"""Model-level quantization, saving, and loading.

quantize_model:  Replace all nn.Linear → TurboQuantLinear (single-pass or residual)
save_quantized / load_quantized: Serialize/deserialize quantized models to disk
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from turboquant_model.codebook import get_codebook
from turboquant_model.rotation import generate_rotation_matrix
from turboquant_model.quantize import pack_4bit
from turboquant_model.module import TurboQuantLinear
from turboquant_model.rotation import (
    generate_rotation_matrix,
    hadamard_rotate,
    hadamard_rotate_inverse,
)

logger = logging.getLogger(__name__)


@dataclass
class TurboQuantConfig:
    """Configuration for TurboQuant weight quantization."""

    bit_width: int = 4
    group_size: Optional[int] = 128
    seed: int = 42
    skip_embeddings: bool = False
    skip_lm_head: bool = False
    # Residual
    residual_bit_width: Optional[int] = None
    residual_seed: int = 1042
    # Rotation method: "qr" (Haar random orthogonal) or "hadamard" (fast Walsh-Hadamard + signs)
    rotation: str = "qr"
    # Rotation strategy for residual passes:
    #   "different" — pass 1 uses seed, pass 2 uses residual_seed (default, best quality)
    #   "shared"    — both passes use the same seed (enables merge_and_requantize)
    #   "alternating" — even passes use seed, odd passes use residual_seed (for multi-pass)
    rotation_strategy: str = "different"
    # MoE (Mixture of Experts) configuration
    moe_offload: bool = False  # Enable expert offloading for MoE models
    expert_cache_size: int = 16  # Number of experts to keep in GPU memory
    offload_path: Optional[str] = None  # Path for mmap'd expert files
    skip_moe_experts: bool = False  # Skip quantizing MoE experts (keep in bf16)

    def save(self, path: str | Path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "TurboQuantConfig":
        with open(path) as f:
            return cls(**json.load(f))

    @property
    def total_bits(self) -> int:
        return self.bit_width + (self.residual_bit_width or 0)


# ---------------------------------------------------------------------------
# Quantize model
# ---------------------------------------------------------------------------


@torch.no_grad()
def quantize_model(model: nn.Module, config: TurboQuantConfig) -> nn.Module:
    """Quantize all nn.Linear layers, replacing them with TurboQuantLinear.

    Supports single-pass and residual (two-pass) quantization.
    All layers use on-the-fly dequantization at inference.

    Args:
        model: HuggingFace model (or any nn.Module with Linear layers)
        config: quantization configuration

    Returns:
        model with TurboQuantLinear modules (modified in-place)
    """
    centroids, boundaries = get_codebook(config.bit_width)
    if config.residual_bit_width:
        r_centroids, r_boundaries = get_codebook(config.residual_bit_width)

    replaced = 0
    total_orig = 0
    total_compressed = 0

    # Collect modules to replace
    replacements = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if config.skip_embeddings and "embed" in name.lower():
            continue
        if config.skip_lm_head and "lm_head" in name.lower():
            continue
        replacements.append((name, module))

    for name, module in replacements:
        W = module.weight.data
        M, N = W.shape
        device = W.device

        group_size = config.group_size or N

        # --- Pass 1: Quantize weight ---
        pass1_packed, pass1_norms, pass1_codebook = _quantize_weight(
            W, config.bit_width, group_size, config.seed, centroids, boundaries, device,
            rotation=config.rotation,
        )

        # --- Create TurboQuantLinear ---
        tq = TurboQuantLinear(
            in_features=N,
            out_features=M,
            bias=module.bias is not None,
            bit_width=config.bit_width,
            group_size=group_size,
            device=device,
            rotation=config.rotation,
        )
        tq.indices_packed.copy_(pass1_packed)
        tq.weight_norms.copy_(pass1_norms)
        tq.codebook.copy_(centroids.to(device))
        tq.set_rotation(config.seed)

        if module.bias is not None:
            tq.bias.copy_(module.bias.data)

        # --- Pass 2: Residual quantization ---
        if config.residual_bit_width:
            # Reconstruct pass 1 to compute residual
            W_hat1 = tq.dequantize().float()
            residual = W.float() - W_hat1

            # Determine residual rotation seed based on strategy
            if config.rotation_strategy == "shared":
                pass2_seed = config.seed
            else:  # "different" or "alternating" — both use residual_seed for pass 2
                pass2_seed = config.residual_seed

            pass2_packed, pass2_norms, pass2_codebook = _quantize_weight(
                residual, config.residual_bit_width, group_size,
                pass2_seed, r_centroids, r_boundaries, device,
                rotation=config.rotation,
            )
            tq.set_pass2(
                indices_packed=pass2_packed,
                weight_norms=pass2_norms,
                codebook=r_centroids.to(device),
                seed=pass2_seed,
            )

        # Replace in model
        _replace_module(model, name, tq)

        orig_bytes = M * N * 2  # bf16
        total_orig += orig_bytes
        total_compressed += tq.memory_bytes()
        replaced += 1

    mode = "residual" if config.residual_bit_width else "single-pass"
    bits = f"{config.bit_width}" if not config.residual_bit_width else f"{config.bit_width}+{config.residual_bit_width}"
    logger.info(
        f"Quantized {replaced} layers ({mode}, {bits}-bit): "
        f"{total_orig / 1024**2:.1f}MB → {total_compressed / 1024**2:.1f}MB "
        f"({total_orig / total_compressed:.1f}x compression)"
    )

    return model


def _quantize_weight(
    W: torch.Tensor,
    bit_width: int,
    group_size: int,
    seed: int,
    centroids: torch.Tensor,
    boundaries: torch.Tensor,
    device: torch.device,
    rotation: str = "qr",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize a single weight matrix and return packed data.

    Returns: (indices_packed, norms, codebook)
    """
    M, N = W.shape
    W = W.float()

    all_norms = []
    all_indices = []

    bnd = boundaries.to(device)
    ctr = centroids.to(device)

    for g_start in range(0, N, group_size):
        g_end = min(g_start + group_size, N)
        g_dim = g_end - g_start
        W_g = W[:, g_start:g_end]

        norms = W_g.norm(dim=1, keepdim=True).clamp(min=1e-8)
        W_norm = W_g / norms
        all_norms.append(norms.squeeze(1))

        if rotation == "hadamard":
            Y = hadamard_rotate(W_norm, seed=seed + g_start)
        else:
            Pi = generate_rotation_matrix(g_dim, seed=seed + g_start).to(device)
            Y = W_norm @ Pi.T
        scale = math.sqrt(g_dim)
        Y_scaled = Y * scale

        indices = torch.searchsorted(bnd, Y_scaled.reshape(-1))
        indices = indices.clamp(0, len(ctr) - 1).reshape(M, g_dim)
        all_indices.append(indices)

    full_indices = torch.cat(all_indices, dim=1)
    norms_out = torch.stack(all_norms, dim=1) if len(all_norms) > 1 else all_norms[0]

    if N % 2 != 0:
        full_indices = torch.nn.functional.pad(full_indices, (0, 1), value=0)

    packed = pack_4bit(full_indices)
    return packed, norms_out, ctr


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------


@torch.no_grad()
def save_quantized(model: nn.Module, config: TurboQuantConfig, save_dir: str | Path):
    """Save quantized model to disk in safetensors format.

    Directory structure:
        save_dir/
        ├── turboquant_config.json
        ├── model.safetensors          # all quantized layer tensors + codebook
        ├── non_quantized.safetensors  # non-linear params (embeddings, norms, etc.)
        └── config.json                # (optional) HuggingFace model config
    """
    from safetensors.torch import save_file

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    config.save(save_dir / "turboquant_config.json")

    # Save HF model config
    if hasattr(model, "config"):
        model.config.save_pretrained(save_dir)

    tensors = {}
    codebook_saved = False
    tq_param_prefixes = set()

    for name, module in model.named_modules():
        if isinstance(module, TurboQuantLinear):
            safe = name.replace(".", "_")
            tensors[f"{safe}.indices"] = module.indices_packed.cpu().contiguous()
            tensors[f"{safe}.norms"] = module.weight_norms.cpu().contiguous()

            if module.bias is not None:
                tensors[f"{safe}.bias"] = module.bias.cpu().contiguous()

            if module.has_residual:
                tensors[f"{safe}.pass2_indices"] = module.pass2_indices_packed.cpu().contiguous()
                tensors[f"{safe}.pass2_norms"] = module.pass2_weight_norms.cpu().contiguous()
                tensors[f"{safe}.pass2_codebook"] = module.pass2_codebook.cpu().clone()

            if not codebook_saved:
                tensors["codebook"] = module.codebook.cpu().clone()
                codebook_saved = True

            tq_param_prefixes.add(name + ".")

    save_file(tensors, save_dir / "model.safetensors")

    # Collect non-quantized parameters
    non_quantized = {}
    for pname, param in model.named_parameters():
        is_tq = any(pname.startswith(prefix) for prefix in tq_param_prefixes)
        if not is_tq:
            non_quantized[pname] = param.data.cpu().contiguous()

    for bname, buf in model.named_buffers():
        is_tq = any(bname.startswith(prefix) for prefix in tq_param_prefixes)
        if not is_tq and bname not in non_quantized:
            non_quantized[bname] = buf.cpu().contiguous()

    save_file(non_quantized, save_dir / "non_quantized.safetensors")

    total = sum(f.stat().st_size for f in save_dir.rglob("*") if f.is_file())
    logger.info(f"Saved quantized model to {save_dir} ({total / 1024**2:.1f} MB)")


@torch.no_grad()
def load_quantized(
    model_name_or_path: str,
    quantized_dir: str | Path,
    device: str = "cuda",
) -> nn.Module:
    """Load a pre-quantized model from disk.

    Supports both safetensors format (model.safetensors) and legacy
    .pt format (layers/*.pt).

    Args:
        model_name_or_path: HF model name or path (for architecture)
        quantized_dir: directory with saved quantized weights
        device: target device

    Returns:
        model with TurboQuantLinear modules, on-the-fly mode
    """
    from transformers import AutoModelForCausalLM, AutoConfig

    quantized_dir = Path(quantized_dir)
    config = TurboQuantConfig.load(quantized_dir / "turboquant_config.json")

    # Load architecture
    if (quantized_dir / "config.json").exists():
        model_config = AutoConfig.from_pretrained(quantized_dir)
    else:
        model_config = AutoConfig.from_pretrained(model_name_or_path)

    model = AutoModelForCausalLM.from_config(model_config).to(torch.bfloat16).to(device)

    # Detect format: safetensors vs legacy .pt
    safetensors_path = quantized_dir / "model.safetensors"
    use_safetensors = safetensors_path.exists()

    if use_safetensors:
        from safetensors.torch import load_file
        tensors = load_file(str(safetensors_path), device=device)
        codebook = tensors["codebook"]
    else:
        tensors = None
        codebook = torch.load(quantized_dir / "codebook.pt", map_location=device, weights_only=True)

    layers_dir = quantized_dir / "layers"

    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if config.skip_embeddings and "embed" in name.lower():
            continue
        if config.skip_lm_head and "lm_head" in name.lower():
            continue

        safe = name.replace(".", "_")

        if use_safetensors:
            indices_key = f"{safe}.indices"
            if indices_key not in tensors:
                continue
        else:
            indices_path = layers_dir / f"{safe}.indices.pt"
            if not indices_path.exists():
                continue

        M, N = module.weight.shape
        group_size = config.group_size or N

        tq = TurboQuantLinear(
            in_features=N,
            out_features=M,
            bias=module.bias is not None,
            bit_width=config.bit_width,
            group_size=group_size,
            device=device,
            rotation=config.rotation,
        )

        if use_safetensors:
            tq.indices_packed = tensors[f"{safe}.indices"]
            tq.weight_norms = tensors[f"{safe}.norms"]
        else:
            tq.indices_packed = torch.load(layers_dir / f"{safe}.indices.pt", map_location=device, weights_only=True)
            tq.weight_norms = torch.load(layers_dir / f"{safe}.norms.pt", map_location=device, weights_only=True)

        tq.codebook = codebook

        if module.bias is not None:
            if use_safetensors:
                bias_key = f"{safe}.bias"
                if bias_key in tensors:
                    tq.bias = tensors[bias_key]
            else:
                bias_path = layers_dir / f"{safe}.bias.pt"
                if bias_path.exists():
                    tq.bias = torch.load(bias_path, map_location=device, weights_only=True)

        tq.set_rotation(config.seed)

        # Load residual pass if present
        if use_safetensors:
            pass2_key = f"{safe}.pass2_indices"
            if pass2_key in tensors:
                tq.set_pass2(
                    indices_packed=tensors[pass2_key],
                    weight_norms=tensors[f"{safe}.pass2_norms"],
                    codebook=tensors[f"{safe}.pass2_codebook"],
                    seed=config.residual_seed,
                )
        else:
            pass2_path = layers_dir / f"{safe}.pass2_indices.pt"
            if pass2_path.exists():
                tq.set_pass2(
                    indices_packed=torch.load(pass2_path, map_location=device, weights_only=True),
                    weight_norms=torch.load(layers_dir / f"{safe}.pass2_norms.pt", map_location=device, weights_only=True),
                    codebook=torch.load(layers_dir / f"{safe}.pass2_codebook.pt", map_location=device, weights_only=True),
                    seed=config.residual_seed,
                )

        _replace_module(model, name, tq)

    # Load non-quantized parameters
    non_quantized_st = quantized_dir / "non_quantized.safetensors"
    if non_quantized_st.exists():
        from safetensors.torch import load_file
        remaining = load_file(str(non_quantized_st), device=device)
    else:
        remaining = torch.load(quantized_dir / "non_quantized.pt", map_location=device, weights_only=True)

    for pname, tensor in remaining.items():
        parts = pname.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        target = getattr(parent, parts[-1], None)
        if target is not None:
            if isinstance(target, nn.Parameter):
                target.data.copy_(tensor)
            elif isinstance(target, torch.Tensor):
                target.copy_(tensor)

    model.eval()
    logger.info(f"Loaded quantized model from {quantized_dir}")
    return model


# ---------------------------------------------------------------------------
# MoE Model Quantization
# ---------------------------------------------------------------------------


@torch.no_grad()
def quantize_moe_model(model: nn.Module, config: TurboQuantConfig) -> nn.Module:
    """Quantize an MoE model with expert offloading support.
    
    Detects MoE layers and quantizes each expert independently.
    Non-MoE layers (attention, embeddings) are quantized normally.
    
    Args:
        model: HuggingFace MoE model (Mixtral, Qwen3-MoE, etc.)
        config: TurboQuantConfig with MoE options
    
    Returns:
        Quantized model with TurboQuantMoELayer modules
    """
    from turboquant_model.moe import (
        detect_moe_layers,
        quantize_expert,
        TurboQuantMoEExpert,
        TurboQuantMoELayer,
        MoELayerInfo,
    )
    
    # Detect MoE layers
    moe_layers = detect_moe_layers(model)
    
    if not moe_layers:
        logger.info("No MoE layers detected, using standard quantization")
        return quantize_model(model, config)
    
    logger.info(f"Detected {len(moe_layers)} MoE layers with {moe_layers[0].num_experts} experts each")
    
    centroids, boundaries = get_codebook(config.bit_width)
    
    # First, quantize non-MoE linear layers
    moe_expert_prefixes = set()
    for moe_info in moe_layers:
        for expert_name in moe_info.expert_names:
            moe_expert_prefixes.add(expert_name)
        if moe_info.router_name:
            moe_expert_prefixes.add(moe_info.router_name)
    
    # Quantize non-MoE layers
    non_moe_replaced = 0
    replacements = []
    
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if config.skip_embeddings and "embed" in name.lower():
            continue
        if config.skip_lm_head and "lm_head" in name.lower():
            continue
        
        # Skip if part of MoE
        is_moe = any(name.startswith(prefix) or prefix in name for prefix in moe_expert_prefixes)
        if is_moe:
            continue
        
        replacements.append((name, module))
    
    for name, module in replacements:
        W = module.weight.data
        M, N = W.shape
        device = W.device
        group_size = config.group_size or N
        
        pass1_packed, pass1_norms, pass1_codebook = _quantize_weight(
            W, config.bit_width, group_size, config.seed, centroids, boundaries, device,
            rotation=config.rotation,
        )
        
        tq = TurboQuantLinear(
            in_features=N,
            out_features=M,
            bias=module.bias is not None,
            bit_width=config.bit_width,
            group_size=group_size,
            device=device,
            rotation=config.rotation,
        )
        tq.indices_packed.copy_(pass1_packed)
        tq.weight_norms.copy_(pass1_norms)
        tq.codebook.copy_(centroids.to(device))
        tq.set_rotation(config.seed)
        
        if module.bias is not None:
            tq.bias.copy_(module.bias.data)
        
        _replace_module(model, name, tq)
        non_moe_replaced += 1
    
    logger.info(f"Quantized {non_moe_replaced} non-MoE linear layers")
    
    # Now quantize MoE layers
    moe_replaced = 0
    total_experts = 0
    
    for moe_info in moe_layers:
        # Get the parent MoE module
        parts = moe_info.layer_name.split('.')
        parent = model
        for part in parts:
            parent = getattr(parent, part)
        
        # Determine expert structure from first expert
        first_expert_name = moe_info.expert_names[0]
        first_expert = model
        for part in first_expert_name.split('.'):
            first_expert = getattr(first_expert, part)
        
        # Get dimensions
        in_features = moe_info.expert_in_features
        intermediate_size = moe_info.expert_out_features
        
        # Detect num_experts_per_tok from router if available
        num_experts_per_tok = 2  # Default
        if hasattr(parent, 'num_experts_per_tok'):
            num_experts_per_tok = parent.num_experts_per_tok
        elif hasattr(parent, 'top_k'):
            num_experts_per_tok = parent.top_k
        
        # Create TurboQuantMoELayer
        tq_moe = TurboQuantMoELayer(
            num_experts=moe_info.num_experts,
            num_experts_per_tok=num_experts_per_tok,
            in_features=in_features,
            intermediate_size=intermediate_size,
            bit_width=config.bit_width,
            group_size=config.group_size or 128,
            rotation=config.rotation,
            device=first_expert.gate_proj.weight.device if hasattr(first_expert, 'gate_proj') else first_expert.weight.device,
        )
        
        # Copy router weights
        if moe_info.router_name:
            router_module = model
            for part in moe_info.router_name.split('.'):
                router_module = getattr(router_module, part)
            if isinstance(router_module, nn.Linear):
                tq_moe.router.weight.data.copy_(router_module.weight.data)
        
        # Quantize each expert
        for i, expert_name in enumerate(moe_info.expert_names):
            expert_module = model
            for part in expert_name.split('.'):
                expert_module = getattr(expert_module, part)
            
            # Quantize expert
            expert_data = quantize_expert(
                expert_module,
                bit_width=config.bit_width,
                group_size=config.group_size or 128,
                seed=config.seed,
                rotation=config.rotation,
            )
            expert_data.expert_id = i
            
            # Load into TurboQuantMoEExpert
            tq_expert = tq_moe.experts[i]
            tq_expert.codebook.copy_(centroids.to(tq_expert.codebook.device))
            tq_expert.set_rotation(config.seed)
            
            tq_expert.load_weights(
                gate_indices=expert_data.gate_indices,
                gate_norms=expert_data.gate_norms,
                gate_bias=expert_data.gate_bias,
                up_indices=expert_data.up_indices,
                up_norms=expert_data.up_norms,
                up_bias=expert_data.up_bias,
                down_indices=expert_data.down_indices,
                down_norms=expert_data.down_norms,
                down_bias=expert_data.down_bias,
            )
            
            total_experts += 1
        
        # Replace the MoE module
        _replace_module(model, moe_info.layer_name, tq_moe)
        moe_replaced += 1
    
    logger.info(f"Quantized {moe_replaced} MoE layers ({total_experts} experts)")
    
    return model


@torch.no_grad()
def save_moe_quantized(
    model: nn.Module,
    config: TurboQuantConfig,
    save_dir: str | Path,
):
    """Save a quantized MoE model with expert offloading support.
    
    Directory structure:
        save_dir/
        ├── turboquant_config.json
        ├── model.safetensors          # Non-expert quantized layers
        ├── non_quantized.safetensors  # Embeddings, norms, etc.
        ├── experts/                    # Expert files for offloading
        │   ├── layer_0/
        │   │   ├── expert_0.tqe
        │   │   ├── expert_1.tqe
        │   │   └── ...
        │   └── ...
        └── config.json                # HF model config
    """
    from safetensors.torch import save_file
    from turboquant_model.moe import TurboQuantMoELayer
    from turboquant_model.offload import save_expert_file
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    config.save(save_dir / "turboquant_config.json")
    
    # Save HF model config
    if hasattr(model, "config"):
        model.config.save_pretrained(save_dir)
    
    tensors = {}
    codebook_saved = False
    tq_param_prefixes = set()
    moe_layer_idx = 0
    
    # Process modules
    for name, module in model.named_modules():
        if isinstance(module, TurboQuantMoELayer):
            # Save router
            safe = name.replace(".", "_")
            tensors[f"{safe}.router.weight"] = module.router.weight.cpu().contiguous()
            
            # Save experts to offload directory
            experts_dir = save_dir / "experts" / f"layer_{moe_layer_idx}"
            experts_dir.mkdir(parents=True, exist_ok=True)
            
            for expert in module.experts:
                save_expert_file(
                    path=experts_dir / f"expert_{expert.expert_id}.tqe",
                    expert_id=expert.expert_id,
                    bit_width=config.bit_width,
                    group_size=config.group_size or 128,
                    in_features=module.in_features,
                    intermediate_size=module.intermediate_size,
                    gate_indices=expert._gate_indices,
                    gate_norms=expert._gate_norms,
                    up_indices=expert._up_indices,
                    up_norms=expert._up_norms,
                    down_indices=expert._down_indices,
                    down_norms=expert._down_norms,
                )
            
            # Save codebook
            if not codebook_saved:
                tensors["codebook"] = module.experts[0].codebook.cpu().clone()
                codebook_saved = True
            
            tq_param_prefixes.add(name + ".")
            moe_layer_idx += 1
        
        elif isinstance(module, TurboQuantLinear):
            safe = name.replace(".", "_")
            tensors[f"{safe}.indices"] = module.indices_packed.cpu().contiguous()
            tensors[f"{safe}.norms"] = module.weight_norms.cpu().contiguous()
            
            if module.bias is not None:
                tensors[f"{safe}.bias"] = module.bias.cpu().contiguous()
            
            if module.has_residual:
                tensors[f"{safe}.pass2_indices"] = module.pass2_indices_packed.cpu().contiguous()
                tensors[f"{safe}.pass2_norms"] = module.pass2_weight_norms.cpu().contiguous()
                tensors[f"{safe}.pass2_codebook"] = module.pass2_codebook.cpu().clone()
            
            if not codebook_saved:
                tensors["codebook"] = module.codebook.cpu().clone()
                codebook_saved = True
            
            tq_param_prefixes.add(name + ".")
    
    save_file(tensors, save_dir / "model.safetensors")
    
    # Collect non-quantized parameters
    non_quantized = {}
    for pname, param in model.named_parameters():
        is_tq = any(pname.startswith(prefix) for prefix in tq_param_prefixes)
        if not is_tq:
            non_quantized[pname] = param.data.cpu().contiguous()
    
    for bname, buf in model.named_buffers():
        is_tq = any(bname.startswith(prefix) for prefix in tq_param_prefixes)
        if not is_tq and bname not in non_quantized:
            non_quantized[bname] = buf.cpu().contiguous()
    
    save_file(non_quantized, save_dir / "non_quantized.safetensors")
    
    total = sum(f.stat().st_size for f in save_dir.rglob("*") if f.is_file())
    logger.info(f"Saved MoE quantized model to {save_dir} ({total / 1024**2:.1f} MB)")


@torch.no_grad()
def load_moe_quantized(
    model_name_or_path: str,
    quantized_dir: str | Path,
    device: str = "cuda",
    offload: bool = True,
) -> nn.Module:
    """Load a pre-quantized MoE model with optional expert offloading.
    
    Args:
        model_name_or_path: HF model name or path (for architecture)
        quantized_dir: directory with saved quantized weights
        device: target device
        offload: if True, use expert offloading (experts loaded on-demand)
    
    Returns:
        Quantized MoE model ready for inference
    """
    from transformers import AutoModelForCausalLM, AutoConfig
    from safetensors.torch import load_file
    from turboquant_model.moe import TurboQuantMoELayer, TurboQuantMoEExpert
    from turboquant_model.offload import ExpertOffloadManager, ExpertFileHeader
    
    quantized_dir = Path(quantized_dir)
    config = TurboQuantConfig.load(quantized_dir / "turboquant_config.json")
    
    # Load architecture
    if (quantized_dir / "config.json").exists():
        model_config = AutoConfig.from_pretrained(quantized_dir)
    else:
        model_config = AutoConfig.from_pretrained(model_name_or_path)
    
    model = AutoModelForCausalLM.from_config(model_config).to(torch.bfloat16).to(device)
    
    # Load tensors
    tensors = load_file(str(quantized_dir / "model.safetensors"), device=device)
    codebook = tensors.get("codebook")
    
    # Check if this is an MoE model
    experts_dir = quantized_dir / "experts"
    is_moe = experts_dir.exists()
    
    if is_moe:
        # Get MoE layer info from experts directory
        moe_layers_info = {}
        for layer_dir in sorted(experts_dir.iterdir()):
            if layer_dir.is_dir() and layer_dir.name.startswith("layer_"):
                layer_idx = int(layer_dir.name.split("_")[1])
                expert_files = list(layer_dir.glob("expert_*.tqe"))
                moe_layers_info[layer_idx] = len(expert_files)
        
        logger.info(f"Found {len(moe_layers_info)} MoE layers in {quantized_dir}")
    
    # Process modules
    moe_layer_idx = 0
    
    for name, module in list(model.named_modules()):
        safe = name.replace(".", "_")
        
        # Check if this is a router (MoE layer)
        router_key = f"{safe}.router.weight"
        if router_key in tensors:
            # This is an MoE layer
            # Reconstruct TurboQuantMoELayer
            layer_dir = experts_dir / f"layer_{moe_layer_idx}"
            expert_files = sorted(layer_dir.glob("expert_*.tqe"))
            
            # Read first expert header to get dimensions
            with open(expert_files[0], 'rb') as f:
                header = ExpertFileHeader.from_bytes(f.read(ExpertFileHeader.header_size()))
            
            num_experts = len(expert_files)
            router_weight = tensors[router_key]
            
            tq_moe = TurboQuantMoELayer(
                num_experts=num_experts,
                num_experts_per_tok=2,  # Will be overridden by model config
                in_features=header.in_features,
                intermediate_size=header.intermediate_size,
                bit_width=header.bit_width,
                group_size=header.group_size,
                rotation=config.rotation,
                device=device,
            )
            
            tq_moe.router.weight.data.copy_(router_weight)
            
            # Set codebook for all experts
            for expert in tq_moe.experts:
                expert.codebook.copy_(codebook)
                expert.set_rotation(config.seed)
            
            _replace_module(model, name, tq_moe)
            moe_layer_idx += 1
            continue
        
        # Check if this is a regular TurboQuant layer
        indices_key = f"{safe}.indices"
        if indices_key in tensors:
            if not isinstance(module, nn.Linear):
                continue
            
            M, N = module.weight.shape
            group_size = config.group_size or N
            
            tq = TurboQuantLinear(
                in_features=N,
                out_features=M,
                bias=module.bias is not None,
                bit_width=config.bit_width,
                group_size=group_size,
                device=device,
                rotation=config.rotation,
            )
            
            tq.indices_packed = tensors[f"{safe}.indices"]
            tq.weight_norms = tensors[f"{safe}.norms"]
            tq.codebook = codebook
            
            if module.bias is not None:
                bias_key = f"{safe}.bias"
                if bias_key in tensors:
                    tq.bias = tensors[bias_key]
            
            tq.set_rotation(config.seed)
            _replace_module(model, name, tq)
    
    # Load non-quantized parameters
    non_quantized_path = quantized_dir / "non_quantized.safetensors"
    if non_quantized_path.exists():
        remaining = load_file(str(non_quantized_path), device=device)
        for pname, tensor in remaining.items():
            parts = pname.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            target = getattr(parent, parts[-1], None)
            if target is not None:
                if isinstance(target, nn.Parameter):
                    target.data.copy_(tensor)
                elif isinstance(target, torch.Tensor):
                    target.copy_(tensor)
    
    # Set up offload manager if requested
    if is_moe and offload:
        from turboquant_model.offload import create_offload_manager
        
        manager = create_offload_manager(
            model=model,
            offload_path=experts_dir,
            cache_size=config.expert_cache_size,
            device=device,
        )
        # Store manager on model for access
        model._expert_offload_manager = manager
    
    model.eval()
    logger.info(f"Loaded MoE quantized model from {quantized_dir}")
    return model


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _replace_module(model: nn.Module, name: str, new_module: nn.Module):
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)
