"""MoE (Mixture of Experts) support for TurboQuant.

Provides:
- MoE layer detection and parsing
- Expert-aware quantization (quantize each expert independently)
- TurboQuantMoEExpert: quantized expert module with offload support
- TurboQuantMoELayer: drop-in replacement for MoE FFN layers

Target architectures:
- Kimi K2.5: 384 experts, 8 active per token, 1T total params
- Mixtral: 8 experts, 2 active per token
- Qwen3-MoE: 64 experts, 8 active per token
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import torch
import torch.nn as nn

from turboquant_model.codebook import get_codebook
from turboquant_model.quantize import pack_4bit, unpack_4bit
from turboquant_model.rotation import generate_rotation_matrix, hadamard_rotate

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MoE Layer Detection
# ---------------------------------------------------------------------------

# Common MoE layer name patterns across different architectures
MOE_PATTERNS = [
    # Mixtral / Qwen3-MoE style
    r".*\.block_sparse_moe\.experts\.\d+",
    r".*\.mlp\.experts\.\d+",
    # DeepSeek / Kimi style
    r".*\.moe\.experts\.\d+",
    r".*\.ffn\.experts\.\d+",
    # Generic patterns
    r".*experts\[\d+\]",
    r".*expert_\d+",
]

# Router patterns
ROUTER_PATTERNS = [
    r".*\.gate$",
    r".*\.router\.gate$",
    r".*\.router$",
    r".*\.gate_proj$",
]

# Shared expert patterns (always in GPU memory)
SHARED_EXPERT_PATTERNS = [
    r".*shared_expert",
    r".*shared_ffn",
]


@dataclass
class MoELayerInfo:
    """Information about a detected MoE layer."""
    layer_name: str  # e.g., "model.layers.0.block_sparse_moe"
    num_experts: int
    expert_names: List[str]  # Full module names for each expert
    router_name: Optional[str]
    shared_expert_name: Optional[str]
    expert_in_features: int
    expert_out_features: int
    has_gate_proj: bool
    has_up_proj: bool
    has_down_proj: bool


def detect_moe_layers(model: nn.Module) -> List[MoELayerInfo]:
    """Detect MoE layers in a model.
    
    Returns a list of MoELayerInfo describing each MoE layer found.
    """
    moe_layers = []
    expert_groups: Dict[str, List[Tuple[str, nn.Module]]] = {}
    routers: Dict[str, str] = {}
    shared_experts: Dict[str, str] = {}
    
    for name, module in model.named_modules():
        # Check for expert patterns
        for pattern in MOE_PATTERNS:
            if re.match(pattern, name):
                # Extract the MoE layer prefix (everything before .experts.N)
                match = re.match(r"(.+\.(?:experts|moe\.experts|mlp\.experts|block_sparse_moe\.experts))\.\d+", name)
                if match:
                    layer_prefix = match.group(1).rsplit('.experts', 1)[0]
                    if layer_prefix not in expert_groups:
                        expert_groups[layer_prefix] = []
                    expert_groups[layer_prefix].append((name, module))
                break
        
        # Check for router patterns
        for pattern in ROUTER_PATTERNS:
            if re.match(pattern, name):
                layer_prefix = name.rsplit('.', 1)[0]
                routers[layer_prefix] = name
                break
        
        # Check for shared expert patterns
        for pattern in SHARED_EXPERT_PATTERNS:
            if re.match(pattern, name):
                layer_prefix = name.rsplit('.', 1)[0]
                shared_experts[layer_prefix] = name
                break
    
    # Build MoELayerInfo for each detected MoE layer
    for layer_prefix, experts in expert_groups.items():
        if len(experts) == 0:
            continue
        
        expert_names = sorted([e[0] for e in experts], 
                              key=lambda x: int(re.search(r'\.(\d+)$', x).group(1)) if re.search(r'\.(\d+)$', x) else 0)
        
        # Analyze expert structure
        sample_expert = experts[0][1]
        has_gate_proj = hasattr(sample_expert, 'gate_proj')
        has_up_proj = hasattr(sample_expert, 'up_proj')
        has_down_proj = hasattr(sample_expert, 'down_proj')
        
        # Get dimensions
        in_features, out_features = 0, 0
        if has_gate_proj and hasattr(sample_expert.gate_proj, 'weight'):
            in_features = sample_expert.gate_proj.weight.shape[1]
            out_features = sample_expert.gate_proj.weight.shape[0]
        elif hasattr(sample_expert, 'weight'):
            in_features = sample_expert.weight.shape[1]
            out_features = sample_expert.weight.shape[0]
        elif isinstance(sample_expert, nn.Linear):
            in_features = sample_expert.in_features
            out_features = sample_expert.out_features
        
        moe_layers.append(MoELayerInfo(
            layer_name=layer_prefix,
            num_experts=len(experts),
            expert_names=expert_names,
            router_name=routers.get(layer_prefix),
            shared_expert_name=shared_experts.get(layer_prefix),
            expert_in_features=in_features,
            expert_out_features=out_features,
            has_gate_proj=has_gate_proj,
            has_up_proj=has_up_proj,
            has_down_proj=has_down_proj,
        ))
    
    return moe_layers


def is_moe_model(model: nn.Module) -> bool:
    """Check if a model uses MoE architecture."""
    return len(detect_moe_layers(model)) > 0


# ---------------------------------------------------------------------------
# Expert Quantization Data
# ---------------------------------------------------------------------------

@dataclass
class QuantizedExpertData:
    """Holds quantized data for a single expert."""
    expert_id: int
    # Gate projection
    gate_indices: Optional[torch.Tensor] = None  # packed uint8
    gate_norms: Optional[torch.Tensor] = None
    gate_bias: Optional[torch.Tensor] = None
    gate_shape: Optional[Tuple[int, int]] = None
    # Up projection
    up_indices: Optional[torch.Tensor] = None
    up_norms: Optional[torch.Tensor] = None
    up_bias: Optional[torch.Tensor] = None
    up_shape: Optional[Tuple[int, int]] = None
    # Down projection
    down_indices: Optional[torch.Tensor] = None
    down_norms: Optional[torch.Tensor] = None
    down_bias: Optional[torch.Tensor] = None
    down_shape: Optional[Tuple[int, int]] = None
    
    def memory_bytes(self) -> int:
        """Calculate compressed storage in bytes."""
        total = 0
        for attr in ['gate_indices', 'up_indices', 'down_indices']:
            tensor = getattr(self, attr)
            if tensor is not None:
                total += tensor.numel()  # uint8
        for attr in ['gate_norms', 'up_norms', 'down_norms']:
            tensor = getattr(self, attr)
            if tensor is not None:
                total += tensor.numel() * 4  # float32
        for attr in ['gate_bias', 'up_bias', 'down_bias']:
            tensor = getattr(self, attr)
            if tensor is not None:
                total += tensor.numel() * 4
        return total


# ---------------------------------------------------------------------------
# Quantize Expert Weights
# ---------------------------------------------------------------------------

@torch.no_grad()
def quantize_expert(
    expert: nn.Module,
    bit_width: int = 4,
    group_size: int = 128,
    seed: int = 42,
    rotation: str = "qr",
) -> QuantizedExpertData:
    """Quantize a single expert module.
    
    Handles various expert structures:
    - SwiGLU (gate_proj + up_proj + down_proj)
    - Standard FFN (fc1 + fc2)
    - Single linear layer
    
    Args:
        expert: The expert module to quantize
        bit_width: Bits per weight (default 4)
        group_size: Quantization group size
        seed: Random seed for rotation
        rotation: "qr" or "hadamard"
    
    Returns:
        QuantizedExpertData with packed indices and norms
    """
    centroids, boundaries = get_codebook(bit_width)
    
    data = QuantizedExpertData(expert_id=-1)  # ID set later
    
    def _quantize_linear(linear: nn.Linear, base_seed: int):
        W = linear.weight.data.float()
        M, N = W.shape
        device = W.device
        
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
                Y = hadamard_rotate(W_norm, seed=base_seed + g_start)
            else:
                Pi = generate_rotation_matrix(g_dim, seed=base_seed + g_start).to(device)
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
        bias = linear.bias.data.clone() if linear.bias is not None else None
        
        return packed.cpu(), norms_out.cpu(), bias.cpu() if bias is not None else None, (M, N)
    
    # Quantize each projection
    if hasattr(expert, 'gate_proj') and isinstance(expert.gate_proj, nn.Linear):
        data.gate_indices, data.gate_norms, data.gate_bias, data.gate_shape = \
            _quantize_linear(expert.gate_proj, seed)
    
    if hasattr(expert, 'up_proj') and isinstance(expert.up_proj, nn.Linear):
        data.up_indices, data.up_norms, data.up_bias, data.up_shape = \
            _quantize_linear(expert.up_proj, seed + 10000)
    
    if hasattr(expert, 'down_proj') and isinstance(expert.down_proj, nn.Linear):
        data.down_indices, data.down_norms, data.down_bias, data.down_shape = \
            _quantize_linear(expert.down_proj, seed + 20000)
    
    # Handle fc1/fc2 style experts
    if hasattr(expert, 'fc1') and isinstance(expert.fc1, nn.Linear):
        data.gate_indices, data.gate_norms, data.gate_bias, data.gate_shape = \
            _quantize_linear(expert.fc1, seed)
    
    if hasattr(expert, 'fc2') and isinstance(expert.fc2, nn.Linear):
        data.down_indices, data.down_norms, data.down_bias, data.down_shape = \
            _quantize_linear(expert.fc2, seed + 20000)
    
    # Handle single Linear expert
    if isinstance(expert, nn.Linear):
        data.gate_indices, data.gate_norms, data.gate_bias, data.gate_shape = \
            _quantize_linear(expert, seed)
    
    return data


# ---------------------------------------------------------------------------
# TurboQuant MoE Expert Module
# ---------------------------------------------------------------------------

class TurboQuantMoEExpert(nn.Module):
    """A single quantized expert with support for memory offloading.
    
    Stores expert weights as packed 4-bit indices + norms.
    Can load/unload from mmap for memory efficiency.
    
    Supports SwiGLU (gate + up + down) and standard FFN (fc1 + fc2) structures.
    """
    
    def __init__(
        self,
        expert_id: int,
        in_features: int,
        intermediate_size: int,
        bit_width: int = 4,
        group_size: int = 128,
        has_gate: bool = True,
        has_up: bool = True,
        has_down: bool = True,
        rotation: str = "qr",
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.expert_id = expert_id
        self.in_features = in_features
        self.intermediate_size = intermediate_size
        self.bit_width = bit_width
        self.group_size = group_size
        self.rotation = rotation
        self.has_gate = has_gate
        self.has_up = has_up
        self.has_down = has_down
        
        self.n_levels = 2 ** bit_width
        self._scale_in = math.sqrt(group_size)
        self._scale_inter = math.sqrt(group_size)
        
        # Codebook (shared across all experts)
        self.register_buffer("codebook", torch.zeros(self.n_levels, dtype=torch.float32))
        
        # Offload state
        self._is_loaded = False
        self._offload_path: Optional[Path] = None
        
        # Weight buffers (None when offloaded)
        self._gate_indices: Optional[torch.Tensor] = None
        self._gate_norms: Optional[torch.Tensor] = None
        self._gate_bias: Optional[torch.Tensor] = None
        
        self._up_indices: Optional[torch.Tensor] = None
        self._up_norms: Optional[torch.Tensor] = None
        self._up_bias: Optional[torch.Tensor] = None
        
        self._down_indices: Optional[torch.Tensor] = None
        self._down_norms: Optional[torch.Tensor] = None
        self._down_bias: Optional[torch.Tensor] = None
        
        # Rotation cache
        self._rotation_cache: Dict[int, torch.Tensor] = {}
        self._rotation_seed: int = 42
    
    def set_rotation(self, seed: int):
        self._rotation_seed = seed
        self._rotation_cache.clear()
    
    def _get_rotation(self, seed: int, g_start: int, g_dim: int) -> torch.Tensor:
        key = (seed, g_start, g_dim)
        if key not in self._rotation_cache:
            device = self.codebook.device
            self._rotation_cache[key] = generate_rotation_matrix(
                g_dim, seed=seed + g_start
            ).to(device)
        return self._rotation_cache[key]
    
    @property
    def is_loaded(self) -> bool:
        return self._is_loaded
    
    def load_weights(
        self,
        gate_indices: Optional[torch.Tensor] = None,
        gate_norms: Optional[torch.Tensor] = None,
        gate_bias: Optional[torch.Tensor] = None,
        up_indices: Optional[torch.Tensor] = None,
        up_norms: Optional[torch.Tensor] = None,
        up_bias: Optional[torch.Tensor] = None,
        down_indices: Optional[torch.Tensor] = None,
        down_norms: Optional[torch.Tensor] = None,
        down_bias: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ):
        """Load quantized weights into GPU memory."""
        if device is None:
            device = self.codebook.device
        
        if gate_indices is not None:
            self._gate_indices = gate_indices.to(device)
        if gate_norms is not None:
            self._gate_norms = gate_norms.to(device)
        if gate_bias is not None:
            self._gate_bias = gate_bias.to(device)
        
        if up_indices is not None:
            self._up_indices = up_indices.to(device)
        if up_norms is not None:
            self._up_norms = up_norms.to(device)
        if up_bias is not None:
            self._up_bias = up_bias.to(device)
        
        if down_indices is not None:
            self._down_indices = down_indices.to(device)
        if down_norms is not None:
            self._down_norms = down_norms.to(device)
        if down_bias is not None:
            self._down_bias = down_bias.to(device)
        
        self._is_loaded = True
    
    def unload_weights(self):
        """Remove weights from GPU memory (for offloading)."""
        self._gate_indices = None
        self._gate_norms = None
        self._gate_bias = None
        self._up_indices = None
        self._up_norms = None
        self._up_bias = None
        self._down_indices = None
        self._down_norms = None
        self._down_bias = None
        self._rotation_cache.clear()
        self._is_loaded = False
    
    def _forward_linear(
        self,
        x: torch.Tensor,
        indices_packed: torch.Tensor,
        norms: torch.Tensor,
        bias: Optional[torch.Tensor],
        in_features: int,
        seed_offset: int,
    ) -> torch.Tensor:
        """Forward pass for a single quantized linear layer."""
        B = x.shape[0]
        N = indices_packed.shape[0]
        device = x.device
        scale = self._scale_in
        n_groups = math.ceil(in_features / self.group_size)
        
        output = torch.zeros(B, N, dtype=torch.float32, device=device)
        
        for g in range(n_groups):
            g_start = g * self.group_size
            g_end = min(g_start + self.group_size, in_features)
            g_dim = g_end - g_start
            
            # Rotate input slice
            if self.rotation == "hadamard":
                x_rot_g = hadamard_rotate(x[:, g_start:g_end], self._rotation_seed + seed_offset + g_start)
            else:
                Pi_g = self._get_rotation(self._rotation_seed + seed_offset, g_start, g_dim)
                x_rot_g = x[:, g_start:g_end] @ Pi_g.T
            
            # Per-group norms
            if norms.dim() == 1:
                norms_g = norms
            else:
                norms_g = norms[:, g]
            
            # Unpack indices for this group
            packed_g = indices_packed[:, g_start // 2 : g_end // 2]
            indices_g = unpack_4bit(packed_g, g_dim)
            
            # Dequant + matmul
            W_g = self.codebook[indices_g.long()]
            out_g = x_rot_g @ W_g.T
            out_g = out_g * (norms_g[None, :] / scale)
            
            output += out_g
        
        if bias is not None:
            output = output + bias[None, :]
        
        return output
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the expert.
        
        Assumes SwiGLU activation: out = down(silu(gate(x)) * up(x))
        Falls back to standard FFN if no up_proj.
        """
        if not self._is_loaded:
            raise RuntimeError(f"Expert {self.expert_id} weights not loaded. Call load_weights() first.")
        
        orig_shape = x.shape
        if x.dim() == 3:
            x = x.reshape(-1, x.shape[-1])
        
        x_f = x.float()
        
        # Gate projection
        gate_out = self._forward_linear(
            x_f, self._gate_indices, self._gate_norms, self._gate_bias,
            self.in_features, seed_offset=0
        )
        
        if self.has_up and self._up_indices is not None:
            # SwiGLU: silu(gate) * up
            up_out = self._forward_linear(
                x_f, self._up_indices, self._up_norms, self._up_bias,
                self.in_features, seed_offset=10000
            )
            hidden = torch.nn.functional.silu(gate_out) * up_out
        else:
            # Standard activation
            hidden = torch.nn.functional.silu(gate_out)
        
        if self.has_down and self._down_indices is not None:
            # Down projection
            output = self._forward_linear(
                hidden, self._down_indices, self._down_norms, self._down_bias,
                self.intermediate_size, seed_offset=20000
            )
        else:
            output = hidden
        
        if len(orig_shape) == 3:
            output = output.reshape(orig_shape[0], orig_shape[1], -1)
        
        return output.to(x.dtype)
    
    def memory_bytes(self) -> int:
        """Calculate GPU memory usage when loaded."""
        total = 0
        for tensor in [self._gate_indices, self._up_indices, self._down_indices]:
            if tensor is not None:
                total += tensor.numel()  # uint8
        for tensor in [self._gate_norms, self._up_norms, self._down_norms]:
            if tensor is not None:
                total += tensor.numel() * 4  # float32
        for tensor in [self._gate_bias, self._up_bias, self._down_bias]:
            if tensor is not None:
                total += tensor.numel() * 4
        return total


# ---------------------------------------------------------------------------
# TurboQuant MoE Layer
# ---------------------------------------------------------------------------

class TurboQuantMoELayer(nn.Module):
    """Complete MoE layer with quantized experts and routing.
    
    Integrates with ExpertOffloadManager for memory-efficient inference.
    Keeps router and shared expert always in GPU memory.
    """
    
    def __init__(
        self,
        num_experts: int,
        num_experts_per_tok: int,
        in_features: int,
        intermediate_size: int,
        bit_width: int = 4,
        group_size: int = 128,
        rotation: str = "qr",
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.in_features = in_features
        self.intermediate_size = intermediate_size
        self.bit_width = bit_width
        self.group_size = group_size
        self.rotation = rotation
        
        # Router (always in GPU memory)
        self.router = nn.Linear(in_features, num_experts, bias=False, device=device)
        
        # Experts
        self.experts = nn.ModuleList([
            TurboQuantMoEExpert(
                expert_id=i,
                in_features=in_features,
                intermediate_size=intermediate_size,
                bit_width=bit_width,
                group_size=group_size,
                rotation=rotation,
                device=device,
            )
            for i in range(num_experts)
        ])
        
        # Shared expert (optional, always in GPU)
        self.shared_expert: Optional[TurboQuantMoEExpert] = None
        
        # Offload manager reference (set externally)
        self._offload_manager = None
    
    def set_offload_manager(self, manager):
        """Set the offload manager for expert loading/unloading."""
        self._offload_manager = manager
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with expert routing and optional offloading.
        
        1. Router computes expert weights
        2. Request needed experts from offload manager
        3. Run selected experts
        4. Combine outputs
        """
        orig_shape = x.shape
        if x.dim() == 3:
            batch_size, seq_len, hidden_dim = x.shape
            x = x.reshape(-1, hidden_dim)
        else:
            batch_size, hidden_dim = x.shape
            seq_len = 1
        
        num_tokens = x.shape[0]
        
        # Router forward
        router_logits = self.router(x)  # (num_tokens, num_experts)
        router_weights = torch.softmax(router_logits, dim=-1)
        
        # Select top-k experts per token
        top_weights, top_indices = torch.topk(
            router_weights, self.num_experts_per_tok, dim=-1
        )  # (num_tokens, k)
        
        # Normalize weights
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)
        
        # Get unique experts needed
        unique_experts = torch.unique(top_indices).tolist()
        
        # Request experts from offload manager
        if self._offload_manager is not None:
            self._offload_manager.ensure_loaded(unique_experts)
        
        # Compute output
        output = torch.zeros(num_tokens, self.in_features, dtype=x.dtype, device=x.device)
        
        # Process each expert
        for expert_idx in unique_experts:
            expert = self.experts[expert_idx]
            
            if not expert.is_loaded:
                logger.warning(f"Expert {expert_idx} not loaded, skipping")
                continue
            
            # Find tokens routed to this expert
            mask = (top_indices == expert_idx).any(dim=-1)  # (num_tokens,)
            if not mask.any():
                continue
            
            token_indices = mask.nonzero(as_tuple=True)[0]
            expert_input = x[token_indices]  # (n_tokens, hidden)
            
            # Get weights for this expert
            weights_for_expert = torch.zeros(token_indices.shape[0], dtype=x.dtype, device=x.device)
            for k in range(self.num_experts_per_tok):
                k_mask = (top_indices[token_indices, k] == expert_idx)
                weights_for_expert[k_mask] = top_weights[token_indices[k_mask], k]
            
            # Expert forward
            expert_output = expert(expert_input)  # (n_tokens, hidden)
            
            # Weighted accumulation
            output[token_indices] += weights_for_expert.unsqueeze(-1) * expert_output
        
        # Add shared expert contribution
        if self.shared_expert is not None and self.shared_expert.is_loaded:
            shared_out = self.shared_expert(x)
            output = output + shared_out
        
        # Restore shape
        if len(orig_shape) == 3:
            output = output.reshape(batch_size, seq_len, -1)
        
        return output
    
    def get_needed_experts(self, x: torch.Tensor) -> List[int]:
        """Pre-compute which experts will be needed for a given input.
        
        Useful for prefetching experts before forward pass.
        """
        with torch.no_grad():
            router_logits = self.router(x.reshape(-1, x.shape[-1]))
            _, top_indices = torch.topk(router_logits, self.num_experts_per_tok, dim=-1)
            return torch.unique(top_indices).tolist()
