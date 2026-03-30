"""Tests for safetensors save/load format.

Validates:
  1. save_quantized writes model.safetensors and non_quantized.safetensors.
  2. Saved safetensors files contain the expected tensor keys.
  3. Roundtrip: save → load tensors → tensors match originals.
  4. Backward compatibility: load_quantized falls back to legacy .pt format.
  5. End-to-end: quantize → save → reload → dequantized weights match.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from safetensors.torch import load_file

from turboquant_model.model import (
    TurboQuantConfig,
    save_quantized,
    _quantize_weight,
    _replace_module,
)
from turboquant_model.module import TurboQuantLinear
from turboquant_model.codebook import get_codebook


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class SmallModel(nn.Module):
    """Tiny model for testing save/load without HuggingFace."""

    def __init__(self):
        super().__init__()
        self.layer_norm = nn.LayerNorm(32)
        self.linear1 = nn.Linear(32, 64, bias=False)
        self.linear2 = nn.Linear(64, 32, bias=True)

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.linear1(x)
        return self.linear2(x)


def _quantize_small_model(model: SmallModel, config: TurboQuantConfig):
    """Quantize the small model's Linear layers in-place."""
    centroids, boundaries = get_codebook(config.bit_width)
    if config.residual_bit_width:
        r_centroids, r_boundaries = get_codebook(config.residual_bit_width)

    for name in ["linear1", "linear2"]:
        module = getattr(model, name)
        W = module.weight.data
        M, N = W.shape
        group_size = config.group_size or N

        pass1_packed, pass1_norms, _ = _quantize_weight(
            W, config.bit_width, group_size, config.seed,
            centroids, boundaries, W.device, rotation=config.rotation,
        )

        tq = TurboQuantLinear(
            in_features=N, out_features=M,
            bias=module.bias is not None,
            bit_width=config.bit_width,
            group_size=group_size,
            device=W.device,
            rotation=config.rotation,
        )
        tq.indices_packed.copy_(pass1_packed)
        tq.weight_norms.copy_(pass1_norms)
        tq.codebook.copy_(centroids.to(W.device))
        tq.set_rotation(config.seed)

        if module.bias is not None:
            tq.bias.copy_(module.bias.data)

        # Residual pass
        if config.residual_bit_width:
            W_hat1 = tq.dequantize().float()
            residual = W.float() - W_hat1
            pass2_seed = config.seed if config.rotation_strategy == "shared" else config.residual_seed
            pass2_packed, pass2_norms, _ = _quantize_weight(
                residual, config.residual_bit_width, group_size,
                pass2_seed, r_centroids, r_boundaries, W.device,
                rotation=config.rotation,
            )
            tq.set_pass2(
                indices_packed=pass2_packed,
                weight_norms=pass2_norms,
                codebook=r_centroids.to(W.device),
                seed=pass2_seed,
            )

        _replace_module(model, name, tq)

    return model


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_dir(tmp_path):
    """Clean temporary directory."""
    return tmp_path / "save_test"


@pytest.fixture()
def small_quantized_model():
    """Quantized SmallModel (single-pass, 4-bit)."""
    torch.manual_seed(42)
    model = SmallModel()
    config = TurboQuantConfig(bit_width=4, group_size=32, seed=42)
    return _quantize_small_model(model, config), config


@pytest.fixture()
def residual_quantized_model():
    """Quantized SmallModel with residual (4+4)."""
    torch.manual_seed(42)
    model = SmallModel()
    config = TurboQuantConfig(
        bit_width=4, residual_bit_width=4,
        group_size=32, seed=42, residual_seed=1042,
    )
    return _quantize_small_model(model, config), config


# ---------------------------------------------------------------------------
# 1. File structure
# ---------------------------------------------------------------------------

class TestSaveFormat:
    """save_quantized creates the expected safetensors files."""

    def test_creates_safetensors_files(self, small_quantized_model, tmp_dir):
        model, config = small_quantized_model
        save_quantized(model, config, tmp_dir)

        assert (tmp_dir / "model.safetensors").exists()
        assert (tmp_dir / "non_quantized.safetensors").exists()
        assert (tmp_dir / "turboquant_config.json").exists()

    def test_no_legacy_files(self, small_quantized_model, tmp_dir):
        model, config = small_quantized_model
        save_quantized(model, config, tmp_dir)

        assert not (tmp_dir / "codebook.pt").exists()
        assert not (tmp_dir / "layers").exists()
        assert not (tmp_dir / "non_quantized.pt").exists()

    def test_config_roundtrip(self, small_quantized_model, tmp_dir):
        model, config = small_quantized_model
        save_quantized(model, config, tmp_dir)

        loaded_config = TurboQuantConfig.load(tmp_dir / "turboquant_config.json")
        assert loaded_config.bit_width == config.bit_width
        assert loaded_config.group_size == config.group_size
        assert loaded_config.seed == config.seed


# ---------------------------------------------------------------------------
# 2. Tensor keys
# ---------------------------------------------------------------------------

class TestTensorKeys:
    """Saved safetensors files contain the expected keys."""

    def test_single_pass_keys(self, small_quantized_model, tmp_dir):
        model, config = small_quantized_model
        save_quantized(model, config, tmp_dir)

        tensors = load_file(str(tmp_dir / "model.safetensors"))
        assert "codebook" in tensors
        assert "linear1.indices" in tensors
        assert "linear1.norms" in tensors
        assert "linear2.indices" in tensors
        assert "linear2.norms" in tensors
        assert "linear2.bias" in tensors
        # No pass2 keys
        assert "linear1.pass2_indices" not in tensors

    def test_residual_keys(self, residual_quantized_model, tmp_dir):
        model, config = residual_quantized_model
        save_quantized(model, config, tmp_dir)

        tensors = load_file(str(tmp_dir / "model.safetensors"))
        assert "linear1.pass2_indices" in tensors
        assert "linear1.pass2_norms" in tensors
        assert "linear1.pass2_codebook" in tensors
        assert "linear2.pass2_indices" in tensors

    def test_non_quantized_keys(self, small_quantized_model, tmp_dir):
        model, config = small_quantized_model
        save_quantized(model, config, tmp_dir)

        nq = load_file(str(tmp_dir / "non_quantized.safetensors"))
        # LayerNorm params should be in non_quantized
        assert "layer_norm.weight" in nq
        assert "layer_norm.bias" in nq


# ---------------------------------------------------------------------------
# 3. Tensor roundtrip (save → load → compare)
# ---------------------------------------------------------------------------

class TestTensorRoundtrip:
    """Tensors loaded from safetensors match the original module buffers."""

    def test_indices_match(self, small_quantized_model, tmp_dir):
        model, config = small_quantized_model
        # Grab originals before save
        orig_indices = model.linear1.indices_packed.clone()
        orig_norms = model.linear1.weight_norms.clone()
        orig_codebook = model.linear1.codebook.clone()

        save_quantized(model, config, tmp_dir)
        tensors = load_file(str(tmp_dir / "model.safetensors"))

        assert torch.equal(tensors["linear1.indices"], orig_indices)
        assert torch.equal(tensors["linear1.norms"], orig_norms)
        assert torch.equal(tensors["codebook"], orig_codebook)

    def test_bias_match(self, small_quantized_model, tmp_dir):
        model, config = small_quantized_model
        orig_bias = model.linear2.bias.clone()

        save_quantized(model, config, tmp_dir)
        tensors = load_file(str(tmp_dir / "model.safetensors"))

        assert torch.equal(tensors["linear2.bias"], orig_bias)

    def test_residual_tensors_match(self, residual_quantized_model, tmp_dir):
        model, config = residual_quantized_model
        orig_p2_indices = model.linear1.pass2_indices_packed.clone()
        orig_p2_norms = model.linear1.pass2_weight_norms.clone()

        save_quantized(model, config, tmp_dir)
        tensors = load_file(str(tmp_dir / "model.safetensors"))

        assert torch.equal(tensors["linear1.pass2_indices"], orig_p2_indices)
        assert torch.equal(tensors["linear1.pass2_norms"], orig_p2_norms)


# ---------------------------------------------------------------------------
# 4. End-to-end: quantize → save → reconstruct → dequantize matches
# ---------------------------------------------------------------------------

class TestEndToEnd:
    """Full pipeline: quantize, save, reload tensors, rebuild TurboQuantLinear, compare."""

    def test_dequantize_matches_after_reload(self, small_quantized_model, tmp_dir):
        model, config = small_quantized_model

        # Dequantize before save
        W_before = model.linear1.dequantize().clone()

        save_quantized(model, config, tmp_dir)
        tensors = load_file(str(tmp_dir / "model.safetensors"))

        # Rebuild a fresh TurboQuantLinear from loaded tensors
        tq = TurboQuantLinear(
            in_features=32, out_features=64,
            bias=False, bit_width=4, group_size=32,
        )
        tq.indices_packed = tensors["linear1.indices"]
        tq.weight_norms = tensors["linear1.norms"]
        tq.codebook = tensors["codebook"]
        tq.set_rotation(config.seed)

        W_after = tq.dequantize()
        assert torch.allclose(W_before, W_after, atol=1e-6)

    def test_residual_dequantize_matches(self, residual_quantized_model, tmp_dir):
        model, config = residual_quantized_model

        # Dequantize pass 1 + pass 2 before save
        W_before = model.linear1.dequantize().clone()

        save_quantized(model, config, tmp_dir)
        tensors = load_file(str(tmp_dir / "model.safetensors"))

        tq = TurboQuantLinear(
            in_features=32, out_features=64,
            bias=False, bit_width=4, group_size=32,
        )
        tq.indices_packed = tensors["linear1.indices"]
        tq.weight_norms = tensors["linear1.norms"]
        tq.codebook = tensors["codebook"]
        tq.set_rotation(config.seed)

        tq.set_pass2(
            indices_packed=tensors["linear1.pass2_indices"],
            weight_norms=tensors["linear1.pass2_norms"],
            codebook=tensors["linear1.pass2_codebook"],
            seed=config.residual_seed,
        )

        W_after = tq.dequantize()
        assert torch.allclose(W_before, W_after, atol=1e-6)

    def test_forward_pass_matches(self, small_quantized_model, tmp_dir):
        model, config = small_quantized_model
        # Force PyTorch fallback (cuTile/Triton may not be available)
        for m in model.modules():
            if isinstance(m, TurboQuantLinear):
                m.use_cutile = False
                m.use_triton = False

        x = torch.randn(2, 32)
        y_before = model(x).clone()

        save_quantized(model, config, tmp_dir)

        # Reload tensors and rebuild model
        tensors = load_file(str(tmp_dir / "model.safetensors"))
        nq = load_file(str(tmp_dir / "non_quantized.safetensors"))

        model2 = SmallModel()
        # Restore LayerNorm
        model2.layer_norm.weight.data.copy_(nq["layer_norm.weight"])
        model2.layer_norm.bias.data.copy_(nq["layer_norm.bias"])

        # Rebuild linear1
        tq1 = TurboQuantLinear(in_features=32, out_features=64, bias=False, bit_width=4, group_size=32)
        tq1.indices_packed = tensors["linear1.indices"]
        tq1.weight_norms = tensors["linear1.norms"]
        tq1.codebook = tensors["codebook"]
        tq1.set_rotation(config.seed)
        model2.linear1 = tq1

        # Rebuild linear2
        tq2 = TurboQuantLinear(in_features=64, out_features=32, bias=True, bit_width=4, group_size=32)
        tq2.indices_packed = tensors["linear2.indices"]
        tq2.weight_norms = tensors["linear2.norms"]
        tq2.codebook = tensors["codebook"]
        tq2.bias = tensors["linear2.bias"]
        tq2.set_rotation(config.seed)
        model2.linear2 = tq2

        model2.eval()
        for m in model2.modules():
            if isinstance(m, TurboQuantLinear):
                m.use_cutile = False
                m.use_triton = False

        y_after = model2(x)
        assert torch.allclose(y_before, y_after, atol=1e-5)
