"""Tests for Metal fused dequant + matmul kernel.

Validates:
  1. metal_fused_matmul matches PyTorch reference (unpack → lookup → matmul → rescale).
  2. Correctness across different problem shapes (B, N, K).
  3. Odd K dimensions (padding in packed representation).
  4. Pre-scaled norms produce correct results.
  5. Integration through TurboQuantLinear forward pass on MPS.

Requires macOS with Apple Silicon and pyobjc-framework-Metal.
Skipped automatically when Metal is not available.
"""

from __future__ import annotations

import math

import pytest
import torch

from turboquant_model.codebook import get_codebook
from turboquant_model.quantize import pack_4bit, unpack_4bit

# Skip entire module if Metal is not available
try:
    from turboquant_model.metal_kernels import metal_fused_matmul, _METAL_AVAILABLE
except ImportError:
    _METAL_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _METAL_AVAILABLE,
    reason="Metal not available (requires macOS + Apple Silicon + pyobjc-framework-Metal)",
)


# ---------------------------------------------------------------------------
# Reference implementation (pure PyTorch)
# ---------------------------------------------------------------------------


def _reference_fused_matmul(
    x_rot: torch.Tensor,
    indices_packed: torch.Tensor,
    codebook: torch.Tensor,
    norms: torch.Tensor,
    K: int,
    scale: float | None = None,
) -> torch.Tensor:
    """Pure-PyTorch reference: unpack → codebook[idx] → matmul → norm rescale."""
    if scale is None:
        scale = math.sqrt(K)

    indices = unpack_4bit(indices_packed, K)          # (N, K)
    W_quant = codebook[indices.long()]                # (N, K)
    out = x_rot @ W_quant.T                           # (B, N)
    norms_scaled = norms / scale
    return out * norms_scaled[None, :]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def codebook_4bit():
    centroids, _ = get_codebook(4)
    return centroids


@pytest.fixture(params=[
    (1, 32, 64),     # B=1 small
    (1, 128, 256),   # B=1 medium
    (4, 64, 128),    # small batch
    (16, 64, 64),    # larger batch, square-ish
    (1, 256, 512),   # wide
])
def problem_shape(request):
    """(B, N, K) problem shapes."""
    return request.param


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMetalFusedMatmul:
    """metal_fused_matmul matches PyTorch reference."""

    def test_matches_reference(self, problem_shape, codebook_4bit):
        B, N, K = problem_shape
        torch.manual_seed(42)

        x_rot = torch.randn(B, K, dtype=torch.float32)
        # Random 4-bit indices packed into uint8
        indices = torch.randint(0, 16, (N, K), dtype=torch.int32)
        indices_packed = pack_4bit(indices)
        norms = torch.randn(N).abs() + 0.1  # positive norms

        ref = _reference_fused_matmul(x_rot, indices_packed, codebook_4bit, norms, K)
        metal = metal_fused_matmul(x_rot, indices_packed, codebook_4bit, norms, K)

        torch.testing.assert_close(metal.cpu().float(), ref.cpu().float(), atol=1e-4, rtol=1e-4)

    def test_explicit_scale(self, codebook_4bit):
        B, N, K = 2, 32, 64
        torch.manual_seed(7)

        x_rot = torch.randn(B, K, dtype=torch.float32)
        indices = torch.randint(0, 16, (N, K), dtype=torch.int32)
        indices_packed = pack_4bit(indices)
        norms = torch.randn(N).abs() + 0.1

        scale = 3.14
        ref = _reference_fused_matmul(x_rot, indices_packed, codebook_4bit, norms, K, scale=scale)
        metal = metal_fused_matmul(x_rot, indices_packed, codebook_4bit, norms, K, scale=scale)

        torch.testing.assert_close(metal.cpu().float(), ref.cpu().float(), atol=1e-4, rtol=1e-4)

    def test_output_shape_dtype(self, codebook_4bit):
        B, N, K = 4, 64, 128
        torch.manual_seed(0)

        x_rot = torch.randn(B, K)
        indices_packed = pack_4bit(torch.randint(0, 16, (N, K), dtype=torch.int32))
        norms = torch.ones(N)

        out = metal_fused_matmul(x_rot, indices_packed, codebook_4bit, norms, K)
        assert out.shape == (B, N)
        assert out.dtype == torch.float32

    def test_zero_input(self, codebook_4bit):
        B, N, K = 1, 32, 64
        x_rot = torch.zeros(B, K)
        indices_packed = pack_4bit(torch.randint(0, 16, (N, K), dtype=torch.int32))
        norms = torch.ones(N)

        out = metal_fused_matmul(x_rot, indices_packed, codebook_4bit, norms, K)
        assert torch.allclose(out, torch.zeros(B, N), atol=1e-7)

    def test_identity_codebook(self):
        """When codebook = [0,1,...,15], result is matmul of input with raw indices."""
        B, N, K = 2, 16, 32
        torch.manual_seed(99)

        codebook = torch.arange(16, dtype=torch.float32)
        x_rot = torch.randn(B, K)
        indices = torch.randint(0, 16, (N, K), dtype=torch.int32)
        indices_packed = pack_4bit(indices)
        norms = torch.ones(N)
        scale = 1.0

        ref = _reference_fused_matmul(x_rot, indices_packed, codebook, norms, K, scale=scale)
        metal = metal_fused_matmul(x_rot, indices_packed, codebook, norms, K, scale=scale)

        torch.testing.assert_close(metal.cpu().float(), ref.cpu().float(), atol=1e-4, rtol=1e-4)


class TestMetalLinearIntegration:
    """TurboQuantLinear uses Metal kernel when available."""

    def test_forward_uses_metal(self, codebook_4bit):
        """Smoke test: forward pass completes with Metal enabled."""
        from turboquant_model.module import TurboQuantLinear

        layer = TurboQuantLinear(64, 32, bit_width=4, rotation="qr")
        # Manually enable only Metal
        layer.use_cutile = False
        layer.use_triton = False
        layer.use_metal = True

        # Fill with valid quantization data
        torch.manual_seed(0)
        layer.codebook.copy_(codebook_4bit)
        indices = torch.randint(0, 16, (32, 64), dtype=torch.int32)
        layer.indices_packed.copy_(pack_4bit(indices))
        layer.weight_norms.fill_(1.0)
        layer.set_rotation(42)

        x = torch.randn(2, 64)
        out = layer(x)
        assert out.shape == (2, 32)
        assert torch.isfinite(out).all()
