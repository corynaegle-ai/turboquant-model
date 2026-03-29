"""Tests for hierarchical successive-refinement (HSR) codebook and merge.

Validates:
  1. Hierarchical codebook structure: sorted centroids, coarse+offset identity.
  2. Hierarchical codebook differs from flat codebook (structural constraint).
  3. merge_and_requantize_hierarchical output format and dequantize roundtrip.
  4. Data-dependent (fit_codebook=True) path produces valid codebook.
  5. Group-wise and odd-column edge cases.
  6. fit_hierarchical_codebook standalone correctness.
"""

from __future__ import annotations

import pytest
import torch

from turboquant_model.codebook import (
    get_codebook,
    get_hierarchical_codebook,
    fit_hierarchical_codebook,
)
from turboquant_model.residual import (
    multi_residual_quantize_packed,
    merge_residual_passes,
    merge_and_requantize,
    merge_and_requantize_hierarchical,
    _dequantize_from_packed,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def weight_matrix():
    """A small reproducible weight matrix for CPU tests."""
    torch.manual_seed(0)
    return torch.randn(16, 32)


@pytest.fixture()
def weight_matrix_odd():
    """Odd-column matrix to exercise padding paths."""
    torch.manual_seed(1)
    return torch.randn(8, 31)


@pytest.fixture()
def packed_3pass(weight_matrix):
    """3-pass shared-rotation packed representation."""
    return multi_residual_quantize_packed(
        weight_matrix, n_passes=3, bit_width=4, seed=42,
    )


# ---------------------------------------------------------------------------
# 1. Hierarchical codebook structure
# ---------------------------------------------------------------------------


class TestHierarchicalCodebook:
    """get_hierarchical_codebook returns a valid HSR codebook."""

    def test_output_shapes(self):
        fc, fb, cc, ro = get_hierarchical_codebook(3, 1)
        assert fc.shape == (16,)
        assert fb.shape == (15,)
        assert cc.shape == (8,)
        assert ro.shape == (8, 2)

    def test_centroids_sorted(self):
        fc, _, _, _ = get_hierarchical_codebook(3, 1)
        assert (fc[1:] > fc[:-1]).all(), "Flat centroids must be strictly sorted"

    def test_coarse_plus_offset_identity(self):
        """flat_centroids[2*i + j] == coarse_centroids[i] + refine_offsets[i, j]."""
        fc, _, cc, ro = get_hierarchical_codebook(3, 1)
        reconstructed = torch.zeros(16)
        for i in range(8):
            for j in range(2):
                reconstructed[i * 2 + j] = cc[i] + ro[i, j]
        assert torch.allclose(reconstructed, fc, atol=1e-6)

    def test_differs_from_flat(self):
        """HSR-4 centroids should differ from flat 4-bit Lloyd-Max."""
        fc_hsr, _, _, _ = get_hierarchical_codebook(3, 1)
        fc_flat, _ = get_codebook(4)
        assert not torch.allclose(fc_hsr, fc_flat, atol=1e-4)

    def test_symmetric(self):
        """For Gaussian, HSR centroids should be approximately symmetric."""
        fc, _, _, _ = get_hierarchical_codebook(3, 1)
        assert torch.allclose(fc, -fc.flip(0), atol=1e-4)

    @pytest.mark.parametrize("coarse_bits,refine_bits", [(2, 2), (3, 1), (1, 3)])
    def test_various_bit_splits(self, coarse_bits, refine_bits):
        total = coarse_bits + refine_bits
        n_total = 2**total
        fc, fb, cc, ro = get_hierarchical_codebook(coarse_bits, refine_bits)
        assert fc.shape == (n_total,)
        assert fb.shape == (n_total - 1,)
        assert cc.shape == (2**coarse_bits,)
        assert ro.shape == (2**coarse_bits, 2**refine_bits)
        assert (fc[1:] > fc[:-1]).all()

    def test_caching(self):
        """Repeated calls return the same object (cached)."""
        result1 = get_hierarchical_codebook(3, 1)
        result2 = get_hierarchical_codebook(3, 1)
        assert result1[0] is result2[0]


# ---------------------------------------------------------------------------
# 2. merge_and_requantize_hierarchical — Gaussian codebook
# ---------------------------------------------------------------------------


class TestMergeRequantizeHierarchical:
    """merge_and_requantize_hierarchical with pre-computed Gaussian codebook."""

    def test_output_format(self, packed_3pass):
        merged = merge_and_requantize_hierarchical(packed_3pass)
        # Standard packed keys
        assert "indices_packed" in merged
        assert "codebook" in merged
        assert "norms" in merged
        assert "seed" in merged
        assert merged["seed"] == 42
        assert merged["shape"] == (16, 32)
        assert merged["bit_width"] == 4
        # Hierarchical metadata
        assert "coarse_codebook" in merged
        assert "refine_offsets" in merged
        assert merged["coarse_bits"] == 3
        assert merged["refine_bits"] == 1
        assert merged["coarse_codebook"].shape == (8,)
        assert merged["refine_offsets"].shape == (8, 2)

    def test_dequant_roundtrip(self, packed_3pass):
        """Dequantized result should be finite and correct shape."""
        merged = merge_and_requantize_hierarchical(packed_3pass)
        W_deq = _dequantize_from_packed(merged, device=torch.device("cpu"))
        assert W_deq.shape == (16, 32)
        assert not torch.isnan(W_deq).any()
        assert not torch.isinf(W_deq).any()

    def test_better_than_random(self, weight_matrix, packed_3pass):
        """Merged HSR result should be closer to W than zero."""
        merged = merge_and_requantize_hierarchical(packed_3pass)
        W_deq = _dequantize_from_packed(merged, device=torch.device("cpu"))
        mse_merged = (weight_matrix - W_deq).pow(2).mean().item()
        mse_zero = weight_matrix.pow(2).mean().item()
        assert mse_merged < mse_zero

    def test_codebook_is_hierarchical(self, packed_3pass):
        """The output codebook should be the hierarchical codebook, not flat."""
        merged = merge_and_requantize_hierarchical(packed_3pass)
        fc_hsr, _, _, _ = get_hierarchical_codebook(3, 1)
        assert torch.allclose(merged["codebook"], fc_hsr)


# ---------------------------------------------------------------------------
# 3. merge_and_requantize_hierarchical — data-dependent (fit_codebook=True)
# ---------------------------------------------------------------------------


class TestMergeRequantizeHierarchicalFit:
    """merge_and_requantize_hierarchical with fit_codebook=True."""

    def test_output_format_fit(self, packed_3pass):
        merged = merge_and_requantize_hierarchical(
            packed_3pass, fit_codebook=True,
        )
        assert merged["codebook"].shape == (16,)
        assert merged["coarse_codebook"].shape == (8,)
        assert merged["refine_offsets"].shape == (8, 2)
        assert merged["bit_width"] == 4

    def test_dequant_roundtrip_fit(self, packed_3pass):
        merged = merge_and_requantize_hierarchical(
            packed_3pass, fit_codebook=True,
        )
        W_deq = _dequantize_from_packed(merged, device=torch.device("cpu"))
        assert W_deq.shape == (16, 32)
        assert not torch.isnan(W_deq).any()
        assert not torch.isinf(W_deq).any()

    def test_fit_codebook_differs_from_gaussian(self, packed_3pass):
        """Data-dependent codebook should differ from the pre-computed one."""
        merged_gauss = merge_and_requantize_hierarchical(
            packed_3pass, fit_codebook=False,
        )
        merged_fit = merge_and_requantize_hierarchical(
            packed_3pass, fit_codebook=True,
        )
        # Codebook values should be different (fitted to actual data)
        assert not torch.allclose(
            merged_gauss["codebook"], merged_fit["codebook"], atol=1e-4,
        )

    def test_better_than_random_fit(self, weight_matrix, packed_3pass):
        merged = merge_and_requantize_hierarchical(
            packed_3pass, fit_codebook=True,
        )
        W_deq = _dequantize_from_packed(merged, device=torch.device("cpu"))
        mse_merged = (weight_matrix - W_deq).pow(2).mean().item()
        mse_zero = weight_matrix.pow(2).mean().item()
        assert mse_merged < mse_zero

    def test_coarse_plus_offset_identity_fit(self, packed_3pass):
        """coarse + offset == flat centroids for data-dependent codebook."""
        merged = merge_and_requantize_hierarchical(
            packed_3pass, fit_codebook=True,
        )
        cc = merged["coarse_codebook"]
        ro = merged["refine_offsets"]
        fc = merged["codebook"]
        reconstructed = torch.zeros(16)
        for i in range(8):
            for j in range(2):
                reconstructed[i * 2 + j] = cc[i] + ro[i, j]
        assert torch.allclose(reconstructed, fc, atol=1e-6)


# ---------------------------------------------------------------------------
# 4. Group-wise and edge cases
# ---------------------------------------------------------------------------


class TestHierarchicalGroupWise:
    """HSR merge with group_size < N and odd dimensions."""

    def test_with_groups(self, weight_matrix):
        packed = multi_residual_quantize_packed(
            weight_matrix, n_passes=3, bit_width=4, group_size=16, seed=42,
        )
        merged = merge_and_requantize_hierarchical(packed)
        assert merged["group_size"] == 16
        W_deq = _dequantize_from_packed(merged, device=torch.device("cpu"))
        assert W_deq.shape == weight_matrix.shape
        assert not torch.isnan(W_deq).any()

    def test_with_groups_fit(self, weight_matrix):
        packed = multi_residual_quantize_packed(
            weight_matrix, n_passes=2, bit_width=4, group_size=16, seed=42,
        )
        merged = merge_and_requantize_hierarchical(packed, fit_codebook=True)
        assert merged["group_size"] == 16
        W_deq = _dequantize_from_packed(merged, device=torch.device("cpu"))
        assert W_deq.shape == weight_matrix.shape

    def test_odd_columns(self, weight_matrix_odd):
        packed = multi_residual_quantize_packed(
            weight_matrix_odd, n_passes=2, bit_width=4, seed=42,
        )
        merged = merge_and_requantize_hierarchical(packed)
        W_deq = _dequantize_from_packed(merged, device=torch.device("cpu"))
        assert W_deq.shape == weight_matrix_odd.shape


# ---------------------------------------------------------------------------
# 5. fit_hierarchical_codebook standalone
# ---------------------------------------------------------------------------


class TestFitHierarchicalCodebook:
    """fit_hierarchical_codebook learns a valid codebook from data."""

    def test_output_shapes(self):
        torch.manual_seed(42)
        values = torch.randn(5000)
        fc, fb, cc, ro = fit_hierarchical_codebook(values, coarse_bits=3, refine_bits=1)
        assert fc.shape == (16,)
        assert fb.shape == (15,)
        assert cc.shape == (8,)
        assert ro.shape == (8, 2)

    def test_centroids_sorted(self):
        torch.manual_seed(42)
        values = torch.randn(5000)
        fc, _, _, _ = fit_hierarchical_codebook(values)
        assert (fc[1:] > fc[:-1]).all()

    def test_coarse_plus_offset_identity(self):
        torch.manual_seed(42)
        values = torch.randn(5000)
        fc, _, cc, ro = fit_hierarchical_codebook(values)
        reconstructed = torch.zeros(16)
        for i in range(8):
            for j in range(2):
                reconstructed[i * 2 + j] = cc[i] + ro[i, j]
        assert torch.allclose(reconstructed, fc, atol=1e-5)

    def test_adapts_to_non_gaussian(self):
        """Codebook fitted on uniform data should differ from Gaussian-based."""
        torch.manual_seed(0)
        values = torch.rand(5000) * 6 - 3  # uniform [-3, 3]
        fc_fit, _, _, _ = fit_hierarchical_codebook(values)
        fc_gauss, _, _, _ = get_hierarchical_codebook(3, 1)
        assert not torch.allclose(fc_fit, fc_gauss, atol=0.1)

    @pytest.mark.parametrize("coarse_bits,refine_bits", [(2, 2), (1, 1)])
    def test_various_bit_splits(self, coarse_bits, refine_bits):
        torch.manual_seed(42)
        values = torch.randn(3000)
        n_total = 2 ** (coarse_bits + refine_bits)
        fc, fb, cc, ro = fit_hierarchical_codebook(
            values, coarse_bits=coarse_bits, refine_bits=refine_bits,
        )
        assert fc.shape == (n_total,)
        assert fb.shape == (n_total - 1,)
        assert (fc[1:] > fc[:-1]).all()
