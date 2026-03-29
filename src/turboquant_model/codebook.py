"""Lloyd-Max optimal codebook computation for Gaussian distribution.

Includes both flat (standard) and hierarchical successive-refinement (HSR)
codebooks, as well as data-dependent codebook fitting via empirical Lloyd-Max.
"""

from __future__ import annotations

import numpy as np
import torch


def _compute_lloyd_max_gaussian(
    n_levels: int, n_iters: int = 200
) -> tuple[np.ndarray, np.ndarray]:
    """Compute optimal Lloyd-Max codebook for N(0,1) distribution.

    Returns:
        centroids: (n_levels,) — sorted codebook centroids
        boundaries: (n_levels+1,) — quantization boundaries
    """
    from scipy.stats import norm

    sigma = 1.0
    boundaries = np.linspace(-3.5 * sigma, 3.5 * sigma, n_levels + 1)
    boundaries[0] = -1e10
    boundaries[-1] = 1e10
    centroids = np.zeros(n_levels)

    for _ in range(n_iters):
        for i in range(n_levels):
            lo, hi = boundaries[i], boundaries[i + 1]
            p = norm.cdf(hi) - norm.cdf(lo)
            if p > 1e-15:
                centroids[i] = (norm.pdf(lo) - norm.pdf(hi)) / p
            else:
                centroids[i] = (max(lo, -3.5) + min(hi, 3.5)) / 2

        for i in range(1, n_levels):
            boundaries[i] = (centroids[i - 1] + centroids[i]) / 2

    return centroids, boundaries


_CODEBOOK_CACHE: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}


def get_codebook(bit_width: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Get precomputed Lloyd-Max codebook for given bit-width.

    Args:
        bit_width: bits per element (1, 2, 3, 4, ...)

    Returns:
        centroids: (2^bit_width,) float32 tensor
        boundaries: (2^bit_width - 1,) float32 tensor (inner boundaries only)
    """
    if bit_width not in _CODEBOOK_CACHE:
        n_levels = 2**bit_width
        centroids, boundaries = _compute_lloyd_max_gaussian(n_levels)
        _CODEBOOK_CACHE[bit_width] = (
            torch.tensor(centroids, dtype=torch.float32),
            torch.tensor(boundaries[1:-1], dtype=torch.float32),
        )
    return _CODEBOOK_CACHE[bit_width]


# ---------------------------------------------------------------------------
# Hierarchical successive-refinement (HSR) codebook
# ---------------------------------------------------------------------------


def _compute_hierarchical_lloyd_max_gaussian(
    coarse_bits: int = 3,
    refine_bits: int = 1,
    n_iters: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute hierarchical successive-refinement codebook for N(0,1).

    Two-stage Lloyd-Max:
      1. Coarse: ``2^coarse_bits`` levels on the full N(0,1) distribution.
      2. Refine: within each coarse Voronoi region, ``2^refine_bits`` sub-levels
         via Lloyd-Max on the truncated Gaussian.

    The flat centroid at index ``coarse_id * n_refine + refine_id`` equals
    ``coarse_centroid[coarse_id] + refine_offset[coarse_id, refine_id]``.

    Returns:
        flat_centroids: (n_total,) sorted reconstruction values
        flat_boundaries: (n_total - 1,) inner boundaries for ``searchsorted``
        coarse_centroids: (n_coarse,) coarse stage centroids
        refine_offsets: (n_coarse, n_refine) offset from coarse centroid per child
    """
    from scipy.stats import norm

    n_coarse = 2**coarse_bits
    n_refine = 2**refine_bits
    n_total = n_coarse * n_refine

    # Stage 1: Coarse Lloyd-Max on full N(0,1)
    coarse_centroids, coarse_full_bnd = _compute_lloyd_max_gaussian(n_coarse, n_iters)

    # Stage 2: Refine within each coarse region
    flat_centroids = np.zeros(n_total)
    refine_offsets = np.zeros((n_coarse, n_refine))

    for i in range(n_coarse):
        lo = coarse_full_bnd[i]
        hi = coarse_full_bnd[i + 1]

        if n_refine == 1:
            flat_centroids[i] = coarse_centroids[i]
            refine_offsets[i, 0] = 0.0
            continue

        # Lloyd-Max with n_refine levels on N(0,1) truncated to [lo, hi]
        sub_bnd = np.linspace(lo, hi, n_refine + 1)
        sub_ctr = np.zeros(n_refine)

        for _ in range(n_iters):
            for j in range(n_refine):
                s_lo, s_hi = sub_bnd[j], sub_bnd[j + 1]
                p = norm.cdf(s_hi) - norm.cdf(s_lo)
                if p > 1e-15:
                    sub_ctr[j] = (norm.pdf(s_lo) - norm.pdf(s_hi)) / p
                else:
                    sub_ctr[j] = (max(s_lo, -3.5) + min(s_hi, 3.5)) / 2
            for j in range(1, n_refine):
                sub_bnd[j] = (sub_ctr[j - 1] + sub_ctr[j]) / 2

        for j in range(n_refine):
            flat_centroids[i * n_refine + j] = sub_ctr[j]
            refine_offsets[i, j] = sub_ctr[j] - coarse_centroids[i]

    # Inner boundaries from sorted flat centroids (nearest-centroid rule)
    flat_boundaries = (flat_centroids[:-1] + flat_centroids[1:]) / 2

    return flat_centroids, flat_boundaries, coarse_centroids, refine_offsets


_HIERARCHICAL_CACHE: dict[
    tuple[int, int],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
] = {}


def get_hierarchical_codebook(
    coarse_bits: int = 3, refine_bits: int = 1
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Get precomputed hierarchical successive-refinement codebook.

    The HSR codebook decomposes a ``(coarse_bits + refine_bits)``-bit codebook
    into a coarse stage and a refinement stage.  The coarse stage partitions
    N(0,1) into ``2^coarse_bits`` regions; the refinement stage further
    subdivides each region into ``2^refine_bits`` sub-regions.

    Dequantization with the flat codebook is the same as the standard path::

        reconstructed = flat_centroids[index]

    Equivalently, using the hierarchical structure::

        coarse_id = index // (2 ** refine_bits)
        refine_id = index %  (2 ** refine_bits)
        reconstructed = coarse_centroids[coarse_id] + refine_offsets[coarse_id, refine_id]

    Args:
        coarse_bits: bits for the coarse stage (default 3)
        refine_bits: bits for the refinement stage (default 1)

    Returns:
        flat_centroids: (2^(coarse_bits+refine_bits),) float32 — sorted centroids
        flat_boundaries: (2^(coarse_bits+refine_bits) - 1,) float32 — inner boundaries
        coarse_centroids: (2^coarse_bits,) float32
        refine_offsets: (2^coarse_bits, 2^refine_bits) float32
    """
    key = (coarse_bits, refine_bits)
    if key not in _HIERARCHICAL_CACHE:
        fc, fb, cc, ro = _compute_hierarchical_lloyd_max_gaussian(
            coarse_bits, refine_bits
        )
        _HIERARCHICAL_CACHE[key] = (
            torch.tensor(fc, dtype=torch.float32),
            torch.tensor(fb, dtype=torch.float32),
            torch.tensor(cc, dtype=torch.float32),
            torch.tensor(ro, dtype=torch.float32),
        )
    return _HIERARCHICAL_CACHE[key]


# ---------------------------------------------------------------------------
# Data-dependent codebook fitting (empirical Lloyd-Max)
# ---------------------------------------------------------------------------


def _lloyd_max_1d_empirical(
    values: torch.Tensor, n_levels: int, n_iters: int = 50
) -> tuple[torch.Tensor, torch.Tensor]:
    """1D Lloyd-Max on empirical data (scalar K-means with sorted centroids).

    Args:
        values: 1-D float tensor
        n_levels: number of quantization levels
        n_iters: number of iterations

    Returns:
        centroids: (n_levels,) sorted
        boundaries: (n_levels + 1,) with ±1e10 at edges
    """
    values = values.float().reshape(-1)

    # Initialize boundaries at quantiles
    quantiles = torch.linspace(0, 1, n_levels + 1, device=values.device)
    boundaries = torch.quantile(values, quantiles)
    boundaries[0] = -1e10
    boundaries[-1] = 1e10

    centroids = torch.zeros(n_levels, dtype=torch.float32, device=values.device)

    for _ in range(n_iters):
        # Assign values to regions via searchsorted on inner boundaries
        assignments = torch.bucketize(values, boundaries[1:-1])

        # Update centroids
        for i in range(n_levels):
            mask = assignments == i
            if mask.sum() > 0:
                centroids[i] = values[mask].mean()
            else:
                centroids[i] = ((boundaries[i] + boundaries[i + 1]) / 2).clamp(-10, 10)

        # Update inner boundaries as midpoints of adjacent centroids
        for i in range(1, n_levels):
            boundaries[i] = (centroids[i - 1] + centroids[i]) / 2

    return centroids, boundaries


@torch.no_grad()
def fit_hierarchical_codebook(
    values: torch.Tensor,
    coarse_bits: int = 3,
    refine_bits: int = 1,
    n_iters: int = 50,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit a hierarchical codebook from empirical data via Lloyd-Max.

    Instead of assuming N(0,1), this function learns the coarse and
    refinement centroids directly from the given values (e.g., the merged
    rotated-domain target ỹ from multi-pass residual quantization).

    Args:
        values: 1-D float tensor of scaled values to fit
        coarse_bits: bits for the coarse stage (default 3)
        refine_bits: bits for the refinement stage (default 1)
        n_iters: Lloyd-Max iterations per stage

    Returns:
        flat_centroids: (2^(coarse_bits+refine_bits),) float32 — sorted
        flat_boundaries: (2^(coarse_bits+refine_bits) - 1,) float32
        coarse_centroids: (2^coarse_bits,) float32
        refine_offsets: (2^coarse_bits, 2^refine_bits) float32
    """
    values = values.float().reshape(-1)
    n_coarse = 2**coarse_bits
    n_refine = 2**refine_bits
    n_total = n_coarse * n_refine

    # Stage 1: Coarse Lloyd-Max on all values
    coarse_centroids, coarse_boundaries = _lloyd_max_1d_empirical(
        values, n_coarse, n_iters
    )

    # Stage 2: Refine within each coarse region
    flat_centroids = torch.zeros(n_total, dtype=torch.float32)
    refine_offsets = torch.zeros(n_coarse, n_refine, dtype=torch.float32)

    coarse_assignments = torch.bucketize(values, coarse_boundaries[1:-1])

    for i in range(n_coarse):
        region_values = values[coarse_assignments == i]

        if n_refine == 1 or region_values.numel() < n_refine:
            flat_centroids[i * n_refine] = coarse_centroids[i]
            refine_offsets[i, 0] = 0.0
            if n_refine > 1:
                for j in range(1, n_refine):
                    flat_centroids[i * n_refine + j] = coarse_centroids[i]
                    refine_offsets[i, j] = 0.0
            continue

        sub_centroids, _ = _lloyd_max_1d_empirical(region_values, n_refine, n_iters)

        for j in range(n_refine):
            flat_centroids[i * n_refine + j] = sub_centroids[j]
            refine_offsets[i, j] = sub_centroids[j] - coarse_centroids[i]

    # Inner boundaries from sorted flat centroids
    flat_boundaries = (flat_centroids[:-1] + flat_centroids[1:]) / 2

    return flat_centroids, flat_boundaries, coarse_centroids, refine_offsets
