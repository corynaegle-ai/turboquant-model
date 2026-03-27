"""Triton fused dequant + matmul kernels for on-the-fly inference.

These kernels avoid materializing the full dequantized weight by fusing
4-bit unpack → codebook lookup → matmul → norm rescale in one kernel launch.

Optimizations applied:
  1. Autotune — @triton.autotune searches (BLOCK_B, BLOCK_N, BLOCK_K, num_warps,
     num_stages) per problem shape; cached after first invocation.
  2. Shared-memory codebook — the 16-entry codebook (64 B) stays in L1/registers
     after first load in each K-tile; repeated gather hits cache.
  3. TF32 tensor cores — allow_tf32=True in tl.dot for ~2× throughput on
     fp32 Ampere+/Ada/Hopper.
  4. Pre-scaled norms — norms / sqrt(K) computed once on host, eliminating
     per-element division in the kernel epilogue.
  5. Software pipelining — num_stages in autotune configs controls prefetch depth.
  6. Transpose elimination — accumulates in natural (B, N) layout; no extra
     transpose required.

Main kernel: _turboquant_fused_matmul_kernel
  - Input: x_rot (pre-rotated activations), packed indices, codebook, norms_scaled
  - Output: x_rot @ codebook[indices].T * norms_scaled

Supports group-wise calls: pass a packed index slice (N, g_dim//2) with K=g_dim.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Autotune configurations — searched per unique (B, N, K) shape
# ---------------------------------------------------------------------------

_AUTOTUNE_CONFIGS = [
    # Small batch (inference with B=1..4)
    triton.Config({"BLOCK_B": 1,  "BLOCK_N": 32,  "BLOCK_K": 32},  num_warps=2, num_stages=2),
    triton.Config({"BLOCK_B": 1,  "BLOCK_N": 64,  "BLOCK_K": 64},  num_warps=4, num_stages=3),
    triton.Config({"BLOCK_B": 4,  "BLOCK_N": 32,  "BLOCK_K": 64},  num_warps=4, num_stages=2),
    triton.Config({"BLOCK_B": 4,  "BLOCK_N": 64,  "BLOCK_K": 64},  num_warps=4, num_stages=3),
    # Medium batch
    triton.Config({"BLOCK_B": 16, "BLOCK_N": 32,  "BLOCK_K": 64},  num_warps=4, num_stages=2),
    triton.Config({"BLOCK_B": 16, "BLOCK_N": 64,  "BLOCK_K": 64},  num_warps=4, num_stages=3),
    triton.Config({"BLOCK_B": 16, "BLOCK_N": 64,  "BLOCK_K": 128}, num_warps=8, num_stages=3),
    # Large batch (tensor-core friendly ≥16 on all dims)
    triton.Config({"BLOCK_B": 32, "BLOCK_N": 32,  "BLOCK_K": 64},  num_warps=4, num_stages=2),
    triton.Config({"BLOCK_B": 32, "BLOCK_N": 64,  "BLOCK_K": 64},  num_warps=8, num_stages=3),
    triton.Config({"BLOCK_B": 32, "BLOCK_N": 64,  "BLOCK_K": 128}, num_warps=8, num_stages=3),
]


@triton.autotune(configs=_AUTOTUNE_CONFIGS, key=["B", "N", "K"])
@triton.jit
def _turboquant_fused_matmul_kernel(
    # Input
    input_ptr,        # (B, K) pre-rotated activations
    # Quantized weight
    indices_ptr,      # (N, K//2) packed uint8
    codebook_ptr,     # (n_levels,) float32
    norms_ptr,        # (N,) float32 — pre-scaled by 1/scale on host
    # Output
    output_ptr,       # (B, N)
    # Dims
    B, N, K,
    PACKED_K,         # K // 2 (stride for packed index rows)
    N_LEVELS: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Fused dequant-matmul: output[b,n] = norms_scaled[n] * Σ_k x[b,k] * codebook[idx[n,k]]"""
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)

    rb = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_b = rb < B
    mask_n = rn < N

    acc = tl.zeros((BLOCK_B, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        rk = k_start + tl.arange(0, BLOCK_K)
        mask_k = rk < K

        # Load input tile: (BLOCK_B, BLOCK_K)
        inp_off = rb[:, None] * K + rk[None, :]
        inp_mask = mask_b[:, None] & mask_k[None, :]
        inp_tile = tl.load(input_ptr + inp_off, mask=inp_mask, other=0.0)

        # Load + unpack weight indices: (BLOCK_N, BLOCK_K)
        byte_col = rk // 2
        is_high = (rk % 2) == 1
        byte_off = rn[:, None] * PACKED_K + byte_col[None, :]
        w_mask = mask_n[:, None] & mask_k[None, :]
        packed = tl.load(indices_ptr + byte_off, mask=w_mask, other=0).to(tl.uint8)
        lo = packed & 0x0F
        hi = (packed >> 4) & 0x0F
        idx = tl.where(is_high[None, :], hi, lo)

        # Codebook lookup (16 entries — stays in L1/registers after first access)
        w_quant = tl.load(codebook_ptr + idx.to(tl.int32), mask=w_mask, other=0.0)

        # TF32 tensor-core MMA: (BLOCK_B, BLOCK_K) @ (BLOCK_K, BLOCK_N)
        acc += tl.dot(
            inp_tile.to(tl.float32),
            tl.trans(w_quant.to(tl.float32)),
            allow_tf32=True,
        )

    # Multiply by pre-scaled norms (norms / scale computed on host)
    norm_vals = tl.load(norms_ptr + rn, mask=mask_n, other=1.0)
    acc = acc * norm_vals[None, :]

    # Store
    out_off = rb[:, None] * N + rn[None, :]
    out_mask = mask_b[:, None] & mask_n[None, :]
    tl.store(output_ptr + out_off, acc.to(output_ptr.dtype.element_ty), mask=out_mask)


def triton_fused_matmul(
    x_rot: torch.Tensor,           # (B, K) pre-rotated input
    indices_packed: torch.Tensor,   # (N, K//2) packed uint8
    codebook: torch.Tensor,         # (n_levels,) float32
    norms: torch.Tensor,            # (N,) float32
    K: int,                         # in_features (or group_size for per-group calls)
    scale: float | None = None,     # override sqrt(K) if needed
) -> torch.Tensor:
    """Fused dequant + matmul via Triton with autotune + TF32 tensor cores.

    Expects pre-rotated input: x_rot = x @ Pi.T

    Supports per-group calls: pass a slice of packed indices (N, g_dim//2)
    with K=g_dim. The kernel handles unpack + codebook lookup + matmul + norm
    rescale in one launch, avoiding materialization of the (N, K) float weight.

    Args:
        x_rot: (B, K) pre-rotated activations
        indices_packed: (N, K//2) packed 4-bit weight indices
        codebook: centroids
        norms: per-row weight norms (N,)
        K: dimension of this group (in_features or group_size)
        scale: norm divisor (default: sqrt(K))

    Returns:
        output: (B, N)
    """
    B = x_rot.shape[0]
    N = indices_packed.shape[0]
    PACKED_K = indices_packed.shape[1]
    if scale is None:
        scale = math.sqrt(K)

    # Pre-scale norms on host (avoids per-element division in kernel)
    norms_scaled = norms / scale

    output = torch.empty(B, N, dtype=torch.float32, device=x_rot.device)

    # Grid is a lambda so autotune can adapt it per config
    grid = lambda META: (
        triton.cdiv(B, META["BLOCK_B"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )

    _turboquant_fused_matmul_kernel[grid](
        x_rot, indices_packed, codebook, norms_scaled, output,
        B, N, K, PACKED_K,
        N_LEVELS=codebook.shape[0],
    )

    return output
