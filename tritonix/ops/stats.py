"""
Standalone order-statistics kernels: topk, kth_largest, weighted_mid.

No tritonix dependencies — plain PyTorch + Triton.
Uses Triton's native @autotune for block_m / num_warps selection.
Launchers compute block_n from n_cols and prune invalid configs via
register-pressure constraints before Triton runs its search.
Falls back to PyTorch when n_cols > 1024 (can't fit even with 8 warps).

Register budget: MAX_REGS = 128 per thread, WARP_SIZE = 32.
Constraint: block_m * block_n / (num_warps * WARP_SIZE) <= MAX_REGS
"""

import math
import torch
import triton
import triton.language as tl
from torch import Tensor

_WARP_SIZE = 32
_MAX_REGS  = 128
_MAX_N     = _MAX_REGS * 8  # 1024: max block_n with 8 warps


# ---------------------------------------------------------------------------
# Pruning
# ---------------------------------------------------------------------------

def _prune_single_load(configs, named_args):
    """Keep only configs that satisfy single-load register constraint."""
    block_n = named_args["block_n"]
    valid = [
        cfg for cfg in configs
        if block_n // cfg.num_warps <= _MAX_REGS
        and cfg.kwargs["block_m"] * block_n // (cfg.num_warps * _WARP_SIZE) <= _MAX_REGS
    ]
    return valid or [configs[0]]


_CONFIGS = [
    triton.Config({"block_m": bm}, num_warps=nw)
    for bm in [1, 2, 4, 8, 16, 32, 64]
    for nw in [1, 2, 4, 8]
]


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------

@triton.autotune(configs=_CONFIGS, key=["m", "n", "k_val"], prune_configs_by={"perf_model": _prune_single_load})
@triton.jit
def _kth_largest_kernel(
    x_ptr, out_ptr, m, n, stride_m, stride_n, k_val,
    block_m: tl.constexpr, block_n: tl.constexpr, k_next_p2: tl.constexpr,
):
    pid  = tl.program_id(0)
    rows = pid * block_m + tl.arange(0, block_m)
    cols = tl.arange(0, block_n)
    mask = (rows[:, None] < m) & (cols[None, :] < n)
    vals = tl.load(x_ptr + rows[:, None] * stride_m + cols[None, :] * stride_n,
                   mask=mask, other=float("-inf"))
    topk = tl.topk(vals, k_next_p2)
    sel  = tl.arange(0, k_next_p2)[None, :]
    kth  = tl.sum(tl.where(sel == (k_val - 1), topk, 0.0), axis=1)
    tl.store(out_ptr + rows, kth.to(x_ptr.dtype.element_ty), mask=rows < m)


@triton.autotune(configs=_CONFIGS, key=["m", "n", "k_val"], prune_configs_by={"perf_model": _prune_single_load})
@triton.jit
def _topk_kernel(
    x_ptr, out_ptr, m, n, stride_m, stride_n, k_val,
    block_m: tl.constexpr, block_n: tl.constexpr, k_next_p2: tl.constexpr,
):
    pid    = tl.program_id(0)
    rows   = pid * block_m + tl.arange(0, block_m)
    cols   = tl.arange(0, block_n)
    mask   = (rows[:, None] < m) & (cols[None, :] < n)
    vals   = tl.load(x_ptr + rows[:, None] * stride_m + cols[None, :] * stride_n,
                     mask=mask, other=float("-inf"))
    topk   = tl.topk(vals, k_next_p2).to(x_ptr.dtype.element_ty)
    k_offs = tl.arange(0, k_next_p2)
    tl.store(out_ptr + rows[:, None] * k_val + k_offs[None, :], topk,
             mask=(rows[:, None] < m) & (k_offs[None, :] < k_val))


@triton.autotune(configs=_CONFIGS, key=["m", "n", "k_val"], prune_configs_by={"perf_model": _prune_single_load})
@triton.jit
def _mid_kernel(
    x_ptr, out_ptr, m, n, stride_m, stride_n, k_val,
    block_m: tl.constexpr, block_n: tl.constexpr, k_next_p2: tl.constexpr,
    k_weight: tl.constexpr = 1.0,
):
    pid  = tl.program_id(0)
    rows = pid * block_m + tl.arange(0, block_m)
    cols = tl.arange(0, block_n)
    mask = (rows[:, None] < m) & (cols[None, :] < n)
    vals   = tl.load(x_ptr + rows[:, None] * stride_m + cols[None, :] * stride_n,
                     mask=mask, other=float("-inf"))
    topk   = tl.topk(vals, k_next_p2)
    sel    = tl.arange(0, k_next_p2)[None, :]
    val_k  = tl.sum(tl.where(sel == (k_val - 1), topk, 0.0), axis=1)
    val_k1 = tl.sum(tl.where(sel == k_val,        topk, 0.0), axis=1)
    result = (k_weight * val_k + val_k1) / (1.0 + k_weight)
    tl.store(out_ptr + rows, result.to(x_ptr.dtype.element_ty), mask=rows < m)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prepare(x: Tensor, dim: int):
    dim = dim % x.ndim
    x = x.movedim(dim, -1).contiguous()
    n_cols = x.shape[-1]
    n_rows = x.numel() // n_cols
    return x.view(n_rows, n_cols), n_rows, n_cols, x.shape[:-1]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def kth_largest(x: Tensor, k: int, dim: int = -1) -> Tensor:
    assert 1 <= k <= x.shape[dim % x.ndim]
    flat, n_rows, n_cols, out_shape = _prepare(x, dim)
    block_n = triton.next_power_of_2(n_cols)

    if block_n > _MAX_N:
        out = torch.kthvalue(flat, n_cols - k + 1, dim=1).values
    else:
        out = torch.empty(n_rows, device=x.device, dtype=x.dtype)
        _kth_largest_kernel[lambda meta: (triton.cdiv(n_rows, meta["block_m"]),)](
            flat, out, n_rows, n_cols, flat.stride(0), flat.stride(1), k,
            block_n=block_n, k_next_p2=triton.next_power_of_2(k),
        )
    return out.view(out_shape) if out_shape else out.squeeze()


def topk(x: Tensor, k: int, dim: int = -1) -> Tensor:
    assert 1 <= k <= x.shape[dim % x.ndim]
    flat, n_rows, n_cols, out_shape = _prepare(x, dim)
    block_n = triton.next_power_of_2(n_cols)

    if block_n > _MAX_N:
        out = torch.topk(flat, k, dim=1).values
    else:
        out = torch.empty((n_rows, k), device=x.device, dtype=x.dtype)
        _topk_kernel[lambda meta: (triton.cdiv(n_rows, meta["block_m"]),)](
            flat, out, n_rows, n_cols, flat.stride(0), flat.stride(1), k,
            block_n=block_n, k_next_p2=triton.next_power_of_2(k),
        )
    return out.view(*out_shape, k) if out_shape else out


def weighted_mid(x: Tensor, k: int, dim: int = -1, weight: float = 1.0) -> Tensor:
    assert 1 <= k and k + 1 <= x.shape[dim % x.ndim]
    flat, n_rows, n_cols, out_shape = _prepare(x, dim)
    block_n = triton.next_power_of_2(n_cols)

    if block_n > _MAX_N:
        n   = flat.shape[1]
        v1  = torch.kthvalue(flat, n - k + 1, dim=1).values
        v2  = torch.kthvalue(flat, n - k,     dim=1).values
        out = (weight * v1 + v2) / (1.0 + weight)
    else:
        out = torch.empty(n_rows, device=x.device, dtype=x.dtype)
        _mid_kernel[lambda meta: (triton.cdiv(n_rows, meta["block_m"]),)](
            flat, out, n_rows, n_cols, flat.stride(0), flat.stride(1), k,
            block_n=block_n, k_next_p2=triton.next_power_of_2(k + 1),
            k_weight=weight,
        )
    return out.view(out_shape) if out_shape else out.squeeze()
