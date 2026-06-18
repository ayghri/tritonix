import triton
import triton.language as tl
import torch
from torch import Tensor

from tritonix.utils.spaces import PowerOfTwo, Range, Choice
from tritonix.autotune import tunable
from tritonix.utils.torch import reduce_dim_strides
from tritonix.dispatcher import dynamic_dispatch

# ---------------------------------------------------------------------------
# Kernel helpers
# ---------------------------------------------------------------------------


@triton.jit
def _get_topmask_and_fullmask(x):
    tl.static_assert(x.dtype.is_int_unsigned(), "must be passed as unsigned bits")
    tm: tl.constexpr = 1 << (-1 + x.dtype.primitive_bitwidth)
    fm: tl.constexpr = (1 << x.dtype.primitive_bitwidth) - 1
    return tl.full(x.shape, tm, dtype=x.dtype), tl.full(x.shape, fm, dtype=x.dtype)


@triton.jit
def _fpval_to_key(x):
    tm, fm = _get_topmask_and_fullmask(x)
    return x ^ tl.where((x & tm) != 0, fm, tm)


@triton.jit
def _key_to_fpval(x):
    tm, fm = _get_topmask_and_fullmask(x)
    return x ^ tl.where((x & tm) == 0, fm, tm)


# ---------------------------------------------------------------------------
# Single-load kernels  (n fits in registers: block_n = next_power_of_2(n))
# ---------------------------------------------------------------------------


@tunable(
    keys=["m", "n", "k_val"],
    space={
        "block_m": PowerOfTwo(4, 128),
        "num_warps": Choice([1, 2, 4, 8]),
    },
    memory_params={"block_m"},
)
@triton.jit
def _topk_kernel(
    x_ptr,
    out_ptr,
    m,
    n,
    stride_m,
    stride_n,
    k_val: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    k_next_p2: tl.constexpr,
):
    """Single-load top-k: writes all k values per row (descending order)."""
    group_idx = tl.program_id(0)
    tl.assume(n <= block_n)
    col_offsets = tl.arange(0, block_n)
    row_indices = group_idx * block_m + tl.arange(0, block_m)
    ptrs = row_indices[:, None] * stride_m + col_offsets[None, :] * stride_n
    row_mask = row_indices < m
    mask = row_mask[:, None] & (col_offsets[None, :] < n)
    x_vals = tl.load(x_ptr + ptrs, mask=mask, other=float("-inf"))
    topk_vals = tl.topk(x_vals, k_next_p2)
    k_offsets = tl.arange(0, k_next_p2)
    out_offsets = row_indices[:, None] * k_val + k_offsets[None, :]
    out_mask = row_mask[:, None] & (k_offsets[None, :] < k_val)
    tl.store(out_ptr + out_offsets, topk_vals, mask=out_mask)


@tunable(
    keys=["m", "n", "k_val"],
    space={
        "block_m": PowerOfTwo(4, 256),
        "num_warps": Choice([1, 2, 4, 8]),
    },
    memory_params={"block_m"},
)
@triton.jit
def _kth_largest_kernel(
    x_ptr,
    out_ptr,
    m,
    n,
    stride_m,
    stride_n,
    k_val: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    k_next_p2: tl.constexpr,
):
    """Single-load k-th largest. Requires n <= block_n."""
    group_idx = tl.program_id(0)
    # --- compiler hints
    tl.assume(n <= block_n)

    offsets_n = tl.arange(0, block_n)
    offsets_m = group_idx * block_m + tl.arange(0, block_m)
    mask_m = offsets_m < m

    mask = mask_m[:, None] & (offsets_n[None, :] < n)

    x_vals = tl.load(
        x_ptr + offsets_m[:, None] * stride_m + offsets_n[None, :] * stride_n,
        mask=mask,
        other=float("-inf"),
    )

    topk_vals = tl.topk(x_vals, k_next_p2)
    select = tl.arange(0, k_next_p2)[None, :]
    kth_val = tl.sum(tl.where(select == (k_val - 1), topk_vals, 0.0), axis=1)

    tl.store(out_ptr + offsets_m, kth_val.to(x_ptr.dtype.element_ty), mask=mask_m)


@tunable(
    keys=["m", "n", "k_val"],
    space={
        "block_m": PowerOfTwo(4, 256),
        "num_warps": Choice([1, 2, 4, 8]),
    },
    memory_params={"block_m"},
)
@triton.jit
def _mid_kth_largest_kernel(
    x_ptr,
    out_ptr,
    m,
    n,
    stride_m,
    stride_n,
    k_val: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    k_next_p2: tl.constexpr,
    k_weight: tl.constexpr = 1.0,
):
    """Single-load midpoint of k-th and (k+1)-th largest."""
    group_idx = tl.program_id(0)
    col_offsets = tl.arange(0, block_n)
    row_indices = group_idx * block_m + tl.arange(0, block_m)
    ptrs = row_indices[:, None] * stride_m + col_offsets[None, :] * stride_n
    row_mask = row_indices < m
    mask = row_mask[:, None] & (col_offsets[None, :] < n)
    data = tl.load(x_ptr + ptrs, mask=mask, other=float("-inf"))
    topk_vals = tl.topk(data, k_next_p2)
    select = tl.arange(0, k_next_p2)[None, :]
    val_k = tl.sum(tl.where(select == (k_val - 1), topk_vals, 0.0), axis=1)
    val_k1 = tl.sum(tl.where(select == k_val, topk_vals, 0.0), axis=1)
    mid = (k_weight * val_k + val_k1) / (1.0 + k_weight)
    tl.store(out_ptr + row_indices, mid, mask=row_mask)


# ---------------------------------------------------------------------------
# Streaming kernels  (chunk-wise for large n)
# ---------------------------------------------------------------------------


@tunable(
    keys=["m", "n", "k_val"],
    space={
        "block_m": PowerOfTwo(2, 64),
        "block_n": PowerOfTwo(64, 1024),
        "num_warps": PowerOfTwo(1, 8),
        "num_stages": Range(1, 3),
    },
    memory_params={"block_m", "block_n", "num_stages"},
)
@triton.jit
def _streaming_kth_largest_kernel(
    x_ptr,
    out_ptr,
    m,
    n,
    stride_m,
    stride_n,
    k_val: tl.constexpr,
    k_next_p2: tl.constexpr,
    double_next_p2: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    n_chunks: tl.constexpr,
):
    group_idx = tl.program_id(0)
    tl.assume(n > k_next_p2)
    row_indices = group_idx * block_m + tl.arange(0, block_m)
    row_mask = row_indices < m
    buf = tl.full([block_m, k_next_p2], float("-inf"), dtype=tl.float32)
    for c in range(n_chunks):
        col_offsets = c * block_n + tl.arange(0, block_n)
        mask = row_mask[:, None] & (col_offsets[None, :] < n)
        ptrs = row_indices[:, None] * stride_m + col_offsets[None, :] * stride_n
        chunk = tl.load(x_ptr + ptrs, mask=mask, other=float("-inf"))
        local_topk = tl.topk(chunk, k_next_p2)
        joined = tl.join(buf, local_topk)
        combined = tl.reshape(joined, [block_m, double_next_p2])
        buf = tl.topk(combined, k_next_p2)
    select = tl.arange(0, k_next_p2)[None, :]
    kth_val = tl.sum(tl.where(select == (k_val - 1), buf, 0.0), axis=1)
    tl.store(out_ptr + row_indices, kth_val.to(x_ptr.dtype.element_ty), mask=row_mask)


@tunable(
    keys=["m", "n", "k_val"],
    space={
        "block_m": PowerOfTwo(2, 64),
        "block_n": PowerOfTwo(64, 1024),
        "num_warps": Choice([1, 2, 4, 8]),
        "num_stages": Range(1, 3),
    },
    memory_params={"block_m", "block_n", "num_stages"},
)
@triton.jit
def _streaming_mid_kth_largest_kernel(
    x_ptr,
    out_ptr,
    m,
    n,
    stride_m,
    stride_n,
    k_val: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    k_next_p2: tl.constexpr,
    n_chunks: tl.constexpr,
    k_weight: tl.constexpr = 1.0,
):
    group_idx = tl.program_id(0)
    row_indices = group_idx * block_m + tl.arange(0, block_m)
    row_mask = row_indices < m
    buf = tl.full([block_m, k_next_p2], float("-inf"), dtype=tl.float32)
    for c in range(n_chunks):
        col_offsets = c * block_n + tl.arange(0, block_n)
        mask = row_mask[:, None] & (col_offsets[None, :] < n)
        ptrs = row_indices[:, None] * stride_m + col_offsets[None, :] * stride_n
        chunk = tl.load(x_ptr + ptrs, mask=mask, other=float("-inf"))
        local_topk = tl.topk(chunk, k_next_p2)
        joined = tl.join(buf, local_topk)
        combined = tl.reshape(joined, [block_m, 2 * k_next_p2])
        buf = tl.topk(combined, k_next_p2)
    select = tl.arange(0, k_next_p2)[None, :]
    val_k = tl.sum(tl.where(select == (k_val - 1), buf, 0.0), axis=1)
    val_k1 = tl.sum(tl.where(select == k_val, buf, 0.0), axis=1)
    result = (k_weight * val_k + val_k1) / (1.0 + k_weight)
    tl.store(out_ptr + row_indices, result.to(x_ptr.dtype.element_ty), mask=row_mask)


@triton.jit
def _radix_kth_largest_kernel(
    x_ptr,
    out_ptr,
    m,
    n,
    stride_m,
    stride_n,
    k_val: tl.constexpr,
    block_n: tl.constexpr,
    block_m: tl.constexpr,
    n_pad: tl.constexpr,
    n_act: tl.constexpr,
):
    """Radix-key streaming k-th largest via sortable-uint packing."""
    x_nbits: tl.constexpr = x_ptr.dtype.element_ty.primitive_bitwidth
    x_utype: tl.constexpr = tl.dtype(f"uint{x_nbits}")
    y_nbits: tl.constexpr = x_nbits * 2 if x_nbits >= 16 else 32
    x_ultype: tl.constexpr = tl.dtype(f"uint{y_nbits}")
    x_dtype: tl.constexpr = x_ptr.dtype.element_ty

    pid = tl.program_id(0)
    offs_m = pid * block_m + tl.arange(0, block_m)
    mask_m = offs_m[:, None] < m

    loop_iterations: tl.constexpr = n_pad // block_n - 1
    offs_n = loop_iterations * block_n + tl.arange(0, block_n)
    mask_n = offs_n[None, :] < n

    x_ptrs = x_ptr + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n
    x = tl.load(x_ptrs, mask=(mask_m & mask_n), other=float("-inf"))
    x = _fpval_to_key(x.to(x_utype, bitcast=True))
    x = (x.to(x_ultype) << 16) | (n_pad - offs_n)[None, :]
    acc = tl.topk(x, n_act, dim=1)

    for _ in (tl.static_range if loop_iterations <= 4 else range)(loop_iterations):
        acc = tl.bitonic_merge(acc)
        x_ptrs -= block_n * stride_n
        offs_n -= block_n
        x = tl.load(x_ptrs, mask=mask_m, other=float("-inf"))
        x = _fpval_to_key(x.to(x_utype, bitcast=True))
        x = (x.to(x_ultype) << 16) | (n_pad - offs_n)[None, :]
        acc = tl.maximum(acc, tl.topk(x, n_act, dim=1))

    acc = tl.sort(acc, dim=1, descending=True)
    y_values_raw = (acc >> 16).to(x_utype)
    y_values = _key_to_fpval(y_values_raw).to(x_dtype, bitcast=True)
    select = tl.arange(0, n_act)[None, :]
    kth_val = tl.sum(tl.where(select == (k_val - 1), y_values, 0.0), axis=1)
    tl.store(out_ptr + offs_m, kth_val, mask=offs_m < m)


_cfg: dict = {}  # keyed by (kernel, n_rows, n_cols, k)


def _launch_topk(x_flat, n_rows, n_cols, sr, sc, k, cfg):
    out = torch.empty((n_rows, k), device=x_flat.device, dtype=x_flat.dtype)
    grid = (triton.cdiv(n_rows, cfg["block_m"]),)
    _topk_kernel[grid](
        x_flat,
        out,
        n_rows,
        n_cols,
        sr,
        sc,
        k,
        block_m=cfg["block_m"],
        block_n=triton.next_power_of_2(n_cols),
        k_next_p2=triton.next_power_of_2(k),
        num_warps=cfg["num_warps"],
    )
    return out


def _launch_kth(x_flat, n_rows, n_cols, sr, sc, k, cfg):
    block_n = triton.next_power_of_2(n_cols)
    if block_n // cfg["num_warps"] > 128:
        raise ValueError(
            f"register pressure too high: block_n={block_n} // num_warps={cfg['num_warps']} > 128"
        )
    out = torch.empty(n_rows, device=x_flat.device, dtype=x_flat.dtype)
    grid = (triton.cdiv(n_rows, cfg["block_m"]),)
    _kth_largest_kernel[grid](
        x_flat,
        out,
        n_rows,
        n_cols,
        sr,
        sc,
        k,
        block_m=cfg["block_m"],
        block_n=triton.next_power_of_2(n_cols),
        k_next_p2=triton.next_power_of_2(k),
        num_warps=cfg["num_warps"],
    )
    return out


def _launch_kth_streaming(x_flat, n_rows, n_cols, sr, sc, k, cfg):
    k_next_p2 = triton.next_power_of_2(k)
    regs_per_thread = 2 * k_next_p2 * cfg["block_m"] // (cfg["num_warps"] * 32)
    if regs_per_thread > 128:
        raise ValueError(
            f"register pressure too high: {regs_per_thread} > 128 "
            f"(k_next_p2={k_next_p2}, block_m={cfg['block_m']}, num_warps={cfg['num_warps']})"
        )
    out = torch.empty(n_rows, device=x_flat.device, dtype=x_flat.dtype)
    grid = (triton.cdiv(n_rows, cfg["block_m"]),)
    _streaming_kth_largest_kernel[grid](
        x_flat,
        out,
        n_rows,
        n_cols,
        sr,
        sc,
        k,
        k_next_p2=k_next_p2,
        double_next_p2=2 * k_next_p2,
        block_m=cfg["block_m"],
        block_n=cfg["block_n"],
        n_chunks=triton.cdiv(n_cols, cfg["block_n"]),
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
    )
    return out


def _launch_mid(x_flat, n_rows, n_cols, sr, sc, k, k_weight, cfg):
    out = torch.empty(n_rows, device=x_flat.device, dtype=x_flat.dtype)
    grid = (triton.cdiv(n_rows, cfg["block_m"]),)
    _mid_kth_largest_kernel[grid](
        x_flat,
        out,
        n_rows,
        n_cols,
        sr,
        sc,
        k,
        block_m=cfg["block_m"],
        block_n=triton.next_power_of_2(n_cols),
        k_next_p2=triton.next_power_of_2(k + 1),
        k_weight=k_weight,
        num_warps=cfg["num_warps"],
    )
    return out


def _launch_mid_streaming(x_flat, n_rows, n_cols, sr, sc, k, cfg, k_weight=1.0):
    k_next_p2 = triton.next_power_of_2(k + 1)
    out = torch.empty(n_rows, device=x_flat.device, dtype=x_flat.dtype)
    grid = (triton.cdiv(n_rows, cfg["block_m"]),)
    _streaming_mid_kth_largest_kernel[grid](
        x_flat,
        out,
        n_rows,
        n_cols,
        sr,
        sc,
        k,
        block_m=cfg["block_m"],
        block_n=cfg["block_n"],
        k_next_p2=k_next_p2,
        n_chunks=triton.cdiv(n_cols, cfg["block_n"]),
        k_weight=k_weight,
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
    )
    return out


_NO_VALID_CONFIG = object()


def _tune(kernel, launcher, n_rows, n_cols, k):
    key = (id(kernel), n_rows, n_cols, k)
    if key not in _cfg:
        result = kernel.tune(launcher)
        _cfg[key] = result if result is not None else _NO_VALID_CONFIG
    result = _cfg[key]
    if result is _NO_VALID_CONFIG:
        raise ValueError(
            f"No valid config for {kernel.kernel.__name__} (n={n_rows}, k={k})"
        )
    return result


def triton_topk(x: Tensor, k: int, dim: int = -1) -> Tensor:
    dim = dim % x.ndim
    assert 1 <= k <= x.shape[dim]
    x_flat, n_rows, n_cols, sr, sc, out_shape = reduce_dim_strides(x, dim)
    cfg = _tune(
        _topk_kernel,
        lambda c: _launch_topk(x_flat, n_rows, n_cols, sr, sc, k, c),
        n_rows,
        n_cols,
        k,
    )
    out = _launch_topk(x_flat, n_rows, n_cols, sr, sc, k, cfg)
    return out.view(*out_shape, k) if out_shape else out


def triton_kth_largest_streaming(x: Tensor, k: int, dim: int = -1) -> Tensor:
    dim = dim % x.ndim
    assert 1 <= k <= x.shape[dim]
    x_flat, n_rows, n_cols, sr, sc, out_shape = reduce_dim_strides(x, dim)
    cfg = _tune(
        _streaming_kth_largest_kernel,
        lambda c: _launch_kth_streaming(x_flat, n_rows, n_cols, sr, sc, k, c),
        n_rows,
        n_cols,
        k,
    )
    out = _launch_kth_streaming(x_flat, n_rows, n_cols, sr, sc, k, cfg)
    return out.view(out_shape) if out_shape else out.squeeze()


def triton_kth_largest(x: Tensor, k: int, dim: int = -1) -> Tensor:
    dim = dim % x.ndim
    assert 1 <= k <= x.shape[dim]
    x_flat, n_rows, n_cols, sr, sc, out_shape = reduce_dim_strides(x, dim)
    cfg = _tune(
        _kth_largest_kernel,
        lambda c: _launch_kth(x_flat, n_rows, n_cols, sr, sc, k, c),
        n_rows,
        n_cols,
        k,
    )
    out = _launch_kth(x_flat, n_rows, n_cols, sr, sc, k, cfg)
    return out.view(out_shape) if out_shape else out.squeeze()


def triton_mid(x: Tensor, k: int, dim: int = -1) -> Tensor:
    dim = dim % x.ndim
    assert 1 <= k and k + 1 <= x.shape[dim]
    x_flat, n_rows, n_cols, sr, sc, out_shape = reduce_dim_strides(x, dim)
    cfg = _tune(
        _streaming_mid_kth_largest_kernel,
        lambda c: _launch_mid_streaming(x_flat, n_rows, n_cols, sr, sc, k, c),
        n_rows,
        n_cols,
        k,
    )
    out = _launch_mid_streaming(x_flat, n_rows, n_cols, sr, sc, k, cfg)
    return out.view(out_shape) if out_shape else out.squeeze()


def triton_weighted_mid(
    x: Tensor, k: int, dim: int = -1, weight: float = 1.0
) -> Tensor:
    dim = dim % x.ndim
    assert 1 <= k and k + 1 <= x.shape[dim]
    x_flat, n_rows, n_cols, sr, sc, out_shape = reduce_dim_strides(x, dim)
    cfg = _tune(
        _streaming_mid_kth_largest_kernel,
        lambda c: _launch_mid_streaming(
            x_flat, n_rows, n_cols, sr, sc, k, c, k_weight=weight
        ),
        n_rows,
        n_cols,
        k,
    )
    out = _launch_mid_streaming(x_flat, n_rows, n_cols, sr, sc, k, cfg, k_weight=weight)
    return out.view(out_shape) if out_shape else out.squeeze()


def torch_topk(x: Tensor, k: int, dim: int = -1) -> Tensor:
    return torch.topk(x, k, dim=dim).values


def torch_kth_largest(x: Tensor, k: int, dim: int = -1) -> Tensor:
    return torch.kthvalue(x, x.shape[dim] - k + 1, dim=dim).values


def torch_mid(x: Tensor, k: int, dim: int = -1) -> Tensor:
    n = x.shape[dim]
    v1 = torch.kthvalue(x, n - k + 1, dim=dim).values
    v2 = torch.kthvalue(x, n - k, dim=dim).values
    return (v1 + v2) / 2.0


def torch_weighted_mid(x: Tensor, k: int, dim: int = -1, weight: float = 1.0) -> Tensor:
    n = x.shape[dim]
    v1 = torch.kthvalue(x, n - k + 1, dim=dim).values
    v2 = torch.kthvalue(x, n - k, dim=dim).values
    return (weight * v1 + v2) / (1.0 + weight)


topk = dynamic_dispatch(
    {"triton": triton_topk, "pytorch": torch_topk}, key=["k", "dim"]
)
kth_largest = dynamic_dispatch(
    {
        "triton_single": triton_kth_largest,
        "triton_streaming": triton_kth_largest_streaming,
        "pytorch": torch_kth_largest,
    },
    key=["k", "dim"],
)
mid = dynamic_dispatch({"triton": triton_mid, "pytorch": torch_mid}, key=["k", "dim"])
weighted_mid = dynamic_dispatch(
    {"triton": triton_weighted_mid, "pytorch": torch_weighted_mid}, key=["k", "dim"]
)
