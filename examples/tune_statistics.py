"""Benchmark: kth_largest — single-load Triton vs streaming Triton vs PyTorch."""

import time
import torch
import triton

from tritonix.ops.statistics import (
    _kth_largest_kernel,
    _launch_kth,
    triton_kth_largest,
    triton_kth_largest_streaming,
    kth_largest,
    torch_kth_largest,
)
from tritonix.utils.torch import reduce_dim_strides

DEVICE = triton.runtime.driver.active.get_active_torch_device()
DTYPE = torch.float16
M, N = 4096, 512
K = 16

x = torch.randn((M, N), device=DEVICE, dtype=DTYPE)


def bench(fn, warmup=50, rep=200):
    return triton.testing.do_bench(fn, warmup=warmup, rep=rep)


def gbps(ms):
    return M * N * torch.finfo(DTYPE).bits / 8 / (ms * 1e-3) / 1e9


def main():
    print(f"Problem: ({M}, {N})  k={K}  dtype={DTYPE}  device={DEVICE}")

    x_flat, n_rows, n_cols, sr, sc, _ = reduce_dim_strides(x, dim=-1)

    # correctness
    cfg = {"block_m": 32, "num_warps": 4}
    out = _launch_kth(x_flat, n_rows, n_cols, sr, sc, K, cfg)
    ref = torch.kthvalue(x, n_cols - K + 1, dim=-1).values
    print("max diff:", (out - ref).abs().max().item())

    # tune single-load
    space_size = len(_kth_largest_kernel.configs())
    print(f"\n[ kth_largest (single-load) ]  space: {space_size} configs")
    print(f"  {'ms':>8}  config")
    print(f"  {'-'*60}")
    t0 = time.perf_counter()
    best = _kth_largest_kernel.tune(
        lambda cfg: _launch_kth(x_flat, n_rows, n_cols, sr, sc, K, cfg),
        warmup=5, rep=20, verbose=True,
    )
    print(f"  best ({time.perf_counter() - t0:.1f}s): {best}")

    # bench all backends
    ms_single = bench(lambda: triton_kth_largest(x, K))
    ms_stream = bench(lambda: triton_kth_largest_streaming(x, K))
    ms_torch  = bench(lambda: torch_kth_largest(x, K))
    print(f"\n  {'backend':<24} {'ms':>7}  {'GB/s':>7}")
    print(f"  {'-'*42}")
    print(f"  {'triton_single':<24} {ms_single:>7.3f}  {gbps(ms_single):>7.1f}")
    print(f"  {'triton_streaming':<24} {ms_stream:>7.3f}  {gbps(ms_stream):>7.1f}")
    print(f"  {'pytorch':<24} {ms_torch:>7.3f}  {gbps(ms_torch):>7.1f}")

    # dispatch
    kth_largest.clear_cache()
    kth_largest(x, K)
    winner  = list(kth_largest.cache.values())[0]
    timings = list(kth_largest.timings.values())[0]
    print(f"\n  dispatch → {winner}")
    for name, ms in timings.items():
        print(f"    {name}: {ms:.3f}ms  {gbps(ms):.1f} GB/s")


if __name__ == "__main__":
    main()
