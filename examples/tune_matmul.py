"""Compare matmul autotuning with and without trie-based OOM pruning."""

import time
import torch
import triton

from tritonix.ops.matmul import matmul_kernel
from tritonix.autotune import TunableKernel

DEVICE = triton.runtime.driver.active.get_active_torch_device()
DTYPE = torch.float16

# Problem size
M, N, K = 2048, 2048, 2048

a = torch.randn((M, K), device=DEVICE, dtype=DTYPE)
b = torch.randn((K, N), device=DEVICE, dtype=DTYPE)


def make_launcher(a, b):
    """Returns a launcher function that takes a config dict."""
    m, k = a.shape
    _, n = b.shape

    def launcher(cfg):
        c = torch.empty((m, n), device=a.device, dtype=a.dtype)
        block_m = cfg["block_m"]
        block_n = cfg["block_n"]
        block_k = cfg["block_k"]
        group_m = cfg["group_m"]
        grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
        matmul_kernel[grid](
            a, b, c,
            m, n, k,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            group_m=group_m,
            num_stages=cfg["num_stages"],
            num_warps=cfg["num_warps"],
        )
        return c

    return launcher


def bench_config(launcher, cfg, warmup=50, rep=200):
    """Benchmark a specific config."""
    ms = triton.testing.do_bench(lambda: launcher(cfg), warmup=warmup, rep=rep)
    if isinstance(ms, tuple):
        ms = ms[0]
    return ms


def main():
    # matmul_kernel is now a TunableKernel with space and smem_params from @tunable
    print(f"Problem: {M}x{K} @ {K}x{N}  dtype={DTYPE}  device={DEVICE}")
    print(f"Search space: {len(matmul_kernel.configs())} total configs")
    print(f"smem_params: {matmul_kernel.smem_params}")
    print()

    launcher = make_launcher(a, b)

    # --- Tune WITHOUT trie ---
    print("=" * 60)
    print("Tuning WITHOUT trie (brute grid search)")
    print("=" * 60)
    tk_no_trie = TunableKernel(
        kernel=matmul_kernel,
        keys=matmul_kernel.keys,
        space=matmul_kernel.space,
        smem_params=[],  # No trie pruning
    )
    t0 = time.perf_counter()
    best_no_trie = tk_no_trie.tune(
        launcher, method="grid", verbose=True, warmup=10, rep=20,
    )
    t_no_trie = time.perf_counter() - t0
    print(f"\nBest config: {best_no_trie}")
    print(f"Tuning time: {t_no_trie:.1f}s")

    # --- Tune WITH trie (using the decorated kernel directly) ---
    print()
    print("=" * 60)
    print("Tuning WITH trie (OOM-pruned grid search)")
    print("=" * 60)
    t0 = time.perf_counter()
    best_trie = matmul_kernel.tune(
        launcher, method="grid", verbose=True, warmup=10, rep=20,
    )
    t_trie = time.perf_counter() - t0
    print(f"\nBest config: {best_trie}")
    print(f"Tuning time: {t_trie:.1f}s")

    # --- Final benchmark: best configs vs PyTorch ---
    print()
    print("=" * 60)
    print("Final benchmark (warmup=100, rep=500)")
    print("=" * 60)

    ms_no_trie = bench_config(launcher, best_no_trie, warmup=100, rep=500)
    ms_trie = bench_config(launcher, best_trie, warmup=100, rep=500)

    torch_ms = triton.testing.do_bench(
        lambda: torch.matmul(a, b), warmup=100, rep=500,
    )
    if isinstance(torch_ms, tuple):
        torch_ms = torch_ms[0]

    flops = 2.0 * M * N * K
    tflops_no_trie = flops / (ms_no_trie * 1e-3) / 1e12
    tflops_trie = flops / (ms_trie * 1e-3) / 1e12
    tflops_torch = flops / (torch_ms * 1e-3) / 1e12

    print(f"\n{'Method':<30} {'ms':>8} {'TFLOPS':>8} {'tune time':>10}")
    print("-" * 60)
    print(f"{'Triton (no trie)':<30} {ms_no_trie:>8.3f} {tflops_no_trie:>8.1f} {t_no_trie:>9.1f}s")
    print(f"{'Triton (trie-pruned)':<30} {ms_trie:>8.3f} {tflops_trie:>8.1f} {t_trie:>9.1f}s")
    print(f"{'torch.matmul':<30} {torch_ms:>8.3f} {tflops_torch:>8.1f} {'n/a':>10}")
    print()

    if t_no_trie > 0:
        speedup = t_no_trie / t_trie if t_trie > 0 else float("inf")
        print(f"Trie tuning speedup: {speedup:.1f}x faster")


if __name__ == "__main__":
    main()
