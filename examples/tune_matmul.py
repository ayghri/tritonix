"""Benchmark: trie+monotonicity grid search vs Bayesian optimization."""

import time
import torch
import triton

from tritonix.ops.matmul import matmul_kernel, triton_matmul, matmul

DEVICE = triton.runtime.driver.active.get_active_torch_device()
DTYPE = torch.float16
M, N, K = 512, 2048, 2048*2

a = torch.randn((M, K), device=DEVICE, dtype=DTYPE)
b = torch.randn((K, N), device=DEVICE, dtype=DTYPE)


def make_launcher(a, b):
    m, k = a.shape
    _, n = b.shape

    def launcher(cfg):
        c = torch.empty((m, n), device=a.device, dtype=a.dtype)
        grid = (triton.cdiv(m, cfg["block_m"]), triton.cdiv(n, cfg["block_n"]))
        matmul_kernel[grid](
            a, b, c, m, n, k,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            **cfg,
        )
        return c

    return launcher


def bench(fn, warmup=100, rep=500):
    return triton.testing.do_bench(fn, warmup=warmup, rep=rep)


def tflops(ms):
    return 2.0 * M * N * K / (ms * 1e-3) / 1e12


def main():
    launcher = make_launcher(a, b)
    space_size = len(matmul_kernel.configs())
    print(f"Problem : {M}x{K} @ {K}x{N}  dtype={DTYPE}  device={DEVICE}")
    print(f"Space   : {space_size} configs")
    print(f"Mem params: {matmul_kernel.memory_params}")
    print()

    results = {}

    # --- Grid search (trie + coordinate monotonicity) ---
    print("Grid search (trie + monotonicity pruning) ...")
    t0 = time.perf_counter()
    best_grid = matmul_kernel.tune(
        launcher, method="grid", warmup=10, rep=20, verbose=True
    )
    results["grid"] = {"cfg": best_grid, "tune_s": time.perf_counter() - t0}

    # --- Bayesian (Ax) ---
    print("\nBayesian (Ax) ...")
    t0 = time.perf_counter()
    best_bayes = matmul_kernel.tune(
        launcher, method="bayesian", max_evals=60, warmup=10, rep=20, verbose=True
    )
    results["bayesian"] = {"cfg": best_bayes, "tune_s": time.perf_counter() - t0}

    # --- Baselines ---
    torch_ms = bench(lambda: torch.matmul(a, b))
    triton_ms = bench(lambda: triton_matmul(a, b))

    # --- Final benchmark ---
    print(f"\n{'Method':<20} {'ms':>7} {'TFLOPS':>8} {'tune time':>11}")
    print("-" * 50)
    for name, r in results.items():
        ms = bench(lambda: launcher(r["cfg"]))
        print(f"{name:<20} {ms:>7.3f} {tflops(ms):>8.1f} {r['tune_s']:>10.1f}s")
        print(f"  config: {r['cfg']}")
    print(f"{'triton_matmul':<20} {triton_ms:>7.3f} {tflops(triton_ms):>8.1f} {'(auto-tuned)':>11}")
    print(f"{'torch.matmul':<20} {torch_ms:>7.3f} {tflops(torch_ms):>8.1f} {'n/a':>11}")

    # --- Dispatch winner ---
    print("\nDispatch benchmark (matmul selects faster backend) ...")
    matmul.clear_cache()
    matmul(a, b)  # triggers benchmarking
    winner = list(matmul.cache.values())[0]
    timings = list(matmul.timings.values())[0]
    print(f"  winner: {winner}")
    for backend, ms in timings.items():
        print(f"  {backend}: {ms:.3f}ms  {tflops(ms):.1f} TFLOPS")


if __name__ == "__main__":
    main()
