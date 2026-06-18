"""Triton vs PyTorch benchmarks: topk, kth_largest, weighted_mid."""

import torch
import triton

from tritonix.ops.stats import kth_largest, topk, weighted_mid, _MAX_N

DEVICE = triton.runtime.driver.active.get_active_torch_device()
DTYPE  = torch.float16


# ── PyTorch equivalents ──────────────────────────────────────────────────────

def torch_kth_largest(x, k, dim=-1):
    return torch.kthvalue(x, x.shape[dim] - k + 1, dim=dim).values

def torch_topk(x, k, dim=-1):
    return torch.topk(x, k, dim=dim).values

def torch_weighted_mid(x, k, dim=-1, weight=1.0):
    n  = x.shape[dim]
    v1 = torch.kthvalue(x, n - k + 1, dim=dim).values
    v2 = torch.kthvalue(x, n - k,     dim=dim).values
    return (weight * v1 + v2) / (1.0 + weight)


# ── Benchmark ────────────────────────────────────────────────────────────────

def bench(fn, warmup=50, rep=200):
    return triton.testing.do_bench(fn, warmup=warmup, rep=rep)


def gbps(ms, m, n):
    return m * n * torch.finfo(DTYPE).bits / 8 / (ms * 1e-3) / 1e9


def run(m, n, k, weight=2.0):
    x       = torch.randn((m, n), device=DEVICE, dtype=DTYPE)
    backend = "triton" if triton.next_power_of_2(n) <= _MAX_N else "pytorch fallback"

    print(f"\n  M={m:>6}  N={n:>5}  k={k:>4}   [{backend}]")
    print(f"  {'op':<16} {'triton':>8}  {'pytorch':>8}  {'triton GB/s':>12}  {'pytorch GB/s':>12}  speedup")
    print(f"  {'-'*75}")

    for name, tri_fn, tor_fn in [
        ("kth_largest",  lambda: kth_largest(x, k),               lambda: torch_kth_largest(x, k)),
        ("topk",         lambda: topk(x, k),                       lambda: torch_topk(x, k)),
        ("weighted_mid", lambda: weighted_mid(x, k, weight=weight), lambda: torch_weighted_mid(x, k, weight=weight)),
    ]:
        tri_ms = bench(tri_fn)
        tor_ms = bench(tor_fn)
        print(f"  {name:<16} {tri_ms:>8.3f}  {tor_ms:>8.3f}  "
              f"{gbps(tri_ms,m,n):>12.1f}  {gbps(tor_ms,m,n):>12.1f}  "
              f"{tor_ms/tri_ms:>6.2f}x")


def main():
    print(f"dtype={DTYPE}  device={DEVICE}")
    configs = [
        (4096*8,   4,  2),
        (4096,   64,  4),
        (4096,  128,  8),
        (4096,  256, 16),
        (4096,  512, 16),
        (4096, 1024, 32),
        (4096, 2048, 32),   # fallback to pytorch (n > 1024)
    ]
    for m, n, k in configs:
        run(m, n, k)


if __name__ == "__main__":
    main()
