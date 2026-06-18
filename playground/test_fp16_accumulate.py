"""Test FP16 dot with FP32 external accumulation vs pure FP16/FP32 accum.

Strategy: use tl.dot(a, b, out_dtype=tl.float16) for the fast FP16-accum
tensor core path, then cast+add into FP32 register accumulator each iter.
"""
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernel variants
# ---------------------------------------------------------------------------

@triton.jit
def matmul_fp32_accum_kernel(
    a_ptr, b_ptr, c_ptr,
    m, n, k,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
):
    """FP32 accumulate inside tl.dot (slower FP32-accum TC path)."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_n = pid_n * block_n + tl.arange(0, block_n)
    offs_k = tl.arange(0, block_k)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((block_m, block_n), dtype=tl.float32)
    for i in range(0, tl.cdiv(k, block_k)):
        k_remaining = k - i * block_k
        mask_k = offs_k < k_remaining
        a = tl.load(a_ptrs, mask=mask_k[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=mask_k[:, None], other=0.0)
        acc = tl.dot(a, b, acc, out_dtype=tl.float32)
        a_ptrs += block_k * stride_ak
        b_ptrs += block_k * stride_bk

    c = acc.to(tl.float16)
    offs_cm = pid_m * block_m + tl.arange(0, block_m)
    offs_cn = pid_n * block_n + tl.arange(0, block_n)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    mask = (offs_cm[:, None] < m) & (offs_cn[None, :] < n)
    tl.store(c_ptrs, c, mask=mask)


@triton.jit
def matmul_fp16_accum_kernel(
    a_ptr, b_ptr, c_ptr,
    m, n, k,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
):
    """Pure FP16 accumulate (fastest TC path, worst precision)."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_n = pid_n * block_n + tl.arange(0, block_n)
    offs_k = tl.arange(0, block_k)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((block_m, block_n), dtype=tl.float16)
    for i in range(0, tl.cdiv(k, block_k)):
        k_remaining = k - i * block_k
        mask_k = offs_k < k_remaining
        a = tl.load(a_ptrs, mask=mask_k[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=mask_k[:, None], other=0.0)
        acc = tl.dot(a, b, acc, out_dtype=tl.float16)
        a_ptrs += block_k * stride_ak
        b_ptrs += block_k * stride_bk

    offs_cm = pid_m * block_m + tl.arange(0, block_m)
    offs_cn = pid_n * block_n + tl.arange(0, block_n)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    mask = (offs_cm[:, None] < m) & (offs_cn[None, :] < n)
    tl.store(c_ptrs, acc, mask=mask)


@triton.jit
def matmul_fp16dot_fp32accum_kernel(
    a_ptr, b_ptr, c_ptr,
    m, n, k,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
):
    """FP16 dot (fast TC) + FP32 external accumulation (best of both)."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_n = pid_n * block_n + tl.arange(0, block_n)
    offs_k = tl.arange(0, block_k)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((block_m, block_n), dtype=tl.float32)
    for i in range(0, tl.cdiv(k, block_k)):
        k_remaining = k - i * block_k
        mask_k = offs_k < k_remaining
        a = tl.load(a_ptrs, mask=mask_k[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=mask_k[:, None], other=0.0)
        acc += tl.dot(a, b, out_dtype=tl.float16).to(tl.float32)
        a_ptrs += block_k * stride_ak
        b_ptrs += block_k * stride_bk

    c = acc.to(tl.float16)
    offs_cm = pid_m * block_m + tl.arange(0, block_m)
    offs_cn = pid_n * block_n + tl.arange(0, block_n)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    mask = (offs_cm[:, None] < m) & (offs_cn[None, :] < n)
    tl.store(c_ptrs, c, mask=mask)


# ---------------------------------------------------------------------------
# Launcher
# ---------------------------------------------------------------------------

def run_kernel(kernel, a, b, block_m=128, block_n=128, block_k=32,
               num_warps=4, num_stages=2):
    m, k = a.shape
    _, n = b.shape
    c = torch.empty((m, n), device=a.device, dtype=torch.float16)
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    kernel[grid](
        a, b, c,
        m, n, k,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        block_m=block_m, block_n=block_n, block_k=block_k,
        num_warps=num_warps, num_stages=num_stages,
    )
    return c


KERNELS = [
    ("FP32 accum",            matmul_fp32_accum_kernel),
    ("FP16 accum (pure)",     matmul_fp16_accum_kernel),
    ("FP16 dot + FP32 accum", matmul_fp16dot_fp32accum_kernel),
]


def error_metrics(c, ref):
    """Compute error metrics between result and reference (in float32).

    Returns: max_err, mean_rel_err, ||err||, ||err||/||ref||
    """
    c_f = c.float()
    r_f = ref.float()
    diff = c_f - r_f
    max_err = diff.abs().max().item()
    mean_rel = (diff.abs() / (r_f.abs() + 1e-6)).mean().item()
    err_norm = torch.norm(diff).item()
    ref_norm = torch.norm(r_f).item()
    rel_norm = err_norm / ref_norm if ref_norm > 0 else float('inf')
    return max_err, mean_rel, err_norm, rel_norm


COL = {
    "name": 25, "ms": 8, "tflops": 8,
    "max_err": 10, "mean_rel": 10, "err_norm": 12, "rel_norm": 14,
}
SEP = "  "


def _header(*cols):
    labels = {
        "name": "name", "ms": "ms", "tflops": "TFLOPS",
        "max_err": "max|err|", "mean_rel": "mean_rel",
        "err_norm": "||err||", "rel_norm": "||err||/||ref||",
    }
    parts = [f"{labels[c]:>{COL[c]}s}" if c != "name" else f"{labels[c]:<{COL[c]}s}"
             for c in cols]
    line = SEP.join(parts)
    print(f"    {line}")
    print(f"    {'-' * len(line)}")


def _row(**kw):
    fmts = {
        "name": lambda v: f"{v:<{COL['name']}s}",
        "ms": lambda v: f"{v:{COL['ms']}.3f}",
        "tflops": lambda v: f"{v:{COL['tflops']}.1f}",
        "max_err": lambda v: f"{v:{COL['max_err']}.4f}",
        "mean_rel": lambda v: f"{v:{COL['mean_rel']}.6f}",
        "err_norm": lambda v: f"{v:{COL['err_norm']}.4f}",
        "rel_norm": lambda v: f"{v:{COL['rel_norm']}.6e}",
    }
    parts = [fmts[k](v) for k, v in kw.items()]
    print(f"    {SEP.join(parts)}")


def main():
    device = "cuda"
    torch.manual_seed(42)

    print(f"Triton version: {triton.__version__}")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"CUDA capability: {torch.cuda.get_device_capability()}")

    bm, bn, bk = 128, 128, 32
    nw, ns = 4, 4

    ALL_COLS = ["name", "ms", "tflops", "max_err", "mean_rel", "err_norm", "rel_norm"]

    shapes = [
        # Square
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        # Tall K (precision stress)
        (1024, 1024, 4096),
        (1024, 1024, 8192),
        (2048, 2048, 8192),
        # Wide / tall M,N (throughput)
        (4096, 4096, 1024),
        (8192, 8192, 1024),
        # Rectangular
        (2048, 4096, 2048),
        (4096, 2048, 4096),
        (512, 8192, 4096),
    ]

    print("\n" + "=" * 120)
    print(f"FP16 MMA ACCUMULATION BENCHMARK")
    print(f"block=({bm},{bn},{bk})  num_warps={nw}  num_stages={ns}")
    print(f"Reference: torch.matmul (FP16 in, FP16 out, cuBLAS FP32 accum)")
    print("=" * 120)

    for m, n, k in shapes:
        a = torch.randn(m, k, device=device, dtype=torch.float16)
        b = torch.randn(k, n, device=device, dtype=torch.float16)
        ref = torch.matmul(a, b)

        print(f"\n  M={m}, N={n}, K={k}   ({2*m*n*k/1e9:.1f} GFLOP)")
        _header(*ALL_COLS)

        for label, kernel in KERNELS:
            try:
                c = run_kernel(kernel, a, b, bm, bn, bk, num_warps=nw, num_stages=ns)
                me, mr, en, rn = error_metrics(c, ref)
                ms = triton.testing.do_bench(
                    lambda kern=kernel: run_kernel(
                        kern, a, b, bm, bn, bk, num_warps=nw, num_stages=ns
                    )
                )
                tflops = 2 * m * n * k / (ms * 1e-3) / 1e12
                _row(name=label, ms=ms, tflops=tflops,
                     max_err=me, mean_rel=mr, err_norm=en, rel_norm=rn)
            except Exception as e:
                print(f"    {label:25s}  FAILED - {e}")

        ms = triton.testing.do_bench(lambda: torch.matmul(a, b))
        tflops = 2 * m * n * k / (ms * 1e-3) / 1e12
        _row(name="torch.matmul (ref)", ms=ms, tflops=tflops,
             max_err=0.0, mean_rel=0.0, err_norm=0.0, rel_norm=0.0)


if __name__ == "__main__":
    main()
