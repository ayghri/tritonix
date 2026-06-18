import torch
import triton
import numpy as np
import configs
from typing import Any

from tritonix.mma.dense import matmul_kernel


def get_mma_config(m, n, k, group_m, num_warps, num_stages):
    return triton.Config(
        {
            "block_m": m,
            "block_n": n,
            "block_k": k,
            "group_m": group_m,
        },
        num_stages=num_stages,
        num_warps=num_warps,
    )


def get_autotune_configs():
    return [
        get_mma_config(64, 32, 16, 8, 4, 4),
        get_mma_config(64, 32, 32, 8, 4, 4),
        get_mma_config(128, 32, 32, 8, 4, 4),
        get_mma_config(128, 64, 16, 4, 4, 4),
        get_mma_config(128, 64, 32, 8, 4, 4),
        get_mma_config(128, 64, 64, 8, 4, 4),
        get_mma_config(128, 64, 64, 8, 8, 4),
        get_mma_config(128, 128, 32, 8, 4, 2),
        get_mma_config(128, 128, 32, 8, 4, 4),
        get_mma_config(128, 128, 32, 8, 4, 6),
        get_mma_config(128, 128, 64, 8, 4, 4),
        get_mma_config(128, 128, 64, 8, 4, 6),
        get_mma_config(128, 128, 64, 8, 8, 4),
    ]


@triton.autotune(
    configs=get_autotune_configs(),
    key=["M", "N", "K"],
)


def matmul(a, b, transpose_b=False):
    # Check constraints.
    if transpose_b:
        assert a.shape[1] == b.shape[1], "Incompatible dimensions"
    else:
        assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    if transpose_b:
        N, K_b = b.shape
    else:
        K_b, N = b.shape
    assert K == K_b

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    def grid(META):
        return (
            triton.cdiv(M, META["block_m"])
            * triton.cdiv(N, META["block_n"]),
        )

    matmul_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(1) if transpose_b else b.stride(0),
        b.stride(0) if transpose_b else b.stride(1),
        c.stride(0),
        c.stride(1),
    )
    return c


def test_correctness(
    M,
    N,
    K,
    dtype=torch.float32,
    transpose_b=False,
    np_float: Any = np.float32,  # type: ignore[no-redef]
):
    torch.manual_seed(0)
    a = torch.randn((M, K), device="cuda", dtype=dtype)
    b = torch.randn((K, N), device="cuda", dtype=dtype)
    if transpose_b:
        b = b.t().contiguous()
        c_torch = torch.matmul(a, b.t())
    else:
        c_torch = torch.matmul(a, b)
    print("Strides: ", a.stride(), b.stride(), c_torch.stride())
    a_np = a.cpu().numpy().astype(np_float)
    b_np = b.cpu().numpy().astype(np_float)
    if transpose_b:
        c_np = np.dot(a_np, b_np.T)
    else:
        c_np = np.dot(a_np, b_np)
    c_np = torch.from_numpy(c_np).to("cuda")
    c_triton = matmul(a, b, transpose_b=transpose_b)
    print(f"fp16 max|torch-triton|: {torch.max(torch.abs(c_torch - c_triton))}")
    c_torch = c_torch.to(torch.float64)
    c_triton = c_triton.to(torch.float64)
    print(
        f"Testing for M={M}, N={N}, K={K}, dtype={dtype}, transpose_b={transpose_b}"
    )
    print(f"max|torch-triton|: {torch.max(torch.abs(c_triton - c_torch))}")
    print(f"max|torch-numpy|: {torch.max(torch.abs(c_torch - c_np))}")
    print(f"max|triton-numpy|: {torch.max(torch.abs(c_triton - c_np))}")
    print(f"||torch-triton||: {torch.norm(c_triton - c_torch)}")
    print(f"||torch-numpy||: {torch.norm(c_torch - c_np)}")
    print(f"||triton-numpy||: {torch.norm(c_triton - c_np)}")


configs = [
    triton.testing.Benchmark(
        x_names=["N", "K"],
        x_vals=[32 * 2**i for i in range(0, 9)],
        line_arg="provider",
        line_vals=["torch", "triton"],
        line_names=["torch", "triton"],
        plot_name="matmul-performance-",
        args={},
    )
]


if __name__ == "__main__":
    DEVICE = triton.runtime.driver.active.get_active_torch_device()
    DTYPE = torch.float16
    FLOAT_SIZE = torch.finfo(DTYPE).bits // 8
    print(f"Using device {DEVICE}, dtype {DTYPE}, {FLOAT_SIZE} bytes.")
    enable_torch_optimizations(fp16_reduced_precision=False)
    # disable_torch_optimizations()

    @triton.testing.perf_report(configs)
    def benchmark(N, K, provider):
        M = 1024*2
        a = torch.randn((M, K), device=DEVICE, dtype=DTYPE)
        b = torch.randn((N, K), device=DEVICE, dtype=DTYPE)
        quantiles = [0.5, 0.2, 0.8]
        ms, min_ms, max_ms = 0, 0, 0
        if provider == "torch":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: torch.matmul(a, b.t()), quantiles=quantiles
            )  # type: ignore[no-untyped-call]
        if provider == "triton":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: matmul(a, b, transpose_b=True), quantiles=quantiles
            )  # type: ignore[no-untyped-call]

        # def perf(ms):
            # return ms
        def perf(ms):
            return FLOAT_SIZE * M * N * K / (ms * 1e-3) / (1024**4)

        return perf(ms), perf(max_ms), perf(min_ms)

    benchmark.run(show_plots=False, print_data=True)
    test_correctness(500, 250, 1000, dtype=DTYPE, np_float=np.float16)
    test_correctness(
        500, 250, 1000, dtype=DTYPE, transpose_b=True, np_float=np.float16
    )
