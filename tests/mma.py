import triton
import torch
import numpy as np
from typing import Any

from kernels.matrix.mma import matmul_kernel
from kernels.utils.triton import get_autotune_configs
from kernels.matrix.mma import gemm_splitk_kernel
from kernels.utils.torch import (
    enable_torch_optimizations,
    # disable_torch_optimizations,
)
from kernels.utils.triton import get_splitk_autotune_configs
from kernels.utils.triton import wrap_autotuner


def is_cuda():
    return True


matmul_K = wrap_autotuner(matmul_kernel, get_autotune_configs())


# matmul_K = wrap_autotuner(
#     gemm_splitk_kernel,
#     get_splitk_autotune_configs(),
#     # reset_to_zero=["c_ptr"],
# )


def matmul(a, b, high_precision=False):
    # Check constraints.
    # if transpose_b:
    #     assert a.shape[1] == b.shape[1], "Incompatible dimensions"
    # else:
    #     assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    # assert a.is_contiguous(), "Matrix A must be contiguous"
    # if transpose_b:
    #     n, k_b = b.shape
    # else:
    #     k_b, n = b.shape
    m, k = a.shape
    k_b, n = b.shape
    assert k == k_b

    # c = torch.zeros((m, n), device=a.device, dtype=a.dtype)
    c = torch.empty((m, n), device=a.device, dtype=torch.float32)
    # split_k = 4
    # group_m = 8

    # total_blocks_m = triton.cdiv(m, block_m)
    # total_blocks_n = triton.cdiv(n, block_n)
    # total_programs_mn = total_blocks_m * total_blocks_n
    # total_programs_k = split_k
    # grid = (total_programs_mn, total_programs_k)

    # def grid(META):
    #     return (
    #         triton.cdiv(m, META["block_m"]),
    #         triton.cdiv(n, META["block_n"]),
    #         META["split_k"],
    #     )

    def grid(META):
        return (
            triton.cdiv(m, META["block_m"]),
            triton.cdiv(n, META["block_n"]),
        )

    matmul_K[grid](
        a,
        b,
        c,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        # use_tf32=False
    )

    #   if transpose_b else b.stride(0),
    #   if transpose_b else b.stride(1),
    return c


def test_correctness(
    m,
    n,
    k,
    dtype=torch.float32,
    np_float: Any = np.float32,
):
    torch.manual_seed(0)
    a = torch.randn((m, k), device="cuda", dtype=dtype)
    b = torch.randn((k, n), device="cuda", dtype=dtype)
    # if transpose_b:
    #     b = b.t().contiguous()
    #     c_torch = torch.matmul(a, b.t())
    # else:
    c_torch = torch.matmul(a, b)
    print("Strides: ", a.stride(), b.stride(), c_torch.stride())
    a_np = a.cpu().numpy().astype(np_float)
    b_np = b.cpu().numpy().astype(np_float)
    # if transpose_b:
    #     c_np = np.dot(a_np, b_np.T)
    # else:
    c_np = np.dot(a_np, b_np)

    c_np = torch.from_numpy(c_np).to("cuda")
    c_triton = matmul(a, b)

    print(f"fp16 max|torch-triton|: {torch.max(torch.abs(c_torch - c_triton))}")

    c_torch = c_torch.to(torch.float64)
    c_triton = c_triton.to(torch.float64)

    print(f"Testing for M={m}, N={n}, K={k}, dtype={dtype}")
    print(f"max|torch-triton|: {torch.max(torch.abs(c_triton - c_torch))}")
    print(f"max|torch-numpy|: {torch.max(torch.abs(c_torch - c_np))}")
    print(f"max|triton-numpy|: {torch.max(torch.abs(c_triton - c_np))}")
    print(f"||torch-triton||: {torch.norm(c_triton - c_torch)}")
    print(f"||torch-numpy||: {torch.norm(c_torch - c_np)}")
    print(f"||triton-numpy||: {torch.norm(c_triton - c_np)}")


configs = []
configs.append(
    triton.testing.Benchmark(
        x_names=["k"],
        x_vals=[512 * i for i in range(4, 32, 2)],
        line_arg="provider",
        line_vals=["torch", "triton"],  # Label name for the lines
        line_names=["torch", "triton"],  # Line styles
        plot_name="Performance-TFLOPS",  # + "fp16"
        args={},  # args={"fp8_inputs": fp8_inputs},
    )
)


if __name__ == "__main__":
    DEVICE = triton.runtime.driver.active.get_active_torch_device()
    DTYPE = torch.float32
    FLOAT_SIZE = torch.finfo(DTYPE).bits // 8
    print(f"Using device {DEVICE}, dtype {DTYPE}, {FLOAT_SIZE} bytes.")
    # enable_torch_optimizations(fp16_reduced_precision=False)
    enable_torch_optimizations()
    # disable_torch_optimizations()

    # m = 1024
    # n = 1024 * 2
    m = n = 1024 * 4

    @triton.testing.perf_report(configs)
    def benchmark(k, provider):
        a = torch.randn((m, k), device=DEVICE, dtype=DTYPE)
        b = torch.randn((k, n), device=DEVICE, dtype=DTYPE)
        # b = torch.randn((k, n), device=DEVICE, dtype=DTYPE)
        quantiles = [0.5, 0.2, 0.8]
        ms, min_ms, max_ms = 0, 0, 0
        if provider == "torch":
            # To get a fair comparison, we compile the torch matmul
            compiled_torch_matmul = torch.compile(
                lambda x, y: torch.matmul(x, y),
                fullgraph=True,
            )
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: compiled_torch_matmul(a, b),
                quantiles=quantiles,
                warmup=100,
                rep=100,
            )  # type: ignore[no-untyped-call]
        if provider == "triton":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: matmul(a, b),
                quantiles=quantiles,
                warmup=100,
                rep=100,
            )  # type: ignore[no-untyped-call]

        def perf(ms):
            return ms

        # def perf(ms):
        #     return FLOAT_SIZE * m * n * k / (ms * 1e-3) / (1024**4)

        return perf(ms), perf(max_ms), perf(min_ms)

    print("Running benchmark... for M=", m, ", N=", n)
    benchmark.run(show_plots=False, print_data=True)
    test_correctness(128, 128, 1024 * 2, dtype=DTYPE, np_float=np.float16)
    # test_correctness( 500, 250, 1000, dtype=DTYPE, transpose_b=True, np_float=np.float16)
