import torch
import triton
import triton.language as tl
import numpy as np
import configs
from typing import Any


def is_cuda():
    return True


@triton.autotune(
    configs=get_autotune_configs(),
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    # num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    # num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    # num_pid_in_group = GROUP_SIZE_M * num_pid_n
    # group_id = pid // num_pid_in_group
    # first_pid_m = group_id * GROUP_SIZE_M
    # group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    # pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    # pid_n = (pid % num_pid_in_group) // group_size_m
    # pid_m, pid_n = serpentine_order_2d(
    # pid, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_groups_per_stripe = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_groups_per_stripe
    group_start = group_id * GROUP_SIZE_M
    group_size = min(num_pid_m - group_start, GROUP_SIZE_M)

    pid_m = group_start + (pid % num_groups_per_stripe) % group_size
    pid_n = (pid % num_groups_per_stripe) // group_size
    # group_m = pid_m // GROUP_SIZE_M
    # start = 0
    # if group_m % 2 == 1:
    #     start = N - 1
    #     pid_n = N - 1 - pid_n
    # if (start - pid_n) % 2 == 1:
    #     pid_m = (
    #         pid_m
    #         - 2 * ((pid % num_groups_per_stripe) % group_size)
    #         - 1
    #         + group_size
    #     )

    # -----------------------------------------------------------
    # Add some integer bound assumptions.
    # This helps to guide integer analysis in the backend to optimize
    # load/store offset address calculation
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    # offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    # offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    # offs_a = offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    # offs_b = offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    # a_ptrs = a_ptr + offs_a
    # b_ptrs = b_ptr + offs_b
    a_ptrs = a_ptr + (
        offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    )
    b_ptrs = b_ptr + (
        offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    )

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_remaining = K - k * BLOCK_SIZE_K

        a_mask = (offs_am[:, None] < M) & (offs_k[None, :] < k_remaining)
        b_mask = (offs_bn[None, :] < N) & (offs_k[:, None] < k_remaining)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        # a = tl.load(a_ptr + offs_a, mask=offs_a < M * K, other=0.0)
        # b = tl.load(b_ptr + offs_b, mask=offs_b < N * K, other=0.0)
        # We accumulate along the K dimension.
        accumulator = tl.dot(a, b, accumulator)  # , allow_tf32=False)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
        # offs_a += BLOCK_SIZE_K * stride_ak
        # offs_b += BLOCK_SIZE_K * stride_bk
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    c = accumulator.to(c_ptr.dtype.element_ty)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


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
            triton.cdiv(M, META["BLOCK_SIZE_M"])
            * triton.cdiv(N, META["BLOCK_SIZE_N"]),
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


configs = []
configs.append(
    triton.testing.Benchmark(
        # Argument names to use as an x-axis for the plot
        x_names=["N", "K"],
        x_vals=[32 * 2**i for i in range(0, 9)],
        line_arg="provider",
        # Possible values for `line_arg`
        # Don't compare to cublas for fp8 cases as torch.matmul doesn't support fp8 at the moment.
        # line_vals=["triton"]
        # if fp8_inputs
        line_vals=["torch", "triton"],  # Label name for the lines
        line_names=["torch", "triton"],  # Line styles
        # styles=[("green", "-"), ("blue", "-")],
        # ylabel="TFLOPS",
        plot_name="matmul-performance-",  # + "fp16"
        args={},
        # args={"fp8_inputs": fp8_inputs},
    )
)


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
