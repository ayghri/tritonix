import torch
import triton
import triton.language as tl
import gc
from configs import get_autotune_configs


def enable_torch_optimizations(allow_tf32=True):
    """
    Enables various optimizations in PyTorch for matmul operations.
    """
    # Enable cuDNN benchmarking
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = True
        print("cuDNN benchmark enabled.")

    # Enable TF32 on Ampere and newer GPUs
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        # For matmul
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        # For cuDNN convolutions
        torch.backends.cudnn.allow_tf32 = allow_tf32
        # A more general way to enable TF32
        torch.set_float32_matmul_precision("high")
        print("TF32 enabled for matmul.")

    # Enable reduced precision reductions for fp16 and bf16
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    print("Reduced precision reductions enabled for fp16/bf16.")


def is_cuda():
    return True
    # return triton.runtime.driver.active.get_current_target().backend == "cuda"


@triton.autotune(
    configs=get_autotune_configs(),
    key=["M", "N", "K"],
)
@triton.jit
def fused_glu_kernel(
    x_ptr,
    w12_ptr,
    w3_ptr,
    z_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_w12k,
    stride_w12n,
    stride_zm,
    stride_zn,
    stride_w3m,
    stride_w3n,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_W3_M: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # This helps to guide integer analysis in the backend to optimize
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(stride_xm > 0)
    tl.assume(stride_xk > 0)
    tl.assume(stride_w12n > 0)
    tl.assume(stride_w12k > 0)
    tl.assume(stride_zm > 0)
    tl.assume(stride_zn > 0)
    tl.assume(stride_w3m > 0)
    tl.assume(stride_w3n > 0)

    # ----------------------------------------------------------
    # Create pointers for the first blocks of X and W12
    offs_xm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_w12n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    x_ptrs = x_ptr + (
        offs_xm[:, None] * stride_xm + offs_k[None, :] * stride_xk
    )
    glu_ptrs = w12_ptr + (
        offs_k[:, None] * stride_w12k + offs_w12n[None, :] * stride_w12n
    )

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(
            x_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0
        )
        b = tl.load(
            glu_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0
        )
        # We accumulate along the K dimension.
        accumulator = tl.dot(a, b, accumulator)  # , allow_tf32=False)
        # Advance the ptrs to the next K block.
        x_ptrs += BLOCK_SIZE_K * stride_xk
        glu_ptrs += BLOCK_SIZE_K * stride_w12k
    accumulator = accumulator.reshape(
        BLOCK_SIZE_M, BLOCK_SIZE_N // 2, 2, can_reorder=True
    )
    activated, raw = tl.split(accumulator)
    activated = leaky_relu(activated)
    a = activated * raw

    for k in range(0, tl.cdiv(M, BLOCK_SIZE_N // 2)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(
            x_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0
        )
        b = tl.load(
            glu_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0
        )
        # We accumulate along the K dimension.
        accumulator = tl.dot(a, b, accumulator)  # , allow_tf32=False)
        # Advance the ptrs to the next K block.
        x_ptrs += BLOCK_SIZE_K * stride_xk
        glu_ptrs += BLOCK_SIZE_K * stride_w12k

    accumulator = accumulator.reshape(
        BLOCK_SIZE_M, BLOCK_SIZE_N // 2, 2, can_reorder=True
    )
    activated, raw = tl.split(accumulator)
    activated = leaky_relu(activated)
    a = activated * raw
    accumulator = tl.zeros(
        (BLOCK_SIZE_N // 2, BLOCK_SIZE_W3_M), dtype=tl.float32
    )

    c = c.to(z_ptr.dtype.element_ty)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N // 2 + tl.arange(0, BLOCK_SIZE_N // 2)
    c_ptrs = z_ptr + stride_zm * offs_cm[:, None] + stride_zn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N // 2)
    tl.store(c_ptrs, c, mask=c_mask)


# We can fuse `leaky_relu` by providing it as an `ACTIVATION` meta-parameter in `matmul_kernel`.
@triton.jit
def leaky_relu(x):
    # return tl.where(x >= 0, x, 0.01 * x)
    return tl.sigmoid(x) * x


def fused_glu(a, b):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    # Allocates output.
    c = torch.empty((M, N // 2), device=a.device, dtype=a.dtype)

    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_SIZE_M"])
            * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )

    fused_glu_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
    )
    return c


ref_lib = "cuBLAS" if is_cuda() else "rocBLAS"

configs = []
configs.append(
    triton.testing.Benchmark(
        # Argument names to use as an x-axis for the plot
        x_names=[
            "M",
            # "N",
            "K",
        ],
        # x_vals=[128 * i for i in range(2, 33)],
        x_vals=[32 * 2**i for i in range(0, 9)],
        line_arg="provider",
        # Possible values for `line_arg`
        # Don't compare to cublas for fp8 cases as torch.matmul doesn't support fp8 at the moment.
        # line_vals=["triton"]
        # if fp8_inputs
        line_vals=[
            ref_lib.lower(),
            ref_lib.lower() + "_packed",
            "triton",
        ],  # Label name for the lines
        # line_names=["Triton"]
        # if fp8_inputs
        line_names=[
            ref_lib,
            ref_lib.lower() + "_packed",
            "Triton",
        ],  # Line styles
        # styles=[("green", "-"), ("blue", "-")],
        ylabel="TFLOPS",  # Label name for the y-axis
        plot_name="matmul-performance-",
        # args={"fp8_inputs": fp8_inputs},
        args={},
    )
)


@torch.compile(fullgraph=True)
def simple_glu(X, WA, WB):
    return torch.nn.functional.silu(X.mm(WA)) * X.mm(WB)


@torch.compile(fullgraph=True)
def packed_glu(X, W, N):
    X1 = X.mm(W)
    return torch.nn.functional.silu(X1[:, :N]) * X1[:, N:]


# @torch.compile(fullgraph=True)
# def packed_glu(X, WAWB):
#     # return torch.nn.functional.silu(X1[:, :N]) * X1[:, N:]
#     return swiglu_packed(X, WAWB, 0.01)


@triton.testing.perf_report(configs)
def benchmark(M, K, provider):
    N = 1024
    # a = torch.randn((M, K), device=DEVICE, dtype=DTYPE)
    # b = torch.randn((K, N), device=DEVICE, dtype=DTYPE)

    WA = torch.randn(K, N, device=DEVICE, dtype=DTYPE).abs()
    WB = torch.randn(K, N, device=DEVICE, dtype=DTYPE).abs()
    W = torch.stack([WA, WB], dim=2).reshape(K, 2 * N).contiguous()
    WA = WA.contiguous()
    WB = WB.contiguous()
    W = W.contiguous()
    WAWB = torch.concat([WA, WB], dim=1).contiguous()
    X = torch.randn(M, K, device=DEVICE, dtype=DTYPE)
    # if TORCH_HAS_FP8 and fp8_inputs:
    # a = a.to(torch.float8_e5m2)
    # b = b.T
    # b = b.to(torch.float8_e5m2)
    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = 0, 0, 0
    if provider == ref_lib.lower():
        ms, min_ms, max_ms = triton.testing.do_bench(
            # lambda: simple_glu(X, WA, WB), quantiles=quantiles
            lambda: simple_glu(X, WA, WB),
            quantiles=quantiles,
        )  # type: ignore[no-untyped-call]
        # print(N, M, K, matmul_kernel.best_config)
    if provider == ref_lib.lower() + "_packed":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: packed_glu(X, WAWB, N), quantiles=quantiles
        )  # type: ignore[no-untyped-call]
        # print(N, M, K, matmul_kernel.best_config)
    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: fused_glu(X, W), quantiles=quantiles
        )  # type: ignore[no-untyped-call]

    # perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    # perf = lambda ms: ms
    def perf(ms):
        return ms

    return perf(ms), perf(max_ms), perf(min_ms)


# benchmark.run(show_plots=False, print_data=True)
if False:
    B = 8
    IN = 8
    OUT = 16
    torch.manual_seed(0)
    WA = torch.randn(IN, OUT, device=DEVICE, dtype=DTYPE).abs()
    WB = torch.randn(IN, OUT, device=DEVICE, dtype=DTYPE).abs()
    X = torch.randn(B, IN, device=DEVICE, dtype=DTYPE)
    W = torch.stack([WA, WB], dim=2).reshape(IN, 2 * OUT)

    print("W:", W)
    print("X:", X)
    W.contiguous()
    Y = torch.nn.functional.leaky_relu(X.mm(WA)) * X.mm(WB)
    print("W shape:", W.shape, "X shape:", X.shape, "Y shape:", Y.shape)
    Y_fused = X.mm(W)
    Y2 = torch.nn.functional.leaky_relu(Y_fused[:, 0::2]) * Y_fused[:, 1::2]
    print("Distance between outputs:", torch.norm(Y - Y2))
    assert torch.allclose(Y, Y2, atol=1e-3, rtol=1e-3), (
        "Output mismatch between two implementations"
    )
    Y3 = fused_glu(X, W)
    print("Y3 shape:", Y3.shape)
    print("Y:", Y)
    print("Y3:", Y3)

    print("Distance between Y and Y3:", torch.norm(Y - Y3))
    assert torch.allclose(Y, Y3, atol=1e-5, rtol=1e-5), (
        "Output mismatch between two implementations"
    )


def get_peak_gpu_memory_usage_mb():
    """Get peak GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0.0


def reset_gpu_memory_stats():
    """Reset GPU memory statistics."""
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def measure_peak_memory_mb(fn, *args, **kwargs):
    """Measure the peak memory usage of a function in MB."""
    reset_gpu_memory_stats()
    start_memory = get_peak_gpu_memory_usage_mb()
    fn(*args, **kwargs)
    torch.cuda.synchronize()
    peak_memory = get_peak_gpu_memory_usage_mb() - start_memory
    return peak_memory


if __name__ == "__main__":
    DEVICE = triton.runtime.driver.active.get_active_torch_device()
    DTYPE = torch.float32
    ALLOW_TF32 = True  # Set to False if you want to disable TF32 for testing

    enable_torch_optimizations(allow_tf32=ALLOW_TF32)
    benchmark.run(show_plots=False, print_data=True)

    N = M = K = 4096 * 2

    X = torch.randn(M, K, device=DEVICE, dtype=DTYPE)
    WA = torch.randn(K, N, device=DEVICE, dtype=DTYPE).abs()
    WB = torch.randn(K, N, device=DEVICE, dtype=DTYPE).abs()

    W = torch.stack([WA, WB], dim=2).reshape(K, 2 * N)
    WA = WA.contiguous()
    WB = WB.contiguous()
    W = W.contiguous()
    # Warm-up
    _ = fused_glu(X, W)
    # _ = simple_glu(X, WA.t(), WB.t())
    _ = simple_glu(X, WA, WB)

    # Benchmark Triton Kernel
    triton_peak_memory = measure_peak_memory_mb(fused_glu, X, W)
    print(f"Triton Kernel Peak Memory: {triton_peak_memory:.4f} MB")

    # Benchmark PyTorch Implementation
    torch_peak_memory = measure_peak_memory_mb(simple_glu, X, WA, WB)
    print(f"PyTorch Peak Memory:      {torch_peak_memory:.4f} MB")

    # Comparison
    if triton_peak_memory < torch_peak_memory:
        print("\nTriton appears to have used less peak memory.")
    elif torch_peak_memory < triton_peak_memory:
        print("\nPyTorch appears to have used less peak memory.")
    else:
        print("\nBoth implementations used a similar amount of peak memory.")
