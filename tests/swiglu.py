import torch
import triton
import gc
from xformers.ops import SwiGLU
from fused_glu import fused_glu


DEVICE = triton.runtime.driver.active.get_active_torch_device()
DTYPE = torch.float32
ALLOW_TF32 = True  # Set to False if you want to disable TF32 for testing


def enable_torch_optimizations():
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
        torch.backends.cuda.matmul.allow_tf32 = ALLOW_TF32
        # For cuDNN convolutions
        torch.backends.cudnn.allow_tf32 = ALLOW_TF32
        # A more general way to enable TF32
        torch.set_float32_matmul_precision("high")
        print("TF32 enabled for matmul.")

    # Enable reduced precision reductions for fp16 and bf16
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    print("Reduced precision reductions enabled for fp16/bf16.")


enable_torch_optimizations()


def is_cuda():
    return True
    # return triton.runtime.driver.active.get_current_target().backend == "cuda"


# @torch.compile(fullgraph=True)
def simple_swiglu(X, WA, WB, WC):
    """
    Simple implementation of the SwigLU activation function using PyTorch.
    """
    return (torch.nn.functional.silu(X.mm(WA)) * X.mm(WB)).mm(WC)


@torch.compile(fullgraph=True)
def fused_swiglu(X, W, WC):
    # X1 = X.mm(W)
    # return (torch.nn.functional.silu(X1[:, :N]) * X1[:, N:]).mm(WC)
    return fused_glu(X, W).mm(WC)


def xformers_swiglu(X, layer):
    """
    Implementation of the SwigLU activation function using xFormers.
    """
    # return swiglu_packed(
    # X, w1w2=WAWB, w3=WC, b1b2=None, b3=None, op=SwiGLUPackedFusedOp
    # )
    with torch.no_grad():
        return layer(X)


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
        line_vals=[
            "simple",
            "fused",
            "xformers",
        ],
        line_names=[
            "simple",
            "fused",
            "xformers",
        ],
        ylabel="TFLOPS",  # Label name for the y-axis
        plot_name="matmul-performance-",
        # args={"fp8_inputs": fp8_inputs},
        args={},
    )
)


# @torch.compile(fullgraph=True)
# def simple_glu(X, WA, WB):
#     return torch.nn.functional.silu(X.mm(WA)) * X.mm(WB)


# @torch.compile(fullgraph=True)
# def fused_glu(X, W, WC, N):
#     X1 = X.mm(W)
#     return (torch.nn.functional.silu(X1[:, :N]) * X1[:, N:]).mm(WC)


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
    WC = torch.randn(N, K, device=DEVICE, dtype=DTYPE).abs()
    W = torch.stack([WA, WB], dim=2).reshape(K, 2 * N).contiguous()
    WA = WA.contiguous()
    WB = WB.contiguous()
    W = W.contiguous()
    X = torch.randn(M, K, device=DEVICE, dtype=DTYPE)

    swig = SwiGLU(
        in_features=K, hidden_features=N, bias=False, _pack_weights=True
    ).to(DEVICE, dtype=DTYPE)

    with torch.no_grad():
        swig.w12.weight.copy_(torch.cat([WA, WB], dim=1).t().contiguous())
        # swig.w1.weight.copy_(WA.t().contiguous())
        # swig.w2.weight.copy_(WB.t().contiguous()u
        # swig.w3.weight.copy_(WC.t().contiguous())
    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = 0, 0, 0
    if provider == "simple":
        ms, min_ms, max_ms = triton.testing.do_bench(
            # lambda: simple_glu(X, WA, WB), quantiles=quantiles
            lambda: simple_swiglu(X, WA, WB, WC),
            quantiles=quantiles,
        )  # type: ignore[no-untyped-call]
        # print(N, M, K, matmul_kernel.best_config)
    elif provider == "fused":
        WC = WC.t().contiguous().t()
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: fused_swiglu(X, W, WC), quantiles=quantiles
        )  # type: ignore[no-untyped-call]
        # print(N, M, K, matmul_kernel.best_config)
    elif provider == "xformers":
        with torch.no_grad():
            _ = xformers_swiglu(X, swig)
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: xformers_swiglu(X, swig), quantiles=quantiles
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
    Y3 = (X, W)
    print("Y3 shape:", Y3.shape)
    print("Y:", Y)
    print("Y3:", Y3)

    print("Distance between Y and Y3:", torch.norm(Y - Y3))
    assert torch.allclose(Y, Y3, atol=1e-5, rtol=1e-5), (
        "Output mismatch between two implementations"
    )

benchmark.run(show_plots=False, print_data=True)


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


# N = M = K = 4096 * 2


# X = torch.randn(M, K, device=DEVICE, dtype=DTYPE)
# WA = torch.randn(K, N, device=DEVICE, dtype=DTYPE).abs()
# WB = torch.randn(K, N, device=DEVICE, dtype=DTYPE).abs()

# W = torch.stack([WA, WB], dim=2).reshape(K, 2 * N)
# WA = WA.contiguous()
# WB = WB.contiguous()
# W = W.contiguous()
# # Warm-up
# # _ = fused_glu(X, W)
# # _ = simple_glu(X, WA.t(), WB.t())
# # _ = simple_glu(X, WA, WB)

# # Benchmark Triton Kernel
# triton_peak_memory = measure_peak_memory_mb(fused_glu, X, W)
# print(f"Triton Kernel Peak Memory: {triton_peak_memory:.4f} MB")

# # Benchmark PyTorch Implementation
# torch_peak_memory = measure_peak_memory_mb(simple_glu, X, WA, WB)
# print(f"PyTorch Peak Memory:      {torch_peak_memory:.4f} MB")

# # Comparison
# if triton_peak_memory < torch_peak_memory:
#     print("\nTriton appears to have used less peak memory.")
# elif torch_peak_memory < triton_peak_memory:
#     print("\nPyTorch appears to have used less peak memory.")
# else:
#     print("\nBoth implementations used a similar amount of peak memory.")
