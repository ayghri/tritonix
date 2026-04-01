import argparse
import statistics
from typing import Dict, Literal, Optional

import torch


def _check_fp16_accumulation() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required for fp16 mma profiling.")
    if not torch.cuda.get_device_capability() >= (7, 0):
        raise RuntimeError("Tensor Cores (sm70+) are required for fp16 mma profiling.")


def profile_mma(
    m: int = 4096,
    n: int = 4096,
    k: int = 4096,
    iters: int = 50,
    warmup: int = 10,
    accum: Literal["fp16", "fp32"] = "fp16",
    use_cuda_graph: bool = False,
) -> Dict[str, float]:
    """Profile a dense GEMM (mma) using torch.matmul with fp16 inputs and accumulator.

    Returns a dictionary with mean, median, and std latency in milliseconds as well as TFLOPs.
    """
    _check_fp16_accumulation()

    device = torch.device("cuda")
    dtype = torch.float16

    # Disable TF32 so fp32 inputs won't silently use TF32 paths.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Control accumulation mode for half inputs: True => allow fp16 reduction, False => prefer fp32
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = accum == "fp16"

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    a = torch.randn((m, k), device=device, dtype=dtype).contiguous()
    b = torch.randn((k, n), device=device, dtype=dtype).contiguous()
    out = torch.empty((m, n), device=device, dtype=dtype)

    # Optional CUDA Graph capture to minimize launch overhead
    graph: Optional[torch.cuda.CUDAGraph] = None
    if use_cuda_graph:
        torch.cuda.synchronize()
        static_a = a.clone()
        static_b = b.clone()
        static_out = torch.empty_like(out)
        stream = torch.cuda.Stream()
        # Warm up ops on the capture stream so kernels/params are initialized there
        with torch.cuda.stream(stream):
            for _ in range(3):
                torch.matmul(static_a, static_b, out=static_out)
            # Capture must occur on a non-default stream
            graph = torch.cuda.CUDAGraph()
            graph.capture_begin()
            torch.matmul(static_a, static_b, out=static_out)
            graph.capture_end()

        # Warm-up graph replays
        for _ in range(warmup):
            graph.replay()
        torch.cuda.synchronize()
    else:
        # Warm-up to stabilize clocks and caches
        for _ in range(warmup):
            torch.matmul(a, b, out=out)
        torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    latencies_ms = []

    for _ in range(iters):
        start_event.record()
        if graph is not None:
            graph.replay()
        else:
            torch.matmul(a, b, out=out)
        end_event.record()
        torch.cuda.synchronize()
        latencies_ms.append(start_event.elapsed_time(end_event))

    mean_ms = statistics.fmean(latencies_ms)
    median_ms = statistics.median(latencies_ms)
    std_ms = statistics.pstdev(latencies_ms)

    # GEMM performs 2 * M * N * K floating point operations
    # Convert ms -> seconds and then to TFLOPs (1e12 FLOPs per TFLOP):
    # tflops = flops / (time_s * 1e12) = flops / ((ms/1e3) * 1e12) = flops / (ms * 1e9)
    tflops = (2.0 * m * n * k) / (mean_ms * 1e9)

    return {
        "mean_ms": mean_ms,
        "median_ms": median_ms,
        "std_ms": std_ms,
        "tflops": tflops,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile fp16 mma via torch.matmul")
    parser.add_argument("--m", type=int, default=4096, help="Rows of A / C")
    parser.add_argument("--n", type=int, default=4096, help="Columns of B / C")
    parser.add_argument("--k", type=int, default=4096, help="Columns of A / Rows of B")
    parser.add_argument("--iters", type=int, default=50, help="Number of timed iterations")
    parser.add_argument("--warmup", type=int, default=10, help="Warm-up iterations")
    parser.add_argument(
        "--accum",
        type=str,
        default="fp16",
        choices=["fp16", "fp32"],
        help="Accumulation precision for fp16 inputs (fp16 or fp32)",
    )
    parser.add_argument(
        "--graph",
        action="store_true",
        help="Use CUDA Graph to reduce launch overhead",
    )
    args = parser.parse_args()

    stats = profile_mma(
        args.m,
        args.n,
        args.k,
        args.iters,
        args.warmup,
        accum=args.accum,
        use_cuda_graph=args.graph,
    )

    props = torch.cuda.get_device_properties(torch.device("cuda"))
    print("FP16 MMA profiling results")
    print(f"Dimensions: {args.m} x {args.k} · {args.k} x {args.n}")
    print(
        f"Device: {props.name} (SM {props.major}{props.minor}), FP16 accum: {args.accum == 'fp16'}"
    )
    print(
        "Settings: TF32 off, allow_fp16_reduced_precision_reduction = "
        f"{torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction}"
    )
    print(f"Mean latency: {stats['mean_ms']:.3f} ms")
    print(f"Median latency: {stats['median_ms']:.3f} ms")
    print(f"Std latency: {stats['std_ms']:.3f} ms")
    print(f"Throughput: {stats['tflops']:.2f} TFLOPs")


if __name__ == "__main__":
    main()
