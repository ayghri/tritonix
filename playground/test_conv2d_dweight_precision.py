#!/usr/bin/env python3
import torch
import torch.nn.grad as G
import numpy as np
import time
import random
from conv2d_bwd import conv2d_dweight
import configs

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
configs.disable_torch_optimizations()


# Test configurations for different precisions and devices
TEST_CONFIGS = [
    {
        "device": "cuda",
        "dtype": torch.float32,
        "name": "CUDA FP32",
        "atol": 1e-6,
        "rtol": 1e-4,
    },
    {
        "device": "cpu",
        "dtype": torch.float32,
        "name": "CPU FP32",
        "atol": 1e-6,
        "rtol": 1e-4,
    },
    {
        "device": "cpu",
        "dtype": torch.float64,
        "name": "CPU FP64",
        "atol": 1e-12,
        "rtol": 1e-10,
    },
]


# @torch.compile
def torch_conv2d_dweight(
    input_tensor, weight_shape, grad_output, stride, padding, dilation
):
    """
    Compute weight gradients using PyTorch's autograd system.
    """

    return G.conv2d_weight(
        input_tensor,
        weight_shape,
        grad_output,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )


def run_single_test(
    config,
    batch,
    in_ch,
    h,
    w,
    out_ch,
    kh,
    kw,
    stride,
    padding,
    dilation,
    test_name,
):
    """Run a single test configuration"""
    device = torch.device(config["device"])
    dtype = config["dtype"]

    if device.type == "cuda" and not torch.cuda.is_available():
        return None, "CUDA not available"

    try:
        # Create input tensors with specified device and dtype
        input_tensor = torch.randn(
            batch, in_ch, h, w, device=device, dtype=dtype
        )
        weight_shape = (out_ch, in_ch, kh, kw)

        # Normalize stride, padding, dilation to tuples
        stride_tuple = (stride, stride) if isinstance(stride, int) else stride
        padding_tuple = (
            (padding, padding) if isinstance(padding, int) else padding
        )
        dilation_tuple = (
            (dilation, dilation) if isinstance(dilation, int) else dilation
        )

        # Calculate output shape for grad_output
        h_out = (h + 2 * padding - dilation * (kh - 1) - 1) // stride + 1
        w_out = (w + 2 * padding - dilation * (kw - 1) - 1) // stride + 1

        if h_out <= 0 or w_out <= 0:
            return None, f"Invalid output dimensions: {h_out}x{w_out}"

        grad_output = torch.randn(
            batch, out_ch, h_out, w_out, device=device, dtype=dtype
        )

        # PyTorch reference implementation
        torch_dweight = torch_conv2d_dweight(
            input_tensor,
            weight_shape,
            grad_output,
            stride_tuple,
            padding_tuple,
            dilation_tuple,
        )

        # Our custom kernel (only works on CUDA with float32)
        if device.type == "cuda" and dtype == torch.float32:
            custom_dweight = conv2d_dweight(
                input_tensor,
                weight_shape,
                grad_output,
                stride_tuple,
                padding_tuple,
                dilation_tuple,
            )

            # Compare results
            max_diff = torch.max(
                torch.abs(torch_dweight - custom_dweight)
            ).item()
            rel_error = torch.norm(torch_dweight - custom_dweight) / (
                torch.norm(torch_dweight) + 1e-12
            )

            # Check if results are close
            is_close = torch.allclose(
                torch_dweight,
                custom_dweight,
                rtol=config["rtol"],
                atol=config["atol"],
            )

            result = {
                "passed": is_close,
                "max_diff": max_diff,
                "rel_error": rel_error.item(),
                "torch_stats": {
                    "min": torch_dweight.min().item(),
                    "max": torch_dweight.max().item(),
                    "mean": torch_dweight.mean().item(),
                    "std": torch_dweight.std().item(),
                },
                "custom_stats": {
                    "min": custom_dweight.min().item(),
                    "max": custom_dweight.max().item(),
                    "mean": custom_dweight.mean().item(),
                    "std": custom_dweight.std().item(),
                },
            }
        else:
            # For CPU or non-float32, we only record PyTorch stats as reference
            result = {
                "passed": True,  # No comparison possible
                "max_diff": 0.0,
                "rel_error": 0.0,
                "torch_stats": {
                    "min": torch_dweight.min().item(),
                    "max": torch_dweight.max().item(),
                    "mean": torch_dweight.mean().item(),
                    "std": torch_dweight.std().item(),
                },
                "custom_stats": None,
                "note": "Custom kernel only available on CUDA FP32",
            }

        return result, None

    except Exception as e:
        return None, str(e)


def test_precision_comparison():
    """Test across different precisions and devices"""

    print("=" * 100)
    print("PRECISION AND DEVICE COMPARISON TEST")
    print("=" * 100)

    # Test configurations
    test_cases = [
        # (batch, in_channels, height, width, out_channels, kernel_h, kernel_w, stride, padding, dilation, test_name)
        (2, 3, 32, 32, 16, 3, 3, 1, 1, 1, "Basic 3x3 conv"),
        (1, 64, 28, 28, 128, 3, 3, 2, 1, 1, "Stride 2"),
        (4, 32, 16, 16, 64, 5, 5, 1, 2, 1, "Large 5x5 kernel"),
        (1, 16, 32, 32, 32, 3, 3, 1, 0, 1, "No padding"),
        (2, 8, 64, 64, 16, 1, 1, 1, 0, 1, "1x1 conv"),
        (1, 32, 14, 14, 64, 3, 3, 1, 1, 2, "Dilation 2"),
    ]

    results = {}

    for test_case in test_cases:
        (
            batch,
            in_ch,
            h,
            w,
            out_ch,
            kh,
            kw,
            stride,
            padding,
            dilation,
            test_name,
        ) = test_case

        print(f"\nTest Case: {test_name}")
        print(
            f"  Config: B={batch}, C_in={in_ch}, H={h}, W={w}, C_out={out_ch}"
        )
        print(f"  Kernel: {kh}x{kw}, S={stride}, P={padding}, D={dilation}")
        print("-" * 80)

        results[test_name] = {}

        # Test across all configurations
        for config in TEST_CONFIGS:
            print(f"  {config['name']:12} | ", end="")

            result, error = run_single_test(
                config,
                batch,
                in_ch,
                h,
                w,
                out_ch,
                kh,
                kw,
                stride,
                padding,
                dilation,
                test_name,
            )

            if error:
                print(f"ERROR: {error}")
                results[test_name][config["name"]] = {"error": error}
                continue

            results[test_name][config["name"]] = result

            if "note" in result:
                print(f"REFERENCE ONLY ({result['note']})")
                print(
                    f"            | PyTorch stats: min={result['torch_stats']['min']:.6e}, max={result['torch_stats']['max']:.6e}, mean={result['torch_stats']['mean']:.6e}"
                )
            else:
                status = "PASS" if result["passed"] else "FAIL"
                print(
                    f"{status:4} | Max diff: {result['max_diff']:.2e}, Rel err: {result['rel_error']:.2e}"
                )

                if not result["passed"]:
                    print(
                        f"            | PyTorch stats: min={result['torch_stats']['min']:.6e}, max={result['torch_stats']['max']:.6e}"
                    )
                    print(
                        f"            | Custom stats:  min={result['custom_stats']['min']:.6e}, max={result['custom_stats']['max']:.6e}"
                    )

    return results


def compare_precision_accuracy():
    """Compare numerical accuracy across different precisions for PyTorch reference"""

    print("\n" + "=" * 100)
    print("NUMERICAL PRECISION COMPARISON (PyTorch Reference)")
    print("=" * 100)

    # Use a test case that should be numerically sensitive
    batch, in_ch, h, w, out_ch, kh, kw = 4, 32, 32, 32, 64, 3, 3
    stride, padding, dilation = 1, 1, 1

    print(
        f"Test configuration: B={batch}, C_in={in_ch}, H={h}, W={w}, C_out={out_ch}, K={kh}x{kw}"
    )

    # Generate the same random inputs but convert to different precisions
    torch.manual_seed(42)
    input_fp32_cpu = torch.randn(batch, in_ch, h, w, dtype=torch.float32)
    grad_output_fp32_cpu = torch.randn(batch, out_ch, h, w, dtype=torch.float32)

    # Convert to different precisions and devices
    configs_for_comparison = []
    if torch.cuda.is_available():
        configs_for_comparison.extend(
            [
                {
                    "tensor_in": input_fp32_cpu.cuda(),
                    "tensor_out": grad_output_fp32_cpu.cuda(),
                    "name": "CUDA FP32",
                },
            ]
        )

    configs_for_comparison.extend(
        [
            {
                "tensor_in": input_fp32_cpu,
                "tensor_out": grad_output_fp32_cpu,
                "name": "CPU FP32",
            },
            {
                "tensor_in": input_fp32_cpu.double(),
                "tensor_out": grad_output_fp32_cpu.double(),
                "name": "CPU FP64",
            },
        ]
    )

    weight_shape = (out_ch, in_ch, kh, kw)
    stride_tuple = (stride, stride)
    padding_tuple = (padding, padding)
    dilation_tuple = (dilation, dilation)

    torch_results = {}

    for config in configs_for_comparison:
        input_tensor = config["tensor_in"]
        grad_output = config["tensor_out"]
        config_name = config["name"]

        print(f"\n{config_name}:")

        try:
            torch_dweight = torch_conv2d_dweight(
                input_tensor,
                weight_shape,
                grad_output,
                stride_tuple,
                padding_tuple,
                dilation_tuple,
            )

            torch_results[config_name] = torch_dweight

            print(f"  Shape: {torch_dweight.shape}")
            print(f"  Dtype: {torch_dweight.dtype}")
            print(f"  Device: {torch_dweight.device}")
            print(
                f"  Stats: min={torch_dweight.min():.8e}, max={torch_dweight.max():.8e}"
            )
            print(
                f"         mean={torch_dweight.mean():.8e}, std={torch_dweight.std():.8e}"
            )

        except Exception as e:
            print(f"  ERROR: {e}")

    # Compare between precisions
    if len(torch_results) >= 2:
        print(f"\n{'Comparison Analysis':^50}")
        print("-" * 50)

        # reference_names = list(torch_results.keys())

        # Compare CPU FP32 vs CPU FP64 if both available
        if "CPU FP32" in torch_results and "CPU FP64" in torch_results:
            fp32_result = torch_results[
                "CPU FP32"
            ].double()  # Convert to fp64 for comparison
            fp64_result = torch_results["CPU FP64"]

            diff = torch.abs(fp32_result - fp64_result)
            max_diff = torch.max(diff).item()
            rel_diff = torch.norm(diff) / torch.norm(fp64_result)

            print("CPU FP32 vs CPU FP64:")
            print(f"  Max absolute difference: {max_diff:.2e}")
            print(f"  Relative difference: {rel_diff:.2e}")

        # Compare CUDA FP32 vs CPU FP32 if both available
        if "CUDA FP32" in torch_results and "CPU FP32" in torch_results:
            cuda_result = torch_results["CUDA FP32"].cpu()
            cpu_result = torch_results["CPU FP32"]

            diff = torch.abs(cuda_result - cpu_result)
            max_diff = torch.max(diff).item()
            rel_diff = torch.norm(diff) / torch.norm(cpu_result)

            print("CUDA FP32 vs CPU FP32:")
            print(f"  Max absolute difference: {max_diff:.2e}")
            print(f"  Relative difference: {rel_diff:.2e}")


def benchmark_precision_performance():
    """Benchmark performance across different precisions"""

    if not torch.cuda.is_available():
        print("Skipping performance benchmarks - CUDA not available")
        return

    print("\n" + "=" * 100)
    print("PERFORMANCE COMPARISON ACROSS PRECISIONS")
    print("=" * 100)

    # Benchmark configuration
    batch, in_ch, h, w, out_ch, kh, kw = 8, 128, 56, 56, 256, 3, 3
    stride, padding, dilation = 1, 1, 1

    print(
        f"Benchmark config: B={batch}, C_in={in_ch}, H={h}, W={w}, C_out={out_ch}, K={kh}x{kw}"
    )

    warmup_iters = 5
    benchmark_iters = 20

    # Create test tensors
    input_tensor = torch.randn(
        batch, in_ch, h, w, device="cuda", dtype=torch.float32
    ).abs()
    h_out = (h + 2 * padding - dilation * (kh - 1) - 1) // stride + 1
    w_out = (w + 2 * padding - dilation * (kw - 1) - 1) // stride + 1
    grad_output = torch.randn(
        batch, out_ch, h_out, w_out, device="cuda", dtype=torch.float32
    ).abs()

    weight_shape = (out_ch, in_ch, kh, kw)
    stride_tuple = (stride, stride)
    padding_tuple = (padding, padding)
    dilation_tuple = (dilation, dilation)

    _ = torch_conv2d_dweight(
        input_tensor,
        weight_shape,
        grad_output,
        stride_tuple,
        padding_tuple,
        dilation_tuple,
    )
    # Benchmark PyTorch CUDA FP32
    for _ in range(warmup_iters):
        _ = torch_conv2d_dweight(
            input_tensor,
            weight_shape,
            grad_output,
            stride_tuple,
            padding_tuple,
            dilation_tuple,
        )
    torch.cuda.synchronize()

    start_time = time.time()
    for _ in range(benchmark_iters):
        _ = torch_conv2d_dweight(
            grad_output,
            input_tensor,
            weight_shape,
            stride_tuple,
            padding_tuple,
            dilation_tuple,
        )
    torch.cuda.synchronize()
    torch_cuda_time = (time.time() - start_time) / benchmark_iters

    # Benchmark Custom Triton CUDA FP32
    for _ in range(warmup_iters):
        _ = conv2d_dweight(
            input_tensor,
            weight_shape,
            grad_output,
            stride_tuple,
            padding_tuple,
            dilation_tuple,
        )
    torch.cuda.synchronize()

    start_time = time.time()
    for _ in range(benchmark_iters):
        _ = conv2d_dweight(
            input_tensor,
            weight_shape,
            grad_output,
            stride_tuple,
            padding_tuple,
            dilation_tuple,
        )
    torch.cuda.synchronize()
    custom_cuda_time = (time.time() - start_time) / benchmark_iters

    # Benchmark PyTorch CPU FP32
    input_cpu = input_tensor.cpu()
    grad_output_cpu = grad_output.cpu()

    start_time = time.time()
    for _ in range(benchmark_iters):
        _ = torch_conv2d_dweight(
            input_cpu,
            weight_shape,
            grad_output_cpu,
            stride_tuple,
            padding_tuple,
            dilation_tuple,
        )
    torch_cpu_fp32_time = (time.time() - start_time) / benchmark_iters

    # Benchmark PyTorch CPU FP64
    input_cpu_fp64 = input_cpu.double()
    grad_output_cpu_fp64 = grad_output_cpu.double()

    start_time = time.time()
    for _ in range(benchmark_iters):
        _ = torch_conv2d_dweight(
            input_cpu_fp64,
            weight_shape,
            grad_output_cpu_fp64,
            stride_tuple,
            padding_tuple,
            dilation_tuple,
        )
    torch_cpu_fp64_time = (time.time() - start_time) / benchmark_iters

    # Results
    print(f"\nPerformance Results (averaged over {benchmark_iters} runs):")
    print(f"  PyTorch CUDA FP32:  {torch_cuda_time * 1000:.3f} ms")
    print(
        f"  Custom Triton CUDA: {custom_cuda_time * 1000:.3f} ms  (Speedup: {torch_cuda_time / custom_cuda_time:.2f}x)"
    )
    print(
        f"  PyTorch CPU FP32:   {torch_cpu_fp32_time * 1000:.3f} ms  (vs CUDA: {torch_cpu_fp32_time / torch_cuda_time:.1f}x slower)"
    )
    print(
        f"  PyTorch CPU FP64:   {torch_cpu_fp64_time * 1000:.3f} ms  (vs FP32: {torch_cpu_fp64_time / torch_cpu_fp32_time:.1f}x slower)"
    )


def main():
    """Run all precision and device comparison tests"""
    print("Conv2D DWeight Kernel - Precision & Device Comparison")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")

    # Run precision comparison tests
    test_results = test_precision_comparison()

    # Run numerical precision analysis
    compare_precision_accuracy()

    # Run performance benchmarks
    benchmark_precision_performance()

    # Summary
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)

    # Count passes/fails for CUDA FP32 tests (where custom kernel can be compared)
    cuda_fp32_results = []
    for test_name, configs in test_results.items():
        if (
            "CUDA FP32" in configs
            and "error" not in configs["CUDA FP32"]
            and "note" not in configs["CUDA FP32"]
        ):
            cuda_fp32_results.append(configs["CUDA FP32"]["passed"])

    if cuda_fp32_results:
        passed_count = sum(cuda_fp32_results)
        total_count = len(cuda_fp32_results)
        print(
            f"Custom kernel correctness (CUDA FP32): {passed_count}/{total_count} tests passed"
        )

    print(
        "Reference computations completed for all available device/precision combinations"
    )
    print(
        "Custom Triton kernel is only available for CUDA FP32 - this is expected"
    )


if __name__ == "__main__":
    main()
