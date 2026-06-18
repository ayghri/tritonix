#!/usr/bin/env python3
"""
Precision comparison test suite for conv2d_dweight kernel.
Compares our float32 Triton kernel against PyTorch's float32 and float64 CPU implementations.
"""

import torch
import numpy as np
import torch.nn.grad as G
from conv2d_bwd import conv2d_dweight
import configs

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
configs.disable_torch_optimizations()


def torch_conv2d_dweight_reference(
    grad_output, input_tensor, weight_shape, stride, padding, dilation
):
    return G.conv2d_weight(
        input_tensor,
        weight_shape,
        grad_output,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )


def run_precision_comparison():
    """
    Compare our float32 CUDA kernel with PyTorch's float32 and float64 CPU implementations.
    """

    print("=" * 80)
    print("PRECISION COMPARISON TEST")
    print("Comparing Triton float32 CUDA kernel vs PyTorch CPU implementations")
    print("=" * 80)

    # Disable optimizations for fair comparison
    configs.disable_torch_optimizations()

    # Test configurations
    test_cases = [
        # (batch, in_ch, h, w, out_ch, kh, kw, stride, padding, dilation, name)
        (2, 8, 16, 16, 16, 3, 3, 1, 1, 1, "Small conv"),
        (4, 32, 32, 32, 64, 3, 3, 1, 1, 1, "Medium conv"),
        (1, 64, 28, 28, 128, 3, 3, 2, 1, 1, "Stride 2 conv"),
        (2, 16, 24, 24, 32, 5, 5, 1, 2, 1, "Large kernel"),
        (1, 32, 14, 14, 64, 3, 3, 1, 1, 2, "Dilated conv"),
    ]

    results = []

    for i, (
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
    ) in enumerate(test_cases):
        print(f"\nTest {i + 1}/{len(test_cases)}: {test_name}")
        print(
            f"  Config: B={batch}, C_in={in_ch}, H={h}, W={w}, C_out={out_ch}"
        )
        print(f"  Kernel: {kh}x{kw}, S={stride}, P={padding}, D={dilation}")

        # Calculate output dimensions
        h_out = (h + 2 * padding - dilation * (kh - 1) - 1) // stride + 1
        w_out = (w + 2 * padding - dilation * (kw - 1) - 1) // stride + 1

        stride_tuple = (stride, stride) if isinstance(stride, int) else stride
        padding_tuple = (
            (padding, padding) if isinstance(padding, int) else padding
        )
        dilation_tuple = (
            (dilation, dilation) if isinstance(dilation, int) else dilation
        )

        weight_shape = (out_ch, in_ch, kh, kw)

        # Create input data in float64 for highest precision reference
        input_data_fp64 = torch.randn(batch, in_ch, h, w, dtype=torch.float64)
        grad_output_data_fp64 = torch.randn(
            batch, out_ch, h_out, w_out, dtype=torch.float64
        )

        # ===== 1. PyTorch float64 CPU (highest precision reference) =====
        print("  Computing PyTorch float64 CPU reference...")
        input_cpu_fp64 = input_data_fp64.to("cpu", dtype=torch.float64)
        grad_output_cpu_fp64 = grad_output_data_fp64.to(
            "cpu", dtype=torch.float64
        )

        torch_dweight_fp64 = torch_conv2d_dweight_reference(
            grad_output_cpu_fp64,
            input_cpu_fp64,
            weight_shape,
            stride_tuple,
            padding_tuple,
            dilation_tuple,
        )

        # ===== 2. PyTorch float32 CPU =====
        print("  Computing PyTorch float32 CPU...")
        input_cpu_fp32 = input_data_fp64.to("cpu", dtype=torch.float32)
        grad_output_cpu_fp32 = grad_output_data_fp64.to(
            "cpu", dtype=torch.float32
        )

        torch_dweight_fp32 = torch_conv2d_dweight_reference(
            grad_output_cpu_fp32,
            input_cpu_fp32,
            weight_shape,
            stride_tuple,
            padding_tuple,
            dilation_tuple,
        )

        # ===== 3. Our Triton float32 CUDA kernel =====
        print("  Computing Triton float32 CUDA...")
        if torch.cuda.is_available():
            input_cuda_fp32 = input_data_fp64.to("cuda", dtype=torch.float32)
            grad_output_cuda_fp32 = grad_output_data_fp64.to(
                "cuda", dtype=torch.float32
            )

            triton_dweight_fp32 = conv2d_dweight(
                input_cuda_fp32,
                weight_shape,
                grad_output_cuda_fp32,
                stride_tuple,
                padding_tuple,
                dilation_tuple,
            )

            # Move back to CPU for comparison
            triton_dweight_fp32_cpu = triton_dweight_fp32.cpu()
        else:
            print("    CUDA not available, skipping Triton kernel")
            triton_dweight_fp32_cpu = None

        # ===== Comparisons =====
        if (
            triton_dweight_fp32_cpu is not None
            and torch_dweight_fp64 is not None
            and torch_dweight_fp32 is not None
        ):
            print("\n  === COMPARISON RESULTS ===")

            # Convert all to same precision for comparison
            torch_fp64_as_fp32 = torch_dweight_fp64.float()
            torch_fp32_native = torch_dweight_fp32
            triton_fp32_native = triton_dweight_fp32_cpu

            # 1. Compare PyTorch FP32 vs FP64 (both on CPU)
            diff_pytorch_precisions = torch.abs(
                torch_fp32_native - torch_fp64_as_fp32
            )
            max_diff_pytorch = torch.max(diff_pytorch_precisions).item()
            rel_error_pytorch = torch.norm(diff_pytorch_precisions) / (
                torch.norm(torch_fp64_as_fp32) + 1e-12
            )

            print(f"  PyTorch FP32 vs FP64 (CPU):")
            print(f"    Max absolute diff: {max_diff_pytorch:.2e}")
            print(f"    Relative error:    {rel_error_pytorch:.2e}")

            # 2. Compare Triton FP32 vs PyTorch FP64 reference
            diff_triton_vs_fp64 = torch.abs(
                triton_fp32_native - torch_fp64_as_fp32
            )
            max_diff_triton_fp64 = torch.max(diff_triton_vs_fp64).item()
            rel_error_triton_fp64 = torch.norm(diff_triton_vs_fp64) / (
                torch.norm(torch_fp64_as_fp32) + 1e-12
            )

            print(f"  Triton FP32 (CUDA) vs PyTorch FP64 (CPU):")
            print(f"    Max absolute diff: {max_diff_triton_fp64:.2e}")
            print(f"    Relative error:    {rel_error_triton_fp64:.2e}")

            # 3. Compare Triton FP32 vs PyTorch FP32
            diff_triton_vs_fp32 = torch.abs(
                triton_fp32_native - torch_fp32_native
            )
            max_diff_triton_fp32 = torch.max(diff_triton_vs_fp32).item()
            rel_error_triton_fp32 = torch.norm(diff_triton_vs_fp32) / (
                torch.norm(torch_fp32_native) + 1e-12
            )

            print(f"  Triton FP32 (CUDA) vs PyTorch FP32 (CPU):")
            print(f"    Max absolute diff: {max_diff_triton_fp32:.2e}")
            print(f"    Relative error:    {rel_error_triton_fp32:.2e}")

            # ===== Accuracy Assessment =====
            # Define tolerance levels
            rtol_strict = 1e-5
            atol_strict = 1e-6
            rtol_relaxed = 1e-4
            atol_relaxed = 1e-5

            triton_vs_fp64_strict = torch.allclose(
                triton_fp32_native,
                torch_fp64_as_fp32,
                rtol=rtol_strict,
                atol=atol_strict,
            )
            triton_vs_fp64_relaxed = torch.allclose(
                triton_fp32_native,
                torch_fp64_as_fp32,
                rtol=rtol_relaxed,
                atol=atol_relaxed,
            )
            triton_vs_fp32_strict = torch.allclose(
                triton_fp32_native,
                torch_fp32_native,
                rtol=rtol_strict,
                atol=atol_strict,
            )
            triton_vs_fp32_relaxed = torch.allclose(
                triton_fp32_native,
                torch_fp32_native,
                rtol=rtol_relaxed,
                atol=atol_relaxed,
            )

            print("\n  === ACCURACY ASSESSMENT ===")
            print(
                f"  Triton vs PyTorch FP64 (strict):  {'✓ PASS' if triton_vs_fp64_strict else '✗ FAIL'}"
            )
            print(
                f"  Triton vs PyTorch FP64 (relaxed): {'✓ PASS' if triton_vs_fp64_relaxed else '✗ FAIL'}"
            )
            print(
                f"  Triton vs PyTorch FP32 (strict):  {'✓ PASS' if triton_vs_fp32_strict else '✗ FAIL'}"
            )
            print(
                f"  Triton vs PyTorch FP32 (relaxed): {'✓ PASS' if triton_vs_fp32_relaxed else '✗ FAIL'}"
            )

            # Store results for summary
            results.append(
                {
                    "name": test_name,
                    "config": f"B={batch}, C_in={in_ch}, H={h}, W={w}, C_out={out_ch}, K={kh}x{kw}",
                    "pytorch_precision_diff": max_diff_pytorch,
                    "triton_vs_fp64_diff": max_diff_triton_fp64,
                    "triton_vs_fp32_diff": max_diff_triton_fp32,
                    "triton_vs_fp64_rel_error": rel_error_triton_fp64.item(),
                    "triton_vs_fp32_rel_error": rel_error_triton_fp32.item(),
                    "triton_vs_fp64_strict": triton_vs_fp64_strict,
                    "triton_vs_fp64_relaxed": triton_vs_fp64_relaxed,
                    "triton_vs_fp32_strict": triton_vs_fp32_strict,
                    "triton_vs_fp32_relaxed": triton_vs_fp32_relaxed,
                }
            )

            # Show some statistics
            print("\n  === TENSOR STATISTICS ===")
            print(
                f"  PyTorch FP64:  min={torch_dweight_fp64.min():.6f}, max={torch_dweight_fp64.max():.6f}, mean={torch_dweight_fp64.mean():.6f}"
            )
            print(
                f"  PyTorch FP32:  min={torch_dweight_fp32.min():.6f}, max={torch_dweight_fp32.max():.6f}, mean={torch_dweight_fp32.mean():.6f}"
            )
            print(
                f"  Triton FP32:   min={triton_dweight_fp32_cpu.min():.6f}, max={triton_dweight_fp32_cpu.max():.6f}, mean={triton_dweight_fp32_cpu.mean():.6f}"
            )
        else:
            print("  Skipping comparisons due to missing results")

    # ===== FINAL SUMMARY =====
    print("\n" + "=" * 80)
    print("PRECISION COMPARISON SUMMARY")
    print("=" * 80)

    if results:
        # Count passes
        fp64_strict_passes = sum(
            1 for r in results if r["triton_vs_fp64_strict"]
        )
        fp64_relaxed_passes = sum(
            1 for r in results if r["triton_vs_fp64_relaxed"]
        )
        fp32_strict_passes = sum(
            1 for r in results if r["triton_vs_fp32_strict"]
        )
        fp32_relaxed_passes = sum(
            1 for r in results if r["triton_vs_fp32_relaxed"]
        )

        total_tests = len(results)

        print(f"Total tests: {total_tests}")
        print(
            f"Triton vs PyTorch FP64 (strict):  {fp64_strict_passes}/{total_tests} passed"
        )
        print(
            f"Triton vs PyTorch FP64 (relaxed): {fp64_relaxed_passes}/{total_tests} passed"
        )
        print(
            f"Triton vs PyTorch FP32 (strict):  {fp32_strict_passes}/{total_tests} passed"
        )
        print(
            f"Triton vs PyTorch FP32 (relaxed): {fp32_relaxed_passes}/{total_tests} passed"
        )

        # Average errors
        avg_triton_vs_fp64_error = np.mean(
            [r["triton_vs_fp64_rel_error"] for r in results]
        )
        avg_triton_vs_fp32_error = np.mean(
            [r["triton_vs_fp32_rel_error"] for r in results]
        )

        print(f"\nAverage relative errors:")
        print(f"Triton vs PyTorch FP64: {avg_triton_vs_fp64_error:.2e}")
        print(f"Triton vs PyTorch FP32: {avg_triton_vs_fp32_error:.2e}")

        # Worst case analysis
        worst_fp64 = max(results, key=lambda x: x["triton_vs_fp64_rel_error"])
        worst_fp32 = max(results, key=lambda x: x["triton_vs_fp32_rel_error"])

        print(f"\nWorst case scenarios:")
        print(
            f"Worst FP64 comparison: {worst_fp64['name']} (rel_error: {worst_fp64['triton_vs_fp64_rel_error']:.2e})"
        )
        print(
            f"Worst FP32 comparison: {worst_fp32['name']} (rel_error: {worst_fp32['triton_vs_fp32_rel_error']:.2e})"
        )

        # Detailed results table
        print(
            f"\n{'Test Name':<20} {'vs FP64':<12} {'vs FP32':<12} {'FP64 Pass':<10} {'FP32 Pass':<10}"
        )
        print("-" * 80)
        for r in results:
            fp64_status = (
                "✓ STRICT"
                if r["triton_vs_fp64_strict"]
                else ("✓ RELAX" if r["triton_vs_fp64_relaxed"] else "✗ FAIL")
            )
            fp32_status = (
                "✓ STRICT"
                if r["triton_vs_fp32_strict"]
                else ("✓ RELAX" if r["triton_vs_fp32_relaxed"] else "✗ FAIL")
            )

            print(
                f"{r['name']:<20} {r['triton_vs_fp64_rel_error']:<12.2e} {r['triton_vs_fp32_rel_error']:<12.2e} {fp64_status:<10} {fp32_status:<10}"
            )

    print("=" * 80)

    return results


def test_precision_edge_cases():
    """Test edge cases that might expose precision issues"""

    print("\n" + "=" * 80)
    print("PRECISION EDGE CASES")
    print("=" * 80)

    edge_cases = [
        # Very small values
        ("Small values", lambda: torch.randn(2, 4, 8, 8) * 1e-3),
        # Large values
        ("Large values", lambda: torch.randn(2, 4, 8, 8) * 1e3),
        # Mixed scales
        (
            "Mixed scales",
            lambda: torch.randn(2, 4, 8, 8)
            * torch.tensor([1e-2, 1e0, 1e2, 1e-1]).view(1, 4, 1, 1),
        ),
        # Near-zero values
        ("Near-zero", lambda: torch.randn(2, 4, 8, 8) * 1e-6),
    ]

    for case_name, data_gen in edge_cases:
        print(f"\nTesting: {case_name}")

        # Generate test data
        input_data = data_gen()
        grad_output_data = data_gen()[:, :2, :, :]  # Reduce output channels

        if torch.cuda.is_available():
            # Test configuration
            batch, in_ch, h, w = input_data.shape
            _, out_ch, h_out, w_out = grad_output_data.shape
            kh, kw = 3, 3

            # PyTorch FP64 reference
            input_fp64 = input_data.to("cpu", dtype=torch.float64)
            grad_fp64 = grad_output_data.to("cpu", dtype=torch.float64)
            torch_result = torch_conv2d_dweight_reference(
                grad_fp64,
                input_fp64,
                (out_ch, in_ch, kh, kw),
                (1, 1),
                (1, 1),
                (1, 1),
            )

            # Triton FP32
            input_fp32 = input_data.to("cuda", dtype=torch.float32)
            grad_fp32 = grad_output_data.to("cuda", dtype=torch.float32)
            triton_result = conv2d_dweight(
                input_fp32,
                (out_ch, in_ch, kh, kw),
                grad_fp32,
                (1, 1),
                (1, 1),
                (1, 1),
            )

            if torch_result is not None:
                # Compare
                diff = torch.abs(triton_result.cpu() - torch_result.float())
                max_diff = torch.max(diff).item()
                rel_error = torch.norm(diff) / (
                    torch.norm(torch_result.float()) + 1e-12
                )

                print(f"  Max diff: {max_diff:.2e}, Rel error: {rel_error:.2e}")

                # Check for numerical issues
                has_nan = (
                    torch.isnan(triton_result).any()
                    or torch.isnan(torch_result).any()
                )
                has_inf = (
                    torch.isinf(triton_result).any()
                    or torch.isinf(torch_result).any()
                )

                if has_nan:
                    print("  ⚠️  NaN detected!")
                if has_inf:
                    print("  ⚠️  Inf detected!")

                if not has_nan and not has_inf and rel_error < 1e-3:
                    print("  ✓ PASSED")
                else:
                    print("  ✗ FAILED or UNSTABLE")
            else:
                print("  ✗ PyTorch reference failed")


def main():
    """Run the precision comparison tests"""

    print("Conv2D DWeight Precision Comparison")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")

    # Run main precision comparison
    results = run_precision_comparison()

    # Run edge case tests
    if torch.cuda.is_available():
        test_precision_edge_cases()
    else:
        print("\nSkipping edge case tests (CUDA not available)")

    # Final assessment
    if results:
        all_relaxed_pass = all(r["triton_vs_fp32_relaxed"] for r in results)
        print(
            f"\nFINAL RESULT: {'✓ ALL TESTS PASSED' if all_relaxed_pass else '✗ SOME TESTS FAILED'}"
        )

    return results


if __name__ == "__main__":
    main()
