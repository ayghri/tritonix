"""Benchmark PyTorch conv2d vs two Triton forward conv2d kernels.

Providers:
  torch       -> torch.nn.functional.conv2d (cuDNN)
  triton_ref  -> implicit_gemm_conv2d_fwd_kernel (baseline)
  triton_opt  -> conv2d_forward_kernel (optimized) via wrapper

Run: python bench_conv2d.py
"""

import torch
import triton

from tritonix.wrappers import conv2d_forward_triton_opt
from tritonix.wrappers import implicit_gemm_conv2d_fwd


def _max_abs(a, b):
    return (a - b).abs().max().item()


if __name__ == "__main__":
    DEVICE = triton.runtime.driver.active.get_active_torch_device()
    DTYPE = torch.float32  # choose fp16 for speed; change if desired
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(0)

    # Problem shape (modifiable)
    C, H, W = 64, 128, 128
    K = 128
    R = 3
    S = 3
    stride = (1, 1)
    padding = (1, 1)
    dilation = (1, 1)

    weight = torch.randn(K, C, R, S, device=DEVICE, dtype=DTYPE)
    bias = torch.randn(K, device=DEVICE, dtype=DTYPE)

    # Quick correctness spot check (small batch)
    test_in = torch.randn(2, C, H, W, device=DEVICE, dtype=DTYPE)
    ref_torch = torch.nn.functional.conv2d(
        test_in, weight, bias, stride, padding, dilation
    )
    ref_base = implicit_gemm_conv2d_fwd(
        test_in, weight, bias, stride, padding, dilation
    )
    ref_opt = conv2d_forward_triton_opt(
        test_in, weight, bias, stride, padding, dilation, use_tf32=True
    )
    print("Max abs diff (baseline vs torch):", _max_abs(ref_base, ref_torch))
    print("Max abs diff (opt vs torch):", _max_abs(ref_opt, ref_torch))

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["N"],
            x_vals=[2**i for i in range(4, 10)],
            x_log=True,
            line_arg="provider",
            line_vals=["torch", "triton_ref", "triton_opt"],
            line_names=["Torch (cuDNN)", "Triton Ref", "Triton Opt"],
            styles=[("blue", "-"), ("green", "-"), ("red", "-")],
            ylabel="Time (ms)",
            plot_name="conv2d-forward",
            args={},
        )
    )
    def benchmark(N, provider):
        inp = torch.randn(N, C, H, W, device=DEVICE, dtype=DTYPE)
        quantiles = [0.5, 0.2, 0.8]
        ms = min_ms = max_ms = 0.0

        @torch.no_grad()
        def torch_conv():
            return torch.nn.functional.conv2d(
                inp, weight, bias, stride, padding, dilation
            )

        def ref_conv():
            return implicit_gemm_conv2d_fwd(
                inp, weight, bias, stride, padding, dilation
            )

        def opt_conv():
            return conv2d_forward_triton_opt(
                inp, weight, bias, stride, padding, dilation, use_tf32=True
            )

        if provider == "torch":
            bench = triton.testing.do_bench(torch_conv, quantiles=quantiles)
        elif provider == "triton_ref":
            bench = triton.testing.do_bench(ref_conv, quantiles=quantiles)
        else:  # triton_opt
            for _ in range(2):
                opt_conv()  # warmup
            bench = triton.testing.do_bench(opt_conv, quantiles=quantiles)
        if bench:
            ms, min_ms, max_ms = bench
        return ms, min_ms, max_ms

    benchmark.run(print_data=True, show_plots=False)
