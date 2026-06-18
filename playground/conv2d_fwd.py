import torch
import triton
from triton.runtime import Autotuner

import configs

# import triton.conv2d_kernels as conv2d_kernels
from tritonix.conv import conv2d_kernels

conv2d_forward_kernel = Autotuner(
    conv2d_kernels.conv2d_forward_kernel,
    conv2d_kernels.conv2d_forward_kernel.arg_names,
    configs=configs.get_autotune_configs_conv2d(),
    key=[
        "BATCH_SIZE",
        "C_IN",
        "H_IN",
        "W_IN",
        "C_OUT",
        "H_OUT",
        "W_OUT",
        "F_H",
        "F_W",
        "str_h",
        "str_w",
        "pad_h",
        "pad_w",
        "dil_h",
        "dil_w",
    ],
    reset_to_zero=None,
    restore_value=None,
)


def conv2d_forward(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride,
    padding,
    dilation,
):
    BATCH_SIZE, C_IN, H_IN, W_IN = input.shape
    C_OUT, t_C, F_H, F_W = weight.shape
    assert t_C == C_IN, "Input and weight channels must match"

    str_h, str_w = stride
    pad_h, pad_w = padding
    dil_h, dil_w = dilation

    H_OUT = (H_IN + 2 * pad_h - dil_h * (F_H - 1) - 1) // str_h + 1
    W_OUT = (W_IN + 2 * pad_w - dil_w * (F_W - 1) - 1) // str_w + 1

    GEMM_M = BATCH_SIZE * H_OUT * W_OUT
    GEMM_K = C_IN * F_H * F_W
    GEMM_N = C_OUT
    output = torch.empty(
        (BATCH_SIZE, C_OUT, H_OUT, W_OUT),
        dtype=input.dtype,
        device=input.device,
    )

    conv2d_forward_kernel[
        lambda META: (
            triton.cdiv(GEMM_M, META["BLOCK_SIZE_M"]),
            triton.cdiv(GEMM_N, META["BLOCK_SIZE_N"]),
        )
    ](
        output,
        input,
        weight,
        bias,
        BATCH_SIZE,
        C_IN,
        H_IN,
        W_IN,
        C_OUT,
        H_OUT,
        W_OUT,
        F_H,
        F_W,
        str_h,
        str_w,
        pad_h,
        pad_w,
        dil_h,
        dil_w,
        GEMM_M,
        GEMM_N,
        GEMM_K,
    )

    return output


if __name__ == "__main__":
    torch.manual_seed(0)
    DEVICE = triton.runtime.driver.active.get_active_torch_device()
    # DTYPE = torch.float16
    DTYPE = torch.float32
    configs.enable_torch_optimizations()
    # configs.disable_torch_optimizations()
    # torch.set_float32_matmul_precision("high")

    C_IN, H, W = 512, 224, 224
    C_OUT = 1024
    K_H = 3
    K_W = 3
    stride = (2, 2)
    padding = (1, 1)
    dilation = (1, 1)

    weight = torch.randn(C_OUT, C_IN, K_H, K_W, device=DEVICE, dtype=DTYPE)
    bias = torch.randn(C_OUT, device=DEVICE, dtype=DTYPE)

    class Conv2d(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(
                C_IN,
                C_OUT,
                (K_H, K_W),
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
            self.conv.weight = torch.nn.Parameter(weight)
            if self.conv.bias is not None:
                self.conv.bias = torch.nn.Parameter(bias)

        def forward(self, x):
            return self.conv(x)

    model = Conv2d().to(DEVICE)
    compiled_model = torch.compile(model)

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["B"],  # Argument names to use as an x-axis for the plot.
            x_vals=[
                2**i for i in range(4, 8)
            ],  # Different possible values for `x_name`.
            x_log=True,  # x axis is logarithmic.
            line_arg="provider",  # Argument name whose value corresponds to a different line in the plot.
            line_vals=[
                "triton",
                "torch",
                "torch-compile",
            ],  # Possible values for `line_arg`.
            line_names=[
                "Triton",
                "Torch",
                "Torch (Compiled)",
            ],  # Label name for the lines.
            styles=[
                ("blue", "-"),
                ("green", "-"),
                ("red", "-"),
            ],  # Line styles.
            ylabel="ms",  # Label name for the y-axis.
            plot_name="conv2d-performance",  # Name for the plot. Used also as a file name for saving the plot.
            args={},  # Values for function arguments not in `x_names` and `y_name`.
        )
    )
    def benchmark(B, provider):
        input = torch.randn(B, C_IN, H, W, device=DEVICE, dtype=DTYPE)

        quantiles = [0.5, 0.2, 0.8]
        ms, min_ms, max_ms = 0.0, 0.0, 0.0

        @torch.no_grad()
        def tr_conv2d():
            return model(input)

        @torch.no_grad()
        def compiled_conv2d():
            return compiled_model(input)

        def tl_conv2d():
            return conv2d_forward(
                input, weight, bias, stride, padding, dilation
            )

        if provider == "torch":
            bench_result = triton.testing.do_bench(
                tr_conv2d, quantiles=quantiles
            )
            if bench_result:
                ms, min_ms, max_ms = bench_result
        elif provider == "triton":
            bench_result = triton.testing.do_bench(
                tl_conv2d, quantiles=quantiles
            )
            if bench_result:
                ms, min_ms, max_ms = bench_result
        elif provider == "torch-compile":
            bench_result = triton.testing.do_bench(
                compiled_conv2d, quantiles=quantiles
            )
            if bench_result:
                ms, min_ms, max_ms = bench_result
        # gbps = (
        #     lambda ms: 3
        #     * input.numel()
        #     * input.element_size()
        #     * 1e-9
        #     / (ms * 1e-3)
        # )
        # return gbps(ms), gbps(max_ms), gbps(min_ms)
        return ms, min_ms, max_ms

    benchmark.run(print_data=True, show_plots=True)

    for B in [2**i for i in range(4, 10)]:
        input = torch.randn(B, C_IN, H, W, device=DEVICE, dtype=DTYPE)
        # weight = torch.randn(C_OUT, C_IN, K_H, K_W, device=DEVICE, dtype=DTYPE)
        # bias = torch.randn(C_OUT, device=DEVICE, dtype=DTYPE)

        @torch.no_grad()
        def tr_conv2d():
            return torch.nn.functional.conv2d(
                input, weight, bias, stride, padding, dilation
            )

        def tl_conv2d():
            return conv2d_forward(
                input, weight, bias, stride, padding, dilation
            )

        triton_out = tl_conv2d()
        torch_out = tr_conv2d()
        print(f"Input shape: {input.shape}")
        print(f"Triton output shape: {triton_out.shape}")
        print(f"Torch output shape: {torch_out.shape}")
        print(f"Norm of Triton output: {triton_out.norm()}")
        print(f"Norm of Torch output: {torch_out.norm()}")
        print(f"Norm of difference: {torch.norm(triton_out - torch_out)}")
        print(f"Max difference: {torch.max(torch.abs(triton_out - torch_out))}")
        # print(
        #     f"Max rel difference: {0.5 * torch.max(torch.abs(triton_out - torch_out) / (torch.abs(torch_out) + torch.abs(triton_out) + 1e-5))}"
        # )
        print(
            f"Output match: {torch.allclose(triton_out, torch_out, atol=1e-2, rtol=1e-2)}"
        )
