import torch
from tabulate import tabulate

# import torch.nn.functional as F
import torch.nn.grad as G
import triton
from triton.runtime import Autotuner


import configs
import conv2d_kernels as kernels


conv2d_dweight_kernel = Autotuner(
    kernels.conv2d_grad_weight_kernel,
    kernels.conv2d_grad_weight_kernel.arg_names,
    configs=configs.get_autotune_conv2d_bwd_configs(),
    key=[
        "BATCH_SIZE",
        "C_IN",
        "H_IN",
        "W_IN",
        "C_OUT",
        "H_OUT",
        "W_OUT",
        "FILTER_H",
        "FILTER_W",
        "stride_h",
        "stride_w",
        "pad_h",
        "pad_w",
        "dil_h",
        "dil_w",
        "GEMM_M",
        "GEMM_N",
        "GEMM_K",
    ],
    reset_to_zero=None,
    restore_value=None,
)

conv2d_dweight_kernel_atomic = Autotuner(
    kernels.conv2d_grad_weight_kernel_atomic,
    kernels.conv2d_grad_weight_kernel_atomic.arg_names,
    configs=configs.get_autotune_conv2d_bwd_configs(),
    key=[
        "BATCH_SIZE",
        "C_IN",
        "H_IN",
        "W_IN",
        "C_OUT",
        "H_OUT",
        "W_OUT",
        "FILTER_H",
        "FILTER_W",
        "stride_h",
        "stride_w",
        "pad_h",
        "pad_w",
        "dil_h",
        "dil_w",
        "GEMM_M",
        "GEMM_N",
        "GEMM_K",
    ],
    reset_to_zero=["grad_weight_ptr"],
    restore_value=None,
)


def conv2d_dweight(
    input_tensor, weight_shape, grad_output, stride, padding, dilation
):
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    dil_h, dil_w = dilation
    batch_size, c_in, h_in, h_in = input_tensor.shape
    c_out, c_in, filter_h, filter_w = weight_shape

    h_out = (h_in + 2 * pad_h - dil_h * (filter_h - 1) - 1) // str_h + 1
    w_out = (h_in + 2 * pad_w - dil_w * (filter_w - 1) - 1) // str_w + 1

    gemm_m = c_out
    gemm_n = c_in * filter_h * filter_w
    gemm_k = h_out * w_out
    # gemm_k = batch_size*h_out * w_out

    grad_weight = torch.zeros(
        (c_out, c_in, filter_h, filter_w),
        dtype=torch.float32,
        device=grad_output.device,
    )

    def grid(META):
        return (
            batch_size,
            triton.cdiv(gemm_m, META["BLOCK_SIZE_M"])
            * triton.cdiv(gemm_n, META["BLOCK_SIZE_N"]),
        )

    # def grid(META):
    #     return (
    #         triton.cdiv(gemm_m, META["BLOCK_SIZE_M"])
    #         * triton.cdiv(gemm_n, META["BLOCK_SIZE_N"]),
    #     )

    conv2d_dweight_kernel_atomic[grid](
    # conv2d_dweight_kernel[grid](
        input_tensor,
        grad_output,
        grad_weight,
        BATCH_SIZE=batch_size,
        C_IN=c_in,
        H_IN=h_in,
        W_IN=w_in,
        C_OUT=c_out,
        H_OUT=h_out,
        W_OUT=w_out,
        FILTER_H=filter_h,
        FILTER_W=filter_w,
        stride_h=stride_h,
        stride_w=stride_w,
        pad_h=pad_h,
        pad_w=pad_w,
        dil_h=dil_h,
        dil_w=dil_w,
        GEMM_M=gemm_m,
        GEMM_N=gemm_n,
        GEMM_K=gemm_k,
    )
    # print(
    return grad_weight


def torch_conv2d_dweight(
    input_tensor, weight_shape, grad_output, stride, padding, dilation
):
    return G.conv2d_weight(
        input_tensor,
        weight_shape,
        grad_output,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    DEVICE = triton.runtime.driver.active.get_active_torch_device()
    DTYPE = torch.float16
    # configs.disable_torch_optimizations()
    configs.enable_torch_optimizations()

    c_in, h_in, w_in = 3, 224, 224
    c_out = 64
    filter_h = 5
    filter_w = 5
    stride = (2, 2)
    padding = (2, 2)
    dilation = (1, 1)
    str_h, str_w = stride
    pad_h, pad_w = padding
    dil_h, dil_w = dilation

    h_out = (h_in + 2 * pad_h - dil_h * (filter_h - 1) - 1) // str_h + 1
    w_out = (w_in + 2 * pad_w - dil_w * (filter_w - 1) - 1) // str_w + 1

    # weight = torch.randn(C_OUT, C_IN, K_H, K_W, device=DEVICE, dtype=DTYPE)
    # bias = torch.randn(C_OUT, device=DEVICE, dtype=DTYPE)

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["B"],  # Argument names to use as an x-axis for the plot.
            x_vals=[
                2**i for i in range(4, 10)
            ],  # Different possible values for `x_name`.
            x_log=True,  # x axis is logarithmic.
            line_arg="provider",  # Argument name whose value corresponds to a different line in the plot.
            line_vals=[
                "triton",
                "torch",
                # "torch-compile",
            ],  # Possible values for `line_arg`.
            line_names=[
                "Triton",
                "Torch",
                # "Torch (Compiled)",
            ],  # Label name for the lines.
            styles=[
                ("blue", "-"),
                ("green", "-"),
                # ("red", "-"),
            ],  # Line styles.
            ylabel="ms",  # Label name for the y-axis.
            plot_name="conv2d-performance",  # Name for the plot. Used also as a file name for saving the plot.
            args={},  # Values for function arguments not in `x_names` and `y_name`.
        )
    )
    # compiled_model = torch.compile(torch_conv2d_dweight)

    def benchmark(B, provider):
        input = torch.randn(B, c_in, h_in, w_in, device=DEVICE, dtype=DTYPE)

        grad_output = torch.randn(
            B, c_out, h_out, w_out, device=DEVICE, dtype=DTYPE
        )

        quantiles = [0.5, 0.2, 0.8]
        ms, min_ms, max_ms = 0.0, 0.0, 0.0

        @torch.no_grad()
        def tr_conv2d():
            return torch_conv2d_dweight(
                input_tensor=input,
                weight_shape=(c_out, c_in, filter_h, filter_w),
                grad_output=grad_output,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )

        # @torch.no_grad()
        # def compiled_conv2d():
        #     return compiled_model(input)

        def tl_conv2d():
            conv2d_dweight(
                input_tensor=input,
                weight_shape=(c_out, c_in, filter_h, filter_w),
                grad_output=grad_output,
                stride=stride,
                padding=padding,
                dilation=dilation,
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
        return ms, min_ms, max_ms

    benchmark.run(print_data=True, show_plots=False)

    # Kernel: 3x3, S=1, P=1, D=1
    # C_IN, H_IN, W_IN = 3, 32, 32
    # C_OUT = 8
    # FILTER_H = 3
    # FILTER_W = 3
    c_in, h_in, w_in = 3, 224, 224
    c_out = 64
    filter_h = 5
    filter_w = 5
    stride = (2, 2)
    padding = (2, 2)
    dilation = (1, 1)
    str_h, str_w = stride
    pad_h, pad_w = padding
    dil_h, dil_w = dilation

    h_out = (h_in + 2 * pad_h - dil_h * (filter_h - 1) - 1) // str_h + 1
    w_out = (w_in + 2 * pad_w - dil_w * (filter_w - 1) - 1) // str_w + 1

    results = []
    Bs = [2**i for i in range(4, 10)]
    for B in Bs:
        input = torch.randn(B, c_in, h_in, w_in, device=DEVICE, dtype=DTYPE)
        grad_output = torch.randn(
            B, c_out, h_out, w_out, device=DEVICE, dtype=DTYPE
        )

        triton_out = conv2d_dweight(
            input_tensor=input,
            weight_shape=(c_out, c_in, filter_h, filter_w),
            grad_output=grad_output,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        torch_out = torch_conv2d_dweight(
            input_tensor=input,
            weight_shape=(c_out, c_in, filter_h, filter_w),
            grad_output=grad_output,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        results.append(
            [
                B,
                triton_out.norm().item(),
                torch_out.norm().item(),
                torch.norm(triton_out - torch_out).item(),
                torch.max(torch.abs(triton_out - torch_out)).item(),
            ]
        )
    print(
        tabulate(
            results,
            headers=["B", "Triton Norm", "Torch Norm", "Norm Diff", "Max Diff"],
        )
    )
