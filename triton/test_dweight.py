import torch
from tabulate import tabulate
import torch.nn.grad as G
import triton
from triton.runtime import Autotuner


import configs
import conv2d_kernels as conv2d_kernels


conv2d_dweight_kernel_atomic = Autotuner(
    conv2d_kernels.conv2d_grad_weight_kernel_atomic,
    conv2d_kernels.conv2d_grad_weight_kernel_atomic.arg_names,
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
    batch_size, c_in, _, _ = input_tensor.shape
    c_out, c_in, filter_h, filter_w = weight_shape

    # h_out = (h_in + 2 * pad_h - dil_h * (filter_h - 1) - 1) // str_h + 1
    # w_out = (h_in + 2 * pad_w - dil_w * (filter_w - 1) - 1) // str_w + 1
    _, _, h_out, w_out = grad_output.shape

    gemm_m = c_out
    gemm_n = c_in * filter_h * filter_w
    gemm_k = h_out * w_out

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

    conv2d_dweight_kernel_atomic[grid](
        input_tensor,
        grad_output,
        grad_weight,
        BATCH_SIZE=batch_size,
        C_IN=C_IN,
        H_IN=H_IN,
        W_IN=W_IN,
        C_OUT=C_OUT,
        H_OUT=H_OUT,
        W_OUT=W_OUT,
        FILTER_H=FILTER_H,
        FILTER_W=FILTER_W,
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
    DTYPE = torch.float32
    configs.disable_torch_optimizations()

    # C_IN, H_IN, W_IN = 3, 224, 224
    # C_OUT = 64
    # FILTER_H = 2
    # FILTER_W = 2
    # # stride = (2, 2)
    # dilation = (1, 1)

    C_IN = 3
    H_IN = 224
    W_IN = 224
    C_OUT = 64
    FILTER_H = 3
    FILTER_W = 3
    stride = (2, 2)
    padding = (0, 0)
    dilation = (1, 1)
    # stride_h = 2
    # stride_w = 2
    # pad_h = 0
    # pad_w = 0
    # dil_h = 1
    # dil_w = 1
    # for stride in [(1, 1), (2, 2)]:
    #     for padding in [(0, 0), (1, 1), (2, 2)]:
    # stride = (stride, stride)
    # print(f"Stride: {stride}, padding: {padding}")
    # padding = (0, 0)
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    dil_h, dil_w = dilation

    H_OUT = (H_IN + 2 * pad_h - dil_h * (FILTER_H - 1) - 1) // stride_h + 1
    W_OUT = (W_IN + 2 * pad_w - dil_w * (FILTER_W - 1) - 1) // stride_w + 1

    results = []
    Bs = [2**i for i in range(4, 10)]
    for B in Bs:
        input = torch.randint(
            0, 3, (B, C_IN, H_IN, W_IN), device=DEVICE, dtype=DTYPE
        )
        grad_output = 2.**torch.randint(
            -4, 4, (B, C_OUT, H_OUT, W_OUT), device=DEVICE, dtype=DTYPE
        )
        torch_out = torch_conv2d_dweight(
            input_tensor=input,
            weight_shape=(C_OUT, C_IN, FILTER_H, FILTER_W),
            grad_output=grad_output,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        triton_out = conv2d_dweight(
            input_tensor=input,
            weight_shape=(C_OUT, C_IN, FILTER_H, FILTER_W),
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
            headers=[
                "B",
                "Triton Norm",
                "Torch Norm",
                "Norm Diff",
                "Max Diff",
            ],
        )
    )
