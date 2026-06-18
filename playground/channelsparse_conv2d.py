import torch
import triton
import triton.language as tl
import numpy as np

ALLOW_TF32 = True  # Set to False if you want to disable TF32 support

torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = ALLOW_TF32
torch.backends.cudnn.allow_tf32 = ALLOW_TF32


def get_autotune_config():
    return [
        triton.Config(
            {
                "BLOCK_SIZE_M": 16,
                "BLOCK_SIZE_N": 16,
                "BLOCK_SIZE_K": 16,
                "GROUP_SIZE_M": 4,
            },
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 32,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 16,
                "GROUP_SIZE_M": 4,
            },
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 32,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 16,
                "GROUP_SIZE_M": 4,
            },
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 16,
                "GROUP_SIZE_M": 8,
            },
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
            },
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 4,
            },
            num_stages=4,
            num_warps=8,
        ),
    ]


@triton.autotune(
    configs=get_autotune_config(),
    key=[
        "N",
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
)
@triton.jit
def channel_sparse_conv2d_forward_kernel(
    output_ptr,
    input_ptr,
    weight_ptr,
    lookup_ptr,
    bias_ptr,
    N,
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
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(GEMM_M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(GEMM_N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    gemm_i = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)

    batch_idx = gemm_i // (H_OUT * W_OUT)
    offset_out = gemm_i % (H_OUT * W_OUT)
    h_out = offset_out // W_OUT
    w_out = offset_out % W_OUT
    c_out = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for idx_k in range(0, tl.cdiv(GEMM_K, BLOCK_SIZE_K)):
        gemm_k = idx_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        c_in = gemm_k // (F_H * F_W)
        crs_residual = gemm_k % (F_H * F_W)
        h_in = crs_residual // F_W
        w_in = crs_residual % F_W

        h = h_out[:, None] * str_h + h_in[None, :] * dil_h - pad_h
        w = w_out[:, None] * str_w + w_in[None, :] * dil_w - pad_w

        weight_offsets = (
            c_out[None, :] * C_IN * F_H * F_W
            + c_in[:, None] * F_H * F_W
            + h_in[:, None] * F_W
            + w_in[:, None]
        )
        weight_mask = (
            (h_in[:, None] < F_H)
            & (w_in[:, None] < F_W)
            & (c_in[:, None] < C_IN)
            & (c_out[None, :] < C_OUT)
        )

        weight_ptrs = weight_ptr + weight_offsets

        weight_data = tl.load(weight_ptrs, mask=weight_mask, other=0.0)
        channel_ins = tl.load(
            lookup_ptr + c_out[:, None] * C_IN + c_in[None, :]
        )

        input_offsets = (
            batch_idx[:, None] * C_IN * H_IN * W_IN
            + channel_ins * H_IN * W_IN
            + h * W_IN
            + w
        )
        input_ptrs = input_ptr + input_offsets
        input_mask = (
            (h >= 0)
            & (h < H_IN)
            & (w >= 0)
            & (w < W_IN)
            & (c_in[None, :] < C_IN)
            & (batch_idx[:, None] < N)
        )

        input_data = tl.load(input_ptrs, mask=input_mask, other=0.0)

        acc = tl.dot(
            input_data,
            weight_data,
            acc,
            # allow_tf32=False,
        )

    if bias_ptr is not None:
        offs_bias = c_out[None, :]
        bias_ptrs = bias_ptr + offs_bias
        bias_data = tl.load(bias_ptrs)
        acc = acc + bias_data

    acc = acc.to(output_ptr.dtype.element_ty)

    output_offsets = (
        batch_idx[:, None] * C_OUT * H_OUT * W_OUT
        + c_out[None, :] * H_OUT * W_OUT
        + h_out[:, None] * W_OUT
        + w_out[:, None]
    )
    output_mask = (
        (batch_idx[:, None] < N)
        & (c_out[None, :] < C_OUT)
        & (h_out[:, None] < H_OUT)
        & (w_out[:, None] < W_OUT)
    )
    output_ptrs = output_ptr + output_offsets
    tl.store(output_ptrs, acc, mask=output_mask)


def conv2d_forward(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    channel_sparse: torch.Tensor,
    stride,
    padding,
    dilation,
):
    N, C_IN, H_IN, W_IN = input.shape
    C_IN = channel_sparse.shape[1]
    C_OUT, t_C, F_H, F_W = weight.shape
    # assert t_C == C_IN, "Input and weight channels must match"

    str_h, str_w = stride
    pad_h, pad_w = padding
    dil_h, dil_w = dilation

    H_OUT = (H_IN + 2 * pad_h - dil_h * (F_H - 1) - 1) // str_h + 1
    W_OUT = (W_IN + 2 * pad_w - dil_w * (F_W - 1) - 1) // str_w + 1

    GEMM_M = N * H_OUT * W_OUT
    GEMM_K = C_IN * F_H * F_W
    GEMM_N = C_OUT
    output = torch.zeros(
        (N, C_OUT, H_OUT, W_OUT), dtype=input.dtype, device=input.device
    )

    channel_sparse_conv2d_forward_kernel[
        lambda META: (
            triton.cdiv(GEMM_M, META["BLOCK_SIZE_M"])
            * triton.cdiv(GEMM_N, META["BLOCK_SIZE_N"]),
        )
    ](
        output,
        input,
        weight,
        channel_sparse,
        bias,
        N,
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

    # # 1. store to output [N*P*Q, K] [GEMM_M, GEMM_N]
    # output = output.view(N, P, Q, K).permute(0, 3, 1, 2).contiguous() # [N*P*Q, K] -> [N, K, P, Q]

    # # 2. store to output [N, P, Q, K]
    # output = output.permute(0, 3, 1, 2).contiguous() # [N, P, Q, K] -> [N, K, P, Q]

    # 3. store to output [N, K, P, Q]
    return output


if __name__ == "__main__":
    DEVICE = triton.runtime.driver.active.get_active_torch_device()
    DTYPE = torch.float32
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)
    print("Backend options:")
    print(f"\tcudnn: {torch.backends.cudnn.enabled}")
    print(f"\tcudnn deterministic: {torch.backends.cudnn.deterministic}")
    print(f"\tcudnn allow tf32: {torch.backends.cudnn.allow_tf32}")
    print(f"\tcuda matmul allow tf32: {torch.backends.cuda.matmul.allow_tf32}")

    N = 4
    C, H, W = 16, 128, 128
    C_OUT = 64
    K_H = 5
    K_W = 7
    stride = 2
    padding = 2
    dilation = 1
    stride = (stride, stride)
    padding = (padding, padding)
    dilation = (dilation, dilation)

    input = torch.randn(N, C, H, W, device=DEVICE, dtype=DTYPE)
    weight = torch.randn(C_OUT, C, K_H, K_W, device=DEVICE, dtype=DTYPE)
    bias = torch.randn(C_OUT, device=DEVICE, dtype=DTYPE)

    c_sparse = []
    for _ in range(C_OUT):
        c_sparse.append(np.sort(np.random.choice(C, size=8, replace=False)))
    c_sparse = torch.tensor(c_sparse, device=DEVICE, dtype=torch.int32)
    print(c_sparse, c_sparse.shape)

    zeroed_weights = weight.clone()
    for i in range(C_OUT):
        zeroed_weights[:, c_sparse[i], :, :] = 0.0

    torch_out = torch.nn.functional.conv2d(
        input,
        zeroed_weights,
        bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )

    triton_out = conv2d_forward(
        input,
        weight,
        bias,
        c_sparse,
        stride,
        padding,
        dilation,
    )
    print(f"torch_out: {torch_out.shape}, triton_out: {triton_out.shape}")
    print(f"torch_out: {torch_out.dtype}, triton_out: {triton_out.dtype}")
    print(f"torch_out: {torch_out.device}, triton_out: {triton_out.device}")
    print(f"torch_out: {torch_out}, triton_out: {triton_out}")
    print(
        f"torch_out == triton_out: {torch.allclose(torch_out, triton_out, atol=1e-5)}"
    )
