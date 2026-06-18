import torch
import triton
import triton.language as tl
import random

ALLOW_TF32 = True


@triton.jit
def conv_forward_kernel(
    # Pointers to matrices
    x_ptr,
    w_ptr,
    b_ptr,
    y_ptr,
    # Matrix dimensions
    N,
    C,
    H,
    W,
    K,
    R,
    S,
    OH,
    OW,
    # Strides
    stride_x_n,
    stride_x_c,
    stride_x_h,
    stride_x_w,
    stride_w_k,
    stride_w_c,
    stride_w_r,
    stride_w_s,
    stride_y_n,
    stride_y_k,
    stride_y_oh,
    stride_y_ow,
    # Convolution parameters
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    dilation_h,
    dilation_w,
    # Meta-parameters
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_OH: tl.constexpr,
    BLOCK_SIZE_OW: tl.constexpr,
    GROUP_SIZE_N: tl.constexpr,
):
    """
    Triton kernel for the forward pass of a 2D convolution.
    """
    # Program IDs
    pid_n = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)
    pid_ohow = tl.program_id(axis=2)

    # Grouping for L2 cache efficiency
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    num_pid_ohow = tl.cdiv(OH * OW, BLOCK_SIZE_OH * BLOCK_SIZE_OW)

    group_id = pid_n // (num_pid_n // GROUP_SIZE_N)
    first_pid_n = group_id * (num_pid_n // GROUP_SIZE_N)
    group_size_n = num_pid_n - first_pid_n

    pid_n = first_pid_n + (pid_n % group_size_n)

    # OH and OW block computation
    pid_oh = pid_ohow // (OW // BLOCK_SIZE_OW)
    pid_ow = pid_ohow % (OW // BLOCK_SIZE_OW)

    # Accumulator for the output block
    acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_K), dtype=tl.float32)

    # Pointers to the weight and bias blocks
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_oh = pid_oh * BLOCK_SIZE_OH + tl.arange(0, BLOCK_SIZE_OH)
    offs_ow = pid_ow * BLOCK_SIZE_OW + tl.arange(0, BLOCK_SIZE_OW)

    # Bias pointer
    b_ptrs = b_ptr + offs_k

    # Loop over the input channels and kernel dimensions
    for c in range(0, C):
        for r in range(0, R):
            for s in range(0, S):
                # Load the weights
                w_ptrs = w_ptr + (
                    offs_k[:, None] * stride_w_k
                    + c * stride_w_c
                    + r * stride_w_r
                    + s * stride_w_s
                )
                weights = tl.load(w_ptrs)

                # Compute the input coordinates
                x_h = offs_oh[:, None] * stride_h + r * dilation_h - padding_h
                x_w = offs_ow[None, :] * stride_w + s * dilation_w - padding_w

                # Boundary checks
                mask_h = (x_h >= 0) & (x_h < H)
                mask_w = (x_w >= 0) & (x_w < W)
                mask_x = mask_h & mask_w

                # Load the input data
                x_ptrs = x_ptr + (
                    offs_n[None, :] * stride_x_n
                    + c * stride_x_c
                    + x_h * stride_x_h
                    + x_w * stride_x_w
                )

                x = tl.load(x_ptrs, mask=mask_x, other=0.0)

                # Matrix multiplication
                acc += tl.dot(x, weights)

    # Add the bias
    bias = tl.load(b_ptrs, mask=offs_k < K)
    acc += bias[None, :]

    # Store the result
    y_ptrs = y_ptr + (
        offs_n[:, None] * stride_y_n
        + offs_k[None, :] * stride_y_k
        + offs_oh[:, None] * stride_y_oh
        + offs_ow[None, :] * stride_y_ow
    )
    tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty))


def conv_forward(x, w, b, stride, padding, dilation):
    """
    Wrapper function for the forward pass of a 2D convolution.
    """
    N, C, H, W = x.shape
    K, _, R, S = w.shape
    stride_h, stride_w = stride
    padding_h, padding_w = padding
    dilation_h, dilation_w = dilation

    OH = (H + 2 * padding_h - dilation_h * (R - 1) - 1) // stride_h + 1
    OW = (W + 2 * padding_w - dilation_w * (S - 1) - 1) // stride_w + 1

    y = torch.empty((N, K, OH, OW), dtype=x.dtype, device=x.device)

    grid = lambda META: (
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
        triton.cdiv(K, META["BLOCK_SIZE_K"]),
        triton.cdiv(OH * OW, META["BLOCK_SIZE_OH"] * META["BLOCK_SIZE_OW"]),
    )

    conv_forward_kernel[grid](
        x,
        w,
        b,
        y,
        N,
        C,
        H,
        W,
        K,
        R,
        S,
        OH,
        OW,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        w.stride(0),
        w.stride(1),
        w.stride(2),
        w.stride(3),
        y.stride(0),
        y.stride(1),
        y.stride(2),
        y.stride(3),
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
        BLOCK_SIZE_N=16,
        BLOCK_SIZE_K=32,
        BLOCK_SIZE_C=4,
        BLOCK_SIZE_OH=2,
        BLOCK_SIZE_OW=2,
        GROUP_SIZE_N=8,
    )
    return y

