import triton
import triton.conv2d_kernels as conv2d_kernels
import torch
import numpy as np
import torch.nn.grad as G
import configs

configs.disable_torch_optimizations()

BATCH_SIZE = 16
C_IN = 3
H_IN = 224
W_IN = 224
C_OUT = 64
FILTER_H = 3
FILTER_W = 3
stride_h = 2
stride_w = 2
pad_h = 0
pad_w = 0
dil_h = 1
dil_w = 1

H_OUT = (H_IN + 2 * pad_h - FILTER_H) // stride_h + 1
W_OUT = (W_IN + 2 * pad_w - FILTER_W) // stride_w + 1
torch.manual_seed(0)
# inputs = torch.randn(1, 3, 4, 4, device="cuda")
inputs = torch.randn(BATCH_SIZE, C_IN, H_IN, W_IN, device="cuda")
# weights = torch.randn(2, 3, 1, 1, device="cuda")
weights = torch.randn(C_OUT, C_IN, FILTER_H, FILTER_W, device="cuda")
# grad_output = torch.randn(1, 2, 4, 4, device="cuda")
grad_output = torch.randn(BATCH_SIZE, C_OUT, H_OUT, W_OUT, device="cuda")
# print(f"inputs: {inputs} shape: {inputs.shape}")
# print(f"weights: {weights} shape: {weights.shape}")
# print(f"grad_output: {grad_output} shape: {grad_output.shape}")
# grad_weights = tl.zeros(weights.shape, dtype=weights.dtype)
grad_weight = torch.zeros(weights.shape, device="cuda", dtype=torch.float32)
GEMM_M = C_OUT
GEMM_N = C_IN * FILTER_H * FILTER_W
GEMM_K = H_OUT * W_OUT
BLOCK_SIZE_M = min(max(2 ** int(np.ceil(np.log2(GEMM_M))), 16), 16)
BLOCK_SIZE_N = min(max(2 ** int(np.ceil(np.log2(GEMM_N))), 16), 16)
BLOCK_SIZE_K = min(max(2 ** int(np.ceil(np.log2(GEMM_K))), 16), 64)
# print(f"BLOCK_SIZE_M: {BLOCK_SIZE_M}, BLOCK_SIZE_N: {BLOCK_SIZE_N}, BLOCK_SIZE_K: {BLOCK_SIZE_K}")
GROUP_SIZE_M = 1


def grid(META):
    return (
        BATCH_SIZE,
        triton.cdiv(GEMM_M, META["BLOCK_SIZE_M"])
        * triton.cdiv(GEMM_N, META["BLOCK_SIZE_N"]),
    )


conv2d_kernels.conv2d_grad_weight_kernel_atomic[grid](
    inputs,
    grad_output,
    grad_weight,
    BATCH_SIZE=BATCH_SIZE,
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
    GEMM_M=GEMM_M,
    GEMM_N=GEMM_N,
    GEMM_K=GEMM_K,
    BLOCK_SIZE_M=BLOCK_SIZE_M,
    BLOCK_SIZE_N=BLOCK_SIZE_N,
    BLOCK_SIZE_K=BLOCK_SIZE_K,
    GROUP_SIZE_M=4, # type: ignore
)

torch_grad = G.conv2d_weight(
    inputs,
    weights.shape,
    grad_output,
    stride=(stride_h, stride_w), # type: ignore
    padding=(pad_h, pad_w), # type: ignore
    dilation=(dil_h, dil_w), # type: ignore
)
print(f"Max difference: {torch.max(torch_grad - grad_weight)}")
print(f"Norm difference: {torch.norm(torch_grad - grad_weight)}")
print(f"Relative error: {torch.norm(torch_grad - grad_weight) / torch.norm(torch_grad)}")
