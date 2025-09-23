#!/usr/bin/env python3
import torch
import triton
import triton.language as tl


# We use the autotuner to find the best configuration for the given problem size.
@triton.autotune(
    configs=[
        # Basic configurations
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 32,
                "BLOCK_SIZE_P": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 32,
                "BLOCK_SIZE_P": 64,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 32,
                "BLOCK_SIZE_P": 64,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        # Configurations with more stages and warps for larger problems
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 64,
                "BLOCK_SIZE_P": 64,
                "GROUP_SIZE_M": 8,
            },
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 32,
                "BLOCK_SIZE_P": 128,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
    ],
    # The key determines when to re-run the autotuning.
    key=["M", "K", "N", "P"],
    # We need to reset the output tensor to zero for each run.
    reset_to_zero=["d_ptr"],
)
@triton.jit
def fused_mlp_atomic_add_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    d_ptr,
    # Matrix dimensions
    M,
    K,
    N,
    P,
    # Strides
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cn,
    stride_cp,
    stride_dm,
    stride_dp,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_P: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    """
    Computes D = relu(A @ B) @ C using a fused kernel with atomic adds.
    - A: (M, K), B: (K, N), C: (N, P), D: (M, P)
    """
    # -----------------------------------------------------------
    # Map program ids to the intermediate matrix's (Z) blocks.
    # This is the same grouped ordering strategy for L2 cache reuse.
    # -----------------------------------------------------------
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # =================================================================
    # PHASE 1: Compute a tile of Z' = relu(A @ B)
    # The result is stored in `accumulator` (registers).
    # =================================================================

    # Pointers to the first blocks of A and B
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (
        offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    )
    b_ptrs = b_ptr + (
        offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    )

    # Accumulator for the Z' tile
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load tiles of A and B, masking for out-of-bounds elements
        k_remaining = K - k * BLOCK_SIZE_K
        mask_a = (offs_am[:, None] < M) & (offs_k[None, :] < k_remaining)
        mask_b = (offs_k[:, None] < k_remaining) & (offs_bn[None, :] < N)

        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)

        accumulator += tl.dot(a, b)

        # Advance pointers to the next K-block
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Apply activation function to the intermediate result


    # if ACTIVATION == "relu":
    

    # =================================================================
    # PHASE 2: Compute D += Z' @ C using the accumulator from Phase 1.
    # We loop over the P dimension of C and D.
    # =================================================================

    # `accumulator` is (BLOCK_M, BLOCK_N)
    # We need to multiply it by C, which is (N, P).
    # We will load tiles of C of size (BLOCK_N, BLOCK_P).

    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_dm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    acc = tl.cast(accumulator, tl.float16)
    acc = tl.maximum(acc, 0.0)

    for p_start in range(0, tl.cdiv(P, BLOCK_SIZE_P)):
        p_offset = p_start * BLOCK_SIZE_P
        offs_p = p_offset + tl.arange(0, BLOCK_SIZE_P)

        # Pointers to tiles in C and D
        c_ptrs = c_ptr + (
            offs_cn[:, None] * stride_cn + offs_p[None, :] * stride_cp
        )
        d_ptrs = d_ptr + (
            offs_dm[:, None] * stride_dm + offs_p[None, :] * stride_dp
        )

        # Load a tile of C
        mask_c = (offs_cn[:, None] < N) & (offs_p[None, :] < P)
        c = tl.load(c_ptrs, mask=mask_c, other=0.0)
        c = tl.cast(c, tl.float32)

        # Compute the partial result for D: Z'[tile] @ C[tile]
        # (BLOCK_M, BLOCK_N) @ (BLOCK_N, BLOCK_P) -> (BLOCK_M, BLOCK_P)
        d_partial = tl.dot(acc, c)

        # Atomically add the partial result to the output matrix D
        mask_d = (offs_dm[:, None] < M) & (offs_p[None, :] < P)
        tl.atomic_add(d_ptrs, d_partial, mask=mask_d)


def fused_mlp(A, B, C):
    # Shape checks
    assert A.shape[1] == B.shape[0], "Incompatible dimensions for A @ B"
    assert B.shape[1] == C.shape[0], "Incompatible dimensions for (A@B) @ C"
    assert A.is_contiguous(), "Matrix A must be contiguous"
    assert B.is_contiguous(), "Matrix B must be contiguous"
    assert C.is_contiguous(), "Matrix C must be contiguous"

    M, K = A.shape
    _K, N = B.shape
    _N, P = C.shape

    # Allocate output tensor, initialized to zeros for atomic add.
    D = torch.zeros((M, P), device=A.device, dtype=A.dtype)

    # The grid is 1D and tiles the intermediate (M, N) matrix.
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"])
        * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    fused_mlp_atomic_add_kernel[grid](
        A,
        B,
        C,
        D,
        M,
        K,
        N,
        P,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        C.stride(0),
        C.stride(1),
        D.stride(0),
        D.stride(1),
        ACTIVATION="relu",
    )
    return D


# --- Main execution block for testing ---
if __name__ == "__main__":
    # Use larger, more realistic dimensions for benchmarking
    M, K, N, P = 512, 1024, 4096, 1024

    A = torch.randn((M, K), device="cuda", dtype=torch.float16)
    B = torch.randn((K, N), device="cuda", dtype=torch.float16)
    C = torch.randn((N, P), device="cuda", dtype=torch.float16)

    # PyTorch reference calculation
    torch_output = torch.nn.functional.relu(A @ B) @ C

    # Our fused Triton kernel calculation
    triton_output = fused_mlp(A, B, C)

    # Print a few values to visually inspect
    print(f"triton_output[:3, :3]=\n{triton_output[:3, :3]}")
    print(f"torch_output[:3, :3]=\n{torch_output[:3, :3]}")

    # Verification
    if torch.allclose(triton_output, torch_output, atol=1e-1, rtol=0.01):
        print("✅ Triton and Torch match")
    else:
        print("❌ Triton and Torch differ")
        diff = torch.max(torch.abs(triton_output - torch_output))
        print(f"   Max absolute difference: {diff.item()}")

    # Benchmark against the compiled torch version
    @torch.compile(mode="max-autotune")
    def compiled_torch_mlp(a, b, c):
        return torch.nn.functional.relu(a @ b) @ c

    print("\nBenchmarking...")
    triton_ms = triton.testing.do_bench(lambda: fused_mlp(A, B, C))
    compiled_torch_ms = triton.testing.do_bench(
        lambda: compiled_torch_mlp(A, B, C)
    )

    # FLOPs calculation: 2*M*K*N (GEMM1) + 2*M*N*P (GEMM2)
    total_flops = 2 * M * K * N + 2 * M * N * P
    triton_tflops = total_flops / (triton_ms * 1e-3) / 1e12
    torch_tflops = total_flops / (compiled_torch_ms * 1e-3) / 1e12

    print(
        f"Triton (Atomic Add): {triton_ms:.4f} ms  | {triton_tflops:.2f} TFLOP/s"
    )
    print(
        f"Torch.compile:       {compiled_torch_ms:.4f} ms  | {torch_tflops:.2f} TFLOP/s"
    )
