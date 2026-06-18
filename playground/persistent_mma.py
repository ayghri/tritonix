import torch
import triton
import triton.language as tl


# Helper function to get the number of SMs on the current GPU
def get_sm_count():
    device = torch.cuda.current_device()
    return torch.cuda.get_device_properties(device).multi_processor_count


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64},
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=4,
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def persistent_gemm_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    # Strides
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    # The number of persistent programs launched
    SM_COUNT: tl.constexpr,
):
    """
    Computes C = A @ B using a persistent kernel design.
    The grid is 1D with size SM_COUNT. Each program is a worker.
    """
    # Each program is a worker with a unique ID
    pid = tl.program_id(axis=0)

    # Calculate the total number of output blocks to compute
    grid_m = tl.cdiv(M, BLOCK_SIZE_M)
    grid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_blocks = grid_m * grid_n

    # --- Persistent Loop ---
    # Each program loops through the total blocks, grabbing the next available one.
    # The `range` function with a step size of `SM_COUNT` handles the work distribution.
    for block_id in range(pid, total_blocks, SM_COUNT):
        # Decompose the 1D block_id into 2D block coordinates (pid_m, pid_n)
        pid_m = block_id // grid_n
        pid_n = block_id % grid_n

        # ----------------------------------------------------------------
        # Standard GEMM logic for a single output block starts here
        # ----------------------------------------------------------------

        # Initialize accumulator for the current block
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        # Pointers to the first tiles of A and B for this specific block
        offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        offs_k = tl.arange(0, BLOCK_SIZE_K)

        a_ptrs = a_ptr + (
            offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
        )
        b_ptrs = b_ptr + (
            offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
        )

        # Loop over the K dimension, accumulating into the accumulator
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            k_remaining = K - k * BLOCK_SIZE_K

            mask_a = (offs_am[:, None] < M) & (offs_k[None, :] < k_remaining)
            mask_b = (offs_k[:, None] < k_remaining) & (offs_bn[None, :] < N)

            a = tl.load(a_ptrs, mask=mask_a, other=0.0)
            b = tl.load(b_ptrs, mask=mask_b, other=0.0)

            accumulator += tl.dot(a, b)

            # Advance pointers to the next K-block
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk

        # Cast the accumulator to the output dtype
        c_block = accumulator.to(c_ptr.dtype.element_ty)

        # Pointers to the output block in C
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + (
            offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
        )

        # Create a mask for storing the output block
        mask_c = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c_block, mask=mask_c)

        # End of standard GEMM logic for one block
        # The loop will now continue to the next `block_id` for this worker


def persistent_gemm(A, B):
    # Shape checks
    assert A.shape[1] == B.shape[0], "Incompatible dimensions"
    assert A.is_contiguous() and B.is_contiguous(), (
        "Matrices must be contiguous"
    )
    M, K = A.shape
    K, N = B.shape

    # Allocate output tensor
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)

    # Launch a 1D grid of SM_COUNT persistent workers
    SM_COUNT = get_sm_count()
    grid = (SM_COUNT,)

    persistent_gemm_kernel[grid](
        A,
        B,
        C,
        M,
        N,
        K,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        C.stride(0),
        C.stride(1),
        SM_COUNT=SM_COUNT,
    )
    return C


if __name__ == "__main__":
    M, N, K = 1024, 2048, 2048

    A = torch.randn((M, K), device="cuda", dtype=torch.float16)
    B = torch.randn((K, N), device="cuda", dtype=torch.float16)

    # PyTorch reference implementations
    torch_output = A @ B

    # @torch.compile(mode="max-autotune", verbose=False)
    @torch.compile()
    def compiled_torch_gemm(a, b):
        return a @ b

    # Run and verify
    triton_output = persistent_gemm(A, B)

    print(f"triton_output[:3, :3]=\n{triton_output[:3, :3]}")
    print(f"torch_output[:3, :3]=\n{torch_output[:3, :3]}")

    if torch.allclose(triton_output, torch_output, atol=0.1, rtol=0.01):
        print("✅ Triton and Torch match")
    else:
        print("❌ Triton and Torch differ")
        diff = torch.max(torch.abs(triton_output - torch_output))
        print(f"   Max absolute difference: {diff.item()}")


    # Benchmark
    print("\nBenchmarking...")
    # Warmup
    _ = persistent_gemm(A, B)
    _ = compiled_torch_gemm(A, B)
    torch.cuda.synchronize()

    triton_ms = triton.testing.do_bench(lambda: persistent_gemm(A, B))
    compiled_torch_ms = triton.testing.do_bench(
        lambda: compiled_torch_gemm(A, B)
    )

    # FLOPs calculation for GEMM: 2 * M * N * K
    total_flops = 2 * M * N * K
    triton_tflops = total_flops / (triton_ms * 1e-3) / 1e12
    torch_tflops = total_flops / (compiled_torch_ms * 1e-3) / 1e12

    print(
        f"Triton (Persistent): {triton_ms:.4f} ms  | {triton_tflops:.2f} TFLOP/s"
    )
    print(
        f"Torch.compile:       {compiled_torch_ms:.4f} ms  | {torch_tflops:.2f} TFLOP/s"
    )
