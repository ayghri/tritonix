import torch
import triton
import triton.language as tl


@triton.jit
def blockify_kernel(
    a_ptr,
    b_ptr,
    m,
    n,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    group_m: tl.constexpr,
    group_n: tl.constexpr,
):
    """
    Reshape a 2D tensor into blocks.
    Each program instance processes an entire row of blocks.

    :param a_ptr: Pointer to the input tensor.
    :param b_ptr: Pointer to the output tensor.
    :param M: Number of rows in the input tensor.
    :param N: Number of columns in the input tensor.
    :param BLOCK_M: Block size for the M dimension.
    :param BLOCK_N: Block size for the N dimension.
    :param NUM_BLOCKS_N: The number of blocks along the N dimension (N // BLOCK_N).
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Calculate the starting row for all blocks this program will process.
    # This row offset is constant for this entire program instance.
    start_m = pid_m * block_m * group_m
    start_n = pid_n * block_n * group_n
    offs_m = start_m + tl.arange(0, block_m * group_m)
    offs_n = start_n + tl.arange(0, block_n * group_n)

    # block_start_n = block_col_idx * block_n
    # offs_n = block_start_n + tl.arange(0, block_n)

    # Create pointers to the current block in the input tensor.
    a_block_ptrs = a_ptr + offs_m[:, None] * n + offs_n[None, :]
    block = tl.load(
        a_block_ptrs, mask=(offs_m[:, None] < m) & (offs_n[None, :] < n)
    )
    block = tl.reshape(block, (group_m, block_m, group_n, block_n))
    block = tl.trans(block, (0, 2, 1, 3)).reshape(
        group_m, group_n * block_m * block_n
    )

    output_block_idx = pid_m * group_m + pid_n
    b_output_ptr = (
        b_ptr
        + output_block_idx * (block_m * block_n)
        + tl.arange(0, block_m * block_n)
    )

    # Reshape the loaded block to be 1D and store it.
    tl.store(b_output_ptr, block)


@triton.jit
def reshape_to_blocks_kernel(
    a_ptr,
    b_ptr,
    m,
    n,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    group_n: tl.constexpr,
    group_m: tl.constexpr,
):
    """
    Optimized Triton kernel to reshape a 2D tensor into blocks.
    Each program instance processes an entire row of blocks.

    :param a_ptr: Pointer to the input tensor.
    :param b_ptr: Pointer to the output tensor.
    :param M: Number of rows in the input tensor.
    :param N: Number of columns in the input tensor.
    :param BLOCK_M: Block size for the M dimension.
    :param BLOCK_N: Block size for the N dimension.
    :param NUM_BLOCKS_N: The number of blocks along the N dimension (N // BLOCK_N).
    """
    block_row_idx = tl.program_id(axis=0)
    block_col_idx = tl.program_id(axis=0)
    block_start_m = block_row_idx * block_m * group_m
    offs_m = block_start_m + tl.arange(0, block_m * group_m)
    offs_n = block_col_idx + tl.arange(0, block_n * group_n)

    a_block_ptr = a_ptr + offs_m[:, None] * n + offs_n[None, :]

    block = tl.load(a_block_ptr)
    block = tl.reshape(block, (group_m, block_m, group_n, block_n))
    block = tl.trans(block, (0, 2, 1, 3))
    block = tl.reshape(block, (group_m, group_n * block_m * block_n))

    output_block_idx = block_row_idx * group_n + block_col_idx
    b_output_ptr = (
        b_ptr
        + output_block_idx * (block_m * block_n)
        + tl.arange(0, block_m * block_n * group_n)
    )
    # Reshape the loaded block to be 1D and store it.
    tl.store(b_output_ptr, block)


# a = torch.tensor([[1,   2,  3,  4],
#                   [5,   6,  7,  8],
#                   [9,  10, 11, 12],
#                   [13, 14, 15, 16]], dtype=torch.float32, device='cuda')
M = 8
N = 6
block_m = 2
block_n = 3
group_n = 2
group_m = 1
a = torch.arange(M * N, dtype=torch.float32).reshape(M, N).cuda()
# Block dimensions


# Create the output tensor on the GPU.
b = torch.empty(
    ((M // block_m) * (N // block_n), block_m, block_n),
    dtype=torch.float32,
    device="cuda",
)

# The grid for launching the kernel is now much smaller.
# We launch one program for each ROW of blocks.
# grid = (M // block_m, N /)
grid = (triton.cdiv(M, block_m), triton.cdiv(N, group_n))

# Launch the optimized kernel.
# reshape_to_blocks_kernel[grid](
#     a,
#     b,
#     M,
#     N,
#     block_m=block_m,
#     block_n=block_n,
#     group_n=group_n,  # Pass as a constexpr
# )
# print(a)

# Reshape the output to the final desired 3D format for verification.
# b = b.reshape(-1, block_m, block_n)

# print("Optimized Triton output:\n", b)

print(a)
# --- PyTorch implementation for verification ---
a_torch = (
    a.view(M // block_m, block_m, N // block_n, block_n)
    .permute(0, 2, 1, 3)
    .reshape(-1, block_m, block_n)
)
# print("\nPyTorch output:\n", a_torch)

# Verify the results are the same.
# assert torch.allclose(b, a_torch)
print("\nOutputs are the same.")


a = torch.arange(8 * 6).reshape(8, 6)
a
# a.reshape(2,4,6)
# a.reshape(2,4,3,2)
# b = a.reshape(8,2,3)
# a
# b[...,0,:], b[...,1,:]

# b = a.reshape(2,4,3,2)
# a
# b[...,0,:], b[...,1,:]
# a
# b.transpose(3,2)
# a
a
a.view(4, 2, 2, 3).permute(0, 2, 1, 3).reshape(8, 2, 3)
# a.view(2, 4, 2, 3).permute(2, 0, 1, 3).reshape(8, 2, 3)
# a.view(2, 4, 2, 3).permute(2, 0, 1, 3).reshape(4, 2,2*3)


# a.view(2, 4, 2, 3).permute(2, 0, 1, 3)
m = 1024
k = 1024 * 8
n = 1024 * 2



blocks = [8, 16, 32]


def blockify(a, block_m, block_n):
    return (
        a.view(m // block_m, block_m, k // block_n, block_n)
        .permute(0, 2, 1, 3)
        .reshape(-1, block_m, block_n)
        .contiguous()
    )


a = torch.randn((m, k), device="cuda", dtype=torch.float32).cuda()
b = torch.randn((k,n), device="cuda", dtype=torch.float32).cuda()
quantiles = [0.5, 0.2]
compiled_blockify = torch.compile(blockify, fullgraph=True)
for block in blocks:
    ms, min_ms = triton.testing.do_bench(
        lambda: blockify(a, block, block),
        quantiles=quantiles,
        warmup=100,
        rep=200,
    ) # type: ignore[no-untyped-call]
    print(f"Block size {block} took {ms:.2f} ms, min {min_ms:.2f} ms")

    ms, min_ms = triton.testing.do_bench(
        lambda: compiled_blockify(a, block, block),
        quantiles=quantiles,
        warmup=100,
        rep=200,
    ) # type: ignore[no-untyped-call]
    print(f"Compiled Block size {block} took {ms:.2f} ms, min {min_ms:.2f} ms")


ms, min_ms = triton.testing.do_bench(
    lambda: torch.matmul(a,b),
    quantiles=quantiles,
    warmup=100,
    rep=200,
) # type: ignore[no-untyped-call]
print(f"Compiled Block size {block} took {ms:.2f} ms, min {min_ms:.2f} ms")