import torch
import triton
import triton.language as tl

DTYPE = torch.float16
# enable TF32 for better performance on Ampere and later GPUs
torch.set_float32_matmul_precision("high")

def prepare_A_dense(m, k, device="cuda", dtype=DTYPE):
    """Creates a dense matrix A."""
    return torch.randn((m, k), device=device, dtype=dtype)


def prepare_A_tiled(dense_a, block_m, block_k):
    """Transforms a dense matrix A into the specified tiled format."""
    m, k = dense_a.shape
    num_blocks_m = m // block_m
    num_blocks_k = k // block_k

    tiled_a = dense_a.reshape(num_blocks_m, block_m, num_blocks_k, block_k)
    tiled_a = tiled_a.permute(0, 2, 1, 3)
    contiguous_a = tiled_a.contiguous().reshape(
        num_blocks_m * num_blocks_k, block_m, block_k
    )

    return contiguous_a


def prepare_B_dense(k, n, device="cuda", dtype=DTYPE):
    """Creates a dense matrix B."""
    return torch.randn((k, n), device=device, dtype=dtype)


def prepare_B_block_sparse(dense_b, block_k, block_n, p, random_seed=0):
    """Transforms a dense matrix B into the specified block-sparse format."""
    k, n = dense_b.shape
    num_blocks_k = k // block_k
    num_blocks_n = n // block_n

    b_values_list = []
    b_metadata = torch.zeros(
        (num_blocks_n, p), dtype=torch.int32, device=dense_b.device
    )

    # For demonstration, we'll create a reproducible sparse pattern
    # In a real scenario, this pattern would be predetermined
    for j in range(num_blocks_n):
        # Create  a random selection of non-zero block indices
        torch.manual_seed(random_seed + j)  # Ensure reproducibility
        non_zero_block_indices = torch.randperm(num_blocks_k)[:p]
        b_metadata[j] = non_zero_block_indices.sort().values

        # non_zero_block_indices = torch.random.r(p, device=dense_b.device) + (
        #     j % (num_blocks_k - p)
        # )
        # b_metadata[j] = non_zero_block_indices

        for i in non_zero_block_indices:
            block = dense_b[
                i * block_k : (i + 1) * block_k, j * block_n : (j + 1) * block_n
            ]
            # Store in column-major to match kernel expectation
            b_values_list.append(block.T.contiguous())

    b_values = torch.stack(b_values_list).reshape(
        num_blocks_n * p, block_n, block_k
    )
    return b_values, b_metadata


@triton.jit
def block_mma_kernel(
    a_ptr,
    b_values_ptr,
    b_metadata_ptr,
    c_ptr,
    m,
    n,
    k,
    stride_a_tile,
    stride_a_m,
    stride_a_k,
    stride_b_values_tile,
    stride_b_values_n,
    stride_b_values_k,
    stride_b_metadata_n,
    stride_b_metadata_p,
    stride_c_m,
    stride_c_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    P: tl.constexpr,
    NUM_BLOCKS_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Pointer to the output block
    c_block_ptr = tl.make_block_ptr(
        base=c_ptr,
        shape=(m, n),
        strides=(stride_c_m, stride_c_n),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Pointer to the metadata for the current block column of B
    b_metadata_col_ptr = b_metadata_ptr + pid_n * stride_b_metadata_n

    for i in range(P):
        # Load the row block index for the current non-zero block of B
        b_block_row_idx = tl.load(b_metadata_col_ptr + i * stride_b_metadata_p)

        # Pointer to the corresponding tile in A
        a_tile_ptr = (
            a_ptr + (pid_m * NUM_BLOCKS_K + b_block_row_idx) * stride_a_tile
        )

        a_block_ptr = tl.make_block_ptr(
            base=a_tile_ptr,
            shape=(BLOCK_M, BLOCK_K),
            strides=(stride_a_m, stride_a_k),
            offsets=(0, 0),
            block_shape=(BLOCK_M, BLOCK_K),
            order=(1, 0),
        )
        a = tl.load(a_block_ptr)

        # Pointer to the non-zero block in B
        b_block_idx = pid_n * P + i
        b_tile_ptr = b_values_ptr + b_block_idx * stride_b_values_tile

        b_block_ptr = tl.make_block_ptr(
            base=b_tile_ptr,
            shape=(
                BLOCK_N,
                BLOCK_K,
            ),  # Shape is (N, K) due to column-major block
            strides=(stride_b_values_n, stride_b_values_k),
            offsets=(0, 0),
            block_shape=(BLOCK_N, BLOCK_K),
            order=(1, 0),
        )
        b = tl.load(b_block_ptr)

        # Transpose B to (BLOCK_K, BLOCK_N) for the dot product
        b = tl.trans(b)

        acc += tl.dot(a, b)

    tl.store(c_block_ptr, acc.to(c_ptr.dtype.element_ty))


@triton.jit
def block_mma_kernel_vectorized(
    a_ptr,
    b_values_ptr,
    b_metadata_ptr,
    c_ptr,
    m,
    n,
    k,
    p,
    size_m,
    size_n,
    size_k,
    stride_am,
    stride_ak,
    stride_c_m,
    stride_c_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_P: tl.constexpr,
    NUM_BLOCKS_K: tl.constexpr,
    
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Accumulator for the output block remains the same.
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    c_block_ptr = tl.make_block_ptr(
        base=c_ptr,
        shape=(m, n),
        strides=(stride_c_m, stride_c_n),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )

    # Pointers to the metadata column for this program instance.
    b_metadata_col_ptr = b_metadata_ptr + pid_n * stride_b_metadata_n

    # Outer loop processes P in chunks of size BLOCK_P.
    for p_group_start in range(0, P, BLOCK_P):
        # --- This is the "vectorized" section ---
        # The following 'for' loop is a compile-time construct.
        # Triton unrolls it, creating BLOCK_P parallel instruction streams.
        # This is the idiomatic way to implement a gather-like operation.

        for p_offset in range(BLOCK_P):
            current_p_idx = p_group_start + p_offset

            # Static guard ensures we don't go past P. This adds no runtime cost.
            if current_p_idx < P:
                # 1. Gather Load: Load one of the scattered A blocks.
                # The compiler will schedule BLOCK_P of these loads in parallel.
                b_block_row_idx = tl.load(
                    b_metadata_col_ptr + current_p_idx * stride_b_metadata_p
                )
                a_tile_ptr = (
                    a_ptr
                    + (pid_m * NUM_BLOCKS_K + b_block_row_idx) * stride_a_tile
                )
                a_block_ptr = tl.make_block_ptr(
                    base=a_tile_ptr,
                    shape=(BLOCK_M, BLOCK_K),
                    strides=(stride_am, stride_ak),
                    offsets=(0, 0),
                    block_shape=(BLOCK_M, BLOCK_K),
                    order=(1, 0),
                )
                a = tl.load(a_block_ptr)

                # 2. Strided Load: Load one of the B blocks.
                # These are contiguous in memory within the chunk, so this is efficient.
                b_block_idx = pid_n * P + current_p_idx
                b_tile_ptr = b_values_ptr + b_block_idx * stride_b_values_tile
                b_block_ptr = tl.make_block_ptr(
                    base=b_tile_ptr,
                    shape=(BLOCK_N, BLOCK_K),
                    strides=(stride_b_values_n, stride_b_values_k),
                    offsets=(0, 0),
                    block_shape=(BLOCK_N, BLOCK_K),
                    order=(1, 0),
                )
                b = tl.load(b_block_ptr)
                b = tl.trans(b)

                # 3. Compute: Perform the dot product.
                # The GPU will overlap these computations with the loads from other "lanes" of the unrolled loop.
                acc += tl.dot(a, b)

    tl.store(c_block_ptr, acc.to(c_ptr.dtype.element_ty))


def benchmark():
    # Matrix dimensions
    m, k, n = 1024, 1024 * 8, 1024 * 2
    b_s = 32
    block_m, block_k, block_n = b_s, b_s, b_s
    # Number of non-zero blocks per block-column in B
    # This represents a sparsity of p / (k / block_k)
    # p = 4
    # p = int(k / block_k * 0.125)
    p = int(k / block_k * 0.125)

    # Create dense matrices
    dense_a = prepare_A_dense(m, k)
    dense_b = prepare_B_dense(k, n)

    # Prepare matrices in the specified formats
    tiled_a = prepare_A_tiled(dense_a, block_m, block_k)
    b_values, b_metadata = prepare_B_block_sparse(
        dense_b, block_k, block_n, p, random_seed=42
    )

    # Output tensor for Triton kernel
    c_triton = torch.empty((m, n), device="cuda", dtype=torch.float32)

    # Grid for launching the kernel
    grid = (m // block_m, n // block_n)
    num_blocks_k = k // block_k

    # --- Benchmark Triton Kernel ---
    triton_benchmark = triton.testing.do_bench(
        lambda: block_mma_kernel_vectorized[grid](
            tiled_a,
            b_values,
            b_metadata,
            c_triton,
            m,
            n,
            k,
            tiled_a.stride(0),
            tiled_a.stride(1),
            tiled_a.stride(2),
            b_values.stride(0),
            b_values.stride(1),
            b_values.stride(2),
            b_metadata.stride(0),
            b_metadata.stride(1),
            c_triton.stride(0),
            c_triton.stride(1),
            BLOCK_M=block_m,  # type: ignore[no-untyped-call]
            BLOCK_N=block_n,  # type: ignore[no-untyped-call]
            BLOCK_K=block_k,  # type: ignore[no-untyped-call]
            P=p,  # type: ignore[no-untyped-call]
            NUM_BLOCKS_K=num_blocks_k,  # type: ignore[no-untyped-call]
            num_stages=3,
            num_warps=2,
        )
    )

    # --- Benchmark PyTorch Dense MMA ---
    pytorch_benchmark = triton.testing.do_bench(
        lambda: torch.matmul(dense_a, dense_b)
    )

    print("Triton Block-Sparse MMA Performance:")
    print(f"  Median Time: {triton_benchmark:.4f} ms")

    print("\nPyTorch Dense MMA Performance:")
    print(f"  Median Time: {pytorch_benchmark:.4f} ms")

    print("\nSpeedup (Triton / PyTorch):", pytorch_benchmark / triton_benchmark)


if __name__ == "__main__":
    benchmark()
