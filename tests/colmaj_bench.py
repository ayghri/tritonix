import torch
import triton
import triton.language as tl
from triton.testing import do_bench



DTYPE = torch.float16
# ---------------------------------------------------------------------------
# The Vectorized Triton Kernel (from previous response)
# ---------------------------------------------------------------------------
@triton.jit
def dense_col_major_x_block_sparse_vectorized_kernel(
    a_ptr,
    b_values_ptr,
    b_indices_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_cm,
    stride_cn,
    P: tl.constexpr,
    B_K: tl.constexpr,
    B_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    accumulator = tl.zeros((BLOCK_SIZE_M, B_N), dtype=tl.float32)

    m_start = pid_m * BLOCK_SIZE_M
    n_start = pid_n * B_N

    num_k_chunks = tl.cdiv(P, BLOCK_P)
    for k_chunk_idx in range(num_k_chunks):
        indices_start_offset = pid_n * P + k_chunk_idx * BLOCK_P
        offs_p_vec = tl.arange(0, BLOCK_P)
        indices_ptrs = b_indices_ptr + indices_start_offset + offs_p_vec
        p_mask = (k_chunk_idx * BLOCK_P + offs_p_vec) < P
        block_row_k_vec = tl.load(indices_ptrs, mask=p_mask, other=0)

        k_offsets_scattered = (
            block_row_k_vec[:, None] * B_K + tl.arange(0, B_K)[None, :]
        )
        k_offsets_flat = tl.reshape(k_offsets_scattered, (BLOCK_P * B_K,))

        a_offs_m = m_start + tl.arange(0, BLOCK_SIZE_M)
        a_ptrs = a_ptr + (
            a_offs_m[:, None] * stride_am + k_offsets_flat[None, :] * stride_ak
        )

        k_mask = tl.broadcast_to(p_mask[:, None], (BLOCK_P, B_K))
        k_mask_flat = tl.reshape(k_mask, (BLOCK_P * B_K,))
        a_tile = tl.load(
            a_ptrs,
            mask=(a_offs_m[:, None] < M) & k_mask_flat[None, :],
            other=0.0,
        )

        b_values_start_offset = indices_start_offset * (B_K * B_N)
        b_offs = tl.arange(0, BLOCK_P * B_K * B_N)
        b_ptrs = b_values_ptr + b_values_start_offset + b_offs

        b_mask = tl.broadcast_to(p_mask[:, None, None], (BLOCK_P, B_K, B_N))
        b_mask_flat = tl.reshape(b_mask, (BLOCK_P * B_K * B_N,))
        b_chunk_flat = tl.load(b_ptrs, mask=b_mask_flat, other=0.0)
        b_tile = tl.reshape(b_chunk_flat, (BLOCK_P * B_K, B_N))

        accumulator = tl.dot(a_tile, b_tile, accumulator)

    # accumulator = accumulator.to(c_ptr.dtype.element_ty)
    offs_c_m = m_start + tl.arange(0, BLOCK_SIZE_M)
    offs_c_n = n_start + tl.arange(0, B_N)
    c_ptrs = c_ptr + (
        offs_c_m[:, None] * stride_cm + offs_c_n[None, :] * stride_cn
    )
    c_mask = (offs_c_m[:, None] < M) & (offs_c_n[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


# ---------------------------------------------------------------------------
# Helper function to create the sparse data structures
# ---------------------------------------------------------------------------
def create_block_sparse_b_and_dense_b(K, N, B_K, B_N, P, device="cuda"):
    """
    Creates both the sparse representation for Triton and a dense, reconstructed
    version for PyTorch's matmul.
    """
    num_block_cols = N // B_N
    b_values_list = []
    b_indices_list = []

    # For a fair benchmark, we use the same random indices for all columns
    # and sort them to improve memory access patterns for the A gather.
    total_k_blocks = K // B_K
    k_block_indices_for_col = torch.randperm(total_k_blocks, device=device)[
        :P
    ].sort()[0]

    # This will be the dense version for PyTorch
    B_dense_reconstructed = torch.zeros(
        (K, N), device=device, dtype=DTYPE
    )

    for j in range(num_block_cols):
        for p_idx in range(P):
            block_row_k = k_block_indices_for_col[p_idx]

            # Create a random block of data
            block = torch.randn((B_K, B_N), device=device, dtype=DTYPE)
            b_values_list.append(block)
            b_indices_list.append(block_row_k)

            # Place the same block in the dense matrix for the reference calculation
            B_dense_reconstructed[
                block_row_k * B_K : (block_row_k + 1) * B_K,
                j * B_N : (j + 1) * B_N,
            ] = block

    b_values = torch.stack(b_values_list).flatten().contiguous()
    b_indices = torch.tensor(
        b_indices_list, dtype=torch.int32, device=device
    ).contiguous()

    return b_values, b_indices, B_dense_reconstructed


# ---------------------------------------------------------------------------
# Main Benchmark Function
# ---------------------------------------------------------------------------
def main():
    # Matrix dimensions
    M, K, N = 1024, 4096, 2048

    # Block sparse parameters
    B_K, B_N = 16, 16

    # Kernel meta-parameters
    BLOCK_SIZE_M = 128
    BLOCK_P = 4  # Vectorized step size

    print("Benchmarking Triton Sparse Kernel vs. PyTorch Dense Matmul")
    print(f"Matrix Dims: A({M}, {K}) @ B({K}, {N})")
    print(
        f"B-Block Dims: ({B_K}, {B_N}). Triton Tile Dims: M={BLOCK_SIZE_M}, N={B_N}, K-Group={BLOCK_P * B_K}"
    )
    print("-" * 60)
    print(
        f"{'Sparsity (%)':<15}{'Triton (ms)':<15}{'PyTorch (ms)':<15}{'Speedup':<10}"
    )
    print("-" * 60)

    # Test a range of sparsity levels
    sparsity_levels = [0.1, 0.25, 0.5, 0.75, 1.0]

    for sparsity in sparsity_levels:
        # Number of non-zero blocks per block column in B
        total_k_blocks = K // B_K
        P = int(total_k_blocks * sparsity)
        if P == 0:
            P = 1
        if P % BLOCK_P != 0 and P > BLOCK_P:
            # Adjust P to be a multiple of BLOCK_P for optimal performance, unless P is very small
            P = (P // BLOCK_P) * BLOCK_P

        # Create input matrices
        # A is created once for all tests
        A_torch = torch.randn((M, K), device="cuda", dtype=DTYPE)
        # We need a column-major layout for our kernel
        A_triton = A_torch.t().contiguous().t()

        # Create sparse B and its dense equivalent
        b_values, b_indices, B_dense = create_block_sparse_b_and_dense_b(
            K, N, B_K, B_N, P
        )

        # Output tensor for Triton
        C_triton = torch.empty((M, N), device="cuda", dtype=DTYPE)

        # --- Benchmark Triton Kernel ---
        grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, B_N))

        triton_ms = do_bench(
            lambda: dense_col_major_x_block_sparse_vectorized_kernel[grid](
                A_triton,
                b_values,
                b_indices,
                C_triton,
                M,
                N,
                K,
                A_triton.stride(0),
                A_triton.stride(1),
                C_triton.stride(0),
                C_triton.stride(1),
                P=P,
                B_K=B_K,
                B_N=B_N,
                BLOCK_SIZE_M=BLOCK_SIZE_M,
                BLOCK_P=BLOCK_P,
            )
        )

        # --- Benchmark PyTorch Dense Kernel ---
        pytorch_ms = do_bench(lambda: torch.matmul(A_torch, B_dense))

        # --- Verification (optional, but good practice) ---
        # reference_output = torch.matmul(A_torch, B_dense)
        # assert torch.allclose(C_triton, reference_output, atol=1e-1, rtol=0), "Verification failed!"

        # Print results
        speedup = pytorch_ms / triton_ms
        sparsity_percent = int((1 - sparsity) * 100)
        print(
            f"{sparsity_percent:<15}{triton_ms:<15.4f}{pytorch_ms:<15.4f}{speedup:<10.2f}x"
        )


if __name__ == "__main__":
    main()
