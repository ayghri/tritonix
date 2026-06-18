import torch
import triton
import triton.language as tl


@triton.jit
def dense_col_major_x_block_sparse_vectorized_kernel(
    # Pointers to matrices
    a_ptr,
    b_values_ptr,
    b_indices_ptr,
    c_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    # Strides for A (col-major) and C (row-major)
    stride_am,
    stride_ak,
    stride_cm,
    stride_cn,
    # Block sparse parameters
    P: tl.constexpr,  # Number of non-zero blocks per column of B
    B_K: tl.constexpr,  # Block size for K dimension
    B_N: tl.constexpr,  # Block size for N dimension
    # Meta-parameters for the kernel
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_P: tl.constexpr,  # Number of B blocks to process per vectorized step
):
    """
    Computes C = A @ B using a fully vectorized approach within each program.
    - Each program computes a (BLOCK_SIZE_M, B_N) tile of C.
    - It iterates over the K-dimension in chunks of (BLOCK_P * B_K).
    - In each step, it gathers `BLOCK_P` tiles from A and loads `BLOCK_P`
      contiguous blocks from B to perform one large tl.dot operation.
    """
    # -----------------------------------------------------------
    # Map program ids to the C tile it computes
    # -----------------------------------------------------------
    pid_m = tl.program_id(axis=0)  # Identifies the row-block of the C tile
    pid_n = tl.program_id(axis=1)  # Identifies the column-block of the C tile

    # -----------------------------------------------------------
    # Local accumulator and C tile pointers
    # -----------------------------------------------------------
    accumulator = tl.zeros((BLOCK_SIZE_M, B_N), dtype=tl.float32)

    m_start = pid_m * BLOCK_SIZE_M
    n_start = pid_n * B_N

    offs_c_m = m_start + tl.arange(0, BLOCK_SIZE_M)
    offs_c_n = n_start + tl.arange(0, B_N)
    c_ptrs = c_ptr + (
        offs_c_m[:, None] * stride_cm + offs_c_n[None, :] * stride_cn
    )
    c_mask = (offs_c_m[:, None] < M) & (offs_c_n[None, :] < N)

    # -----------------------------------------------------------
    # Loop over chunks of BLOCK_P non-zero blocks.
    # The body of this loop is now fully vectorized.
    # -----------------------------------------------------------
    num_k_chunks = tl.cdiv(P, BLOCK_P)
    for k_chunk_idx in range(num_k_chunks):
        # --- VECTORIZED LOAD OF B INDICES ---
        # 1. Calculate the starting offset for the current chunk of indices
        indices_start_offset = pid_n * P + k_chunk_idx * BLOCK_P
        offs_p_vec = tl.arange(0, BLOCK_P)

        # 2. Load BLOCK_P `k`-indices at once
        # block_row_k_vec is a 1D tensor of shape [BLOCK_P]
        indices_ptrs = b_indices_ptr + indices_start_offset + offs_p_vec
        # Boundary check for the last chunk if P is not a multiple of BLOCK_P
        p_mask = (k_chunk_idx * BLOCK_P + offs_p_vec) < P
        block_row_k_vec = tl.load(indices_ptrs, mask=p_mask, other=0)

        # --- VECTORIZED GATHER FROM A ---
        # 3. Construct scattered K-offsets for A based on the loaded indices
        # Shape: (BLOCK_P, B_K) -> flatten to (BLOCK_P * B_K)
        k_offsets_scattered = (
            block_row_k_vec[:, None] * B_K + tl.arange(0, B_K)[None, :]
        )
        k_offsets_flat = tl.reshape(k_offsets_scattered, (BLOCK_P * B_K,))

        # 4. Create pointers for a (BLOCK_SIZE_M, BLOCK_P * B_K) tile of A
        # The k-offsets are a tensor, resulting in a gather operation.
        a_offs_m = m_start + tl.arange(0, BLOCK_SIZE_M)
        a_ptrs = a_ptr + (
            a_offs_m[:, None] * stride_am + k_offsets_flat[None, :] * stride_ak
        )

        # 5. Load the A tile. Mask the M-rows and also the K-columns based on the p_mask.
        # This prevents loading data for padding blocks.
        a_mask = (a_offs_m[:, None] < M) & (
            p_mask[:, None, None] & (tl.arange(0, B_K)[None, None, :] < B_K)
        )
        # A more direct way to mask columns is to expand p_mask
        k_mask = tl.broadcast_to(p_mask[:, None], (BLOCK_P, B_K))
        k_mask_flat = tl.reshape(k_mask, (BLOCK_P * B_K,))
        a_tile = tl.load(
            a_ptrs,
            mask=(a_offs_m[:, None] < M) & k_mask_flat[None, :],
            other=0.0,
        )

        # --- VECTORIZED LOAD FROM B ---
        # 6. Load one large contiguous chunk of BLOCK_P blocks from B
        b_values_start_offset = indices_start_offset * (B_K * B_N)
        b_offs = tl.arange(0, BLOCK_P * B_K * B_N)
        b_ptrs = b_values_ptr + b_values_start_offset + b_offs
        # We also mask the load from B to avoid reading padding data
        b_mask = tl.broadcast_to(p_mask[:, None, None], (BLOCK_P, B_K, B_N))
        b_mask_flat = tl.reshape(b_mask, (BLOCK_P * B_K * B_N,))
        b_chunk_flat = tl.load(b_ptrs, mask=b_mask_flat, other=0.0)

        # 7. Reshape B into a (BLOCK_P * B_K, B_N) tile
        b_tile = tl.reshape(b_chunk_flat, (BLOCK_P * B_K, B_N))

        # --- SINGLE, LARGE MMA ---
        # 8. Perform one matmul on the (BLOCK_SIZE_M, BLOCK_P * B_K) tile from A
        # and the (BLOCK_P * B_K, B_N) tile from B.
        accumulator = tl.dot(a_tile, b_tile, accumulator, allow_tf32=False)

    # --- FINAL STORE ---
    # After the loop, write the final accumulated tile to C.
    tl.store(c_ptrs, accumulator, mask=c_mask)


def create_block_sparse_b(dense_b, B_K, B_N, P, device="cuda"):
    K, N = dense_b.shape
    num_block_cols = N // B_N
    b_values_list, b_indices_list = [], []
    k_block_indices = torch.arange(P, device=device)
    for j in range(num_block_cols):
        for p_idx in range(P):
            block_row_k = k_block_indices[p_idx]
            block = dense_b[
                block_row_k * B_K : (block_row_k + 1) * B_K,
                j * B_N : (j + 1) * B_N,
            ]
            b_values_list.append(block)
            b_indices_list.append(block_row_k)
    return (
        torch.stack(b_values_list).flatten().contiguous(),
        torch.tensor(
            b_indices_list, dtype=torch.int32, device=device
        ).contiguous(),
    )


def main():
    M, K, N = 1024, 1024, 512
    B_K, B_N = 16, 16
    sparsity = 0.5
    P = int((K // B_K) * sparsity)
    if P == 0:
        P = 1

    BLOCK_SIZE_M, BLOCK_P = 64, 4

    A_torch = torch.randn((M, K), device="cuda", dtype=torch.float32)
    A_triton = A_torch.t().contiguous().t()
    B_dense = torch.randn((K, N), device="cuda", dtype=torch.float32)
    b_values, b_indices = create_block_sparse_b(B_dense, B_K, B_N, P)
    C = torch.empty((M, N), device="cuda", dtype=torch.float32)

    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, B_N))

    dense_col_major_x_block_sparse_vectorized_kernel[grid](
        A_triton,
        b_values,
        b_indices,
        C,
        M,
        N,
        K,
        A_triton.stride(0),
        A_triton.stride(1),
        C.stride(0),
        C.stride(1),
        P=P,
        B_K=B_K,
        B_N=B_N,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_P=BLOCK_P,
    )

    B_reconstructed = torch.zeros_like(B_dense)
    num_block_cols = N // B_N
    for j in range(num_block_cols):
        for p_idx in range(P):
            nnz_idx = j * P + p_idx
            block_row_k = b_indices[nnz_idx]
            vals = b_values[
                nnz_idx * B_K * B_N : (nnz_idx + 1) * B_K * B_N
            ].view(B_K, B_N)
            B_reconstructed[
                block_row_k * B_K : (block_row_k + 1) * B_K,
                j * B_N : (j + 1) * B_N,
            ] = vals

    reference_output = torch.matmul(A_torch, B_reconstructed)

    print("Verifying correctness...")
    print("Diff norm: ", torch.norm(C - reference_output).item())
    print("C shape: ", C.shape, "Reference shape: ", reference_output.shape)
    print("C dtype: ", C.dtype, "Reference dtype: ", reference_output.dtype)
    print("C device: ", C.device, "Reference device: ", reference_output.device)
    print("Max diff: ", torch.max(torch.abs(C - reference_output)).item())
    assert torch.allclose(C, reference_output, atol=1e-2, rtol=0), (
        "Incorrect result!"
    )
    print("✅ Triton implementation is correct.")



if __name__ == "__main__":
    main()
