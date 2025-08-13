import torch
import triton
import triton.language as tl
from triton.testing import do_bench

def create_block_sparse_b_sorted(K, N, B_K, B_M, P, dtype, device="cuda"):
    """
    Creates a block-sparse representation where, within each block-column,
    the non-zero blocks are sorted by their K-index.
    This improves memory locality for the gather operation on matrix A.
    """
    num_block_cols = N // B_M

    # These will hold the final, sorted data
    final_b_values = torch.empty(
        num_block_cols * P * B_K * B_M, dtype=dtype, device=device
    )
    final_b_indices = torch.empty(
        num_block_cols * P, dtype=torch.int32, device=device
    )

    # Temporary dense matrix to pull data from
    B_dense_temp = torch.randn((K, N), device=device, dtype=dtype)

    total_k_blocks = K // B_K

    for j in range(num_block_cols):
        # 1. Select P random K-indices for this column
        k_indices_for_col = torch.randperm(total_k_blocks, device=device)[:P]

        # 2. <<< THE KEY STEP: SORT THE INDICES >>>
        sorted_k_indices, sort_order = torch.sort(k_indices_for_col)

        # Store the sorted indices for this block-column
        final_b_indices[j * P : (j + 1) * P] = sorted_k_indices

        # 3. Permute the blocks according to the sort order and store them
        for p_idx in range(P):
            original_k_idx = k_indices_for_col[sort_order[p_idx]]

            # This is not efficient, but correct for generation.
            # In a real scenario, you'd permute existing blocks.
            block = B_dense_temp[
                sorted_k_indices[p_idx] * B_K : (sorted_k_indices[p_idx] + 1)
                * B_K,
                j * B_M : (j + 1) * B_M,
            ]

            # Place the correctly ordered block into the final values tensor
            final_b_values[
                (j * P + p_idx) * (B_K * B_M) : (j * P + p_idx + 1)
                * (B_K * B_M)
            ] = block.flatten()

    # Create the dense version for PyTorch using the same sorted data
    B_dense_reconstructed = torch.zeros((K, N), device=device, dtype=dtype)
    for j in range(num_block_cols):
        for p_idx in range(P):
            nnz_idx = j * P + p_idx
            block_row_k = final_b_indices[nnz_idx]
            vals = final_b_values[
                nnz_idx * B_K * B_M : (nnz_idx + 1) * B_K * B_M
            ].view(B_K, B_M)
            B_dense_reconstructed[
                block_row_k * B_K : (block_row_k + 1) * B_K,
                j * B_M : (j + 1) * B_M,
            ] = vals

    return final_b_values, final_b_indices, B_dense_reconstructed


@triton.jit
def dense_x_sparse_atomic_vectorized_kernel(
    a_ptr, b_values_ptr, b_indices_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak, stride_cm, stride_cn,
    P: tl.constexpr, B_K: tl.constexpr, B_M: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_P: tl.constexpr,
):
    """
    Computes C = A @ B using atomic additions.
    The grid is 2D:
    - axis 0: M dimension of A/C, in steps of BLOCK_SIZE_M.
    - axis 1: The non-zero blocks of B, processed in groups of BLOCK_P.
    Each program computes BLOCK_P small matrix products and adds them to C.
    """
    # --- Program ID Mapping ---
    pid_m = tl.program_id(axis=0)
    pid_nnz_group = tl.program_id(axis=1)

    # --- M-dimension offsets for A and C (constant for this program) ---
    offs_m_a = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    a_mask_m = offs_m_a < M

    # --- Loop over the BLOCK_P non-zero blocks assigned to this program ---
    # This loop is small and its cost is amortized.
    for p_iter in range(BLOCK_P):
        # Calculate the absolute index of the current non-zero block
        nnz_block_idx = pid_nnz_group * BLOCK_P + p_iter

        # Find which block column (j) this block belongs to. This determines the C-columns.
        block_col_j = nnz_block_idx // P
        
        # Load the row index (k) for this non-zero block of B. This determines the A-rows.
        block_row_k = tl.load(b_indices_ptr + nnz_block_idx)

        # --- Load A Tile (Gather) ---
        offs_k_a = block_row_k * B_K + tl.arange(0, B_K)
        a_ptrs = a_ptr + (offs_m_a[:, None] * stride_am + offs_k_a[None, :] * stride_ak)
        a_tile = tl.load(a_ptrs, mask=a_mask_m[:, None], other=0.0)

        # --- Load B Block (Contiguous) ---
        b_values_offset = nnz_block_idx * (B_K * B_M)
        b_offs = tl.arange(0, B_K * B_M)
        b_ptrs = b_values_ptr + b_values_offset + b_offs
        b_block_flat = tl.load(b_ptrs)
        b_block = tl.reshape(b_block_flat, (B_K, B_M))

        # --- Compute MMA ---
        # Accumulator is re-initialized for each block, not carried over.
        result_tile = tl.dot(a_tile, b_block).to(c_ptr.dtype.element_ty)

        # --- Atomically Add to C ---
        offs_n_c = block_col_j * B_M + tl.arange(0, B_M)
        c_ptrs = c_ptr + (offs_m_a[:, None] * stride_cm + offs_n_c[None, :] * stride_cn)
        c_mask = (a_mask_m[:, None]) & (offs_n_c[None, :] < N)
        tl.atomic_add(c_ptrs, result_tile, mask=c_mask, sem="relaxed")

# (Assume create_block_sparse_b_sorted is defined as before)

def main():
    # Problem definition from L40s benchmark
    M, K, N = 512, 2048, 8192
    B_K, B_M = 16, 16
    DTYPE = torch.float16
    SPARSITY = 0.5
    
    total_k_blocks = K // B_K
    P = int(total_k_blocks * SPARSITY)

    # --- Aggressive Atomic Kernel Configuration ---
    # We want many small, fast programs.
    config = {'BLOCK_SIZE_M': 64, 'BLOCK_P': 4, 'num_warps': 4}

    # Adjust P to be a multiple of BLOCK_P
    if P % config['BLOCK_P'] != 0:
        P = (P // config['BLOCK_P']) * config['BLOCK_P']

    # Create data
    A_torch = torch.randn((M, K), device='cuda', dtype=DTYPE)
    A_triton = A_torch.t().contiguous().t()
    b_values, b_indices, B_dense = create_block_sparse_b_sorted(K, N, B_K, B_M, P, DTYPE)
    
    # Benchmark PyTorch baseline
    pytorch_ms = do_bench(lambda: torch.matmul(A_torch, B_dense))
    
    # --- Benchmark the new Atomic Kernel ---
    # CRITICAL: Output tensor MUST be initialized to zeros for atomic add.
    C_triton = torch.zeros((M, N), device='cuda', dtype=DTYPE)
    
    # Calculate the new grid size
    total_nnz_blocks = (N // B_M) * P
    grid = (triton.cdiv(M, config['BLOCK_SIZE_M']), total_nnz_blocks // config['BLOCK_P'])
    
    print(f"Testing Atomic-Add Strategy on L40S (Grid Size: {grid})")
    
    triton_ms = do_bench(lambda: dense_x_sparse_atomic_vectorized_kernel[grid](
        A_triton, b_values, b_indices, C_triton,
        M, N, K, A_triton.stride(0), A_triton.stride(1), C_triton.stride(0), C_triton.stride(1),
        P=P, B_K=B_K, B_M=B_M,
        **config
    ))
    
    speedup = pytorch_ms / triton_ms

    print("-" * 60)
    print(f"PyTorch Dense Baseline:      {pytorch_ms:.4f} ms")
    print(f"Triton Atomic-Add Kernel:    {triton_ms:.4f} ms")
    print(f"Speedup:                     {speedup:.2f}x")
    print("-" * 60)
    
    # Verification
    print("Verifying correctness...")
    reference_output = torch.matmul(A_torch, B_dense)
    is_correct = torch.allclose(C_triton, reference_output, atol=1e-1, rtol=0) # Atomics can have slightly different accumulation order
    if is_correct:
        print("✅ Verification PASSED.")
    else:
        print("❌ Verification FAILED.")

if __name__ == "__main__":
    main()