import torch


def create_blocksparse(K, N, SIZE_K, SIZE_N, P, dtype, device="cuda", seed=None):
    """
    Creates a block-sparse representation where, within each block-column,
    the non-zero blocks are sorted by their K-index.
    """
    num_block_cols = N // SIZE_N
    total_k_blocks = K // SIZE_K
    if seed is not None:
        torch.manual_seed(seed)

    b_values = torch.randn(
        (num_block_cols * P, SIZE_K, SIZE_N), dtype=dtype, device=device
    )
    b_indices = torch.empty(
        num_block_cols * P, dtype=torch.int32, device=device
    )

    for j in range(num_block_cols):
        col_block_indices = torch.randperm(total_k_blocks, device=device)[:P]
        col_block_indices, _ = torch.sort(col_block_indices)
        b_indices[j * P : (j + 1) * P] = col_block_indices

    B_dense = torch.zeros((K, N), device=device, dtype=dtype)
    for j in range(num_block_cols):
        for p_idx in range(P):
            nnz_idx = j * P + p_idx
            block_row_k = b_indices[nnz_idx]
            vals = b_values[nnz_idx]
            B_dense[
                block_row_k * SIZE_K : (block_row_k + 1) * SIZE_K,
                j * SIZE_N : (j + 1) * SIZE_N,
            ] = vals

    return b_values, b_indices, B_dense
