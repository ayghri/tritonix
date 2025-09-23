import torch


def create_bsr_matrix(
    shape, block_size, sparsity, device="cpu", dtype=torch.float16
):
    """
    Creates a random block sparse row (BSR) matrix.

    Args:
        shape (tuple): The desired shape of the dense matrix (rows, cols).
        block_size (tuple): The size of each block (block_rows, block_cols).
        sparsity (float): The desired sparsity of the block matrix (0.0 to 1.0).
        device (str): The device to create the tensors on.
        dtype (torch.dtype): The data type of the matrix.

    Returns:
        torch.sparse.Tensor: A sparse BSR tensor.
    """
    dense_rows, dense_cols = shape
    block_rows, block_cols = block_size

    if dense_rows % block_rows != 0 or dense_cols % block_cols != 0:
        raise ValueError(
            "Matrix dimensions must be divisible by block dimensions."
        )

    grid_rows = dense_rows // block_rows
    grid_cols = dense_cols // block_cols

    # Create a random layout of blocks
    num_blocks = grid_rows * grid_cols
    num_non_zero_blocks = int(num_blocks * (1 - sparsity))

    # Create random indices for the non-zero blocks
    indices = torch.randperm(num_blocks, device=device)[:num_non_zero_blocks]
    row_indices = indices // grid_cols
    col_indices = indices % grid_cols

    # Sort indices for BSR format
    sorted_indices = torch.argsort(row_indices)
    row_indices = row_indices[sorted_indices]
    col_indices = col_indices[sorted_indices]

    # Create indptr
    indptr = torch.zeros(grid_rows + 1, dtype=torch.long, device=device)
    # This is a more efficient way to create indptr
    indptr[1:] = torch.cumsum(
        torch.bincount(row_indices, minlength=grid_rows), dim=0
    )

    # Create random data for the non-zero blocks
    values = torch.randn(
        num_non_zero_blocks, block_rows, block_cols, device=device, dtype=dtype
    )

    bsr = torch.sparse_bsr_tensor(
        indptr, col_indices, values, size=shape, dtype=dtype, device=device
    )
    return bsr
