import torch
import pytest
from tritonix.matrix.sparse import dense_blocksparse_mm
from tritonix.utils.initialize import create_blocksparse
import triton



@pytest.mark.parametrize(
    "M, K, N, size_k, size_n, P, BLOCK_M, BLOCK_P, group_m, dtype",
    [
        # Basic test case from sparse.py
        (256, 128, 256, 8, 16, 8, 32, 8, 4, torch.float32),
        # Larger sizes
        (512, 256, 512, 16, 32, 8, 64, 4, 8, torch.float32),
        # Different data type
        (64, 32, 64, 2, 16, 8, 16, 8, 2, torch.float16),
        # M not a multiple of BLOCK_M
        (100, 64, 128, 4, 16, 8, 16, 4, 2, torch.float32),
        # N not a multiple of size_n
        (128, 64, 100, 4, 16, 8, 16, 4, 2, torch.float32),
        # K not a multiple of size_k
        (128, 60, 128, 4, 16, 10, 16, 5, 2, torch.float32),
        # P not a multiple of BLOCK_P
        (128, 64, 128, 4, 16, 10, 16, 5, 2, torch.float32),
    ],
)
def test_dense_blocksparse_mm(
    M, K, N, size_k, size_n, P, BLOCK_M, BLOCK_P, group_m, dtype
):
    torch.manual_seed(0)

    b_values, b_indices, B_dense = create_blocksparse(
        K, N, size_k, size_n, P=P, dtype=dtype, seed=0
    )
    a = torch.randn((K, M), device="cuda", dtype=dtype).t()

    max_k_block_idx = K // size_k
    if b_indices.numel() > 0 and b_indices.max() >= max_k_block_idx:
        print(f"ERROR: Invalid index in b_indices!")
        print(f"Max K block index should be < {max_k_block_idx}, but found {b_indices.max().item()}")

    c = torch.empty((M, N), device="cuda", dtype=torch.float32)
    torch.cuda.synchronize() 

    # dense_blocksparse_mm[(triton.cdiv(M, BLOCK_M), N // size_n)](
    dense_blocksparse_mm[(triton.cdiv(M, BLOCK_M), triton.cdiv(N, size_n))](
        a_ptr=a,
        b_values_ptr=b_values,
        b_indices_ptr=b_indices,
        c_ptr=c,
        m=M,
        n=N,
        k=K,
        stride_cm=c.stride(0),  # Assuming float32
        stride_cn=c.stride(1),  # Assuming float32
        p=P,
        size_k=size_k,
        size_n=size_n,
        block_m=BLOCK_M,
        block_p=BLOCK_P,
        group_m=group_m,
    )
    torch.cuda.synchronize() 

    golden = torch.matmul(a, B_dense).float()

    # Using a tolerance that's appropriate for the data type
    atol = 1e-2 if dtype == torch.float16 else 1e-4
    rtol = 1e-2 if dtype == torch.float16 else 1e-4
    assert torch.allclose(c, golden, atol=atol, rtol=rtol), (
        f"Max diff: {torch.max(torch.abs(c - golden))}"
    )


if __name__ == "__main__":
    pytest.main([__file__])
