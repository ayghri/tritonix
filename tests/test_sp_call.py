import torch
import triton
from kernels.matrix.sparse import dense_blocksparse_mm  # Your kernel
from kernels.utils.initialize import create_blocksparse  # Your utility
from triton.testing import do_bench


def is_power_of_two(n: int) -> bool:
    """Checks if a number is a power of two."""
    return (n > 0) and (n & (n - 1) == 0)


def efficient_blocksparse_mm(
    a: torch.Tensor,
    b_values: torch.Tensor,
    b_indices: torch.Tensor,
    n_dim: int,
    p: int,
    size_k: int,
    size_n: int,
    block_m: int = 32,
    block_p: int = 8,
    group_m: int = 8,
    num_warps: int = 4,
) -> torch.Tensor:
    """
    A safe and efficient caller for the dense_blocksparse_mm Triton kernel.

    This function validates inputs to match the kernel's performance assumptions
    and abstracts away the grid calculation and launch configuration.

    Args:
        a (torch.Tensor): The dense input matrix A, with shape (M, K).
        b_values (torch.Tensor): The non-zero blocks of the sparse matrix B.
        b_indices (torch.Tensor): The row indices of the non-zero blocks.
        n_dim (int): The N dimension of the output matrix C.
        p (int): The number of non-zero blocks per column in matrix B.
        size_k (int): The height of the sparse blocks. Must be a power of 2 >= 16.
        size_n (int): The width of the sparse blocks. Must be a power of 2 >= 16.
        block_m (int): Tuning: Block size for the M dimension. Must be a power of 2 >= 16.
        block_p (int): Tuning: Number of P blocks to process per iteration. Must be a power of 2.
        group_m (int): Tuning: Swizzle factor.

    Returns:
        torch.Tensor: The resulting output matrix C, with shape (M, N).
    """
    # 1. --- Input Validation ---
    # Enforce the power-of-2 and >= 16 constraints for performance.
    for name, value in [
        ("size_k", size_k),
        ("size_n", size_n),
        ("block_m", block_m),
    ]:
        if not is_power_of_two(value) or value < 16:
            raise ValueError(
                f"{name} must be a power of 2 and >= 16, but got {value}."
            )
    if not is_power_of_two(block_p):
        raise ValueError(f"block_p must be a power of 2, but got {block_p}.")

    # 2. --- Dimension and Tensor Setup ---
    m_dim, k_dim = a.shape

    # Ensure the output tensor is created with the correct data type for accumulation
    c = torch.empty((m_dim, n_dim), device="cuda", dtype=torch.float32)

    # 3. --- Grid Calculation ---
    # Use triton.cdiv to safely calculate the grid size, even for non-multiple dimensions.
    grid = (triton.cdiv(m_dim, block_m), triton.cdiv(n_dim, size_n))

    # 4. --- Kernel Launch ---
    # Call the kernel with validated and configured parameters.
    dense_blocksparse_mm[grid](
        a_ptr=a,
        b_values_ptr=b_values,
        b_indices_ptr=b_indices,
        c_ptr=c,
        m=m_dim,
        n=n_dim,
        k=k_dim,
        stride_cm=c.stride(0),
        stride_cn=c.stride(1),
        p=p,
        size_k=size_k,
        size_n=size_n,
        block_m=block_m,
        block_p=block_p,
        group_m=group_m,
        num_warps=num_warps,
        # num_stages=3,
    )

    return c


# if __name__ == "__main__":
#     torch.manual_seed(0)

#     # Define parameters that satisfy the wrapper's validation rules
#     M, K, N = 512, 256, 512
#     DTYPE = torch.float32

#     # Kernel-specific parameters that are now validated by the caller
#     SIZE_K = 16  # Power of 2, >= 16
#     SIZE_N = 32  # Power of 2, >= 16
#     P = 8  # Number of non-zero blocks per column

#     # Create the test matrices
#     b_values, b_indices, B_dense = create_blocksparse(
#         K, N, SIZE_K, SIZE_N, P=P, dtype=DTYPE, seed=0
#     )
#     a = torch.randn((K, M), device="cuda", dtype=DTYPE).t()

#     # --- Use the new, safe caller function ---
#     # Notice we don't need to calculate the grid or worry about constexpr values here.
#     c_triton = efficient_blocksparse_mm(
#         a=a,
#         b_values=b_values,
#         b_indices=b_indices,
#         n_dim=N,
#         p=P,
#         size_k=SIZE_K,
#         size_n=SIZE_N,
#         # We can optionally override tuning params, e.g., block_m=64
#     )
#     # -----------------------------------------

#     # Verification
#     print("Verifying the output...")
#     c_torch = torch.matmul(a, B_dense)

#     # Use appropriate tolerance for the data type
#     atol = 1e-2 if DTYPE == torch.float16 else 1e-4
#     rtol = 1e-2 if DTYPE == torch.float16 else 1e-4

#     is_close = torch.allclose(c_triton, c_torch, atol=atol, rtol=rtol)
#     print(f"Outputs are close: {is_close}")
#     if not is_close:
#         print(f"Max difference: {torch.max(torch.abs(c_triton - c_torch))}")

#     # Example of what happens with invalid parameters
#     try:
#         print("\nTesting invalid parameters (this should fail)...")
#         efficient_blocksparse_mm(
#             a,
#             b_values,
#             b_indices,
#             N,
#             P,
#             size_k=10,  # Not a power of 2
#             size_n=16,
#         )
#     except ValueError as e:
#         print(f"Successfully caught expected error: {e}")


def main():
    # Fixed problem parameters
    M, K, N = 512, 1024 * 2, 1024 * 8
    SIZE_K, SIZE_N = 16, 16
    DTYPE = torch.float32
    SPARSITY = 0.5  # Our target!

    total_k_blocks = K // SIZE_K
    P = int(total_k_blocks * SPARSITY)

    # --- Define the configurations to test ---
    configs = [
        {"BLOCK_SIZE_M": 64, "BLOCK_P": 4, "num_warps": 4},
        {"BLOCK_SIZE_M": 64, "BLOCK_P": 8, "num_warps": 4},
        {"BLOCK_SIZE_M": 64, "BLOCK_P": 8, "num_warps": 8},
        {"BLOCK_SIZE_M": 128, "BLOCK_P": 4, "num_warps": 4},
        {"BLOCK_SIZE_M": 128, "BLOCK_P": 4, "num_warps": 4},
        {"BLOCK_SIZE_M": 128, "BLOCK_P": 4, "num_warps": 8},
        {"BLOCK_SIZE_M": 128, "BLOCK_P": 4, "num_warps": 8},
        {"BLOCK_SIZE_M": 128, "BLOCK_P": 8, "num_warps": 8},
        {"BLOCK_SIZE_M": 128, "BLOCK_P": 8, "num_warps": 16},
        {"BLOCK_SIZE_M": 128, "BLOCK_P": 4, "num_warps": 2},
        {"BLOCK_SIZE_M": 128, "BLOCK_P": 4, "num_warps": 2},
        {"BLOCK_SIZE_M": 128, "BLOCK_P": 4, "num_warps": 4},
        {"BLOCK_SIZE_M": 128, "BLOCK_P": 4, "num_warps": 4},
        {"BLOCK_SIZE_M": 128, "BLOCK_P": 8, "num_warps": 4},
        {"BLOCK_SIZE_M": 128, "BLOCK_P": 8, "num_warps": 8},
        {"BLOCK_SIZE_M": 256, "BLOCK_P": 4, "num_warps": 4},
        {"BLOCK_SIZE_M": 256, "BLOCK_P": 4, "num_warps": 8},
        {"BLOCK_SIZE_M": 256, "BLOCK_P": 8, "num_warps": 8},
        {"BLOCK_SIZE_M": 64, "BLOCK_P": 16, "num_warps": 4},
        {"BLOCK_SIZE_M": 64, "BLOCK_P": 32, "num_warps": 4},
        {"BLOCK_SIZE_M": 64, "BLOCK_P": 32, "num_warps": 8},
        {"BLOCK_SIZE_M": 128, "BLOCK_P": 16, "num_warps": 2},
        {"BLOCK_SIZE_M": 128, "BLOCK_P": 16, "num_warps": 2},
        {"BLOCK_SIZE_M": 128, "BLOCK_P": 16, "num_warps": 4},
        {"BLOCK_SIZE_M": 128, "BLOCK_P": 16, "num_warps": 4},
        {"BLOCK_SIZE_M": 128, "BLOCK_P": 32, "num_warps": 4},
        {"BLOCK_SIZE_M": 128, "BLOCK_P": 32, "num_warps": 8},
        {"BLOCK_SIZE_M": 256, "BLOCK_P": 16, "num_warps": 4},
        {"BLOCK_SIZE_M": 256, "BLOCK_P": 16, "num_warps": 8},
        {"BLOCK_SIZE_M": 256, "BLOCK_P": 32, "num_warps": 8},
    ]

    # Pre-create data
    A_torch = torch.randn((M, K), device="cuda", dtype=DTYPE)
    A_triton = A_torch.t().contiguous().t()
    # b_values, b_indices, B_dense = create_block_sparse_b_and_dense_b(
    #     K, N, SIZE_K, SIZE_N, P, DTYPE
    # )
    # b_values, b_indices, B_dense = create_block_sparse_b_sorted(
    #     K, N, SIZE_K, SIZE_N, P, DTYPE
    # )
    # Create the test matrices
    b_values, b_indices, B_dense = create_blocksparse(
        K, N, SIZE_K, SIZE_N, P=P, dtype=DTYPE, seed=0
    )
    # Benchmark PyTorch once as a baseline
    pytorch_ms = do_bench(lambda: torch.matmul(A_torch, B_dense))
    C_torch = torch.matmul(A_torch, B_dense)

    print("Tuning Experiment for A100 at 50% Sparsity, matrix sizes:")
    print(f"M: {M}, K: {K}, N: {N}, SIZE_K: {SIZE_K}, SIZE_N: {SIZE_N}, P: {P}")
    print(f"PyTorch Dense Baseline: {pytorch_ms:.4f} ms")
    print("-" * 70)
    print(
        f"{'BLOCK_M':<10}{'BLOCK_P':<10}{'num_warps':<12}{'Triton (ms)':<15}{'Speedup':<10}{'MSE':<15}"
    )
    print("-" * 70)

    for cfg in configs:
        triton_ms = 1.0
        triton_ms = do_bench(
            lambda: efficient_blocksparse_mm(
                A_triton,
                b_values=b_values,
                b_indices=b_indices,
                n_dim=N,
                p=P,
                size_k=SIZE_K,
                size_n=SIZE_N,
                block_m=cfg["BLOCK_SIZE_M"],
                block_p=cfg["BLOCK_P"],
                group_m=2,
                num_warps=cfg["num_warps"],
            )
        )

        # except Exception as e:
        #     print(f"Error with config {cfg}: {e}")
        #     triton_ms = float("inf")
        #     continue

        speedup = pytorch_ms / triton_ms  # type: ignore
        # print(torch.abs(C_triton).mean(), C_torch.abs().mean())
        C_triton = efficient_blocksparse_mm(
            A_triton,
            b_values=b_values,
            b_indices=b_indices,
            n_dim=N,
            p=P,
            size_k=SIZE_K,
            size_n=SIZE_N,
            block_m=cfg["BLOCK_SIZE_M"],
            block_p=cfg["BLOCK_P"],
            group_m=2,
            num_warps=cfg["num_warps"],
        )

        print(
            f"{cfg['BLOCK_SIZE_M']:<10}{cfg['BLOCK_P']:<10}{cfg['num_warps']:<12}"
            f"{triton_ms:<15.4f}{speedup:<10.2f}x{(C_triton - C_torch).square().mean():.5e}"
        )


if __name__ == "__main__":
    main()
