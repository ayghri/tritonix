import torch
import triton
from triton.testing import do_bench
import triton.language as tl
from tritonix.matrix.sparse import dense_block_sparse_kernel_v0 as dense_block_sparse_kernel
from tritonix.utils.initialize import create_blocksparse


@triton.jit
def dense_col_major_x_block_sparse_pipelined_kernel(
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
    BLOCK_M: tl.constexpr,
    BLOCK_P: tl.constexpr,
    # GROUP_M=tl.constexpr(4),
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    # num_n = tl.cdiv(N, B_N)
    # num_m = tl.cdiv(M, BLOCK_M)
    # pid_m, pid_n = tl.swizzle2d(pid_m, pid_n, num_m, num_n, GROUP_M)

    m_start = pid_m * BLOCK_M
    n_start = pid_n * B_N
    accumulator = tl.zeros((BLOCK_M, B_N), dtype=tl.float32)

    a_offs_m = m_start + tl.arange(0, BLOCK_M)
    a_mask_m = a_offs_m < M

    # k_chunk_idx = 0

    # indices_start_offset = pid_n * P + k_chunk_idx * BLOCK_P
    offs_p_vec = tl.arange(0, BLOCK_P)
    # p_mask = (offs_p_vec) < P
    # indices_ptrs = b_indices_ptr + indices_start_offset
    # block_row_k_vec = tl.load(indices_ptrs + offs_p_vec, mask=p_mask, other=0)

    # Load first A tile (gather)
    # k_offsets_scattered = (
    #     block_row_k_vec[:, None] * B_K + tl.arange(0, B_K)[None, :]
    # )
    # k_offsets_flat = tl.reshape(k_offsets_scattered, (BLOCK_P * B_K,))
    # k_mask = tl.broadcast_to(p_mask[:, None], (BLOCK_P, B_K))
    # k_mask_flat = tl.reshape(k_mask, (BLOCK_P * B_K,))
    # a_ptrs = a_ptr + (
    #     a_offs_m[:, None] * stride_am + k_offsets_flat[None, :] * stride_ak
    # )
    # a_tile = tl.load(
    #     a_ptrs, mask=a_mask_m[:, None] & k_mask_flat[None, :], other=0.0
    # )

    # Load first B tile
    # b_values_start_offset = indices_start_offset * (B_K * B_N)
    b_offs = tl.arange(0, BLOCK_P * B_K * B_N)
    # b_ptrs = b_values_ptr + b_values_start_offset + b_offs
    # b_mask = tl.broadcast_to(p_mask[:, None, None], (BLOCK_P, B_K, B_N))
    # b_mask_flat = tl.reshape(b_mask, (BLOCK_P * B_K * B_N,))
    # b_chunk_flat = tl.load(b_ptrs, mask=b_mask_flat, other=0.0)
    # b_tile = tl.reshape(b_chunk_flat, (BLOCK_P * B_K, B_N))

    # --- Main Pipelined Loop ---
    num_k_chunks = tl.cdiv(P, BLOCK_P)

    # for k_chunk_idx in range(0, num_k_chunks):
    for k_chunk_idx in range(0, num_k_chunks):
        # 1. MMA on the current chunk (data is already in registers)

        # accumulator = tl.dot(a_tile, b_tile, accumulator)

        # 2. Asynchronously load data for the NEXT chunk while computing
        indices_start_offset = pid_n * P + k_chunk_idx * BLOCK_P
        p_mask = (k_chunk_idx * BLOCK_P + offs_p_vec) < P

        # Load next indices
        indices_ptrs = b_indices_ptr + indices_start_offset
        block_row_k_vec = tl.load(
            indices_ptrs + offs_p_vec, mask=p_mask, other=0
        )

        # Load next A tile
        k_offsets_scattered = (
            block_row_k_vec[:, None] * B_K + tl.arange(0, B_K)[None, :]
        )
        k_offsets_flat = tl.reshape(k_offsets_scattered, (BLOCK_P * B_K,))
        k_mask_flat = tl.reshape(
            tl.broadcast_to(p_mask[:, None], (BLOCK_P, B_K)), (BLOCK_P * B_K,)
        )
        a_ptrs = a_ptr + (
            a_offs_m[:, None] * stride_am + k_offsets_flat[None, :] * stride_ak
        )
        a_tile = tl.load(
            a_ptrs, mask=a_mask_m[:, None] & k_mask_flat[None, :], other=0.0
        )

        # Load next B tile
        b_values_start_offset = indices_start_offset * (B_K * B_N)
        b_mask_flat = tl.reshape(
            tl.broadcast_to(p_mask[:, None, None], (BLOCK_P, B_K, B_N)),
            (BLOCK_P * B_K * B_N,),
        )
        b_ptrs = b_values_ptr + b_values_start_offset + b_offs
        b_chunk_flat = tl.load(b_ptrs, mask=b_mask_flat, other=0.0)
        b_tile = tl.reshape(b_chunk_flat, (BLOCK_P * B_K, B_N))

        accumulator = tl.dot(a_tile, b_tile, accumulator)

    # accumulator = tl.dot(a_tile, b_tile, accumulator)

    accumulator = accumulator.to(c_ptr.dtype.element_ty)
    offs_c_m = m_start + tl.arange(0, BLOCK_M)
    offs_c_n = n_start + tl.arange(0, B_N)
    c_ptrs = c_ptr + (
        offs_c_m[:, None] * stride_cm + offs_c_n[None, :] * stride_cn
    )
    c_mask = (a_mask_m[:, None]) & (offs_c_n[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


# # Helper function (can be simplified as we only need one B)
# def create_block_sparse_b_and_dense_b(K, N, B_K, B_M, P, dtype, device="cuda"):
#     total_k_blocks = K // B_K
#     k_block_indices_for_col = torch.randperm(total_k_blocks, device=device)[
#         :P
#     ].sort()[0]
#     B_dense_reconstructed = torch.zeros((K, N), device=device, dtype=dtype)
#     b_values_list, b_indices_list = [], []
#     for j in range(N // B_M):
#         for p_idx in range(P):
#             block_row_k = k_block_indices_for_col[p_idx]
#             block = torch.randn((B_K, B_M), device=device, dtype=dtype)
#             b_values_list.append(block)
#             b_indices_list.append(block_row_k)
#             B_dense_reconstructed[
#                 block_row_k * B_K : (block_row_k + 1) * B_K,
#                 j * B_M : (j + 1) * B_M,
#             ] = block
#     b_values = torch.stack(b_values_list).flatten().contiguous()
#     b_indices = torch.tensor(
#         b_indices_list, dtype=torch.int32, device=device
#     ).contiguous()
#     return b_values, b_indices, B_dense_reconstructed


# --- Main Tuning Script ---
def main():
    # Fixed problem parameters
    # M, K, N = 512, 1024 * 4, 1024 * 4
    M = K = N = 1024 * 4

    # M = 512
    B_K = B_N = 16
    DTYPE = torch.float16
    SPARSITY = 0.5  # Our target!

    total_k_blocks = K // B_K
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
        # {"BLOCK_SIZE_M": 64, "BLOCK_P": 16, "num_warps": 4},
        # {"BLOCK_SIZE_M": 64, "BLOCK_P": 32, "num_warps": 4},
        # {"BLOCK_SIZE_M": 64, "BLOCK_P": 32, "num_warps": 8},
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
    A_triton = (A_torch*1.).t().contiguous().t()
    # A_triton = A_torch
    # b_values, b_indices, B_dense = create_block_sparse_b_and_dense_b(
    #     K, N, B_K, B_M, P, DTYPE
    # )
    b_values, b_indices, B_dense = create_blocksparse(K, N, B_K, B_N, P, DTYPE)

    print("Sparsity of B_dense:", (B_dense != 0).sum().item()/K/N)
    print(
        "Dense abs mean vs B_values mean:",
        B_dense.abs().mean().item()/b_values.numel() * B_dense.numel(),
        b_values.abs().mean().item(),
    )

    # Benchmark PyTorch once as a baseline
    pytorch_ms = do_bench(lambda: torch.matmul(A_torch, B_dense))
    C_torch = torch.matmul(A_torch, B_dense)

    print(
        f"Tuning Experiment for A100 at {SPARSITY * 100}% Sparsity, matrix sizes:"
    )
    print(f"M: {M}, K: {K}, N: {N}, B_K: {B_K}, B_M: {B_N}, P: {P}")
    print(f"PyTorch Dense Baseline: {pytorch_ms:.4f} ms")
    print("-" * 70)
    print(
        f"{'BLOCK_M':<10}{'BLOCK_P':<10}{'num_warps':<12}{'Triton (ms)':<15}{'Speedup':<10}{'MSE':<15}"
    )
    print("-" * 70)

    # for num_stages in [1, 3, 5, 6]:

    print(
        "Torch flops:", K * M * N / pytorch_ms / 1024**3 / 1e-3 / DTYPE.itemsize
    )
    for cfg in configs:
        # Adjust P to be a multiple of BLOCK_P for optimal perf
        if P % cfg["BLOCK_P"] != 0:
            P_adj = (P // cfg["BLOCK_P"]) * cfg["BLOCK_P"]
        else:
            P_adj = P

        # We need to re-JIT the kernel for each set of constexpr arguments
        # kernel = dense_col_major_x_block_sparse_vectorized_kernel

        # kernel = dense_col_major_x_block_sparse_pipelined_kernel
        kernel = dense_block_sparse_kernel

        # The grid needs to be recomputed for each BLOCK_SIZE_M
        grid = (triton.cdiv(M, cfg["BLOCK_SIZE_M"]), triton.cdiv(N, B_N))

        C_triton = torch.zeros((M, N), device="cuda", dtype=DTYPE)

        # Run benchmark
        def run_kernel(a):
            # A = a.t().contiguous().t()  # Ensure column-major order
            A = A_triton
            # C_triton = torch.empty((M, N), device="cuda", dtype=DTYPE)
            return kernel[grid](
                a,
                b_values,
                b_indices,
                C_triton,
                M,
                N,
                K,
                A.stride(0),
                A.stride(1),
                C_triton.stride(0),
                C_triton.stride(1),
                P=P_adj,
                B_K=B_K,
                B_N=B_N,
                BLOCK_M=cfg["BLOCK_SIZE_M"],
                BLOCK_P=cfg["BLOCK_P"],
                # GROUP_M=2,
                num_warps=cfg["num_warps"],  # type: ignore
                num_stages=3,
            )

        # try:
        triton_ms = 1.0
        triton_ms = do_bench(lambda: run_kernel(A_torch))
        # except Exception as e:
        #     print(f"Error with config {cfg}: {e}")
        #     triton_ms = float("inf")
        #     continue

        speedup = pytorch_ms / triton_ms  # type: ignore
        # print(torch.abs(C_triton).mean(), C_torch.abs().mean())

        print(
            f"{cfg['BLOCK_SIZE_M']:<10}{cfg['BLOCK_P']:<10}{cfg['num_warps']:<12}"
            f"{triton_ms:<15.4f}x{speedup:<10.2f}{(C_triton - C_torch).abs().max()}"
        )


if __name__ == "__main__":
    main()
