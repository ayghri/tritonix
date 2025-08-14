import triton
import triton.language as tl
import torch
from kernels.utils.initialize import create_blocksparse


@triton.jit
def dense_block_sparse_kernel(
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
    GROUP_M: tl.constexpr = tl.constexpr(2),
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)


    m_start = pid_m * BLOCK_M
    n_start = pid_n * B_N
    a_offs_m = m_start + tl.arange(0, BLOCK_M)
    a_mask_m = a_offs_m < M
    offs_p_vec = tl.arange(0, BLOCK_P)
    b_offs = tl.arange(0, BLOCK_P * B_K * B_N)

    accumulator = tl.zeros((BLOCK_M, B_N), dtype=tl.float32)

    for k_chunk_idx in range(0, tl.cdiv(P, BLOCK_P)):
        indices_start_offset = pid_n * P + k_chunk_idx * BLOCK_P
        p_mask = offs_p_vec < P - k_chunk_idx * BLOCK_P

        indices_ptrs = b_indices_ptr + indices_start_offset
        block_row_k_vec = tl.load(
            indices_ptrs + offs_p_vec, mask=p_mask, other=0
        )

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

        b_values_start_offset = indices_start_offset * (B_K * B_N)
        b_mask_flat = tl.reshape(
            tl.broadcast_to(p_mask[:, None, None], (BLOCK_P, B_K, B_N)),
            (BLOCK_P * B_K * B_N,),
        )
        b_ptrs = b_values_ptr + b_values_start_offset + b_offs
        b_chunk_flat = tl.load(b_ptrs, mask=b_mask_flat, other=0.0)
        b_tile = tl.reshape(b_chunk_flat, (BLOCK_P * B_K, B_N))

        accumulator = tl.dot(a_tile, b_tile, accumulator)

    accumulator = accumulator.to(c_ptr.dtype.element_ty)
    offs_c_m = m_start + tl.arange(0, BLOCK_M)
    offs_c_n = n_start + tl.arange(0, B_N)
    c_ptrs = c_ptr + (
        offs_c_m[:, None] * stride_cm + offs_c_n[None, :] * stride_cn
    )
    c_mask = (a_mask_m[:, None]) & (offs_c_n[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


@triton.jit
def dense_blocksparse_mm(
    a_ptr,
    b_values_ptr,
    b_indices_ptr,
    c_ptr,
    m,
    n,
    k,
    stride_cm,
    stride_cn,
    p: tl.constexpr,
    size_k: tl.constexpr,
    size_n: tl.constexpr,
    block_m: tl.constexpr,
    block_p: tl.constexpr,  # How many P blocks to process in one iteration
    group_m: tl.constexpr,  # How many M blocks to process in parallel (swizzle)
):
    pid_m = tl.program_id(axis=0)  # this is in [0, cdiv(M, BLOCK_M)]
    pid_n = tl.program_id(axis=1)  # this is in [0, cdiv(N,B_N)]

    # swizzle the program ids
    # num_pid_m = tl.cdiv(m, block_m)
    # num_pid_n = tl.cdiv(n, size_n)
    # pid_m, pid_n = tl.swizzle2d(pid_m, pid_n, num_pid_m, num_pid_n, group_m)

    # compiler hints
    # tl.assume(pid_m >= 0)
    # tl.assume(pid_n >= 0)
    # tl.assume(stride_cm > 0)
    # tl.assume(stride_cn > 0)
    # tl.multiple_of(n, size_n)
    # tl.multiple_of(k, size_k)
    # ----------------------

    accumulator = tl.zeros((block_m, size_n), dtype=tl.float32)
    m_start = pid_m * block_m
    n_start = pid_n * size_n

    indices_start_offset = pid_n * p  # which column to start from

    offs_p_vec = tl.arange(0, block_p)
    indices_ptrs = b_indices_ptr + indices_start_offset + offs_p_vec

    a_offs_m = m_start + tl.arange(0, block_m)

    b_offs = (
        offs_p_vec[:, None] * size_k * size_n
        + tl.arange(0, size_k * size_n)[None, :]
    )
    k_offs = tl.arange(0, size_k)
    b_ptrs = b_values_ptr + indices_start_offset * (size_k * size_n) + b_offs

    # compiler hints
    # tl.multiple_of(a_offs_m, block_m)
    # ----------------------
    # max_offs_p = tl.cdiv(k, size_k)

    for k_chunk_idx in range(tl.cdiv(p, block_p)):
        p_mask = offs_p_vec < p - k_chunk_idx * block_p

        # shape (block_p,)
        # block_row_k_vec = tl.load(indices_ptrs, mask=p_mask, other=max_offs_p)
        block_row_k_vec = tl.load(indices_ptrs, mask=p_mask, other=0.0)
        # print((pid_m, pid_n), block_row_k_vec)

        k_offsets_scattered = (
            block_row_k_vec[:, None] * size_k + k_offs[None, :]
        )
        k_offsets_flat = tl.reshape(k_offsets_scattered, (block_p * size_k,))

        a_ptrs = a_ptr + a_offs_m[:, None] + k_offsets_flat[None, :] * m
        # a_mask = tl.reshape((p_mask[:, None]) & (k_offsets_scattered < k), (block_p, size_k))

        # Shape (block_m, block_p * size_k)
        a_tile = tl.load(
            a_ptrs,
            mask=k_offsets_flat[None, :] < k,
            other=0.0,
        )

        # Shape (block_p, size_k * size_n)
        b_tile = tl.load(b_ptrs, mask=p_mask[:, None], other=0.0)
        b_tile = tl.reshape(b_tile, (block_p * size_k, size_n))
        # print((pid_m, pid_n), b_tile)

        # print("Shapes:", a_tile.shape, b_tile.shape, accumulator.shape)
        accumulator = tl.dot(a_tile, b_tile, accumulator, allow_tf32=False)

        b_ptrs += block_p * size_k * size_n
        indices_ptrs += block_p

    accumulator = accumulator.to(c_ptr.dtype.element_ty)
    offs_c_m = m_start + tl.arange(0, block_m)
    offs_c_n = n_start + tl.arange(0, size_n)
    c_ptrs = c_ptr + (
        offs_c_m[:, None] * stride_cm + offs_c_n[None, :] * stride_cn
    )
    c_mask = (offs_c_m[:, None] < m) & (offs_c_n[None, :] < n)
    tl.store(c_ptrs, accumulator, mask=c_mask)


if __name__ == "__main__":
    torch.manual_seed(0)

    # m, k, n, size_k, size_n, p, BLOCK_M, BLOCK_P, group_m = ( 128, 64, 128, 4, 16, 10, 16, 4, 2,)
    # m, k, n, size_k, size_n, p, BLOCK_M, BLOCK_P, group_m = ( 128, 64, 100, 4, 16, 8, 16, 4, 2,)
    # m, k, n, size_k, size_n, p, BLOCK_M, BLOCK_P, group_m = 64,32,64,2,16,8,16,8,2
    # DTYPE = torch.float16

    for m, k, n, size_k, size_n, p, BLOCK_M, BLOCK_P, group_m, DTYPE in [
        # Basic test case from sparse.py
        (256, 128, 256, 8, 16, 8, 32, 8, 4, torch.float32),
        # Larger sizes
        (512, 256, 512, 16, 32, 8, 64, 4, 8, torch.float32),
        # Different data type
        (64, 32, 64, 2, 16, 8, 16, 8, 2, torch.float16),
        # M not a multiple of BLOCK_M
        (100, 64, 128, 4, 16, 8, 16, 4, 2, torch.float32),
        # P not a multiple of BLOCK_P
        (128, 64, 128, 4, 16, 10, 16, 4, 2, torch.float32),
    ]:
        print("M,K,N,size_k,size_n,P,BLOCK_M,BLOCK_P,group_m,DTYPE")
        print(
            f"{m},{k},{n},{size_k},{size_n},{p},{BLOCK_M},{BLOCK_P},{group_m},{DTYPE}"
        )
        assert BLOCK_P * size_k >= 16, (
            f"BLOCK_P * size_k must be greater than 16, got {BLOCK_P} x {size_k}"
        )
        b_values, b_indices, B_dense = create_blocksparse(
            k, n, size_k, size_n, P=p, dtype=DTYPE, seed=0
        )

        a = torch.randn((k, m), device="cuda", dtype=DTYPE).t()

        c = torch.empty((m, n), device="cuda", dtype=torch.float32)

        dense_blocksparse_mm[(triton.cdiv(m, BLOCK_M), triton.cdiv(n, size_n))](
            a_ptr=a,
            b_values_ptr=b_values,
            b_indices_ptr=b_indices,
            c_ptr=c,
            m=m,
            n=n,
            k=k,
            stride_cm=c.stride(0),
            stride_cn=c.stride(1),
            p=p,  # type: ignore
            size_k=size_k,  # type: ignore
            size_n=size_n,  # type: ignore
            block_m=BLOCK_M,  # type: ignore
            block_p=BLOCK_P,  # type: ignore
            group_m=group_m,  # type: ignore
        )
        # print("torch dense matmul:\n", torch.matmul(a, B_dense))
        # print("triton dense blocksparse mm:\n", c)
        print("Difference:\n", torch.abs(torch.matmul(a, B_dense) - c))
        print(
            "Max difference:",
            torch.max(torch.abs(torch.matmul(a, B_dense) - c)),
        )
        print(
            "Min difference:",
            torch.min(torch.abs(torch.matmul(a, B_dense) - c)),
        )
        print(
            "Mean difference:",
            torch.mean(torch.abs(torch.matmul(a, B_dense) - c)),
        )
        print(
            "Sum of differences:",
            torch.sum(torch.abs(torch.matmul(a, B_dense).float() - c)),
        )
        print(
            "All close:",
            torch.allclose(
                torch.matmul(a, B_dense).float(), c, atol=1e-5, rtol=1e-5
            ),
        )
