import torch
import triton
import triton.language as tl
from triton.testing import do_bench

DTYPE = torch.float16


@triton.jit
def kblock_grouped_local_n_kernel(
    a_ptr,  # [M, K]
    b_values_ptr,  # flattened values per (k_block, n_group)
    b_indices_ptr,  # flattened local indices per (k_block, n_group) (int32, values in [0, GROUP_SIZE))
    c_ptr,  # [M, N] float32 accumulation
    M,
    N,
    K,
    num_groups,  # (N / B_N) / GROUP_SIZE
    stride_am,
    stride_ak,
    stride_cm,
    stride_cn,
    B_K: tl.constexpr,
    B_N: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    NNZ_PER_GROUP: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_g = tl.program_id(2)

    k_block = pid_k
    group_id = pid_g

    m_offs = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    k_offs = k_block * B_K + tl.arange(0, B_K)

    # Load A tile (BLOCK_SIZE_M, B_K)
    a_ptrs = a_ptr + m_offs[:, None] * stride_am + k_offs[None, :] * stride_ak
    a_tile = tl.load(a_ptrs, mask=(m_offs[:, None] < M), other=0.0)

    # Offsets for indices for this (k_block, group)
    indices_offset = (k_block * num_groups + group_id) * NNZ_PER_GROUP
    indices_base = b_indices_ptr + indices_offset

    # Base pointer for values for this (k_block, group)
    values_group_offset = (k_block * num_groups + group_id) * (
        NNZ_PER_GROUP * (B_K * B_N)
    )
    values_base = b_values_ptr + values_group_offset

    n_arange = tl.arange(0, B_N)
    v_offs = tl.arange(0, B_K * B_N)

    # Iterate nonzeros inside this group
    for i in tl.static_range(0, NNZ_PER_GROUP):
        local_idx = tl.load(indices_base + i)  # in [0, GROUP_SIZE)
        # Absolute N-block number
        n_block = group_id * GROUP_SIZE + local_idx
        n_offs = n_block * B_N + n_arange
        # Load this value block
    v_ptrs = values_base + i * (B_K * B_N) + v_offs
    b_block_flat = tl.load(v_ptrs)
    b_block = tl.reshape(b_block_flat, (B_K, B_N))
    c_partial = tl.dot(a_tile, b_block)
    c_ptrs = c_ptr + m_offs[:, None] * stride_cm + n_offs[None, :] * stride_cn
    mask = (m_offs[:, None] < M) & (n_offs[None, :] < N)
    tl.atomic_add(c_ptrs, c_partial, mask=mask)


def generate_local_grouped_sparse(
    K, N, B_K, B_N, NNZ_PER_GROUP, GROUP_SIZE=16, device="cuda", dtype=DTYPE
):
    assert K % B_K == 0
    assert N % B_N == 0
    n_tiles = N // B_N
    assert n_tiles % GROUP_SIZE == 0, "N/B_N must be divisible by GROUP_SIZE"
    num_groups = n_tiles // GROUP_SIZE
    k_blocks = K // B_K

    # Storage
    indices = []  # shape (k_blocks, num_groups, NNZ_PER_GROUP)
    values = []  # list of blocks (B_K, B_N)
    dense_B = torch.zeros((K, N), device=device, dtype=dtype)

    for kb in range(k_blocks):
        for g in range(num_groups):
            # choose NNZ_PER_GROUP distinct local indices in [0, GROUP_SIZE)
            chosen_local = torch.randperm(GROUP_SIZE, device=device)[
                :NNZ_PER_GROUP
            ]
            chosen_local, _ = torch.sort(chosen_local)
            indices.append(chosen_local)
            for li in chosen_local:
                global_block = g * GROUP_SIZE + li.item()
                block_vals = torch.randn((B_K, B_N), device=device, dtype=dtype)
                values.append(block_vals)
                k_row_start = kb * B_K
                n_col_start = global_block * B_N
                dense_B[
                    k_row_start : k_row_start + B_K,
                    n_col_start : n_col_start + B_N,
                ] = block_vals

    indices_t = torch.stack(indices).to(torch.int32).contiguous().view(-1)
    values_t = torch.stack(values).contiguous().view(-1)
    return values_t, indices_t, dense_B, num_groups


def run_once(
    M=1024,
    K=4096,
    N=2048,
    B_K=16,
    B_N=16,
    GROUP_SIZE=16,
    NNZ_PER_GROUP=4,
    BLOCK_SIZE_M=128,
    dtype=DTYPE,
    verify=True,
):
    device = "cuda"
    A = torch.randn((M, K), device=device, dtype=dtype)
    b_values, b_indices, B_dense, num_groups = generate_local_grouped_sparse(
        K, N, B_K, B_N, NNZ_PER_GROUP, GROUP_SIZE, device=device, dtype=dtype
    )

    # Accumulation buffer
    C_acc = torch.zeros((M, N), device=device, dtype=torch.float32)

    grid = (triton.cdiv(M, BLOCK_SIZE_M), K // B_K, num_groups)
    kblock_grouped_local_n_kernel[grid](
        A,
        b_values,
        b_indices,
        C_acc,
        M,
        N,
        K,
        num_groups,
        A.stride(0),
        A.stride(1),
        C_acc.stride(0),
        C_acc.stride(1),
        B_K=tl.constexpr(B_K),
        B_N=tl.constexpr(B_N),
        GROUP_SIZE=tl.constexpr(GROUP_SIZE),
        NNZ_PER_GROUP=tl.constexpr(NNZ_PER_GROUP),
        BLOCK_SIZE_M=tl.constexpr(BLOCK_SIZE_M),
    )

    # Separate timing buffer to avoid multi-accumulation
    C_time = torch.zeros_like(C_acc)

    def _bench():
        C_time.zero_()
        kblock_grouped_local_n_kernel[grid](
            A,
            b_values,
            b_indices,
            C_time,
            M,
            N,
            K,
            num_groups,
            A.stride(0),
            A.stride(1),
            C_time.stride(0),
            C_time.stride(1),
            B_K=tl.constexpr(B_K),
            B_N=tl.constexpr(B_N),
            GROUP_SIZE=tl.constexpr(GROUP_SIZE),
            NNZ_PER_GROUP=tl.constexpr(NNZ_PER_GROUP),
            BLOCK_SIZE_M=tl.constexpr(BLOCK_SIZE_M),
        )

    kernel_ms = do_bench(_bench)
    dense_ms = do_bench(lambda: torch.matmul(A, B_dense))

    max_abs = float("nan")
    rel_l2 = float("nan")
    if verify:
        ref = torch.matmul(A, B_dense).float()
        diff = C_acc - ref
        max_abs = torch.max(torch.abs(diff)).item()
        rel_l2 = torch.norm(diff).item() / (torch.norm(ref).item() + 1e-9)
        if max_abs > 1e-1:
            nnz_ref = (ref != 0).sum().item()
            nnz_out = (C_acc != 0).sum().item()
            print(
                f"[DEBUG] mismatch max_abs={max_abs:.3e} rel_l2={rel_l2:.3e} nnz_out={nnz_out} nnz_ref={nnz_ref}"
            )
            # Show a small slice
            print(
                "[DEBUG] First nonzero ref col idx:",
                (ref.abs().sum(0) != 0).nonzero()[:10].view(-1).tolist(),
            )

    # Sparsity: per group density = NNZ_PER_GROUP / GROUP_SIZE
    sparsity_pct = 100 * (1 - NNZ_PER_GROUP / GROUP_SIZE)

    return {
        "kernel_ms": kernel_ms,
        "dense_ms": dense_ms,
        "speedup": float(dense_ms) / float(kernel_ms),
        "NNZ_PER_GROUP": NNZ_PER_GROUP,
        "sparsity_pct": sparsity_pct,
        "max_abs": max_abs,
        "rel_l2": rel_l2,
    }


def main():
    print("Per-Group (16 N-tiles) Block Sparse: each group has 2/4/8 nnz tiles")
    print(
        f"{'NNZ':<6}{'Spars%':<10}{'Time(ms)':<12}{'Dense(ms)':<12}{'Speedup':<10}{'MaxAbs':<12}{'RelL2':<12}"
    )
    print("-" * 82)
    for nnz in [2, 4, 8]:
        stats = run_once(NNZ_PER_GROUP=nnz)
        print(
            f"{nnz:<6}{stats['sparsity_pct']:<10.1f}{stats['kernel_ms']:<12.4f}{stats['dense_ms']:<12.4f}{stats['speedup']:<10.2f}{stats['max_abs']:<12.3e}{stats['rel_l2']:<12.3e}"
        )


if __name__ == "__main__":
    main()
