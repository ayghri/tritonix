import torch
import triton
import triton.language as tl
from triton.testing import do_bench


"""
Grouped Block Sparse GEMM Benchmark
===================================

Pattern: For every GROUP_SIZE (=16) contiguous 16x16 blocks along the K dimension,
only NNZ_PER_GROUP (2, 4, 8, or 12) blocks are non–zero. This is a structured block sparsity
pattern analogous to extended N:M at a 16x16 block granularity.

Storage Layout:
  Indices  : [num_col_blocks, k_group_count, NNZ_PER_GROUP]
  Values   : [num_col_blocks, k_group_count, NNZ_PER_GROUP, B_K, B_N]
Flattened in row‑major order (last dimension fastest). The kernel expects both tensors
flattened to 1-D contiguous buffers with the above logical order.

Kernel Mapping:
  Each program instance (pid_m, pid_n) computes a tile of C of shape
  (BLOCK_SIZE_M, B_N). It iterates over all K groups and, inside each group,
  iterates the NNZ_PER_GROUP non‑zero 16x16 blocks, performing one dot per block.

Limitations / Assumptions:
  - K must be divisible by (GROUP_SIZE * B_K)
  - N must be divisible by B_N
        - Block size B_K x B_N = 16 x 16
"""


DTYPE = torch.float16


@triton.jit
def grouped_block_sparse_kernel(
    a_ptr,  # [M, K]
    b_values_ptr,  # flattened values
    b_indices_ptr,  # flattened indices (int32)
    c_ptr,  # [M, N]
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_cm,
    stride_cn,
    B_K: tl.constexpr,  # 16 (K dimension of a block)
    B_N: tl.constexpr,  # 16 (N dimension of a block)
    GROUP_SIZE: tl.constexpr,  # 16 contiguous blocks in a group along K
    NNZ_PER_GROUP: tl.constexpr,  # 2,4,8
    BLOCK_SIZE_M: tl.constexpr,  # tile size along M
    GROUP_M: tl.constexpr
):

    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)

    pid_m, pid_n = tl.swizzle2d()

    m_start = pid_m * BLOCK_SIZE_M
    n_start = pid_n * B_N

    offs_m = m_start + tl.arange(0, BLOCK_SIZE_M)
    offs_n = n_start + tl.arange(0, B_N)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    accumulator = tl.zeros((BLOCK_SIZE_M, B_N), dtype=tl.float32)

    # Derived quantities
    k_blocks = K // B_K
    groups_per_col = k_blocks // GROUP_SIZE  # K group count

    # Base pointers for this column block (pid_n)
    indices_per_col = groups_per_col * NNZ_PER_GROUP
    values_per_col = indices_per_col * B_K * B_N

    col_indices_base = b_indices_ptr + pid_n * indices_per_col
    col_values_base = b_values_ptr + pid_n * values_per_col

    # Iterate over groups
    for g in range(groups_per_col):
        # Base pointer for this group's indices
        group_indices_ptr = col_indices_base + g * NNZ_PER_GROUP

        # Iterate over non‑zero blocks in the group (scalar load per block)
        for i in tl.static_range(0, NNZ_PER_GROUP):
            block_idx_in_group = tl.load(group_indices_ptr + i)
            # global block index along K dimension
            global_block_k = g * GROUP_SIZE + block_idx_in_group
            k_base = global_block_k * B_K

            offs_k = k_base + tl.arange(0, B_K)

            a_ptrs = (
                a_ptr
                + offs_m[:, None] * stride_am
                + offs_k[None, :] * stride_ak
            )
            a_tile = tl.load(a_ptrs, mask=(offs_m[:, None] < M), other=0.0)

            # Load B block (B_K, B_N)
            block_linear_index = g * NNZ_PER_GROUP + i
            b_block_base = col_values_base + block_linear_index * (B_K * B_N)
            offs_b_k = tl.arange(0, B_K)
            offs_b_n = tl.arange(0, B_N)
            b_ptrs = b_block_base + offs_b_k[:, None] * B_N + offs_b_n[None, :]
            b_block = tl.load(b_ptrs)

            accumulator += tl.dot(a_tile, b_block)

    tl.store(c_ptrs, accumulator, mask=c_mask)


@triton.jit
def grouped_block_sparse_kernel_vec(
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
    B_K: tl.constexpr,
    B_N: tl.constexpr,
    GROUP_SIZE: tl.constexpr,  # 16
    NNZ_PER_GROUP: tl.constexpr,  # 2,4,8,12
    BLOCK_SIZE_M: tl.constexpr,
):
    """Vectorized variant (group-level):
    For each group g:
        1. Load NNZ_PER_GROUP local block indices.
        2. Map to global block indices (add g * GROUP_SIZE).
        3. Gather all (NNZ_PER_GROUP * B_K) columns from A into a single tile.
        4. Load contiguous B values segment for the group.
        5. Perform one tl.dot and accumulate.
    This reduces loop overhead inside the group and enables a larger fused dot.
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    m_start = pid_m * BLOCK_SIZE_M
    n_start = pid_n * B_N
    offs_m = m_start + tl.arange(0, BLOCK_SIZE_M)
    offs_n = n_start + tl.arange(0, B_N)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    accumulator = tl.zeros((BLOCK_SIZE_M, B_N), dtype=tl.float32)

    k_blocks = K // B_K
    groups_per_col = k_blocks // GROUP_SIZE
    indices_per_col = groups_per_col * NNZ_PER_GROUP
    values_per_col = indices_per_col * B_K * B_N
    col_indices_base = b_indices_ptr + pid_n * indices_per_col
    col_values_base = b_values_ptr + pid_n * values_per_col
    offs_i = tl.arange(0, NNZ_PER_GROUP)
    b_offs = tl.arange(0, NNZ_PER_GROUP * B_K * B_N)

    for g in range(groups_per_col):
        group_indices_ptr = col_indices_base + g * NNZ_PER_GROUP
        b_group_start = col_values_base + g * NNZ_PER_GROUP * (B_K * B_N)
        b_ptrs = b_group_start + b_offs
        b_group_flat = tl.load(b_ptrs)

        local_block_indices = tl.load(group_indices_ptr + offs_i)
        global_block_indices = g * GROUP_SIZE + local_block_indices
        k_offsets_scattered = (
            global_block_indices[:, None] * B_K + tl.arange(0, B_K)[None, :]
        )
        k_offsets_flat = tl.reshape(k_offsets_scattered, (NNZ_PER_GROUP * B_K,))
        a_ptrs = (
            a_ptr
            + offs_m[:, None] * stride_am
            + k_offsets_flat[None, :] * stride_ak
        )
        a_tile = tl.load(a_ptrs, mask=(offs_m[:, None] < M), other=0.0)

        b_tile = tl.reshape(b_group_flat, (NNZ_PER_GROUP * B_K, B_N))
        accumulator += tl.dot(a_tile, b_tile)

    tl.store(c_ptrs, accumulator, mask=c_mask)


def generate_grouped_block_sparse_B(
    K,
    N,
    B_K=16,
    B_N=16,
    GROUP_SIZE=16,
    NNZ_PER_GROUP=2,
    device="cuda",
    dtype=DTYPE,
):
    assert K % (GROUP_SIZE * B_K) == 0, "K must be divisible by GROUP_SIZE*B_K"
    assert N % B_N == 0, "N must be divisible by B_N"
    k_blocks = K // B_K
    groups_per_col = k_blocks // GROUP_SIZE
    num_col_blocks = N // B_N

    values_list = []
    indices_list = []
    dense_B = torch.zeros((K, N), device=device, dtype=dtype)

    for col in range(num_col_blocks):
        for g in range(groups_per_col):
            # choose NNZ_PER_GROUP unique block indices within group
            chosen = torch.randperm(GROUP_SIZE, device=device)[:NNZ_PER_GROUP]
            chosen, _ = torch.sort(chosen)
            for idx in chosen:
                block_k_global = g * GROUP_SIZE + idx.item()
                block_vals = torch.randn((B_K, B_N), device=device, dtype=dtype)
                values_list.append(block_vals)
                indices_list.append(idx)
                # place into dense for reference
                k_row_start = block_k_global * B_K
                n_col_start = col * B_N
                dense_B[
                    k_row_start : k_row_start + B_K,
                    n_col_start : n_col_start + B_N,
                ] = block_vals

    values = torch.stack(values_list).contiguous().view(-1)
    indices = torch.tensor(
        indices_list, dtype=torch.int32, device=device
    ).contiguous()
    return values, indices, dense_B


def run_once(
    M=1024,
    K=4096,
    N=2048,
    B_K=16,
    B_N=16,
    GROUP_SIZE=16,
    NNZ_PER_GROUP=2,
    BLOCK_SIZE_M=None,  # If None choose adaptively based on NNZ_PER_GROUP
    dtype=DTYPE,
    verify=True,
    verbose_verify=False,
    vectorized=False,
):
    device = "cuda"
    A = torch.randn((M, K), device=device, dtype=dtype)
    # Make A column-major friendly (optionally); here we keep standard layout.
    A_col_major = A.t().contiguous().t()

    b_values, b_indices, B_dense = generate_grouped_block_sparse_B(
        K, N, B_K, B_N, GROUP_SIZE, NNZ_PER_GROUP, device=device, dtype=dtype
    )
    C_out = torch.empty((M, N), device=device, dtype=dtype)

    # Adaptive BLOCK_SIZE_M selection to reduce resource pressure for high NNZ
    if BLOCK_SIZE_M is None:
        if NNZ_PER_GROUP <= 4:
            bs_m = 64
        elif NNZ_PER_GROUP <= 8:
            bs_m = 64  # still fits resources empirically
        else:  # 12 or larger
            bs_m = 64  # shrink to stay within shared mem limits
    else:
        bs_m = BLOCK_SIZE_M

    grid = (triton.cdiv(M, bs_m), N // B_N)

    # First functional run (also triggers JIT compilation) -----------------
    kernel = (
        grouped_block_sparse_kernel_vec
        if vectorized
        else grouped_block_sparse_kernel
    )
    kernel[grid](
        A_col_major,
        b_values,
        b_indices,
        C_out,
        M,
        N,
        K,
        A_col_major.stride(0),
        A_col_major.stride(1),
        C_out.stride(0),
        C_out.stride(1),
        B_K=tl.constexpr(B_K),
        B_N=tl.constexpr(B_N),
        GROUP_SIZE=tl.constexpr(GROUP_SIZE),
        NNZ_PER_GROUP=tl.constexpr(NNZ_PER_GROUP),
        BLOCK_SIZE_M=tl.constexpr(bs_m),
        num_stages=5
    )

    # Correctness check BEFORE timing so timing excludes ref matmul work ---
    max_abs = float("nan")
    rel_l2 = float("nan")
    if verify:
        ref = torch.matmul(A, B_dense)
        diff = C_out - ref
        max_abs = torch.max(torch.abs(diff)).item()
        l2 = torch.norm(diff).item()
        denom = torch.norm(ref).item() + 1e-9
        rel_l2 = l2 / denom
        ok = torch.allclose(C_out, ref, atol=1e-1, rtol=0)
        if verbose_verify or not ok:
            print(
                f"[VERIFY] NNZ/G={NNZ_PER_GROUP} max_abs={max_abs:.3e} rel_l2={rel_l2:.3e} pass={ok}"
            )
        if not ok:
            raise AssertionError(
                f"Verification failed: max_abs={max_abs:.3e} rel_l2={rel_l2:.3e}"
            )

    # Benchmark
    triton_ms = do_bench(
        lambda: kernel[grid](
            A_col_major,
            b_values,
            b_indices,
            C_out,
            M,
            N,
            K,
            A_col_major.stride(0),
            A_col_major.stride(1),
            C_out.stride(0),
            C_out.stride(1),
            B_K=tl.constexpr(B_K),
            B_N=tl.constexpr(B_N),
            GROUP_SIZE=tl.constexpr(GROUP_SIZE),
            NNZ_PER_GROUP=tl.constexpr(NNZ_PER_GROUP),
            BLOCK_SIZE_M=tl.constexpr(bs_m),
        )
    )

    dense_ms = do_bench(lambda: torch.matmul(A, B_dense))

    active_ratio = NNZ_PER_GROUP / GROUP_SIZE
    sparsity_pct = 100 * (1 - active_ratio)
    speedup = float(dense_ms) / float(triton_ms)
    # print(B_dense.view(K//B_K,B_K, N//B_N,B_K).abs().sum(1).sum(2)[0][:16])
    return {
        "M": M,
        "K": K,
        "N": N,
        "NNZ_PER_GROUP": NNZ_PER_GROUP,
        "triton_ms": triton_ms,
        "dense_ms": dense_ms,
        "speedup": speedup,
        "sparsity_pct": sparsity_pct,
        "max_abs": max_abs,
        "rel_l2": rel_l2,
        "vectorized": vectorized,
        "BLOCK_SIZE_M": bs_m,
    }


def main():
    print(
        "Grouped Block Sparse GEMM (group size 16 over 16x16 blocks; NNZ=2/4/8/12)"
    )
    print("Baseline vs Vectorized")
    print(
        f"{'Mode':<10}{'NNZ/G':<6}{'Spars%':<8}{'BM':<5}{'Time(ms)':<10}{'Dense(ms)':<11}{'Speedup':<8}{'MaxAbs':<12}{'RelL2':<12}"
    )
    print("-" * 90)
    for nnz in [2, 4, 8, 12]:
        base = run_once(
            NNZ_PER_GROUP=nnz, B_N=16, GROUP_SIZE=16, vectorized=False
        )
        print(
            f"{'base':<10}{nnz:<6}{base['sparsity_pct']:<8.1f}{base['BLOCK_SIZE_M']:<5}{base['triton_ms']:<10.4f}{base['dense_ms']:<11.4f}{base['speedup']:<8.2f}{base['max_abs']:<12.3e}{base['rel_l2']:<12.3e}"
        )
        # Only run vectorized variant for power-of-two NNZ (2,4,8,16)
        if nnz & (nnz - 1) == 0:
            vec = run_once(
                NNZ_PER_GROUP=nnz, B_N=16, GROUP_SIZE=16, vectorized=True
            )
            print(
                f"{'vec':<10}{nnz:<6}{vec['sparsity_pct']:<8.1f}{vec['BLOCK_SIZE_M']:<5}{vec['triton_ms']:<10.4f}{vec['dense_ms']:<11.4f}{vec['speedup']:<8.2f}{vec['max_abs']:<12.3e}{vec['rel_l2']:<12.3e}"
            )
        else:
            print(
                f"{'vec-skip':<10}{nnz:<6}{base['sparsity_pct']:<8.1f}{base['BLOCK_SIZE_M']:<5}{'-':<10}{base['dense_ms']:<11.4f}{'-':<8}{'-':<12}{'-':<12}"
            )


if __name__ == "__main__":
    main()
