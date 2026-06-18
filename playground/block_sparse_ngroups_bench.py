import torch
import triton
import triton.language as tl
from triton.testing import do_bench

DTYPE = torch.float16

@triton.jit
def kblock_grouped_along_n_kernel(
    a_ptr,            # [M, K]
    b_values_ptr,     # flattened values per K-block
    b_indices_ptr,    # flattened N-block indices per K-block (int32)
    c_ptr,            # [M, N] (float32 accumulation)
    M, N, K,
    stride_am, stride_ak,
    stride_cm, stride_cn,
    B_K: tl.constexpr,
    B_N: tl.constexpr,
    NNZ_PER_GROUP: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    k_block = pid_k
    m_offs = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    k_offs = k_block * B_K + tl.arange(0, B_K)

    # Load A tile (BLOCK_SIZE_M, B_K)
    a_ptrs = a_ptr + m_offs[:, None] * stride_am + k_offs[None, :] * stride_ak
    a_tile = tl.load(a_ptrs, mask=(m_offs[:, None] < M), other=0.0)

    # Load indices for this K-block (NNZ_PER_GROUP is guaranteed in {2,4,8})
    indices_base = b_indices_ptr + k_block * NNZ_PER_GROUP
    # We'll scalar-load indices inside the loop to avoid tensor indexing issues

    # Base pointer to values for this K-block
    block_elems = B_K * B_N  # 256 for 16x16
    values_base = b_values_ptr + k_block * (NNZ_PER_GROUP * block_elems)

    n_arange = tl.arange(0, B_N)
    v_offs = tl.arange(0, B_K * B_N)
    # Loop over each present N-block
    for i in tl.static_range(0, NNZ_PER_GROUP):
        n_block = tl.load(indices_base + i)
        n_offs = n_block * B_N + n_arange
        # Load this (B_K, B_N) block's values
        v_ptrs = values_base + i * block_elems + v_offs
        b_block_flat = tl.load(v_ptrs)
        b_block = tl.reshape(b_block_flat, (B_K, B_N))
        # Matmul contribution
        c_partial = tl.dot(a_tile, b_block)
        c_ptrs = c_ptr + m_offs[:, None] * stride_cm + n_offs[None, :] * stride_cn
        mask = (m_offs[:, None] < M) & (n_offs[None, :] < N)
        tl.atomic_add(c_ptrs, c_partial, mask=mask)


def generate_kblock_sparse_along_n(K, N, B_K, B_N, NNZ_PER_GROUP, device="cuda", dtype=DTYPE):
    assert K % B_K == 0
    assert N % B_N == 0
    k_blocks = K // B_K
    n_blocks = N // B_N
    # Indices: [k_blocks, NNZ_PER_GROUP]
    indices = []
    values = []
    dense_B = torch.zeros((K, N), device=device, dtype=dtype)
    for kb in range(k_blocks):
        chosen = torch.randperm(n_blocks, device=device)[:NNZ_PER_GROUP]
        chosen, _ = torch.sort(chosen)
        indices.append(chosen)
        for j, nb in enumerate(chosen):
            block_vals = torch.randn((B_K, B_N), device=device, dtype=dtype)
            values.append(block_vals)
            k_row_start = kb * B_K
            n_col_start = nb * B_N
            dense_B[k_row_start:k_row_start+B_K, n_col_start:n_col_start+B_N] = block_vals
    indices_t = torch.stack(indices).to(torch.int32).contiguous().view(-1)
    values_t = torch.stack(values).contiguous().view(-1)
    return values_t, indices_t, dense_B


def run_once(
    M=1024,
    K=4096,
    N=2048,
    B_K=16,
    B_N=16,
    NNZ_PER_GROUP=4,
    BLOCK_SIZE_M=128,
    dtype=DTYPE,
    verify=True,
):
    device = "cuda"
    A = torch.randn((M, K), device=device, dtype=dtype)
    b_values, b_indices, B_dense = generate_kblock_sparse_along_n(K, N, B_K, B_N, NNZ_PER_GROUP, device=device, dtype=dtype)
    # Float32 accumulation buffer for correctness
    C_acc = torch.zeros((M, N), device=device, dtype=torch.float32)

    grid = (triton.cdiv(M, BLOCK_SIZE_M), K // B_K)
    kblock_grouped_along_n_kernel[grid](
        A,
        b_values,
        b_indices,
        C_acc,
        M, N, K,
    A.stride(0), A.stride(1),
        C_acc.stride(0), C_acc.stride(1),
    B_K=tl.constexpr(B_K), B_N=tl.constexpr(B_N),
    NNZ_PER_GROUP=tl.constexpr(NNZ_PER_GROUP), BLOCK_SIZE_M=tl.constexpr(BLOCK_SIZE_M),
    )

    # Timing buffer (avoid accumulating multiple times into correctness buffer)
    C_acc_timing = torch.zeros_like(C_acc)
    def _bench_kernel():
        C_acc_timing.zero_()
        kblock_grouped_along_n_kernel[grid](
            A, b_values, b_indices, C_acc_timing,
            M, N, K,
            A.stride(0), A.stride(1),
            C_acc_timing.stride(0), C_acc_timing.stride(1),
            B_K=tl.constexpr(B_K), B_N=tl.constexpr(B_N),
            NNZ_PER_GROUP=tl.constexpr(NNZ_PER_GROUP), BLOCK_SIZE_M=tl.constexpr(BLOCK_SIZE_M),
        )
    kernel_ms = do_bench(_bench_kernel)
    dense_ms = do_bench(lambda: torch.matmul(A, B_dense))

    max_abs = float('nan')
    rel_l2 = float('nan')
    if verify:
        ref = torch.matmul(A, B_dense).float()
        diff = C_acc - ref
        max_abs = torch.max(torch.abs(diff)).item()
        rel_l2 = torch.norm(diff).item() / (torch.norm(ref).item() + 1e-9)
        assert torch.allclose(C_acc, ref, atol=1e-1, rtol=0), f"Mismatch max_abs={max_abs:.3e} rel_l2={rel_l2:.3e}"

    sparsity_pct = 100 * (1 - NNZ_PER_GROUP / (N // B_N))
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
    print("K-Block Sparse with N-grouping (each K-block has NNZ selected N-blocks)")
    print(f"{'NNZ':<6}{'Spars%':<10}{'Time(ms)':<12}{'Dense(ms)':<12}{'Speedup':<10}{'MaxAbs':<12}{'RelL2':<12}")
    print('-'*80)
    for nnz in [2,4,8]:
        stats = run_once(NNZ_PER_GROUP=nnz)
        print(f"{nnz:<6}{stats['sparsity_pct']:<10.1f}{stats['kernel_ms']:<12.4f}{stats['dense_ms']:<12.4f}{stats['speedup']:<10.2f}{stats['max_abs']:<12.3e}{stats['rel_l2']:<12.3e}")


if __name__ == "__main__":
    main()
