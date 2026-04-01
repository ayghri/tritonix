# Dense × Block‑Sparse MMA in PyTorch with Triton
# ------------------------------------------------
# C = A @ B, where A is dense [M,K], B is block‑sparse [K,N] with blocks [BK, BN]
#
# This file provides:
#  - A compact BSR (Block Sparse Row) container for B
#  - Triton kernels for forward, dA, and dB
#  - An autograd Function + convenience API
#  - Utilities to pack from dense + masks and to densify back
#
# Notes
# -----
# * Layouts assumed contiguous row‑major for A, C; B values packed as [nnzb, BK, BN]
# * Kernels use atomic adds on fp32 accumulators for simplicity and parallelism
# * Supported dtypes: float16, bfloat16, float32 (compute in fp32)
# * Tune BLOCK_M, BK, BN for your GPU; defaults are reasonable for Ampere/Ada
# * This is a starting point—tile sizes and num_warps/num_stages can be tuned

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

# -------------------------
# Packing: Dense -> BSR
# -------------------------


def _validate_block_dims(K: int, N: int, BK: int, BN: int):
    assert K % BK == 0 and N % BN == 0, (
        f"K={K}, N={N} must be multiples of BK={BK}, BN={BN}"
    )


def pack_bsr(
    B_dense: torch.Tensor,
    BK: int,
    BN: int,
    *,
    mask: Optional[torch.Tensor] = None,
    threshold: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert dense B [K,N] into BSR with block size [BK,BN].

    Args:
      B_dense: [K, N], contiguous, on device (cuda recommended)
      BK, BN: block size
      mask: optional boolean mask [K//BK, N//BN] selecting nonzero blocks
      threshold: if set, mark a block "nonzero" if its L2 norm >= threshold

    Returns:
      values: [nnzb, BK, BN]
      rowptr: [K//BK + 1] (int32)
      colind: [nnzb] (int32)  # block column indices
      rowids: [nnzb] (int32)  # block row index per block
    """
    assert B_dense.dim() == 2, "B must be 2D"
    K, N = B_dense.shape
    _validate_block_dims(K, N, BK, BN)
    KB = K // BK
    NB = N // BN

    if mask is None:
        # derive mask by threshold on block L2 norm
        assert threshold is not None, "Provide either mask or threshold"
        # compute norms per block
        Bd = B_dense.detach()
        norms = torch.empty(
            (KB, NB), device=B_dense.device, dtype=torch.float32
        )
        for rb in range(KB):
            k0 = rb * BK
            k1 = k0 + BK
            for cb in range(NB):
                n0 = cb * BN
                n1 = n0 + BN
                block = Bd[k0:k1, n0:n1]
                norms[rb, cb] = torch.linalg.vector_norm(block.float())
        mask = norms >= float(threshold)
    else:
        assert mask.shape == (KB, NB)
        mask = mask.to(torch.bool)

    # Build CSR‑like structure over block rows
    row_counts = mask.sum(dim=1)
    nnzb = int(row_counts.sum().item())
    rowptr = torch.empty(KB + 1, device=B_dense.device, dtype=torch.int32)
    rowptr[0] = 0
    torch.cumsum(row_counts.to(torch.int32), dim=0, out=rowptr[1:])

    colind = torch.empty(nnzb, device=B_dense.device, dtype=torch.int32)
    rowids = torch.empty(nnzb, device=B_dense.device, dtype=torch.int32)
    values = torch.empty(
        (nnzb, BK, BN), device=B_dense.device, dtype=B_dense.dtype
    )

    write_ptr = 0
    for rb in range(KB):
        k0 = rb * BK
        k1 = k0 + BK
        cols = torch.nonzero(mask[rb], as_tuple=False).flatten().tolist()
        for cb in cols:
            n0 = cb * BN
            n1 = n0 + BN
            values[write_ptr].copy_(B_dense[k0:k1, n0:n1])
            colind[write_ptr] = int(cb)
            rowids[write_ptr] = int(rb)
            write_ptr += 1

    return (
        values.contiguous(),
        rowptr.contiguous(),
        colind.contiguous(),
        rowids.contiguous(),
    )


def bsr_to_dense(
    values: torch.Tensor,
    rowptr: torch.Tensor,
    colind: torch.Tensor,
    BK: int,
    BN: int,
    K: int,
    N: int,
) -> torch.Tensor:
    """Materialize a dense [K,N] from BSR pieces."""
    B = torch.zeros((K, N), device=values.device, dtype=values.dtype)
    KB = K // BK
    for rb in range(KB):
        start = int(rowptr[rb].item())
        end = int(rowptr[rb + 1].item())
        for p in range(start, end):
            cb = int(colind[p].item())
            k0 = rb * BK
            n0 = cb * BN
            B[k0 : k0 + BK, n0 : n0 + BN] = values[p]
    return B


# -----------------------------------
# BSR -> CSC (for faster forward)
# -----------------------------------

def bsr_to_csc(values: torch.Tensor, rowids: torch.Tensor, colind: torch.Tensor, BK: int, BN: int, K: int, N: int):
    """Reorder blocks into column-major (CSC) structure.

    Returns:
      values_csc: [nnzb, BK, BN] reordered so blocks with same column are contiguous
      colptr: int32 [NB+1]
      rowind: int32 [nnzb] row block index for each block in column order
    """
    if values.numel() == 0:
        NB = N // BN
        colptr = torch.zeros(NB + 1, device=values.device, dtype=torch.int32)
        rowind = torch.empty(0, device=values.device, dtype=torch.int32)
        return values, colptr, rowind
    # KB = K // BK  # not used here
    NB = N // BN
    nnzb = values.shape[0]
    # build counts per column
    counts = torch.zeros(NB, device=values.device, dtype=torch.int32)
    counts.scatter_add_(0, colind, torch.ones_like(colind, dtype=torch.int32))
    colptr = torch.empty(NB + 1, device=values.device, dtype=torch.int32)
    colptr[0] = 0
    torch.cumsum(counts, 0, out=colptr[1:])
    # prefix sums gives insertion offsets
    write_ptr = colptr[:-1].clone()
    rowind = torch.empty_like(colind)
    order = torch.empty_like(colind)
    for idx in range(nnzb):
        cb = int(colind[idx].item())
        dst = int(write_ptr[cb].item())
        rowind[dst] = rowids[idx]
        order[dst] = idx
        write_ptr[cb] += 1
    values_csc = values[order]
    return values_csc, colptr, rowind


# -----------------------------------
# Triton Kernels (forward / backward)
# -----------------------------------


@triton.jit
def _dense_bsr_forward(
    A_ptr,  # *fp16/bf16/fp32  [M,K]
    Bvals_ptr,  # *fp16/bf16/fp32  [nnzb, BK, BN] packed
    Brow_ptr,  # *int32           [nnzb]  block row id per block
    Bcol_ptr,  # *int32           [nnzb]  block col id per block
    Cacc_ptr,  # *fp32            [M,N] accumulator (zeros on entry)
    M: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    nnzb: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)

    # Guard tiles
    m0 = pid_m * BLOCK_M
    m_mask = m0 + tl.arange(0, BLOCK_M)
    valid_m = m_mask < M

    if pid_b >= nnzb:
        return

    rb = tl.load(Brow_ptr + pid_b).to(tl.int32)
    cb = tl.load(Bcol_ptr + pid_b).to(tl.int32)

    k0 = rb * BK
    n0 = cb * BN

    offs_m = m0 + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BK)
    offs_n = n0 + tl.arange(0, BN)

    # Pointers for A tile [BLOCK_M, BK]
    A_ptrs = A_ptr + (offs_m[:, None] * K + (k0 + offs_k)[None, :])
    # Pointers for B block [BK, BN]
    B_ptrs = Bvals_ptr + (
        pid_b * BK * BN + offs_k[:, None] * BN + tl.arange(0, BN)[None, :]
    )
    # Pointers for C tile [BLOCK_M, BN]
    C_ptrs = Cacc_ptr + (offs_m[:, None] * N + offs_n[None, :])

    a = tl.load(
        A_ptrs, mask=valid_m[:, None] & ((k0 + offs_k)[None, :] < K), other=0.0
    )
    b = tl.load(B_ptrs)
    acc = tl.zeros((BLOCK_M, BN), dtype=tl.float32)
    # Compute partial product: [Mtile,BK] @ [BK,BN]
    acc += tl.dot(a.to(tl.float32), b.to(tl.float32))

    # Atomic accumulate into C
    cmask = valid_m[:, None] & (offs_n[None, :] < N)
    tl.atomic_add(C_ptrs, acc, mask=cmask)


@triton.jit
def _dense_bsr_backward_dA(
    dC_ptr,  # *fp16/bf16/fp32 [M,N]
    Bvals_ptr,  # *fp16/bf16/fp32 [nnzb, BK, BN]
    Brow_ptr,  # *int32          [nnzb]
    Bcol_ptr,  # *int32          [nnzb]
    dAacc_ptr,  # *fp32           [M,K]
    M: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    nnzb: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)

    m0 = pid_m * BLOCK_M
    if pid_b >= nnzb:
        return

    rb = tl.load(Brow_ptr + pid_b).to(tl.int32)
    cb = tl.load(Bcol_ptr + pid_b).to(tl.int32)

    k0 = rb * BK
    n0 = cb * BN

    offs_m = m0 + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BK)
    offs_n = n0 + tl.arange(0, BN)

    dC_ptrs = dC_ptr + (offs_m[:, None] * N + offs_n[None, :])
    B_ptrs = Bvals_ptr + (
        pid_b * BK * BN + offs_k[:, None] * BN + tl.arange(0, BN)[None, :]
    )
    dA_ptrs = dAacc_ptr + (offs_m[:, None] * K + (k0 + offs_k)[None, :])

    dc = tl.load(
        dC_ptrs, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N), other=0.0
    )
    b = tl.load(B_ptrs)

    # dA += dC @ B^T
    acc = tl.dot(dc.to(tl.float32), tl.trans(b.to(tl.float32)))  # [BLOCK_M, BK]
    tl.atomic_add(
        dA_ptrs, acc, mask=(offs_m[:, None] < M) & ((k0 + offs_k)[None, :] < K)
    )


@triton.jit
def _dense_bsr_backward_dB(
    A_ptr,  # *fp16/bf16/fp32 [M,K]
    dC_ptr,  # *fp16/bf16/fp32 [M,N]
    Brow_ptr,  # *int32          [nnzb]
    Bcol_ptr,  # *int32          [nnzb]
    dBacc_ptr,  # *fp32           [nnzb, BK, BN]
    M: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    nnzb: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    # One program per (m‑tile, block). Accumulate into dB with atomics.
    pid_m = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)

    if pid_b >= nnzb:
        return

    m0 = pid_m * BLOCK_M
    rb = tl.load(Brow_ptr + pid_b).to(tl.int32)
    cb = tl.load(Bcol_ptr + pid_b).to(tl.int32)

    k0 = rb * BK
    n0 = cb * BN

    offs_m = m0 + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BK)
    offs_n = tl.arange(0, BN)

    # Tiles
    A_ptrs = A_ptr + (
        offs_m[:, None] * K + (k0 + offs_k)[None, :]
    )  # [Mtile,BK]
    dC_ptrs = dC_ptr + (
        offs_m[:, None] * N + (n0 + offs_n)[None, :]
    )  # [Mtile,BN]

    a = tl.load(
        A_ptrs,
        mask=(offs_m[:, None] < M) & ((k0 + offs_k)[None, :] < K),
        other=0.0,
    )
    dc = tl.load(
        dC_ptrs,
        mask=(offs_m[:, None] < M) & ((n0 + offs_n)[None, :] < N),
        other=0.0,
    )

    # dB += A^T @ dC  -> [BK,BN]
    acc = tl.dot(tl.trans(a.to(tl.float32)), dc.to(tl.float32))

    dB_ptrs = dBacc_ptr + (
        pid_b * BK * BN + offs_k[:, None] * BN + offs_n[None, :]
    )
    tl.atomic_add(dB_ptrs, acc)


# ------------------------
# Autograd + User API
# ------------------------


class _DenseBsrMM(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        A: torch.Tensor,
        values: torch.Tensor,
        rowptr: torch.Tensor,
        colind: torch.Tensor,
        rowids: torch.Tensor,
        BK: int,
        BN: int,
        BLOCK_M: int,
    ):
        assert A.is_cuda and values.is_cuda, "Use CUDA tensors"
        M, K = A.shape
    # KB = K // BK  # unused
        # Rebuild N from metadata
        max_cb = int(colind.max().item()) if colind.numel() > 0 else 0
        N = (max_cb + 1) * BN
        nnzb = values.shape[0]

        # fp32 accumulator for stable atomics
        Cacc = torch.zeros((M, N), device=A.device, dtype=torch.float32)

        grid = (triton.cdiv(M, BLOCK_M), nnzb)
        _dense_bsr_forward[grid](
            A,
            values,
            rowids,
            colind,
            Cacc,
            M,
            K,
            N,
            nnzb,
            BK,
            BN,
            BLOCK_M,
            num_warps=4,
            num_stages=2,
        )

        C = Cacc.to(A.dtype)
        ctx.save_for_backward(A, values, rowptr, colind, rowids)
        ctx.dims = (M, K, N, BK, BN, BLOCK_M)
        return C

    @staticmethod
    def backward(ctx, dC: torch.Tensor):  # PyTorch expects signature (ctx, *grad_outputs)
        A, values, rowptr, colind, rowids = ctx.saved_tensors
        M, K, N, BK, BN, BLOCK_M = ctx.dims
        nnzb = values.shape[0]

        # dA
        dAacc = torch.zeros((M, K), device=A.device, dtype=torch.float32)
        grid = (triton.cdiv(M, BLOCK_M), nnzb)
        _dense_bsr_backward_dA[grid](
            dC,
            values,
            rowids,
            colind,
            dAacc,
            M,
            K,
            N,
            nnzb,
            BK,
            BN,
            BLOCK_M,
            # launch meta left default if unsupported
        )
        dA = dAacc.to(A.dtype)

        # dB
        dBacc = torch.zeros_like(values, dtype=torch.float32)
        _dense_bsr_backward_dB[grid](
            A,
            dC,
            rowids,
            colind,
            dBacc,
            M,
            K,
            N,
            nnzb,
            BK,
            BN,
            BLOCK_M,
            # launch meta left default if unsupported
        )
        dB = dBacc.to(values.dtype)

        return dA, dB, None, None, None, None, None, None


## (Experimental CSC kernel removed due to issues; keeping atomic forward only for now)


@dataclass
class BsrMatrix:
    values: torch.Tensor  # [nnzb, BK, BN]
    rowptr: torch.Tensor  # [KB+1]
    colind: torch.Tensor  # [nnzb]
    rowids: torch.Tensor  # [nnzb]
    BK: int
    BN: int
    K: int
    N: int

    @staticmethod
    def from_dense(
        B: torch.Tensor,
        BK: int,
        BN: int,
        *,
        mask: Optional[torch.Tensor] = None,
        threshold: Optional[float] = None,
    ) -> "BsrMatrix":
        K, N = B.shape
        values, rowptr, colind, rowids = pack_bsr(
            B.contiguous(), BK, BN, mask=mask, threshold=threshold
        )
        max_cb = int(colind.max().item()) if colind.numel() > 0 else 0
        Nrec = (max_cb + 1) * BN
        return BsrMatrix(values, rowptr, colind, rowids, BK, BN, K, Nrec)

    def to_dense(self) -> torch.Tensor:
        return bsr_to_dense(
            self.values,
            self.rowptr,
            self.colind,
            self.BK,
            self.BN,
            self.K,
            self.N,
        )

    def to(self, device=None, dtype=None) -> "BsrMatrix":
        return BsrMatrix(
            self.values.to(device=device, dtype=dtype),
            self.rowptr.to(device=device),
            self.colind.to(device=device),
            self.rowids.to(device=device),
            self.BK,
            self.BN,
            self.K,
            self.N,
        )

    def mm(self, A: torch.Tensor, *, block_m: int = 128) -> torch.Tensor:
        """Compute A @ B where A is [M,K] dense."""
        assert A.shape[1] == self.K
    out = _DenseBsrMM.apply(
            A.contiguous(),
            self.values,
            self.rowptr,
            self.colind,
            self.rowids,
            self.BK,
            self.BN,
            block_m,
        )
    return out

    # (mm_csc removed)


# ------------------------
# Quick self‑test / usage
# ------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda"

    M, K, N = 1024, 2048, 1536
    BK, BN = 32, 32

    A = torch.randn(M, K, device=device, dtype=torch.float16)
    B = torch.randn(K, N, device=device, dtype=torch.float16)

    # Build a random 20% density mask over blocks
    KB, NB = K // BK, N // BN
    density = 0.20
    mask = torch.rand(KB, NB, device=device) < density

    B_bsr = BsrMatrix.from_dense(B, BK, BN, mask=mask)

    # Forward
    C = B_bsr.mm(A)

    # Verify against dense (materialized) result
    B_dense_masked = B_bsr.to_dense()
    C_ref = A @ B_dense_masked
    max_err = (C.float() - C_ref.float()).abs().max().item()
    print(f"max |C - C_ref| = {max_err:.4e}")

    # Backward check (finite sample)
    A.requires_grad_(True)
    values = B_bsr.values.clone().detach().requires_grad_(True)
    rowptr = B_bsr.rowptr
    colind = B_bsr.colind
    rowids = B_bsr.rowids

    C2 = _DenseBsrMM.apply(A, values, rowptr, colind, rowids, BK, BN, 128)
    # loss = C2.float().pow(2).mean()
    # loss.backward()
    # print(
    #     f"dA norm: {A.grad.norm().item():.3e}, dB norm: {values.grad.norm().item():.3e}"
    # )

    # --------------------------------------------------
    # Benchmark: dense_bsr_forward vs torch dense matmul
    # --------------------------------------------------
    # We benchmark forward pass only (no autograd) for a target sparsity >= 0.5
    # Sparsity here means fraction of zero blocks.
    print("\n[Benchmark] dense_bsr_forward (atomic) vs torch.matmul (masked dense)")

    # Parameters (feel free to tweak)
    M, K, N = 2048, 4096, 2048
    BK, BN = 32, 32
    SPARSITY = 0.75  # >= 0.5 as requested (fraction of zero blocks)
    assert 0.5 <= SPARSITY < 1.0
    BLOCK_DENSITY = 1.0 - SPARSITY

    A = torch.randn(M, K, device=device, dtype=torch.float16)
    B = torch.randn(K, N, device=device, dtype=torch.float16)

    KB, NB = K // BK, N // BN
    mask = (torch.rand(KB, NB, device=device) < BLOCK_DENSITY)

    B_bsr = BsrMatrix.from_dense(B, BK, BN, mask=mask)
    nnzb = B_bsr.values.shape[0]
    nnz_elems = nnzb * BK * BN

    # Materialize masked dense for reference (zeroed blocks)
    B_masked_dense = B_bsr.to_dense().to(B.dtype)

    # Helper timing util
    def _time(fn, iters=50, warmup=10):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end) / iters  # average ms
        return ms

    with torch.no_grad():
        C_ref = A @ B_masked_dense
        C_sp = B_bsr.mm(A)
        err = (C_ref.float() - C_sp.float()).abs().max().item()
        print(f"Accuracy max|C_sp - C_ref| = {err:.3e}")
        ms_sparse = _time(lambda: B_bsr.mm(A))
        ms_dense = _time(lambda: A @ B_masked_dense)

    # FLOPs
    dense_flops = 2.0 * M * K * N  # canonical dense GEMM count
    sparse_flops_effective = 2.0 * M * nnz_elems  # only nonzero elements

    gflops_sparse_effective = sparse_flops_effective / (ms_sparse * 1e6)
    gflops_sparse_vs_dense_equivalent = dense_flops / (ms_sparse * 1e6)
    gflops_dense = dense_flops / (ms_dense * 1e6)

    print("--- Benchmark Results ---")
    print(f"Shape: A=({M},{K})  B=({K},{N})  Block=({BK},{BN})")
    print(f"Block density: {BLOCK_DENSITY*100:.1f}%  (sparsity {SPARSITY*100:.1f}%)  nnz_blocks={nnzb}  nnz_elems={nnz_elems}")
    print(f"Sparse (atomic): {ms_sparse:.3f} ms | Eff GFLOP/s: {gflops_sparse_effective:.2f} | Eqv GFLOP/s: {gflops_sparse_vs_dense_equivalent:.2f}")
    print(f"Dense masked:   {ms_dense:.3f} ms | Dense GFLOP/s: {gflops_dense:.2f}")
    print(f"Speedup (dense / sparse) = {ms_dense / ms_sparse:.2f}x")

    # NOTE: For higher sparsity (e.g., 75%), adjust SPARSITY above.
