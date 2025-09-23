import time
import torch
import triton
import triton.language as tl
from torch import nn
from torch.autograd import Function


# ---------------------------
# Forward kernel (A row-major)
# ---------------------------
@triton.jit
def _fwd_rm_lut(
    a_ptr,  # [M, K], row-major
    b_packed_ptr,  # flat packed chunks
    c_ptr,  # [M, N], row- or col-major (pass strides)
    ck0_ptr,  # int32 [total_chunks]  (K-block start)
    cnb_ptr,  # int32 [total_chunks]  (#blocks in chunk <= BLOCK_P)
    cboff_ptr,  # int32 [total_chunks]  (element offset into b_packed)
    cstarts_ptr,  # int32 [num_col_blocks+1] (prefix sum over chunks per col-block)
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_cm,
    stride_cn,
    B_K: tl.constexpr,
    B_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_P: tl.constexpr,
    MAX_CHUNKS_PER_COL: tl.constexpr,
    ACC_DTYPE: tl.constexpr = tl.constexpr(tl.float32),
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # pid_m, pid_n = tl.swizzle2d(
    #     pid_m, pid_n, tl.cdiv(M, BLOCK_M), tl.cdiv(N, B_N), 16
    # )  # type: ignore

    m0 = pid_m * BLOCK_M
    n0 = pid_n * B_N
    cb = pid_n

    offs_m = m0 + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    col_start = tl.load(cstarts_ptr + cb, mask=True, other=0)
    col_end = tl.load(cstarts_ptr + cb + 1, mask=True, other=0)
    n_chunks = col_end - col_start

    K_CHUNK: tl.constexpr = BLOCK_P * B_K
    offs_k = tl.arange(0, K_CHUNK)
    offs_n = tl.arange(0, B_N)

    acc = tl.zeros((BLOCK_M, B_N), dtype=ACC_DTYPE)

    for rel in range(0, MAX_CHUNKS_PER_COL):
        active = rel < n_chunks
        idx = col_start + rel

        k0_blk = tl.load(ck0_ptr + idx, mask=active, other=0)
        nb = tl.load(cnb_ptr + idx, mask=active, other=0)
        boff = tl.load(cboff_ptr + idx, mask=active, other=0)

        valid_k = offs_k < (nb * B_K)
        k0 = k0_blk * B_K

        # A tile [BLOCK_M, K_CHUNK]
        a_ptrs = (
            a_ptr
            + (offs_m[:, None] * stride_am)
            + ((k0 + offs_k)[None, :] * stride_ak)
        )
        a_tile = tl.load(
            a_ptrs,
            mask=(mask_m[:, None] & valid_k[None, :] & active),
            other=0.0,
        )

        # B tile [(K_CHUNK), B_N] contiguous
        b_flat_mask = tl.reshape(
            valid_k[:, None] & (offs_n[None, :] < B_N), (K_CHUNK * B_N,)
        )
        b_flat = tl.load(
            b_packed_ptr + boff + tl.arange(0, K_CHUNK * B_N),
            mask=(b_flat_mask & active),
            other=0.0,
        )
        b_tile = tl.reshape(b_flat, (K_CHUNK, B_N))

        acc = tl.dot(a_tile, b_tile, acc)

    c_ptrs = (
        c_ptr
        + (offs_m[:, None] * stride_cm)
        + ((n0 + offs_n)[None, :] * stride_cn)
    )
    tl.store(
        c_ptrs,
        acc.to(c_ptr.dtype.element_ty),
        mask=(mask_m[:, None] & ((n0 + offs_n)[None, :] < N)),
    )


# -----------------------------------
# dA kernel without packing B^T: (m,k)
# -----------------------------------
@triton.jit
def _dA_gather_kchunk(
    gout_ptr,  # [M, N] dC, row-major
    b_packed_ptr,  # packed B values (same as forward)
    dA_ptr,  # [M, K], row- or col-major (pass strides)
    # chunk meta (by chunk id)
    ck0_ptr,  # int32 [total_chunks]  K-block start of chunk
    cnb_ptr,  # int32 [total_chunks]  #blocks in chunk (<= BLOCK_P)
    cboff_ptr,  # int32 [total_chunks]  element offset into b_packed
    ccol_ptr,  # int32 [total_chunks]  column-block index cb
    # k-chunk -> list of chunk-ids
    klist_k0_ptr,  # int32 [num_kchunks]   k0_block value for each k-chunk
    klist_starts_ptr,  # int32 [num_kchunks+1] prefix sum over chunk-ids per k-chunk
    klist_chunk_ids_ptr,  # int32 [total_chunks]  flattened lists of chunk-ids
    M,
    N,
    K,
    stride_gm,
    stride_gn,
    stride_dm,
    stride_dk,
    B_K: tl.constexpr,
    B_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_P: tl.constexpr,
    MAX_CHUNKS_PER_K: tl.constexpr,
    ACC_DTYPE: tl.constexpr = tl.constexpr(tl.float32),
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    # tile
    m0 = pid_m * BLOCK_M
    offs_m = m0 + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    K_CHUNK = BLOCK_P * B_K
    offs_k = tl.arange(0, K_CHUNK)
    offs_n = tl.arange(0, B_N)

    # which k-chunk?
    k0_blk = tl.load(klist_k0_ptr + pid_k)
    k0 = k0_blk * B_K

    start = tl.load(klist_starts_ptr + pid_k)
    end = tl.load(klist_starts_ptr + pid_k + 1)
    n_items = end - start

    acc = tl.zeros((BLOCK_M, K_CHUNK), dtype=ACC_DTYPE)

    for rel in range(0, MAX_CHUNKS_PER_K):
        active = rel < n_items
        cid = tl.load(klist_chunk_ids_ptr + start + rel, mask=active, other=0)

        nb = tl.load(cnb_ptr + cid, mask=active, other=0)
        boff = tl.load(cboff_ptr + cid, mask=active, other=0)
        cb = tl.load(ccol_ptr + cid, mask=active, other=0)

        valid_k = offs_k < (nb * B_K)
        n0 = cb * B_N

        # dC tile [BLOCK_M, B_N]
        g_ptrs = (
            gout_ptr
            + (offs_m[:, None] * stride_gm)
            + ((n0 + offs_n)[None, :] * stride_gn)
        )
        g_tile = tl.load(
            g_ptrs,
            mask=(mask_m[:, None] & ((n0 + offs_n)[None, :] < N) & active),
            other=0.0,
        )

        # B chunk [(K_CHUNK), B_N], then transpose in registers
        b_flat_mask = tl.reshape(
            valid_k[:, None] & (offs_n[None, :] < B_N), (K_CHUNK * B_N,)
        )
        b_flat = tl.load(
            b_packed_ptr + boff + tl.arange(0, K_CHUNK * B_N),
            mask=(b_flat_mask & active),
            other=0.0,
        )
        b_tile = tl.reshape(b_flat, (K_CHUNK, B_N))

        # acc += dC_tile @ (B_chunk)^T  → [BLOCK_M, K_CHUNK]
        acc += tl.dot(g_tile, tl.trans(b_tile))

    # write dA tile
    dA_ptrs = (
        dA_ptr
        + (offs_m[:, None] * stride_dm)
        + ((k0 + offs_k)[None, :] * stride_dk)
    )
    tl.store(
        dA_ptrs,
        acc.to(dA_ptr.dtype.element_ty),
        mask=(mask_m[:, None] & ((k0 + offs_k)[None, :] < K)),
    )


# ------------------------------------------
# dB kernel: one CTA per chunk (reduce over M)
# ------------------------------------------
@triton.jit
def _dB_per_chunk(
    a_ptr,
    gout_ptr,
    bgrad_ptr,
    ck0_ptr,
    cnb_ptr,
    cboff_ptr,
    ccol_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_gm,
    stride_gn,
    total_chunks,
    B_K: tl.constexpr,
    B_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_P: tl.constexpr,
    ACC_DTYPE: tl.constexpr = tl.float32,  # type: ignore
):
    cid = tl.program_id(0)
    if cid >= total_chunks:
        return

    k0_blk = tl.load(ck0_ptr + cid)
    nb = tl.load(cnb_ptr + cid)
    boff = tl.load(cboff_ptr + cid)
    cb = tl.load(ccol_ptr + cid)

    K_CHUNK = BLOCK_P * B_K
    offs_k = tl.arange(0, K_CHUNK)
    offs_n = tl.arange(0, B_N)
    valid_k = offs_k < (nb * B_K)
    n0 = cb * B_N
    k0 = k0_blk * B_K

    acc = tl.zeros((K_CHUNK, B_N), dtype=ACC_DTYPE)

    for m0 in range(0, M, BLOCK_M):
        offs_m = m0 + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M

        a_ptrs = (
            a_ptr
            + (offs_m[:, None] * stride_am)
            + ((k0 + offs_k)[None, :] * stride_ak)
        )
        a_tile = tl.load(
            a_ptrs, mask=(mask_m[:, None] & valid_k[None, :]), other=0.0
        )

        g_ptrs = (
            gout_ptr
            + (offs_m[:, None] * stride_gm)
            + ((n0 + offs_n)[None, :] * stride_gn)
        )
        g_tile = tl.load(
            g_ptrs,
            mask=(mask_m[:, None] & ((n0 + offs_n)[None, :] < N)),
            other=0.0,
        )

        acc += tl.dot(tl.trans(a_tile), g_tile)

    b_ptrs = bgrad_ptr + boff + tl.arange(0, K_CHUNK * B_N)
    b_mask_flat = tl.reshape(
        valid_k[:, None] & (offs_n[None, :] < B_N), (K_CHUNK * B_N,)
    )
    tl.store(
        b_ptrs,
        tl.reshape(acc, (K_CHUNK * B_N,)).to(bgrad_ptr.dtype.element_ty),
        mask=b_mask_flat,
    )


# --------------------------
# Packing & metadata builders
# --------------------------
@torch.no_grad()
def pack_bs_colblocks_rowmajor(
    row_block_idx,  # list[Tensor(nnz_j,)] of K-block ids per column-block j
    values,  # list[Tensor(nnz_j, B_K, B_N)]
    B_K: int,
    B_N: int,
    BLOCK_P: int,
    align_elems: int = 64,
    device=None,
    dtype=None,
):
    assert len(row_block_idx) == len(values)
    if device is None:
        device = values[0].device
    if dtype is None:
        dtype = values[0].dtype

    def pad_to(n, m):
        return (m - (n % m)) % m

    b_parts = []
    ck0 = []
    cnb = []
    cboff = []
    ccol = []
    cstarts = [0]
    elems = 0

    num_cols = len(values)
    for j in range(num_cols):
        idx = row_block_idx[j].to(torch.int64)
        val = values[j]
        if idx.numel() == 0:
            cstarts.append(cstarts[-1])
            continue

        order = torch.argsort(idx)
        idx_s = idx[order]
        val_s = val[order]

        # find consecutive runs
        runs = []
        k_prev = int(idx_s[0])
        start = k_prev
        rlen = 1
        for t in range(1, idx_s.numel()):
            k = int(idx_s[t])
            if k == k_prev + 1:
                rlen += 1
            else:
                runs.append((start, rlen))
                start = k
                rlen = 1
            k_prev = k
        runs.append((start, rlen))

        pos = {int(k): i for i, k in enumerate(idx_s.tolist())}

        for k0, rlen in runs:
            t = 0
            while t < rlen:
                nb = min(BLOCK_P, rlen - t)
                k0c = k0 + t
                p0 = pos[k0c]
                blk = val_s[p0 : p0 + nb]  # (nb, B_K, B_N)
                flat = blk.reshape(-1)

                pad = pad_to(elems, align_elems)
                if pad:
                    b_parts.append(torch.zeros(pad, dtype=dtype, device=device))
                    elems += pad

                cboff.append(elems)
                ck0.append(k0c)
                cnb.append(nb)
                ccol.append(j)

                b_parts.append(flat)
                elems += flat.numel()
                t += nb

        cstarts.append(len(ck0))

    b_packed = (
        torch.cat(b_parts, dim=0)
        if b_parts
        else torch.empty(0, dtype=dtype, device=device)
    )
    return (
        b_packed,
        torch.tensor(ck0, dtype=torch.int32, device=device),
        torch.tensor(cnb, dtype=torch.int32, device=device),
        torch.tensor(cboff, dtype=torch.int32, device=device),
        torch.tensor(cstarts, dtype=torch.int32, device=device),
        torch.tensor(ccol, dtype=torch.int32, device=device),
    )


@torch.no_grad()
def build_kchunk_lists_from_pack(ck0, cnb, ccol):
    """
    Build K-chunk adjacency from pack metadata ONLY (no values duplication).
    Each entry in klist corresponds to a unique k0_block where at least one column-block has a chunk.
    Returns:
      klist_k0     : [num_kchunks] int32 of k0_block values
      klist_starts : [num_kchunks+1] prefix-sum over flattened ids
      klist_ids    : [total_chunks] chunk ids grouped by k0
      MAX_CHUNKS_PER_K : int (host-side)
    """
    device = ck0.device
    # unique k0 values (sorted)
    k0_unique, inverse = torch.unique(
        ck0.to(torch.int64), sorted=True, return_inverse=True
    )
    num_kchunks = k0_unique.numel()
    counts = torch.bincount(inverse, minlength=num_kchunks)
    starts = torch.zeros(num_kchunks + 1, dtype=torch.int32, device=device)
    starts[1:] = counts.cumsum(0).to(torch.int32)
    ids = torch.empty_like(ck0, dtype=torch.int32)
    # stable fill
    cursor = torch.zeros(num_kchunks, dtype=torch.int32, device=device)
    for cid in range(ck0.numel()):
        group = int(inverse[cid].item())
        pos = int(starts[group].item() + cursor[group].item())
        ids[pos] = cid
        cursor[group] += 1
    max_per_k = int(counts.max().item()) if num_kchunks > 0 else 0
    return (
        k0_unique.to(torch.int32),
        starts,
        ids,
        max_per_k,
    )


# -----------------------
# Host launch convenience
# -----------------------
def launch_fwd(
    A, pack, B_K, B_N, BLOCK_M=128, BLOCK_P=8, num_warps=4, num_stages=2
):
    (b_packed, ck0, cnb, cboff, cstarts, ccol) = pack
    M, K = A.shape
    num_cols = cstarts.numel() - 1
    N = num_cols * B_N
    C = torch.empty((M, N), dtype=A.dtype, device=A.device)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, B_N))
    MAX_CHUNKS_PER_COL = (
        int((cstarts[1:] - cstarts[:-1]).max().item()) if num_cols > 0 else 0
    )

    _fwd_rm_lut[grid](
        A,
        b_packed,
        C,
        ck0,
        cnb,
        cboff,
        cstarts,
        M,
        N,
        K,
        A.stride(0),
        A.stride(1),
        C.stride(0),
        C.stride(1),
        B_K=B_K,
        B_N=B_N,
        BLOCK_M=BLOCK_M,
        BLOCK_P=BLOCK_P,
        MAX_CHUNKS_PER_COL=MAX_CHUNKS_PER_COL,
        num_warps=num_warps,  # type: ignore
        num_stages=num_stages,  # type: ignore
    )
    return C


def launch_dA(
    dC, pack, klist, B_K, B_N, BLOCK_M=128, BLOCK_P=8, num_warps=4, num_stages=2
):
    (b_packed, ck0, cnb, cboff, cstarts, ccol) = pack
    (klist_k0, klist_starts, klist_ids, MAX_CHUNKS_PER_K) = klist
    M, N = dC.shape
    # K is inferred from max K-block
    K = (int(ck0.max().item()) + 1) * B_K if ck0.numel() > 0 else 0
    dA = torch.zeros((M, K), dtype=dC.dtype, device=dC.device)
    grid = (triton.cdiv(M, BLOCK_M), klist_k0.numel())

    _dA_gather_kchunk[grid](
        dC,
        b_packed,
        dA,
        ck0,
        cnb,
        cboff,
        ccol,
        klist_k0,
        klist_starts,
        klist_ids,
        M,
        N,
        K,
        dC.stride(0),
        dC.stride(1),
        dA.stride(0),
        dA.stride(1),
        B_K=B_K,
        B_N=B_N,
        BLOCK_M=BLOCK_M,
        BLOCK_P=BLOCK_P,
        MAX_CHUNKS_PER_K=MAX_CHUNKS_PER_K,
        num_warps=num_warps,  # type: ignore
        num_stages=num_stages,  # type: ignore
    )
    return dA


def launch_dB(
    A, dC, pack, B_K, B_N, BLOCK_M=128, BLOCK_P=8, num_warps=4, num_stages=2
):
    (b_packed, ck0, cnb, cboff, cstarts, ccol) = pack
    M, K = A.shape
    total_chunks = ck0.numel()
    bgrad = torch.zeros_like(b_packed)
    grid = (total_chunks,)
    _dB_per_chunk[grid](
        A,
        dC,
        bgrad,
        ck0,
        cnb,
        cboff,
        ccol,
        M,
        dC.shape[1],
        K,
        A.stride(0),
        A.stride(1),
        dC.stride(0),
        dC.stride(1),
        total_chunks,
        B_K=B_K,
        B_N=B_N,
        BLOCK_M=BLOCK_M,
        BLOCK_P=BLOCK_P,
        num_warps=num_warps,  # type: ignore
        num_stages=num_stages,  # type: ignore
    )
    return bgrad


# --------------------------
# Autograd + nn.Module (bias)
# --------------------------
class _BSLinearFn(Function):
    @staticmethod
    def forward(ctx, A, bias, pack, klist, B_K, B_N, BLOCK_M, BLOCK_P):
        C = launch_fwd(A, pack, B_K, B_N, BLOCK_M=BLOCK_M, BLOCK_P=BLOCK_P)
        if bias is not None:
            C = C + bias.view(1, -1)
        ctx.save_for_backward(
            A, bias if bias is not None else torch.tensor([], device=A.device)
        )
        ctx.pack = pack
        ctx.klist = klist
        ctx.meta = (B_K, B_N, BLOCK_M, BLOCK_P)
        return C

    @staticmethod
    def backward(ctx, dC):  # type: ignore
        A, bias = ctx.saved_tensors
        B_K, B_N, BLOCK_M, BLOCK_P = ctx.meta
        # dA (no B^T packing)
        dA = launch_dA(
            dC, ctx.pack, ctx.klist, B_K, B_N, BLOCK_M=BLOCK_M, BLOCK_P=BLOCK_P
        )
        # dB (packed layout)
        dB_packed = launch_dB(
            A, dC, ctx.pack, B_K, B_N, BLOCK_M=BLOCK_M, BLOCK_P=BLOCK_P
        )
        # dbias
        dbias = dC.sum(dim=0) if bias.numel() != 0 else None
        return dA, dbias, dB_packed, None, None, None, None, None


class BlockSparseLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        B_K,
        B_N,
        row_block_idx,
        init_dense_weight=None,
        bias=True,
        BLOCK_P=8,
        BLOCK_M=128,
        dtype=torch.float16,
        device="cuda",
    ):
        super().__init__()
        assert out_features % B_N == 0 and in_features % B_K == 0
        self.in_features, self.out_features = in_features, out_features
        self.B_K, self.B_N = B_K, B_N
        self.BLOCK_P, self.BLOCK_M = BLOCK_P, BLOCK_M

        num_cols = out_features // B_N
        vals = []
        rbi = []
        if init_dense_weight is None:
            for j in range(num_cols):
                idx = row_block_idx[j].to(torch.int64).to(device)
                rbi.append(idx)
                vals.append(
                    torch.zeros(
                        (idx.numel(), B_K, B_N), dtype=dtype, device=device
                    )
                )
        else:
            W = init_dense_weight.to(dtype=dtype, device=device)  # [out,in]
            W_B = W.t().contiguous()  # [K,N]
            for j in range(num_cols):
                idx = row_block_idx[j].to(torch.int64).to(device)
                blocks = []
                for kblk in idx.tolist():
                    ks, ns = kblk * B_K, j * B_N
                    blocks.append(
                        W_B[ks : ks + B_K, ns : ns + B_N].contiguous()
                    )
                rbi.append(idx)
                vals.append(
                    torch.stack(blocks, dim=0)
                    if blocks
                    else torch.empty((0, B_K, B_N), dtype=dtype, device=device)
                )

        # pack (values become Parameter)
        pack = pack_bs_colblocks_rowmajor(
            rbi, vals, B_K, B_N, BLOCK_P, device=device, dtype=dtype
        )
        b_values = nn.Parameter(pack[0].clone().detach())
        self.pack = (b_values, *pack[1:])  # replace values with parameter

        # k-chunk lists for dA (metadata only)
        (_bvals, ck0, cnb, cboff, cstarts, ccol) = self.pack
        self.klist = build_kchunk_lists_from_pack(ck0, cnb, ccol)

        self.bias = (
            nn.Parameter(torch.zeros(out_features, dtype=dtype, device=device))
            if bias
            else None
        )

    def forward(self, A):
        return _BSLinearFn.apply(
            A,
            self.bias,
            self.pack,
            self.klist,
            self.B_K,
            self.B_N,
            self.BLOCK_M,
            self.BLOCK_P,
        )


# -----------------------------
# Helpers: pattern & evaluation
# -----------------------------
def make_random_pattern(K, N, B_K, B_N, p_keep=0.15, device="cuda"):
    num_kb = K // B_K
    num_cb = N // B_N
    rbi = []
    for j in range(num_cb):
        mask = torch.rand(num_kb, device=device) < p_keep
        idx = torch.nonzero(mask, as_tuple=False).flatten()
        rbi.append(idx)
    return rbi


@torch.no_grad()
def dense_from_blocks(rbi, vals, K, N, B_K, B_N):
    W = torch.zeros((K, N), dtype=vals[0].dtype, device=vals[0].device)
    for j, idx in enumerate(rbi):
        for t, kblk in enumerate(idx.tolist()):
            ks, ns = kblk * B_K, j * B_N
            W[ks : ks + B_K, ns : ns + B_N] = vals[j][t]
    return W


def build_vals_from_dense(rbi, W_B, B_K, B_N):
    vals = []
    for j, idx in enumerate(rbi):
        blocks = []
        for kblk in idx.tolist():
            ks, ns = kblk * B_K, j * B_N
            blocks.append(W_B[ks : ks + B_K, ns : ns + B_N].contiguous())
        vals.append(
            torch.stack(blocks, dim=0)
            if blocks
            else torch.empty((0, B_K, B_N), dtype=W_B.dtype, device=W_B.device)
        )
    return vals


# -----------------------------
# Correctness & Benchmark suite
# -----------------------------
def test_correctness_and_bench(
    M=2048,
    K=2048,
    N=2048,
    B_K=8,
    B_N=8,
    p_keep=0.10,
    dtype=torch.float16,
    seed=0,
    reps=50,
    warmup=10,
    device="cuda",
):
    torch.manual_seed(seed)
    A = torch.randn(M, K, device=device, dtype=dtype)
    # [out,in]
    W_dense_out_in = torch.randn(N, K, device=device, dtype=dtype) * 0.1
    W_B = W_dense_out_in.t().contiguous()  # [K,N]
    bias = torch.randn(N, device=device, dtype=dtype)

    # pattern & vals
    rbi = make_random_pattern(K, N, B_K, B_N, p_keep=p_keep, device=device)
    vals = build_vals_from_dense(rbi, W_B, B_K, B_N)

    # pack
    pack = pack_bs_colblocks_rowmajor(
        rbi,
        vals,
        B_K,
        B_N,
        BLOCK_P=max(8, 32 // B_K),
        device=device,
        dtype=dtype,
    )
    # klist = build_kchunk_lists_from_pack(pack[1], pack[2], pack[5])

    # forward compare
    C_ref = A @ W_B + bias
    C_sparse = (
        launch_fwd(A, pack, B_K, B_N, BLOCK_M=128, BLOCK_P=max(8, 32 // B_K))
        + bias
    )
    max_abs = (C_ref - C_sparse).abs().max().item()
    rel = (C_ref.abs().max().clamp_min(1e-6)).item()
    print(
        f"[FWD] max abs err: {max_abs:.3e}  (rel vs max|ref|: {max_abs / rel:.3e})"
    )

    # backward compare
    A_req = A.clone().detach().requires_grad_(True)
    bparam = nn.Parameter(pack[0].clone().detach())
    pack_run = (bparam, *pack[1:])
    # klist_run = klist

    # our path
    C = (
        launch_fwd(
            A_req, pack_run, B_K, B_N, BLOCK_M=128, BLOCK_P=max(8, 32 // B_K)
        )
        + bias
    )
    loss = (C**2).mean()
    loss.backward()
    dA_ours = A_req.grad.detach().clone()  # type: ignore
    dB_ours = bparam.grad.detach().clone()  # type: ignore
    dbias_ours = (
        (2 * C / C.numel()).sum(dim=0).detach()
    )  # same as dC.sum(dim=0) scaled by loss deriv.

    # explicit dense reference grads
    dC = 2 * C_ref / C_ref.numel()
    dA_true = dC @ W_B.t()
    dW_true = A.t() @ dC
    dbias_true = dC.sum(dim=0)
    # re-pack dW_true for comparison
    vals_grad = build_vals_from_dense(rbi, dW_true, B_K, B_N)
    bgrad_true, *_ = pack_bs_colblocks_rowmajor(
        rbi,
        vals_grad,
        B_K,
        B_N,
        BLOCK_P=max(8, 32 // B_K),
        device=device,
        dtype=dtype,
    )

    da_err = (dA_ours - dA_true).abs().max().item()
    db_err = (dB_ours - bgrad_true).abs().max().item()
    dbias_err = (dbias_ours - dbias_true).abs().max().item()
    print(
        f"[BWD] max|dA-dA_ref|: {da_err:.3e}   max|dB-dB_ref|: {db_err:.3e}   max|dbias-dbias_ref|: {dbias_err:.3e}"
    )

    # simple forward benchmark
    BLOCK_P = max(8, 32 // B_K)

    def bench_fwd():
        torch.cuda.synchronize()
        for _ in range(warmup):
            _ = launch_fwd(A, pack, B_K, B_N, BLOCK_M=128, BLOCK_P=BLOCK_P)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(reps):
            _ = launch_fwd(A, pack, B_K, B_N, BLOCK_M=128, BLOCK_P=BLOCK_P)
        torch.cuda.synchronize()
        return (time.time() - t0) / reps

    t = bench_fwd()
    nnz_blocks = sum([len(x) for x in rbi])
    eff_flops = (
        2.0 * M * nnz_blocks * (B_K * B_N)
    )  # theoretical work actually done
    tflops = eff_flops / t / 1e12
    density = nnz_blocks / ((K // B_K) * (N // B_N))
    print(
        f"[BENCH] density={density:.3f}  time={t * 1e3:.2f} ms  eff-TFLOPs={tflops:.2f}  (M={M},K={K},N={N}, B={B_K}x{B_N})"
    )


if __name__ == "__main__":
    # Run a small sanity check
    test_correctness_and_bench(
        M=2048,
        K=2048,
        N=2048,
        B_K=16,
        B_N=16,
        p_keep=0.10,
        dtype=torch.float16,
        reps=50,
        warmup=10,
        device="cuda",
    )

    K, N = 4096, 4096
    B_K, B_N = 16, 16
    rbi = make_random_pattern(K, N, B_K, B_N, p_keep=0.12, device="cuda")
    layer = BlockSparseLinear(
        in_features=K,
        out_features=N,
        B_K=B_K,
        B_N=B_N,
        row_block_idx=rbi,
        init_dense_weight=None,
        bias=True,
        BLOCK_P=max(8, 32 // B_K),
        BLOCK_M=128,
        dtype=torch.float16,
        device="cuda",
    )
    x = torch.randn(1024, K, device="cuda", dtype=torch.float16)
    y = layer(x)
    print("Output:", y.shape)
