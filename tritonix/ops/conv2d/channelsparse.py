import triton
import triton.language as tl

"""Channel-sparse forward Conv2D kernel.
Each output channel uses exactly NC selected input channels (indexed per-output-channel).
Weights layout: [C_OUT, NC, FILTER_H, FILTER_W]
Channel indices layout: [C_OUT, NC] (int32)
Input layout:  [N, C_IN, H_IN, W_IN]
Output layout: [N, C_OUT, H_OUT, W_OUT]

We compute:
  out[n, co, ho, wo] = bias[co] + sum_{i=0..NC-1} sum_{fh,fw} w[co,i,fh,fw] * inp[n, idx[co,i], ho*stride_h - pad_h + fh*dil_h, wo*stride_w - pad_w + fw*dil_w]

This version is a straightforward implementation optimized for small NC; it performs
nested loops across NC, FH, FW and vectorizes over output channels (N-axis tile) and
(M-axis tile = batch * H_out * W_out).

Potential future optimizations:
 - Unroll FH/FW for common sizes (3x3, 5x5)
 - Split NC loop into BLOCK_SIZE_K with partial accumulation
 - Prefetch next spatial row (software pipelining)
 - Use shared memory to stage channel indices when NC large.
"""


@triton.jit
def conv2d_forward_channelsparse_kernel(
    output_ptr,  # *fp16/fp32
    input_ptr,  # *fp16/fp32
    weight_ptr,  # *fp16/fp32 (C_OUT, NC, FH, FW)
    bias_ptr,  # *fp16/fp32 or nullptr
    ch_idx_ptr,  # *int32 (C_OUT, NC)
    BATCH_SIZE,
    C_IN,  # total input channels (for bounds check)
    H_IN,
    W_IN,
    C_OUT,
    H_OUT,
    W_OUT,
    FILTER_H: tl.constexpr,
    FILTER_W: tl.constexpr,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    dil_h,
    dil_w,
    NC: tl.constexpr,  # number of channels per output (constexpr for unrolling)
    GEMM_M,  # = BATCH_SIZE * H_OUT * W_OUT
    GEMM_N,  # = C_OUT
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  # tile of NC indices to stage
    GROUP_SIZE_M: tl.constexpr,
    OUT_DTYPE: tl.constexpr = tl.constexpr(tl.float16),
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    num_pid_m = tl.cdiv(GEMM_M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(GEMM_N, BLOCK_SIZE_N)
    pid_m, pid_n = tl.swizzle2d(
        pid_m, pid_n, num_pid_m, num_pid_n, GROUP_SIZE_M
    )

    gemm_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cout = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # Derive (batch, h_out, w_out) from gemm_m
    batch_idx = gemm_m // (H_OUT * W_OUT)
    offs_out = gemm_m % (H_OUT * W_OUT)
    offs_hout = offs_out // W_OUT
    offs_wout = offs_out % W_OUT

    # Base components
    a_base_batch = batch_idx[:, None] * (C_IN * H_IN * W_IN)
    a_base_h0 = offs_hout[:, None] * stride_h - pad_h
    a_base_w0 = offs_wout[:, None] * stride_w - pad_w

    m_mask = (
        (batch_idx[:, None] < BATCH_SIZE)
        & (offs_hout[:, None] < H_OUT)
        & (offs_wout[:, None] < W_OUT)
    )
    n_mask_row = offs_cout < C_OUT  # (BN,)
    n_mask = n_mask_row[None, :]    # (1, BN) for broadcasting with (BM, BN)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    weight_cout_base = offs_cout[None, :] * (NC * FILTER_H * FILTER_W)

    # Tile NC dimension: load BLOCK_SIZE_K (power-of-two) indices once per chunk and reuse over FH*FW
    for ic_base in tl.static_range(0, NC, BLOCK_SIZE_K):
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        ic_mask = ic_base + offs_k < NC  # (BK,)
        # Build pointer matrix shape (BK, BN) for transposed load (k major)
        ch_idx_tile_ptrs = (
            ch_idx_ptr
            + (ic_base + offs_k)[:, None]
            + offs_cout[None, :] * NC
        )
        # Load channel indices tile [BK, BN]
        ch_idx_tile = tl.load(
            ch_idx_tile_ptrs,
            mask=(ic_mask[:, None] & n_mask_row[None, :]),
            other=0,
        )
        # Iterate over filter spatial domain
        for fh in tl.static_range(0, FILTER_H):
            hin = a_base_h0 + fh * dil_h
            in_h_ok = (hin >= 0) & (hin < H_IN)
            for fw in tl.static_range(0, FILTER_W):
                win = a_base_w0 + fw * dil_w
                in_w_ok = (win >= 0) & (win < W_IN)
                spatial_mask = m_mask & n_mask & in_h_ok & in_w_ok
                # Iterate over staged channel indices
                for k in tl.static_range(0, BLOCK_SIZE_K):
                    use_ic = ic_base + k < NC
                    cin_idx_vec = ch_idx_tile[k, :]  # (BN,)
                    valid_cin = (cin_idx_vec < C_IN) & use_ic
                    mask_in = spatial_mask & valid_cin[None, :]
                    input_offs = (
                        a_base_batch
                        + cin_idx_vec[None, :] * (H_IN * W_IN)
                        + hin * W_IN
                        + win
                    )
                    inp = tl.load(input_ptr + input_offs, mask=mask_in, other=0.0)
                    weight_offs = (
                        weight_cout_base
                        + (ic_base + k) * (FILTER_H * FILTER_W)
                        + fh * FILTER_W
                        + fw
                    )
                    w = tl.load(
                        weight_ptr + weight_offs, mask=(n_mask & use_ic), other=0.0, cache_modifier=".ca"
                    )
                    acc += inp * w

    if bias_ptr is not None:
        bias = tl.load(
            bias_ptr + offs_cout[None, :], mask=n_mask, other=0.0
        ).to(tl.float32)
        acc += bias

    # Store
    output_offsets = (
        batch_idx[:, None] * C_OUT * H_OUT * W_OUT
        + offs_cout[None, :] * H_OUT * W_OUT
        + offs_hout[:, None] * W_OUT
        + offs_wout[:, None]
    )
    out_mask = m_mask & n_mask
    tl.store(output_ptr + output_offsets, acc.to(OUT_DTYPE), mask=out_mask)
