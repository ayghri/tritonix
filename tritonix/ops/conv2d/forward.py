import torch
import triton
import triton.language as tl

from tritonix.autotune import tunable, PowerOfTwo, Choice, Range


@tunable(
    keys=["GEMM_M", "GEMM_N", "GEMM_K"],
    space={
        "BLOCK_SIZE_M": PowerOfTwo(32, 256),
        "BLOCK_SIZE_N": PowerOfTwo(32, 256),
        "BLOCK_SIZE_K": PowerOfTwo(16, 64),
        "GROUP_SIZE_M": Choice([4, 8, 16]),
        "num_stages": Range(2, 5),
        "num_warps": Choice([4, 8]),
    },
)
@triton.jit
def conv2d_forward_kernel(
    output_ptr,
    input_ptr,
    weight_ptr,
    bias_ptr,
    BATCH_SIZE,
    C_IN,
    H_IN,
    W_IN,
    C_OUT,
    H_OUT,
    W_OUT,
    FILTER_H,
    FILTER_W,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    dil_h,
    dil_w,
    GEMM_M,  # = BATCH_SIZE * H_OUT * W_OUT
    GEMM_N,  # = C_OUT
    GEMM_K,  # = C_IN * FILTER_H * FILTER_W
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    OUT_DTYPE: tl.constexpr = tl.constexpr(tl.float16),
    USE_TF32: tl.constexpr = tl.constexpr(True),
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
    gemm_k = tl.arange(0, BLOCK_SIZE_K)

    ### for compiler
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(BATCH_SIZE > 0)
    tl.assume(C_IN > 0)
    tl.assume(C_OUT > 0)
    tl.assume(FILTER_H > 0)
    tl.assume(FILTER_W > 0)
    tl.assume(H_IN > 0)
    tl.assume(H_OUT > 0)
    tl.assume(W_IN > 0)
    tl.assume(W_OUT > 0)
    tl.assume(dil_h > 0)
    tl.assume(dil_w > 0)
    tl.assume(pad_h >= 0)
    tl.assume(pad_w >= 0)
    tl.assume(stride_h > 0)
    tl.assume(stride_w > 0)
    tl.multiple_of(gemm_m, 16)
    tl.multiple_of(offs_cout, 16)
    tl.multiple_of(gemm_k, 16)  # if BLOCK_SIZE_K % 16 == 0
    tl.max_contiguous(offs_cout, 16)

    batch_idx = gemm_m // (H_OUT * W_OUT)
    offs_out = gemm_m % (H_OUT * W_OUT)
    offs_hout = offs_out // W_OUT
    offs_wout = offs_out % W_OUT

    a_base_batch = batch_idx[:, None] * (C_IN * H_IN * W_IN)  # (BM,1)
    a_base_h0 = offs_hout[:, None] * stride_h - pad_h  # (BM,1)
    a_base_w0 = offs_wout[:, None] * stride_w - pad_w  # (BM,1)
    w_base_cout = offs_cout[None, :] * (C_IN * FILTER_H * FILTER_W)  # (1, BN)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    m_mask = (
        (batch_idx[:, None] < BATCH_SIZE)
        & (offs_hout[:, None] < H_OUT)
        & (offs_wout[:, None] < W_OUT)
    )
    n_mask = offs_cout[None, :] < C_OUT

    for idx_k in tl.range(0, tl.cdiv(GEMM_K, BLOCK_SIZE_K)):
        offs_cin = gemm_k // (FILTER_H * FILTER_W)
        c_fh_fw_residual = gemm_k % (FILTER_H * FILTER_W)
        offs_fh = c_fh_fw_residual // FILTER_W
        offs_fw = c_fh_fw_residual % FILTER_W

        offs_hin = a_base_h0 + offs_fh[None, :] * dil_h
        offs_win = a_base_w0 + offs_fw[None, :] * dil_w

        input_offs = (
            a_base_batch
            + offs_cin[None, :] * (H_IN * W_IN)
            + offs_hin * W_IN
            + offs_win
        )
        input_mask = (
            m_mask
            & (offs_hin >= 0)  # because of -pad_h
            & (offs_hin < H_IN)
            & (offs_win >= 0)  # because of -pad_w
            & (offs_win < W_IN)
            & (offs_cin[None, :] < C_IN)
        )

        weight_mask = n_mask & (offs_cin[:, None] < C_IN)

        weight_offs = (
            w_base_cout
            + offs_cin[:, None] * (FILTER_H * FILTER_W)
            + offs_fh[:, None] * FILTER_W
            + offs_fw[:, None]
        )

        weight_ptrs = weight_ptr + weight_offs
        input_ptrs = input_ptr + input_offs

        weight_data = tl.load(
            weight_ptrs, mask=weight_mask, other=0.0, cache_modifier=".ca"
        )
        input_data = tl.load(
            input_ptrs, mask=input_mask, other=0.0, cache_modifier=".cv"
        )
        acc = tl.dot(input_data, weight_data, acc, allow_tf32=USE_TF32)
        gemm_k += BLOCK_SIZE_K

    if bias_ptr is not None:
        bias_ptrs = bias_ptr + offs_cout[None, :]
        bias_data = tl.load(
            bias_ptrs, mask=offs_cout[None, :] < GEMM_N, other=0.0
        ).to(tl.float32)
        acc = acc + bias_data

    output_offsets = (
        batch_idx[:, None] * C_OUT * H_OUT * W_OUT
        + offs_cout[None, :] * H_OUT * W_OUT
        + offs_hout[:, None] * W_OUT
        + offs_wout[:, None]
    )
    output_mask = m_mask & n_mask
    output_ptrs = output_ptr + output_offsets
    tl.store(
        output_ptrs, acc.to(OUT_DTYPE), mask=output_mask, cache_modifier=".cs"
    )


# ---------------------------------------------------------------------------
# Autotuning
# ---------------------------------------------------------------------------

_CONV_FWD_CONFIG_CACHE: dict[tuple, dict] = {}
_CANDIDATE_CONFIGS = [
    # (BM, BN, BK, GROUP)
    (128, 128, 32, 8),
    (128, 64, 32, 8),
    (64, 128, 32, 8),
    (64, 64, 32, 8),
    (128, 128, 64, 8),
    (64, 128, 64, 8),
    (128, 64, 64, 8),
]


def _time_kernel(fn, iters=6, warmup=2):
    for _ in range(warmup):
        fn()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iters


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def conv2d_forward(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride: tuple[int, int] = (1, 1),
    padding: tuple[int, int] = (0, 0),
    dilation: tuple[int, int] = (1, 1),
    use_tf32: bool = True,
    config: dict | None = None,
) -> torch.Tensor:
    """Triton conv2d forward with lightweight autotuning.

    Drop-in replacement for torch.nn.functional.conv2d (forward only).
    Returns output tensor of shape [N, C_OUT, H_OUT, W_OUT].
    """
    assert input.is_contiguous(), "Input must be contiguous NCHW"
    assert weight.is_contiguous(), "Weight must be contiguous KCRS"
    N, C_IN, H_IN, W_IN = input.shape
    C_OUT, C_INw, R, S = weight.shape
    assert C_IN == C_INw
    str_h, str_w = stride
    pad_h, pad_w = padding
    dil_h, dil_w = dilation
    H_OUT = (H_IN + 2 * pad_h - dil_h * (R - 1) - 1) // str_h + 1
    W_OUT = (W_IN + 2 * pad_w - dil_w * (S - 1) - 1) // str_w + 1

    GEMM_M = N * H_OUT * W_OUT
    GEMM_N = C_OUT
    GEMM_K = C_IN * R * S

    out = input.new_empty((N, C_OUT, H_OUT, W_OUT))

    if out.dtype == torch.float16:
        out_ty = tl.float16
    elif out.dtype == torch.bfloat16:
        out_ty = tl.bfloat16
    else:
        out_ty = tl.float32

    key = (N, C_IN, H_IN, W_IN, C_OUT, R, S, stride, padding, dilation, input.dtype)

    if config is None:
        cfg = _CONV_FWD_CONFIG_CACHE.get(key)
        if cfg is None:
            candidates = []
            for BM, BN, BK, G in _CANDIDATE_CONFIGS:
                if BM > GEMM_M and GEMM_M >= 64:
                    continue
                if BN > GEMM_N and GEMM_N >= 64:
                    continue
                if BK > GEMM_K:
                    continue
                candidates.append((BM, BN, BK, G))
            if not candidates:
                candidates = [(64, 64, 32 if GEMM_K % 32 == 0 else 16, 8)]

            tmp = out
            best_time = float("inf")
            best = None
            for BM, BN, BK, G in candidates:
                grid = (triton.cdiv(GEMM_M, BM), triton.cdiv(GEMM_N, BN))

                def launch():
                    conv2d_forward_kernel[grid](
                        tmp, input, weight, bias,
                        N, C_IN, H_IN, W_IN, C_OUT, H_OUT, W_OUT, R, S,
                        str_h, str_w, pad_h, pad_w, dil_h, dil_w,
                        GEMM_M, GEMM_N, GEMM_K,
                        BLOCK_SIZE_M=BM,
                        BLOCK_SIZE_N=BN,
                        BLOCK_SIZE_K=BK,
                        GROUP_SIZE_M=G,
                        OUT_DTYPE=out_ty,
                        USE_TF32=use_tf32,
                    )

                try:
                    t = _time_kernel(launch)
                except Exception:
                    continue
                if t < best_time:
                    best_time = t
                    best = dict(
                        BLOCK_SIZE_M=BM, BLOCK_SIZE_N=BN,
                        BLOCK_SIZE_K=BK, GROUP_SIZE_M=G,
                    )
            if best is None:
                best = {
                    "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64,
                    "BLOCK_SIZE_K": 32 if GEMM_K % 32 == 0 else 16,
                    "GROUP_SIZE_M": 8,
                }
            _CONV_FWD_CONFIG_CACHE[key] = best
            cfg = best
        BLOCK_SIZE_M = cfg["BLOCK_SIZE_M"]
        BLOCK_SIZE_N = cfg["BLOCK_SIZE_N"]
        BLOCK_SIZE_K = cfg["BLOCK_SIZE_K"]
        GROUP_SIZE_M = cfg["GROUP_SIZE_M"]
    else:
        BLOCK_SIZE_M = config["BLOCK_SIZE_M"]
        BLOCK_SIZE_N = config["BLOCK_SIZE_N"]
        BLOCK_SIZE_K = config["BLOCK_SIZE_K"]
        GROUP_SIZE_M = config.get("GROUP_SIZE_M", 8)

    grid = (triton.cdiv(GEMM_M, BLOCK_SIZE_M), triton.cdiv(GEMM_N, BLOCK_SIZE_N))
    conv2d_forward_kernel[grid](
        out, input, weight, bias,
        N, C_IN, H_IN, W_IN, C_OUT, H_OUT, W_OUT, R, S,
        str_h, str_w, pad_h, pad_w, dil_h, dil_w,
        GEMM_M, GEMM_N, GEMM_K,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        OUT_DTYPE=out_ty,
        USE_TF32=use_tf32,
    )
    return out


def torch_conv2d_forward(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride: tuple[int, int] = (1, 1),
    padding: tuple[int, int] = (0, 0),
    dilation: tuple[int, int] = (1, 1),
) -> torch.Tensor:
    """PyTorch reference for benchmarking."""
    return torch.nn.functional.conv2d(input, weight, bias, stride, padding, dilation)
