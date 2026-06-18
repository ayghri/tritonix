import argparse
import math
import os
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# -------------------- Helpers --------------------


def cdiv(x, y):  # ceil division
    return (x + y - 1) // y


def compute_out_hw(
    H_in, W_in, kH, kW, stride_h, stride_w, pad_h, pad_w, dil_h, dil_w
):
    H_out = (H_in + 2 * pad_h - dil_h * (kH - 1) - 1) // stride_h + 1
    W_out = (W_in + 2 * pad_w - dil_w * (kW - 1) - 1) // stride_w + 1
    return int(H_out), int(W_out)


def set_torch_flags(tf32=True):
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    torch.backends.cudnn.benchmark = True





# ---------- 1x1 (pointwise) ----------
@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 128},
            num_warps=8,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 128},
            num_warps=4,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64},
            num_warps=4,
            num_stages=3,
        ),
    ],
    key=["GEMM_M", "GEMM_N", "GEMM_K"],
)
@triton.jit
def conv2d_forward_1x1_kernel(
    out_ptr,
    in_ptr,
    w_ptr,
    b_ptr,
    BATCH,
    C_IN,
    H_IN,
    W_IN,
    C_OUT,
    H_OUT,
    W_OUT,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    GEMM_M,
    GEMM_N,
    GEMM_K,  # M=B*H_out*W_out, N=C_out, K=C_in
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    USE_TF32: tl.constexpr = tl.constexpr(True),
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # map m -> (b, h_out, w_out)
    b = m // (H_OUT * W_OUT)
    mw = m % (H_OUT * W_OUT)
    h = mw // W_OUT
    w = mw % W_OUT

    # alignment hints (choose BLOCK_* multiples of 16)
    tl.multiple_of(m, 16)
    tl.multiple_of(n, 16)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Hoisted bases
    a_base_batch = b[:, None] * (C_IN * H_IN * W_IN)  # (BM,1)
    hin = h[:, None] * stride_h - pad_h  # (BM,1) (fh=0)
    win = w[:, None] * stride_w - pad_w  # (BM,1) (fw=0)
    a_hw = hin * W_IN + win  # (BM,1)
    w_base_cout = n[None, :] * (C_IN)  # (1,BN)

    # bounds masks
    m_ok = (
        (b[:, None] < BATCH)
        & (h[:, None] >= 0)
        & (h[:, None] < H_IN)
        & (w[:, None] >= 0)
        & (w[:, None] < W_IN)
    )
    n_ok = n[None, :] < C_OUT

    # K loop over input channels
    for k0 in tl.range(0, tl.cdiv(GEMM_K, BLOCK_K)):
        k = k0 * BLOCK_K + tl.arange(0, BLOCK_K)  # cin
        tl.multiple_of(k, 16)

        # A (input) pointers
        a_ptrs = in_ptr + a_base_batch + k[None, :] * (H_IN * W_IN) + a_hw
        a_mask = m_ok & (k[None, :] < C_IN)

        # B (weights) pointers: layout OIHW with 1x1 -> offset = cout*C_IN + cin
        b_ptrs = w_ptr + w_base_cout + k[:, None]
        b_mask = n_ok & (k[:, None] < C_IN)

        A = tl.load(a_ptrs, mask=a_mask, other=0.0, cache_modifier=".cs")
        B = tl.load(b_ptrs, mask=b_mask, other=0.0, cache_modifier=".ca")

        acc = tl.dot(A, B, acc, allow_tf32=USE_TF32)

    # Bias
    if b_ptr is not None:
        bias = tl.load(b_ptr + n[None, :], mask=n_ok, other=0.0).to(tl.float32)
        acc += bias

    # Store
    out_offsets = (
        b[:, None] * (C_OUT * H_OUT * W_OUT)
        + n[None, :] * (H_OUT * W_OUT)
        + h[:, None] * W_OUT
        + w[:, None]
    )
    out_mask = m_ok & n_ok
    tl.store(
        out_ptr + out_offsets,
        acc.to(out_ptr.dtype.element_ty),
        mask=out_mask,
        cache_modifier=".cs",
    )


# ---------- 3x3 (unrolled taps) ----------
@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64},
            num_warps=4,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64},
            num_warps=4,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32},
            num_warps=8,
            num_stages=4,
        ),
    ],
    key=["GEMM_M", "GEMM_N", "GEMM_K"],
)
@triton.jit
def conv2d_forward_3x3_kernel(
    out_ptr,
    in_ptr,
    w_ptr,
    b_ptr,
    BATCH,
    C_IN,
    H_IN,
    W_IN,
    C_OUT,
    H_OUT,
    W_OUT,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    GEMM_M,
    GEMM_N,
    GEMM_K,  # K == C_IN * 9
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    USE_TF32: tl.constexpr = tl.constexpr(True),
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    b = m // (H_OUT * W_OUT)
    mw = m % (H_OUT * W_OUT)
    h = mw // W_OUT
    w = mw % W_OUT

    tl.multiple_of(m, 16)
    tl.multiple_of(n, 16)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Hoisted bases
    a_base_batch = b[:, None] * (C_IN * H_IN * W_IN)
    base_h0 = h[:, None] * stride_h - pad_h
    base_w0 = w[:, None] * stride_w - pad_w
    w_base_cout = n[None, :] * (C_IN * 3 * 3)

    m_row_ok = (
        (b[:, None] < BATCH) & (h[:, None] < H_OUT) & (w[:, None] < W_OUT)
    )
    n_ok = n[None, :] < C_OUT

    # K loop over cin tiles (unroll 3x3 taps inside)
    for k0 in tl.range(0, tl.cdiv(C_IN, BLOCK_K)):
        cin = k0 * BLOCK_K + tl.arange(0, BLOCK_K)
        tl.multiple_of(cin, 16)

        # weights common: offset per (cout, cin, fh, fw)
        cin_ok_row = cin[None, :] < C_IN

        # Unroll 3x3 taps
        for fh in tl.static_range(0, 3):
            hin = base_h0 + fh
            h_ok = (hin >= 0) & (hin < H_IN)

            for fw in tl.static_range(0, 3):
                win = base_w0 + fw
                w_ok = (win >= 0) & (win < W_IN)

                # Input pointers for this tap
                a_ptrs = (
                    in_ptr
                    + a_base_batch
                    + cin[None, :] * (H_IN * W_IN)
                    + hin * W_IN
                    + win
                )
                a_mask = m_row_ok & h_ok & w_ok & cin_ok_row

                # Weights for this tap
                w_ptrs = (
                    w_ptr + w_base_cout + cin[:, None] * (3 * 3) + fh * 3 + fw
                )
                w_mask = n_ok & (cin[:, None] < C_IN)

                A = tl.load(
                    a_ptrs, mask=a_mask, other=0.0, cache_modifier=".cv"
                )
                B = tl.load(
                    w_ptrs, mask=w_mask, other=0.0, cache_modifier=".ca"
                )
                acc = tl.dot(A, B, acc, allow_tf32=USE_TF32)

    if b_ptr is not None:
        bias = tl.load(b_ptr + n[None, :], mask=n_ok, other=0.0).to(tl.float32)
        acc += bias

    out_offsets = (
        b[:, None] * (C_OUT * H_OUT * W_OUT)
        + n[None, :] * (H_OUT * W_OUT)
        + h[:, None] * W_OUT
        + w[:, None]
    )
    out_mask = (b[:, None] < BATCH) & (n[None, :] < C_OUT)
    tl.store(
        out_ptr + out_offsets,
        acc.to(out_ptr.dtype.element_ty),
        mask=out_mask,
        cache_modifier=".cs",
    )


# ---------- Thin Python launcher ----------
def launch_conv2d_optimized(
    out_ptr,
    in_ptr,
    w_ptr,
    b_ptr,
    BATCH,
    C_IN,
    H_IN,
    W_IN,
    C_OUT,
    H_OUT,
    W_OUT,
    FILTER_H,
    FILTER_W,
    stride_h=1,
    stride_w=1,
    pad_h=0,
    pad_w=0,
    dil_h=1,
    dil_w=1,
    use_tf32=True,
    num_warps=None,
    num_stages=None,
):
    GEMM_M = BATCH * H_OUT * W_OUT
    GEMM_N = C_OUT
    if FILTER_H == 1 and FILTER_W == 1:
        GEMM_K = C_IN
        # grid = (M tiles, N tiles)
        grid = (
            triton.cdiv(
                GEMM_M, 128
            ),  # must match largest BLOCK_M in autotune set
            triton.cdiv(GEMM_N, 128),
        )
        conv2d_forward_1x1_kernel[grid](
            out_ptr,
            in_ptr,
            w_ptr,
            b_ptr,
            BATCH,
            C_IN,
            H_IN,
            W_IN,
            C_OUT,
            H_OUT,
            W_OUT,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            dil_h,
            dil_w,
            GEMM_M,
            GEMM_N,
            GEMM_K,
            num_warps=num_warps if num_warps is not None else 8,
            num_stages=num_stages if num_stages is not None else 4,
            USE_TF32=use_tf32,
        )
    elif FILTER_H == 3 and FILTER_W == 3:
        GEMM_K = C_IN * 9
        grid = (
            triton.cdiv(GEMM_M, 128),
            triton.cdiv(GEMM_N, 128),
        )
        conv2d_forward_3x3_kernel[grid](
            out_ptr,
            in_ptr,
            w_ptr,
            b_ptr,
            BATCH,
            C_IN,
            H_IN,
            W_IN,
            C_OUT,
            H_OUT,
            W_OUT,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            dil_h,
            dil_w,
            GEMM_M,
            GEMM_N,
            GEMM_K,
            num_warps=num_warps if num_warps is not None else 8,
            num_stages=num_stages if num_stages is not None else 4,
            USE_TF32=use_tf32,
        )
    else:
        raise ValueError("Use the generic kernel for other filter sizes.")


# --------------- Launcher ----------------


def run_triton_conv(x, w, bias, stride, padding, dilation, use_tf32=True):
    B, Cin, H, W = x.shape
    Cout, Cin_w, kH, kW = w.shape
    assert Cin == Cin_w, "Cin mismatch"
    H_out, W_out = compute_out_hw(
        H,
        W,
        kH,
        kW,
        stride[0],
        stride[1],
        padding[0],
        padding[1],
        dilation[0],
        dilation[1],
    )
    M = B * H_out * W_out
    N = Cout
    K = Cin * kH * kW

    out = torch.empty((B, Cout, H_out, W_out), device=x.device, dtype=x.dtype)

    # grid uses the largest BLOCKs from autotune sets (128)
    grid = (cdiv(M, 128), cdiv(N, 128))

    if kH == 1 and kW == 1:
        conv2d_forward_1x1_kernel[grid](
            out,
            x,
            w,
            bias,
            B,
            Cin,
            H,
            W,
            Cout,
            H_out,
            W_out,
            stride[0],
            stride[1],
            padding[0],
            padding[1],
            M,
            N,
            Cin,  # GEMM_K = Cin
            USE_TF32=use_tf32,
        )
    elif kH == 3 and kW == 3:
        conv2d_forward_3x3_kernel[grid](
            out,
            x,
            w,
            bias,
            B,
            Cin,
            H,
            W,
            Cout,
            H_out,
            W_out,
            stride[0],
            stride[1],
            padding[0],
            padding[1],
            M,
            N,
            K,
            USE_TF32=use_tf32,
        )
    else:
        raise ValueError("This launcher only supports 1x1 and 3x3.")

    return out


# --------------- Benchmarking ---------------


@dataclass
class BenchResult:
    B: int
    Cin: int
    Cout: int
    H: int
    W: int
    kH: int
    kW: int
    dtype: str
    tf32: bool
    ms_triton: float
    ms_torch: float
    ms_torch_compiled: Optional[float]
    tflops_triton: float
    tflops_torch: float
    tflops_torch_compiled: Optional[float]


def measure_ms(fn, iters=100, warmup=30):
    # Use CUDA events for stable measurements
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    # Warmup
    for _ in range(warmup):
        y = fn()
    torch.cuda.synchronize()
    # Timed
    times = []
    for _ in range(iters):
        start.record()
        y = fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))  # ms
    return sum(times) / len(times)


def approx_tflops(B, Cin, Cout, H, W, kH, kW, stride, padding, dilation, ms):
    H_out, W_out = compute_out_hw(
        H,
        W,
        kH,
        kW,
        stride[0],
        stride[1],
        padding[0],
        padding[1],
        dilation[0],
        dilation[1],
    )
    M = B * H_out * W_out
    N = Cout
    K = Cin * kH * kW
    flops = 2.0 * M * N * K  # MACs -> 2 flops
    tflops = flops / (ms * 1e-3) / 1e12
    return tflops


def run_suite(
    B_list: List[int],
    Cin: int,
    Cout: int,
    H: int,
    W: int,
    kH: int,
    kW: int,
    dtype: str,
    stride: Tuple[int, int],
    padding: Tuple[int, int],
    dilation: Tuple[int, int],
    iters: int,
    warmup: int,
    use_tf32: bool,
    use_compile: bool,
):
    device = torch.device("cuda")
    torch.manual_seed(0)
    set_torch_flags(tf32=use_tf32)

    dtype_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    dt = dtype_map[dtype]

    results: List[BenchResult] = []

    for B in B_list:
        x = torch.randn(B, Cin, H, W, device=device, dtype=dt)
        w = torch.randn(Cout, Cin, kH, kW, device=device, dtype=dt) * (
            1.0 / math.sqrt(Cin * kH * kW)
        )
        bias = torch.randn(Cout, device=device, dtype=dt)

        # Reference conv
        def torch_op():
            return F.conv2d(
                x,
                w,
                bias=bias,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )

        # Optional compiled path (PyTorch 2.1+)
        torch_compiled = None
        if use_compile and hasattr(torch, "compile"):
            try:
                torch_compiled = torch.compile(torch_op, mode="max-autotune")
                # one warm call to trigger graph creation
                torch_compiled()
                torch.cuda.synchronize()
            except Exception as e:
                print(f"[warn] torch.compile failed: {e}")
                torch_compiled = None

        # Triton op
        def triton_op():
            return run_triton_conv(
                x, w, bias, stride, padding, dilation, use_tf32=use_tf32
            )

        # Correctness check once
        y_torch = torch_op()
        y_triton = triton_op()
        # Tolerances depend on dtype/TF32
        if dtype == "fp32" and not use_tf32:
            rtol, atol = 1e-4, 1e-4
        else:
            rtol, atol = 2e-2, 2e-2
        max_abs = (y_triton - y_torch).abs().max().item()
        max_rel = (
            ((y_triton - y_torch).abs() / (y_torch.abs() + 1e-8)).max().item()
        )
        if not (max_abs <= atol + 1e-7 or max_rel <= rtol + 1e-7):
            print(
                f"[mismatch] B={B} | max_abs={max_abs:.3e}, max_rel={max_rel:.3e} (rtol={rtol}, atol={atol})"
            )

        # Measure
        ms_torch = measure_ms(torch_op, iters=iters, warmup=warmup)
        ms_triton = measure_ms(triton_op, iters=iters, warmup=warmup)
        ms_torch_comp = None
        if torch_compiled is not None:

            def torch_compiled_op():
                return torch_compiled()

            ms_torch_comp = measure_ms(
                torch_compiled_op, iters=iters, warmup=warmup
            )

        tflops_torch = approx_tflops(
            B, Cin, Cout, H, W, kH, kW, stride, padding, dilation, ms_torch
        )
        tflops_triton = approx_tflops(
            B, Cin, Cout, H, W, kH, kW, stride, padding, dilation, ms_triton
        )
        tflops_torch_comp = (
            approx_tflops(
                B,
                Cin,
                Cout,
                H,
                W,
                kH,
                kW,
                stride,
                padding,
                dilation,
                ms_torch_comp,
            )
            if ms_torch_comp is not None
            else None
        )

        results.append(
            BenchResult(
                B=B,
                Cin=Cin,
                Cout=Cout,
                H=H,
                W=W,
                kH=kH,
                kW=kW,
                dtype=dtype,
                tf32=use_tf32,
                ms_triton=ms_triton,
                ms_torch=ms_torch,
                ms_torch_compiled=ms_torch_comp,
                tflops_triton=tflops_triton,
                tflops_torch=tflops_torch,
                tflops_torch_compiled=tflops_torch_comp,
            )
        )

        print(
            f"[B={B:>4}] Triton: {ms_triton:8.3f} ms ({tflops_triton:6.2f} TF/s) | "
            f"Torch: {ms_torch:8.3f} ms ({tflops_torch:6.2f} TF/s)"
            + (
                f" | Torch(comp): {ms_torch_comp:8.3f} ms ({tflops_torch_comp:6.2f} TF/s)"
                if ms_torch_comp is not None
                else ""
            )
        )

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filter",
        type=int,
        choices=[1, 3],
        default=3,
        help="kernel size: 1 or 3",
    )
    parser.add_argument(
        "--B",
        type=int,
        nargs="+",
        default=[16, 32, 64, 128],
        help="batch sizes to test",
    )
    parser.add_argument("--Cin", type=int, default=32)
    parser.add_argument("--Cout", type=int, default=64)
    parser.add_argument("--H", type=int, default=224)
    parser.add_argument("--W", type=int, default=224)
    parser.add_argument("--stride", type=int, nargs=2, default=[1, 1])
    parser.add_argument(
        "--pad",
        type=int,
        nargs=2,
        default=[0, 0],
        help="padding (h,w). For 3x3 typical is 1 1",
    )
    parser.add_argument("--dil", type=int, nargs=2, default=[1, 1])
    parser.add_argument(
        "--dtype", type=str, choices=["fp16", "bf16", "fp32"], default="fp16"
    )
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument(
        "--tf32", action="store_true", help="enable TF32 for matmul/cudnn"
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="benchmark torch.compile() as well",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run 1x1 (stride1,pad0) and 3x3 (stride1,pad1) presets",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="optional path to write CSV results",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemError(
            "CUDA device not available. Please run on a GPU machine."
        )

    if args.all:
        # 1x1 preset
        res1 = run_suite(
            B_list=args.B,
            Cin=args.Cin,
            Cout=args.Cout,
            H=args.H,
            W=args.W,
            kH=1,
            kW=1,
            dtype=args.dtype,
            stride=(1, 1),
            padding=(0, 0),
            dilation=(1, 1),
            iters=args.iters,
            warmup=args.warmup,
            use_tf32=args.tf32,
            use_compile=args.compile,
        )
        print("\n--- 3x3 preset (pad=1) ---\n")
        # 3x3 preset (same spatial dims, pad 1)
        res2 = run_suite(
            B_list=args.B,
            Cin=args.Cin,
            Cout=args.Cout,
            H=args.H,
            W=args.W,
            kH=3,
            kW=3,
            dtype=args.dtype,
            stride=(1, 1),
            padding=(1, 1),
            dilation=(1, 1),
            iters=args.iters,
            warmup=args.warmup,
            use_tf32=args.tf32,
            use_compile=args.compile,
        )
        results = res1 + res2
    else:
        k = args.filter
        results = run_suite(
            B_list=args.B,
            Cin=args.Cin,
            Cout=args.Cout,
            H=args.H,
            W=args.W,
            kH=k,
            kW=k,
            dtype=args.dtype,
            stride=tuple(args.stride),
            padding=tuple(args.pad),
            dilation=tuple(args.dil),
            iters=args.iters,
            warmup=args.warmup,
            use_tf32=args.tf32,
            use_compile=args.compile,
        )

    # Pretty print summary and optional CSV
    try:
        import pandas as pd

        rows = [r.__dict__ for r in results]
        df = pd.DataFrame(rows)
        print("\n==== Summary ====")
        print(
            df.to_string(
                index=False,
                formatters={
                    "ms_triton": "{:.3f}".format,
                    "ms_torch": "{:.3f}".format,
                    "tflops_triton": "{:.2f}".format,
                    "tflops_torch": "{:.2f}".format,
                },
            )
        )
        if args.csv:
            df.to_csv(args.csv, index=False)
            print(f"\nSaved CSV: {args.csv}")
    except Exception as e:
        print("[info] pandas not available; skipping table.\n", e)


if __name__ == "__main__":
    main()
