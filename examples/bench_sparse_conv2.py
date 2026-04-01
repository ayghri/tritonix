import time
import random
import torch
import triton

from tritonix.wrappers import conv2d_forward_channelsparse

"""Benchmark: Channel-sparse conv2d (NC fixed per output) vs dense PyTorch conv2d.

Configuration:
  - Vary NC in {6, 8, 16, 48}
  - Keep (N, C_IN, C_OUT, H, W, kernel, stride, padding, dilation) fixed.
  - Construct random channel index lists (without replacement) per output channel.
  - Build sparse weight (C_OUT, NC, KH, KW) and dense weight (C_OUT, C_IN, KH, KW) where
    only the chosen channels are populated (others zero) so correctness check is exact.
  - Measure latency (ms) for: dense (torch), sparse (triton).
  - Report speedup and effective TFLOPs relative to actual non-zero MAC count.

Run: python bench_sparse_conv2.py
"""

DTYPE = torch.float16
DEVICE = triton.runtime.driver.active.get_active_torch_device()


def _select_channels(
    C_in: int, NC: int, C_out: int, deterministic: bool = False
):
    rng = random.Random(0 if deterministic else time.time())
    idx = torch.empty((C_out, NC), dtype=torch.int32, device=DEVICE)
    all_channels = list(range(C_in))
    for co in range(C_out):
        chosen = rng.sample(all_channels, NC)
        idx[co] = torch.tensor(chosen, dtype=torch.int32, device=DEVICE)
    return idx


def _build_dense_weight(sparse_w: torch.Tensor, idx: torch.Tensor, C_in: int):
    # sparse_w: [C_OUT, NC, KH, KW], idx: [C_OUT, NC]
    C_out, NC, KH, KW = sparse_w.shape
    dense_w = torch.zeros(
        (C_out, C_in, KH, KW), device=sparse_w.device, dtype=sparse_w.dtype
    )
    for co in range(C_out):
        dense_w[co, idx[co].long()] = sparse_w[co]
    return dense_w


@torch.no_grad()
def _check_correct(
    input, dense_w, sparse_w, idx, bias, stride, padding, dilation
):
    ref = torch.nn.functional.conv2d(
        input, dense_w, bias, stride, padding, dilation
    )
    out_sparse = conv2d_forward_channelsparse(
        input,
        sparse_w,
        idx,
        bias,
        stride,
        padding,
        dilation,
        out_dtype=input.dtype,
    )
    max_diff = (ref - out_sparse).abs().max().item()
    return max_diff


@torch.no_grad()
def _time(fn, warmup=10, iters=50):
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    end = time.time()
    return (end - start) * 1000.0 / iters  # ms


def benchmark():
    torch.manual_seed(0)
    random.seed(0)
    torch.backends.cudnn.benchmark = True

    # Base problem shape
    N = 32
    C_IN = 256
    C_OUT = 256
    H = W = 64
    KH = KW = 3
    stride = (1, 1)
    padding = (1, 1)
    dilation = (1, 1)

    input = torch.randn(N, C_IN, H, W, device=DEVICE, dtype=DTYPE)
    bias = torch.randn(C_OUT, device=DEVICE, dtype=DTYPE)

    NC_list = [6, 8, 16, 48]

    print(
        "Channel-Sparse Conv2D Benchmark (dtype={}, device={})".format(
            DTYPE, DEVICE
        )
    )
    print(f"Base shape: N={N} C_IN={C_IN} C_OUT={C_OUT} H=W={H} KH=KW={KH}")
    header = f"{'NC':>4}  {'Dense ms':>10}  {'Sparse ms':>10}  {'Speedup':>8}  {'Eff TFLOPs':>10}  {'MaxDiff':>8}"
    print(header)
    print("-" * len(header))

    for NC in NC_list:
        idx = _select_channels(C_IN, NC, C_OUT, deterministic=True)
        sparse_w = torch.randn(C_OUT, NC, KH, KW, device=DEVICE, dtype=DTYPE)
        dense_w = _build_dense_weight(sparse_w, idx, C_IN)

        # Correctness
        max_diff = _check_correct(
            input[:2], dense_w, sparse_w, idx, bias, stride, padding, dilation
        )

        # Timing
        dense_ms = _time(
            lambda: torch.nn.functional.conv2d(
                input, dense_w, bias, stride, padding, dilation
            )
        )
        sparse_ms = _time(
            lambda: conv2d_forward_channelsparse(
                input,
                sparse_w,
                idx,
                bias,
                stride,
                padding,
                dilation,
                out_dtype=DTYPE,
            )
        )

        # Effective FLOPs: 2 * (N * C_OUT * H_out * W_out * (NC * KH * KW))
        H_OUT = (H + 2 * padding[0] - dilation[0] * (KH - 1) - 1) // stride[
            0
        ] + 1
        W_OUT = (W + 2 * padding[1] - dilation[1] * (KW - 1) - 1) // stride[
            1
        ] + 1
        macs = N * C_OUT * H_OUT * W_OUT * NC * KH * KW
        eff_tflops = (2 * macs) / (sparse_ms * 1e-3) / 1e12
        speedup = dense_ms / sparse_ms

        print(
            f"{NC:4d}  {dense_ms:10.3f}  {sparse_ms:10.3f}  {speedup:8.2f}  {eff_tflops:10.2f}  {max_diff:8.4f}"
        )


if __name__ == "__main__":
    benchmark()
