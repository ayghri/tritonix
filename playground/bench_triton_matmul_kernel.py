import statistics

import torch
import triton


SHAPES = [
    (256, 256, 256),
    (512, 512, 512),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
]

DTYPE = torch.float16
WARMUP = 10
ITERS = 50


def tflops(m: int, n: int, k: int, mean_ms: float) -> float:
    return (2.0 * m * n * k) / (mean_ms * 1e9)


def time_loop_ms(fn, *, warmup: int, iters: int) -> tuple[float, float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    lapses = []

    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        lapses.append(start.elapsed_time(end))
    return (
        statistics.fmean(lapses),
        statistics.median(lapses),
        statistics.pstdev(lapses),
    )


def main() -> None:
    from tritonix.mma.dense import matmul_kernel
    from tritonix.utils.triton import get_autotune_configs, wrap_autotuner

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    print(
        f"Device: {props.name} (SM {props.major}{props.minor}), dtype={DTYPE}"
    )
    print(f"Warmup/iters: {WARMUP}/{ITERS}")

    configs = get_autotune_configs()
    tuned = wrap_autotuner(matmul_kernel, configs)

    print(
        "M      N      K | best_config                 |  triton_ms  triton_TF |   torch_ms   torch_TF | speedup"
    )
    print("-" * 110)

    for m, n, k in SHAPES:
        a = torch.randn((m, k), device=device, dtype=DTYPE).contiguous()
        b = torch.randn((k, n), device=device, dtype=DTYPE).contiguous()
        out_triton = torch.empty((m, n), device=device, dtype=DTYPE)
        out_torch = torch.empty((m, n), device=device, dtype=DTYPE)

        def grid(meta):
            return (
                triton.cdiv(m, meta["block_m"]),
                triton.cdiv(n, meta["block_n"]),
            )

        # Compile + autotune once per shape.
        tuned[grid](
            a,
            b,
            out_triton,
            m,
            n,
            k,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            out_triton.stride(0),
            out_triton.stride(1),
        )
        torch.cuda.synchronize()

        cache_key = (m, n, k, str(a.dtype), str(b.dtype), str(out_triton.dtype))
        best = tuned.cache.get(cache_key)
        if best is None:
            best_str = "(missing)"
        else:
            kw = best.kwargs
            best_str = (
                f"{kw['block_m']}x{kw['block_n']}x{kw['block_k']} "
                f"gm{kw['group_m']} w{best.num_warps} s{best.num_stages}"
            )

        def run_triton() -> None:
            tuned[grid](
                a,
                b,
                out_triton,
                m,
                n,
                k,
                a.stride(0),
                a.stride(1),
                b.stride(0),
                b.stride(1),
                out_triton.stride(0),
                out_triton.stride(1),
            )

        def run_torch() -> None:
            torch.matmul(a, b, out=out_torch)

        tr_mean, _, _ = time_loop_ms(run_triton, warmup=WARMUP, iters=ITERS)
        th_mean, _, _ = time_loop_ms(run_torch, warmup=WARMUP, iters=ITERS)

        tr_tf = tflops(m, n, k, tr_mean)
        th_tf = tflops(m, n, k, th_mean)
        speedup = th_mean / tr_mean if tr_mean > 0 else float("inf")

        print(
            f"{m:6d} {n:6d} {k:6d} | "
            f"{best_str:<22s} | "
            f"{tr_mean:10.3f} {tr_tf:10.2f} | "
            f"{th_mean:10.3f} {th_tf:10.2f} | "
            f"{speedup:5.2f}x"
        )


if __name__ == "__main__":
    main()
