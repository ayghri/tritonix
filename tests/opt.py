import torch
import triton
import triton.language as tl
from skopt.space import Real, Integer
from skopt.utils import use_named_args

# from time import time
import time


from kernels.matrix.mma import matmul_kernel
from kernels.matrix.mma import gemm_splitk_kernel
from kernels.utils.trie import MonotonicCascadeTrie


DEVICE = triton.runtime.driver.active.get_active_torch_device()
DTYPE = torch.float16
FLOAT_SIZE = torch.finfo(DTYPE).bits // 8
print(f"Using device {DEVICE}, dtype {DTYPE}, {FLOAT_SIZE} bytes.")
torch.set_float32_matmul_precision("high")
m = 1024 * 8
n = 1024 * 16
k = 1024 * 2
a = torch.randn((m, k), device=DEVICE, dtype=DTYPE)
b = torch.randn((k, n), device=DEVICE, dtype=DTYPE)


def matmul(
    block_m, block_n, block_k, group_m, num_stages, num_warps, split_k=1
):
    c = torch.empty((m, n), device=a.device, dtype=a.dtype)
    

    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    # grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n), split_k)
    # def grid(META):
    #     return (
    #         triton.cdiv(m, META["block_m"]),
    #         triton.cdiv(n, META["block_n"]),
    #     )
    # def grid(META):
    #     return (
    #         triton.cdiv(m, META["block_m"]),
    #         triton.cdiv(n, META["block_n"]),
    #         META["split_k"],
    #     )

    # matmul_kernel[grid](
    matmul_kernel[grid](
        # gemm_splitk_kernel[grid](
        a,
        b,
        c,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        block_m=tl.constexpr(block_m),
        block_n=tl.constexpr(block_n),
        block_k=tl.constexpr(block_k),
        group_m=tl.constexpr(group_m),
        # split_k=tl.constexpr(split_k),
        num_stages=num_stages,  # type: ignore[no-untyped-call]
        num_warps=num_warps,  # type: ignore[no-untyped-call]
    )

    return c


quantiles = [0.5, 0.2]


def benchmark(
    block_m, block_n, block_k, group_m, num_stages, num_warps, split_k=1
):
    try:
        ms, min_ms = triton.testing.do_bench(
            lambda: matmul(
                block_m=block_m,
                block_n=block_n,
                block_k=block_k,
                group_m=group_m,
                num_stages=num_stages,
                num_warps=num_warps,
                split_k=split_k,
            ),
            quantiles=quantiles,
            warmup=100,
            rep=200,
        )  # type: ignore[no-untyped-call]
        loss = -1 / ms
    except Exception as _:
        # time.sleep(0.1)
        loss = 0
    return loss


# print("Running benchmark...")
# result = benchmark(
#     block_m=32,
#     block_n=32,
#     block_k=32,
#     group_m=8,
#     num_stages=2,
#     num_warps=4)
# print(f"Benchmark result: {result}")

# result = benchmark(
#     block_m=128,
#     block_n=128,
#     block_k=128,
#     group_m=8,
#     num_stages=20,
#     num_warps=4)
# print(f"Benchmark result: {result}")


space = [
    Integer(4, 8, name="b_m"),
    Integer(4, 8, name="b_n"),
    Integer(4, 7, name="b_k"),
    Integer(1, 24, name="group_m"),
    # Integer(2, 16, name="split_k"),
    Integer(1, 6, name="n_stages"),
    Integer(1, 6, name="n_warps"),
    # Integer(1, 10, name="n_warps"),
]


@use_named_args(space)
def objective(**params):
    # print(params)
    # print(
    #     "Arguments are:",
    #     f"b_m={b_m}, b_n={b_n}, b_k={b_k}, group_m={group_m}, n_stages={n_stages}, n_w={n_w}",
    # )
    # print("Running benchmark with params:", params)
    return benchmark(
        block_m=int(2 ** params["b_m"]),
        block_n=int(2 ** params["b_n"]),
        block_k=int(2 ** params["b_k"]),
        group_m=int(params["group_m"]),
        # split_k=int(params["split_k"]),
        num_stages=int(params["n_stages"]),
        num_warps=int(params["n_warps"]),
    )


from skopt import gp_minimize
#from skopt import forest_minimize as minimize
# from skopt import gbrt_minimize as minimize


def callback(res):
    time.sleep(0.1)
    # print(res.x)


res_gp = minimize(
    objective,
    space,
    n_calls=100,
    # random_state=42,
    verbose=True,
    callback=callback,
)

print("Best score=%.4f" % res_gp.fun)


# params = best
# for kk, v in params.items():
#     params[kk] = int(v)
# block_m = 2 ** params["b_m"]
# block_n = 2 ** params["b_n"]
# block_k = 2 ** params["b_k"]
# group_m = params["group_m"]
# num_stages = params["n_stages"]
# num_warps = 2 ** params["n_w"]
print("Best parameters:", res_gp.x)
res_gp.x = [int(x) for x in res_gp.x]
block_m = 2 ** res_gp.x[0]
block_n = 2 ** res_gp.x[1]
block_k = 2 ** res_gp.x[2]
group_m = res_gp.x[3]
# split_k = res_gp.x[4]
num_stages = res_gp.x[-2]
num_warps = res_gp.x[-1]
if __name__ == "__main__":
    block_m = 128
    block_n = 128
    block_k = 32
    group_m = 4
    # split_k = res_gp.x[4]
    num_stages = 2
    num_warps = 4

    print(
        f"Best parameters: block_m={block_m}, block_n={block_n}, block_k={block_k}, "
        # f"group_m={group_m}, split_k={split_k}, num_stages={num_stages}, num_warps={num_warps}"
        f"group_m={group_m}, num_stages={num_stages}, num_warps={num_warps}"
    )

    ms, min_ms = triton.testing.do_bench(
        lambda: matmul(
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            group_m=group_m,
            # split_k=split_k,
            num_stages=num_stages,
            num_warps=num_warps,
        ),
        quantiles=quantiles,
        warmup=200,
        rep=200,
    )  # type: ignore[no-untyped-call]
    print(f"Benchmark result {m}x{k} . {k}x{n}: {ms} ms, min: {min_ms} ms")
    print(
        f"Triton FLOPS: {FLOAT_SIZE * m * n * k / (ms * 1e-3) / (1024**4)} TFLOPS"
    )

    torch_matmul = torch.compile(lambda: torch.matmul(a, b))
    ms, min_ms = triton.testing.do_bench(
        lambda: torch_matmul(),
        quantiles=quantiles,
        warmup=200,
        rep=200,
    )  # type: ignore[no-untyped-call]
    print(
        f"Benchmark torch result {m}x{k} . {k}x{n}: {ms} ms, min: {min_ms} ms"
    )
    print(
        f"Torch FLOPS: {FLOAT_SIZE * m * n * k / (ms * 1e-3) / (1024**4)} TFLOPS"
    )
