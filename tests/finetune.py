import torch
import triton
from hyperopt import STATUS_FAIL, STATUS_OK
from hyperopt import hp
from hyperopt.pyll import scope
from hyperopt import fmin, tpe, Trials
import triton.language as tl
triton.runtime.jit


from tritonix.matrix.mma import matmul_kernel


DEVICE = triton.runtime.driver.active.get_active_torch_device()
DTYPE = torch.float32
FLOAT_SIZE = torch.finfo(DTYPE).bits // 8
print(f"Using device {DEVICE}, dtype {DTYPE}, {FLOAT_SIZE} bytes.")
m = 1024
n = 1024 * 2
k = 1024 * 16
a = torch.randn((m, k), device=DEVICE, dtype=DTYPE)
b = torch.randn((k, n), device=DEVICE, dtype=DTYPE)


def matmul(block_m, block_n, block_k, group_m, num_stages, num_warps):
    c = torch.empty((m, n), device=a.device, dtype=a.dtype)

    # def grid(META):
    #     return (
    #         triton.cdiv(m, META["block_m"]),
    #         triton.cdiv(n, META["block_n"]),
    #         META["split_k"],
    #     )

    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    # def grid(META):
    #     return (
    #         triton.cdiv(m, META["block_m"]),
    #         triton.cdiv(n, META["block_n"]),
    #     )

    matmul_kernel[grid](
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
        num_stages=num_stages,  # type: ignore[no-untyped-call]
        num_warps=num_warps,  # type: ignore[no-untyped-call]
    )

    return c


quantiles = [0.5, 0.2]


def benchmark(block_m, block_n, block_k, group_m, num_stages, num_warps):
    try:
        ms, min_ms = triton.testing.do_bench(
            lambda: matmul(
                block_m=block_m,
                block_n=block_n,
                block_k=block_k,
                group_m=group_m,
                num_stages=num_stages,
                num_warps=num_warps,
            ),
            quantiles=quantiles,
            warmup=200,
            rep=200,
        )  # type: ignore[no-untyped-call]
        loss = -1 / ms
        status = STATUS_OK
    except Exception as _:
        loss = 0
        status = STATUS_FAIL
    return {"loss": loss, "status": status}


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


@scope.define
def objective(params):
    # print(params)
    # print(
    #     "Arguments are:",
    #     f"b_m={b_m}, b_n={b_n}, b_k={b_k}, group_m={group_m}, n_stages={n_stages}, n_w={n_w}",
    # )
    return benchmark(
        block_m=int(2 ** params["b_m"]),
        block_n=int(2 ** params["b_n"]),
        block_k=int(2 ** params["b_k"]),
        group_m=int(params["group_m"]),
        num_stages=int(params["n_stages"]),
        num_warps=int(2 ** params["n_w"]),
    )


space = {
    "b_m": hp.choice("b_m", list(range(4, 9))),
    "b_n": hp.choice("b_n", list(range(4, 9))),
    "b_k": hp.choice("b_k", list(range(4, 9))),
    "group_m": hp.choice("group_m", [1]+list(range(2, 25, 2))),
    "n_stages": hp.choice("n_stages", list(range(1, 9))),
    "n_w": hp.choice("n_w", list(range(0, 9))),
}


trials = Trials()
best = fmin(
    objective, space=space, algo=tpe.suggest, max_evals=100, trials=trials
)
print(trials.trials)

params = best
for kk, v in params.items():
    params[kk] = int(v)
block_m = 2 ** params["b_m"]
block_n = 2 ** params["b_n"]
block_k = 2 ** params["b_k"]
group_m = params["group_m"]
num_stages = params["n_stages"]
num_warps = 2 ** params["n_w"]

print(
    f"Best parameters: block_m={block_m}, block_n={block_n}, block_k={block_k}, "
    f"group_m={group_m}, num_stages={num_stages}, num_warps={num_warps}"
)

ms, min_ms = triton.testing.do_bench(
    lambda: matmul(
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        group_m=group_m,
        num_stages=num_stages,
        num_warps=num_warps,
    ),
    quantiles=quantiles,
    warmup=200,
    rep=200,
)  # type: ignore[no-untyped-call]
print(f"Benchmark result {m}x{k} . {k}x{n}: {ms} ms, min: {min_ms} ms")

torch_matmul = torch.compile(lambda: torch.matmul(a, b), mode="fullgraph")
ms, min_ms = triton.testing.do_bench(
    lambda: torch_matmul(),
    quantiles=quantiles,
    warmup=200,
    rep=200,
)  # type: ignore[no-untyped-call]
print(f"Benchmark torch result {m}x{k} . {k}x{n}: {ms} ms, min: {min_ms} ms")
