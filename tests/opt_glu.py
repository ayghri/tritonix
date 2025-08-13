import torch
import triton
import triton.language as tl
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from xformers.ops import SwiGLU
import torch.nn.functional as F

# from time import time
import time

from kernels.fused.glu import glu_kernel
from kernels.matrix.mma import matmul_kernel


# TUNE = True
torch.manual_seed(42)
TUNE = False
DEVICE = triton.runtime.driver.active.get_active_torch_device()
DTYPE = torch.float16
FLOAT_SIZE = torch.finfo(DTYPE).bits // 8
print(f"Using device {DEVICE}, dtype {DTYPE}, {FLOAT_SIZE} bytes.")
torch.set_float32_matmul_precision("high")
M = 1024 * 4
N = 1024 * 8
K = 1024 * 2
a = torch.randn((M, K), device=DEVICE, dtype=DTYPE)
b = torch.randn((K, 2 * N), device=DEVICE, dtype=DTYPE)
# b = torch.randn((2 * N, K), device=DEVICE, dtype=DTYPE).t()
d = torch.randn((N, K), device=DEVICE, dtype=DTYPE)


def glu(a, b, block_m, block_n, block_k, group_m, num_stages, num_warps):
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = (triton.cdiv(M, block_m), triton.cdiv(2 * N, block_n))

    glu_kernel[grid](
        a,
        b,
        c,
        M,
        2 * N,
        K,
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
    # return c

    # return torch.matmul(c, d)
    # return F.do
    # grid = (triton.cdiv(M, block_m), triton.cdiv(K, block_k))
    # e = torch.empty((M, K), device=a.device, dtype=a.dtype)

    # matmul_kernel[grid](
    #     c,
    #     d,
    #     e,
    #     M,
    #     K,
    #     N,
    #     c.stride(0),
    #     c.stride(1),
    #     d.stride(0),
    #     d.stride(1),
    #     e.stride(0),
    #     e.stride(1),
    #     block_m=tl.constexpr(block_m),
    #     block_n=tl.constexpr(block_n),
    #     block_k=tl.constexpr(block_k),
    #     group_m=tl.constexpr(group_m),
    #     num_stages=num_stages,  # type: ignore[no-untyped-call]
    #     num_warps=num_warps,  # type: ignore[no-untyped-call]
    # )
    # return e
    return c


quantiles = [0.5, 0.2]


def benchmark(
    block_m, block_n, block_k, group_m, num_stages, num_warps, split_k=1
):
    try:
        ms, min_ms = triton.testing.do_bench(
            lambda: glu(
                a,
                b,
                block_m=block_m,
                block_n=block_n,
                block_k=block_k,
                group_m=group_m,
                num_stages=num_stages,
                num_warps=num_warps,
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
    Integer(1, 8, name="group_m"),
    Integer(1, 6, name="n_stages"),
    Integer(1, 6, name="n_warps"),
]


@use_named_args(space)
def objective(**params):
    return benchmark(
        block_m=int(2 ** params["b_m"]),
        block_n=int(2 ** params["b_n"]),
        block_k=int(2 ** params["b_k"]),
        group_m=int(params["group_m"]),
        num_stages=int(params["n_stages"]),
        num_warps=int(params["n_warps"]),
    )


from xformers.ops.swiglu_op import _SwiGLUFusedFunc, custom_fwd, DualGemmSiluOp

# w12 = b.view(2, K , N)
# w1  = w12[0]
# w2 = w12[1]
w1 = torch.randn((K, N), device=DEVICE, dtype=DTYPE).t().contiguous()
w2 = torch.randn((K, N), device=DEVICE, dtype=DTYPE).t().contiguous()


class SwiGLUFFNFused(SwiGLU):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        bias: bool = False,
    ) -> None:
        out_features = out_features
        hidden_features = hidden_features

        super().__init__(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            bias=bias,
        )


class CustomGLUOp(_SwiGLUFusedFunc):
    @classmethod
    @custom_fwd(device_type="cuda")
    def forward(cls, ctx, x, w1, b1, w2, b2, w3, b3):
        x1, x2, x4 = DualGemmSiluOp.OPERATOR(x, w1, b1, w2, b2)
        return x4


swiglu_layer = SwiGLUFFNFused(
    in_features=K,
    hidden_features=N,
    out_features=K,
    bias=False,
).to(DEVICE, dtype=DTYPE)
# from skopt import gp_minimize as minimize
from skopt import forest_minimize as minimize

# from skopt import gbrt_minimize as minimize


# # params = best
# for kk, v in params.items():
#     params[kk] = int(v)
# block_m = 2 ** params["b_m"]
# block_n = 2 ** params["b_n"]
# block_k = 2 ** params["b_k"]
# group_m = params["group_m"]
# num_stages = params["n_stages"]
# num_warps = 2 ** params["n_w"]

if __name__ == "__main__":
    if TUNE:

        def callback(res):
            time.sleep(0.1)
            # print(res.x)

        res_gp = minimize(
            objective,
            space,
            n_calls=100,
            random_state=42,
            verbose=True,
            callback=callback,
        )

        print("Best score=%.4f" % res_gp.fun)
        print("Best parameters:", res_gp.x)
        res_gp.x = [int(x) for x in res_gp.x]
        block_m = 2 ** res_gp.x[0]
        block_n = 2 ** res_gp.x[1]
        block_k = 2 ** res_gp.x[2]
        group_m = res_gp.x[3]
        num_stages = res_gp.x[-2]
        num_warps = res_gp.x[-1]
    else:
        # block_m = 128
        # block_n = 128
        # block_k = 32
        # group_m = 4
        # num_stages = 2
        # num_warps = 4

        block_m = 128
        block_n = 128
        block_k = 32
        group_m = 8
        num_stages = 3
        num_warps = 4

    print(
        f"Best parameters: block_m={block_m}, block_n={block_n}, block_k={block_k}, "
        f"group_m={group_m}, num_stages={num_stages}, num_warps={num_warps}"
    )
    _ = swiglu_layer(a)

    ms, min_ms = triton.testing.do_bench(
        lambda: glu(
            a,
            b,
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            group_m=group_m,
            num_stages=num_stages,
            num_warps=num_warps,
        ),
        quantiles=quantiles,
        warmup=400,
        rep=400,
    )  # type: ignore[no-untyped-call]
    print(f"Benchmark result {M}x{K} . {K}x{N}: {ms} ms, min: {min_ms} ms")
    print(
        f"Triton FLOPS: {2 * FLOAT_SIZE * M * N * K / (ms * 1e-3) / (1024**4)} TFLOPS"
    )

    print("W shapes:", w1.shape, w2.shape)
    @torch.no_grad()
    def torch_glu(a):
        # return torch.nn.functional.glu(a @ b, dim=-1)
        # return F.glu(swiglu_layer.w12(a),dim=-1)

        _, _, x4= DualGemmSiluOp.OPERATOR(a, w1, None, w2, None)
        # return CustomGLUOp.apply(
        # a, w1, None, w2, None, None, None
        # )
        # return DualGemmSiluOp.apply(a, w1, w1, w1, w1, 0.0, 0.0)
        # return x1, x2
        return x4

    # torch_matmul = torch.compile(lambda: torch.matmul(a, b))
    # compiled_torch_glu = torch.compile(torch_glu)
    compiled_torch_glu = torch_glu

    print(
        "Output shape:",
        compiled_torch_glu(a).shape,
        # compiled_torch_glu(a)[0].shape,
        # compiled_torch_glu(a)[1].shape,
    )

    ms, min_ms = triton.testing.do_bench(
        lambda: torch_glu(a),
        quantiles=quantiles,
        warmup=400,
        rep=400,
    )  # type: ignore[no-untyped-call]
    print(
        f"Benchmark torch result {M}x{K} . {K}x{N}: {ms} ms, min: {min_ms} ms"
    )
    print(
        f"Torch FLOPS: {2 * FLOAT_SIZE * M * N * K / (ms * 1e-3) / (1024**4)} TFLOPS"
    )
    # ms, min_ms = triton.testing.do_bench(
    #     lambda: compiled_torch_glu(a),
    #     quantiles=quantiles,
    #     warmup=200,
    #     rep=200,
    # )  # type: ignore[no-untyped-call]
    # print(
    #     f"Benchmark torch result {M}x{K} . {K}x{N}: {ms} ms, min: {min_ms} ms"
    # )
    # print(
    #     f"Torch FLOPS: {2 * FLOAT_SIZE * M * N * K / (ms * 1e-3) / (2**40)} TFLOPS"
    # )
    # print(swiglu_layer.w12, swiglu_layer.op)
