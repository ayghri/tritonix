import torch
from tritonix.utils.torch import enable_cudnn_optimizations
m = 1024
n = 2048
k = 1024 * 8
a = torch.randn((m, k), device="cuda", dtype=torch.float16)
b = torch.randn((k, n), device="cuda", dtype=torch.float16)
enable_cudnn_optimizations()

# matmul = torch.compile(lambda a, b: torch.matmul(a, b), fullgraph=True)
matmul = torch.matmul

for _ in range(5):
    d = matmul(a, b)

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CUDA,
    ],
    schedule=torch.profiler.schedule(
        wait=1,
        warmup=1,
        active=3),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./runs/mma'),
) as prof:
    with torch.cuda.nvtx.range("My_GEMM"):
        c = matmul(a, b)
print("Gemm done, profile:")