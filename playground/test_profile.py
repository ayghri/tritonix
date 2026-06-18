import torch
from profile_mma_fp16 import profile_mma

if torch.cuda.is_available():
    print("CUDA available:", torch.cuda.get_device_name(0))
    for accum in ("fp16", "fp32"):
        stats = profile_mma(
            1024,
            1024,
            1024,
            iters=10,
            warmup=5,
            accum=accum,
            use_cuda_graph=True,
        )
        print(accum, stats)
else:
    print("CUDA not available on this machine")
