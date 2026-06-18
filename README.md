# Tritonix

[Triton](https://github.com/triton-lang/triton) GPU kernels for neural network ops with an OOM-aware autotuning framework and automatic backend dispatch.

[![Documentation](https://readthedocs.org/projects/tritonix/badge/?version=latest)](https://tritonix.readthedocs.io)

## Installation

Requires Python ≥ 3.11, PyTorch ≥ 2.7, Triton ≥ 3.4.

```bash
pip install tritonix

# from source
git clone https://github.com/ayghri/tritonix.git && cd tritonix && pip install -e .

# with Bayesian tuning (Ax)
pip install tritonix[bayesian]
```

## Quick start

```python
import torch
from tritonix import matmul, conv2d_forward

a = torch.randn(1024, 512, device="cuda", dtype=torch.float16)
b = torch.randn(512, 768, device="cuda", dtype=torch.float16)

# First call benchmarks Triton vs PyTorch and caches the winner per shape
c = matmul(a, b)

# Force a specific backend
with matmul.force_backend("triton"):
    c = matmul(a, b)
```

### Autotuning

Every `@tunable` kernel exposes `.tune()` to search its config space:

```python
from tritonix.ops.matmul import matmul_kernel

# Grid search with trie + unimodality pruning
best = matmul_kernel.tune(launcher, method="grid", verbose=True)

# Bayesian optimization (requires ax-platform)
best = matmul_kernel.tune(launcher, method="bayesian", max_evals=60)
```

See [`examples/tune_matmul.py`](examples/tune_matmul.py) for a full comparison.

## Package structure

```
tritonix/
  autotune.py              # @tunable, TunableKernel, grid/Bayesian search
  dispatcher.py            # DynamicDispatcher, dynamic_dispatch
  ops/
    matmul.py              # matmul_kernel, triton_matmul, matmul (dispatcher)
    swiglu.py              # glu_kernel, swiglu_kernel
    block_sparse_linear.py # BlockSparseLinear nn.Module
    conv2d/
      forward.py           # conv2d_forward_kernel, conv2d_forward
      backward.py          # grad_weight, grad_input, grad_bias kernels
      channelsparse.py     # channel-sparse variant
  utils/
    spaces.py              # PowerOfTwo, Range, Choice, ConfigSpace
    pruners.py             # MonotonicCascadeTrie, CoordinateMonotonicFunction
    torch.py               # TF32/cuDNN optimization toggles
    triton.py              # Triton config helpers
    hilbert.py             # Hilbert curve swizzle
    initialize.py          # Block-sparse tensor creation
```

## Kernels

**Matmul.** Tiled GEMM with 2D swizzle for L2 locality. Split-K variant partitions the K dimension across blocks and reduces via `atomic_add`.

**Conv2D.** Implicit GEMM: input is virtually unfolded into `(N·H_out·W_out, C_in·R·S)` and multiplied against `(C_in·R·S, C_out)`. Supports stride, padding, dilation. Backward kernels for weight, input, and bias gradients.

**Fused GLU/SwiGLU.** Computes `σ(W₁x) ⊙ (W₂x)` in one kernel by interleaving W₁/W₂ columns.

**Block-sparse linear.** `nn.Module` for structured sparsity using lookup tables to index packed non-zero blocks.

## How autotuning works

Triton kernels have many tunable `constexpr` parameters (block sizes, pipeline stages, warp counts). The optimal config depends on problem shape and hardware. Tritonix uses two pruning strategies to cut the search space.

### Monotonic trie (OOM pruning)

Shared memory grows monotonically with block sizes and pipeline stages. If `(block_m=128, block_n=128)` OOMs, then everything with values ≥ those in all memory dimensions will also OOM.

The trie drives the search in **midpoint order**: the first probe is the middle of the unpruned space. On OOM, the upper half is pruned and the lower half is explored — O(log n) probes to find the boundary.

### Performance pruner (unimodality)

For each parameter, with all others fixed, latency is assumed unimodal (single minimum, non-decreasing tails). If benchmarking a slice reveals `L(a) < L(b)` with `a < b`, all configs ≥ b in that slice are pruned.

### Declaring a tunable kernel

```python
from tritonix.autotune import tunable, PowerOfTwo, Choice, Range
import triton
import triton.language as tl

@tunable(
    keys=["m", "n", "k"],
    space={
        "block_m": PowerOfTwo(32, 256),   # {32, 64, 128, 256}
        "block_n": PowerOfTwo(32, 256),
        "block_k": PowerOfTwo(16, 128),   # {16, 32, 64, 128}
        "group_m": Choice([4, 8]),
        "num_stages": Range(2, 5),        # {2, 3, 4}
        "num_warps": Choice([4, 8]),
    },
    memory_params={"block_m", "block_n", "block_k", "num_stages"},
)
@triton.jit
def my_kernel(..., block_m: tl.constexpr, block_n: tl.constexpr, ...):
    ...
```

`memory_params` tells the trie which parameters affect shared memory — only these drive the OOM boundary search.

## Backend dispatch

`matmul` (and other ops) are `DynamicDispatcher` instances that benchmark Triton and PyTorch backends on first call per shape and cache the winner:

```python
from tritonix import matmul

c = matmul(a, b)               # benchmarks on first call, cached after
print(matmul.cache)            # {shape_key: "triton" | "pytorch"}
print(matmul.timings)          # {shape_key: {"triton": ms, "pytorch": ms}}

matmul.clear_cache()           # force re-benchmark

with matmul.force_backend("pytorch"):
    ref = matmul(a, b)         # bypass cache for correctness checks
```

## Examples

| File | What it does |
|------|-------------|
| `examples/tune_matmul.py` | Grid vs Bayesian tuning comparison |

## License

[CC BY-NC 4.0](LICENSE)
