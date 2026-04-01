# Tritonix

[Triton](https://github.com/triton-lang/triton) GPU kernels for neural network ops with an autotuning framework that prunes OOM configurations before benchmarking.

## Installation

Requires Python 3.10+, PyTorch >= 2.7, Triton >= 3.4.

```bash
pip install tritonix

# or from source
git clone https://github.com/ayghri/tritonix.git && cd tritonix && pip install -e .

# for Bayesian tuning (Ax)
pip install tritonix[bayesian]
```

## Quick start

```python
import torch
from tritonix import matmul, conv2d_forward

# matmul
a = torch.randn(1024, 512, device="cuda", dtype=torch.float16)
b = torch.randn(512, 768, device="cuda", dtype=torch.float16)
c = matmul(a, b)

# conv2d
x = torch.randn(8, 64, 32, 32, device="cuda", dtype=torch.float16)
w = torch.randn(128, 64, 3, 3, device="cuda", dtype=torch.float16)
y = conv2d_forward(x, w, padding=(1, 1))
```

### Autotuning

Every `@tunable` kernel exposes `.tune()` to search its parameter space. See [`examples/tune_matmul.py`](examples/tune_matmul.py) for a full working example.

```python
from tritonix.ops.matmul import matmul_kernel

best = matmul_kernel.tune(launcher, method="grid", verbose=True)
```

## Package structure

```
tritonix/
  autotune.py              # @tunable, TunableKernel, grid/Bayesian search
  ops/
    matmul.py              # matmul_kernel, gemm_splitk_kernel
    swiglu.py              # glu_kernel, swiglu_kernel
    block_sparse_linear.py # BlockSparseLinear nn.Module
    conv2d/
      forward.py           # conv2d_forward_kernel
      backward.py          # grad_weight, grad_input, grad_bias kernels
      channelsparse.py     # channel-sparse variant
  utils/
    trie.py                # MonotonicCascadeTrie
    triton.py              # config builders, wrap_autotuner
    torch.py               # TF32/cuDNN optimization toggles
    hilbert.py             # Hilbert curve swizzle
    initialize.py          # block-sparse tensor creation
```

## Kernels

**Matmul.** Tiled GEMM with 2D swizzle for L2 locality. The split-K variant partitions the reduction dimension across blocks and reduces via `atomic_add`.

**Conv2D.** Implicit GEMM: the input is virtually unfolded into `(N*H_out*W_out, C_in*R*S)` and multiplied against `(C_in*R*S, C_out)`. Supports stride, padding, dilation. Backward kernels for weight, input, and bias gradients.

**Fused GLU/SwiGLU.** Computes `sigma(W1 @ X) * (W2 @ X)` in one kernel by interleaving W1/W2 columns. `swiglu_kernel` adds a second matmul stage with `atomic_add` reduction.

**Block-sparse linear.** `nn.Module` for structured sparsity using lookup tables to index packed non-zero blocks.

## How autotuning works

Triton kernels have many tunable `constexpr` parameters (block sizes, pipeline stages, warp counts). The optimal config depends on problem size and hardware. Tritonix uses two pruning strategies to cut 30-60% of the search space.

### Monotonic trie (OOM pruning)

Shared memory grows monotonically with block sizes and pipeline stages. If `(block_m=128, block_n=128, block_k=64)` OOMs, then anything with values >= those in *all* smem dimensions will also OOM.

The trie tracks minimal failure points and prunes subtrees in two phases:

1. **Boundary detection.** Probe the midpoint of unpruned space, then binary-search each dimension upward to find the OOM boundary in O(log n) probes.
2. **Exhaustive sweep.** Iterate remaining configs, skipping pruned subtrees via prefix checks.

### Performance pruner (unimodality)

For each parameter, with all others fixed, latency is assumed unimodal (single minimum). If two benchmarks in the same slice show `L(a) < L(b)` with `a < b`, everything >= b is pruned.

### Declaring a tunable kernel

```python
from tritonix.autotune import tunable, PowerOfTwo, Choice, Range

@tunable(
    keys=["m", "n", "k"],
    space={
        "block_m": PowerOfTwo(32, 256),    # {32, 64, 128, 256}
        "block_n": PowerOfTwo(32, 256),
        "block_k": PowerOfTwo(16, 128),    # {16, 32, 64, 128}
        "group_m": Choice([4, 8]),
        "num_stages": Range(2, 5),         # {2, 3, 4, 5}
        "num_warps": Choice([4, 8]),
    },
)
@triton.jit
def my_kernel(...):
    ...
```

`PowerOfTwo(lo, hi)` generates powers of 2 in [lo, hi]. `Range(lo, hi)` generates integers. `Choice(list)` passes values through.

## Examples

| File | What it does |
|------|-------------|
| `examples/tune_matmul.py` | Tuning with/without trie pruning |
| `examples/bench_triton_matmul_kernel.py` | Matmul TFLOPS benchmark |
| `examples/bench_conv2d.py` | Conv2D benchmark |
| `examples/dense_block_sparse_mma.py` | Block-sparse MMA demo |

## License

[CC BY-NC 4.0](LICENSE)
