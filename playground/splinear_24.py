import torch
from torch.sparse import to_sparse_semi_structured, SparseSemiStructuredTensor
from torch.utils.benchmark import Timer
SparseSemiStructuredTensor._FORCE_CUTLASS = True

# mask Linear weight to be 2:4 sparse
M, N , K =  4096, 4096, 4096
mask = torch.Tensor([0, 0, 1, 1]).tile((3072, 2560)).cuda().bool()
linear = torch.nn.Linear(4096, 4096).half().cuda().eval()
linear.weight = torch.nn.Parameter(mask * linear.weight)

x = torch.rand(3072, 10240).half().cuda()

with torch.inference_mode():
    dense_output = linear(x)
    dense_t = Timer(stmt="linear(x)",
                    globals={"linear": linear,
                             "x": x}).blocked_autorange().median * 1e3

    # accelerate via SparseSemiStructuredTensor
    linear.weight = torch.nn.Parameter(to_sparse_semi_structured(linear.weight))

    sparse_output = linear(x)
    sparse_t = Timer(stmt="linear(x)",
                    globals={"linear": linear,
                             "x": x}).blocked_autorange().median * 1e3

    # sparse and dense matmul are numerically equivalent
    # On an A100 80GB, we see: `Dense: 0.870ms Sparse: 0.630ms | Speedup: 1.382x`
    assert torch.allclose(sparse_output, dense_output, atol=1e-3)
    print(f"Dense: {dense_t:.3f}ms Sparse: {sparse_t:.3f}ms | Speedup: {(dense_t / sparse_t):.3f}x")