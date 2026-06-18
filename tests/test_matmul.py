import torch
import pytest
from tritonix.ops.matmul import matmul, triton_matmul

DEVICE = "cuda"
DTYPE = torch.float16


@pytest.mark.parametrize(
    "M,N,K", [(512, 512, 512), (1024, 2048, 512), (2048, 2048, 2048)]
)
def test_matmul_correctness(M, N, K):
    a = torch.randn(M, K, device=DEVICE, dtype=DTYPE)
    b = torch.randn(K, N, device=DEVICE, dtype=DTYPE)
    ref = torch.matmul(a, b)
    out = triton_matmul(a, b)
    # fp16 accumulation error grows with K; 1.0 is appropriate for K=2048
    assert torch.allclose(
        ref, out, atol=1.0, rtol=1e-2
    ), f"max diff: {(ref - out).abs().max()}"


@pytest.mark.parametrize("M,N,K", [(512, 512, 512), (2048, 2048, 2048)])
def test_dispatch_correctness(M, N, K):
    a = torch.randn(M, K, device=DEVICE, dtype=DTYPE)
    b = torch.randn(K, N, device=DEVICE, dtype=DTYPE)
    ref = torch.matmul(a, b)

    with matmul.force_backend("triton"):
        tri = matmul(a, b)
    with matmul.force_backend("pytorch"):
        pt = matmul(a, b)

    assert torch.allclose(
        ref, tri, atol=1.0, rtol=1e-2
    ), f"triton max diff: {(ref - tri).abs().max()}"
    assert torch.allclose(
        ref, pt, atol=1.0, rtol=1e-2
    ), f"pytorch max diff: {(ref - pt).abs().max()}"


def test_dispatch_caches_winner():
    matmul.clear_cache()
    a = torch.randn(1024, 1024, device=DEVICE, dtype=DTYPE)
    b = torch.randn(1024, 1024, device=DEVICE, dtype=DTYPE)

    out1 = matmul(a, b)
    assert len(matmul.cache) == 1
    assert list(matmul.cache.values())[0] in ("triton", "pytorch")

    out2 = matmul(a, b)
    assert torch.allclose(out1, out2)
