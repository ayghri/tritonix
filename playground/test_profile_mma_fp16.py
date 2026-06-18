import pytest
import torch

from profile_mma_fp16 import profile_mma


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required for fp16 mma test")
def test_profile_mma_fp16_executes():
    stats = profile_mma(m=128, n=128, k=128, iters=2, warmup=1)

    assert stats["mean_ms"] > 0
    assert stats["median_ms"] > 0
    assert stats["std_ms"] >= 0
    assert stats["tflops"] > 0
    # Sanity: TFLOPs should be well below an unreasonable upper bound
    # (catches unit mistakes like ms vs s or 1e6 vs 1e9 scaling)
    assert stats["tflops"] < 1e4
