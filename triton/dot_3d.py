import torch
import triton
import triton.language as tl
import numpy as np


@triton.jit
def dot_3d_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    L: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # pid = tl.program_id(axis=0)

    # if pid == 0:
    M_idx = tl.arange(0, M)
    N_idx = tl.arange(0, N)
    batch_idx = tl.arange(0, L)
    acc = tl.zeros((L, M, N), dtype=tl.float32)
    for i in range(0, K, BLOCK_K):
        # a = tl.load(a_ptr + tl.arange(0, L * M * K).reshape((L, M, K)))
        # a_idx = LM_idx[:, None] * K + K_idx[None, :]
        K_idx = tl.arange(i, i + BLOCK_K)
        a_idx = (
            batch_idx[:, None, None] * M * K
            + M_idx[None, :, None] * K
            + K_idx[None, None, :]
        )
        # b_idx = LN_idx[None, :] * N + K_idx[:, None]
        b_idx = (
            batch_idx[:, None, None] * K * N
            + K_idx[None, :, None] * N
            + N_idx[None, None, :]
        )
        # b = tl.load(b_ptr + tl.arange(0, L * K * N).reshape((L, K, N)))
        print("a_idx", a_idx)  # shaped (L, M, K)
        print("b_idx", b_idx)  # shaped (L, K, N
        a = tl.load(a_ptr + a_idx)
        b = tl.load(b_ptr + b_idx)
        print("a.shape", a.shape)  # shaped (L, M, K)
        print("b.shape", b.shape)  # shaped (L, K, N)

        acc += tl.dot(
            a,
            b,
            allow_tf32=False,
        )
    print("a.shape", a.shape)  # shaped (L, M, K)
    print("b.shape", b.shape)  # shaped (L, K, N)
    print("c.shape", acc.shape)  # shaped (L, M, N)

    tl.store(c_ptr + tl.arange(0, L * M * N).reshape(L, M, N), acc)


L = 2
M = 4
K = 16
BLOCK_K = 8
K_TRUE = 4
N = 8
torch.manual_seed(0)
a_t = torch.randn(L, M, K, device="cuda")
b_t = torch.randn(L, K, N, device="cuda")
c_t = torch.empty(L, M, N, device="cuda")
lookup_t = []
for i in range(K):
    lookup_t.append(np.sort(np.random.choice(K, size=K_TRUE, replace=False)))
lookup_t = np.array(lookup_t, dtype=np.int32)  # (K, K_TRUE)
lookup_t = torch.tensor(
    lookup_t, device="cuda", dtype=torch.int32
)  # (K, K_TRUE)


dot_3d_kernel[(1,)](
    a_t,
    b_t,
    c_t,
    tl.constexpr(L),
    tl.constexpr(M),
    tl.constexpr(N),
    tl.constexpr(K),
    tl.constexpr(BLOCK_K),
)

norm = torch.norm(
    c_t - torch.einsum("lmk,lkn->lmn", a_t, b_t)
)  # Should be close to 0
print("Norm of difference:", norm.item())
