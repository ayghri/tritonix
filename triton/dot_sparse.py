import torch
import triton
import triton.language as tl
import numpy as np


@triton.jit
def dot_3d_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    lookup_t_ptr,
    L: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    M_idx = tl.arange(0, M)
    N_idx = tl.arange(0, N)
    acc = tl.zeros((N, M, 1), dtype=tl.float32)
    for i in range(0, L, BLOCK_L):
        # K_idx = tl.arange(i, i + BLOCK_K)
        # (L,N)
        K_idx = tl.load(
            lookup_t_ptr
            + N_idx[:, None] * L
            + tl.arange(i, i + BLOCK_L)[None, :]
        )
        a_idx = M_idx[None, :, None] * K + K_idx[:, None, :]
        # j_idx = K_idx % BLOCK_L + i
        # b_idx = j_idx[:, :, None] + N_idx[:, None, None] * K
        b_idx = K_idx[:, :, None] + N_idx[:, None, None] * K
        print("Using features:", K_idx)
        # * N + N_idx[None, :]  # (L, N, N)
        print("a_idx", a_idx)  # shaped (M, K)
        print("b_idx", b_idx)  # shaped (K, N)

        a = tl.load(a_ptr + a_idx)
        b = tl.load(b_ptr + b_idx)
        print("a.shape", a.shape)  # shaped (M, K)
        print("b.shape", b.shape)  # shaped (K, N)
        print(a, b)

        acc += tl.dot(
            a,
            b,
            allow_tf32=False,
        )
    # print("a.shape", a.shape)  # shaped (M, K)
    # print("b.shape", b.shape)  # shaped (K, N)
    # print("c.shape", acc.shape)  # shaped (M, N)
    acc = acc.reshape(N, M)
    tl.store(c_ptr + tl.arange(0, M * N).reshape(M, N), tl.trans(acc))


M = 2
K = 8
# BLOCK_K = 4
BLOCK_L = 4
L = 4
N = 2
torch.manual_seed(0)
a_t = torch.randn(M, K, device="cuda")
b_t = torch.randn(N, K, device="cuda")
c_t = torch.empty(M, N, device="cuda")
lookup_t = []
np.random.seed(0)
for i in range(N):
    lookup_t.append(np.sort(np.random.choice(K, size=L, replace=False)))
lookup_t = torch.tensor(np.array(lookup_t, dtype=np.int32), device="cuda")
print(a_t)
print(b_t)
print(lookup_t)

dot_3d_kernel[(1,)](
    a_t,
    b_t,
    c_t,
    lookup_t,
    tl.constexpr(L),
    tl.constexpr(M),
    tl.constexpr(N),
    tl.constexpr(K),
    tl.constexpr(BLOCK_L),
)

for i in range(N):
    c = b_t[i, lookup_t[i]]
    b_t[i].zero_()
    b_t[i, lookup_t[i]] = c

print(b_t)

norm = torch.norm(c_t - a_t.mm(b_t.t()))
print("Norm of difference:", norm.item())
