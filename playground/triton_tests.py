import torch
import triton
import triton.language as tl

@triton.jit
def split_kernel(
    x_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
):
    # pid = tl.program_id(axis=0)
    x = tl.load(x_ptr + tl.arange(0, M)[:, None] * N + tl.arange(0,N)[None, :])
    x = x.reshape(M, (N//2), 2)
    print(x)
    # x = x.split()
    # print(x)
    z = tl.arange(0,4)
    print(z)
    # return x


x = torch.arange(32).view(4, 8)
print(x)
split_kernel[(1,)](x, 4, 8)

