import triton
import triton.language as tl


@triton.jit
def swizzle2d_rows(i, j, size_i, size_j, size_g):
    ij = i * size_j + j
    num_groups_per_stripe = size_g * size_j
    group_id = ij // num_groups_per_stripe
    group_start = group_id * size_g
    group_size = tl.minimum(size_i - group_start, size_g)

    pid_m = group_start + (ij % num_groups_per_stripe) % group_size
    pid_n = (ij % num_groups_per_stripe) // group_size
    group_m = pid_m // size_g
    start = 0
    if group_m % 2 == 1:
        start = size_j - 1
        pid_n = size_j - 1 - pid_n
    if (start - pid_n) % 2 == 1:
        pid_m = (
            pid_m
            - 2 * ((ij % num_groups_per_stripe) % group_size)
            - 1
            + group_size
        )
    return pid_m, pid_n


@triton.jit
def matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    m,
    n,
    k,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    group_m: tl.constexpr,
    use_tf32: tl.constexpr = tl.constexpr(True),
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    num_pid_m = tl.cdiv(m, block_m)
    num_pid_n = tl.cdiv(n, block_n)

    pid_m, pid_n = tl.swizzle2d(pid_m, pid_n, num_pid_m, num_pid_n, group_m)

    # for compiler
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)
    # ----------------------
    offs_am = pid_m * block_m + tl.arange(0, block_m)
    offs_bn = pid_n * block_n + tl.arange(0, block_n)
    offs_k = tl.arange(0, block_k)

    a_ptrs = a_ptr + (
        offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    )

    b_ptrs = b_ptr + (
        offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    )

    accumulator = tl.zeros((block_m, block_n), dtype=tl.float32)
    for k_loop in range(0, tl.cdiv(k, block_k)):
        k_remaining = k - k_loop * block_k

        # a_mask = (offs_am[:, None] < m) & (offs_k[None, :] < k_remaining)
        # b_mask = (offs_bn[None, :] < n) & (offs_k[:, None] < k_remaining)
        mask = offs_k < k_remaining
        a = tl.load(a_ptrs, mask=mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=mask[:, None], other=0.0)

        if use_tf32:
            accumulator = tl.dot(a, b, accumulator, input_precision="tf32")
            # accumulator = tl.dot(a, b, accumulator)
        else:
            accumulator = tl.dot(a, b, accumulator, input_precision="ieee")
        a_ptrs += block_k * stride_ak
        b_ptrs += block_k * stride_bk

    c = accumulator.to(c_ptr.dtype.element_ty)

    offs_cm = pid_m * block_m + tl.arange(0, block_m)
    offs_cn = pid_n * block_n + tl.arange(0, block_n)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < m) & (offs_cn[None, :] < n)
    tl.store(c_ptrs, c, mask=c_mask)


matmul_kernel.specialize_keys = [
    "m",
    "n",
    "k",
]


@triton.jit
def grouped_launch(
    pid,
    m,
    n,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    group_m: tl.constexpr,
):
    grid_m = tl.cdiv(m, block_m)
    grid_n = tl.cdiv(n, block_n)

    width = group_m * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * group_m, group_m)

    pid_m = group_id * group_m + (pid % group_size)
    pid_n = (pid % width) // group_size

    return pid_m, pid_n


@triton.jit
def gemm_splitk_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    m,
    n,
    k,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    group_m: tl.constexpr,
    split_k: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    grid_k = tl.cdiv(k, block_k * split_k)
    num_pid_m = tl.cdiv(m, block_m)
    num_pid_n = tl.cdiv(n, block_n)
    pid_m, pid_n = tl.swizzle2d(pid_m, pid_n, num_pid_m, num_pid_n, group_m)
    # pid_m, pid_n = swizzle2d_rows(pid_m, pid_n, num_pid_m, num_pid_n, group_m)

    # pid = tl.program_id(0)
    # pid_k = tl.program_id(1)
    # grid_k = tl.cdiv(k, block_k * split_k)
    # pid_m, pid_n = tl.swizzle2d(pid_m, pid_n, num_pid_m, num_pid_n, group_m)
    # pid_m, pid_n = grouped_launch(pid, m, n, block_m, block_n, group_m)

    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_n = pid_n * block_n + tl.arange(0, block_n)
    offs_k = pid_k * block_k + tl.arange(0, block_k)

    offs_am = tl.max_contiguous(tl.multiple_of(offs_m, block_m), block_m)
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_n, block_n), block_n)

    # a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    # b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    a_ptrs = a_ptr + (
        offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    )
    b_ptrs = b_ptr + (
        offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    )

    # Create the masks for M and N dimensions
    # mask_m = (offs_m < m)[:, None]
    # mask_n = (offs_n < n)[None, :]

    acc = tl.zeros((block_m, block_n), dtype=tl.float32)
    for k_loop in range(0, grid_k):
        k_remaining = k - k_loop * (block_k * split_k)
        a = tl.load(a_ptrs, mask=(offs_k[None, :] < k_remaining), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < k_remaining), other=0.0)

        acc = tl.dot(a, b, acc, out_dtype=tl.float32)

        a_ptrs += block_k * split_k * stride_ak
        b_ptrs += block_k * split_k * stride_bk

    acc = acc.to(c_ptr.dtype.element_ty)

    # Note: The write operation was already correctly masked
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_n = pid_n * block_n + tl.arange(0, block_n)

    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    mask = (offs_m < m)[:, None] & (offs_n < n)[None, :]
    # mask = mask_m & mask_n
    # if split_k > 1:
    tl.atomic_add(c_ptrs, acc, mask=mask, sem="relaxed")
    # else:
    # tl.store(c_ptrs, acc, mask=mask)


def zero_output(*args, **kwargs):
    if kwargs["split_k"] > 1:
        args[2].zero_()


gemm_splitk_kernel.add_pre_run_hook(zero_output)
# @triton.jit
# def gemm_splitk_kernel(
#     a_ptr,
#     b_ptr,
#     c_ptr,
#     m,
#     n,
#     k,
#     stride_am,
#     stride_ak,
#     stride_bk,
#     stride_bn,
#     stride_cm,
#     stride_cn,
#     block_m: tl.constexpr,
#     block_n: tl.constexpr,
#     block_k: tl.constexpr,
#     split_k: tl.constexpr,
#     group_m: tl.constexpr,
# ):
#     pid_m = tl.program_id(0)
#     pid_n = tl.program_id(1)
#     pid_k = tl.program_id(2)
#     grid_k = tl.cdiv(k, block_k * split_k)

#     # pid_m, pid_n = grouped_launch(pid, m, n, block_m, block_n, group_m)
#     pid_m, pid_n = tl.swizzle2d(
#         pid_m, pid_n, tl.cdiv(m, block_m), tl.cdiv(n, block_n), group_m
#     )

#     offs_m = pid_m * block_m + tl.arange(0, block_m)
#     offs_n = pid_n * block_n + tl.arange(0, block_n)
#     offs_k = pid_k * block_k + tl.arange(0, block_k)

#     offs_am = tl.max_contiguous(tl.multiple_of(offs_m, block_m), block_m)
#     offs_bn = tl.max_contiguous(tl.multiple_of(offs_n, block_n), block_n)

#     a_ptrs = a_ptr + (
#         offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
#     )
#     b_ptrs = b_ptr + (
#         offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
#     )
#     # a_offs = (
#     #     offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
#     # )
#     # b_offs = (
#     #     offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
#     # )


#     acc = tl.zeros((block_m, block_n), dtype=tl.float32)
#     for k_ in range(0, grid_k):
#         # print("pid_m=", pid_m, "pid_n=", pid_n, "pid_k=", pid_k,
#             #   "a_offs=", a_offs, "b_offs=", b_offs)
#         k_remaining = k - k_ * (block_k * split_k)

#         a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
#         b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)

#         acc = tl.dot(a, b, acc, out_dtype=tl.float32)

#         a_ptrs += block_k * split_k * stride_ak
#         b_ptrs += block_k * split_k * stride_bk
#         # a_offs += block_k * split_k * stride_ak
#         # b_offs += block_k * split_k * stride_bk

#     acc.to(c_ptr.dtype.element_ty)

#     offs_m = pid_m * block_m + tl.arange(0, block_m)
#     offs_n = pid_n * block_n + tl.arange(0, block_n)

#     c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
#     mask = (offs_m < m)[:, None] & (offs_n < n)[None, :]

#     tl.atomic_add(c_ptrs, acc, mask=mask)


gemm_splitk_kernel.specialize_keys = [
    "m",
    "n",
    "k",
]
