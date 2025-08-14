import triton
import triton.language as tl


@triton.jit
def silu(x):
    return tl.sigmoid(x) * x


@triton.jit
def glu_kernel(
    x_ptr,
    w12_ptr,
    z_ptr,
    m,
    n,
    k,
    stride_xm,
    stride_xk,
    stride_wk,
    stride_wn,
    stride_zm,
    stride_zn,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    group_m: tl.constexpr,
    use_tf32: tl.constexpr = tl.constexpr(True),
    activation: tl.constexpr = tl.constexpr("silu"),
):
    """
    Kernel for computing the GLU C = sigma(W1@X).(W12@X)
    Done by using interleaved columns of W1 and W2 as W12,
    then returning  sigma(A[:,::2]).A[:,1::2]) with A = W12@X,
    """

    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    num_pid_m = tl.cdiv(m, block_m)
    num_pid_n = tl.cdiv(n, block_n)

    pid_m, pid_n = tl.swizzle2d(pid_m, pid_n, num_pid_m, num_pid_n, group_m)

    # for compiler
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(stride_xm > 0)
    tl.assume(stride_xk > 0)
    tl.assume(stride_wn > 0)
    tl.assume(stride_wk > 0)
    tl.assume(stride_zm > 0)
    tl.assume(stride_zn > 0)
    # ----------------------

    offs_am = pid_m * block_m + tl.arange(0, block_m)
    offs_bn = pid_n * block_n + tl.arange(0, block_n)
    offs_k = tl.arange(0, block_k)

    x_ptrs = x_ptr + (
        offs_am[:, None] * stride_xm + offs_k[None, :] * stride_xk
    )

    w_ptrs = w12_ptr + (
        offs_k[:, None] * stride_wk + offs_bn[None, :] * stride_wn
    )

    accumulator = tl.zeros((block_m, block_n), dtype=tl.float32)

    for k_loop in range(0, tl.cdiv(k, block_k)):
        k_remaining = k - k_loop * block_k

        mask = offs_k < k_remaining
        a = tl.load(x_ptrs, mask=mask[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=mask[:, None], other=0.0)

        if use_tf32:
            accumulator = tl.dot(a, w, accumulator)
            # accumulator = tl.dot(a, b, accumulator)
        else:
            accumulator = tl.dot(a, w, accumulator, input_precision="ieee")
        x_ptrs += block_k * stride_xk
        w_ptrs += block_k * stride_wk


    accumulator = accumulator.reshape(block_m, block_n // 2, 2)
    z, raw = tl.split(accumulator)
    if activation == "silu":
        z = silu(z)
    z = z * raw

    z = z.to(z_ptr.dtype.element_ty)

    offs_cm = pid_m * block_m + tl.arange(0, block_m)
    offs_cn = pid_n * (block_n // 2) + tl.arange(0, block_n // 2)
    z_ptrs = z_ptr + stride_zm * offs_cm[:, None] + stride_zn * offs_cn[None, :]
    z_mask = (offs_cm[:, None] < m) & (offs_cn[None, :] < n // 2)
    tl.store(z_ptrs, z, mask=z_mask)


@triton.jit
def swiglu_kernel(
    x_ptr,
    w12_ptr,
    w3_ptr,
    z_ptr,
    m,
    n,
    k,
    l,
    stride_xm,
    stride_xk,
    stride_wk,
    stride_wn,
    stride_w3n,
    stride_w3l,
    stride_zm,
    stride_zl,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    group_m: tl.constexpr,
    block_l: tl.constexpr,
    use_tf32: tl.constexpr = tl.constexpr(True),
    activation: tl.constexpr = tl.constexpr("silu"),
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
    tl.assume(stride_xm > 0)
    tl.assume(stride_xk > 0)
    tl.assume(stride_wn > 0)
    tl.assume(stride_wk > 0)
    tl.assume(stride_zm > 0)
    tl.assume(stride_zl > 0)
    # ----------------------

    offs_xm = pid_m * block_m + tl.arange(0, block_m)
    offs_wn = pid_n * block_n + tl.arange(0, block_n)
    offs_k = tl.arange(0, block_k)

    x_ptrs = x_ptr + (
        offs_xm[:, None] * stride_xm + offs_k[None, :] * stride_xk
    )

    w_ptrs = w12_ptr + (
        offs_k[:, None] * stride_wk + offs_wn[None, :] * stride_wn
    )

    accumulator = tl.zeros((block_m, block_n), dtype=tl.float32)
    for k_loop in range(0, tl.cdiv(k, block_k)):
        k_remaining = k - k_loop * block_k

        mask = offs_k < k_remaining
        a = tl.load(x_ptrs, mask=mask[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=mask[:, None], other=0.0)

        if use_tf32:
            accumulator = tl.dot(a, w, accumulator)
        else:
            accumulator = tl.dot(a, w, accumulator, input_precision="ieee")
        x_ptrs += block_k * stride_xk
        w_ptrs += block_k * stride_wk

    accumulator = accumulator.reshape(block_m, block_n // 2, 2)
    activated, raw = tl.split(accumulator)

    if activation == "silu":
        activated = silu(activated)
    a = activated * raw

    offs_wn = pid_n * block_n + tl.arange(0, block_n // 2)
    offs_l = tl.arange(0, block_l)

    w3_ptrs = w3_ptr + (
        offs_wn[:, None] * stride_w3n + offs_l[None, :] * stride_w3l
    )

    z_ptrs = z_ptr + stride_zm * offs_xm[:, None] + stride_zl * offs_l[None, :]
    for l_loop in range(0, tl.cdiv(l, block_l)):
        l_remaining = l - l_loop * block_k
        mask = offs_l < l_remaining

        w3 = tl.load(w3_ptrs, mask=mask[None, :], other=0.0)

        if use_tf32:
            accumulator = tl.dot(a, w3)
        else:
            accumulator = tl.dot(a, w3, input_precision="ieee")
        accumulator = accumulator.to(z_ptr.dtype.element_ty)

        z_mask = (offs_xm[:, None] < m) & (mask[None, :] < l_remaining)
        tl.atomic_add(z_ptrs, accumulator, mask=z_mask, sem="relaxed")

        w3_ptrs += block_l * stride_w3l
