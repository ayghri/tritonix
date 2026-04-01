import triton
import triton.language as tl


@triton.jit
def conv2d_grad_weight_kernel(
    input_ptr,
    grad_output_ptr,
    grad_weight_ptr,
    BATCH_SIZE,
    C_IN,
    H_IN,
    W_IN,
    C_OUT,
    H_OUT,
    W_OUT,
    FILTER_H,
    FILTER_W,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    dil_h,
    dil_w,
    GEMM_M,  # = C_OUT
    GEMM_N,  # = C_IN * FILTER_H * FILTER_W
    GEMM_K,  # = BATCH_SIZE * H_OUT * W_OUT
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(GEMM_M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(GEMM_N, BLOCK_SIZE_N)

    num_pid_in_group = GROUP_SIZE_M * num_pid_m
    group_id = pid // num_pid_in_group
    first_pid_n = group_id * GROUP_SIZE_M
    group_size_n = min(num_pid_n - first_pid_n, GROUP_SIZE_M)
    pid_n = first_pid_n + ((pid % num_pid_in_group) % group_size_n)
    pid_m = (pid % num_pid_in_group) // group_size_n

    ### for compiler
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(pid_m < num_pid_m)
    tl.assume(pid_n < num_pid_n)
    tl.assume(BATCH_SIZE > 0)
    tl.assume(C_IN > 0)
    tl.assume(C_OUT > 0)
    tl.assume(FILTER_H > 0)
    tl.assume(FILTER_W > 0)
    tl.assume(H_IN > 0)
    tl.assume(H_OUT > 0)
    tl.assume(W_IN > 0)
    tl.assume(W_OUT > 0)
    tl.assume(dil_h > 0)
    tl.assume(dil_w > 0)
    tl.assume(stride_h > 0)
    tl.assume(stride_w > 0)
    tl.assume(pad_h >= 0)
    tl.assume(pad_w >= 0)
    ###

    gemm_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    gemm_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    offs_cout = gemm_m
    offs_cin = gemm_n // (FILTER_H * FILTER_W)
    offs_filter = gemm_n % (FILTER_H * FILTER_W)
    offs_fh = offs_filter // FILTER_W
    offs_fw = offs_filter % FILTER_W

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    gemm_k = tl.arange(0, BLOCK_SIZE_K)

    for _ in range(0, tl.cdiv(GEMM_K, BLOCK_SIZE_K)):
        batch_idx = gemm_k // (H_OUT * W_OUT)
        hw_out_residual = gemm_k % (H_OUT * W_OUT)
        offs_hout = hw_out_residual // W_OUT
        offs_wout = hw_out_residual % W_OUT

        offs_hin = (
            offs_hout[:, None] * stride_h + offs_fh[None, :] * dil_h - pad_h
        )
        offs_win = (
            offs_wout[:, None] * stride_w + offs_fw[None, :] * dil_w - pad_w
        )

        offs_input = (
            batch_idx[:, None] * (C_IN * H_IN * W_IN)
            + offs_cin[None, :] * H_IN * W_IN
            + (offs_hin * W_IN + offs_win)
        )
        mask_input = (
            (offs_cin[None, :] < C_IN)
            & (batch_idx[:, None] < BATCH_SIZE)
            & (
                (offs_hin < H_IN)
                & (offs_win < W_IN)
                & (offs_hin >= 0)
                & (offs_win >= 0)
            )
        )

        offs_doutput = (
            batch_idx * (C_OUT * H_OUT * W_OUT) + offs_hout * W_OUT + offs_wout
        )[None, :] + offs_cout[:, None] * H_OUT * W_OUT

        mask_doutput = (
            (batch_idx < BATCH_SIZE) & (offs_hout < H_OUT) & (offs_wout < W_OUT)
        )[None, :] & (offs_cout < C_OUT)[:, None]

        doutput_ptrs = grad_output_ptr + offs_doutput
        input_ptrs = input_ptr + offs_input

        doutput_data = tl.load(doutput_ptrs, mask=mask_doutput, other=0.0)
        input_data = tl.load(input_ptrs, mask=mask_input, other=0.0)

        acc = tl.dot(doutput_data, input_data, acc, allow_tf32=False)

        gemm_k = gemm_k + BLOCK_SIZE_K

    acc = acc.to(grad_weight_ptr.dtype.element_ty)

    offs_weight = gemm_m[:, None] * GEMM_N + gemm_n[None, :]
    mask_weight = (gemm_m[:, None] < GEMM_M) & (gemm_n[None, :] < GEMM_N)
    dweight_ptrs = grad_weight_ptr + offs_weight
    tl.store(dweight_ptrs, acc, mask=mask_weight)


@triton.jit
def conv2d_grad_weight_kernel_atomic(
    input_ptr,
    grad_output_ptr,
    grad_weight_ptr,
    BATCH_SIZE,
    C_IN,
    H_IN,
    W_IN,
    C_OUT,
    H_OUT,
    W_OUT,
    FILTER_H,
    FILTER_W,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    dil_h,
    dil_w,
    GEMM_M,  # = C_OUT
    GEMM_N,  # = C_IN * FILTER_H * FILTER_W
    GEMM_K,  # = H_OUT * W_OUT
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Per-batch weight gradient with atomic accumulation across batches."""
    batch_id = tl.program_id(axis=0)
    pid = tl.program_id(axis=1)
    num_pid_m = tl.cdiv(GEMM_M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(GEMM_N, BLOCK_SIZE_N)
    num_groups_per_stripe = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_groups_per_stripe
    group_start = group_id * GROUP_SIZE_M
    group_size = min(num_pid_m - group_start, GROUP_SIZE_M)

    pid_m = group_start + (pid % num_groups_per_stripe) % group_size
    pid_n = (pid % num_groups_per_stripe) // group_size

    ### for compiler
    tl.assume(batch_id >= 0)
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(pid_m < num_pid_m)
    tl.assume(pid_n < num_pid_n)
    tl.assume(BATCH_SIZE > 0)
    tl.assume(C_IN > 0)
    tl.assume(C_OUT > 0)
    tl.assume(FILTER_H > 0)
    tl.assume(FILTER_W > 0)
    tl.assume(H_IN > 0)
    tl.assume(H_OUT > 0)
    tl.assume(W_IN > 0)
    tl.assume(W_OUT > 0)
    tl.assume(dil_h > 0)
    tl.assume(dil_w > 0)
    tl.assume(stride_h > 0)
    tl.assume(stride_w > 0)
    tl.assume(pad_h >= 0)
    tl.assume(pad_w >= 0)
    ###

    gemm_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    gemm_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    offs_cout = gemm_m
    offs_cin = gemm_n // (FILTER_H * FILTER_W)
    offs_filter = gemm_n % (FILTER_H * FILTER_W)
    offs_fh = offs_filter // FILTER_W
    offs_fw = offs_filter % FILTER_W

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k_idx in range(0, tl.cdiv(GEMM_K, BLOCK_SIZE_K)):
        gemm_k = k_idx * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        offs_hout = gemm_k // W_OUT
        offs_wout = gemm_k % W_OUT

        offs_hin = (
            offs_hout[:, None] * stride_h + offs_fh[None, :] * dil_h - pad_h
        )
        offs_win = (
            offs_wout[:, None] * stride_w + offs_fw[None, :] * dil_w - pad_w
        )

        offs_input = (
            batch_id * (C_IN * H_IN * W_IN)
            + offs_cin[None, :] * H_IN * W_IN
            + (offs_hin * W_IN + offs_win)
        )
        mask_input = (offs_cin < C_IN)[None, :] & (
            (offs_hin < H_IN)
            & (offs_win < W_IN)
            & (offs_hin >= 0)
            & (offs_win >= 0)
        )

        offs_doutput = (
            batch_id * (C_OUT * H_OUT * W_OUT)
            + offs_cout[:, None] * H_OUT * W_OUT
            + (offs_hout * W_OUT + offs_wout)[None, :]
        )

        mask_doutput = ((offs_hout < H_OUT) & (offs_wout < W_OUT))[None, :] & (
            offs_cout < C_OUT
        )[:, None]
        doutput_ptrs = grad_output_ptr + offs_doutput
        input_ptrs = input_ptr + offs_input

        doutput_data = tl.load(doutput_ptrs, mask=mask_doutput, other=0.0)
        input_data = tl.load(input_ptrs, mask=mask_input, other=0.0)

        acc = tl.dot(doutput_data, input_data, acc, allow_tf32=False)

    acc = acc.to(grad_weight_ptr.dtype.element_ty)

    offs_weight = gemm_m[:, None] * GEMM_N + gemm_n[None, :]
    mask_weight = (gemm_m[:, None] < GEMM_M) & (gemm_n[None, :] < GEMM_N)
    dweight_ptrs = grad_weight_ptr + offs_weight
    tl.atomic_add(dweight_ptrs, acc, mask=mask_weight)


@triton.jit
def conv2d_grad_bias_kernel(
    grad_output,
    grad_bias,
    BATCH_SIZE,
    C_OUT,
    H_OUT,
    W_OUT,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_COUT: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_cout = pid + tl.arange(0, BLOCK_COUT)
    offs_hwout = tl.arange(0, BLOCK_SIZE)
    acc = tl.zeros((BLOCK_COUT,), dtype=tl.float32)
    grad_output_ptrs = grad_output + (
        offs_cout[None, :] * H_OUT * W_OUT + offs_hwout[:, None]
    )
    grad_output_mask = (offs_cout[None, :] < C_OUT) & (
        offs_hwout[:, None] < H_OUT * W_OUT
    )
    for _ in range(0, BATCH_SIZE):
        grad_output_data = tl.load(
            grad_output_ptrs, mask=grad_output_mask, other=0.0
        )
        acc = acc + tl.sum(grad_output_data, dim=0)
        grad_output_ptrs += C_OUT * H_OUT * W_OUT

    acc = acc.to(grad_bias.dtype.element_ty)
    tl.store(grad_bias + offs_cout, acc, mask=(offs_cout < C_OUT))


@triton.jit
def conv2d_grad_input_kernel(
    grad_input_ptr,
    grad_output_ptr,
    weight_ptr,
    BATCH_SIZE,
    C_IN,
    H_IN,
    W_IN,
    C_OUT,
    H_OUT,
    W_OUT,
    FILTER_H,
    FILTER_W,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    dil_h,
    dil_w,
    GEMM_M,
    GEMM_N,
    GEMM_K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(GEMM_M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(GEMM_N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    gemm_i = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    gemm_j = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    n = gemm_i // (H_IN * W_IN)
    nhw_residual = gemm_i % (H_IN * W_IN)
    h = nhw_residual // W_IN
    w = nhw_residual % W_IN
    c = gemm_j

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for idx_k in range(0, tl.cdiv(GEMM_K, BLOCK_SIZE_K)):
        gemm_k = idx_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        k = gemm_k // (FILTER_H * FILTER_W)
        krs_residual = gemm_k % (FILTER_H * FILTER_W)
        r = krs_residual // FILTER_W
        s = krs_residual % FILTER_W

        h_tmp = h[:, None] + pad_h - r[None, :] * dil_h
        p = h_tmp // stride_h
        mask_p = (h_tmp % stride_h == 0) & (p >= 0) & (p < H_OUT)
        w_tmp = w[:, None] + pad_w - s[None, :] * dil_w
        q = w_tmp // stride_w
        mask_q = (w_tmp % stride_w == 0) & (q >= 0) & (q < W_OUT)
        mask_doutput = (
            (n[:, None] < BATCH_SIZE) & (k[None, :] < C_OUT) & mask_p & mask_q
        )
        mask_weight = (
            (k[:, None] < C_OUT)
            & (c[None, :] < C_IN)
            & (r[:, None] < FILTER_H)
            & (s[:, None] < FILTER_W)
        )

        offs_doutput = (
            n[:, None] * C_OUT * H_OUT * W_OUT
            + k[None, :] * H_OUT * W_OUT
            + p * W_OUT
            + q
        )
        offs_weight = (
            k[:, None] * C_IN * FILTER_H * FILTER_W
            + c[None, :] * FILTER_H * FILTER_W
            + r[:, None] * FILTER_W
            + s[:, None]
        )

        doutput_ptrs = grad_output_ptr + offs_doutput
        weight_ptrs = weight_ptr + offs_weight

        doutput_data = tl.load(doutput_ptrs, mask=mask_doutput, other=0.0)
        weight_data = tl.load(weight_ptrs, mask=mask_weight, other=0.0)

        acc = tl.dot(doutput_data, weight_data, acc, allow_tf32=False)

    acc = acc.to(grad_input_ptr.dtype.element_ty)
    offs_nchw = (
        n[:, None] * C_IN * H_IN * W_IN
        + c[None, :] * H_IN * W_IN
        + h[:, None] * W_IN
        + w[:, None]
    )
    mask_nchw = (
        (n[:, None] < BATCH_SIZE)
        & (c[None, :] < C_IN)
        & (h[:, None] < H_IN)
        & (w[:, None] < W_IN)
    )
    dinput_ptrs = grad_input_ptr + offs_nchw
    tl.store(dinput_ptrs, acc, mask=mask_nchw)
