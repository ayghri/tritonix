
@triton.jit
def conv2d_dweight_kernel(
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
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

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

    for idx_k in range(0, tl.cdiv(GEMM_K, BLOCK_SIZE_K)):
        gemm_k = idx_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        # valid_k = gemm_k < GEMM_K  # Master mask for K-dim

        batch_idx = gemm_k // (H_OUT * W_OUT)
        hw_out = gemm_k % (H_OUT * W_OUT)
        offs_hout = hw_out // W_OUT
        offs_wout = hw_out % W_OUT
        # offs_hout = tl.where(mask_k, hw_out // W_OUT, 0)
        # offs_wout = tl.where(mask_k, hw_out % W_OUT, 0)

        offs_hin = (
            offs_hout[:, None] * stride_h + offs_fh[None, :] * dil_h - pad_h
        )
        offs_win = (
            offs_wout[:, None] * stride_w + offs_fw[None, :] * dil_w - pad_w
        )
        # hin_valid = (offs_hin >= 0) & (offs_hin < H_IN)
        # win_valid = (offs_win >= 0) & (offs_win < W_IN)
        # mask_input = valid_k[:, None] & hin_valid & win_valid

        offs_input = (
            batch_idx[:, None] * (C_IN * H_IN * W_IN)
            + offs_cin[None, :] * (H_IN * W_IN)
            + (offs_hin * W_IN + offs_win)
        )
        mask_input = (
            (batch_idx[:, None] < BATCH_SIZE)
            & (offs_cin[None, :] < C_IN)
            & (offs_hin < H_IN)
            & (offs_win < W_IN)
            & (offs_hin >= 0)
            & (offs_win >= 0)
        )

        offs_grad_output = (
            batch_idx * (C_OUT * H_OUT * W_OUT) + offs_hout * W_OUT + offs_wout
        )[None, :] + (offs_cout * (H_OUT * W_OUT))[:, None]

        # [None, :]
        # mask_grad_output = valid_k[None, :]

        mask_grad_output = (offs_cout[:, None] < C_OUT) & (
            (batch_idx < BATCH_SIZE) & (offs_hout < H_OUT) & (offs_wout < W_OUT)
        )[None, :]
        doutput_ptrs = grad_output_ptr + offs_grad_output
        input_ptrs = input_ptr + offs_input

        doutput_data = tl.load(doutput_ptrs, mask=mask_grad_output, other=0.0)
        input_data = tl.load(input_ptrs, mask=mask_input, other=0.0)
        # doutput_data = tl.load(
        # doutput_ptr + offs_doutput, mask=mask_k[None, :], other=0.0
        # )
        # input_data = tl.load(input_ptr + offs_input, mask=mask_input, other=0.0)

        acc = tl.dot(doutput_data, input_data, acc, allow_tf32=False)
        # acc = tl.dot(doutput_data, input_data, acc)
        # acc = tl.dot(doutput_data, input_data, acc, input_precision="ieee")

    acc = acc.to(grad_weight_ptr.dtype.element_ty)

    # offs_weight: [BLOCK_SIZE_M, BLOCK_SIZE_N]
    offs_weight = gemm_m[:, None] * GEMM_N + gemm_n[None, :]
    mask_weight = (gemm_m[:, None] < GEMM_M) & (gemm_n[None, :] < GEMM_N)
    dweight_ptrs = grad_weight_ptr + offs_weight
    tl.store(dweight_ptrs, acc, mask=mask_weight)


