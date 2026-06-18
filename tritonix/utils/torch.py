import torch

Tensor = torch.Tensor


def reduce_dim_strides(x: Tensor, dim: int):
    """Compute (x_flat, n_rows, n_cols, stride_row, stride_col, out_shape)
    for reducing along ``dim``, zero-copy when possible."""
    ndim = x.ndim
    dim = dim % ndim
    n_cols = x.shape[dim]
    stride_col = x.stride(dim)

    if ndim == 1:
        return x, 1, n_cols, 1, stride_col, ()

    if ndim == 2:
        row_dim = 1 - dim
        n_rows = x.shape[row_dim]
        stride_row = x.stride(row_dim)
        out_shape = (n_rows,)
        return x, n_rows, n_cols, stride_row, stride_col, out_shape

    # N-D: permute dim to last, flatten leading dims
    perm = [i for i in range(ndim) if i != dim] + [dim]
    x_perm = x.permute(perm)
    out_shape = x_perm.shape[:-1]
    n_rows = x_perm.shape[:-1].numel()

    # Check if leading dims are contiguous so we can flatten without copy
    leading_contiguous = True
    for i in range(len(out_shape) - 1):
        if x_perm.stride(i) != x_perm.stride(i + 1) * x_perm.shape[i + 1]:
            leading_contiguous = False
            break

    if leading_contiguous:
        stride_row = x_perm.stride(len(out_shape) - 1)
        stride_col = x_perm.stride(-1)
        x_flat = x_perm.as_strided(
            (n_rows, n_cols),
            (stride_row, stride_col),
            storage_offset=x_perm.storage_offset(),
        )
    else:
        x_flat = x_perm.contiguous()
        stride_row = n_cols
        stride_col = 1

    return x_flat, n_rows, n_cols, stride_row, stride_col, out_shape


def enable_cudnn_optimizations():
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = True
        print("cuDNN benchmark enabled.")
    else:
        print("cuDNN is not available.")


def enable_torch_optimizations(
    allow_tf32=True,
    fp16_reduced_precision=False,
    # high_precision=True,
):
    """
    Enables various optimizations in PyTorch for matmul operations.
    """
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = True
        print("cuDNN benchmark enabled.")
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        if allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("TF32 enabled for matmul.")
        else:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            print("TF32 disabled for matmul.")

            # torch.set_float32_matmul_precision("high")
            # print("High precision matmul enabled.")

    if fp16_reduced_precision:
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        print("Reduced precision reductions enabled for fp16/bf16.")


def disable_torch_optimizations():
    """
    Disables various optimizations in PyTorch for matmul operations to enforce float32 precision.
    """
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        print("cuDNN benchmark disabled.")
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        print("TF32 disabled for matmul.")

        torch.set_float32_matmul_precision("highest")
        print("Highest precision matmul enabled.")

    # torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    # torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    # print("Reduced precision reductions disabled for fp16/bf16.")
