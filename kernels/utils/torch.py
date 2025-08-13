import torch


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
