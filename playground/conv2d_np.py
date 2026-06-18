import torch
import triton
import numpy as np

# from conv2d_fwd import conv2d_forward_kernel
from triton.conv2d_kernels import conv2d_forward_kernel
import configs
# No special configs needed as we are not using the autotuner for this comparison

# ---- Triton Kernel and Wrapper (Corrected Version) ----
# Note: I'm using a simplified, non-autotuned kernel for clarity in this test.


def conv2d_forward_triton(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride,
    padding,
    dilation,
):
    BATCH_SIZE, C_IN, H_IN, W_IN = input.shape
    C_OUT, t_C, F_H, F_W = weight.shape
    assert t_C == C_IN

    str_h, str_w = stride
    pad_h, pad_w = padding
    dil_h, dil_w = dilation

    H_OUT = (H_IN + 2 * pad_h - dil_h * (F_H - 1) - 1) // str_h + 1
    W_OUT = (W_IN + 2 * pad_w - dil_w * (F_W - 1) - 1) // str_w + 1

    GEMM_M = BATCH_SIZE * H_OUT * W_OUT
    GEMM_K = C_IN * F_H * F_W
    GEMM_N = C_OUT
    output = torch.zeros(
        (BATCH_SIZE, C_OUT, H_OUT, W_OUT),
        dtype=input.dtype,
        device=input.device,
    )

    # Simplified grid for direct comparison
    grid = lambda META: (
        triton.cdiv(GEMM_M, META["BLOCK_SIZE_M"])
        * triton.cdiv(GEMM_N, META["BLOCK_SIZE_N"]),
    )

    # We pass some default block sizes, no autotuning
    conv2d_forward_kernel[grid](
        output,
        input,
        weight,
        bias,
        BATCH_SIZE,
        C_IN,
        H_IN,
        W_IN,
        C_OUT,
        H_OUT,
        W_OUT,
        F_H,
        F_W,
        str_h,
        str_w,
        pad_h,
        pad_w,
        dil_h,
        dil_w,
        GEMM_M,
        GEMM_N,
        GEMM_K,
        BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32, GROUP_SIZE_M=2,  # type: ignore[call-arg]
    )
    return output


# ---- NumPy / SciPy Float64 Reference Implementation ----
def conv2d_forward_numpy(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride,
    padding,
    dilation,
):
    # Convert inputs to float64 numpy arrays
    inp_np = input.cpu().numpy().astype(np.float64)
    w_np = weight.cpu().numpy().astype(np.float64)
    b_np = bias.cpu().numpy().astype(np.float64) if bias is not None else None

    BATCH_SIZE, C_IN, H_IN, W_IN = inp_np.shape
    C_OUT, _, F_H, F_W = w_np.shape
    str_h, str_w = stride
    pad_h, pad_w = padding
    dil_h, dil_w = dilation

    H_OUT = (H_IN + 2 * pad_h - dil_h * (F_H - 1) - 1) // str_h + 1
    W_OUT = (W_IN + 2 * pad_w - dil_w * (F_W - 1) - 1) // str_w + 1

    # Add padding to the input tensor
    padded_inp = np.pad(
        inp_np,
        ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)),
        mode="constant",
        constant_values=0,
    )

    # Initialize output with float64
    output_np = np.zeros((BATCH_SIZE, C_OUT, H_OUT, W_OUT), dtype=np.float64)

    # Manually perform convolution
    for n in range(BATCH_SIZE):
        for c_out in range(C_OUT):
            for h_out in range(H_OUT):
                for w_out in range(W_OUT):
                    # Find the top-left corner of the input patch
                    h_start = h_out * str_h
                    w_start = w_out * str_w

                    # Select the input patch with dilation
                    input_patch = padded_inp[
                        n,
                        :,
                        h_start : h_start + F_H * dil_h : dil_h,
                        w_start : w_start + F_W * dil_w : dil_w,
                    ]

                    # Perform dot product and add bias
                    conv_sum = np.sum(input_patch * w_np[c_out, :, :, :])
                    if b_np is not None:
                        conv_sum += b_np[c_out]
                    output_np[n, c_out, h_out, w_out] = conv_sum

    return output_np


if __name__ == "__main__":
    # disable_torch_optimizations()
    # configs.enable_torch_optimizations()
    configs.disable_torch_optimizations()

    torch.manual_seed(42)
    np.random.seed(42)

    # Use a small batch size for focused comparison
    B = 32
    DEVICE = "cuda"
    DTYPE = torch.float32

    # Define network parameters
    C_IN, H, W = 64, 97, 130
    C_OUT = 32
    K_H, K_W = 5, 7
    stride = (3, 2)
    padding = (2, 3)
    dilation = (1, 1)
    H_OUT = (H + 2 * padding[0] - dilation[0] * (K_H - 1) - 1) // stride[0] + 1
    W_OUT = (W + 2 * padding[1] - dilation[1] * (K_W - 1) - 1) // stride[1] + 1

    # Create random tensors
    input_tensor = torch.randn(B, C_IN, H, W, device=DEVICE, dtype=DTYPE)
    weight_tensor = torch.randn(
        C_OUT, C_IN, K_H, K_W, device=DEVICE, dtype=DTYPE
    )
    bias_tensor = torch.randn(C_OUT, device=DEVICE, dtype=DTYPE)

    # --- Run all three implementations ---
    print("Running implementations...")
    # 1. Triton
    triton_out = conv2d_forward_triton(
        input_tensor, weight_tensor, bias_tensor, stride, padding, dilation
    )
    # 2. PyTorch (cuDNN)
    torch_out = torch.nn.functional.conv2d(
        input_tensor, weight_tensor, bias_tensor, stride, padding, dilation
    )
    # 3. NumPy (Float64 Ground Truth)
    numpy_out = conv2d_forward_numpy(
        input_tensor, weight_tensor, bias_tensor, stride, padding, dilation
    )

    # Convert GPU tensors to CPU numpy arrays for comparison
    triton_out_np = triton_out.cpu().numpy()
    torch_out_np = torch_out.cpu().numpy()

    print("\n--- Comparing Output Values at Specific Indices ---")

    # Define a set of indices to check
    indices_to_check = [
        (0, 0, 0, 0),  # Corner
        (0, C_OUT // 2, H_OUT // 2, W_OUT // 2),  # Center
        (0, C_OUT - 1, H_OUT // 2, W_OUT // 2),  # Center
        (0, C_OUT - 1, H_OUT - 1, W_OUT - 1),  # Opposite corner
        (0, C_OUT - 1, H_OUT // 3 - 1, W_OUT // 3 - 1),  # Opposite corner
        (
            0,
            np.random.randint(C_OUT),
            np.random.randint(H_OUT),
            np.random.randint(W_OUT),
        ),  # Random
    ]

    for idx in indices_to_check:
        triton_val = triton_out_np[idx]
        torch_val = torch_out_np[idx]
        numpy_val = numpy_out[idx]

        diff_triton_numpy = np.abs(triton_val - numpy_val)
        diff_torch_numpy = np.abs(torch_val - numpy_val)
        diff_torch_triton = np.abs(torch_val - triton_val)

        print(f"\nComparing at index: {idx}")
        print(f"  NumPy (float64)  : {numpy_val:.8f}")
        print(
            f"  Triton (float32) : {triton_val:.8f} (Difference from NumPy: {diff_triton_numpy:.8f})"
        )
        print(
            f"  Torch (float32)  : {torch_val:.8f} (Difference from NumPy: {diff_torch_numpy:.8f})"
        )
        print(f"  Difference Triton-Torch: {diff_torch_triton:.8f}")
        if diff_triton_numpy < diff_torch_numpy:
            print("  -> Triton is closer to the NumPy reference.")
        else:
            print("  -> Torch is closer to the NumPy reference.")

    print("\n--- Summary of Differences ---")
    triton_diff = np.abs(triton_out_np - numpy_out)
    torch_diff = np.abs(torch_out_np - numpy_out)
    triton_torch_diff = np.abs(triton_out_np - torch_out_np).mean()
    print(f"  Mean Difference Triton-NumPy: {triton_diff.mean():.8f}")
    print(f"  Mean Difference Torch-NumPy: {torch_diff.mean():.8f}")
    print(f"  Mean Difference Triton-Torch: {triton_torch_diff:.8f}")
    print(
        f" relative triton-numpy: {triton_diff.mean() / np.abs(numpy_out).mean():.8f}"
    )
    print(
        f" relative torch-numpy: {torch_diff.mean() / np.abs(numpy_out).mean():.8f}"
    )
    print(
        f" max relative triton-numpy ratio: {np.max(triton_diff / np.abs(numpy_out)):.8f}"
    )
    print(
        f" max relative torch-numpy ratio: {np.max(torch_diff / np.abs(numpy_out)):.8f}"
    )
