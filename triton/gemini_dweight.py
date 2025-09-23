import torch
import triton
from triton.runtime import Autotuner
import configs
import triton.conv2d_kernels as conv2d_kernels
from torch.nn import grad as G


# Define the parameters for the convolution
BATCH_SIZE = 32
IN_CHANNELS = 64
OUT_CHANNELS = 128
INPUT_HEIGHT = 56
INPUT_WIDTH = 56
KERNEL_HEIGHT = 3
KERNEL_WIDTH = 3
STRIDE = 1
PADDING = 1

torch.nn.functional.conv2d
# Create random input tensors on the GPU
input_tensor = torch.randn(
    (BATCH_SIZE, IN_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH),
    device="cuda",
    dtype=torch.float16,
)
grad_output = torch.randn(
    (BATCH_SIZE, OUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH),
    device="cuda",
    dtype=torch.float16,
)
weight_size = (OUT_CHANNELS, IN_CHANNELS, KERNEL_HEIGHT, KERNEL_WIDTH)

def pytorch_grad_weight():
  """
  Computes the gradient of the weight using PyTorch's native function.
  """
  return G.conv2d_weight(
      input_tensor,
      weight_size,
      grad_output,
      stride=STRIDE,
      padding=PADDING
  )

def triton_grad_weight():
    """
    Wrapper function to launch the Triton kernel.
    """
    grad_weight = torch.empty(weight_size, device="cuda", dtype=torch.float16)
    grid = (triton.cdiv(OUT_CHANNELS, 16), triton.cdiv(IN_CHANNELS, 16))
    conv2d_kernels.conv2d_naive_dweight_kernel[grid](
        input_tensor,
        grad_output,
        grad_weight,
        BATCH_SIZE,
        IN_CHANNELS,
        OUT_CHANNELS,
        INPUT_HEIGHT,
        INPUT_WIDTH,
        KERNEL_HEIGHT,
        KERNEL_WIDTH,
        STRIDE,
        PADDING,
        BLOCK_SIZE_OC=16,
        BLOCK_SIZE_IC=16,
        BLOCK_SIZE_KH=4,
        BLOCK_SIZE_KW=4,
    )
    return grad_weight

# Run both implementations
pytorch_result = pytorch_grad_weight()
triton_result = triton_grad_weight()

# Compare the results
print(f"Are the results close? {torch.allclose(pytorch_result, triton_result, atol=1e-2, rtol=0)}")