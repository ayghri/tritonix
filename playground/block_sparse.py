from matmul0 import MatMul
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
# Initial layout

layout = torch.tensor(
    [
        [1, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 0, 1, 1],
        [0, 0, 1, 1],
    ],
    dtype=torch.int64,
    device=device,
)

# Instantiate the MatMul Class
sparse_matmul = MatMul(layout=layout, block=16, mode="dds")

# --- Later in your code ---

# FIX: When you update the layout, add the batch dimension with unsqueeze(0)
sparse_matmul.layout = layout.to(device).unsqueeze(0)

# Create other tensors...
a = torch.randn(2, 32, 64, dtype=torch.float16, device=device)
num_non_zero_blocks = int(layout.sum().item())
b = torch.randn(num_non_zero_blocks, 16, 16, dtype=torch.float16, device=device)

# This will now work correctly.
c = sparse_matmul(a, b)

print("Shape of output 'c':", getattr(c, "shape", None))
