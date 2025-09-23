import torch
from matmul0 import MatMul  # Assuming your modified MatMul class is in matmul.py
import torch.utils.benchmark as benchmark
import numpy as np


# --- Helper Function to Create a Dense Matrix from a Sparse Layout ---
def densify_sparse_matrix(layout_2d, sparse_blocks, block_size):
    """
    Converts a sparse block matrix into a dense PyTorch tensor.

    Args:
        layout_2d (torch.Tensor): The 2D layout tensor (0s and 1s).
        sparse_blocks (torch.Tensor): The packed tensor of non-zero blocks.
        block_size (int): The size of each square block.

    Returns:
        torch.Tensor: The equivalent dense matrix.
    """
    device = sparse_blocks.device
    dtype = sparse_blocks.dtype

    rows, cols = layout_2d.shape
    dense_shape = (rows * block_size, cols * block_size)

    # Create an empty (zero) dense matrix
    dense_matrix = torch.zeros(dense_shape, dtype=dtype, device=device)

    # Find the coordinates of the non-zero blocks
    non_zero_coords = torch.nonzero(layout_2d)

    # Place each non-zero block into the dense matrix
    for i, (r, c) in enumerate(non_zero_coords):
        row_start, col_start = r * block_size, c * block_size
        row_end, col_end = row_start + block_size, col_start + block_size
        dense_matrix[row_start:row_end, col_start:col_end] = sparse_blocks[i]

    return dense_matrix


# --- 1. SETUP ---

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

if device.type == "cpu":
    print("This benchmark requires a GPU to run.")
    exit()

# Configuration
block_size = 16
mode = "dds"

# Define the 2D layout on the target device
# layout = torch.tensor(
#     [
#         [1, 0, 1, 0, 0, 1],
#         [0, 1, 0, 0, 1, 1],
#         [1, 0, 1, 1, 0, 0],
#         [0, 0, 1, 1, 1, 0],
#         [0, 1, 0, 0, 1, 1],
#         [1, 0, 1, 1, 0, 0],
#     ],
#     dtype=torch.int64,
#     device=device,
# )
layout_n = 128
# layout = torch.randint(0, 2, (layout_n, layout_n), dtype=torch.int64, device=device)  # Random layout for testing
layout = (
    torch.from_numpy(np.random.uniform(0, 1, (layout_n, layout_n)) > 0.6)
    .to(torch.int64)
    .to(device)
)

# Instantiate the custom MatMul Class
# The constructor expects a 3D layout, so we unsqueeze it
sparse_matmul = MatMul(layout=layout.unsqueeze(0), block=block_size, mode=mode)


# Create the input tensors
# We'll use larger dimensions for a more meaningful benchmark
batch_size, M, K = 1, 1024, block_size * layout.shape[0]
a = torch.randn(batch_size, M, K, dtype=torch.float16, device=device)

# Create the packed sparse blocks for our custom kernel
num_non_zero_blocks = int(layout.sum().item())
b_sparse = torch.randn(
    num_non_zero_blocks,
    block_size,
    block_size,
    dtype=torch.float16,
    device=device,
)

# Create the dense equivalent of 'b' for torch.matmul
# This adds a batch dimension of 1, which will broadcast correctly with 'a'
b_dense = densify_sparse_matrix(layout, b_sparse, block_size).unsqueeze(0)

print("--- Benchmark Setup ---")
print(f"Dense Matrix 'a' shape:      {a.shape}")
print(
    f"Sparse Matrix 'b' layout:    {layout.shape} (conceptual shape: {b_dense.shape[1:]})"
)
print(f"Packed sparse 'b' shape:     {b_sparse.shape}")
print(f"Densified Matrix 'b' shape:  {b_dense.shape}")
print("-" * 25)

# --- 2. VERIFY CORRECTNESS ---

print("\nVerifying correctness...")
# Run both operations once to get the results
c_sparse = sparse_matmul(a, b_sparse)
c_dense = torch.matmul(a, b_dense)

# Compare the results. A small tolerance is needed for float16 arithmetic.
are_close = torch.allclose(c_sparse, c_dense, atol=1e-2)
print(f"Results are close: {are_close}")
if not are_close:
    print("Warning: Results do not match. Benchmark may be invalid.")
    print("Sparse result shape:", c_sparse.shape)
    print("Dense result shape:", c_dense.shape)
    print(c_sparse[:5, :5])
    print(c_dense[:5, :5])

print("-" * 25)


# --- 3. BENCHMARK ---

print("\nRunning benchmark...")

t0 = benchmark.Timer(
    stmt="sparse_matmul(a, b_sparse)",
    globals={"sparse_matmul": sparse_matmul, "a": a, "b_sparse": b_sparse},
)

t1 = benchmark.Timer(
    stmt="torch.matmul(a, b_dense)",
    globals={"torch": torch, "a": a, "b_dense": b_dense},
)

# Use blocked_autorange to get stable measurements
sparse_time = t0.blocked_autorange(min_run_time=1).mean * 1000
dense_time = t1.blocked_autorange(min_run_time=1).mean * 1000

print("\n--- Benchmark Results ---")
print(f"Custom DDS Sparse Matmul: {sparse_time:.3f} ms")
print(f"Plain Torch Dense Matmul: {dense_time:.3f} ms")
print("-" * 25)

# --- 4. CONCLUSION ---

if sparse_time < dense_time:
    speedup = dense_time / sparse_time
    print(
        f"\nConclusion: The custom sparse kernel is {speedup:.2f}x faster for this configuration."
    )
else:
    slowdown = sparse_time / dense_time
    print(
        f"\nConclusion: The custom sparse kernel is {slowdown:.2f}x slower for this configuration."
    )
    print(
        "This can happen for small matrices or low sparsity where kernel launch overhead dominates."
    )
