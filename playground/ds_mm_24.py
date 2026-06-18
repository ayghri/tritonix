import torch
from torch.sparse import to_sparse_semi_structured, SparseSemiStructuredTensor

# Using the CUTLASS backend is still recommended for broader support
SparseSemiStructuredTensor._FORCE_CUTLASS = True

def create_structured_sparsity_mask(
    tensor: torch.Tensor, 
    n: int, 
    m: int
) -> torch.Tensor:
    """
    Creates a boolean mask for n:m structured sparsity.

    This function identifies the `n` elements with the largest absolute values 
    within every contiguous block of `m` elements along the last dimension of 
    the input tensor.

    Args:
        tensor (torch.Tensor): The input weight tensor. Can be of any shape, but
                               sparsity is applied along the last dimension.
        n (int): The number of elements to keep in each block (e.g., 2 for 2:4).
        m (int): The size of each block (e.g., 4 for 2:4).

    Returns:
        torch.Tensor: A boolean tensor of the same shape as the input, with `True`
                      indicating the weights to keep and `False` for those to prune.
                      
    Raises:
        ValueError: If the last dimension of the tensor is not divisible by `m`.
    """
    # --- Input Validation ---
    if tensor.dim() == 0:
        raise ValueError("Input tensor must have at least one dimension.")
        
    if tensor.shape[-1] % m != 0:
        raise ValueError(
            f"The last dimension of the tensor (shape[-1]={tensor.shape[-1]}) "
            f"must be divisible by m={m}."
        )

    # --- Mask Creation ---
    # Work with the absolute values to find the largest magnitudes
    abs_tensor = torch.abs(tensor)
    
    # Reshape the tensor to isolate the m-sized blocks
    # Example: (..., K) -> (..., K/m, m)
    reshaped_tensor = abs_tensor.view(-1, m)
    
    # Use torch.topk to find the indices of the n largest values in each block
    # This returns the values and their indices within each block of size m.
    _, top_k_indices = torch.topk(reshaped_tensor, k=n, dim=-1)
    
    # Create a new boolean mask filled with False
    mask = torch.zeros_like(reshaped_tensor, dtype=torch.bool)
    
    # Use scatter_ to place `True` at the locations of the top-k indices.
    # This is a highly efficient way to populate the mask.
    mask.scatter_(dim=-1, index=top_k_indices, value=True)
    
    # Reshape the mask back to the original tensor's shape
    return mask.view(tensor.shape)

M, N, K = 3072, 3072, 10240
DTYPE = torch.float16
device = "cuda"

# 1. Create original dense input and weight
x = torch.randn(M, K, device=device, dtype=DTYPE)
weight_original = torch.randn(N, K, device=device, dtype=DTYPE)

# weight_original = weight_original.reshape(N, K//4, 4)
# Mask to create a 2:4 sparse pattern
# mask = torch.tensor([0, 0, 1, 1], device=device, dtype=DTYPE).repeat(N // 4, 1)

# weight_original = weight_original * mask.unsqueeze(1)
mask = create_structured_sparsity_mask(weight_original, n=2, m=4).to(device)
weight_original = weight_original * mask

# 2. Create the official 2:4 sparse tensor (this performs pruning)
sparse_weight = to_sparse_semi_structured(weight_original)

# 3. Create the correct dense baseline tensor for comparison
weight_pruned_dense = sparse_weight.to_dense()

sparse_weight = sparse_weight.t()
weight_pruned_dense = weight_pruned_dense.t()

# --- Manual timing function ---
def benchmark_op(op, op_name, iterations=100, warmup=10):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    print(f"--- Benchmarking: {op_name} ---")
    print("Running warmup...")
    for _ in range(warmup):
        op()
    torch.cuda.synchronize()
    print(f"Running measurement ({iterations} iterations)...")
    start_event.record()
    for _ in range(iterations):
        op()
    end_event.record()
    torch.cuda.synchronize()
    avg_ms = start_event.elapsed_time(end_event) / iterations
    print(f"Average time: {avg_ms:.3f} ms")
    return avg_ms


# --- Run Benchmarks ---
dense_time = benchmark_op(
    lambda: torch.mm(x, weight_pruned_dense), "Dense (Pruned)"
)
sparse_time = benchmark_op(
    lambda: torch.mm(x, sparse_weight), "Sparse (2:4)"
)


# --- Final Summary ---
speedup = dense_time / sparse_time
print("\n" + "=" * 30)
print("--- Final Results ---")
print(f"Dense (Pruned) average time: {dense_time:.3f} ms")
print(f"Sparse (2:4) average time:   {sparse_time:.3f} ms")
print(f"\nSpeedup: {speedup:.3f}x")
print("=" * 30)

# 
print("Sparsity of dense weight:", weight_pruned_dense.count_nonzero().item() / (N * K))
# --- Verification ---
print("\nVerifying numerical equivalence with appropriate tolerance...")
# output_dense = torch.mm(x, weight_pruned_dense.t())
# output_sparse = torch.mm(x, sparse_weight.t())
output_dense = torch.mm(x, weight_pruned_dense)
output_sparse = torch.mm(x, sparse_weight)

max_diff = torch.max(torch.abs(output_dense - output_sparse))
print(f"Max difference between outputs: {max_diff.item():.2f}")

# Increase the absolute tolerance to account for float16 accumulation differences
# across different GPU kernels. A tolerance of 1.0 is reasonable for this scale.
assert torch.allclose(output_dense, output_sparse, atol=1, rtol=1e-2)
print(
    "Verification successful: The outputs are numerically equivalent within a reasonable tolerance."
)
print(weight_pruned_dense[:5, :5])
print(weight_original[:5, :5])
print(weight_original.shape)
print(sparse_weight.packed_t.shape)
print(sparse_weight.meta_t.shape)
