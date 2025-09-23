import torch
import torch.utils.benchmark as benchmark
from iht.bsr import create_bsr_matrix

def benchmark_matmul():
    """
    Benchmarks dense vs. sparse matrix multiplication.
    """
    # Matrix properties
    shape = (4096, 4096)
    block_size = (8, 8)
    sparsity = 0.999
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float16

    if device != 'cuda':
        print("CUDA not available, running on CPU. Benchmark might be slow or not work for sparse operations.")
        # Fallback to CPU-compatible settings if needed
        # return 
    
    print(f"Benchmarking on {device} with dtype {dtype}")
    print(f"Matrix shape: {shape}")
    print(f"Block size: {block_size}")
    print(f"Sparsity: {sparsity}")
    print("-" * 30)

    # Create dense matrices
    A_dense = torch.randn(shape, device=device, dtype=dtype)
    B_dense = torch.randn(shape, device=device, dtype=dtype)

    # Create BSR matrix
    A_bsr = create_bsr_matrix(shape, block_size, sparsity, device=device, dtype=dtype)

    # Warm-up GPU
    if device == 'cuda':
        for _ in range(10):
            _ = torch.matmul(A_dense, B_dense)
            _ = torch.matmul(A_bsr, B_dense)
            torch.cuda.synchronize()

    # Benchmark dense * dense
    t_dense = benchmark.Timer(
        stmt='torch.matmul(A_dense, B_dense)',
        globals={'A_dense': A_dense, 'B_dense': B_dense},
        label='Matrix-Matrix Multiplication',
        sub_label='Dense x Dense',
        description=f'Shape: {shape}, Sparsity: 0.0'
    )

    # Benchmark sparse * dense
    t_sparse = benchmark.Timer(
        stmt='torch.matmul(A_bsr, B_dense)',
        globals={'A_bsr': A_bsr, 'B_dense': B_dense},
        label='Matrix-Matrix Multiplication',
        sub_label='Sparse (BSR) x Dense',
        description=f'Shape: {shape}, Sparsity: {sparsity}'
    )

    dense_measurement = t_dense.blocked_autorange(min_run_time=1)
    sparse_measurement = t_sparse.blocked_autorange(min_run_time=1)

    print(dense_measurement)
    print(sparse_measurement)

    # speedup
    speedup = dense_measurement.mean / sparse_measurement.mean
    print(f"Speedup of Sparse vs Dense: {speedup:.2f}x")


if __name__ == "__main__":
    benchmark_matmul()
