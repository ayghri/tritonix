import torch
import triton
import triton.language as tl
import time


# (The _transpose_kernel and transpose function from the previous response)
@triton.jit
def _transpose_kernel(
    input_ptr,
    output_ptr,
    M,
    N,
    stride_in_m,
    stride_in_n,
    stride_out_m,
    stride_out_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Triton kernel for out-of-place matrix transpose.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for the input block
    offsets_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # Pointers to the input block
    input_ptrs = (
        input_ptr
        + offsets_m[:, None] * stride_in_m
        + offsets_n[None, :] * stride_in_n
    )

    # Load the input block with masking
    mask = (offsets_m[:, None] < M) & (offsets_n[None, :] < N)
    input_block = tl.load(input_ptrs, mask=mask)

    # Transpose the block in registers
    transposed_block = tl.trans(input_block)

    # Offsets for the output block (swapped dimensions)
    offsets_out_m = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offsets_out_n = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)

    # Pointers to the output block
    output_ptrs = (
        output_ptr
        + offsets_out_m[:, None] * stride_out_m
        + offsets_out_n[None, :] * stride_out_n
    )

    # Store the transposed block with masking
    output_mask = (offsets_out_m[:, None] < N) & (offsets_out_n[None, :] < M)
    tl.store(output_ptrs, transposed_block, mask=output_mask)


def triton_transpose(tensor: torch.Tensor):
    """
    Out-of-place transpose of a 2D tensor using a Triton kernel.
    """
    M, N = tensor.shape
    output = torch.empty((N, M), device=tensor.device, dtype=tensor.dtype)

    # Optimal block sizes can be hardware-dependent.
    # 32x32 is a good starting point for many GPUs.
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32

    grid = (
        triton.cdiv(M, BLOCK_SIZE_M),
        triton.cdiv(N, BLOCK_SIZE_N),
    )

    _transpose_kernel[grid](
        tensor,
        output,
        M,
        N,
        tensor.stride(0),
        tensor.stride(1),
        output.stride(0),
        output.stride(1),
        BLOCK_SIZE_M=tl.constexpr(BLOCK_SIZE_M),
        BLOCK_SIZE_N=tl.constexpr(BLOCK_SIZE_N),
    )

    return output


def benchmark(M, N, device="cuda"):
    """
    Benchmarks Triton and PyTorch transpose operations.
    """
    tensor = torch.randn(M, N, device=device)

    # Warm-up
    for _ in range(5):
        _ = triton_transpose(tensor)
        _ = torch.transpose(tensor, 0, 1)
        _ = tensor.T

    triton_output = triton_transpose(tensor)
    pytorch_output_t = torch.transpose(tensor, 0, 1)
    pytorch_output_T = tensor.T.contiguous()
    torch.cuda.synchronize()

    # Benchmark Triton
    start_time = time.time()
    for _ in range(100):
        triton_output = triton_transpose(tensor)
    torch.cuda.synchronize()
    triton_time = (time.time() - start_time) / 100

    # Benchmark PyTorch (torch.transpose)
    start_time = time.time()
    for _ in range(100):
        pytorch_output_t = torch.transpose(tensor, 0, 1).contiguous()
    torch.cuda.synchronize()
    pytorch_time_t = (time.time() - start_time) / 100

    # Benchmark PyTorch (.T)
    # Note: .T is often a view, but we materialize it with .contiguous() for a fair comparison
    start_time = time.time()
    for _ in range(100):
        pytorch_output_T = tensor.T.contiguous()
    torch.cuda.synchronize()
    pytorch_time_T = (time.time() - start_time) / 100

    # Verification
    assert torch.allclose(triton_output, pytorch_output_t)
    assert torch.allclose(triton_output, pytorch_output_T)

    return triton_time, pytorch_time_t, pytorch_time_T


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA is not available. This benchmark requires a GPU.")
        exit()

    sizes = [
        (512, 512),
        # (1024, 1024),
        # (2048, 2048),
        # (4096, 4096),
        # (1024, 4096),
        # (4096, 1024),
        # (4096, 4096),
        # (4096, 1024*8),
        # (256, 1024*8),
    ] + [(512, 1024 * i) for i in range(1, 9)]

    print(
        f"{'Matrix Size':<15} | {'Triton (ms)':<15} | {'PyTorch.transpose (ms)':<25} | {'PyTorch .T (ms)':<15}"
    )
    print("-" * 80)

    for M, N in sizes:
        try:
            triton_time, pytorch_time_t, pytorch_time_T = benchmark(M, N)
            print(
                f"({M}, {N}){'':<6} | {triton_time * 1000:.4f}{'':<10} | {pytorch_time_t * 1000:.4f}{'':<18} | {pytorch_time_T * 1000:.4f}"
            )
        except Exception as e:
            print(f"Could not run for size ({M}, {N}). Error: {e}")
