import torch
import triton
import triton.language as tl

# def torch_blockify(a, block_m, block_n):
#     """
#     Reshape a matrix into blocks.
#     """
#     assert a.shape[0] % block_m == 0 and a.shape[1] % block_n == 0
#     b = a.view(a.shape[0] // block_m, block_m, a.shape[1] // block_n, block_n)
#     return b.permute(0, 2, 1, 3).reshape(-1, block_m, block_n)

# m, n = 6, 8
# block_m = 3
# block_n = 4
# a = torch.arange(n*m).reshape(m, n).cuda()
# b = torch_blockify(a, block_m, block_n)
# print(a)
# print(b)


# (The prepare_A and prepare_B functions from before)
def prepare_A(dense_a, block_m, block_k):
    m, k = dense_a.shape
    num_blocks_m = m // block_m
    num_blocks_k = k // block_k

    tiled_a = dense_a.reshape(num_blocks_m, block_m, num_blocks_k, block_k)
    tiled_a = tiled_a.permute(0, 2, 1, 3)
    contiguous_a = tiled_a.contiguous().reshape(
        num_blocks_m * num_blocks_k, block_m, block_k
    )

    return contiguous_a


def prepare_B(dense_b, block_k, block_n, p):
    k, n = dense_b.shape
    num_blocks_k = k // block_k
    num_blocks_n = n // block_n

    b_values_list = []
    b_metadata = torch.zeros((num_blocks_n, p), dtype=torch.int32)

    for j in range(num_blocks_n):
        non_zero_block_indices = torch.randperm(num_blocks_k)[:p]
        b_metadata[j] = non_zero_block_indices.sort().values

        for i in sorted(non_zero_block_indices.tolist()):
            block = dense_b[
                i * block_k : (i + 1) * block_k, j * block_n : (j + 1) * block_n
            ]
            b_values_list.append(block.T.contiguous())

    b_values = torch.stack(b_values_list)
    return b_values.cuda(), b_metadata.cuda()


@triton.jit
def block_mma_kernel(
    a_ptr,
    b_values_ptr,
    b_metadata_ptr,
    c_ptr,
    m,
    n,
    k,
    stride_a_tile,
    stride_a_m,
    stride_a_k,
    stride_b_values_tile,
    stride_b_values_n,
    stride_b_values_k,
    stride_b_metadata_n,
    stride_b_metadata_p,
    stride_c_m,
    stride_c_n,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    p: tl.constexpr,
    num_blocks_k: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    c_block_ptr = tl.make_block_ptr(
        base=c_ptr,
        shape=(m, n),
        strides=(stride_c_m, stride_c_n),
        offsets=(pid_m * block_m, pid_n * block_n),
        block_shape=(block_m, block_n),
        order=(1, 0),
    )

    acc = tl.zeros((block_m, block_n), dtype=tl.float32)

    b_metadata_col_ptr = b_metadata_ptr + pid_n * stride_b_metadata_n

    for i in range(p):
        b_block_row_idx = tl.load(b_metadata_col_ptr + i * stride_b_metadata_p)

        a_tile_ptr = (
            a_ptr + (pid_m * num_blocks_k + b_block_row_idx) * stride_a_tile
        )

        a_block_ptr = tl.make_block_ptr(
            base=a_tile_ptr,
            shape=(block_m, block_k),
            strides=(stride_a_m, stride_a_k),
            offsets=(0, 0),
            block_shape=(block_m, block_k),
            order=(1, 0),
        )
        a = tl.load(a_block_ptr)

        b_block_idx = pid_n * p + i
        b_tile_ptr = b_values_ptr + b_block_idx * stride_b_values_tile

        b_block_ptr = tl.make_block_ptr(
            base=b_tile_ptr,
            shape=(block_n, block_k),
            strides=(stride_b_values_n, stride_b_values_k),
            offsets=(0, 0),
            block_shape=(block_n, block_k),
            order=(1, 0),
        )
        b = tl.load(b_block_ptr)
        b = tl.trans(b)

        acc += tl.dot(a, b)

    tl.store(c_block_ptr, acc.to(c_ptr.dtype.element_ty))


def main():
    # Matrix dimensions
    m, k, n = 512, 512, 512
    block_m, block_k, block_n = 64, 32, 64
    p = 4  # Number of non-zero blocks per block-column in B

    # Create dense matrices
    torch.manual_seed(0)
    dense_a = torch.randn((m, k), device="cuda", dtype=torch.float16)
    dense_b = torch.randn((k, n), device="cuda", dtype=torch.float16)

    # Prepare matrices in the specified formats
    a = prepare_A(dense_a, block_m, block_k)
    b_values, b_metadata = prepare_B(dense_b, block_k, block_n, p)

    # Output tensor
    c = torch.empty((m, n), device="cuda", dtype=torch.float16)

    # Grid for launching the kernel
    grid = (m // block_m, n // block_n)

    num_blocks_k = k // block_k

    block_mma_kernel[grid](
        a,
        b_values,
        b_metadata,
        c,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        a.stride(2),
        b_values.stride(0),
        b_values.stride(1),
        b_values.stride(2),
        b_metadata.stride(0),
        b_metadata.stride(1),
        c.stride(0),
        c.stride(1),
        block_m, # type: ignore[no-untyped-call] 
        block_n, # type: ignore[no-untyped-call]
        block_k, # type: ignore[no-untyped-call]
        p, # type: ignore[no-untyped-call]
        num_blocks_k, # type: ignore[no-untyped-call]
    )

    # Verification
    # Reconstruct sparse B to compute reference C
    dense_b_reconstructed = torch.zeros_like(dense_b)
    for j in range(n // block_n):
        for i_idx, i in enumerate(b_metadata[j]):
            block_data = b_values[j * p + i_idx].T
            dense_b_reconstructed[
                i * block_k : (i + 1) * block_k, j * block_n : (j + 1) * block_n
            ] = block_data

    ref_c = torch.matmul(dense_a, dense_b_reconstructed)

    # Compare results
    print(
        "Triton output matches reference:",
        torch.allclose(c, ref_c, atol=1e-3, rtol=1e-2),
    )


if __name__ == "__main__":
    main()
