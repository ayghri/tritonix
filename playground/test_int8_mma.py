"""Test if Triton supports INT8 MMA with block_k=16."""
import torch
import triton
import triton.language as tl


@triton.jit
def matmul_int8_kernel(
    a_ptr, b_ptr, c_ptr,
    m, n, k,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_n = pid_n * block_n + tl.arange(0, block_n)
    offs_k = tl.arange(0, block_k)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((block_m, block_n), dtype=tl.int32)
    for i in range(0, tl.cdiv(k, block_k)):
        k_remaining = k - i * block_k
        mask_k = offs_k < k_remaining
        a = tl.load(a_ptrs, mask=mask_k[None, :], other=0)
        b = tl.load(b_ptrs, mask=mask_k[:, None], other=0)
        acc = tl.dot(a, b, acc, out_dtype=tl.int32)
        a_ptrs += block_k * stride_ak
        b_ptrs += block_k * stride_bk

    offs_cm = pid_m * block_m + tl.arange(0, block_m)
    offs_cn = pid_n * block_n + tl.arange(0, block_n)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    mask = (offs_cm[:, None] < m) & (offs_cn[None, :] < n)
    tl.store(c_ptrs, acc, mask=mask)


def run_int8(m, n, k, block_m, block_n, block_k):
    a = torch.randint(-128, 127, (m, k), device="cuda", dtype=torch.int8)
    b = torch.randint(-128, 127, (k, n), device="cuda", dtype=torch.int8)
    c = torch.empty((m, n), device="cuda", dtype=torch.int32)
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    matmul_int8_kernel[grid](
        a, b, c,
        m, n, k,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        block_m=block_m, block_n=block_n, block_k=block_k,
    )
    # torch doesn't have int8 matmul, use float32 reference
    ref = (a.float() @ b.float()).int()
    max_err = (c - ref).abs().max().item()
    return c, ref, max_err


def main():
    print(f"Triton version: {triton.__version__}")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"CUDA capability: {torch.cuda.get_device_capability()}")
    print()

    m, n = 256, 256
    block_m, block_n = 64, 64

    print("=== Correctness ===")
    for block_k in [16, 32, 48, 64, 128]:
        k = max(block_k, 128)
        print(f"block_k={block_k:3d}, K={k}: ", end="", flush=True)
        try:
            _, _, max_err = run_int8(m, n, k, block_m, block_n, block_k)
            print(f"OK  max_err={max_err}")
        except Exception as e:
            # Print full error for diagnosis
            print(f"FAILED")
            print(f"    {e}")

    # Benchmark the ones that work
    print("\n=== Benchmark (M=4096, N=4096, K=4096) ===")
    m, n, k = 4096, 4096, 4096
    block_m, block_n = 128, 128

    for block_k in [16, 32, 64, 128]:
        print(f"\n  block_k={block_k}:", end=" ", flush=True)
        try:
            a = torch.randint(-128, 127, (m, k), device="cuda", dtype=torch.int8)
            b = torch.randint(-128, 127, (k, n), device="cuda", dtype=torch.int8)

            def run(bk=block_k):
                c = torch.empty((m, n), device="cuda", dtype=torch.int32)
                grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
                matmul_int8_kernel[grid](
                    a, b, c, m, n, k,
                    a.stride(0), a.stride(1),
                    b.stride(0), b.stride(1),
                    c.stride(0), c.stride(1),
                    block_m=block_m, block_n=block_n, block_k=bk,
                )
                return c

            ms = triton.testing.do_bench(run)
            tops = 2 * m * n * k / (ms * 1e-3) / 1e12
            print(f"{ms:.3f} ms  ({tops:.1f} TOPS)")
        except Exception as e:
            err = str(e).split("\n")[0]
            print(f"FAILED  {err}")


if __name__ == "__main__":
    main()
