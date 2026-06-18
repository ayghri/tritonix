import torch
import triton
import triton.language as tl

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


@triton.jit
def fused_mlp_workspace_sync(
    # Pointers to matrices
    x_ptr,
    w1_ptr,
    w2_ptr,
    y_ptr,
    workspace_z_ptr,  # Global workspace for intermediate σ(Z)
    sync_workspace_ptr,  # Global workspace for row-block completion flags
    # Matrix dimensions
    M,
    K,
    N,
    P,
    # Strides
    stride_xm,
    stride_xk,
    stride_w1k,
    stride_w1n,
    stride_w2n,
    stride_w2p,
    stride_ym,
    stride_yp,
    stride_wzm,
    stride_wzn,  # Strides for the intermediate workspace_Z
    # Tile dimensions
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_P: tl.constexpr,
    # Misc
    SM_COUNT: tl.constexpr,
):
    """
    Triton kernel for Y = σ(x @ W1) @ W2 using a persistent grid and
    a global workspace for synchronization.
    """
    # =================================================================
    # PHASE 1: COMPUTE Z' = σ(x @ W1) and store in workspace_Z
    # =================================================================
    # Work distribution for Phase 1: iterate over all blocks of the intermediate Z matrix
    pid = tl.program_id(axis=0)
    grid_m_z = tl.cdiv(M, BLOCK_M)
    grid_n_z = tl.cdiv(N, BLOCK_N)
    total_blocks_z = grid_m_z * grid_n_z

    # Persistent loop for Phase 1
    # Each program computes blocks where (block_id % SM_COUNT == pid)
    for block_id_z in range(pid, total_blocks_z, SM_COUNT):
        # Derive 2D block coordinates for the intermediate Z matrix
        pid_m = block_id_z // grid_n_z
        pid_n = block_id_z % grid_n_z

        # --- Standard GEMM logic for a block of Z ---
        # Create an accumulator for the Z block
        acc_z = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # Pointers for inputs
        offs_x = (
            pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
        ) * stride_xm + tl.arange(0, BLOCK_K)[None, :]
        offs_w1 = (
            tl.arange(0, BLOCK_K)[:, None] * stride_w1k
            + (pid_n * BLOCK_N + tl.arange(0, BLOCK_N))[None, :]
        )

        # Loop over K dimension
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            x_ptrs = x_ptr + offs_x + k * BLOCK_K
            w1_ptrs = w1_ptr + offs_w1 + k * BLOCK_K * stride_w1k

            mask_x = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]) < M
            mask_w1 = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N))[None, :] < N

            x_tile = tl.load(x_ptrs, mask=mask_x, other=0.0)
            w1_tile = tl.load(w1_ptrs, mask=mask_w1, other=0.0)

            acc_z += tl.dot(x_tile, w1_tile)

        # Apply activation function (e.g., ReLU)
        z_block = tl.maximum(acc_z, 0.0)

        # --- Store result to global workspace_Z ---
        offs_wz = (
            pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
        ) * stride_wzm + (pid_n * BLOCK_N + tl.arange(0, BLOCK_N))[None, :]
        wz_ptrs = workspace_z_ptr + offs_wz
        mask_wz = ((pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]) < M) & (
            pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        )[None, :] < N
        tl.store(wz_ptrs, z_block, mask=mask_wz)

        # --- Signal completion of this block in the sync workspace ---
        # The index corresponds to the row block
        sync_ptr = sync_workspace_ptr + pid_m
        tl.atomic_add(sync_ptr, 1)

    # =================================================================
    # PHASE 2: COMPUTE Y = Z' @ W2, reading Z' from workspace_Z
    # =================================================================
    # Work distribution for Phase 2: iterate over all blocks of the final Y matrix
    grid_m_y = tl.cdiv(M, BLOCK_M)
    grid_p_y = tl.cdiv(P, BLOCK_P)
    total_blocks_y = grid_m_y * grid_p_y

    # Persistent loop for Phase 2
    for block_id_y in range(pid, total_blocks_y, SM_COUNT):
        # Derive 2D block coordinates for the final Y matrix
        pid_m = block_id_y // grid_p_y
        pid_p = block_id_y % grid_p_y

        # --- Synchronization: Spin-wait for the required row block of Z' to be ready ---
        sync_ptr = sync_workspace_ptr + pid_m
        # The row is ready when all N-dimension blocks have been computed
        required_val = tl.cdiv(N, BLOCK_N)
        # Poll the sync flag until it's ready
        while tl.load(sync_ptr) < required_val:
            pass  # Spin-wait

        # --- Standard GEMM logic for a block of Y ---
        acc_y = tl.zeros((BLOCK_M, BLOCK_P), dtype=tl.float32)

        # Pointers for inputs (Z' is read from workspace_z)
        offs_wz = (
            pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
        ) * stride_wzm + tl.arange(0, BLOCK_N)[None, :]
        offs_w2 = (
            tl.arange(0, BLOCK_N)[:, None] * stride_w2n
            + (pid_p * BLOCK_P + tl.arange(0, BLOCK_P))[None, :]
        )

        # Loop over N dimension (which was the K dimension for GEMM2)
        for n in range(0, tl.cdiv(N, BLOCK_N)):
            wz_ptrs = workspace_z_ptr + offs_wz + n * BLOCK_N
            w2_ptrs = w2_ptr + offs_w2 + n * BLOCK_N * stride_w2n

            mask_wz = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]) < M
            mask_w2 = (pid_p * BLOCK_P + tl.arange(0, BLOCK_P))[None, :] < P

            wz_tile = tl.load(wz_ptrs, mask=mask_wz, other=0.0)
            w2_tile = tl.load(w2_ptrs, mask=mask_w2, other=0.0)

            acc_y += tl.dot(wz_tile, w2_tile)

        # --- Store final result to Y ---
        offs_y = (
            pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
        ) * stride_ym + (pid_p * BLOCK_P + tl.arange(0, BLOCK_P))[None, :]
        y_ptrs = y_ptr + offs_y
        mask_y = ((pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]) < M) & (
            pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
        )[None, :] < P
        tl.store(y_ptrs, acc_y, mask=mask_y)


# This is a helper function to get the number of SMs
# NOTE: This is an example, a more robust method might be needed
def get_sm_count():
    device = torch.cuda.current_device()
    return torch.cuda.get_device_properties(device).multi_processor_count


def fused_mlp(x, w1, w2):
    M, K = x.shape
    K, N = w1.shape
    N, P = w2.shape

    # Block sizes
    BLOCK_M, BLOCK_K, BLOCK_N, BLOCK_P = 128, 64, 64, 128

    # Output tensor
    y = torch.empty((M, P), device=x.device, dtype=x.dtype)

    # Workspace for intermediate Z' matrix
    workspace_z = torch.empty((M, N), device=x.device, dtype=x.dtype)

    # Workspace for synchronization flags (one flag per row-block of Z)
    # MUST be initialized to zero
    sync_workspace_size = triton.cdiv(M, BLOCK_M)  # BLOCK_M should match here
    sync_workspace = torch.zeros(
        sync_workspace_size, device=x.device, dtype=torch.int32
    )

    # Triton launch grid
    SM_COUNT = get_sm_count()
    grid = (SM_COUNT,)  # Launch one program per SM

    fused_mlp_workspace_sync[grid](
        x,
        w1,
        w2,
        y,
        workspace_z,
        sync_workspace,
        M,
        K,
        N,
        P,
        x.stride(0),
        x.stride(1),
        w1.stride(0),
        w1.stride(1),
        w2.stride(0),
        w2.stride(1),
        y.stride(0),
        y.stride(1),
        workspace_z.stride(0),
        workspace_z.stride(1),
        BLOCK_M=BLOCK_M,  # type: ignore
        BLOCK_K=BLOCK_K,  # type: ignore
        BLOCK_N=BLOCK_N,  # type: ignore
        BLOCK_P=BLOCK_P,  # type: ignore
        SM_COUNT=SM_COUNT,
        num_stages=3,
        num_warps=4,  # Adjust based on your GPU architecture
    )
    return y


if __name__ == "__main__":
    # --- Verification ---
    torch.manual_seed(0)
    x = torch.randn((512, 512), device="cuda", dtype=torch.float16)
    w1 = torch.randn((512, 1024), device="cuda", dtype=torch.float16)
    w2 = torch.randn((1024, 512), device="cuda", dtype=torch.float16)

    # PyTorch reference implementation
    y_torch = torch.nn.functional.relu(x @ w1) @ w2

    # Triton implementation
    y_triton = fused_mlp(x, w1, w2)

    # Compare results
    print(f"Max difference: {torch.max(torch.abs(y_torch - y_triton))}")
    assert torch.allclose(y_torch, y_triton, atol=1e-2, rtol=1e-2)
    print("✅ Verification successful!")

    def torch_mlp(x, w1, w2):
        return torch.nn.functional.relu(x @ w1) @ w2

    compiled_torch_mlp = torch.compile(
        torch_mlp, mode="max-autotune", fullgraph=True
    )

    # ---------------------------------------------------------------------
    # 3. Benchmarking Harness
    # ---------------------------------------------------------------------

    @triton.testing.perf_report(
        [
            triton.testing.Benchmark(
                x_names=["N"],  # The dimension we are varying
                x_vals=[256, 512, 1024, 2048, 4096, 8192],
                line_arg="provider",  # This will create a different line for each provider
                line_vals=["torch_eager", "torch_compile", "triton_workspace"],
                line_names=[
                    "PyTorch Eager",
                    "Torch.compile",
                    "Triton (Workspace Sync)",
                ],
                styles=[("blue", "-"), ("green", "-"), ("red", "-")],
                ylabel="TFLOP/s",
                plot_name="mlp-benchmark-vary-M",
                args={"M": 512, "K": 2048, "P": 2048},
            ),
            triton.testing.Benchmark(
                x_names=["K"],  # Varying the hidden dimension
                x_vals=[256, 512, 1024, 2048, 4096, 8192],
                line_arg="provider",
                line_vals=["torch_eager", "torch_compile", "triton_workspace"],
                line_names=[
                    "PyTorch Eager",
                    "Torch.compile",
                    "Triton (Workspace Sync)",
                ],
                styles=[("blue", "-"), ("green", "-"), ("red", "-")],
                ylabel="TFLOP/s",
                plot_name="mlp-benchmark-vary-N",
                args={"M": 512, "N": 1024, "P": 1024},
            ),
        ]
    )
    def benchmark(M, K, N, P, provider):
        x = torch.randn((M, K), device="cuda", dtype=torch.float16)
        w1 = (
            torch.randn((N, K), device="cuda", dtype=torch.float16)
            .t()
            .contiguous()
        )
        w2 = (
            torch.randn((P, N), device="cuda", dtype=torch.float16)
            .t()
            .contiguous()
        )

        quantiles = [0.5, 0.2, 0.8]
        ms = min_ms = max_ms = 0.0
        if provider == "torch_eager":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: torch_mlp(x, w1, w2), quantiles=quantiles
            )
        if provider == "torch_compile":
            # First call is a warmup for compilation
            compiled_torch_mlp(x, w1, w2)
            torch.cuda.synchronize()
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: compiled_torch_mlp(x, w1, w2),
                quantiles=quantiles,
                warmup=500,
            )
        if provider == "triton_workspace":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: fused_mlp(x, w1, w2), quantiles=quantiles
            )

        # Total FLOPs for the MLP
        flops = 2 * M * K * N + 2 * M * N * P
        tflops = flops / (ms * 1e-3) / 1024**4
        return tflops

    benchmark.run(print_data=True, show_plots=True)
