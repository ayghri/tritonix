import triton
from triton.runtime import Autotuner


def get_gemm_config(m, n, k, group_m, num_stages, num_warps):
    return triton.Config(
        {
            "block_m": m,
            "block_n": n,
            "block_k": k,
            "group_m": group_m,
        },
        num_stages=num_stages,
        num_warps=num_warps,
    )


def get_gemm_splitk_config(
    block_m, block_n, block_k, group_m, split_k, num_stages, num_warps
):
    return triton.Config(
        {
            "block_m": block_m,
            "block_n": block_n,
            "block_k": block_k,
            "group_m": group_m,
            "split_k": split_k,
        },
        num_stages=num_stages,
        num_warps=num_warps,
    )


def get_autotune_configs():
    return [
        get_gemm_config(32, 32, 256, 8, 2, 4),
        get_gemm_config(32, 64, 64, 24, 1, 8),
        get_gemm_config(64, 32, 16, 8, 4, 4),
        get_gemm_config(64, 32, 32, 8, 4, 4),
        get_gemm_config(64, 64, 32, 8, 4, 4),
        get_gemm_config(64, 64, 128, 8, 4, 4),
        get_gemm_config(64, 128, 128, 8, 4, 4),
        get_gemm_config(128, 32, 32, 8, 4, 4),
        get_gemm_config(128, 64, 16, 4, 4, 4),
        get_gemm_config(128, 64, 32, 8, 4, 4),
        get_gemm_config(128, 64, 64, 8, 4, 4),
        get_gemm_config(128, 64, 64, 8, 8, 4),
        # get_gemm_config(128, 128, 32, 8, 4, 2),
        # get_gemm_config(128, 128, 32, 8, 4, 4),
        # get_gemm_config(128, 128, 32, 4, 4, 4),
        # get_gemm_config(128, 128, 32, 8, 4, 4),
        # get_gemm_config(128, 128, 64, 8, 4, 4),
        # get_gemm_config(128, 128, 64, 4, 4, 4),
        # get_gemm_config(128, 128, 64, 8, 4, 16),
        # get_gemm_config(128, 128, 64, 8, 3, 4),
        # get_gemm_config(128, 128, 64, 16, 2, 4),
        # get_gemm_config(128, 128, 128, 8, 2, 4),
    ]


def get_splitk_autotune_configs():
    return [
        # get_gemm_splitk_config(64, 32, 16, 8, 2, 4, 4),
        # get_gemm_splitk_config(64, 32, 32, 8, 2, 4, 4),
        # get_gemm_splitk_config(64, 64, 128, 16, 4, 3, 4),
        # get_gemm_splitk_config(64, 64, 128, 8, 8, 3, 4),
        # get_gemm_splitk_config(64, 64, 128, 8, 16, 3, 4),
        # get_gemm_splitk_config(128, 128, 64, 8, 16, 4, 4),
        # Best parameters: block_m=64, block_n=32, block_k=64, group_m=24, num_stages=1, num_warps=8
        get_gemm_splitk_config(32, 64, 64, 24, 1, 1, 8),
        get_gemm_splitk_config(64, 128, 16, 8, 6, 3, 4),
        get_gemm_splitk_config(64, 128, 32, 8, 6, 3, 4),
        get_gemm_splitk_config(64, 64, 64, 8, 8, 3, 4),
        get_gemm_splitk_config(64, 64, 64, 16, 8, 3, 4),
        get_gemm_splitk_config(128, 128, 32, 16, 5, 3, 4), # 128x128x32 split 5 used by cutlass for 1024*32
        get_gemm_splitk_config(128, 128, 32, 16, 5, 4, 4),
        get_gemm_splitk_config(128, 128, 32, 8, 5, 4, 4),
        get_gemm_splitk_config(128, 128, 32, 8, 5, 3, 4),
        get_gemm_splitk_config(128, 128, 64, 16, 8, 3, 4),
        get_gemm_splitk_config(128, 128, 64, 16, 1, 3, 4),
        get_gemm_splitk_config(128, 64, 64, 16, 4, 3, 4),
        get_gemm_splitk_config(128, 64, 64, 16, 8, 3, 4),
        get_gemm_splitk_config(256, 128, 32, 8, 4, 4, 4),
        get_gemm_splitk_config(256, 128, 32, 8, 2, 4, 4),
        get_gemm_splitk_config(256, 128, 32, 8, 1, 4, 4),
        get_gemm_splitk_config(256, 128, 32, 4, 1, 4, 4),
        get_gemm_splitk_config(256, 128, 32, 8, 1, 4, 8),
        get_gemm_splitk_config(256, 128, 16, 8, 2, 3, 4),
        # get_gemm_splitk_config(64, 64, 64, 16, 1, 3, 4),
        # get_gemm_splitk_config(32, 32, 128, 8, 8, 4, 4),
        # get_gemm_splitk_config(64, 64, 128, 16, 8, 4, 8),
        # get_gemm_splitk_config(64, 64, 256, 16, 8, 2, 8),
        # get_gemm_splitk_config(128, 128, 64, 8, 4, 4, 4),
        # get_gemm_splitk_config(128, 128, 64, 8, 8, 8, 4),
    ]


def get_autotune_conv2d_bwd_configs():
    return [
        # BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M, num_warps, num_stages
        # GEMM_M=C_OUT, GEMM_N=C_IN*FILTER_H*FILTER_W, GEMM_K=BATCH_SIZE*H_OUT*W_OUT
        get_gemm_config(16, 16, 32, 4, 4, 4),
        get_gemm_config(16, 16, 32, 4, 4, 6),
        get_gemm_config(16, 16, 32, 4, 4, 8),
        get_gemm_config(16, 16, 64, 4, 4, 4),
        get_gemm_config(16, 16, 64, 4, 4, 6),
        get_gemm_config(16, 16, 64, 4, 4, 8),
        get_gemm_config(16, 16, 64, 8, 4, 4),
        get_gemm_config(16, 16, 128, 4, 4, 4),
        get_gemm_config(16, 16, 128, 8, 4, 4),
        get_gemm_config(16, 32, 64, 8, 4, 4),
        get_gemm_config(16, 32, 128, 4, 4, 4),
        get_gemm_config(16, 32, 128, 8, 4, 4),
        get_gemm_config(32, 32, 64, 4, 4, 4),
        get_gemm_config(32, 32, 64, 8, 4, 4),
        get_gemm_config(32, 32, 64, 8, 4, 6),
        get_gemm_config(32, 32, 64, 8, 4, 8),
        get_gemm_config(32, 32, 128, 8, 4, 4),
        get_gemm_config(32, 64, 64, 8, 4, 4),
    ]


def wrap_autotuner(kernel, configs, reset_to_zero=None, restore_value=None):
    """
    Wraps a Triton kernel with autotuning capabilities.
    """
    return Autotuner(
        kernel,
        kernel.arg_names,
        configs=configs,
        key=kernel.specialize_keys,
        reset_to_zero=reset_to_zero,
        restore_value=restore_value,
    )
