import numpy as np

import triton
triton.language.swizzle2d


def triton_swizzle2d(i,j, size_i, size_j, size_g):
    """
    Transforms the indices of a row-major `size_i * size_j` matrix into
    the indices of a column-major matrix for each group of `size_g` rows.

    For example, for :code:`size_i = size_j = 4` and :code:`size_g = 2`, it will
    transform ::

        [[0 , 1 , 2 , 3 ],
         [4 , 5 , 6 , 7 ],
         [8 , 9 , 10, 11],
         [12, 13, 14, 15]]

    into ::

        [[0, 2,  4 , 6 ],
         [1, 3,  5 , 7 ],
         [8, 10, 12, 14],
         [9, 11, 13, 15]]
    """
    ij = i * size_j + j
    size_gj = size_g * size_j
    # index of the group in which (i,j) is
    group_id = ij // size_gj
    # row-index of the first element of this group
    off_i = group_id * size_g
    # last group may have fewer rows
    size_g = min(size_i - off_i, size_g)
    # linear index with respect to the first element in this group
    ij = ij % size_gj
    # new row and column indices
    new_i = off_i + ij % size_g
    new_j = ij // size_g
    # new_ij = new_i * size_j + new_j
    return new_i, new_j
# def swizzle2d(pid_m, pid_n, num_pid_m, num_pid_n, group_size):
def swizzle2d(pid_m, pid_n, num_pid_m, num_pid_n, group_size):
    # # pid = tl.program_id(axis=0)
    # # pid_m = tl.program_id(axis=0)
    # # pid_n = tl.program_id(axis=1)
    # # num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    # # num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = group_size * num_pid_n
    pid = (pid_m*num_pid_n + pid_n)
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * group_size
    group_size_m = min(num_pid_m - first_pid_m, group_size)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    return pid_m, pid_n
    # ij = pid_m * num_pid_n + pid_n
    # size_gj = group_size * num_pid_n
    # group_id = ij // size_gj
    # off_i = group_id * group_size
    # # Last group may have fewer rows
    # size_g = min(num_pid_m - off_i, group_size)
    # # linear index with respect to the first element in this group
    # ij = ij % size_gj
    # # new row and column indices
    # new_i = off_i + ij % size_g
    # new_j = ij // size_g
    # return new_i, new_j

def swizzle2d_rows(i, j, size_i, size_j, size_g):
    ij = i * size_j + j
    num_groups_per_stripe = size_g * size_j
    group_id = ij // num_groups_per_stripe
    group_start = group_id * size_g
    group_size = min(size_i - group_start, size_g)

    pid_m = group_start + (ij % num_groups_per_stripe) % group_size
    pid_n = (ij % num_groups_per_stripe) // group_size
    group_m = pid_m // size_g
    start = 0
    if group_m % 2 == 1:
        start = size_j - 1
        pid_n = size_j - 1 - pid_n
    if (start - pid_n) % 2 == 1:
        pid_m = (
            pid_m
            - 2 * ((ij % num_groups_per_stripe) % group_size)
            - 1
            + group_size
        )
    return pid_m, pid_n

# # j = i
# M = 4
# N = 4
# G = 2

# p = np.arange(M*N).reshape(M,N)
# z = np.zeros((M,N), dtype=int)
# w = np.zeros((M,N), dtype=int)
# for i in range(M):
#     for j in range(N):
#         # print(f"i={i}, j={j} -> ", end="")
#         # print(f"({i,j}){swizzle2d(i, M, N, G)}", end=", ")
#         z[i,j] = swizzle2d(p[i,j], M, N, G)
#         w[i,j] = triton_swizzle2d(p[i,j], M, N, G)
#     # print()
# print(p)
# print(z)
# print(w)

# j = i
M = 7
N = 8
G = 3

p = np.arange(M * N).reshape(M, N)
z = np.zeros((M, N), dtype=int)
w = np.zeros((M, N), dtype=int)
for i in range(M):
    for j in range(N):
        # print(f"i={i}, j={j} -> ", end="")
        # print(f"({i,j}){swizzle2d(i, M, N, G)}", end=", ")
        new_i, new_j = swizzle2d(i, j, M, N, G)
        z[new_i, new_j] = p[i, j]
        # z[i, j] = p[new_i, new_j]
        # new_i_triton, new_j_triton = triton_swizzle2d(i,j, M, N, G)

        new_i_triton, new_j_triton = swizzle2d_rows(i,j, M, N, G)
        w[new_i_triton, new_j_triton] = p[i, j]
# print()
print("Original:")
print(p)
print("swizzle2d output:")
print(z)
print("triton_swizzle2d output:")
print(w)
