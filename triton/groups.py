import numpy as np


def cdiv(a, b):
    return (a + b - 1) // b


BLOCK_M = 8
BLOCK_N = 8
GROUP_M = 4
M, N = 13, 12

# num_pid_m = cdiv(M, BLOCK_M)
# num_pid_n = cdiv(N, BLOCK_N)
# num_pid_in_group = GROUP_M * num_pid_n

# pid = np.arange(0, num_pid_m * num_pid_n)
# group_id = pid // num_pid_in_group
# first_pid_m = group_id * GROUP_M
# group_size_m = np.minimum(num_pid_m - first_pid_m, GROUP_M)
# pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
# pid_n = (pid % num_pid_in_group) // group_size_m


def grouped_ordering(i, M, N, G_M):
    num_groups_per_stripe = G_M * N
    group_id = i // num_groups_per_stripe
    group_start = group_id * G_M
    group_size = np.minimum(M - group_start, G_M)
    m = group_start + ((i % num_groups_per_stripe) % group_size)
    n = (i % num_groups_per_stripe) // group_size
    return m, n


def z_order_2d(x, y):
    answer = 0
    bits = max(len(bin(x)), len(bin(y))) - 2
    for i in range(bits):
        mshifted = 1 << i
        shift = i
        answer |= ((x & mshifted) << shift) | ((y & mshifted) << (shift + 1))
    return answer


def serpentine_order_2d(i, M, N, G_M):
    num_groups_per_strip = N * G_M
    group_id = i // num_groups_per_strip
    group_start = group_id * G_M
    group_size = np.minimum(M - group_start, G_M)
    m = group_start + (i % num_groups_per_strip) % group_size
    n = (i % num_groups_per_strip) // group_size
    group_m = m // G_M
    start = 0
    if group_m % 2 == 1:
        start = N - 1
        n = N - 1 - n
    if (start - n) % 2 == 1:
        m = m - 2 * ((i % num_groups_per_strip) % group_size) - 1 + group_size
    return m, n


# for p, p_m, p_n in zip(pid, pid_m, pid_n):
#     normal_m = p // num_pid_n
#     normal_n = p % num_pid_n
#     z_order = z_order_2d(normal_m, normal_n)
#     zorder_m = z_order // num_pid_n
#     zorder_n= z_order % num_pid_n
#     # print("pid_m:", pid_m)
#     # print("pid_n:", pid_n)
#     print(f"pid: {p:3d}|, z-order:{z_order:3d},no_group: ({normal_m:2d},{normal_n:2d})|"
#          f" with group ({p_m:2d},{p_n:2d}), zorder (m,n): {zorder_m:2d},{zorder_n:2d}")

for p in range(N * M):
    m, n = serpentine_order_2d(p, M, N, GROUP_M)
    m_g, n_g = grouped_ordering(p, M, N, GROUP_M)
    print(f"pid: {p:3d}|,(m,n): ({m_g:2d},{n_g:2d})|({m:2d},{n:2d})")
# print("N, M, GROUP_M:", N, M, GROUP_M)
