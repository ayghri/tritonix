from __future__ import annotations

from typing import Dict, List, Tuple


def _is_power_of_two(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0


def hilbert_d2xy(n: int, d: int) -> Tuple[int, int]:
    """Convert Hilbert distance d to (x, y) for an n x n grid.

    Requirements: n must be a power of two.
    Implementation follows the classic Butz algorithm.
    """
    if not _is_power_of_two(n):
        raise ValueError(f"n must be a power of two, got {n}")

    x = 0
    y = 0
    t = d
    s = 1
    while s < n:
        rx = 1 & (t // 2)
        ry = 1 & (t ^ rx)
        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            # rotate
            x, y = y, x
        x += s * rx
        y += s * ry
        t //= 4
        s *= 2
    return x, y


def hilbert_rect_coords(m: int, n: int) -> List[Tuple[int, int]]:
    """Return a list of (i, j) block coordinates of an m x n rectangle
    ordered by a Hilbert space-filling traversal.

    The rectangle is embedded into the next power-of-two square and we walk
    the Hilbert curve, selecting only coordinates that fall within [0, m) x [0, n)
    until we've collected m*n elements. This yields a bijection over the rectangle.
    """
    if m <= 0 or n <= 0:
        return []
    # side length of the embedding square
    N = 1 << (max(m, n) - 1).bit_length()
    coords: List[Tuple[int, int]] = []
    d = 0
    total = m * n
    # Iterate d along the Hilbert distance, picking in-bounds cells
    while len(coords) < total:
        x, y = hilbert_d2xy(N, d)
        if y < m and x < n:
            coords.append((y, x))  # (row=i, col=j)
        d += 1
    return coords


def hilbert_permutation(m: int, n: int) -> List[int]:
    """Return a permutation P of length m*n mapping row-major index -> Hilbert rank.

    If r = i*n + j (row-major), then P[r] gives the position along the
    Hilbert traversal order for the cell (i, j).
    """
    coords = hilbert_rect_coords(m, n)
    pos: Dict[Tuple[int, int], int] = {(i, j): k for k, (i, j) in enumerate(coords)}
    out = [0] * (m * n)
    k = 0
    for i in range(m):
        base = i * n
        for j in range(n):
            out[base + j] = pos[(i, j)]
            k += 1
    return out


def hilbert_inverse_permutation(m: int, n: int) -> List[int]:
    """Return inverse permutation I of length m*n mapping Hilbert rank -> row-major index.

    That is, if P = hilbert_permutation(m, n) then I[P[r]] = r.
    """
    P = hilbert_permutation(m, n)
    inv_perm = [0] * (m * n)
    for r, h in enumerate(P):
        inv_perm[h] = r
    return inv_perm


def hilbert_swizzled_coords(m: int, n: int) -> List[Tuple[int, int]]:
    """Convenience: identical to hilbert_rect_coords(m, n).

    Provided for naming clarity when used to swizzle MMA block coordinates.
    """
    return hilbert_rect_coords(m, n)
