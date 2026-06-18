from tritonix.utils.hilbert import (
    hilbert_rect_coords,
    hilbert_permutation,
    hilbert_inverse_permutation,
    hilbert_swizzled_coords,
)


def test_rect_coords_cover_rectangle_small():
    coords = hilbert_rect_coords(3, 5)
    assert len(coords) == 15
    assert len(set(coords)) == 15
    assert all(0 <= i < 3 and 0 <= j < 5 for i, j in coords)


def test_permutation_inverse_roundtrip():
    for m, n in [(1, 1), (2, 2), (3, 4), (4, 3), (5, 5)]:
        P = hilbert_permutation(m, n)
        inv_perm = hilbert_inverse_permutation(m, n)
        # I[P[r]] == r for all r
        for r, h in enumerate(P):
            assert inv_perm[h] == r


def test_swizzled_coords_is_bijection():
    m, n = 4, 7
    coords = hilbert_swizzled_coords(m, n)
    assert len(coords) == m * n
    assert len(set(coords)) == m * n
