import numpy as np


def load_balance(sizes, offset=None):
    # segment size
    # heuristics taken from OpenAI blocksparse code
    # https://github.com/openai/blocksparse/blob/master/blocksparse/matmul.py#L95
    # max_size = sizes.max()
    # if max_size > min_size * 2.0:
    #  seg_max = max(triton.cdiv(max_size, 4), min_size*2)
    # else:
    #  seg_max = max_size
    # seg_max = max_size
    seg_max = sizes.max()
    seg_min = max(int(np.ceil(seg_max / 4)), 4)
    # split reduction into segments
    div = sizes // seg_max
    rem = sizes % seg_max
    packs = div + (sizes < seg_min) + (rem >= seg_min)
    width = packs.sum()
    segments = np.empty(width, dtype=sizes.dtype)
    column = np.empty_like(segments)
    lockid = np.zeros_like(segments)
    maxid = np.zeros_like(segments)
    nlocks = 0
    current = 0
    col_idx = 0
    for i in range(len(sizes)):
        d, r = div[i], rem[i]
        isempty = sizes[i] < seg_min
        last = current + d + (r >= seg_min) + isempty
        # column id
        column[current:last] = col_idx
        # lock id
        if d > 1 or (d == 1 and r >= seg_min):
            nlocks += 1
            lockid[current:last] = nlocks
            maxid[current:last] = last - current
        # segment size
        segments[current : current + d] = seg_max
        if r < seg_min and not isempty:
            segments[current + d - 1] += r
        if r >= seg_min or isempty:
            segments[current + d] = r
        current = last
        col_idx += 1
    offsets = np.zeros_like(segments)
    offsets[1:] = np.cumsum(segments[:-1], axis=0)
    return segments, column, lockid, maxid, offset


if __name__ == "__main__":
    # Example usage
    sizes = np.array([10, 20, 30, 40, 50, 20, 30, 90, 120, 20])
    segments, column, lockid, maxid, offset = load_balance(sizes)
    print("Segments:", segments)
    print("Column IDs:", column)
    print("Lock IDs:", lockid)
    print("Max IDs:", maxid)
    print("Offsets:", offset)
