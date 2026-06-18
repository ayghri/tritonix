from typing import Optional
# This file implements a block-sparse matrix multiplication using Triton.
# It is designed to be a drop-in replacement for torch.matmul, but for block-sparse matrices.
# The implementation is heavily inspired by the OpenAI block-sparse kernels.
# Three modes of matrix multiplication are supported:
#  - SDD (Sparse = Dense x Dense): The output matrix is sparse.
#  - DSD (Dense = Sparse x Dense): The first input matrix is sparse.
#  - DDS (Dense = Dense x Sparse): The second input matrix is sparse.
#
# The main components are:
# 1. `_kernel`: A Triton kernel that performs the matrix multiplication on the GPU.
# 2. `_sparse_matmul`: A `torch.autograd.Function` that wraps the Triton kernel and handles
#    the forward and backward passes. It also contains helper functions for creating
#    Look-Up Tables (LUTs) that guide the sparse computation.
# 3. `MatMul`: A user-facing class that provides a convenient API for block-sparse matrix
#    multiplication, managing layouts and LUTs automatically.

import importlib
import torch

import triton
import triton.language as tl
import triton._C.libtriton as libtriton
from deepspeed.accelerator import get_accelerator


@triton.jit
def _kernel(
    A,
    B,
    C,
    stride_za,
    stride_ha,
    stride_ma,
    stride_ka,
    stride_zb,
    stride_hb,
    stride_kb,
    stride_nb,
    stride_zc,
    stride_hc,
    stride_mc,
    stride_nc,
    DS0,
    DS1,
    SDD_K,
    SDD_off_width,
    lut,
    locks,
    nlocks,
    **meta,
):
    """
    Triton kernel for block-sparse matrix multiplication.
    This kernel can handle three modes of matrix multiplication:
    - SDD (Sparse = Dense x Dense)
    - DSD (Dense = Sparse x Dense)
    - DDS (Dense = Dense x Sparse)

    The kernel is organized into three main parts:
    1. Prologue: Computes offsets for the input and output matrices based on the program ID and the LUT.
    2. Inner Loop: Performs the matrix multiplication using dot products of tiles.
    3. Epilogue: Writes the result back to the output matrix, handling atomic updates if necessary.
    """
    TM = meta["TM"]
    TN = meta["TN"]
    TK = meta["TK"]
    TZ = meta["TZ"]
    BLOCK = meta["BLOCK"]
    # ------------#
    # - Prologue -#
    # ------------#
    # Get program IDs for the three dimensions of the grid.
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    pidz = tl.program_id(2)

    # SDD (Sparse = Dense x Dense) mode
    if meta["SDD"]:
        # Adjust program ID for width offset in case of large grids.
        pid1 = pid1 + SDD_off_width
        # Calculate block IDs within the tile.
        blockidm = tl.arange(0, TM) // BLOCK
        blockidn = tl.arange(0, TN) // BLOCK
        # Calculate offsets into the LUT.
        offlutm = blockidm * (TN // BLOCK) * 4
        offlutn = blockidn * 4
        # Load metadata from the LUT header.
        header = lut + pid1 * (TM // BLOCK) * (TN // BLOCK) * 4
        z = tl.load(header + 0)  # Batch/head index
        i = tl.load(header + 1 + offlutm)  # Row block index
        j = tl.load(header + 2 + offlutn)  # Column block index
        AS1 = SDD_K // TZ  # Reduction size
        lockid = tl.where(TZ > 1, 1, 0) # Use locks if reduction is across Z dimension
        # Initial offsets for the K dimension.
        offka = pid0 * AS1
        offkb = pid0 * AS1
        # Offsets for M, N, and C are 0 as they are determined by LUT.
        offmc = 0
        offnc = 0
        offpa = 0
        offpb = 0
        maxid = TZ
        # Offsets for the H dimension (heads).
        offhc = 0
        offha = z
        offhb = z
        # Pointers to the M and N dimensions for the output.
        ram = i * BLOCK + (tl.arange(0, TM) % BLOCK)
        rbn = j * BLOCK + (tl.arange(0, TN) % BLOCK)
    # DSD (Dense = Sparse x Dense) or DDS (Dense = Dense x Sparse) mode
    else:
        # Load metadata from the LUT header.
        header = lut + pid0 * 6
        offset = tl.load(header + 0)  # Offset to pointer increments
        AS1 = tl.load(header + 1)  # Reduction size for this block
        column = tl.load(header + 2)  # Column index of the output block
        depth = tl.load(header + 3)  # Depth/head index
        lockid = tl.load(header + 4)  # Lock ID for atomic updates
        maxid = tl.load(header + 5)  # Maximum value for the lock counter
        pinc = lut + offset  # Pointer to increments table
        offhc = depth

        # DSD (Dense = Sparse x Dense) mode
        if meta["DSD"]:
            # output offset
            offnc = pid1 * TN
            offmc = column * TM
            offpc = 0
            # dense input (B) offset
            offnb = pid1 * TN
            offkb = tl.load(pinc)
            offkb = tl.multiple_of(offkb, 8)  # compiler hint
            offpb = 0
            # sparse input (A) offset
            offma = 0
            offka = 0
            offpa = tl.load(pinc + 1)
            offpa = tl.multiple_of(offpa, 8)  # compiler hint
            offpa = offpa * BLOCK * BLOCK
            offha = 0
            offhb = depth
        # DDS (Dense = Dense x Sparse) mode
        else:
            # output offset
            offmc = pid1 * TM
            offnc = column * TN
            offpc = 0
            # dense input (A) offset
            offma = pid1 * TM
            offka = tl.load(pinc)
            offka = tl.multiple_of(offka, 8)  # compiler hint
            offpa = 0
            # sparse input (B) offset
            offnb = 0
            offkb = 0
            offpb = tl.load(pinc + 1)
            offpb = tl.multiple_of(offpb, 8)  # compiler hint
            offpb = offpb * BLOCK * BLOCK
            offha = depth
            offhb = 0
        # Pointers to the M and N dimensions for the output.
        ram = offma + tl.arange(0, TM)
        rbn = offnb + tl.arange(0, TN)

    # initialize a, b pointers
    # Pointers to the K dimension.
    rka = offka + tl.arange(0, TK)
    rkb = offkb + tl.arange(0, TK)
    # Pointers to the input matrices A and B.
    pa = (
        A
        + pidz * stride_za
        + offha * stride_ha
        + offpa
        + ram[:, None] * stride_ma
        + rka[None, :] * stride_ka
    )
    pb = (
        B
        + pidz * stride_zb
        + offhb * stride_hb
        + offpb
        + rbn[None, :] * stride_nb
        + rkb[:, None] * stride_kb
    )
    # Create masks for loading data, important for handling padding.
    if meta["DDS"]:
        checkam = ram[:, None] < DS0
    else:
        checkam = AS1 > 0
    if meta["DSD"]:
        checkbn = rbn[None, :] < DS0
    else:
        checkbn = AS1 > 0
    # Load the first tile of A and B.
    a = tl.load(pa, mask=checkam, other=0.0)
    b = tl.load(pb, mask=checkbn, other=0.0)

    ## ---------------- ##
    ##    Inner Loop    ##
    ## ---------------- ##
    # Initialize accumulator.
    acc = tl.zeros((TM, TN), dtype=tl.float32)
    # Loop over the K dimension.
    for k in range(AS1, 0, -TK):
        # Perform matrix multiplication on the current tiles.
        acc += tl.dot(a, b)
        # Update pointers for the next iteration.
        if meta["SDD"]:
            inc_a = TK * stride_ka
            inc_b = TK * stride_kb
        else:
            pinc += 2
        if meta["DSD"]:
            # For DSD, B is dense and A is sparse.
            # We load increments for both from the LUT.
            inc_b = tl.load(pinc)
            inc_a = tl.load(pinc + 1)
            inc_b = tl.multiple_of(inc_b, 8)
            inc_a = tl.multiple_of(inc_a, 8)
            inc_b = inc_b * stride_kb
        if meta["DDS"]:
            # For DDS, A is dense and B is sparse.
            # We load increments for both from the LUT.
            inc_a = tl.load(pinc)
            inc_b = tl.load(pinc + 1)
            inc_a = tl.multiple_of(inc_a, 8)
            inc_b = tl.multiple_of(inc_b, 8)
            inc_a = inc_a * stride_ka
        pa += inc_a
        pb += inc_b
        # pre-fetch next tiles
        checkak = k > TK
        checkbk = k > TK
        checka = checkam & checkak
        checkb = checkbn & checkbk
        a = tl.load(pa, mask=checka)
        b = tl.load(pb, mask=checkb)
    # Convert accumulator to the output data type.
    c = acc.to(C.dtype.element_ty)

    # ---------------- #
    #    - Epilogue -   #
    # ---------------- #
    # Calculate output pointers and write back the result.
    if meta["SDD"]:
        checkc = True
        # Calculate block IDs and offsets for C from the LUT.
        rr_blockidm = tl.arange(0, TM) // BLOCK
        rr_blockidn = tl.arange(0, TN) // BLOCK
        rr_offlutm = rr_blockidm * (TN // BLOCK) * 4
        rr_offlutn = rr_blockidn * 4
        off_bkid = 3 + rr_offlutm[:, None] + rr_offlutn[None, :]
        bkid = tl.load(header + off_bkid)
        offpc = bkid * BLOCK * BLOCK
        rcm = tl.arange(0, TM) % BLOCK
        rcn = tl.arange(0, TN) % BLOCK
    else:
        rcm = offmc + tl.arange(0, TM)
        rcn = offnc + tl.arange(0, TN)
    # Create masks for writing data.
    if meta["DSD"]:
        checkc = rcn[None, :] < DS0
    if meta["DDS"]:
        checkc = rcm[:, None] < DS0

    # Pointer to the output matrix C.
    pc = (
        C
        + offpc
        + offhc * stride_hc
        + pidz * stride_zc
        + rcm[:, None] * stride_mc
        + rcn[None, :] * stride_nc
    )
    # write-back directly if no lock is needed.
    if lockid == 0:
        tl.store(pc, c, mask=checkc)
    # accumulate partial results using spin-locks if a lock is needed.
    else:
        # Get lock pointers.
        plock = (
            locks
            + tl.program_id(2) * nlocks * tl.num_programs(1)
            + tl.program_id(1) * nlocks
            + lockid
            - 1
        )
        pcount = plock + tl.num_programs(2) * tl.num_programs(1) * nlocks
        # Acquire lock.
        while tl.atomic_cas(plock, 0, 1) == 1:
            pass
        count = tl.load(pcount)
        # If it's the first time writing to this location, just store.
        if count == 0:
            tl.store(pc, c, mask=checkc)
        # Otherwise, accumulate.
        else:
            d = tl.load(pc, mask=checkc)
            tl.store(pc, d + c, mask=checkc)
        # Increment counter and release lock.
        tl.atomic_xchg(pcount, (count + 1) % maxid)
        tl.atomic_xchg(plock, 0)


##############
#  MAIN API  #
##############
class _sparse_matmul(torch.autograd.Function):
    """
    PyTorch autograd function for block-sparse matrix multiplication.
    This class wraps the Triton kernel and provides forward and backward passes.
    It also includes helper methods for creating Look-Up Tables (LUTs) that
    encode the sparse layout for efficient computation on the GPU.
    """
    # Caches for LUTs and locks to avoid recomputation.
    sdd_cache = dict()
    dsd_cache = dict()
    dds_cache = dict()
    locks = dict()

    # Given an array sizes representing reduction size for each
    # column of a block-mode matrix multiplication,
    # performs load-balancing to achieve more smaller reductions
    # between `seg_size` elements
    @staticmethod
    def load_balance(sizes, block):
        """
        Performs load balancing for reduction operations.
        This function takes a tensor of reduction sizes and splits them into smaller,
        more manageable segments. This helps to improve parallelism and GPU utilization,
        especially when the reduction sizes are highly variable.
        The implementation is based on heuristics from the OpenAI blocksparse library.
        """
        # global triton
        # if triton is None:
        #    triton = importlib.import_module('triton')
        # segment size
        # heuristics taken from OpenAI blocksparse code
        # https://github.com/openai/blocksparse/blob/master/blocksparse/matmul.py#L95
        max_size = sizes.max()
        # min_size = sizes[sizes != 0].min()
        # if max_size > min_size * 2.0:
        #  seg_max = max(triton.cdiv(max_size, 4), min_size*2)
        # else:
        #  seg_max = max_size
        seg_max = max_size
        seg_min = max(triton.cdiv(seg_max, 4), 4)
        # split reduction into segments
        div = sizes // seg_max
        rem = sizes % seg_max
        packs = div + (sizes < seg_min).long() + (rem >= seg_min).long()
        width = packs.sum()
        segments = torch.empty(width, dtype=sizes.dtype)
        column = torch.empty_like(segments)
        lockid = torch.zeros_like(segments)
        maxid = torch.zeros_like(segments)
        nlocks = 0
        current = 0
        col_idx = 0
        for i in range(len(sizes)):
            d, r = div[i], rem[i]
            isempty = sizes[i] < seg_min
            last = current + d + (r >= seg_min) + isempty
            # column id
            column[current:last] = col_idx
            # lock id: if a reduction is split, it needs a lock for atomic updates.
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
        offsets = torch.zeros_like(segments)
        offsets[1:] = torch.cumsum(segments[:-1], dim=0)
        return segments, column, lockid, maxid, offsets

    @staticmethod
    def get_locks(size, dev):
        """
        Allocates or retrieves a cached tensor for locks.
        Locks are used for atomic updates in the Triton kernel when multiple
        thread blocks write to the same output location.
        """
        if dev not in _sparse_matmul.locks or size > _sparse_matmul.locks[
            dev
        ].size(0):
            _sparse_matmul.locks[dev] = torch.zeros(
                size, dtype=torch.int32, device=dev
            )
        return _sparse_matmul.locks[dev]

    ##########################
    # SPARSE = DENSE x DENSE #
    ##########################

    @staticmethod
    def make_sdd_lut(layout, block, dtype, device):
        """
        Creates the Look-Up Table (LUT) for SDD (Sparse = Dense x Dense) mode.
        In SDD mode, the output matrix is sparse. The LUT encodes the locations
        of the non-zero blocks in the output.
        This function uses a C++ backend (`libtriton.superblock`) to efficiently
        generate the LUT from the sparsity layout.
        """
        # _sparse_matmul._load_utils()
        # start_width = 64 // block
        # segmented = _sparse_matmul.sdd_segment(layout.type(torch.int32), start_width)
        start_width = (128 if block > 16 else 32) // block
        layout = layout.type(torch.int32)
        segmented = libtriton.superblock(
            layout.data_ptr(),
            layout.shape[0],
            layout.shape[1],
            layout.shape[2],
            start_width,
        )
        luts, widths, packs = [], [], []
        for size, nnz in segmented:
            """ width = nnz.shape[0] // (size * size)
            h = nnz[:, 0]
            i = nnz[:, 1]
            j = nnz[:, 2]
            b = nnz[:, 3]
            lut = torch.stack((h, i, j, b), dim=1).view(-1).contiguous()
            luts.append(lut.type(torch.int32).to(device))
            widths.append(width)
            packs.append(size) """
            nnz = nnz.reshape(-1, 4)
            width = nnz.shape[0] // (size * size)
            luts.append(torch.from_numpy(nnz).type(torch.int32).to(device))
            widths.append(width)
            packs.append(size)
        # create locks
        return luts, None, widths, packs

    @staticmethod
    def _sdd_matmul(
        a,
        b,
        trans_a,
        trans_b,
        trans_c,
        spdims,
        block,
        luts,
        num_locks,
        widths,
        packs,
        bench,
        time,
    ):
        """
        Performs SDD (Sparse = Dense x Dense) matrix multiplication.
        This function sets up the grid and launches the Triton kernel.
        It handles cases where the grid size exceeds the maximum allowed by the GPU
        by decomposing the operation into multiple kernel calls.
        """
        if trans_c:
            a, b = b, a
            trans_a, trans_b = not trans_b, not trans_a
        AS0 = a.size(0)
        # Shape check
        a_dim = -2 if trans_a else -1
        b_dim = -1 if trans_b else -2
        a_inner, b_inner = a.shape[a_dim], b.shape[b_dim]
        if a_inner != b_inner:
            raise ValueError(
                f"Size of tensor A along the {a_dim} dim ({a_inner}) must match size "
                f"of tensor B along the {b_dim} dim ({b_inner})"
            )
        if a_inner % 16 != 0:
            raise ValueError("Reduction size for SDD must be a multiple of 16")

        batch_size = a.size(0)
        a_outer = a.size(3 if trans_a else 2)
        dtype = a.dtype
        is_16_multiple = a_inner % 16 == 0
        is_32_multiple = a_inner % 32 == 0
        is_64_multiple = a_inner % 64 == 0
        if not is_16_multiple:
            raise ValueError("Reduction size for SDD must be a multiple of 16")
        # device = a.device
        # create kernel
        total_width = sum(
            [width * pack * pack for width, pack in zip(widths, packs)]
        )
        c = torch.empty(
            (batch_size, total_width, block, block),
            dtype=dtype,
            device=a.device,
        )
        for lut, width, pack in zip(luts, widths, packs):
            F32TK = [8, 16]
            F16TK = [16]
            F16TK += [32] if is_32_multiple else []
            F16TK += [64] if is_64_multiple else []
            TK = {torch.float32: F32TK, torch.float16: F16TK}[dtype]
            num_lock = 1
            meta = {
                "TM": block * pack,
                "TN": block * pack,
                "BLOCK": block,
                "TK": TK[0],
                "TZ": 1,
                "SDD": True,
                "DSD": False,
                "DDS": False,
            }
            # create output
            locks = _sparse_matmul.get_locks(
                2 * width * AS0 * num_lock, a.device
            )
            # maximum grid size is 65535
            # so operation might be decomposed into multiple
            # kernel calls
            max_width = 49152
            # total = 0 if bench else None
            for off_width in range(0, width, max_width):
                def grid(meta):
                    return [
                        meta["TZ"],
                        min(max_width, width - off_width),
                        batch_size,
                    ]
                _kernel[grid](
                    a,
                    b,
                    c,
                    a.stride(0),
                    a.stride(1),
                    a.stride(3 if trans_a else 2),
                    a.stride(2 if trans_a else 3),
                    b.stride(0),
                    b.stride(1),
                    b.stride(3 if trans_b else 2),
                    b.stride(2 if trans_b else 3),
                    c.stride(0),
                    c.stride(0),
                    c.stride(2),
                    c.stride(3),
                    a_outer,
                    a_outer,
                    a_inner,
                    off_width,
                    lut,
                    locks,
                    num_lock,
                    num_warps=4,
                    **meta,
                )
        # save for backward pass
        return c

    ##########################
    # DENSE = DENSE x SPARSE #
    ##########################

    # Given a binary layout of 0s and 1s,
    # Construct look-up table for efficient execution on GPUs
    @staticmethod
    def make_dxx_lut(
        layout, block, step, trans, device, transform=lambda idx: idx
    ):
        """
        Creates the Look-Up Table (LUT) for DSD (Dense = Sparse x Dense) and
        DDS (Dense = Dense x Sparse) modes.
        The LUT contains metadata for each reduction segment, including its size,
        output column, and pointers to the increments for the input matrices.
        This function uses the `load_balance` method to handle variable reduction sizes.
        """
        # load-balancing
        _empty = torch.tensor([], dtype=torch.int64, device=layout.device)
        segments = _empty.clone()
        column = _empty.clone()
        depth = _empty.clone()
        lockid = _empty.clone()
        maxid = _empty.clone()
        offsets = _empty.clone()
        current_offset = 0
        current_maxid = 0
        for z in range(layout.size(0)):
            if trans:
                sizes = torch.sum(layout[z, :, :], 1)
            else:
                sizes = torch.sum(layout[z, :, :], 0)
            z_segments, z_column, z_lockid, z_maxid, z_offsets = (
                _sparse_matmul.load_balance(sizes, block)
            )
            z_depth = z * torch.ones_like(z_segments)
            z_lockid[z_lockid > 0] += current_maxid
            current_maxid = z_lockid.max()
            # concatenate depth
            segments = torch.cat((segments, z_segments))
            column = torch.cat((column, z_column))
            depth = torch.cat((depth, z_depth))
            maxid = torch.cat((maxid, z_maxid))
            offsets = torch.cat((offsets, current_offset + z_offsets))
            lockid = torch.cat((lockid, z_lockid))
            current_offset += layout[z, :, :].sum()
        segments *= step
        # pointer increments
        if trans:
            nnz = layout.nonzero()
        else:
            nnz = layout.transpose(1, 2).nonzero()
        num_blocks = nnz.size(0)
        offsets = torch.min(
            offsets, (num_blocks - 1) * torch.ones_like(offsets)
        )
        idx = transform(nnz[:, 2] * block)
        xincs = idx.clone()
        xincs[1:] -= idx[:-1]
        # divide block into multiple steps
        div = block // step
        xincs = xincs.view(-1, 1).repeat(1, div)
        xincs[:, 1:] = step
        xincs[:, 0] -= (div - 1) * step
        # first increment for each reduction is actually the offset
        xincs[offsets[segments > 0], 0] = idx[offsets[segments > 0]]
        xincs = xincs.view(-1)
        # block-mode input increments
        if trans:
            widx = torch.arange(num_blocks)
        else:
            widx = _empty.clone()
            current_offset = 0
            for z in range(layout.size(0)):
                layoutw = layout[z, :, :].clone()
                msum = layoutw.sum()
                layoutw[layoutw > 0] = 1 + torch.arange(msum)
                widx = torch.cat(
                    (widx, current_offset + layoutw.T[layoutw.T > 0] - 1)
                )
                current_offset += msum
        widx = widx
        wincs = widx * block * block
        wincs[1:] -= widx[:-1] * block * block
        wincs = wincs.view(-1, 1).repeat(1, div)
        if trans:
            wincs[:, 1:] = step
            wincs[:, 0] -= (div - 1) * step
        else:
            wincs[:, 1:] = step * block
            wincs[:, 0] -= (div - 1) * step * block
        wincs[offsets[segments > 0], 0] = widx[offsets[segments > 0]]
        wincs = wincs.view(-1)
        # adjust offset and segment size
        offsets *= 2 * div
        segments *= div
        # create header containing metadata for each reduction segment
        width = column.size(0)
        offsets += 6 * width
        header = (
            torch.stack(
                (offsets, segments, column, depth, lockid, maxid), dim=1
            )
            .view(-1)
            .contiguous()
        )
        # create increments table
        incs = torch.stack((xincs, wincs), dim=1).view(-1).contiguous()
        incs = torch.cat(
            (incs, torch.zeros(2, device=incs.device, dtype=incs.dtype))
        )
        # create lut
        lut = torch.cat((header, incs))
        lut = lut.type(torch.int32).to(device)
        # create locks
        num_locks = max(1, lockid.max().item())
        return lut, num_locks, width, None

    @staticmethod
    def _dds_matmul(
        a,
        b,
        trans_a,
        trans_b,
        trans_c,
        spdims,
        block,
        lut,
        num_locks,
        width,
        packs,
        bench,
        time,
    ):
        """
        Performs DDS (Dense = Dense x Sparse) matrix multiplication.
        This function sets up the grid and launches the Triton kernel.
        """
        global triton
        if triton is None:
            triton = importlib.import_module("triton")

        # shapes / dtypes
        AS0 = a.size(0)
        AS1 = a.size(1)
        AS2 = a.size(3 if trans_a else 2)
        # AS3 = a.size(2 if trans_a else 3)
        # BS0 = spdims[0]
        # BS1 = block * spdims[2 if trans_b else 1]
        BS2 = block * spdims[1 if trans_b else 2]
        dtype = a.dtype
        # kernel
        meta = {
            "TN": block,
            "TM": 128,
            "TK": 16,
            "BLOCK": block,
            "TZ": 1,
            "SDD": False,
            "DSD": False,
            "DDS": True,
        }
        # output
        CS0 = AS0
        CS1 = AS1
        CS2 = BS2 if trans_c else AS2
        CS3 = AS2 if trans_c else BS2
        locks = _sparse_matmul.get_locks(
            2 * AS0 * AS2 // 32 * num_locks, a.device
        )
        c = torch.empty((CS0, CS1, CS2, CS3), dtype=dtype, device=a.device)
        def grid(meta):
            return [width, triton.cdiv(AS2, meta["TM"]), AS0]
        _kernel[grid](
            a,
            b,
            c,
            a.stride(0),
            a.stride(1),
            a.stride(3 if trans_a else 2),
            a.stride(2 if trans_a else 3),
            b.stride(0),
            b.stride(1),
            b.stride(3 if trans_b else 2),
            b.stride(2 if trans_b else 3),
            c.stride(0),
            c.stride(1),
            c.stride(3 if trans_c else 2),
            c.stride(2 if trans_c else 3),
            AS2,
            BS2,
            0,
            0,
            lut,
            locks,
            num_locks,
            num_warps=4,
            **meta,
        )
        return c

    @staticmethod
    def _dsd_matmul(
        a,
        b,
        trans_a,
        trans_b,
        trans_c,
        spdims,
        block,
        lut,
        num_locks,
        width,
        packs,
        bench,
        time,
    ):
        """
        Performs DSD (Dense = Sparse x Dense) matrix multiplication.
        This function sets up the grid and launches the Triton kernel.
        """
        global triton
        if triton is None:
            triton = importlib.import_module("triton")

        # shapes / dtypes
        # AS0 = spdims[0]
        AS1 = block * spdims[2 if trans_a else 1]
        # AS2 = block * spdims[1 if trans_a else 2]
        BS0 = b.size(0)
        BS1 = b.size(1)
        # BS2 = b.size(3 if trans_b else 2)
        BS3 = b.size(2 if trans_b else 3)
        dtype = a.dtype
        # kernel

        meta = {
            "TM": block,
            "TN": 128,
            "TK": 16,
            "BLOCK": block,
            "TZ": 1,
            "SDD": False,
            "DSD": True,
            "DDS": False,
        }
        # output
        CS0 = BS0
        CS1 = BS1
        CS2 = BS3 if trans_c else AS1
        CS3 = AS1 if trans_c else BS3
        locks = _sparse_matmul.get_locks(
            2 * BS0 * BS3 // 32 * num_locks, a.device
        )
        c = torch.empty((CS0, CS1, CS2, CS3), dtype=dtype, device=a.device)
        def grid(meta):
            return [width, triton.cdiv(BS3, meta["TN"]), BS0]
        _kernel[grid](
            a,
            b,
            c,
            a.stride(0),
            a.stride(1),
            a.stride(3 if trans_a else 2),
            a.stride(2 if trans_a else 3),
            b.stride(0),
            b.stride(1),
            b.stride(3 if trans_b else 2),
            b.stride(2 if trans_b else 3),
            c.stride(0),
            c.stride(1),
            c.stride(2),
            c.stride(3),
            BS3,
            AS1,
            0,
            0,
            lut,
            locks,
            num_locks,
            num_warps=4,
            **meta,
        )
        return c

    # Dictionary mapping mode to the corresponding matmul function.
    fn = {
        "sdd": _sdd_matmul.__get__(object),
        "dsd": _dsd_matmul.__get__(object),
        "dds": _dds_matmul.__get__(object),
    }

    @staticmethod
    def forward(
        ctx,
        a,
        b,
        trans_a,
        trans_b,
        trans_c,
        mode,
        spdims,
        block,
        c_lut,
        c_num_locks,
        c_width,
        c_packs,
        c_bench,
        c_time,
        da_lut,
        da_num_locks,
        da_width,
        da_packs,
        da_bench,
        da_time,
        db_lut,
        db_num_locks,
        db_width,
        db_packs,
        db_bench,
        db_time,
    ):
        """
        Forward pass for the block-sparse matrix multiplication.
        This function calls the appropriate matmul function based on the `mode`
        and saves the necessary tensors and parameters for the backward pass.
        """
        c = _sparse_matmul.fn[mode](
            a,
            b,
            trans_a,
            trans_b,
            trans_c,
            spdims,
            block,
            c_lut,
            c_num_locks,
            c_width,
            c_packs,
            c_bench,
            c_time,
        )
        # save for backward
        ctx.save_for_backward(a, b)
        ctx.da_num_locks = da_num_locks
        ctx.da_lut = da_lut
        ctx.da_width = da_width
        ctx.da_packs = da_packs
        ctx.da_bench = da_bench
        ctx.da_time = da_time
        ctx.db_lut = db_lut
        ctx.db_num_locks = db_num_locks
        ctx.db_width = db_width
        ctx.db_bench = db_bench
        ctx.db_packs = db_packs
        ctx.db_time = db_time
        ctx.mode = mode
        ctx.spdims = spdims
        ctx.block = block
        ctx.trans_a = trans_a
        ctx.trans_b = trans_b
        return c

    @staticmethod
    def backward(ctx, *grad_outputs):
        """
        Backward pass for the block-sparse matrix multiplication.
        This function computes the gradients with respect to the input tensors `a` and `b`.
        The gradient computation is also a block-sparse matrix multiplication.
        """
        dc = grad_outputs[0]
        # saved for backward
        a, b = ctx.saved_tensors
        mode = ctx.mode
        da = None
        db = None
        # gradients w.r.t. a
        if ctx.needs_input_grad[0]:
            # The mode for the backward pass is derived from the forward pass mode.
            # For example, if forward is DSD, backward for `da` is SDD.
            mode_da = mode[1] + mode[0] + mode[2]
            da = _sparse_matmul.fn[mode_da](
                dc,
                b,
                False,
                not ctx.trans_b,
                ctx.trans_a,
                ctx.spdims,
                ctx.block,
                ctx.da_lut,
                ctx.da_num_locks,
                ctx.da_width,
                ctx.da_packs,
                ctx.da_bench,
                ctx.da_time,
            )
        # gradients w.r.t. b
        if ctx.needs_input_grad[1]:
            # The mode for the backward pass is derived from the forward pass mode.
            # For example, if forward is DSD, backward for `db` is DSD.
            mode_db = mode[2] + mode[1] + mode[0]
            db = _sparse_matmul.fn[mode_db](
                a,
                dc,
                not ctx.trans_a,
                False,
                ctx.trans_b,
                ctx.spdims,
                ctx.block,
                ctx.db_lut,
                ctx.db_num_locks,
                ctx.db_width,
                ctx.db_packs,
                ctx.db_bench,
                ctx.db_time,
            )
        return (
            da,
            db,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class MatMul:
    """Block-Sparse MatMul class; this class handles three types of matrix-multiplication:
       - sparse = dense X dense
       - dense = sparse X dense
       - dense = dense X sparse

    For more details about sparsity config, please see `Generative Modeling with Sparse Transformers`: https://arxiv.org/abs/1904.10509
    """

    def make_lut(self, dtype, device):
        """
        Generates and caches the Look-Up Tables (LUTs) for the forward and backward passes.
        The LUTs are generated based on the sparsity layout and the matrix multiplication mode.
        This function creates LUTs for the output (c), and the gradients of the inputs (da, db).
        """
        key = (dtype, device)
        if key in self.lut_cache:
            return self.lut_cache[key]
        # C look-up table
        layout, block = self.layout, self.block
        step = 16
        c_lut, c_num_locks, c_width, c_packs = None, None, None, None
        da_lut, da_num_locks, da_width, da_packs = None, None, None, None
        db_lut, db_num_locks, db_width, db_packs = None, None, None, None
        if self.mode == "sdd":
            c_lut, c_num_locks, c_width, c_packs = _sparse_matmul.make_sdd_lut(
                layout, block, dtype, device
            )
        elif self.mode == "dsd":
            c_lut, c_num_locks, c_width, c_packs = _sparse_matmul.make_dxx_lut(
                layout, block, step, not self.trans_a, device
            )
        elif self.mode == "dds":
            c_lut, c_num_locks, c_width, c_packs = _sparse_matmul.make_dxx_lut(
                layout, block, step, self.trans_b, device
            )
        # DA look-up table (for gradient w.r.t. a)
        if self.mode == "sdd":
            da_lut, da_num_locks, da_width, da_packs = (
                _sparse_matmul.make_dxx_lut(layout, block, step, True, device)
            )
        elif self.mode == "dsd":
            da_lut, da_num_locks, da_width, da_packs = (
                _sparse_matmul.make_sdd_lut(layout, block, dtype, device)
            )
        elif self.mode == "dds":
            da_lut, da_num_locks, da_width, da_packs = (
                _sparse_matmul.make_dxx_lut(
                    layout, block, step, not self.trans_b, device
                )
            )
        # DB look-up table (for gradient w.r.t. b)
        if self.mode == "sdd":
            db_lut, db_num_locks, db_width, db_packs = (
                _sparse_matmul.make_dxx_lut(layout, block, step, False, device)
            )
        elif self.mode == "dsd":
            db_lut, db_num_locks, db_width, db_packs = (
                _sparse_matmul.make_dxx_lut(
                    layout, block, step, self.trans_a, device
                )
            )
        elif self.mode == "dds":
            db_lut, db_num_locks, db_width, db_packs = (
                _sparse_matmul.make_sdd_lut(layout, block, dtype, device)
            )
        self.lut_cache[key] = (
            c_lut,
            c_num_locks,
            c_width,
            c_packs,
            da_lut,
            da_num_locks,
            da_width,
            da_packs,
            db_lut,
            db_num_locks,
            db_width,
            db_packs,
        )
        return self.lut_cache[key]

    def __init__(
        self, layout, block, mode, trans_a=False, trans_b=False, bench=False
    ):
        """Initialize the Block-Sparse MatMul class.

        Arguments:
             layout: required: sparsity layout tensor
             block: required: an integer determining the block size.
             mode: required: a string determining type of matmul; ('sdd') sparse = dense X dense, ('dsd') dense = sparse X dense, ('dds') dense = dense X sparse
             trans_a: optional: a boolean determining if multiplication needs to be applied on transpose of input a; default is false
             trans_b: optional: a boolean determining if multiplication needs to be applied on transpose of input b; default is false
             bench: optional: set if you want to do benchmarking
        """

        if mode not in ["sdd", "dsd", "dds"]:
            raise NotImplementedError("Supported modes are: sdd, dsd, dds")
        # look-up table cache
        self.lut_cache = dict()
        # attributes
        self.trans_a = trans_a
        self.trans_b = trans_b
        self.mode = mode
        self.block = block
        self.layout = layout
        layout_dim = layout.ndim
        assert layout_dim in (2, 3), (
            "Layout should be a 2 or 3 dimensional tensor of 0s and 1s"
        )
        if not mode == "sdd":
            # Dims to be reduced on the 'inside' of the matmul, either -1 or -2
            trans_dense, trans_sparse, sparse_inner = (
                (trans_b, trans_a, -1)
                if mode == "dsd"
                else (trans_a, trans_b, -2)
            )
            self.dense_inner_dim = (
                -((sparse_inner % 2) + 1) if not trans_dense else sparse_inner
            )
            sparse_inner = (
                sparse_inner if not trans_sparse else -((sparse_inner % 2) + 1)
            )

            # Inner dim of the dense input should be equal to the inner dim of the sparse input
            self.dense_inner_size = layout.shape[sparse_inner] * block
            # Expected shape for sparse inputs
            self.sparse_shape = (layout.sum().item(), block, block)

        # Support using the same layout across attention heads etc.
        if layout_dim == 2:
            layout = layout.unsqueeze(0)

        layout = (
            layout.long()
        )  # Above code assumes the layout tensor is an integral type

        self.spdims = layout.shape
        # timings
        self.bench = bench
        self.time_c = None
        self.time_da = None
        self.time_db = None

    # pad shapes of a tensor to make it
    # compatible with kernel calls
    @staticmethod
    def _pad_shape(x, is_sparse):
        """
        Pads the shape of a tensor with leading singleton dimensions to make it
        compatible with the 4D tensors expected by the Triton kernel.
        """
        max_dim = 3 if is_sparse else 4
        for i in range(max_dim - x.dim()):
            x = x.unsqueeze(0)
        return x

    def __call__(self, a, b):
        """Applies Block-Sparse MatMul.

        For more details about sparsity config, please see `Generative Modeling with Sparse Transformers`: https://arxiv.org/abs/1904.10509

        Arguments:
             a: required: a dense/block-sparse tensor; first input of mat-mul
             b: required: a dense/block-sparse tensor; second input of mat-mul

        Return:
             c: a dense/block-sparse tensor result of a X b
        """
        (
            c_lut,
            c_num_locks,
            c_width,
            c_packs,
            da_lut,
            da_num_locks,
            da_width,
            da_packs,
            db_lut,
            db_num_locks,
            db_width,
            db_packs,
        ) = self.make_lut(a.dtype, a.device)
        # timings
        time_c = [None]
        time_da = [None]
        time_db = [None]

        original_dims = max(a.ndim, b.ndim)
        a, b = self._validate_inputs(a, b)

        # pad shapes with ones
        a = MatMul._pad_shape(a, self.mode == "dsd")
        b = MatMul._pad_shape(b, self.mode == "dds")
        # execute

        c: Optional[torch.Tensor] = _sparse_matmul.apply(
            a,
            b,
            self.trans_a,
            self.trans_b,
            False,
            self.mode,
            self.spdims,
            self.block,
            c_lut,
            c_num_locks,
            c_width,
            c_packs,
            self.bench,
            time_c,
            da_lut,
            da_num_locks,
            da_width,
            da_packs,
            self.bench,
            time_da,
            db_lut,
            db_num_locks,
            db_width,
            db_packs,
            self.bench,
            time_db,
        )

        # This removes any leading singleton dimensions we may have added to the tensor that weren't in the input
        if c is not None:
            dims_to_trim = c.ndim - original_dims
            for _ in range(dims_to_trim):
                c = c.squeeze(0)

        self.time_c = time_c[0]
        self.time_da = time_da[0]
        self.time_db = time_db[0]
        return c

    def _validate_inputs(self, a, b):
        """
        Validates the input tensors for device, dtype, and shape compatibility.
        It also handles autocasting to float16 if enabled.
        """
        if a.device != b.device:
            raise ValueError(
                f"Inputs must be on the same device; got {a.device} for tensor A "
                f"and {b.device} for tensor B"
            )
        accelerator = get_accelerator()
        if not accelerator or not accelerator.on_accelerator(a):
            raise ValueError("Only GPU devices are supported for now")

        # When autocast is enabled, torch.matmul autocasts to float16, so we do the same here
        if torch.is_autocast_enabled():
            a, b = a.half(), b.half()
        elif a.dtype != b.dtype:
            raise ValueError(
                f"Inputs must be the same dtype; got {a.dtype} for A and {b.dtype} for B"
            )

        mode = self.mode
        if mode != "sdd":
            # One input is sparse, so we need to check its shape.
            dense, dense_name, sparse, sparse_name = (
                (a, "A", b, "B") if mode == "dds" else (b, "B", a, "A")
            )
            dense_inner = dense.shape[self.dense_inner_dim]
            if dense_inner != self.dense_inner_size:
                raise ValueError(
                    f"Expected tensor {dense_name} to have size {self.dense_inner_size} at dim "
                    f"{self.dense_inner_dim % dense.ndim}, got {dense_inner}."
                )

            if sparse.shape[-len(self.sparse_shape) :] != self.sparse_shape:
                raise ValueError(
                    f"Expected tensor with trailing dimensions of shape {self.sparse_shape} for argument "
                    f"{sparse_name}, got {sparse.shape}"
                )

        def add_extra_dims(x):
            # Add extra leading singleton dimensions if needed
            dims_needed = 4 - x.ndim
            if dims_needed > 0:
                singletons = [1] * dims_needed
                x = x.view(*singletons, *x.shape)
            elif dims_needed < 0:
                raise ValueError(
                    "Tensors with more than 4 dimensions are not currently supported"
                )

            return x

        # Pad shapes with leading singleton dimensions
        a = add_extra_dims(a)
        b = add_extra_dims(b)

        return a, b
