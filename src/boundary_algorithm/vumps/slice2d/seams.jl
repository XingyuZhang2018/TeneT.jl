# ═══════════════════════════════════════════════════════════════════════════════
# M5: vumps_step assembly + the QR gather seam.
# Design: docs/2026-06-15-m5-vumps-step-slice2d-assembly-design.md
#
# The four M4 solvers cover calls 2-5 of the 6-call vumps_step pipeline. M5 adds the
# two ENDPOINTS (ALCtoAC entry, ACCtoALAR exit) + orchestration + block init. Both
# endpoints follow the SAME pattern (R1-confirmed lowest-risk): gather the operand
# StructArray to FULL, run the UNMODIFIED serial kernel, scatter the output(s) back to
# blocks. C is REPLICATED throughout (never gathered/scattered). This reuses the M4
# gather/scatter rrules (take-my-block / allreduce) with ZERO new rrules, and inherits
# the serial kernels' exact per-cell accumulation (errL/errR) for free.
# ═══════════════════════════════════════════════════════════════════════════════

# StructArray-level full↔block shims (differentiable: StructArray ctor rrule +
# per-cell slice2d_gather/slice2d_scatter rrules). gather/scatter map over .data (the
# UNIQUE cells), so they gather each unique cell exactly once and are rank-uniform
# (the .pattern is replicated). bcast is init-only (outside AD) — no rrule needed.
gather_struct(SA, grid::Slice2DGrid)  = StructArray([slice2d_gather(t, grid)  for t in SA.data], SA.pattern)
scatter_struct(SA, grid::Slice2DGrid) = StructArray([slice2d_scatter(t, grid) for t in SA.data], SA.pattern)
bcast_struct(SA, root::Integer, comm) = StructArray([MPI.bcast(t, root, comm) for t in SA.data], SA.pattern)

"""
    AC_blk = ALCtoAC_slice2d_gather_ref(AL_blk, C, grid)

Distributed entry seam (call 1). Block AL's contracted last leg (`r2`-split) ×
replicated full C needs a reduction over the `r2` partition; v1 does it by gathering
AL to full, running the verbatim serial `ALCtoAC`, and scattering the result AC back
to blocks (Option A — uniform with the QR seam, lowest risk). AD: gather take-my-block
adjoint → serial @tensor → scatter allreduce adjoint. Parity-load-bearing: `ACenv_slice2d`
seeds a FINITE power iteration from this exact AC, so it must match serial bit-equally.
"""
function ALCtoAC_slice2d_gather_ref(AL_blk, C, grid::Slice2DGrid)
    AL_full = gather_struct(AL_blk, grid)
    AC_full = ALCtoAC(AL_full, C)
    return scatter_struct(AC_full, grid)
end

function _ALCtoAC_slice2d_one(AL_blk, Ck, grid::Slice2DGrid)
    l_rs = split_ranges(size(Ck, 2), grid.N2)
    br = l_rs[grid.r2 + 1]
    partial = similar(AL_blk, size(AL_blk, 1), size(AL_blk, 2), size(AL_blk, 3), size(Ck, 2))
    Cb = Ck[br, :]
    @tensor partial[a, i, j, d] := AL_blk[a, i, j, b] * Cb[b, d]
    return _slice2d_row_reduce_scatter_last(partial, grid, l_rs)
end

function ALCtoAC_slice2d(AL_blk, C, grid::Slice2DGrid)
    AC_data = map(eachindex(AL_blk.data)) do k
        _ALCtoAC_slice2d_one(AL_blk.data[k], C.data[k], grid)
    end
    return StructArray(AC_data, AL_blk.pattern)
end

"""
    AL_blk, AR_blk, errL, errR = ACCtoALAR_slice2d_gather_ref(AC_blk, C, grid)

The QR GATHER SEAM (exit, call 6). The per-cell full-χ QR/LQ (`qrpos`/`lqpos`) cannot
run on a χ-block, so gather the whole AC StructArray to FULL, run the UNMODIFIED serial
`ACCtoALAR` (which factorizes the full χ matrices on every rank identically), and scatter
the full AL/AR back to blocks. C is already replicated (never gathered). errL/errR are
replicated QR residuals — computed identically on every rank, NEVER allreduced (an
allreduce would multiply by P). The exact serial accumulation index sets (ACCtoAL over
all positions, ACCtoAR over unique data + jr=mod1(j-1,Nj)) are inherited by reusing the
serial kernel verbatim. AD: gather take-my-block → qrpos/lqpos rrules (replicated) →
scatter allreduce. MUST use the full `slice2d_gather` (take-my-block), NOT the row/col
reduce-scatter wrappers — those over-count the AC gradient by N1/N2 (the QR is replicated,
not a distributed contraction). Design §2.
"""
function ACCtoALAR_slice2d_gather_ref(AC_blk, C, grid::Slice2DGrid)
    AC_full = gather_struct(AC_blk, grid)
    AL_full, AR_full, errL, errR = ACCtoALAR(AC_full, C)
    return scatter_struct(AL_full, grid), scatter_struct(AR_full, grid), errL, errR
end


"""
    AL_blk, errL = ACCtoAL_slice2d_gather_ref(AC_blk, C, grid)

Left-only QR gather seam (plaquette has no AR): gather AC to full, run the verbatim
serial `ACCtoAL`, scatter AL. C replicated. Same AD as M5 ACCtoALAR_slice2d minus the AR half.
"""
function ACCtoAL_slice2d_gather_ref(AC_blk, C, grid::Slice2DGrid)
    AC_full = gather_struct(AC_blk, grid)
    AL_full, errL = ACCtoAL(AC_full, C)
    return scatter_struct(AL_full, grid), errL
end

ACCtoAL_slice2d(AC_blk, C, grid::Slice2DGrid) = ACCtoAL_tsqr_slice2d(AC_blk, C, grid)

function _tsqr_front_col_axis(A_mat, grid::Slice2DGrid)
    Qloc, Rloc = qrpos(A_mat)
    chi = size(Rloc, 1)
    r_rs = split_ranges(grid.N1 * chi, grid.N1)
    Rstack = slice2d_gather_col(Rloc, grid, r_rs)
    Q2stack, R = qrpos(Rstack)
    Q2 = Q2stack[r_rs[grid.r1 + 1], :]
    return Qloc * Q2, R
end

function _tsqr_front_row_axis(A_mat, grid::Slice2DGrid)
    Qloc, Rloc = qrpos(A_mat)
    chi = size(Rloc, 1)
    r_rs = split_ranges(grid.N2 * chi, grid.N2)
    Rstack = slice2d_gather_first_row(Rloc, grid, r_rs)
    Q2stack, R = qrpos(Rstack)
    Q2 = Q2stack[r_rs[grid.r2 + 1], :]
    return Qloc * Q2, R
end

function _tail_adjoint_rowblock(AC_col)
    return reshape(permutedims(conj(AC_col), (2, 3, 4, 1)),
                   size(AC_col, 2) * size(AC_col, 3) * size(AC_col, 4),
                   size(AC_col, 1))
end

function _qtail_from_qr_rows(Q_rows, AC_col)
    return reshape(Q_rows', size(AC_col))
end

function _tsqr_front_rowblock(A_row, grid::Slice2DGrid)
    # For a row block with a local first-chi leg and a full last-chi leg, the
    # serial `_to_front` equivalent is (local_a * D * D) x full_chi.
    A_mat = reshape(A_row, size(A_row, 1) * size(A_row, 2) * size(A_row, 3), size(A_row, 4))
    return _tsqr_front_col_axis(A_mat, grid)
end

qrpos_colrep(C, grid::Slice2DGrid) = qrpos(C)
lqpos_colrep(C, grid::Slice2DGrid) = lqpos(C)

function _acc_to_al_tsqr_one(AC_blk, C, grid::Slice2DGrid)
    χ = ChainRulesCore.ignore_derivatives() do
        MPI.Allreduce(size(AC_blk, 1), +, grid.col_comm)
    end
    l_rs = split_ranges(χ, grid.N2)
    AC_row = slice2d_gather_row(AC_blk, grid, l_rs)
    QAC, RAC = _tsqr_front_rowblock(AC_row, grid)
    QC, RC = qrpos_colrep(C, grid)
    AL_row = reshape(QAC * QC', size(AC_row))
    AL_blk = AL_row[ntuple(i -> i == ndims(AL_row) ? l_rs[grid.r2 + 1] : Colon(), ndims(AL_row))...]
    return AL_blk, norm(RAC - RC)
end

"""
    AL_blk, errL = ACCtoAL_tsqr_slice2d(AC_blk, C, grid)

 Left QR seam using TSQR. `AC_blk` stays block-distributed:
the last chi leg is row-gathered, the tall-skinny QR is reduced over
`col_comm`, and the resulting row block is sliced back to this rank's last-leg
block. The communication adjoints use the existing Slice2D gather wrappers, so
this path is valid in the AD loop as well as the forward observable loop.
"""
function ACCtoAL_tsqr_slice2d(AC_blk, C, grid::Slice2DGrid)
    blocks = map(eachindex(AC_blk.data)) do k
        _acc_to_al_tsqr_one(AC_blk.data[k], C.data[k], grid)
    end
    AL_data = [first(block) for block in blocks]
    errL = sum(last, blocks)
    return StructArray(AL_data, AC_blk.pattern), errL
end

function _acc_to_ar_tslq_one(AC_blk, Cjr, grid::Slice2DGrid)
    chi = ChainRulesCore.ignore_derivatives() do
        MPI.Allreduce(size(AC_blk, 1), +, grid.col_comm)
    end
    a_rs = split_ranges(chi, grid.N1)
    AC_col = slice2d_gather_col(AC_blk, grid, a_rs)
    A_tail_adj_rows = _tail_adjoint_rowblock(AC_col)
    Q_rows, Rqr = _tsqr_front_row_axis(A_tail_adj_rows, grid)
    QAC_col = _qtail_from_qr_rows(Q_rows, AC_col)
    LAC = Rqr'
    LC, QC = lqpos_colrep(Cjr, grid)
    AR_col = reshape(QC' * reshape(QAC_col, size(QAC_col, 1), size(QAC_col, 2) * size(QAC_col, 3) * size(QAC_col, 4)),
                     size(QAC_col))
    AR_blk = AR_col[a_rs[grid.r1 + 1], :, :, :]
    return AR_blk, norm(LAC - LC)
end

function ACCtoAR_tslq_slice2d(AC_blk, C, grid::Slice2DGrid)
    Nj = size(AC_blk, 2)
    blocks = map(eachindex(AC_blk.data)) do p
        i, j = Tuple(findfirst(==(p), AC_blk.pattern))
        jr = mod1(j - 1, Nj)
        _acc_to_ar_tslq_one(AC_blk[i, j], C[i, jr], grid)
    end
    AR_data = [first(block) for block in blocks]
    errR = sum(last, blocks)
    return StructArray(AR_data, AC_blk.pattern), errR
end

function ACCtoALAR_dist_slice2d(AC_blk, C, grid::Slice2DGrid)
    AL_blk, errL = ACCtoAL_tsqr_slice2d(AC_blk, C, grid)
    AR_blk, errR = ACCtoAR_tslq_slice2d(AC_blk, C, grid)
    return AL_blk, AR_blk, errL, errR
end
