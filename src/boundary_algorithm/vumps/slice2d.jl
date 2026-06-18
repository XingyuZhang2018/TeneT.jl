# M4: env-level Slice2D integration with gather hoisting.
# Design: docs/2026-06-15-m4-env-slice2d-integration-design.md
#
# Distributed boundary-environment solvers on block-distributed tensors over a
# square N×N Slice2D grid. The FIXED boundary operands (ALu/ALd for leftenv) are
# gathered ONCE outside the power iteration (gather hoisting); the per-power-step
# map is the already-sliced `FLmap_slice2d_sliced` (no per-call gather of the fixed
# operands; the iterate is row-gathered once through the NCCL-capable gather wrapper).
# `simple_eig`
# runs with the distributed `slice2d_dot` / `slice2d_norm` / `orth_for_ad_slice2d`
# hooks so the eigenvalue, normalization and ⊥-projection are GLOBAL across the
# grid (a plain local `dot`/`norm` would use only the per-rank block).
#
# The serial `leftenv` (general.jl) is byte-untouched; this is a parallel path,
# gated by env-level eigenpair parity vs the serial reference (test_slice2d_m4.jl).
# Batch A = leftenv only (FLmap template). M4 scope: leg5 single-layer-pair M,
# ifsimple_eig=true, no obs, no mixed precision (asserted at entry).

# Hoisted, checkpoint-friendly per-row eigensolve. Takes the ALREADY-GATHERED
# fixed slices (so checkpoint(eig_checkpoint, Recompute()) re-runs only the local
# chain on the captured slices, and the gather rrules fire once on the OUTER tape).
function _simple_eig_FLmap_slice2d(FLij_blk, ALu_row, ALd_col, M_i, grid::Slice2DGrid;
                                  power_iter, forloop_iter,
                                  segment_checkpoint::CheckpointMethod=Plain())
    Nj = length(ALu_row)
    function f(v)
        for j in 1:Nj
            v = FLmap_slice2d_sliced(v, ALu_row[j], ALd_col[j], M_i[j], grid; forloop_iter)
        end
        return v
    end
    return simple_eig(f, FLij_blk; power_iter, segment_checkpoint,
                      inner_product = (x, y) -> slice2d_dot(x, y, grid),
                      norm_fn       = x -> slice2d_norm(x, grid),
                      orth_fn       = v -> orth_for_ad_slice2d(v, grid))
end

"""
    λL, FL_blk = leftenv_slice2d(ALu_blk, ALd_blk, M, FL_blk, grid; alg)

Distributed `leftenv` on a square N×N Slice2D grid. `ALu_blk`/`ALd_blk`/`FL_blk` are
block-stored `StructArray`s (`slice2d_scatter` convention: first χ leg by r1, last by
r2); `M` is the REPLICATED (full) leg5 MPO `StructArray`. Returns the per-cell
eigenvalues `λL` and the block-stored left environment `FL_blk`, same convention as
the input (so it iterates). The gathers of the fixed `ALu`/`ALd` slices are hoisted
once per unit-cell row (outside the power iteration); the per-power-step map is the
sliced `FLmap_slice2d_sliced`. Mirrors the serial `leftenv` control flow (pattern
dedup, ir=mod1(i+1,Ni) down-partner, j=2:Nj cycle) — that loop is rank-uniform
because `M.pattern` is replicated, so the collectives never desync.

Preconditions (asserted, fail loud — the deferred features are M5): square grid,
`ifsimple_eig=true`, no mixed precision (`inner_etype`/`whole_vumps_etype`/polish
unset), no obs, leg5 M. Design: docs/2026-06-15-m4-env-slice2d-integration-design.md.
"""
function leftenv_slice2d(ALu_blk, ALd_blk, M, FL_blk, grid::Slice2DGrid; alg, ifobs=false, model=nothing)
    @assert grid.N1 == grid.N2 "leftenv_slice2d: M3 v1 requires a square grid (N1==N2)"
    @assert alg.ifsimple_eig "leftenv_slice2d: requires ifsimple_eig=true (the eigsolve branch bypasses the slice2d rrules)"
    @assert alg.inner_etype === nothing && alg.whole_vumps_etype === nothing && alg.simple_eig_polish_steps == 0 "leftenv_slice2d: mixed precision is M5 (inner_etype/whole_vumps_etype/simple_eig_polish_steps must be unset)"
    @assert ndims(M.data[1]) == 5 "leftenv_slice2d: leg5 single-layer-pair M only (leg4/leg8 are M5)"
    @unpack power_iter, forloop_iter, segment_checkpoint, eig_checkpoint = alg
    power_iter = ifobs ? alg.power_iter_obs : power_iter   # obs (mixed) env uses the larger obs power iter

    Ni, Nj = size(M)
    # χ is the GLOBAL bond dim (recovered from the local a-block over col_comm);
    # the same for every cell, so recover it once. Rank-uniform. The recovery is
    # a non-differentiable integer/size computation — hide its MPI.Allreduce
    # foreigncall from Zygote (it has no rrule and is not on the value path).
    χ, a_rs, l_rs = ChainRulesCore.ignore_derivatives() do
        c = MPI.Allreduce(size(ALu_blk[1, 1], 1), +, grid.col_comm)
        (c, split_ranges(c, grid.N1), split_ranges(c, grid.N2))
    end

    λL = Zygote.Buffer(randSA(Array, M.pattern))
    FL′ = Zygote.Buffer(FL_blk)
    processed_indices = Set{Int}()
    for i in 1:Ni
        # ifobs (obs/mixed env): down-row partner is Ni+1-i (or model's obs_index),
        # AND the caller passes ALu=ALd=AL (no conj) — same FLmap kernel, just the partner.
        ir = ifobs ? (model === nothing ? Ni + 1 - i : obs_index(typeof(model), i, Ni)) : mod1(i + 1, Ni)
        # HOIST (unconditional + rank-uniform): gather the FIXED ALu/ALd slices
        # once for every column of this row, OUTSIDE the eig checkpoint. Immutable
        # Tuples (not a Zygote.Buffer) so Zygote accumulates the slice cotangents
        # across the power-iteration reuses before the single gather pullback.
        ALu_row = ntuple(j -> slice2d_gather_row(ALu_blk[i, j],  grid, l_rs), Nj)
        ALd_col = ntuple(j -> slice2d_gather_col(ALd_blk[ir, j], grid, a_rs), Nj)
        M_i     = ntuple(j -> M[i, j], Nj)

        p = FL_blk.pattern[i, 1]
        if p ∉ processed_indices
            λLs, FLi1s = checkpoint(eig_checkpoint, _simple_eig_FLmap_slice2d,
                                    FL_blk[i, 1], ALu_row, ALd_col, M_i, grid;
                                    power_iter, forloop_iter, segment_checkpoint)
            λL[i, 1], FL′[i, 1] = selectpos(λLs, FLi1s, Nj)
            push!(processed_indices, p)
            length(processed_indices) == length(FL_blk.data) && break
        end
        for j in 2:Nj
            p2 = FL_blk.pattern[i, j]
            if p2 ∉ processed_indices
                FL′[i, j] = FLmap_slice2d_sliced(FL′[i, j-1], ALu_row[j-1], ALd_col[j-1], M_i[j-1], grid; forloop_iter)
                λL[i, j] = λL[i, 1]
                push!(processed_indices, p2)
                length(processed_indices) == length(FL_blk.data) && break
            end
        end
    end
    return copy(λL), copy(FL′)
end

# ── Batch B: rightenv (FRmap) ────────────────────────────────────────────────
# Mirrors serial rightenv (general.jl:355-421): leading cell at COLUMN Nj, the
# inner cycle runs j = Nj-1:-1:1, and FRmap sweeps the row BACKWARD (j = Nj:-1:1).
# Hoisted FIXED partners per row: ARu_row[j] (row-gather, full d) + ARd_col[j]
# (col-gather, full i). The ITERATE FR is col-gathered per-call inside the map.
function _simple_eig_FRmap_slice2d(FRiNj_blk, ARu_row, ARd_col, M_i, grid::Slice2DGrid;
                                  power_iter, forloop_iter,
                                  segment_checkpoint::CheckpointMethod=Plain())
    Nj = length(ARu_row)
    function f(v)
        for j in Nj:-1:1
            v = FRmap_slice2d_sliced(v, ARu_row[j], ARd_col[j], M_i[j], grid; forloop_iter)
        end
        return v
    end
    return simple_eig(f, FRiNj_blk; power_iter, segment_checkpoint,
                      inner_product = (x, y) -> slice2d_dot(x, y, grid),
                      norm_fn       = x -> slice2d_norm(x, grid),
                      orth_fn       = v -> orth_for_ad_slice2d(v, grid))
end

"""
    λR, FR_blk = rightenv_slice2d(ARu_blk, ARd_blk, M, FR_blk, grid; alg)

Distributed `rightenv` on a square N×N Slice2D grid (gather hoisting). Block-stored
ARu/ARd/FR; replicated leg5 M. Same preconditions/fail-loud asserts as leftenv_slice2d
(square, ifsimple_eig, no mixed precision, no obs, leg5). Design Batch B.
"""
function rightenv_slice2d(ARu_blk, ARd_blk, M, FR_blk, grid::Slice2DGrid; alg, ifobs=false, model=nothing)
    @assert grid.N1 == grid.N2 "rightenv_slice2d: square grid (N1==N2)"
    @assert alg.ifsimple_eig "rightenv_slice2d: requires ifsimple_eig=true"
    @assert alg.inner_etype === nothing && alg.whole_vumps_etype === nothing && alg.simple_eig_polish_steps == 0 "rightenv_slice2d: mixed precision is M5"
    @assert ndims(M.data[1]) == 5 "rightenv_slice2d: leg5 single-layer-pair M only"
    @unpack power_iter, forloop_iter, segment_checkpoint, eig_checkpoint = alg
    power_iter = ifobs ? alg.power_iter_obs : power_iter   # obs (mixed) env uses the larger obs power iter

    Ni, Nj = size(M)
    χ, p_rs = ChainRulesCore.ignore_derivatives() do
        c = MPI.Allreduce(size(ARu_blk[1, 1], 1), +, grid.col_comm)
        (c, split_ranges(c, grid.N1))
    end
    λR = Zygote.Buffer(randSA(Array, M.pattern))
    FR′ = Zygote.Buffer(FR_blk)
    processed_indices = Set{Int}()
    for i in 1:Ni
        ir = ifobs ? (model === nothing ? Ni + 1 - i : obs_index(typeof(model), i, Ni)) : mod1(i + 1, Ni)
        ARu_row = ntuple(j -> slice2d_gather_row(ARu_blk[i, j],  grid, p_rs), Nj)   # full d
        ARd_col = ntuple(j -> slice2d_gather_col(ARd_blk[ir, j], grid, p_rs), Nj)   # full i
        M_i     = ntuple(j -> M[i, j], Nj)
        p = FR_blk.pattern[i, Nj]
        if p ∉ processed_indices
            λRs, FR1s = checkpoint(eig_checkpoint, _simple_eig_FRmap_slice2d,
                                   FR_blk[i, Nj], ARu_row, ARd_col, M_i, grid;
                                   power_iter, forloop_iter, segment_checkpoint)
            λR[i, Nj], FR′[i, Nj] = selectpos(λRs, FR1s, Nj)
            push!(processed_indices, p)
            length(processed_indices) == length(FR_blk.data) && break
        end
        for j in Nj-1:-1:1
            p2 = FR_blk.pattern[i, j]
            if p2 ∉ processed_indices
                FR′[i, j] = FRmap_slice2d_sliced(FR′[i, j+1], ARu_row[j+1], ARd_col[j+1], M_i[j+1], grid; forloop_iter)
                λR[i, j] = λR[i, Nj]
                push!(processed_indices, p2)
                length(processed_indices) == length(FR_blk.data) && break
            end
        end
    end
    return copy(λR), copy(FR′)
end

# ── Batch C: ACenv (ACmap) — AXIS-TRANSPOSED vs leftenv ──────────────────────
# Mirrors serial ACenv (general.jl:562-623): OUTER loop over the unit-cell COLUMN
# j, the eig chains over the cell ROW i (ACmap sweeps i = 1:Ni), selectpos uses Ni,
# and the non-leading cells (i = 2:Ni) NORMALIZE — with the GLOBAL `slice2d_norm`
# (NOT a local `norm`, which would divide a block by its per-rank norm). Hoisted
# FIXED partners per column: FL_col[i'] (row-gather, full i) + FR_col[i'] (col-
# gather, full d). The ITERATE AC is row-gathered per-call inside the map. NB the
# AC-tensor leg `i` (grid r1 split) is a DIFFERENT index from the cell-row i here.
function _simple_eig_ACmap_slice2d(AC1j_blk, FL_col, FR_col, M_j, grid::Slice2DGrid;
                                  power_iter, forloop_iter,
                                  segment_checkpoint::CheckpointMethod=Plain())
    Ni = length(FL_col)
    function f(v)
        for i in 1:Ni
            v = ACmap_slice2d_sliced(v, FL_col[i], FR_col[i], M_j[i], grid; forloop_iter)
        end
        return v
    end
    return simple_eig(f, AC1j_blk; power_iter, segment_checkpoint,
                      inner_product = (x, y) -> slice2d_dot(x, y, grid),
                      norm_fn       = x -> slice2d_norm(x, grid),
                      orth_fn       = v -> orth_for_ad_slice2d(v, grid))
end

"""
    λAC, AC_blk = ACenv_slice2d(AC_blk, FL_blk, M, FR_blk, grid; alg)

Distributed `ACenv` on a square N×N Slice2D grid (gather hoisting). Block-stored
AC/FL/FR; replicated leg5 M. Same fail-loud asserts as leftenv_slice2d. Loops the
unit-cell COLUMN (outer j); the eig chains the cell ROW i; non-leading cells use the
GLOBAL `slice2d_norm`. Design Batch C / R1-M13.
"""
function ACenv_slice2d(AC_blk, FL_blk, M, FR_blk, grid::Slice2DGrid; alg)
    @assert grid.N1 == grid.N2 "ACenv_slice2d: square grid (N1==N2)"
    @assert alg.ifsimple_eig "ACenv_slice2d: requires ifsimple_eig=true"
    @assert alg.inner_etype === nothing && alg.whole_vumps_etype === nothing && alg.simple_eig_polish_steps == 0 "ACenv_slice2d: mixed precision is M5"
    @assert ndims(M.data[1]) == 5 "ACenv_slice2d: leg5 single-layer-pair M only"
    @unpack power_iter, forloop_iter, segment_checkpoint, eig_checkpoint = alg

    Ni, Nj = size(M)
    χ, p_rs = ChainRulesCore.ignore_derivatives() do
        c = MPI.Allreduce(size(AC_blk[1, 1], 1), +, grid.col_comm)
        (c, split_ranges(c, grid.N1))
    end
    λAC = Zygote.Buffer(randSA(Array, M.pattern))
    AC′ = Zygote.Buffer(AC_blk)
    processed_indices = Set{Int}()
    for j in 1:Nj
        FL_col = ntuple(ip -> slice2d_gather_row(FL_blk[ip, j], grid, p_rs), Ni)   # full i
        FR_col = ntuple(ip -> slice2d_gather_col(FR_blk[ip, j], grid, p_rs), Ni)   # full d
        M_j    = ntuple(ip -> M[ip, j], Ni)
        p = AC_blk.pattern[1, j]
        if p ∉ processed_indices
            λACs, ACs = checkpoint(eig_checkpoint, _simple_eig_ACmap_slice2d,
                                   AC_blk[1, j], FL_col, FR_col, M_j, grid;
                                   power_iter, forloop_iter, segment_checkpoint)
            λAC[1, j], AC′[1, j] = selectpos(λACs, ACs, Ni)
            push!(processed_indices, p)
            length(processed_indices) == length(AC_blk.data) && break
        end
        for i in 2:Ni
            p2 = AC_blk.pattern[i, j]
            if p2 ∉ processed_indices
                ACij = ACmap_slice2d_sliced(AC′[i-1, j], FL_col[i-1], FR_col[i-1], M_j[i-1], grid; forloop_iter)
                AC′[i, j] = ACij / slice2d_norm(ACij, grid)   # GLOBAL norm (not per-rank local)
                λAC[i, j] = λAC[1, j]
                push!(processed_indices, p2)
                length(processed_indices) == length(AC_blk.data) && break
            end
        end
    end
    return copy(λAC), copy(AC′)
end

# ── Batch D: Cenv (Cmap) — the REPLICATED-OUTPUT case ────────────────────────
# Cmap's output (and the iterate C) is the FULL χ×χ replicated tensor, so this is
# the simplest batch: NO sliced map, NO new rrule. We hoist the FIXED FL/FR to FULL
# with `slice2d_gather` (its adjoint is TAKE-MY-BLOCK — correct for a replicated
# downstream; reduce-scatter would over-count by P, R1-M3/§2.4), then run the SERIAL
# `Cmap` kernel on the gathered-full tensors (its chain_apply rrule supplies the AD)
# with PLAIN dot/norm/orth (C replicated → local == global; NO slice2d hooks). Mirrors
# serial Cenv (general.jl:640-681): the jr=mod1(j+1,Nj) FL-column offset, the
# i=2:Ni non-leading cells normalized by `norm` (local == global on a replicated C).
# Cmap has no MPO arg and no eig_checkpoint (cheap map — serial Cenv skips it too).
# Grid-agnostic (Cmap works on any grid; the others assert square).
function _simple_eig_Cmap_slice2d(C1j, FL_full, FR_full; power_iter,
                                 segment_checkpoint::CheckpointMethod=Plain())
    Ni = length(FL_full)
    function f(v)
        for i in 1:Ni
            v = Cmap(v, FL_full[i], FR_full[i])      # SERIAL kernel on full tensors
        end
        return v
    end
    return simple_eig(f, C1j; power_iter, segment_checkpoint)   # plain dot/norm/orth (C replicated)
end

"""
    λC, C = Cenv_slice2d(C, FL_blk, FR_blk, grid; alg)

Distributed `Cenv`. `C` is the REPLICATED (full χ×χ) center matrix; FL/FR are
block-stored and gathered to FULL once per column (gather hoisting; take-my-block
adjoint). Output `C` is full χ×χ replicated. Plain dot/norm (C replicated). Design
Batch D / §2.4.
"""
function Cenv_slice2d(C, FL_blk, FR_blk, grid::Slice2DGrid; alg)
    @assert alg.ifsimple_eig "Cenv_slice2d: requires ifsimple_eig=true"
    @unpack power_iter, segment_checkpoint = alg
    Ni, Nj = size(C)
    λC = Zygote.Buffer(randSA(Array, C.pattern))
    C′ = Zygote.Buffer(C)
    processed_indices = Set{Int}()
    for j in 1:Nj
        jr = mod1(j + 1, Nj)
        # HOIST: gather FL[:,jr] / FR[:,j] to FULL (replicated; take-my-block adjoint).
        FL_full = ntuple(ip -> slice2d_gather(FL_blk[ip, jr], grid), Ni)
        FR_full = ntuple(ip -> slice2d_gather(FR_blk[ip, j],  grid), Ni)
        p = C.pattern[1, j]
        if p ∉ processed_indices
            λCs, Cs = _simple_eig_Cmap_slice2d(C[1, j], FL_full, FR_full; power_iter, segment_checkpoint)
            λC[1, j], C′[1, j] = selectpos(λCs, Cs, Ni)
            push!(processed_indices, p)
            length(processed_indices) == length(C.data) && break
        end
        for i in 2:Ni
            p2 = C.pattern[i, j]
            if p2 ∉ processed_indices
                Cij = Cmap(C′[i-1, j], FL_full[i-1], FR_full[i-1])
                C′[i, j] = Cij / norm(Cij)            # local norm OK — C replicated (local == global)
                λC[i, j] = λC[1, j]
                push!(processed_indices, p2)
                length(processed_indices) == length(C.data) && break
            end
        end
    end
    return copy(λC), copy(C′)
end

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
    rt′, err = vumps_step_slice2d(rt, M, grid, alg)

One distributed VUMPS step on a square N×N Slice2D grid. Mirrors the serial `vumps_step`
(general.jl:873) call-for-call — old AL/AR into the env updates, a single AC/C solve (NOT
the `vumps_step_power` re-solve variant) — replacing the four solvers with their `_slice2d`
analogs and the two endpoints with the gather seams. AL/AR/FL/FR are block-stored; C is
replicated full χ×χ. Same checkpoint wrapping as serial (leftenv/rightenv/ACenv/ACCtoALAR
under `subop_checkpoint`; ALCtoAC and Cenv unwrapped). Square-grid/leg5/ifsimple_eig/no-
mixed-precision asserts fire inside the `_slice2d` solvers.
"""
function vumps_step_slice2d(rt::VUMPSRuntime, M::StructArray, grid::Slice2DGrid, alg::VUMPS{General})
    # The slice2d env solvers / seam maps do NOT thread inner_checkpoint into their per-map
    # calls (Cmap/FLmap/etc.), unlike serial Cenv/leftenv. It is a no-op at the default
    # Plain(), but a non-Plain inner_checkpoint would silently diverge from serial — fail loud.
    @assert alg.inner_checkpoint isa Plain "vumps_step_slice2d: inner_checkpoint other than Plain() is not supported on the slice2d path (the slice2d maps don't thread it); got $(alg.inner_checkpoint)"
    @unpack AL, C, AR, FL, FR = rt
    sub = alg.subop_checkpoint
    AC = ALCtoAC_slice2d(AL, C, grid)
    _, FL = checkpoint(sub, (a, b, m, fl) -> leftenv_slice2d(a, b, m, fl, grid; alg), AL, conj(AL), M, FL)
    _, FR = checkpoint(sub, (a, b, m, fr) -> rightenv_slice2d(a, b, m, fr, grid; alg), AR, conj(AR), M, FR)
    _, AC = checkpoint(sub, (ac, fl, m, fr) -> ACenv_slice2d(ac, fl, m, fr, grid; alg), AC, FL, M, FR)
    _, C  = Cenv_slice2d(C, FL, FR, grid; alg)          # C replicated χ×χ — no checkpoint (mirrors serial)
    AL, AR, errL, errR = checkpoint(sub, (ac, c) -> ACCtoALAR_dist_slice2d(ac, c, grid), AC, C)
    err = errL + errR
    alg.verbosity >= 4 && err > 1e-8 && println("errL=$errL, errR=$errR")
    C = for_gc(C)
    return VUMPSRuntime(AL, AR, C, FL, FR), err
end

# Fixed seed for the distributed init: every rank seeds identically so initial_A / FLint / FRint
# draw the SAME full tensors → the scattered blocks are consistent WITHOUT broadcasting full-χ
# tensors. (The old bcast_struct used MPI.bcast, which serializes each tensor to a >2^31-byte
# buffer and overflows MPI's Cint count at χ≳700 — e.g. 2.42 GB tensors at χ768 D16 → hang.)
const _SLICE2D_INIT_SEED = 1234567

"""
    rt = init_VUMPSRuntime_slice2d(M, χ, grid, alg)

Build a block-distributed initial runtime — each rank PERSISTS only its χ-block. Canonicalization
(`left_canonical`/`right_canonical`) is irreducibly full-χ (the canonical form AL†AL=I is a global
property), so it runs redundantly per rank on a full A built from a SHARED seed (cheap, ~GB
transient, freed before the solve). The initial FL/FR are then solved **block-distributed** via
`leftenv_slice2d`/`rightenv_slice2d` — NOT a serial full-χ `leftenv`/`rightenv` (that was ~108 GB/GPU
at χ768 D16 and forced the full-χ bcast that overflowed MPI's 2^31 count). C stays replicated.
The shared seed (R1-F1 consistency) replaces the bcast: same seed → same full A/FL → consistent
blocks. RNG state is saved & restored so the seed is invisible to the caller.
"""
function init_VUMPSRuntime_slice2d(M::StructArray, χ::Int, grid::Slice2DGrid, alg::VUMPS{General})
    rng_bak = copy(Random.default_rng()); Random.seed!(_SLICE2D_INIT_SEED)
    A = initial_A(M, χ)
    AL, L, _ = left_canonical(A)
    R, AR, _ = right_canonical(AL)
    C  = LRtoC(L, R)
    FL0, FR0 = FLint(AL, M), FRint(AR, M)
    copy!(Random.default_rng(), rng_bak)
    AL_blk, AR_blk = scatter_struct(AL, grid), scatter_struct(AR, grid)
    _, FL_blk = leftenv_slice2d(AL_blk, conj(AL_blk), M, scatter_struct(FL0, grid), grid; alg)
    _, FR_blk = rightenv_slice2d(AR_blk, conj(AR_blk), M, scatter_struct(FR0, grid), grid; alg)
    return VUMPSRuntime(AL_blk, AR_blk, C, FL_blk, FR_blk)
end

# ═══════════════════════════════════════════════════════════════════════════════
# M5-plaq: Plaquette-mode slice2d (vumps_step_slice2d for VUMPS{<:Plaquette}).
# Plaquette VUMPS is a SIMPLER mirror of General: no AR / rightenv (left-canonical
# only), and ACenv/Cenv use FL on BOTH transfer sides instead of FL+FR. The serial
# Plaquette path (plaquette.jl) calls the SAME ACmap(AC,FL,FR,M) / Cmap(C,FL,FR)
# kernels as General, just passing a second FL slice into the FR slot. So the slice2d
# versions REUSE the General slice2d machinery wholesale (gather/scatter seam,
# ACmap_slice2d_sliced, _simple_eig_{AC,C}map_slice2d, leftenv_slice2d, ALCtoAC_slice2d,
# slice2d_dot/norm rrules) — the only changes: FR-operand sourced from FL[:,jr], and the
# AR/rightenv half dropped (ACCtoAL not ACCtoALAR). AD-correctness is identical to M5.
# ═══════════════════════════════════════════════════════════════════════════════

# jr column of the FR-slot FL slice (copied from serial ACenv_plaq/Cenv_plaq:58-64).
function _plaq_jr(::Type{L}, j::Int, Nj::Int) where {L <: Plaquette}
    if L <: Plaquette{Square}
        return mod1(j + 1, Nj)
    elseif L <: Plaquette{Honeycomb{:brickwall_h}}
        return mod1(Nj - j, Nj)
    else
        error("Plaquette slice2d: unsupported lattice $L (only Square / Honeycomb brickwall_h)")
    end
end

"""
    λAC, AC_blk = ACenv_plaq_slice2d(AC_blk, FL_blk, M, grid; alg)

Distributed plaquette `ACenv_plaq`. FL on BOTH sides: FL[:,j] (row-gather, "FL" slot)
and FL[:,jr] (col-gather, "FR" slot). Reuses `_simple_eig_ACmap_slice2d` /
`ACmap_slice2d_sliced` verbatim; mirrors ACenv_slice2d (non-leading cells use GLOBAL slice2d_norm).
"""
function ACenv_plaq_slice2d(AC_blk, FL_blk, M, grid::Slice2DGrid; alg::VUMPS{L}) where {L <: Plaquette}
    @assert grid.N1 == grid.N2 "ACenv_plaq_slice2d: square grid (N1==N2)"
    @assert alg.ifsimple_eig "ACenv_plaq_slice2d: requires ifsimple_eig=true"
    @assert alg.inner_etype === nothing && alg.whole_vumps_etype === nothing && alg.simple_eig_polish_steps == 0 "ACenv_plaq_slice2d: mixed precision is not supported on the slice2d path"
    @assert ndims(M.data[1]) == 5 "ACenv_plaq_slice2d: leg5 single-layer-pair M only"
    @unpack power_iter, forloop_iter, segment_checkpoint, eig_checkpoint = alg

    Ni, Nj = size(M)
    χ, p_rs = ChainRulesCore.ignore_derivatives() do
        c = MPI.Allreduce(size(AC_blk[1, 1], 1), +, grid.col_comm)
        (c, split_ranges(c, grid.N1))
    end
    λAC = Zygote.Buffer(randSA(Array, M.pattern))
    AC′ = Zygote.Buffer(AC_blk)
    processed_indices = Set{Int}()
    for j in 1:Nj
        jr = _plaq_jr(L, j, Nj)
        FLj_col  = ntuple(ip -> slice2d_gather_row(FL_blk[ip, j],  grid, p_rs), Ni)   # FL slot (full i)
        FLjr_col = ntuple(ip -> slice2d_gather_col(FL_blk[ip, jr], grid, p_rs), Ni)   # FR slot (full d)
        M_j      = ntuple(ip -> M[ip, j], Ni)
        p = AC_blk.pattern[1, j]
        if p ∉ processed_indices
            λACs, ACs = checkpoint(eig_checkpoint, _simple_eig_ACmap_slice2d,
                                   AC_blk[1, j], FLj_col, FLjr_col, M_j, grid;
                                   power_iter, forloop_iter, segment_checkpoint)
            λAC[1, j], AC′[1, j] = selectpos(λACs, ACs, Ni)
            push!(processed_indices, p)
            length(processed_indices) == length(AC_blk.data) && break
        end
        for i in 2:Ni
            p2 = AC_blk.pattern[i, j]
            if p2 ∉ processed_indices
                ACij = ACmap_slice2d_sliced(AC′[i-1, j], FLj_col[i-1], FLjr_col[i-1], M_j[i-1], grid; forloop_iter)
                AC′[i, j] = ACij / slice2d_norm(ACij, grid)
                λAC[i, j] = λAC[1, j]
                push!(processed_indices, p2)
                length(processed_indices) == length(AC_blk.data) && break
            end
        end
    end
    return copy(λAC), copy(AC′)
end

"""
    λC, C = Cenv_plaq_slice2d(C, FL_blk, grid; alg)

Distributed plaquette `Cenv_plaq`: Cmap(C, FL[:,jl], FL[:,jr]). C REPLICATED (gather FL
slices take-my-block, plain norm). Mirrors Cenv_slice2d.
"""
function Cenv_plaq_slice2d(C, FL_blk, grid::Slice2DGrid; alg::VUMPS{L}) where {L <: Plaquette}
    @assert alg.ifsimple_eig "Cenv_plaq_slice2d: requires ifsimple_eig=true"
    @unpack power_iter, segment_checkpoint = alg
    Ni, Nj = size(C)
    λC = Zygote.Buffer(randSA(Array, C.pattern))
    C′ = Zygote.Buffer(C)
    processed_indices = Set{Int}()
    for j in 1:Nj
        jl = mod1(j + 1, Nj)          # FL slot column (serial Cenv_plaq:117, all lattices)
        jr = _plaq_jr(L, j, Nj)       # FR slot column
        FLjl_full = ntuple(ip -> slice2d_gather(FL_blk[ip, jl], grid), Ni)
        FLjr_full = ntuple(ip -> slice2d_gather(FL_blk[ip, jr], grid), Ni)
        p = C.pattern[1, j]
        if p ∉ processed_indices
            λCs, Cs = _simple_eig_Cmap_slice2d(C[1, j], FLjl_full, FLjr_full; power_iter, segment_checkpoint)
            λC[1, j], C′[1, j] = selectpos(λCs, Cs, Ni)
            push!(processed_indices, p)
            length(processed_indices) == length(C.data) && break
        end
        for i in 2:Ni
            p2 = C.pattern[i, j]
            if p2 ∉ processed_indices
                Cij = Cmap(C′[i-1, j], FLjl_full[i-1], FLjr_full[i-1])
                C′[i, j] = Cij / norm(Cij)    # local norm OK — C replicated
                λC[i, j] = λC[1, j]
                push!(processed_indices, p2)
                length(processed_indices) == length(C.data) && break
            end
        end
    end
    return copy(λC), copy(C′)
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

"""
    rt′, err = vumps_step_slice2d(rt::PlaquetteVUMPSRuntime, M, grid, alg)

One distributed plaquette VUMPS step, mirroring serial vumps_step (plaquette.jl:162):
ALCtoAC → leftenv → ACenv_plaq → Cenv_plaq → ACCtoAL (no rightenv/AR). AL/FL block, C replicated.
"""
function vumps_step_slice2d(rt::PlaquetteVUMPSRuntime, M::StructArray, grid::Slice2DGrid, alg::VUMPS{L}) where {L <: Plaquette}
    @assert alg.inner_checkpoint isa Plain "vumps_step_slice2d(Plaquette): inner_checkpoint other than Plain() is not supported on the slice2d path; got $(alg.inner_checkpoint)"
    @unpack AL, C, FL = rt
    sub = alg.subop_checkpoint
    AC = ALCtoAC_slice2d(AL, C, grid)
    _, FL = checkpoint(sub, (a, b, m, fl) -> leftenv_slice2d(a, b, m, fl, grid; alg), AL, conj(AL), M, FL)
    _, AC = checkpoint(sub, (a, fl, m) -> ACenv_plaq_slice2d(a, fl, m, grid; alg), AC, FL, M)
    _, C  = Cenv_plaq_slice2d(C, FL, grid; alg)
    AL, err = checkpoint(sub, (ac, c) -> ACCtoAL_tsqr_slice2d(ac, c, grid), AC, C)
    C = for_gc(C)
    return PlaquetteVUMPSRuntime(AL, C, FL), err
end

"""
    rt = init_VUMPSRuntime_slice2d(M, χ, grid, alg::VUMPS{<:Plaquette})

Block-distributed plaquette init: serial canonicalization (C = LRtoC(L,L), no right),
unconditional bcast over grid.comm, then scatter AL/FL (C replicated).
"""
function init_VUMPSRuntime_slice2d(M::StructArray, χ::Int, grid::Slice2DGrid, alg::VUMPS{L}) where {L <: Plaquette}
    if alg.distributed_qr
        rng_bak = copy(Random.default_rng()); Random.seed!(_SLICE2D_INIT_SEED + grid.rank)
        A_blk = _initial_A_slice2d_block(M, χ, grid)
        AL_blk, Lg = _left_canonical_tsqr_slice2d(A_blk, grid)
        C = LRtoC(Lg, Lg)
        FL0_blk = _initial_FL_slice2d_block(M, χ, grid)
        copy!(Random.default_rng(), rng_bak)
        _, FL_blk = leftenv_slice2d(AL_blk, conj(AL_blk), M, FL0_blk, grid; alg)
        return PlaquetteVUMPSRuntime(AL_blk, C, FL_blk)
    end
    # DISTRIBUTED init (see the General method above): shared-seed cross-rank consistency instead of
    # the overflowing full-χ bcast; left_canonical stays (global, cheap); FL solved block-distributed
    # via leftenv_slice2d (no serial full-χ leftenv / 108 GB). Plaquette is left-canonical only (no AR/FR).
    rng_bak = copy(Random.default_rng()); Random.seed!(_SLICE2D_INIT_SEED)
    A = initial_A(M, χ)
    AL, Lg, _ = left_canonical(A)
    C   = LRtoC(Lg, Lg)
    FL0 = FLint(AL, M)
    copy!(Random.default_rng(), rng_bak)
    AL_blk = scatter_struct(AL, grid)
    _, FL_blk = leftenv_slice2d(AL_blk, conj(AL_blk), M, scatter_struct(FL0, grid), grid; alg)
    return PlaquetteVUMPSRuntime(AL_blk, C, FL_blk)
end

function _initial_A_slice2d_block(M::StructArray, χ::Int, grid::Slice2DGrid)
    a_rs = split_ranges(χ, grid.N1)
    l_rs = split_ranges(χ, grid.N2)
    sizes = [(D = size(m, 4); (length(a_rs[grid.r1 + 1]), D, D, length(l_rs[grid.r2 + 1]))) for m in M.data]
    return randSA(M, sizes)
end

function _initial_FL_slice2d_block(M::StructArray, χ::Int, grid::Slice2DGrid)
    a_rs = split_ranges(χ, grid.N1)
    l_rs = split_ranges(χ, grid.N2)
    sizes = [(D = size(m, 1); (length(a_rs[grid.r1 + 1]), D, D, length(l_rs[grid.r2 + 1]))) for m in M.data]
    return randSA(M, sizes)
end

function _left_canonical_tsqr_slice2d(A_blk::StructArray, grid::Slice2DGrid)
    blocks = map(eachindex(A_blk.data)) do k
        χ = MPI.Allreduce(size(A_blk.data[k], 1), +, grid.col_comm)
        l_rs = split_ranges(χ, grid.N2)
        A_row = slice2d_gather_row(A_blk.data[k], grid, l_rs)
        Q, R = _tsqr_front_rowblock(A_row, grid)
        AL_row = reshape(Q, size(A_row))
        AL_blk = AL_row[ntuple(i -> i == ndims(AL_row) ? l_rs[grid.r2 + 1] : Colon(), ndims(AL_row))...]
        AL_blk, R / norm(R)
    end
    AL_data = [first(block) for block in blocks]
    L_data = [last(block) for block in blocks]
    return StructArray(AL_data, A_blk.pattern), StructArray(L_data, A_blk.pattern)
end

# ═══════════════════════════════════════════════════════════════════════════════
# slice2d ObsEnv: distributed observation environment.
# The obs envs (FLo/FRo) use the SAME FLmap/FRmap kernels as the bulk envs — leftenv/
# rightenv with ifobs=true just change the down-row partner (ir=Ni+1-i) and power_iter
# (power_iter_obs); ALu=ALd=AL (no conj). So leftenv_slice2d/rightenv_slice2d (ifobs flag
# now live) compute FLo/FRo block-distributed. v1 then GATHERS the obs env to full and
# hands it to the serial `energy_value` (energy expectation is not yet distributed — a
# v2 item; energy_value is cheaper than leading_boundary, and the full env is replicated
# after gather so every rank computes the identical scalar). The ObsEnv guard lives in
# general.jl (dispatch on alg.grid). gather_env reuses gather_struct (take-my-block adjoint).
# ═══════════════════════════════════════════════════════════════════════════════
gather_env(env::VUMPSEnv, grid::Slice2DGrid) = VUMPSEnv(
    gather_struct(env.ACu, grid), gather_struct(env.ARu, grid),
    gather_struct(env.ACd, grid), gather_struct(env.ARd, grid),
    gather_struct(env.FLu, grid), gather_struct(env.FRu, grid),
    gather_struct(env.FLo, grid), gather_struct(env.FRo, grid))
gather_env(env::PlaquetteVUMPSEnv, grid::Slice2DGrid) = PlaquetteVUMPSEnv(
    gather_struct(env.AL, grid), env.C,                 # C replicated (not gathered)
    gather_struct(env.FLu, grid), gather_struct(env.FLo, grid))
