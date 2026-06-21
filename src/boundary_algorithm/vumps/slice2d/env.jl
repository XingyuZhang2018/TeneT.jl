# M4: env-level Slice2D integration with gather hoisting.
# Design: docs/2026-06-15-m4-env-slice2d-integration-design.md
#
# Distributed boundary-environment solvers on block-distributed tensors over an
# N1×N2 Slice2D grid. The FIXED boundary operands (ALu/ALd for leftenv) are
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

Distributed `leftenv` on an N1×N2 Slice2D grid. `ALu_blk`/`ALd_blk`/`FL_blk` are
block-stored `StructArray`s (`slice2d_scatter` convention: first χ leg by r1, last by
r2); `M` is the REPLICATED (full) leg5 MPO `StructArray`. Returns the per-cell
eigenvalues `λL` and the block-stored left environment `FL_blk`, same convention as
the input (so it iterates). The gathers of the fixed `ALu`/`ALd` slices are hoisted
once per unit-cell row (outside the power iteration); the per-power-step map is the
sliced `FLmap_slice2d_sliced`. Mirrors the serial `leftenv` control flow (pattern
dedup, ir=mod1(i+1,Ni) down-partner, j=2:Nj cycle) — that loop is rank-uniform
because `M.pattern` is replicated, so the collectives never desync.

Preconditions (asserted, fail loud — the deferred features are M5):
`ifsimple_eig=true`, no mixed precision (`inner_etype`/`whole_vumps_etype`/polish
unset), no obs, leg5 M. Design: docs/2026-06-15-m4-env-slice2d-integration-design.md.
"""
function leftenv_slice2d(ALu_blk, ALd_blk, M, FL_blk, grid::Slice2DGrid; alg, ifobs=false, model=nothing)
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

Distributed `rightenv` on an N1×N2 Slice2D grid (gather hoisting). Block-stored
ARu/ARd/FR; replicated leg5 M. Same preconditions/fail-loud asserts as leftenv_slice2d
(ifsimple_eig, no mixed precision, no obs, leg5). Design Batch B.
"""
function rightenv_slice2d(ARu_blk, ARd_blk, M, FR_blk, grid::Slice2DGrid; alg, ifobs=false, model=nothing)
    @assert alg.ifsimple_eig "rightenv_slice2d: requires ifsimple_eig=true"
    @assert alg.inner_etype === nothing && alg.whole_vumps_etype === nothing && alg.simple_eig_polish_steps == 0 "rightenv_slice2d: mixed precision is M5"
    @assert ndims(M.data[1]) == 5 "rightenv_slice2d: leg5 single-layer-pair M only"
    @unpack power_iter, forloop_iter, segment_checkpoint, eig_checkpoint = alg
    power_iter = ifobs ? alg.power_iter_obs : power_iter   # obs (mixed) env uses the larger obs power iter

    Ni, Nj = size(M)
    χ, a_rs, l_rs = ChainRulesCore.ignore_derivatives() do
        c = MPI.Allreduce(size(ARu_blk[1, 1], 1), +, grid.col_comm)
        (c, split_ranges(c, grid.N1), split_ranges(c, grid.N2))
    end
    λR = Zygote.Buffer(randSA(Array, M.pattern))
    FR′ = Zygote.Buffer(FR_blk)
    processed_indices = Set{Int}()
    for i in 1:Ni
        ir = ifobs ? (model === nothing ? Ni + 1 - i : obs_index(typeof(model), i, Ni)) : mod1(i + 1, Ni)
        ARu_row = ntuple(j -> slice2d_gather_row(ARu_blk[i, j],  grid, l_rs), Nj)   # full d
        ARd_col = ntuple(j -> slice2d_gather_col(ARd_blk[ir, j], grid, a_rs), Nj)   # full i
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

Distributed `ACenv` on an N1×N2 Slice2D grid (gather hoisting). Block-stored
AC/FL/FR; replicated leg5 M. Same fail-loud asserts as leftenv_slice2d. Loops the
unit-cell COLUMN (outer j); the eig chains the cell ROW i; non-leading cells use the
GLOBAL `slice2d_norm`. Design Batch C / R1-M13.
"""
function ACenv_slice2d(AC_blk, FL_blk, M, FR_blk, grid::Slice2DGrid; alg)
    @assert alg.ifsimple_eig "ACenv_slice2d: requires ifsimple_eig=true"
    @assert alg.inner_etype === nothing && alg.whole_vumps_etype === nothing && alg.simple_eig_polish_steps == 0 "ACenv_slice2d: mixed precision is M5"
    @assert ndims(M.data[1]) == 5 "ACenv_slice2d: leg5 single-layer-pair M only"
    @unpack power_iter, forloop_iter, segment_checkpoint, eig_checkpoint = alg

    Ni, Nj = size(M)
    χ, a_rs, l_rs = ChainRulesCore.ignore_derivatives() do
        c = MPI.Allreduce(size(AC_blk[1, 1], 1), +, grid.col_comm)
        (c, split_ranges(c, grid.N1), split_ranges(c, grid.N2))
    end
    λAC = Zygote.Buffer(randSA(Array, M.pattern))
    AC′ = Zygote.Buffer(AC_blk)
    processed_indices = Set{Int}()
    for j in 1:Nj
        FL_col = ntuple(ip -> slice2d_gather_row(FL_blk[ip, j], grid, l_rs), Ni)   # full i
        FR_col = ntuple(ip -> slice2d_gather_col(FR_blk[ip, j], grid, a_rs), Ni)   # full d
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
# Grid-agnostic (Cmap works on any grid; the cross-axis maps also carry separate
# r1/r2 partitions on rectangular grids).
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
    @assert alg.ifsimple_eig "ACenv_plaq_slice2d: requires ifsimple_eig=true"
    @assert alg.inner_etype === nothing && alg.whole_vumps_etype === nothing && alg.simple_eig_polish_steps == 0 "ACenv_plaq_slice2d: mixed precision is not supported on the slice2d path"
    @assert ndims(M.data[1]) == 5 "ACenv_plaq_slice2d: leg5 single-layer-pair M only"
    @unpack power_iter, forloop_iter, segment_checkpoint, eig_checkpoint = alg

    Ni, Nj = size(M)
    χ, a_rs, l_rs = ChainRulesCore.ignore_derivatives() do
        c = MPI.Allreduce(size(AC_blk[1, 1], 1), +, grid.col_comm)
        (c, split_ranges(c, grid.N1), split_ranges(c, grid.N2))
    end
    λAC = Zygote.Buffer(randSA(Array, M.pattern))
    AC′ = Zygote.Buffer(AC_blk)
    processed_indices = Set{Int}()
    for j in 1:Nj
        jr = _plaq_jr(L, j, Nj)
        FLj_col  = ntuple(ip -> slice2d_gather_row(FL_blk[ip, j],  grid, l_rs), Ni)   # FL slot (full i)
        FLjr_col = ntuple(ip -> slice2d_gather_col(FL_blk[ip, jr], grid, a_rs), Ni)   # FR slot (full d)
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
