# M4: env-level Cannon integration with gather hoisting.
# Design: docs/2026-06-15-m4-env-cannon-integration-design.md
#
# Distributed boundary-environment solvers on block-distributed tensors over a
# square N×N Cannon grid. The FIXED boundary operands (ALu/ALd for leftenv) are
# gathered ONCE outside the power iteration (gather hoisting); the per-power-step
# map is the already-sliced `FLmap_cannon_sliced` (no per-call gather of the fixed
# operands — the iterate is the ring-carried block, never gathered). `simple_eig`
# runs with the distributed `cannon_dot` / `cannon_norm` / `orth_for_ad_cannon`
# hooks so the eigenvalue, normalization and ⊥-projection are GLOBAL across the
# grid (a plain local `dot`/`norm` would use only the per-rank block).
#
# The serial `leftenv` (general.jl) is byte-untouched; this is a parallel path,
# gated by env-level eigenpair parity vs the serial reference (test_cannon_m4.jl).
# Batch A = leftenv only (FLmap template). M4 scope: leg5 single-layer-pair M,
# ifsimple_eig=true, no obs, no mixed precision (asserted at entry).

# Hoisted, checkpoint-friendly per-row eigensolve. Takes the ALREADY-GATHERED
# fixed slices (so checkpoint(eig_checkpoint, Recompute()) re-runs only the local
# chain on the captured slices, and the gather rrules fire once on the OUTER tape).
function _simple_eig_FLmap_cannon(FLij_blk, ALu_row, ALd_col, M_i, grid::CannonGrid;
                                  power_iter, forloop_iter,
                                  segment_checkpoint::CheckpointMethod=Plain())
    Nj = length(ALu_row)
    function f(v)
        for j in 1:Nj
            v = FLmap_cannon_sliced(v, ALu_row[j], ALd_col[j], M_i[j], grid; forloop_iter)
        end
        return v
    end
    return simple_eig(f, FLij_blk; power_iter, segment_checkpoint,
                      inner_product = (x, y) -> cannon_dot(x, y, grid),
                      norm_fn       = x -> cannon_norm(x, grid),
                      orth_fn       = v -> orth_for_ad_cannon(v, grid))
end

"""
    λL, FL_blk = leftenv_cannon(ALu_blk, ALd_blk, M, FL_blk, grid; alg)

Distributed `leftenv` on a square N×N Cannon grid. `ALu_blk`/`ALd_blk`/`FL_blk` are
block-stored `StructArray`s (`cannon_scatter` convention: first χ leg by r1, last by
r2); `M` is the REPLICATED (full) leg5 MPO `StructArray`. Returns the per-cell
eigenvalues `λL` and the block-stored left environment `FL_blk`, same convention as
the input (so it iterates). The gathers of the fixed `ALu`/`ALd` slices are hoisted
once per unit-cell row (outside the power iteration); the per-power-step map is the
sliced `FLmap_cannon_sliced`. Mirrors the serial `leftenv` control flow (pattern
dedup, ir=mod1(i+1,Ni) down-partner, j=2:Nj cycle) — that loop is rank-uniform
because `M.pattern` is replicated, so the collectives never desync.

Preconditions (asserted, fail loud — the deferred features are M5): square grid,
`ifsimple_eig=true`, no mixed precision (`inner_etype`/`whole_vumps_etype`/polish
unset), no obs, leg5 M. Design: docs/2026-06-15-m4-env-cannon-integration-design.md.
"""
function leftenv_cannon(ALu_blk, ALd_blk, M, FL_blk, grid::CannonGrid; alg, ifobs=false, model=nothing)
    @assert grid.N1 == grid.N2 "leftenv_cannon: M3 v1 requires a square grid (N1==N2)"
    @assert alg.ifsimple_eig "leftenv_cannon: requires ifsimple_eig=true (the eigsolve branch bypasses the cannon rrules)"
    @assert alg.inner_etype === nothing && alg.whole_vumps_etype === nothing && alg.simple_eig_polish_steps == 0 "leftenv_cannon: mixed precision is M5 (inner_etype/whole_vumps_etype/simple_eig_polish_steps must be unset)"
    @assert ndims(M.data[1]) == 5 "leftenv_cannon: leg5 single-layer-pair M only (leg4/leg8 are M5)"
    @assert !ifobs && model === nothing "leftenv_cannon: obs/model env is deferred to M5 (R1-M12)"
    @unpack power_iter, forloop_iter, segment_checkpoint, eig_checkpoint = alg

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
        ir = mod1(i + 1, Ni)
        # HOIST (unconditional + rank-uniform): gather the FIXED ALu/ALd slices
        # once for every column of this row, OUTSIDE the eig checkpoint. Immutable
        # Tuples (not a Zygote.Buffer) so Zygote accumulates the slice cotangents
        # across the power-iteration reuses before the single gather pullback.
        ALu_row = ntuple(j -> cannon_gather_row(ALu_blk[i, j],  grid, l_rs), Nj)
        ALd_col = ntuple(j -> cannon_gather_col(ALd_blk[ir, j], grid, a_rs), Nj)
        M_i     = ntuple(j -> M[i, j], Nj)

        p = FL_blk.pattern[i, 1]
        if p ∉ processed_indices
            λLs, FLi1s = checkpoint(eig_checkpoint, _simple_eig_FLmap_cannon,
                                    FL_blk[i, 1], ALu_row, ALd_col, M_i, grid;
                                    power_iter, forloop_iter, segment_checkpoint)
            λL[i, 1], FL′[i, 1] = selectpos(λLs, FLi1s, Nj)
            push!(processed_indices, p)
            length(processed_indices) == length(FL_blk.data) && break
        end
        for j in 2:Nj
            p2 = FL_blk.pattern[i, j]
            if p2 ∉ processed_indices
                FL′[i, j] = FLmap_cannon_sliced(FL′[i, j-1], ALu_row[j-1], ALd_col[j-1], M_i[j-1], grid; forloop_iter)
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
function _simple_eig_FRmap_cannon(FRiNj_blk, ARu_row, ARd_col, M_i, grid::CannonGrid;
                                  power_iter, forloop_iter,
                                  segment_checkpoint::CheckpointMethod=Plain())
    Nj = length(ARu_row)
    function f(v)
        for j in Nj:-1:1
            v = FRmap_cannon_sliced(v, ARu_row[j], ARd_col[j], M_i[j], grid; forloop_iter)
        end
        return v
    end
    return simple_eig(f, FRiNj_blk; power_iter, segment_checkpoint,
                      inner_product = (x, y) -> cannon_dot(x, y, grid),
                      norm_fn       = x -> cannon_norm(x, grid),
                      orth_fn       = v -> orth_for_ad_cannon(v, grid))
end

"""
    λR, FR_blk = rightenv_cannon(ARu_blk, ARd_blk, M, FR_blk, grid; alg)

Distributed `rightenv` on a square N×N Cannon grid (gather hoisting). Block-stored
ARu/ARd/FR; replicated leg5 M. Same preconditions/fail-loud asserts as leftenv_cannon
(square, ifsimple_eig, no mixed precision, no obs, leg5). Design Batch B.
"""
function rightenv_cannon(ARu_blk, ARd_blk, M, FR_blk, grid::CannonGrid; alg, ifobs=false, model=nothing)
    @assert grid.N1 == grid.N2 "rightenv_cannon: square grid (N1==N2)"
    @assert alg.ifsimple_eig "rightenv_cannon: requires ifsimple_eig=true"
    @assert alg.inner_etype === nothing && alg.whole_vumps_etype === nothing && alg.simple_eig_polish_steps == 0 "rightenv_cannon: mixed precision is M5"
    @assert ndims(M.data[1]) == 5 "rightenv_cannon: leg5 single-layer-pair M only"
    @assert !ifobs && model === nothing "rightenv_cannon: obs/model env is deferred to M5 (R1-M12)"
    @unpack power_iter, forloop_iter, segment_checkpoint, eig_checkpoint = alg

    Ni, Nj = size(M)
    χ, p_rs = ChainRulesCore.ignore_derivatives() do
        c = MPI.Allreduce(size(ARu_blk[1, 1], 1), +, grid.col_comm)
        (c, split_ranges(c, grid.N1))
    end
    λR = Zygote.Buffer(randSA(Array, M.pattern))
    FR′ = Zygote.Buffer(FR_blk)
    processed_indices = Set{Int}()
    for i in 1:Ni
        ir = mod1(i + 1, Ni)
        ARu_row = ntuple(j -> cannon_gather_row(ARu_blk[i, j],  grid, p_rs), Nj)   # full d
        ARd_col = ntuple(j -> cannon_gather_col(ARd_blk[ir, j], grid, p_rs), Nj)   # full i
        M_i     = ntuple(j -> M[i, j], Nj)
        p = FR_blk.pattern[i, Nj]
        if p ∉ processed_indices
            λRs, FR1s = checkpoint(eig_checkpoint, _simple_eig_FRmap_cannon,
                                   FR_blk[i, Nj], ARu_row, ARd_col, M_i, grid;
                                   power_iter, forloop_iter, segment_checkpoint)
            λR[i, Nj], FR′[i, Nj] = selectpos(λRs, FR1s, Nj)
            push!(processed_indices, p)
            length(processed_indices) == length(FR_blk.data) && break
        end
        for j in Nj-1:-1:1
            p2 = FR_blk.pattern[i, j]
            if p2 ∉ processed_indices
                FR′[i, j] = FRmap_cannon_sliced(FR′[i, j+1], ARu_row[j+1], ARd_col[j+1], M_i[j+1], grid; forloop_iter)
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
# and the non-leading cells (i = 2:Ni) NORMALIZE — with the GLOBAL `cannon_norm`
# (NOT a local `norm`, which would divide a block by its per-rank norm). Hoisted
# FIXED partners per column: FL_col[i'] (row-gather, full i) + FR_col[i'] (col-
# gather, full d). The ITERATE AC is row-gathered per-call inside the map. NB the
# AC-tensor leg `i` (grid r1 split) is a DIFFERENT index from the cell-row i here.
function _simple_eig_ACmap_cannon(AC1j_blk, FL_col, FR_col, M_j, grid::CannonGrid;
                                  power_iter, forloop_iter,
                                  segment_checkpoint::CheckpointMethod=Plain())
    Ni = length(FL_col)
    function f(v)
        for i in 1:Ni
            v = ACmap_cannon_sliced(v, FL_col[i], FR_col[i], M_j[i], grid; forloop_iter)
        end
        return v
    end
    return simple_eig(f, AC1j_blk; power_iter, segment_checkpoint,
                      inner_product = (x, y) -> cannon_dot(x, y, grid),
                      norm_fn       = x -> cannon_norm(x, grid),
                      orth_fn       = v -> orth_for_ad_cannon(v, grid))
end

"""
    λAC, AC_blk = ACenv_cannon(AC_blk, FL_blk, M, FR_blk, grid; alg)

Distributed `ACenv` on a square N×N Cannon grid (gather hoisting). Block-stored
AC/FL/FR; replicated leg5 M. Same fail-loud asserts as leftenv_cannon. Loops the
unit-cell COLUMN (outer j); the eig chains the cell ROW i; non-leading cells use the
GLOBAL `cannon_norm`. Design Batch C / R1-M13.
"""
function ACenv_cannon(AC_blk, FL_blk, M, FR_blk, grid::CannonGrid; alg)
    @assert grid.N1 == grid.N2 "ACenv_cannon: square grid (N1==N2)"
    @assert alg.ifsimple_eig "ACenv_cannon: requires ifsimple_eig=true"
    @assert alg.inner_etype === nothing && alg.whole_vumps_etype === nothing && alg.simple_eig_polish_steps == 0 "ACenv_cannon: mixed precision is M5"
    @assert ndims(M.data[1]) == 5 "ACenv_cannon: leg5 single-layer-pair M only"
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
        FL_col = ntuple(ip -> cannon_gather_row(FL_blk[ip, j], grid, p_rs), Ni)   # full i
        FR_col = ntuple(ip -> cannon_gather_col(FR_blk[ip, j], grid, p_rs), Ni)   # full d
        M_j    = ntuple(ip -> M[ip, j], Ni)
        p = AC_blk.pattern[1, j]
        if p ∉ processed_indices
            λACs, ACs = checkpoint(eig_checkpoint, _simple_eig_ACmap_cannon,
                                   AC_blk[1, j], FL_col, FR_col, M_j, grid;
                                   power_iter, forloop_iter, segment_checkpoint)
            λAC[1, j], AC′[1, j] = selectpos(λACs, ACs, Ni)
            push!(processed_indices, p)
            length(processed_indices) == length(AC_blk.data) && break
        end
        for i in 2:Ni
            p2 = AC_blk.pattern[i, j]
            if p2 ∉ processed_indices
                ACij = ACmap_cannon_sliced(AC′[i-1, j], FL_col[i-1], FR_col[i-1], M_j[i-1], grid; forloop_iter)
                AC′[i, j] = ACij / cannon_norm(ACij, grid)   # GLOBAL norm (not per-rank local)
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
# with `cannon_gather` (its adjoint is TAKE-MY-BLOCK — correct for a replicated
# downstream; reduce-scatter would over-count by P, R1-M3/§2.4), then run the SERIAL
# `Cmap` kernel on the gathered-full tensors (its chain_apply rrule supplies the AD)
# with PLAIN dot/norm/orth (C replicated → local == global; NO cannon hooks). Mirrors
# serial Cenv (general.jl:640-681): the jr=mod1(j+1,Nj) FL-column offset, the
# i=2:Ni non-leading cells normalized by `norm` (local == global on a replicated C).
# Cmap has no MPO arg and no eig_checkpoint (cheap map — serial Cenv skips it too).
# Grid-agnostic (Cmap works on any grid; the others assert square).
function _simple_eig_Cmap_cannon(C1j, FL_full, FR_full; power_iter,
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
    λC, C = Cenv_cannon(C, FL_blk, FR_blk, grid; alg)

Distributed `Cenv`. `C` is the REPLICATED (full χ×χ) center matrix; FL/FR are
block-stored and gathered to FULL once per column (gather hoisting; take-my-block
adjoint). Output `C` is full χ×χ replicated. Plain dot/norm (C replicated). Design
Batch D / §2.4.
"""
function Cenv_cannon(C, FL_blk, FR_blk, grid::CannonGrid; alg)
    @assert alg.ifsimple_eig "Cenv_cannon: requires ifsimple_eig=true"
    @unpack power_iter, segment_checkpoint = alg
    Ni, Nj = size(C)
    λC = Zygote.Buffer(randSA(Array, C.pattern))
    C′ = Zygote.Buffer(C)
    processed_indices = Set{Int}()
    for j in 1:Nj
        jr = mod1(j + 1, Nj)
        # HOIST: gather FL[:,jr] / FR[:,j] to FULL (replicated; take-my-block adjoint).
        FL_full = ntuple(ip -> cannon_gather(FL_blk[ip, jr], grid), Ni)
        FR_full = ntuple(ip -> cannon_gather(FR_blk[ip, j],  grid), Ni)
        p = C.pattern[1, j]
        if p ∉ processed_indices
            λCs, Cs = _simple_eig_Cmap_cannon(C[1, j], FL_full, FR_full; power_iter, segment_checkpoint)
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
