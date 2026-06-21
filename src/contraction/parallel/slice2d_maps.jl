# ─── Stage kernels (leg5) ─────────────────────────────────────────────────
#
# FLmap splits into row-slice sum + fold + stage 2 so distributed FLOPs stay exactly
# serial/P:
#   row-slice sum (stage 1) contracts the i leg only, accumulating the pre-fold
#     intermediate H. The M fold is deliberately NOT in the ring: its cost is
#     independent of the i-block extent, so folding per step would redo it N2
#     times (overhead growing with grid size);
#   fold contracts M1/M2 into H once per map call;
#   stage 2 contracts a/b/c with ALu, leaving a full-length d leg for the
#     column reduce-scatter.
# Transient accounting: peak is ≈ (1+d)·|H| during the fold (H and the owned
# pairwise intermediate T coexist) — the serial transient peak / P.

# Deterministic device-memory release for chunk transients. Plain @tensor
# never eagerly frees on CuArray (DefaultAllocator's tensorfree! is a no-op),
# so without explicit frees the pool rides at high-water across chunks and
# reactive GC pollutes timings. unsafe_free! is stream-ordered and the
# finalizer skips already-freed arrays.
_free!(x::CuArray) = CUDA.unsafe_free!(x)
_free!(x) = nothing

function _slice2d_stage1(FL, ALd)
    @tensor H[a, e, f, j, k, l] := FL[a, e, f, i] * ALd[i, j, k, l]
    return H
end

# In-place accumulating variant for the row-slice sum (avoids a second H-sized
# temporary). The rrule backward recomputes H with the same kernels and then
# walks the hand adjoints below — no Zygote anywhere in the chunk body.
function _slice2d_stage1_add!(H, FL, ALd)
    @tensor H[a, e, f, j, k, l] += FL[a, e, f, i] * ALd[i, j, k, l]
    return H
end

# Fold split into its two pairwise steps so the d·|H| intermediate T is an
# owned array (freeable eagerly) instead of a dead @tensor-internal temporary
# — TensorOperations' DefaultAllocator never frees on CuArray, and these
# temporaries are the largest single transient (job 1274256 OOM).
function _slice2d_fold1(H, M1)
    @tensor T[a, f, k, g, b, p, l] := H[a, e, f, j, k, l] * M1[e, j, g, b, p]
    return T
end

function _slice2d_fold2(T, M2)
    @tensor G[a, b, c, g, h, l] := T[a, f, k, g, b, p, l] * M2[f, k, h, c, p]
    return G
end

function _slice2d_fold(H, M1, M2)
    T = _slice2d_fold1(H, M1)
    G = _slice2d_fold2(T, M2)
    _free!(T)
    return G
end

function _slice2d_stage2(G, ALu)
    @tensor P[d, g, h, l] := G[a, b, c, g, h, l] * ALu[a, b, c, d]
    return P
end

# Hand-written adjoints of _slice2d_stage1 (H = FL·ALd): two contractions
# instead of Zygote's three (whose pullback re-runs the stage-1 forward only
# to discard it — the cause of the large-χ backward slowdown in job 1265371).
function _slice2d_stage1_dFL(dH, ALd)
    @tensor dFL[a, e, f, i] := dH[a, e, f, j, k, l] * conj(ALd[i, j, k, l])
    return dFL
end

function _slice2d_stage1_dALd(dH, FL)
    @tensor dALd[i, j, k, l] := conj(FL[a, e, f, i]) * dH[a, e, f, j, k, l]
    return dALd
end

# Hand adjoints of stage 2 (P = G·ALu, contracting a,b,c):
function _slice2d_stage2_dG(dP, ALu)
    @tensor dG[a, b, c, g, h, l] := dP[d, g, h, l] * conj(ALu[a, b, c, d])
    return dG
end
function _slice2d_stage2_dALu(dP, G)
    @tensor dALu[a, b, c, d] := conj(G[a, b, c, g, h, l]) * dP[d, g, h, l]
    return dALu
end

# Hand adjoints of fold step 2 (G = T·M2, contracting f,k,p):
function _slice2d_fold2_dT(dG, M2)
    @tensor dT[a, f, k, g, b, p, l] := dG[a, b, c, g, h, l] * conj(M2[f, k, h, c, p])
    return dT
end
function _slice2d_fold2_dM2(dG, T)
    @tensor dM2[f, k, h, c, p] := conj(T[a, f, k, g, b, p, l]) * dG[a, b, c, g, h, l]
    return dM2
end

# Hand adjoints of fold step 1 (T = H·M1, contracting e,j):
function _slice2d_fold1_dH(dT, M1)
    @tensor dH[a, e, f, j, k, l] := dT[a, f, k, g, b, p, l] * conj(M1[e, j, g, b, p])
    return dH
end
function _slice2d_fold1_dM1(dT, H)
    @tensor dM1[e, j, g, b, p] := conj(H[a, e, f, j, k, l]) * dT[a, f, k, g, b, p, l]
    return dM1
end

# ─── Forward ──────────────────────────────────────────────────────────────

# Core shared by the replicated (FLmap_slice2d) and distributed
# (FLmap_slice2d_dist) paths and their rrules. Takes the ALREADY-SLICED per-rank
# AL working set:
#   ALu_row = ALu[a_r1, :, :, :]   (local a block, FULL d leg)
#   ALd_col = ALd[:, :, :, l_r2]   (FULL i leg, local l block)
# Returns (result_blk, FL_row): the FL iterate is row-gathered once into a
# local-a/full-i working slice (χ²D²/N1 per rank); the local l range is then
# processed in `forloop_iter` chunks, each running stage1 →
# fold → stage2 fully locally with transients bounded by (2+d)·|H|/forloop_iter.
function _slice2d_forward_sliced(FL_blk, ALu_row, ALd_col, M1, M2, grid::Slice2DGrid; forloop_iter = 1)
    N1, N2, r1, r2 = grid.N1, grid.N2, grid.r1, grid.r2
    χ = size(ALu_row, 4)              # full d leg of the row slice
    a_rs = split_ranges(χ, N1)
    i_rs = split_ranges(χ, N2)
    nl = size(ALd_col, 4)             # local l extent
    @assert size(FL_blk, 1) == length(a_rs[r1 + 1]) && size(FL_blk, 4) == nl == length(i_rs[r2 + 1]) "slice2d forward: FL block shape $(size(FL_blk)) inconsistent with grid ($(N1)×$(N2)) and χ=$χ"
    @assert forloop_iter ≥ 1 "FLmap_slice2d: forloop_iter must be ≥ 1"

    FL_row = slice2d_gather_row(FL_blk, grid, i_rs)

    # Local pipeline per l-chunk: stage 1 accumulate → fold once → stage 2.
    # Chunk ranges are LOCAL (within the l block): ALd_col's last leg is the
    # local block, so chunks index it directly.
    Dg, Dh = size(M1, 3), size(M2, 3)
    partial = similar(FL_blk, χ, Dg, Dh, nl)
    l_chunks = split_ranges(nl, min(forloop_iter, nl))
    for ch in l_chunks
        ALd_chunk = view(ALd_col, :, :, :, ch)
        Hc = _slice2d_stage1(FL_row, ALd_chunk)
        Gc = _slice2d_fold(Hc, M1, M2)
        _free!(Hc)
        Pc = _slice2d_stage2(Gc, ALu_row)
        _free!(Gc)
        view(partial, :, :, :, ch) .= Pc
        _free!(Pc)
    end
    result = _slice2d_col_reduce_scatter(partial, grid, a_rs)
    return result, FL_row
end

# Replicated-AL path: build the per-rank slices as views (no copies) and run
# the shared core. Signature/behavior identical to the pre-refactor version.
function _slice2d_forward(FL_blk, ALu, ALd, M1, M2, grid::Slice2DGrid; forloop_iter = 1)
    χ = size(ALu, 1)
    a_rs = split_ranges(χ, grid.N1)
    i_rs = split_ranges(χ, grid.N2)
    ALu_row = view(ALu, a_rs[grid.r1 + 1], :, :, :)
    ALd_col = view(ALd, :, :, :, i_rs[grid.r2 + 1])
    return _slice2d_forward_sliced(FL_blk, ALu_row, ALd_col, M1, M2, grid; forloop_iter)
end

"""
    FLmap_slice2d(FL_blk, ALu, ALd, M, grid; forloop_iter=1, inner_etype=nothing) -> result_blk

Distributed FLmap on an N1×N2 Slice2D grid. `FL_blk` and the returned block
follow the convention: first χ leg split N1-ways by r1, last χ leg split
N2-ways by r2. `M` is a leg5 tensor or an `(M1, M2)` tuple; ALu/ALd/M are
replicated on every rank. Collective over `grid.comm`.
`forloop_iter` sub-slices the local l range: forward per-chunk transients are
≈(1+d)·χ²D⁴/(P·forloop_iter); backward ≈(2+2d)·χ²D⁴/(P·forloop_iter) (fully
hand-written adjoint chain, every intermediate freed after its last use) —
size `forloop_iter` by the backward bound when gradients are needed.
See docs/2026-06-10-slice2d-flmap-design.md.
"""
function FLmap_slice2d(FL_blk, ALu, ALd, M, grid::Slice2DGrid; forloop_iter = 1, inner_etype = nothing)
    M1, M2 = M isa Tuple ? M : (M, conj(M))
    T_orig = eltype(FL_blk)
    do_cast = inner_etype !== nothing && inner_etype != real(T_orig)
    if do_cast
        FL_blk = _downcast_eltype(inner_etype, FL_blk)
        ALu = _downcast_eltype(inner_etype, ALu)
        ALd = _downcast_eltype(inner_etype, ALd)
        M1 = _downcast_eltype(inner_etype, M1)
        M2 = _downcast_eltype(inner_etype, M2)
    end
    result, _ = _slice2d_forward(FL_blk, ALu, ALd, M1, M2, grid; forloop_iter)
    return do_cast ? T_orig.(result) : result
end

"""
    FLmap_slice2d_dist(FL_blk, ALu_blk, ALd_blk, M, grid; forloop_iter=1, inner_etype=nothing)

Fully distributed FLmap: FL, ALu, ALd all block-stored (first χ leg by r1,
last by r2; `slice2d_scatter` convention), M replicated. Internally assembles
the row slice ALu[a_r1, :, :, :] and column slice ALd[:, :, :, l_r2] (the
irreducible per-rank working set) and runs the same ring/chunk pipeline as
`FLmap_slice2d`. Collective over `grid.comm`. In an iteration with fixed
ALu/ALd, hoisting the two gathers out of the loop is the natural
optimization (leftenv round).
"""
function FLmap_slice2d_dist(FL_blk, ALu_blk, ALd_blk, M, grid::Slice2DGrid;
                           forloop_iter = 1, inner_etype = nothing)
    M1, M2 = M isa Tuple ? M : (M, conj(M))
    T_orig = eltype(FL_blk)
    do_cast = inner_etype !== nothing && inner_etype != real(T_orig)
    if do_cast
        FL_blk = _downcast_eltype(inner_etype, FL_blk)
        ALu_blk = _downcast_eltype(inner_etype, ALu_blk)
        ALd_blk = _downcast_eltype(inner_etype, ALd_blk)
        M1 = _downcast_eltype(inner_etype, M1)
        M2 = _downcast_eltype(inner_etype, M2)
    end
    χ = MPI.Allreduce(size(ALu_blk, 1), +, grid.col_comm)
    a_rs = split_ranges(χ, grid.N1)
    l_rs = split_ranges(χ, grid.N2)
    ALu_row = _slice2d_row_allgather(ALu_blk, grid, l_rs)
    ALd_col = _slice2d_col_allgather(ALd_blk, grid, a_rs)
    result, _ = _slice2d_forward_sliced(FL_blk, ALu_row, ALd_col, M1, M2, grid; forloop_iter)
    return do_cast ? T_orig.(result) : result
end


"""
    FLmap_slice2d_sliced(FL_blk, ALu_row, ALd_col, M, grid; forloop_iter=1) -> result_blk

Hoisted FLmap: the FIXED boundary slices `ALu_row` (local-a, FULL-d) and `ALd_col`
(FULL-i, local-l) are supplied PRE-GATHERED (gathered once, outside the power
iteration, by `slice2d_gather_row`/`slice2d_gather_col`). The iterate `FL_blk` is
row-gathered once per map call, matching the NCCL-capable gather wrappers. This is the per-power-step
map of `leftenv_slice2d`; its rrule (autodiff/rules.jl) is the `FLmap_slice2d_dist` rrule
MINUS the input gathers (now args) MINUS the type-B fixed-operand reduce-scatters (now
in the gather wrappers' rrules), returning the SLICE gradients `dALu_row`/`dALd_col`.
No `inner_etype` — mixed precision is M5 (the env @asserts it unset). `M` is leg5
single-layer-pair (or an `(M1,M2)` tuple); square grid not required (FLmap is on-axis).
Design: docs/2026-06-15-m4-env-slice2d-integration-design.md §2.2b/§4.
"""
function FLmap_slice2d_sliced(FL_blk, ALu_row, ALd_col, M, grid::Slice2DGrid; forloop_iter = 1)
    M1, M2 = M isa Tuple ? M : (M, conj(M))
    result, _ = _slice2d_forward_sliced(FL_blk, ALu_row, ALd_col, M1, M2, grid; forloop_iter)
    return result
end

"""
    FRmap_slice2d_sliced(FR_blk, ARu_g, ARd_g, M, grid; forloop_iter=1) -> result_blk

Hoisted FRmap (Batch B): the FIXED partners `ARu_g` (a-block, FULL-d; row-gathered)
and `ARd_g` (FULL-i, l-block; col-gathered) are supplied PRE-GATHERED (hoisted once
by `slice2d_gather_row`/`slice2d_gather_col`). The ITERATE `FR_blk` is col-gathered to
full-d INTERNALLY per call (FR is not fixed across the power iteration). Per-step map
of `rightenv_slice2d`. rrule (rules.jl) = the FRmap_slice2d_dist rrule minus the fixed
ARu/ARd gathers + their reduce-scatters (→ slice grads), keeping the iterate FR
col-gather + its col_reduce_scatter adjoint. Supports N1×N2 grids via separate
r1/r2 partitions. See §2.2b/§4.
"""
function FRmap_slice2d_sliced(FR_blk, ARu_g, ARd_g, M, grid::Slice2DGrid; forloop_iter = 1)
    M1, M2 = M isa Tuple ? M : (M, conj(M))
    χ = size(ARd_g, 1)                                  # full i (ARd_g = [i-full, j, k, l-block])
    a_rs = split_ranges(χ, grid.N1)
    l_rs = split_ranges(χ, grid.N2)
    FR_g = _slice2d_col_allgather(FR_blk, grid, a_rs)    # iterate gather (full d), per-call
    result, _, _, _ = _frmap_slice2d_forward_sliced(ARd_g, FR_g, ARu_g, M1, M2, grid, a_rs, l_rs; forloop_iter)
    return result
end

function Rmap_slice2d_sliced(R_blk, ARu_g, ARd_g, grid::Slice2DGrid)
    χ = size(ARd_g, 1)
    a_rs = split_ranges(χ, grid.N1)
    l_rs = split_ranges(χ, grid.N2)
    R_g = _slice2d_col_allgather(R_blk, grid, a_rs)
    if ndims(ARu_g) == 3
        @tensor partial[a, d] := ARu_g[a, b, c] * R_g[c, e] * ARd_g[d, b, e]
    else
        @tensor partial[a, d] := ARu_g[a, b, f, c] * R_g[c, e] * ARd_g[d, b, f, e]
    end
    result = _slice2d_row_reduce_scatter_last(partial, grid, l_rs)
    _free!(R_g)
    _free!(partial)
    return result
end

"""
    ACmap_slice2d_sliced(AC_blk, FL_g, FR_g, M, grid; forloop_iter=1) -> result_blk

Hoisted ACmap (Batch C): the FIXED partners `FL_g` (a-block, FULL-i; row-gathered)
and `FR_g` (FULL-d, l-block; col-gathered) are supplied PRE-GATHERED (hoisted once).
The ITERATE `AC_blk` is row-gathered to full-d INTERNALLY per call. Per-step map of
`ACenv_slice2d`. rrule (rules.jl) = the ACmap_slice2d_dist rrule minus the fixed FL/FR
gathers + their reduce-scatters (→ slice grads), keeping the iterate AC row-gather +
its row_reduce_scatter_last adjoint. Supports N1×N2 grids via separate r1/r2
partitions. See §2.2b/§4.
"""
function ACmap_slice2d_sliced(AC_blk, FL_g, FR_g, M, grid::Slice2DGrid; forloop_iter = 1)
    M1, M2 = M isa Tuple ? M : (M, conj(M))
    χ = size(FL_g, 4)                                   # full i (FL_g = [a-block, e, f, i-full])
    a_rs = split_ranges(χ, grid.N1)
    l_rs = split_ranges(χ, grid.N2)
    AC_g = _slice2d_row_allgather(AC_blk, grid, l_rs)    # iterate gather (full d), per-call
    result, _, _, _ = _acmap_slice2d_forward_sliced(AC_g, FR_g, FL_g, M1, M2, grid, a_rs; forloop_iter)
    return result
end

# ─── Cmap (replicated class — C replicated, FL/FR distributed) ───────────────
# Cmap leg4 result[e,f] := FL[a,c,d,e] C[a,b] FR[b,c,d,f]; leg3 result[d,e] :=
# FL[a,c,d] C[a,b] FR[b,c,e]. C stays REPLICATED (per the M3 design — C is
# tiny χ×χ); only FL/FR are block-stored. Output is the FULL χ×χ tensor,
# identical on every rank (every rank gathers the SAME full FL/FR and runs the
# SAME full chain → identical replicated output, NO allreduce). No ring, no
# output scatter, no square-grid assertion. Local einsum is CMAP_LEG*_CHAIN via
# chain_apply (whole-chain engine API), NOT a hand kernel.
# NO inner_etype kwarg: Cmap deliberately has no precondition/boundary downcast
# path (C is tiny χ×χ, no production inner_etype caller — cf. chain_maps.jl:167).
# The gather-class maps (FRmap/ACmap/ACdmap) DO take inner_etype and implement
# the FLmap-style do_cast — do NOT copy this no-cast signature to them.
# (CMAP_LEG*_CHAIN are defined in chain_maps.jl, included after this file, and
# resolve at call time via Julia's global late-binding.)
function Cmap_slice2d(C, FL_blk, FR_blk, grid::Slice2DGrid)
    χ = MPI.Allreduce(size(FL_blk, 1), +, grid.col_comm)   # full a/b extent (r1)
    a_rs = split_ranges(χ, grid.N1)
    e_rs = split_ranges(χ, grid.N2)
    # Make FL/FR fully local: gather the r1 (a/b) leg over col, the r2 (e/f)
    # leg over row. After both gathers each rank holds the IDENTICAL FULL FL and
    # FR, so the local chain produces the complete replicated χ×χ output — NO
    # allreduce needed (design (a) below).
    FL_full = _slice2d_col_allgather(_slice2d_row_allgather(FL_blk, grid, e_rs), grid, a_rs)
    FR_full = _slice2d_col_allgather(_slice2d_row_allgather(FR_blk, grid, e_rs), grid, a_rs)
    chain = ndims(FL_blk) == 3 ? CMAP_LEG3_CHAIN : CMAP_LEG4_CHAIN
    out = chain_apply(chain, (FL_full, C, FR_full))
    return out   # full χ×χ, replicated
end

# cuTENSOR contracts the M3.5 ring chain's 7-dim intermediate I2 (≈ na·local_l·
# D⁴·d_phys) through 32-bit-indexed StridedViews; once it exceeds ~2^31 elements
# cuTENSOR throws an illegal-address (code 700). Observed on a 2×2 grid at D=10
# χ=768 (I2≈2.95e9 > 2^31) with a single l-chunk; a 4×4 grid never hit it (I2 is
# χ²/P smaller). Floor the l-chunk COUNT so each chunk's I2 stays < 2e9, taking the
# max with the caller's forloop_iter (never fewer chunks than asked). Uniform-D
# production: Dmax⁴·d_phys is exact; non-uniform bonds: a safe over-estimate (a few
# extra chunks). Used by both reorder forwards (slice2d.jl) and rrules (rules.jl).
function _ring_l_chunks(na::Int, nl::Int, M1, M2, forloop_iter::Int)
    Dmax = max(size(M1,1), size(M1,2), size(M1,3), size(M1,4),
               size(M2,1), size(M2,2), size(M2,3), size(M2,4))
    dphys = size(M1, 5)
    inter_n1 = na * nl * Dmax^4 * dphys            # I2 element count at a single l-chunk
    n_floor  = cld(inter_n1, 2_000_000_000)        # keep each chunk's I2 < 2e9 (< 2^31, margin)
    n_eff    = clamp(max(forloop_iter, n_floor), 1, nl)
    return split_ranges(nl, n_eff)
end

# ─── FRmap (cross-axis gather class, RING reorder, single l-chunk) ────────────
# result[a,e,f,i] := ARd[i,j,k,l] FR[d,g,h,l] M1[e,j,g,b,p] M2[f,k,h,c,p] ARu[a,b,c,d]
# M3.5 RING reorder (docs/2026-06-15-m35-slice2d-ring-reorder-design.md): the local
# chain is FRMAP_LEG5_SLICE2D_CHAIN = ops (FR, ARu, M1, M2, ARd) — FR·ARu kills the
# cross-axis CONTRACTED leg d at link 1, so NO full-i×full-d plane ever forms.
# Every carried intermediate is a-block × l-block (χ²D⁴/P, like FLmap); the
# cross-axis OUTPUT leg i is born full only in the output buffer (a-block × i-FULL
# = χ²D²/N). So a SINGLE l-chunk that ACCUMULATES Σ_l over the local l-block bounds
# the chain to χ²D⁴/(P·forloop_iter) — identical to FLmap/ACmap, replacing the old
# 2-level d/i chunk. Gathers (in FRmap_slice2d_dist) are UNCHANGED (i full on ARd, d
# full on FR+ARu); F5 row_reduce_scatter_last (full i last leg, sum l over row, keep
# i-block r2) is UNCHANGED. Leg placement: l is on FR_g.4 and ARd_g.4 (the chunked
# local block); ARu_g has NO l (full d). The first χ leg uses the r1 partition
# `a_rs = split_ranges(χ, N1)` and the last χ leg uses the r2 partition
# `l_rs = split_ranges(χ, N2)`, so rectangular grids do not need a block transpose.
# Local einsum is FRMAP_LEG5_SLICE2D_CHAIN via chain_apply (whole-chain API), NOT a
# ring, NOT a hand kernel. (FRMAP_LEG5_SLICE2D_CHAIN is defined in chain_maps.jl,
# included after this file, and resolves at call time via Julia's global late-binding.)
function _frmap_slice2d_forward_sliced(ARd_g, FR_g, ARu_g, M1, M2, grid, a_rs, l_rs; forloop_iter = 1)
    # ARd_g = ARd[i∈1:χ, j, k, l∈l_rs[r2+1]]   (col_allgather, full i)
    # FR_g  = FR[d∈1:χ, g, h, l∈l_rs[r2+1]]    (col_allgather, full d)
    # ARu_g = ARu[a∈a_rs[r1+1], b, c, d∈1:χ]   (row_allgather, full d)
    χ = sum(length, a_rs)
    na = length(a_rs[grid.r1 + 1])             # local a extent
    nl = size(FR_g, 4)                          # local l extent (FR_g/ARd_g last leg)
    # out (a,e,f,i): e = M1's FIRST leg (:e in (:e,:j,:g,:b,:p)) = size(M1,1);
    # f = M2's FIRST leg (:f in (:f,:k,:h,:c,:p)) = size(M2,1); i is FULL.
    # (Verified with non-uniform bonds; uniform-D test would mask a wrong index.)
    partial = similar(ARu_g, na, size(M1,1), size(M2,1), χ)   # [a-block, e, f, i∈1:χ]
    l_chunks = _ring_l_chunks(na, nl, M1, M2, forloop_iter)   # 2^31 cuTENSOR floor (see _ring_l_chunks)
    firstchunk = true
    for ch in l_chunks                           # ACCUMULATE Σ_l (l is the aligned contracted leg)
        Pc = chain_apply(FRMAP_LEG5_SLICE2D_CHAIN,
            (FR_g[:, :, :, ch], ARu_g, M1, M2, ARd_g[:, :, :, ch]))
        if firstchunk
            partial .= Pc; firstchunk = false
        else
            partial .+= Pc
        end
        _free!(Pc)
    end
    # F5: sum l over row (full last leg i), keep i-block r2.
    @assert size(partial, 4) == χ "FRmap F5: partial last leg must be full i"
    result = _slice2d_row_reduce_scatter_last(partial, grid, l_rs)   # tag 760
    return result, ARd_g, FR_g, ARu_g
end

"""
    FRmap_slice2d_dist(FR_blk, ARu_blk, ARd_blk, M, grid; forloop_iter=1, inner_etype=nothing)

Distributed FRmap on an N1×N2 Slice2D grid (RING class via the M3.5 reorder).
FR/ARu/ARd are block-stored (first χ leg by r1, last χ leg by
r2; `slice2d_scatter` convention), M replicated. Two cross-axis legs (contracted
`d`, output `i`) are gathered to full before the local chain so the off-diagonal
`(a,i)` blocks are actually contracted (the diagonal trap). The r1/r2 partitions
are tracked separately as `a_rs = split_ranges(χ, N1)` and
`l_rs = split_ranges(χ, N2)`. `forloop_iter` is the FLmap-style SINGLE l-chunk
count (= #l-chunks): the M3.5 reorder
(`FRMAP_LEG5_SLICE2D_CHAIN`, FR·ARu kills the cross-axis contracted `d` at link 1)
makes every carried intermediate a-block×l-block (χ²D⁴/(P·forloop_iter)), so a
single l-loop that accumulates `Σ_l` suffices — no full-i×full-d plane, no 2-level
`d/i` chunk. Local einsum is `FRMAP_LEG5_SLICE2D_CHAIN` via `chain_apply`
(whole-chain engine API), NOT a ring, NOT a hand kernel. Collective over
`grid.comm` (gathers + `row_reduce_scatter_last` UNCHANGED from the gather-class
version — only the local chain order/chunk changed). See
docs/2026-06-15-m35-slice2d-ring-reorder-design.md and Batch B of
docs/2026-06-13-m3-slice2d-wrappers-plan.md.
"""
function FRmap_slice2d_dist(FR_blk, ARu_blk, ARd_blk, M, grid::Slice2DGrid;
                           forloop_iter = 1, inner_etype = nothing)
    M1, M2 = M isa Tuple ? M : (M, conj(M))
    T_orig = eltype(FR_blk)
    do_cast = inner_etype !== nothing && inner_etype != real(T_orig)
    if do_cast
        FR_blk  = _downcast_eltype(inner_etype, FR_blk)
        ARu_blk = _downcast_eltype(inner_etype, ARu_blk)
        ARd_blk = _downcast_eltype(inner_etype, ARd_blk)
        M1 = _downcast_eltype(inner_etype, M1); M2 = _downcast_eltype(inner_etype, M2)
    end
    χ = MPI.Allreduce(size(ARd_blk, 1), +, grid.col_comm)       # F0
    a_rs = split_ranges(χ, grid.N1)
    l_rs = split_ranges(χ, grid.N2)
    ARd_g = _slice2d_col_allgather(ARd_blk, grid, a_rs)         # F1: full i (tag 730)
    ARu_g = _slice2d_row_allgather(ARu_blk, grid, l_rs)         # F2: full d (tag 750)
    FR_g  = _slice2d_col_allgather(FR_blk,  grid, a_rs)         # F3: full d (tag 730)
    result, _, _, _ = _frmap_slice2d_forward_sliced(ARd_g, FR_g, ARu_g, M1, M2, grid, a_rs, l_rs; forloop_iter)
    return do_cast ? T_orig.(result) : result
end

# ─── ACmap (cross-axis gather class, SINGLE l-chunk) ─────────────────────────
# Design: docs/2026-06-13-acmap-slice2d-dataflow-design.md §2-§4.
# result[i,j,k,l] := AC[a,b,c,d] FR[d,g,h,l] M1[e,j,g,b,p] M2[f,k,h,c,p] FL[a,e,f,i]
# Two cross-axis legs: contracted d (AC.4=r2, FR.1=r1) + output i (FL.4=r2,
# result.1=r1). The diagonal trap (i and l both on r2) is defeated by gathering
# FL's free leg i to FULL before the local chain (F2). The first χ leg uses
# `a_rs = split_ranges(χ, N1)` and the last χ leg uses
# `l_rs = split_ranges(χ, N2)`.
# SINGLE l-chunk (NOT the 2-level d/i chunk of FRmap/ACdmap): ACmap's pinned
# chain intermediates I1=(a,c,h,l,b,g), I2=(a,l,e,j,c,h,p), I3=(l,j,k,a,e,f)
# carry only the LOCAL a-block + the (chunked) output l; the output i and
# contracted d never appear as intermediate legs because FL is the chain's LAST
# operand (i enters at the last link; d is consumed at link 2). So chunking the
# local l-block bounds them, identical to FLmap — no full-i×full-d plane exists.
# Local einsum is ACMAP_LEG5_CHAIN via chain_apply (whole-chain API), NOT a ring,
# NOT a hand kernel. Leg placement: i is on FL.4 (last leg of FL_g, gathered
# full) and partial.1 (full); d is on AC.4 (last leg of AC_g, gathered full) and
# FR.1 (first leg of FR_g, gathered full); l is on FR.4 (last leg of FR_g, local
# block) and partial.4 (local block) — the chunked leg.
# (ACMAP_LEG5_CHAIN is defined in chain_maps.jl, included after this file, and
# resolves at call time via Julia's global late-binding.)
function _acmap_slice2d_forward_sliced(AC_g, FR_g, FL_g, M1, M2, grid, a_rs; forloop_iter = 1)
    # AC_g = AC[a∈a_rs[r1+1], b, c, d∈1:χ]   (row_allgather, full d)
    # FR_g = FR[d∈1:χ, g, h, l∈l_rs[r2+1]]   (col_allgather, full d)
    # FL_g = FL[a∈a_rs[r1+1], e, f, i∈1:χ]   (row_allgather, full i)
    χ = sum(length, a_rs)
    nl = size(FR_g, 4)                       # local l extent
    # out (i,j,k,l): j = M1's leg :j in (:e,:j,:g,:b,:p) = size(M1,2);
    # k = M2's leg :k in (:f,:k,:h,:c,:p) = size(M2,2). (i is FULL; l is the local
    # block.) These are the ACmap output legs — NOT size(M1,1)/size(M2,1) (those
    # are :e/:f, the FRmap output legs).
    partial = similar(AC_g, χ, size(M1,2), size(M2,2), nl)   # [i∈1:χ, j, k, l-block]
    l_chunks = split_ranges(nl, min(forloop_iter, nl))
    for ch in l_chunks
        # F4: single local chain over full-i FL_g; full i, local l-chunk. The
        # chain contracts a at the LAST link (FL), so it sums only the LOCAL
        # a-block — the col_reduce_scatter in F5 completes Σ_a.
        Pc = chain_apply(ACMAP_LEG5_CHAIN, (AC_g, FR_g[:, :, :, ch], M1, M2, FL_g))
        view(partial, :, :, :, ch) .= Pc
        _free!(Pc)
    end
    # F5: sum a over col (full first leg i), keep i-block r1.
    @assert size(partial, 1) == χ "ACmap F5: partial first leg must be full i (risk 5)"
    result = _slice2d_col_reduce_scatter(partial, grid, a_rs)   # tag 710
    return result, AC_g, FR_g, FL_g
end

"""
    ACmap_slice2d_dist(AC_blk, FL_blk, FR_blk, M, grid; forloop_iter=1, inner_etype=nothing)

Distributed ACmap on an N1×N2 Slice2D grid (gather class). AC/FL/FR are
block-stored (first χ leg by r1, last χ leg by r2;
`slice2d_scatter` convention), M replicated. Two cross-axis legs (contracted
`d`, output `i`) are gathered to full before the local chain so the off-diagonal
`(i,l)` blocks are actually contracted (the diagonal trap defeated by gathering
the cross-axis FREE leg `i` on FL). The r1/r2 partitions are tracked separately as
`a_rs = split_ranges(χ, N1)` and `l_rs = split_ranges(χ, N2)`. `forloop_iter` sets
a SINGLE memory chunk over the LOCAL `l`-block (ACmap's intermediates carry
local-a + chunked-l, never a full-i×full-d plane, so one l-loop suffices — unlike
FRmap/ACdmap's 2-level chunk). Output `[i,j,k,l]` is block-distributed (i on r1, l on r2 — same
convention as the AC input, so the map iterates). Local einsum is
`ACMAP_LEG5_CHAIN` via `chain_apply` (whole-chain engine API), NOT a ring, NOT a
hand kernel. Collective over `grid.comm`. See
docs/2026-06-13-m3-slice2d-wrappers-plan.md Batch C and
docs/2026-06-13-acmap-slice2d-dataflow-design.md §2-§4.
"""
function ACmap_slice2d_dist(AC_blk, FL_blk, FR_blk, M, grid::Slice2DGrid;
                           forloop_iter = 1, inner_etype = nothing)
    M1, M2 = M isa Tuple ? M : (M, conj(M))
    T_orig = eltype(AC_blk)
    do_cast = inner_etype !== nothing && inner_etype != real(T_orig)
    if do_cast
        AC_blk = _downcast_eltype(inner_etype, AC_blk); FL_blk = _downcast_eltype(inner_etype, FL_blk)
        FR_blk = _downcast_eltype(inner_etype, FR_blk)
        M1 = _downcast_eltype(inner_etype, M1); M2 = _downcast_eltype(inner_etype, M2)
    end
    χ = MPI.Allreduce(size(AC_blk, 1), +, grid.col_comm)      # F0
    a_rs = split_ranges(χ, grid.N1)
    l_rs = split_ranges(χ, grid.N2)
    AC_g = _slice2d_row_allgather(AC_blk, grid, l_rs)          # F1: full d (tag 750)
    FL_g = _slice2d_row_allgather(FL_blk, grid, l_rs)          # F2: full i (tag 750)
    FR_g = _slice2d_col_allgather(FR_blk, grid, a_rs)          # F3: full d (tag 730)
    result, _, _, _ = _acmap_slice2d_forward_sliced(AC_g, FR_g, FL_g, M1, M2, grid, a_rs; forloop_iter)
    return do_cast ? T_orig.(result) : result
end

# ─── ACdmap (cross-axis gather class, RING reorder, single l-chunk) ───────────
# Design: docs/2026-06-15-m35-slice2d-ring-reorder-design.md (supersedes the 2-level
# i/d chunk of docs/2026-06-13-acmap-slice2d-dataflow-design.md §5-§6).
# result[a,b,c,d] := ACd[i,j,k,l] FR[d,g,h,l] M1[e,j,g,b,p] M2[f,k,h,c,p] FL[a,e,f,i]
# Cross-axis: contracted i (ACd.1=r1, FL.4=r2) + output d (FR.1=r1, result.4=r2);
# co-dist output a (FL.1=r1, local r1); aligned contracted l (ACd.4=r2, FR.4=r2,
# local r2). M3.5 RING reorder: the local chain is ACDMAP_LEG5_SLICE2D_CHAIN = ops
# (FL, ACd, M1, M2, FR) — FL·ACd kills the cross-axis CONTRACTED leg i at link 1,
# so NO full-i×full-d plane forms. Every carried intermediate is a-block×l-block
# (χ²D⁴/P, like FLmap); the cross-axis OUTPUT leg d is born full only in the output
# buffer (a-block × d-FULL = χ²D²/N). A SINGLE l-chunk that ACCUMULATES Σ_l over the
# local l-block bounds the chain to χ²D⁴/(P·forloop_iter) — identical to
# FLmap/ACmap, replacing the old 2-level i/d chunk (the former §5.1 BLOCKER is gone:
# the plane no longer exists, so there is nothing to bound by chunking i/d).
# Gathers (in ACdmap_slice2d_dist) are UNCHANGED (i full on ACd+FL, d full on FR);
# F5 row_reduce_scatter_last (full d last leg, sum l over row, keep d-block r2) is
# UNCHANGED. Leg placement: l is on ACd_g.4 and FR_g.4 (the chunked local block);
# FL_g has NO l (a-block × full i). The first χ leg uses
# `a_rs = split_ranges(χ, N1)` and the last χ leg uses
# `l_rs = split_ranges(χ, N2)`, so rectangular grids do not need a block transpose.
# Local einsum is ACDMAP_LEG5_SLICE2D_CHAIN via chain_apply (whole-chain API), NOT a
# ring, NOT a hand kernel. (ACDMAP_LEG5_SLICE2D_CHAIN is in chain_maps.jl, included
# after this file, resolved at call time via global late-binding.)
function _acdmap_slice2d_forward_sliced(ACd_g, FR_g, FL_g, M1, M2, grid, a_rs, l_rs; forloop_iter = 1)
    # ACd_g = ACd[i∈1:χ, j, k, l∈l_rs[r2+1]]  (col_allgather, full i)
    # FR_g  = FR[d∈1:χ, g, h, l∈l_rs[r2+1]]   (col_allgather, full d)
    # FL_g  = FL[a∈a_rs[r1+1], e, f, i∈1:χ]   (row_allgather, full i)
    χ = sum(length, a_rs)
    na = length(a_rs[grid.r1 + 1])             # local a extent
    nl = size(ACd_g, 4)                         # local l extent (ACd_g/FR_g last leg)
    # out (a,b,c,d): b = M1's leg :b in (:e,:j,:g,:b,:p) = size(M1,4);
    # c = M2's leg :c in (:f,:k,:h,:c,:p) = size(M2,4). NOT size(M1,2)/size(M2,2)
    # (those are :j/:k, the ACmap output legs). Verified with non-uniform bonds
    # (Db≠Dc); a uniform-D=3 test masks the wrong index — the Db≠Dc testset guards it.
    partial = similar(FL_g, na, size(M1,4), size(M2,4), χ)   # [a-block, b, c, d∈1:χ]
    l_chunks = _ring_l_chunks(na, nl, M1, M2, forloop_iter)  # 2^31 cuTENSOR floor (see _ring_l_chunks)
    firstchunk = true
    for ch in l_chunks                           # ACCUMULATE Σ_l (l is the aligned contracted leg)
        Pc = chain_apply(ACDMAP_LEG5_SLICE2D_CHAIN,
            (FL_g, ACd_g[:, :, :, ch], M1, M2, FR_g[:, :, :, ch]))
        if firstchunk
            partial .= Pc; firstchunk = false
        else
            partial .+= Pc
        end
        _free!(Pc)
    end
    # F5: sum l over row (full last leg d), keep d-block r2.
    @assert size(partial, 4) == χ "ACdmap F5: partial last leg must be full d (mirror of ACmap risk-5)"
    result = _slice2d_row_reduce_scatter_last(partial, grid, l_rs)   # tag 760
    return result, ACd_g, FR_g, FL_g
end

"""
    ACdmap_slice2d_dist(ACd_blk, FL_blk, FR_blk, M, grid; forloop_iter=1, inner_etype=nothing)

Distributed ACdmap on an N1×N2 Slice2D grid (RING class via the M3.5 reorder).
ACd/FL/FR are block-stored (first χ leg by r1, last χ leg by r2;
`slice2d_scatter` convention), M replicated. Two cross-axis legs (contracted `i`,
output `d`) are gathered to full before the local chain so the off-diagonal `(a,d)`
blocks are actually contracted (the diagonal trap, transposed vs ACmap). The r1/r2
partitions are tracked separately as `a_rs = split_ranges(χ, N1)` and
`l_rs = split_ranges(χ, N2)`. `forloop_iter` is the FLmap-style SINGLE l-chunk
count (= #l-chunks): the M3.5 reorder
(`ACDMAP_LEG5_SLICE2D_CHAIN`, FL·ACd kills the cross-axis contracted `i` at link 1)
makes every carried intermediate a-block×l-block (χ²D⁴/(P·forloop_iter)), so a
single l-loop that accumulates `Σ_l` suffices — no full-i×full-d plane, no 2-level
`i/d` chunk (the former §5.1 BLOCKER is gone). Output `[a,b,c,d]` is
block-distributed (a on r1, d on r2). Local einsum is `ACDMAP_LEG5_SLICE2D_CHAIN`
via `chain_apply` (whole-chain engine API), NOT a ring, NOT a hand kernel.
Collective over `grid.comm` (gathers + `row_reduce_scatter_last` UNCHANGED from the
gather-class version — only the local chain order/chunk changed). See
docs/2026-06-15-m35-slice2d-ring-reorder-design.md and Batch D of
docs/2026-06-13-m3-slice2d-wrappers-plan.md.
"""
function ACdmap_slice2d_dist(ACd_blk, FL_blk, FR_blk, M, grid::Slice2DGrid;
                            forloop_iter = 1, inner_etype = nothing)
    M1, M2 = M isa Tuple ? M : (M, conj(M))
    T_orig = eltype(ACd_blk)
    do_cast = inner_etype !== nothing && inner_etype != real(T_orig)
    if do_cast
        ACd_blk = _downcast_eltype(inner_etype, ACd_blk); FL_blk = _downcast_eltype(inner_etype, FL_blk)
        FR_blk = _downcast_eltype(inner_etype, FR_blk)
        M1 = _downcast_eltype(inner_etype, M1); M2 = _downcast_eltype(inner_etype, M2)
    end
    χ = MPI.Allreduce(size(ACd_blk, 1), +, grid.col_comm)        # F0
    a_rs = split_ranges(χ, grid.N1)
    l_rs = split_ranges(χ, grid.N2)
    ACd_g = _slice2d_col_allgather(ACd_blk, grid, a_rs)          # F1: full i (tag 730)
    FL_g  = _slice2d_row_allgather(FL_blk, grid, l_rs)           # F2: full i (tag 750)
    FR_g  = _slice2d_col_allgather(FR_blk, grid, a_rs)           # F3: full d (tag 730)
    result, _, _, _ = _acdmap_slice2d_forward_sliced(ACd_g, FR_g, FL_g, M1, M2, grid, a_rs, l_rs; forloop_iter)
    return do_cast ? T_orig.(result) : result
end
