# Cannon-style 2D distributed FLmap (map level).
# Design: docs/2026-06-10-cannon-flmap-design.md
#
# Rank (r1, r2) on an N1×N2 grid holds
#   FL block:      FL[a_range(r1), :, :, i_range(r2)]
#   result block:  result[d_range(r1), :, :, l_range(r2)]
# ALu / ALd / M are fully replicated (AL distribution comes with the later
# leftenv integration round). Stage 1 rotates FL blocks along the row ring
# while accumulating the stationary intermediate G; stage 2 contracts with
# ALu and reduce-scatters along the column. Output distribution = input
# distribution, so the map iterates without redistribution.

struct CannonGrid
    N1::Int
    N2::Int
    r1::Int
    r2::Int
    rank::Int
    comm::MPI.Comm
    row_comm::MPI.Comm   # fixed r1; rank within == r2 (size N2)
    col_comm::MPI.Comm   # fixed r2; rank within == r1 (size N1)
end

const _cannon_grid_cache = Ref{Union{Nothing, CannonGrid}}(nothing)

"""
    cannon_grid(N1, N2; comm=MPI.COMM_WORLD) -> CannonGrid

Build (and cache) the N1×N2 process grid with row/col sub-communicators.
Collective over `comm` on cache miss — must be called by all ranks with
identical `N1, N2`.
"""
function cannon_grid(N1::Integer, N2::Integer; comm = MPI.COMM_WORLD)
    g = _cannon_grid_cache[]
    if g !== nothing && g.N1 == N1 && g.N2 == N2 && g.comm == comm
        return g
    end
    nprocs = MPI.Comm_size(comm)
    @assert nprocs == N1 * N2 "cannon_grid: nprocs=$nprocs ≠ N1*N2=$(N1 * N2)"
    rank = MPI.Comm_rank(comm)
    r1, r2 = divrem(rank, N2)
    row_comm = MPI.Comm_split(comm, r1, r2)
    col_comm = MPI.Comm_split(comm, r2, r1)
    g = CannonGrid(N1, N2, r1, r2, rank, comm, row_comm, col_comm)
    _cannon_grid_cache[] = g
    return g
end

# ─── Stage kernels (leg5) ─────────────────────────────────────────────────
#
# FLmap splits into ring + fold + stage 2 so distributed FLOPs stay exactly
# serial/P:
#   ring (stage 1) contracts the i leg only, accumulating the pre-fold
#     intermediate H. The M fold is deliberately NOT in the ring: its cost is
#     independent of the i-block extent, so folding per step would redo it N2
#     times (overhead growing with grid size);
#   fold contracts M1/M2 into H once per map call;
#   stage 2 contracts a/b/c with ALu, leaving a full-length d leg for the
#     column reduce-scatter.
# Transient accounting: peak is ≈ (2+d)·|H| during the fold (@tensor pairwise
# temporaries) — the serial transient peak / P.

function _cannon_stage1(FL, ALd)
    @tensor H[a, e, f, j, k, l] := FL[a, e, f, i] * ALd[i, j, k, l]
    return H
end

# In-place accumulating variant for the forward ring (avoids a second H-sized
# temporary). Backward uses the non-mutating version through Zygote.pullback.
function _cannon_stage1_add!(H, FL, ALd)
    @tensor H[a, e, f, j, k, l] += FL[a, e, f, i] * ALd[i, j, k, l]
    return H
end

function _cannon_fold(H, M1, M2)
    @tensor G[a, b, c, g, h, l] := H[a, e, f, j, k, l] *
                                   M1[e, j, g, b, p] * M2[f, k, h, c, p]
    return G
end

function _cannon_stage2(G, ALu)
    @tensor P[d, g, h, l] := G[a, b, c, g, h, l] * ALu[a, b, c, d]
    return P
end

# ─── Boundary shims: full ↔ distributed blocks ────────────────────────────

"""
    cannon_scatter(T_full, grid) -> block

Local block of a replicated tensor: first leg split N1-ways (by r1), last leg
split N2-ways (by r2). Pure indexing — no communication.
"""
function cannon_scatter(T_full::AbstractArray, grid::CannonGrid)
    n = ndims(T_full)
    a_rs = split_ranges(size(T_full, 1), grid.N1)
    i_rs = split_ranges(size(T_full, n), grid.N2)
    inds = ntuple(j -> j == 1 ? a_rs[grid.r1 + 1] :
                       (j == n ? i_rs[grid.r2 + 1] : Colon()), n)
    return T_full[inds...]
end

"""
    cannon_gather(blk, grid) -> full

Reassemble the full tensor from all ranks' blocks (allgatherv on flattened
blocks, then per-block placement). Test/integration shim — not on the hot path.
"""
function cannon_gather(blk::AbstractArray, grid::CannonGrid)
    n = ndims(blk)
    χ1 = MPI.Allreduce(size(blk, 1), +, grid.col_comm)
    χ2 = MPI.Allreduce(size(blk, n), +, grid.row_comm)
    a_rs = split_ranges(χ1, grid.N1)
    i_rs = split_ranges(χ2, grid.N2)
    mid = size(blk)[2:n-1]
    P = grid.N1 * grid.N2
    counts = Cint[length(a_rs[divrem(r, grid.N2)[1] + 1]) * prod(mid) *
                  length(i_rs[divrem(r, grid.N2)[2] + 1]) for r in 0:P-1]
    displs = cumsum([0; counts[1:end-1]])
    buf = similar(blk, sum(counts))
    copyto!(view(buf, displs[grid.rank + 1] + 1 : displs[grid.rank + 1] + counts[grid.rank + 1]),
            vec(blk))
    allgatherv_p2p!(buf, counts, grid.comm)
    full = similar(blk, χ1, mid..., χ2)
    for r in 0:P-1
        s, t = divrem(r, grid.N2)
        seg = reshape(view(buf, displs[r + 1] + 1 : displs[r + 1] + counts[r + 1]),
                      length(a_rs[s + 1]), mid..., length(i_rs[t + 1]))
        inds = ntuple(j -> j == 1 ? a_rs[s + 1] : (j == n ? i_rs[t + 1] : Colon()), n)
        view(full, inds...) .= seg
    end
    return full
end
