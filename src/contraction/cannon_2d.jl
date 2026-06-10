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
# FLmap splits into two stages so distributed FLOPs stay exactly serial/P:
#   stage 1 contracts the i leg and folds in M (G is the only big transient,
#   sized (χ/N1)·D⁴·(χ/N2) — the serial intermediate / P);
#   stage 2 contracts a/b/c with ALu, leaving a full-length d leg for the
#   column reduce-scatter.

function _cannon_stage1(FL, ALd, M1, M2)
    @tensor G[a, b, c, g, h, l] := FL[a, e, f, i] * ALd[i, j, k, l] *
                                   M1[e, j, g, b, p] * M2[f, k, h, c, p]
    return G
end

# In-place accumulating variant for the forward ring (avoids a second G-sized
# temporary). Backward uses the non-mutating version through Zygote.pullback.
function _cannon_stage1_add!(G, FL, ALd, M1, M2)
    @tensor G[a, b, c, g, h, l] += FL[a, e, f, i] * ALd[i, j, k, l] *
                                   M1[e, j, g, b, p] * M2[f, k, h, c, p]
    return G
end

function _cannon_stage2(G, ALu)
    @tensor P[d, g, h, l] := G[a, b, c, g, h, l] * ALu[a, b, c, d]
    return P
end
