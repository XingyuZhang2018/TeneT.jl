# Cannon-style 2D distributed FLmap (map level).
# Design: docs/2026-06-10-cannon-flmap-design.md
#
# Rank (r1, r2) on an N1×N2 grid holds
#   FL block:      FL[a_range(r1), :, :, i_range(r2)]
#   result block:  result[d_range(r1), :, :, l_range(r2)]
# ALu / ALd / M are fully replicated (AL distribution comes with the later
# leftenv integration round). Stage 1 rotates FL blocks along the row ring
# while accumulating the pre-fold intermediate H; M is folded once after the
# ring, then stage 2 contracts with ALu and reduce-scatters along the column.
# Output distribution = input distribution, so the map iterates without
# redistribution.

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
# Transient accounting: peak is ≈ (1+d)·|H| during the fold (H and the owned
# pairwise intermediate T coexist) — the serial transient peak / P.

# Deterministic device-memory release for chunk transients. Plain @tensor
# never eagerly frees on CuArray (DefaultAllocator's tensorfree! is a no-op),
# so without explicit frees the pool rides at high-water across chunks and
# reactive GC pollutes timings. unsafe_free! is stream-ordered and the
# finalizer skips already-freed arrays.
_free!(x::CuArray) = CUDA.unsafe_free!(x)
_free!(x) = nothing

function _cannon_stage1(FL, ALd)
    @tensor H[a, e, f, j, k, l] := FL[a, e, f, i] * ALd[i, j, k, l]
    return H
end

# In-place accumulating variant for the ring (avoids a second H-sized
# temporary). The rrule backward recomputes H with the same kernels and then
# walks the hand adjoints below — no Zygote anywhere in the chunk body.
function _cannon_stage1_add!(H, FL, ALd)
    @tensor H[a, e, f, j, k, l] += FL[a, e, f, i] * ALd[i, j, k, l]
    return H
end

# Fold split into its two pairwise steps so the d·|H| intermediate T is an
# owned array (freeable eagerly) instead of a dead @tensor-internal temporary
# — TensorOperations' DefaultAllocator never frees on CuArray, and these
# temporaries are the largest single transient (job 1274256 OOM).
function _cannon_fold1(H, M1)
    @tensor T[a, f, k, g, b, p, l] := H[a, e, f, j, k, l] * M1[e, j, g, b, p]
    return T
end

function _cannon_fold2(T, M2)
    @tensor G[a, b, c, g, h, l] := T[a, f, k, g, b, p, l] * M2[f, k, h, c, p]
    return G
end

function _cannon_fold(H, M1, M2)
    T = _cannon_fold1(H, M1)
    G = _cannon_fold2(T, M2)
    _free!(T)
    return G
end

function _cannon_stage2(G, ALu)
    @tensor P[d, g, h, l] := G[a, b, c, g, h, l] * ALu[a, b, c, d]
    return P
end

# Hand-written adjoints of _cannon_stage1 (H = FL·ALd): two contractions
# instead of Zygote's three (whose pullback re-runs the stage-1 forward only
# to discard it — the cause of the large-χ backward slowdown in job 1265371).
function _cannon_stage1_dFL(dH, ALd)
    @tensor dFL[a, e, f, i] := dH[a, e, f, j, k, l] * conj(ALd[i, j, k, l])
    return dFL
end

function _cannon_stage1_dALd(dH, FL)
    @tensor dALd[i, j, k, l] := conj(FL[a, e, f, i]) * dH[a, e, f, j, k, l]
    return dALd
end

# Hand adjoints of stage 2 (P = G·ALu, contracting a,b,c):
function _cannon_stage2_dG(dP, ALu)
    @tensor dG[a, b, c, g, h, l] := dP[d, g, h, l] * conj(ALu[a, b, c, d])
    return dG
end
function _cannon_stage2_dALu(dP, G)
    @tensor dALu[a, b, c, d] := conj(G[a, b, c, g, h, l]) * dP[d, g, h, l]
    return dALu
end

# Hand adjoints of fold step 2 (G = T·M2, contracting f,k,p):
function _cannon_fold2_dT(dG, M2)
    @tensor dT[a, f, k, g, b, p, l] := dG[a, b, c, g, h, l] * conj(M2[f, k, h, c, p])
    return dT
end
function _cannon_fold2_dM2(dG, T)
    @tensor dM2[f, k, h, c, p] := conj(T[a, f, k, g, b, p, l]) * dG[a, b, c, g, h, l]
    return dM2
end

# Hand adjoints of fold step 1 (T = H·M1, contracting e,j):
function _cannon_fold1_dH(dT, M1)
    @tensor dH[a, e, f, j, k, l] := dT[a, f, k, g, b, p, l] * conj(M1[e, j, g, b, p])
    return dH
end
function _cannon_fold1_dM1(dT, H)
    @tensor dM1[e, j, g, b, p] := conj(H[a, e, f, j, k, l]) * dT[a, f, k, g, b, p, l]
    return dM1
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

# ─── Ring / column communication ──────────────────────────────────────────

# Send `cur` to the left row neighbor (r2-1) and receive the next block from
# the right (r2+1). Fresh exact-size buffer per step (uneven χ blocks differ
# in size; CUDA pool makes the allocation cheap). Never mutates `cur`.
function _cannon_row_shift(cur, grid::CannonGrid, recv_size; tag = _TAG_BASE + 700)
    dest = mod(grid.r2 - 1, grid.N2)
    src  = mod(grid.r2 + 1, grid.N2)
    recv = similar(cur, recv_size)
    synchronize(cur)
    req_r = MPI.Irecv!(recv, grid.row_comm; source = src, tag = tag)
    req_s = MPI.Isend(cur, grid.row_comm; dest = dest, tag = tag)
    MPI.Waitall([req_s, req_r])
    return recv
end

# Sum `partial` (full d leg, local l block) over the column and keep the local
# d block. Direct algorithm: each rank sends every other rank its chunk and
# accumulates the N1-1 contributions for its own chunk. Chunk extraction via
# getindex (allocating) keeps MPI buffers contiguous. All buffers are
# allocated and filled before the single synchronize, so stream-ordered
# allocations are complete before any Irecv! is posted.
function _cannon_col_reduce_scatter(partial, grid::CannonGrid, d_rs)
    N1, r1 = grid.N1, grid.r1
    N1 == 1 && return partial
    acc = partial[d_rs[r1 + 1], :, :, :]
    recvbufs = Vector{typeof(acc)}(undef, N1)
    sendbufs = Vector{typeof(acc)}(undef, N1)
    for j in 0:N1-1
        j == r1 && continue
        recvbufs[j + 1] = similar(acc)
        sendbufs[j + 1] = partial[d_rs[j + 1], :, :, :]
    end
    synchronize(partial)            # one sync covers allocs + chunk copies
    reqs = MPI.Request[]
    for j in 0:N1-1
        j == r1 && continue
        push!(reqs, MPI.Irecv!(recvbufs[j + 1], grid.col_comm; source = j, tag = _TAG_BASE + 710))
    end
    for j in 0:N1-1
        j == r1 && continue
        push!(reqs, MPI.Isend(sendbufs[j + 1], grid.col_comm; dest = j, tag = _TAG_BASE + 710))
    end
    MPI.Waitall(reqs)
    for j in 0:N1-1
        j == r1 && continue
        acc .+= recvbufs[j + 1]
    end
    return acc
end

# Adjoint of _cannon_col_reduce_scatter: gather all column ranks' d-blocks
# into a full-d tensor (identical on every column rank). All buffers are
# allocated before the single synchronize, matching the reduce-scatter's
# ordering discipline.
function _cannon_col_allgather(dblk, grid::CannonGrid, d_rs)
    N1, r1 = grid.N1, grid.r1
    nmid = ndims(dblk) - 1
    χ = sum(length, d_rs)
    full = similar(dblk, χ, size(dblk)[2:end]...)
    view(full, d_rs[r1 + 1], ntuple(_ -> Colon(), nmid)...) .= dblk
    N1 == 1 && return full
    recvbufs = Vector{typeof(full)}(undef, N1)
    for j in 0:N1-1
        j == r1 && continue
        recvbufs[j + 1] = similar(dblk, length(d_rs[j + 1]), size(dblk)[2:end]...)
    end
    synchronize(dblk)
    reqs = MPI.Request[]
    for j in 0:N1-1
        j == r1 && continue
        push!(reqs, MPI.Irecv!(recvbufs[j + 1], grid.col_comm; source = j, tag = _TAG_BASE + 730))
    end
    for j in 0:N1-1
        j == r1 && continue
        push!(reqs, MPI.Isend(dblk, grid.col_comm; dest = j, tag = _TAG_BASE + 730))
    end
    MPI.Waitall(reqs)
    for j in 0:N1-1
        j == r1 && continue
        view(full, d_rs[j + 1], ntuple(_ -> Colon(), nmid)...) .= recvbufs[j + 1]
    end
    return full
end

# Sum per-destination dFL contribution blocks over the row and deliver block
# t to rank (r1, t). Direct pairwise on row_comm, same buffer/sync discipline
# as the column reduce-scatter: allocate everything, one synchronize, post
# all Irecv! before all Isend.
function _cannon_row_reduce_scatter(contribs::Vector, grid::CannonGrid)
    N2, r2 = grid.N2, grid.r2
    acc = contribs[r2 + 1]
    N2 == 1 && return acc
    recvbufs = Vector{typeof(acc)}(undef, N2)
    for j in 0:N2-1
        j == r2 && continue
        recvbufs[j + 1] = similar(acc)
    end
    synchronize(acc)
    reqs = MPI.Request[]
    for j in 0:N2-1
        j == r2 && continue
        push!(reqs, MPI.Irecv!(recvbufs[j + 1], grid.row_comm; source = j, tag = _TAG_BASE + 740))
    end
    for j in 0:N2-1
        j == r2 && continue
        push!(reqs, MPI.Isend(contribs[j + 1], grid.row_comm; dest = j, tag = _TAG_BASE + 740))
    end
    MPI.Waitall(reqs)
    for j in 0:N2-1
        j == r2 && continue
        acc .+= recvbufs[j + 1]
    end
    return acc
end

# ─── Forward ──────────────────────────────────────────────────────────────

# Shared by FLmap_cannon and its rrule. Returns (result_blk, blocks): the ring
# rotates ONCE caching the N2 visiting FL blocks (χ²D²/N1 per rank — the AD
# capture, far smaller than H); the local l range is then processed in
# `forloop_iter` chunks, each running stage1 → fold → stage2 fully locally
# with transients bounded by (2+d)·|H|/forloop_iter.
function _cannon_forward(FL_blk, ALu, ALd, M1, M2, grid::CannonGrid; forloop_iter = 1)
    N1, N2, r1, r2 = grid.N1, grid.N2, grid.r1, grid.r2
    χ = size(ALu, 1)
    a_rs = split_ranges(χ, N1)
    i_rs = split_ranges(χ, N2)
    l_rng = i_rs[r2 + 1]
    @assert size(FL_blk, 1) == length(a_rs[r1 + 1]) && size(FL_blk, 4) == length(l_rng) "FLmap_cannon: block shape $(size(FL_blk)) inconsistent with grid ($(N1)×$(N2)) and χ=$χ"
    @assert forloop_iter ≥ 1 "FLmap_cannon: forloop_iter must be ≥ 1"

    # Ring: rotate once, cache the visiting FL blocks by their i-block index.
    blocks = Vector{typeof(FL_blk)}(undef, N2)
    cur = FL_blk
    for k in 0:N2-1
        t = mod(r2 + k, N2)
        blocks[t + 1] = cur
        if k < N2 - 1
            t_next = mod(r2 + k + 1, N2)
            cur = _cannon_row_shift(cur, grid,
                (length(a_rs[r1 + 1]), size(FL_blk, 2), size(FL_blk, 3), length(i_rs[t_next + 1])))
        end
    end

    # Local pipeline per l-chunk: stage 1 accumulate → fold once → stage 2.
    Dg, Dh = size(M1, 3), size(M2, 3)
    partial = similar(FL_blk, χ, Dg, Dh, length(l_rng))
    ALu_slice = view(ALu, a_rs[r1 + 1], :, :, :)
    l_chunks = split_ranges(length(l_rng), min(forloop_iter, length(l_rng)))
    for ch in l_chunks
        l_glob = l_rng[ch]
        local Hc
        for t in 0:N2-1
            ALd_slice = view(ALd, i_rs[t + 1], :, :, l_glob)
            if t == 0
                Hc = _cannon_stage1(blocks[t + 1], ALd_slice)
            else
                _cannon_stage1_add!(Hc, blocks[t + 1], ALd_slice)
            end
        end
        Gc = _cannon_fold(Hc, M1, M2)
        _free!(Hc)
        Pc = _cannon_stage2(Gc, ALu_slice)
        _free!(Gc)
        view(partial, :, :, :, ch) .= Pc
        _free!(Pc)
    end
    result = _cannon_col_reduce_scatter(partial, grid, a_rs)
    return result, blocks
end

"""
    FLmap_cannon(FL_blk, ALu, ALd, M, grid; forloop_iter=1, inner_etype=nothing) -> result_blk

Distributed FLmap on an N1×N2 Cannon grid. `FL_blk` and the returned block
follow the convention: first χ leg split N1-ways by r1, last χ leg split
N2-ways by r2. `M` is a leg5 tensor or an `(M1, M2)` tuple; ALu/ALd/M are
replicated on every rank. Collective over `grid.comm`.
`forloop_iter` sub-slices the local l range: forward per-chunk transients are
≈(1+d)·χ²D⁴/(P·forloop_iter); backward ≈(2+2d)·χ²D⁴/(P·forloop_iter) (fully
hand-written adjoint chain, every intermediate freed after its last use) —
size `forloop_iter` by the backward bound when gradients are needed.
See docs/2026-06-10-cannon-flmap-design.md.
"""
function FLmap_cannon(FL_blk, ALu, ALd, M, grid::CannonGrid; forloop_iter = 1, inner_etype = nothing)
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
    result, _ = _cannon_forward(FL_blk, ALu, ALd, M1, M2, grid; forloop_iter)
    return do_cast ? T_orig.(result) : result
end
