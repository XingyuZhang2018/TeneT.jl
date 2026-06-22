# Slice2D-style 2D distributed FLmap (map level).
# Design: docs/2026-06-10-slice2d-flmap-design.md
#
# Rank (r1, r2) on an N1×N2 grid holds
#   FL block:      FL[a_range(r1), :, :, i_range(r2)]
#   result block:  result[d_range(r1), :, :, l_range(r2)]
# M is replicated (tiny). ALu/ALd: `FLmap_slice2d` takes them replicated;
# `FLmap_slice2d_dist` takes them block-stored in the same convention and
# assembles the irreducible per-rank slices ALu[a_r1, :, :, :] (row
# allgather) and ALd[:, :, :, l_r2] (column allgather). Stage 1 gathers the
# FL iterate's row slice once and accumulates the pre-fold intermediate H;
# M is folded once after the row-slice sum, then stage 2 contracts with ALu and
# reduce-scatters along the column. Output distribution = input
# distribution, so the map iterates without redistribution.

struct Slice2DGrid
    N1::Int
    N2::Int
    r1::Int
    r2::Int
    rank::Int
    comm::MPI.Comm
    row_comm::MPI.Comm   # fixed r1; rank within == r2 (size N2)
    col_comm::MPI.Comm   # fixed r2; rank within == r1 (size N1)
end

const _slice2d_grid_cache = Ref{Union{Nothing, Slice2DGrid}}(nothing)

"""
    slice2d_grid(N1, N2; comm=MPI.COMM_WORLD) -> Slice2DGrid

Build (and cache) the N1×N2 process grid with row/col sub-communicators.
Collective over `comm` on cache miss — must be called by all ranks with
identical `N1, N2`.
"""
function slice2d_grid(N1::Integer, N2::Integer; comm = MPI.COMM_WORLD)
    g = _slice2d_grid_cache[]
    if g !== nothing && g.N1 == N1 && g.N2 == N2 && g.comm == comm
        return g
    end
    nprocs = MPI.Comm_size(comm)
    @assert nprocs == N1 * N2 "slice2d_grid: nprocs=$nprocs ≠ N1*N2=$(N1 * N2)"
    rank = MPI.Comm_rank(comm)
    r1, r2 = divrem(rank, N2)
    row_comm = MPI.Comm_split(comm, r1, r2)
    col_comm = MPI.Comm_split(comm, r2, r1)
    g = Slice2DGrid(N1, N2, r1, r2, rank, comm, row_comm, col_comm)
    _slice2d_grid_cache[] = g
    return g
end

# ─── Boundary shims: full ↔ distributed blocks ────────────────────────────

"""
    slice2d_scatter(T_full, grid) -> block

Local block of a replicated tensor: first leg split N1-ways (by r1), last leg
split N2-ways (by r2). Pure indexing — no communication.
"""
function slice2d_scatter(T_full::AbstractArray, grid::Slice2DGrid)
    n = ndims(T_full)
    a_rs = split_ranges(size(T_full, 1), grid.N1)
    i_rs = split_ranges(size(T_full, n), grid.N2)
    inds = ntuple(j -> j == 1 ? a_rs[grid.r1 + 1] :
                       (j == n ? i_rs[grid.r2 + 1] : Colon()), n)
    return T_full[inds...]
end

"""
    slice2d_gather(blk, grid) -> full

Reassemble the full tensor from all ranks' blocks (allgatherv on flattened
blocks, then per-block placement). Test/integration shim — not on the hot path.
"""
function slice2d_gather(blk::AbstractArray, grid::Slice2DGrid)
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

# ─── Helpers for distributed observable contractions (oc_22 qrpos seam) ────
#
# Full χ from a χ-block: the first tensor leg is split by r1 across the column
# communicator, so allreduce the local first-dim over col_comm (mirrors
# FLmap_slice2d_dist line ~540).
_slice2d_full_chi(blk, grid::Slice2DGrid) = MPI.Allreduce(size(blk, 1), +, grid.col_comm)

# Rank-consistent random Q0 (χf,D1,D2,χf) block for oc_22's one power iteration.
# CORRECTNESS-CRITICAL: every rank must hold blocks of the SAME full Q0, else the
# distributed maps contract mismatched blocks (silent garbage, not a crash). So
# generate full Q0 on rank 0, Bcast over grid.comm, then slice2d_scatter to this
# rank's block (reusing the audited r1/r2 slicer). Caller wraps in Zygote.@ignore.
function _slice2d_random_Q0(template, χf::Int, D1::Int, D2::Int, grid::Slice2DGrid)
    Q0 = _arraytype(template)(randn(eltype(template), χf, D1, D2, χf))
    MPI.Bcast!(Q0, 0, grid.comm)
    return slice2d_scatter(Q0, grid)
end

# ─── Distributed inner product / norm over block tiles ────────────────────
#
# Blocks tile the full tensor disjointly (first χ leg by r1, last by r2), so
# the global inner product is the sum of local ones — a single scalar
# allreduce. Collective over `grid.comm`. These are the `inner_product` /
# `norm_fn` hooks for running simple_eig fully distributed on Slice2D blocks.

function slice2d_dot(x_blk, y_blk, grid::Slice2DGrid)
    return MPI.Allreduce(dot(x_blk, y_blk), +, grid.comm)
end

function slice2d_norm(x_blk, grid::Slice2DGrid)
    return sqrt(MPI.Allreduce(sum(abs2, x_blk), +, grid.comm))
end

# ─── Row / column communication ───────────────────────────────────────────

# Sum `partial` (full d leg, local l block) over the column and keep the local
# d block. Direct algorithm: each rank sends every other rank its chunk and
# accumulates the N1-1 contributions for its own chunk. Chunk extraction via
# getindex (allocating) keeps MPI buffers contiguous. All buffers are
# allocated and filled before the single synchronize, so stream-ordered
# allocations are complete before any Irecv! is posted.
function _slice2d_col_reduce_scatter(partial, grid::Slice2DGrid, d_rs)
    N1, r1 = grid.N1, grid.r1
    N1 == 1 && return partial
    if _use_nccl() && partial isa CuArray && _equal_blocks(d_rs)   # NCCL fast path (cross-node col axis)
        return _nccl_slice2d_reduce_scatter!(partial, grid.col_comm, true)
    end
    tail = ntuple(_ -> Colon(), ndims(partial) - 1)
    acc = partial[d_rs[r1 + 1], tail...]
    recvbufs = Vector{typeof(acc)}(undef, N1)
    sendbufs = Vector{typeof(acc)}(undef, N1)
    for j in 0:N1-1
        j == r1 && continue
        recvbufs[j + 1] = similar(acc)
        sendbufs[j + 1] = partial[d_rs[j + 1], tail...]
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

# Adjoint of _slice2d_col_reduce_scatter: gather all column ranks' d-blocks
# into a full-d tensor (identical on every column rank). All buffers are
# allocated before the single synchronize, matching the reduce-scatter's
# ordering discipline.
function _slice2d_col_allgather(dblk, grid::Slice2DGrid, d_rs)
    N1, r1 = grid.N1, grid.r1
    if N1 > 1 && _use_nccl() && dblk isa CuArray && _equal_blocks(d_rs)   # NCCL fast path (cross-node col axis)
        return _nccl_slice2d_allgather!(dblk, grid.col_comm, true)
    end
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

# Assemble the local-a, full-d row slice of a block-distributed tensor:
# gather the N2 row peers' blocks and concatenate along the LAST leg.
# Mirror of _slice2d_col_allgather; same buffer/sync discipline. Tag 750.
function _slice2d_row_allgather(blk, grid::Slice2DGrid, l_rs)
    N2, r2 = grid.N2, grid.r2
    if N2 > 1 && _use_nccl() && blk isa CuArray && _equal_blocks(l_rs)   # NCCL fast path (row axis, last leg)
        return _nccl_slice2d_allgather!(blk, grid.row_comm, false)
    end
    nfront = ndims(blk) - 1
    χ2 = sum(length, l_rs)
    front = size(blk)[1:nfront]
    full = similar(blk, front..., χ2)
    view(full, ntuple(_ -> Colon(), nfront)..., l_rs[r2 + 1]) .= blk
    N2 == 1 && return full
    recvbufs = Vector{typeof(full)}(undef, N2)
    for j in 0:N2-1
        j == r2 && continue
        recvbufs[j + 1] = similar(blk, front..., length(l_rs[j + 1]))
    end
    synchronize(blk)
    reqs = MPI.Request[]
    for j in 0:N2-1
        j == r2 && continue
        push!(reqs, MPI.Irecv!(recvbufs[j + 1], grid.row_comm; source = j, tag = _TAG_BASE + 750))
    end
    for j in 0:N2-1
        j == r2 && continue
        push!(reqs, MPI.Isend(blk, grid.row_comm; dest = j, tag = _TAG_BASE + 750))
    end
    MPI.Waitall(reqs)
    for j in 0:N2-1
        j == r2 && continue
        view(full, ntuple(_ -> Colon(), nfront)..., l_rs[j + 1]) .= recvbufs[j + 1]
    end
    return full
end

function _slice2d_row_allgather_first(blk, grid::Slice2DGrid, a_rs)
    N2, r2 = grid.N2, grid.r2
    nrest = ndims(blk) - 1
    chi = sum(length, a_rs)
    full = similar(blk, chi, size(blk)[2:end]...)
    view(full, a_rs[r2 + 1], ntuple(_ -> Colon(), nrest)...) .= blk
    N2 == 1 && return full
    recvbufs = Vector{typeof(blk)}(undef, N2)
    for j in 0:N2-1
        j == r2 && continue
        recvbufs[j + 1] = similar(blk, length(a_rs[j + 1]), size(blk)[2:end]...)
    end
    synchronize(blk)
    reqs = MPI.Request[]
    for j in 0:N2-1
        j == r2 && continue
        push!(reqs, MPI.Irecv!(recvbufs[j + 1], grid.row_comm; source = j, tag = _TAG_BASE + 770))
    end
    for j in 0:N2-1
        j == r2 && continue
        push!(reqs, MPI.Isend(blk, grid.row_comm; dest = j, tag = _TAG_BASE + 770))
    end
    MPI.Waitall(reqs)
    for j in 0:N2-1
        j == r2 && continue
        view(full, a_rs[j + 1], ntuple(_ -> Colon(), nrest)...) .= recvbufs[j + 1]
    end
    return full
end

# Sum the row peers' full-d slice gradients and keep the local d block:
# adjoint of _slice2d_row_allgather. Direct pairwise on row_comm, tag 760.
# Recv sizing: every row peer's slice has MY l-block columns at MY range, so
# `similar(acc)` is right even for uneven χ (same reasoning as the sendbuf /
# recvbuf split in _slice2d_col_reduce_scatter).
function _slice2d_row_reduce_scatter_last(dslice, grid::Slice2DGrid, l_rs)
    N2, r2 = grid.N2, grid.r2
    if N2 > 1 && _use_nccl() && dslice isa CuArray && _equal_blocks(l_rs)   # NCCL fast path (row axis, last leg)
        return _nccl_slice2d_reduce_scatter!(dslice, grid.row_comm, false)
    end
    nfront = ndims(dslice) - 1
    cols = ntuple(_ -> Colon(), nfront)
    acc = dslice[cols..., l_rs[r2 + 1]]
    N2 == 1 && return acc
    recvbufs = Vector{typeof(acc)}(undef, N2)
    sendbufs = Vector{typeof(acc)}(undef, N2)
    for j in 0:N2-1
        j == r2 && continue
        recvbufs[j + 1] = similar(acc)
        sendbufs[j + 1] = dslice[cols..., l_rs[j + 1]]
    end
    synchronize(dslice)
    reqs = MPI.Request[]
    for j in 0:N2-1
        j == r2 && continue
        push!(reqs, MPI.Irecv!(recvbufs[j + 1], grid.row_comm; source = j, tag = _TAG_BASE + 760))
    end
    for j in 0:N2-1
        j == r2 && continue
        push!(reqs, MPI.Isend(sendbufs[j + 1], grid.row_comm; dest = j, tag = _TAG_BASE + 760))
    end
    MPI.Waitall(reqs)
    for j in 0:N2-1
        j == r2 && continue
        acc .+= recvbufs[j + 1]
    end
    return acc
end

function _slice2d_row_reduce_scatter_first(dfull, grid::Slice2DGrid, a_rs)
    N2, r2 = grid.N2, grid.r2
    nrest = ndims(dfull) - 1
    cols = ntuple(_ -> Colon(), nrest)
    acc = dfull[a_rs[r2 + 1], cols...]
    N2 == 1 && return acc
    recvbufs = Vector{typeof(acc)}(undef, N2)
    sendbufs = Vector{typeof(acc)}(undef, N2)
    for j in 0:N2-1
        j == r2 && continue
        recvbufs[j + 1] = similar(acc)
        sendbufs[j + 1] = dfull[a_rs[j + 1], cols...]
    end
    synchronize(dfull)
    reqs = MPI.Request[]
    for j in 0:N2-1
        j == r2 && continue
        push!(reqs, MPI.Irecv!(recvbufs[j + 1], grid.row_comm; source = j, tag = _TAG_BASE + 780))
    end
    for j in 0:N2-1
        j == r2 && continue
        push!(reqs, MPI.Isend(sendbufs[j + 1], grid.row_comm; dest = j, tag = _TAG_BASE + 780))
    end
    MPI.Waitall(reqs)
    for j in 0:N2-1
        j == r2 && continue
        acc .+= recvbufs[j + 1]
    end
    return acc
end

# ─── M4: gather hoisting — differentiable gather wrappers ─────────────────────
# Design: docs/2026-06-15-m4-env-slice2d-integration-design.md §2.2a.
# These are the DISTRIBUTED-OUTPUT gather primitives used by the env-level
# gather hoisting (leftenv/rightenv/ACenv): the FIXED boundary slices are
# gathered ONCE outside the power iteration, and their rrule adjoint is the
# matching reduce-scatter (`slice2d_gather_row` ↔ `_slice2d_row_reduce_scatter_last`,
# `slice2d_gather_col` ↔ `_slice2d_col_reduce_scatter`; rrules in autodiff/rules.jl).
# NAMED DISTINCTLY from the replicated-output `slice2d_gather` (whose adjoint is
# take-my-block) so the wrong adjoint can never leak onto a replicated map (Cmap).
# Forward is just the existing allgather (no behaviour change); the rrule supplies
# the reduce-scatter so the type-B input-gather adjoint fires ONCE at the hoist
# boundary (linearity: sum-then-scatter ≡ scatter-then-sum). The ranges argument
# is the SAME the *_slice2d_dist wrappers pass: `l_rs` (N2 partition) for the row
# gather, `a_rs` (N1 partition) for the column gather.
slice2d_gather_row(blk, grid::Slice2DGrid, l_rs) = _slice2d_row_allgather(blk, grid, l_rs)
slice2d_gather_first_row(blk, grid::Slice2DGrid, a_rs) = _slice2d_row_allgather_first(blk, grid, a_rs)
slice2d_gather_col(blk, grid::Slice2DGrid, a_rs) = _slice2d_col_allgather(blk, grid, a_rs)
