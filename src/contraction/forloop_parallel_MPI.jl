"""
    split_count(N::Integer, n::Integer)

Return a vector of `n` integers which are approximately equally sized and sum to `N`.
"""
function split_count(N::Integer, n::Integer)
    q,r = divrem(N, n)
    return [i <= r ? q+1 : q for i = 1:n]
end

function split_ranges(counts::Vector)
    nprocs = length(counts)
    cumulative = cumsum([0; counts])
    return [cumulative[i] + 1 : cumulative[i+1] for i in 1:nprocs]
end

function split_ranges(N::Integer, n::Integer)
    counts = split_count(N, n)
    return split_ranges(counts)
end

# ─── Pre-allocated communication buffers ──────────────────────────────────

const _comm_sendbuf = Ref{Any}(nothing)
const _comm_recvbuf = Ref{Any}(nothing)
const _comm_local = Ref{Any}(nothing)     # node-local communicator
const _comm_leaders = Ref{Any}(nothing)   # inter-node leaders communicator
const _comm_siblings = Ref{Any}(nothing)  # inter-node comm split by local_rank

# MPI tag namespace for internal p2p collectives. Keep well above any tag the
# caller might use (current callers use tag = rank ∈ 0..nprocs-1).
const _TAG_BASE = 1000

function _ensure_buf!(ref, buf, n)
    if isnothing(ref[]) || length(ref[]) < n || eltype(ref[]) != eltype(buf)
        ref[] = similar(buf, n)
    end
    return view(ref[], 1:n)
end

function _get_local_comm(comm)
    if isnothing(_comm_local[])
        rank = MPI.Comm_rank(comm)
        local_comm = MPI.Comm_split_type(comm, MPI.COMM_TYPE_SHARED, rank)
        local_rank = MPI.Comm_rank(local_comm)
        # All ranks participate in Comm_split; leaders get color=0, others color=1
        leaders_comm = MPI.Comm_split(comm, local_rank == 0 ? 0 : 1, rank)
        _comm_local[] = local_comm
        # Only store leaders_comm for local_rank==0; others don't use it
        _comm_leaders[] = local_rank == 0 ? leaders_comm : nothing
    end
    return _comm_local[], _comm_leaders[]
end

# Sibling communicator: groups ranks with the same local_rank across nodes.
# Used for Phase 2 of allreduce to do per-slice cross-node rings in parallel.
function _per_local_rank_comm(comm, local_comm)
    if _comm_siblings[] === nothing
        local_rank = MPI.Comm_rank(local_comm)
        rank = MPI.Comm_rank(comm)
        _comm_siblings[] = MPI.Comm_split(comm, local_rank, rank)
    end
    return _comm_siblings[]
end

# ─── Ring topology + node-layout helpers ─────────────────────────────────

"""
    _ring_neighbors(rank, size) -> (prev, next)

Cyclic predecessor/successor under the standard ring ordering `0..size-1`.
"""
_ring_neighbors(rank, size) = (mod(rank - 1, size), mod(rank + 1, size))

"""
    _node_ranges(counts, local_size, n_nodes)

Given per-global-rank `counts` and a symmetric layout of `local_size` ranks on
each of `n_nodes` nodes, return `n_nodes`-long vector of `(first, last)` index
tuples (1-indexed, inclusive) spanning each node's contiguous slab of the
allgatherv buffer.
"""
function _node_ranges(counts, local_size, n_nodes)
    displs = cumsum([0; counts[1:end-1]])
    ranges = Vector{Tuple{Int,Int}}(undef, n_nodes)
    for k in 1:n_nodes
        first_global = (k - 1) * local_size + 1
        last_global  = k * local_size
        first_idx = displs[first_global] + 1
        last_idx  = displs[last_global] + counts[last_global]
        ranges[k] = (first_idx, last_idx)
    end
    return ranges
end

# ─── Hierarchical Allgatherv ──────────────────────────────────────────────

"""
    allgatherv_p2p!(buf, counts, comm)

Bandwidth-optimal hierarchical allgatherv via three phases:

1. Intra-node: concurrent `Irecv!`/`Isend` all-to-all (NVSwitch saturates
   pairwise transfers).
2. Leader inter-node: ring allgatherv across node leaders over IB.
3. Intra-node: leader broadcasts the other-node chunks (local chunks are
   already in place from Phase 1).

Degenerates cleanly to Phase 1 only when `n_nodes == 1`. Pre-allocated
`_comm_sendbuf` keeps GPU memory registrations from accumulating.
"""
function allgatherv_p2p!(buf, counts, comm)
    rank   = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)
    nprocs == 1 && return buf
    synchronize(buf)

    local_comm, _ = _get_local_comm(comm)
    local_rank = MPI.Comm_rank(local_comm)
    local_size = MPI.Comm_size(local_comm)
    n_nodes    = nprocs ÷ local_size
    @assert nprocs == n_nodes * local_size "allgatherv_p2p! requires symmetric node layout; got nprocs=$nprocs, local_size=$local_size"

    displs   = cumsum([0; counts[1:end-1]])
    my_count = counts[rank + 1]

    sendbuf = _ensure_buf!(_comm_sendbuf, buf, my_count)
    copyto!(sendbuf, view(buf, displs[rank+1]+1 : displs[rank+1]+my_count))
    synchronize(buf)

    # ── Phase 1: intra-node concurrent all-to-all ──
    node_of_me = rank ÷ local_size
    reqs = MPI.Request[]
    for lr in 0:local_size-1
        lr == local_rank && continue
        peer = node_of_me * local_size + lr
        rc   = counts[peer + 1]
        rview = view(buf, displs[peer+1]+1 : displs[peer+1]+rc)
        push!(reqs, MPI.Irecv!(rview, comm; source=peer, tag=_TAG_BASE + peer))
    end
    for lr in 0:local_size-1
        lr == local_rank && continue
        peer = node_of_me * local_size + lr
        push!(reqs, MPI.Isend(sendbuf, comm; dest=peer, tag=_TAG_BASE + rank))
    end
    MPI.Waitall(reqs)

    # ── Phase 2: leader inter-node ring allgatherv ──
    if n_nodes > 1 && local_rank == 0
        _allgatherv_ring_leaders!(buf, counts, local_size, n_nodes, comm)
    end

    # ── Phase 3: intra-node broadcast of other-node chunks ──
    if n_nodes > 1 && local_size > 1
        _allgatherv_broadcast_other_nodes!(buf, counts, local_comm,
                                           local_rank, local_size,
                                           node_of_me, n_nodes)
    end

    return buf
end

# Phase 2: leaders only. Walks a ring of `leaders_comm`, at step s sending the
# node range the leader received s-1 steps ago (step 1 sends own node range).
function _allgatherv_ring_leaders!(buf, counts, local_size, n_nodes, comm)
    _, leaders_comm = _get_local_comm(comm)
    leaders_comm === nothing && return buf
    leaders_size = MPI.Comm_size(leaders_comm)
    leaders_size <= 1 && return buf
    leaders_rank = MPI.Comm_rank(leaders_comm)

    prev, next = _ring_neighbors(leaders_rank, leaders_size)
    ranges     = _node_ranges(counts, local_size, n_nodes)

    for step in 1:leaders_size-1
        send_idx = mod(leaders_rank - step + 1, leaders_size) + 1
        recv_idx = mod(leaders_rank - step,     leaders_size) + 1
        fs, ls = ranges[send_idx]
        fr, lr_ = ranges[recv_idx]
        synchronize(buf)
        req_send = MPI.Isend(view(buf, fs:ls),   leaders_comm;
                             dest=next, tag=_TAG_BASE + 100 + step)
        req_recv = MPI.Irecv!(view(buf, fr:lr_), leaders_comm;
                              source=prev, tag=_TAG_BASE + 100 + step)
        MPI.Waitall([req_send, req_recv])
    end
    return buf
end

# Phase 3: intra-node broadcast. Leader sends every other-node slab to each
# local peer; peers Irecv into buf. Local-node slab is already correct from
# Phase 1, so we skip it and save ~1/n_nodes of broadcast bandwidth.
function _allgatherv_broadcast_other_nodes!(buf, counts, local_comm,
                                            local_rank, local_size,
                                            node_of_me, n_nodes)
    ranges = _node_ranges(counts, local_size, n_nodes)
    synchronize(buf)
    reqs = MPI.Request[]
    if local_rank == 0
        for k in 1:n_nodes
            k - 1 == node_of_me && continue
            fs, ls = ranges[k]
            sub = view(buf, fs:ls)
            for peer in 1:local_size-1
                push!(reqs, MPI.Isend(sub, local_comm;
                                      dest=peer, tag=_TAG_BASE + 200 + k))
            end
        end
    else
        for k in 1:n_nodes
            k - 1 == node_of_me && continue
            fs, ls = ranges[k]
            push!(reqs, MPI.Irecv!(view(buf, fs:ls), local_comm;
                                   source=0, tag=_TAG_BASE + 200 + k))
        end
    end
    MPI.Waitall(reqs)
    return buf
end

# ─── Hierarchical Allreduce ──────────────────────────────────────────────

"""
    allreduce_p2p!(buf, +, comm)

Hierarchical allreduce: tree reduce within node (NVLink), then between
node leaders (IB), then broadcast back within node. Uses pre-allocated
fixed buffers to avoid GPU registration accumulation.
"""
function allreduce_p2p!(buf, ::typeof(+), comm)
    rank = MPI.Comm_rank(comm)
    P = MPI.Comm_size(comm)
    P == 1 && return buf
    synchronize(buf)
    N = length(buf)

    local_comm, leaders_comm = _get_local_comm(comm)
    local_rank = MPI.Comm_rank(local_comm)
    local_size = MPI.Comm_size(local_comm)

    _ensure_buf!(_comm_recvbuf, buf, N)
    recvbuf = reshape(view(_comm_recvbuf[], 1:N), size(buf))

    # Phase 1: Tree reduce within node to local_rank 0
    step = 1
    while step < local_size
        if local_rank % (2 * step) == 0
            partner = local_rank + step
            if partner < local_size
                MPI.Recv!(recvbuf, local_comm; source=partner, tag=step)
                buf .+= recvbuf
            end
        elseif local_rank % (2 * step) == step
            synchronize(buf)
            sendbuf = _ensure_buf!(_comm_sendbuf, buf, N)
            copyto!(sendbuf, buf)
            synchronize(buf)
            MPI.Send(sendbuf, local_comm; dest=local_rank - step, tag=step)
            # This rank is done reducing, wait for broadcast
        end
        step *= 2
    end

    # Phase 2: Reduce between node leaders (local_rank 0 only)
    if local_rank == 0 && !isnothing(leaders_comm)
        leaders_size = MPI.Comm_size(leaders_comm)
        leaders_rank = MPI.Comm_rank(leaders_comm)
        if leaders_size > 1
            synchronize(buf)
            step = 1
            while step < leaders_size
                if leaders_rank % (2 * step) == 0
                    partner = leaders_rank + step
                    if partner < leaders_size
                        MPI.Recv!(recvbuf, leaders_comm; source=partner, tag=100+step)
                        buf .+= recvbuf
                    end
                elseif leaders_rank % (2 * step) == step
                    synchronize(buf)
                    sendbuf = _ensure_buf!(_comm_sendbuf, buf, N)
                    copyto!(sendbuf, buf)
                    synchronize(buf)
                    MPI.Send(sendbuf, leaders_comm; dest=leaders_rank - step, tag=100+step)
                end
                step *= 2
            end
            # Broadcast back to all leaders (use pre-allocated buffers to avoid GPU registration accumulation)
            synchronize(buf)
            if leaders_rank == 0
                sendbuf = _ensure_buf!(_comm_sendbuf, buf, N)
                copyto!(sendbuf, buf)
                synchronize(buf)
                reqs = [MPI.Isend(sendbuf, leaders_comm; dest=r, tag=200) for r in 1:leaders_size-1]
                MPI.Waitall(reqs)
            else
                MPI.Recv!(recvbuf, leaders_comm; source=0, tag=200)
                copyto!(buf, recvbuf)
            end
        end
    end

    # Phase 3: Broadcast within node from local_rank 0 (use pre-allocated buffers)
    synchronize(buf)
    if local_size > 1
        if local_rank == 0
            sendbuf = _ensure_buf!(_comm_sendbuf, buf, N)
            copyto!(sendbuf, buf)
            synchronize(buf)
            reqs = [MPI.Isend(sendbuf, local_comm; dest=r, tag=300) for r in 1:local_size-1]
            MPI.Waitall(reqs)
        else
            MPI.Recv!(recvbuf, local_comm; source=0, tag=300)
            copyto!(buf, recvbuf)
        end
    end
    return buf
end

# ─── Boundary cast helpers for mixed-precision at the parallel/forloop level ───
#
# Strategy: cast args to `inner_etype` ONCE at function entry, run the whole
# for-loop body (kernel contractions + MPI gather) in that precision, cast the
# result back ONCE at function exit. This is equivalent to the per-kernel
# `inner_etype` threading but:
#  - does 1 down-cast + 1 up-cast per `parallel()` call (vs `2*forloop_iter`
#    at the kernel level),
#  - halves the MPI allgatherv/allreduce payload (Float32 = 2× bandwidth),
#  - halves the intermediate `result` VRAM footprint,
#  - keeps QR, eigsolve, norm and the outer VUMPS loop in native precision
#    (avoids the AD breakage that `whole_vumps_etype` exhibits).
_boundary_cast(::Nothing, a)            = a
_boundary_cast(T::Type, a::Tuple)       = map(t -> _downcast_eltype(T, t), a)
_boundary_cast(T::Type, a::StructArray) = _downcast_eltype(T, a)
_boundary_cast(T::Type, a::AbstractArray) = _downcast_eltype(T, a)

function forloop(f, args...; forloop_iter, N_in, N_out, size_out, inner_etype=nothing)
    T_orig = eltype(args[1])
    do_cast = inner_etype !== nothing && inner_etype != real(T_orig)
    if do_cast
        args = map(a -> _boundary_cast(inner_etype, a), args)
    end

    if forloop_iter == 1
        result = f(args...)
    else
        D_split = size(args[N_in[1]])[N_in[2]]
        result = similar(args[1], size_out)
        D_split_ranges = split_ranges(D_split, forloop_iter)

        for range in D_split_ranges
            cols_in = (j == N_in[2] ? range : (:) for j in 1:ndims(args[N_in[1]]))
            cols_out = (j == N_out ? range : (:) for j in 1: ndims(result))
            split_args = Tuple(j == N_in[1] ? @view(args[j][cols_in...]) : args[j] for j in 1:length(args))
            result[cols_out...] = f(split_args...)
        end
    end

    return do_cast ? T_orig.(result) : result
end

function parallel(f, args...; forloop_iter, N_in, N_out, size_out, inner_etype=nothing)
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    T_orig = eltype(args[1])
    do_cast = inner_etype !== nothing && inner_etype != real(T_orig)
    if do_cast
        args = map(a -> _boundary_cast(inner_etype, a), args)
    end

    D_split = size(args[N_in[1]])[N_in[2]]
    result = similar(args[1], size_out)
    D_split_ranges = split_ranges(D_split, nprocs*forloop_iter)

    for i in 1:forloop_iter
        ind = forloop_iter * rank + i
        cols_in = (j == N_in[2] ? D_split_ranges[ind] : (:) for j in 1:ndims(args[N_in[1]]))
        cols_out = (j == N_out ? D_split_ranges[ind] : (:) for j in 1: ndims(result))
        split_args = Tuple(j == N_in[1] ? @view(args[j][cols_in...]) : args[j] for j in 1:length(args))
        result[cols_out...] = f(split_args...)
        synchronize(args[1])
    end

    element_size = prod(size_out) ÷ D_split
    counts = Cint[sum([length(D_split_ranges[(i-1)*forloop_iter+j]) for j in 1:forloop_iter]) * element_size for i in 1:nprocs]
    allgatherv_p2p!(result, counts, comm)

    return do_cast ? T_orig.(result) : result
end


function forloop_sum(f, args...; forloop_iter, N_in1, N_in2, size_out, inner_etype=nothing)
    T_orig = eltype(args[1])
    do_cast = inner_etype !== nothing && inner_etype != real(T_orig)
    if do_cast
        args = map(a -> _boundary_cast(inner_etype, a), args)
    end

    if forloop_iter == 1
        result = f(args...)
    else
        D_split = size(args[N_in1[1]])[N_in1[2]]
        result = similar(args[1], size_out)
        result .= 0
        D_split_ranges = split_ranges(D_split, forloop_iter)

        for range in D_split_ranges
            cols_in1 = (j == N_in1[2] ? range : (:) for j in 1:ndims(args[N_in1[1]]))
            cols_in2 = (j == N_in2[2] ? range : (:) for j in 1:ndims(args[N_in2[1]]))
            split_args = (j == N_in1[1] ? @view(args[j][cols_in1...]) : (j == N_in2[1] ? @view(args[j][cols_in2...]) : args[j]) for j in 1:length(args))
            result .+= f(split_args...)
        end
    end

    return do_cast ? T_orig.(result) : result
end

function parallel_sum(f, args...; forloop_iter, N_in1, N_in2, size_out, inner_etype=nothing)
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    T_orig = eltype(args[1])
    do_cast = inner_etype !== nothing && inner_etype != real(T_orig)
    if do_cast
        args = map(a -> _boundary_cast(inner_etype, a), args)
    end

    D_split = size(args[N_in1[1]])[N_in1[2]]
    result = similar(args[1], size_out)
    result .= 0
    D_split_ranges = split_ranges(D_split, nprocs*forloop_iter)

    for i in 1:forloop_iter
        ind = forloop_iter * rank + i
        cols_in1 = (j == N_in1[2] ? D_split_ranges[ind] : (:) for j in 1:ndims(args[N_in1[1]]))
        cols_in2 = (j == N_in2[2] ? D_split_ranges[ind] : (:) for j in 1:ndims(args[N_in2[1]]))
        split_args = (j == N_in1[1] ? @view(args[j][cols_in1...]) : (j == N_in2[1] ? @view(args[j][cols_in2...]) : args[j]) for j in 1:length(args))
        result .+= f(split_args...)
    end

    allreduce_p2p!(result, +, comm)

    return do_cast ? T_orig.(result) : result
end

function FLmap_parallel(FL, ALu, ALd, M; ifparallel, forloop_iter, inner_etype=nothing)
    N_in = (3, ndims(ALd))
    N_out = ndims(ALd)
    χ = size(FL, 1)
    if M isa Tuple
        D1 = size(M[1], 3)
        D2 = size(M[2], 3)
        size_out = (χ,D1,D2,χ)
    elseif ndims(M) == 5
        D = size(M, 3)
        size_out = (χ,D,D,χ)
    else
        D = size(M, 3)
        size_out = (χ,D,χ)
    end
    # inner_etype is threaded to parallel/forloop (boundary cast), NOT to the
    # kernel. This reduces F64↔F32 conversion to once per parallel() call and
    # lets MPI allgatherv run in the lower precision (2× bandwidth).
    if ifparallel
        return parallel(FLmap, FL, ALu, ALd, M; forloop_iter, N_in, N_out, size_out, inner_etype)
    else
        return forloop(FLmap, FL, ALu, ALd, M; forloop_iter, N_in, N_out, size_out, inner_etype)
    end
end

function FRmap_parallel(FR, ARu, ARd, M; ifparallel, forloop_iter, inner_etype=nothing)
    N_in = (3, 1)
    N_out = ndims(ARd)
    χ = size(ARd, 1)
    if M isa Tuple
        D1 = size(M[1], 1)
        D2 = size(M[2], 1)
        size_out = (χ,D1,D2,χ)
    elseif ndims(M) == 5
        D = size(M, 1)
        size_out = (χ,D,D,χ)
    else
        D = size(M, 1)
        size_out = (χ,D,χ)
    end
    if ifparallel
        return parallel(FRmap, FR, ARu, ARd, M; forloop_iter, N_in, N_out, size_out, inner_etype)
    else
        return forloop(FRmap, FR, ARu, ARd, M; forloop_iter, N_in, N_out, size_out, inner_etype)
    end
end

function ACmap_parallel(AC, FL, FR, M; ifparallel, forloop_iter, inner_etype=nothing)
    N_in = (3, ndims(FR))
    N_out = ndims(FR)
    χ = size(FR, 1)
    if M isa Tuple
        D1 = size(M[1], 2)
        D2 = size(M[2], 2)
        size_out = (χ,D1,D2,χ)
    elseif ndims(M) == 5
        D = size(M, 2)
        size_out = (χ,D,D,χ)
    else
        D = size(M, 2)
        size_out = (χ,D,χ)
    end
    if ifparallel
        return parallel(ACmap, AC, FL, FR, M; forloop_iter, N_in, N_out, size_out, inner_etype)
    else
        return forloop(ACmap, AC, FL, FR, M; forloop_iter, N_in, N_out, size_out, inner_etype)
    end
end

function ACdmap_parallel(ACd, FL, FR, M; ifparallel, forloop_iter, inner_etype=nothing)
    N_in = (3, 1)
    N_out = ndims(FR)
    χ = size(FR, 1)
    if M isa Tuple
        D1 = size(M[1], 4)
        D2 = size(M[2], 4)
        size_out = (χ,D1,D2,χ)
    elseif ndims(M) == 5
        D = size(M, 4)
        size_out = (χ,D,D,χ)
    else
        D = size(M, 4)
        size_out = (χ,D,χ)
    end
    if ifparallel
        return parallel(ACdmap, ACd, FL, FR, M; forloop_iter, N_in, N_out, size_out, inner_etype)
    else
        return forloop(ACdmap, ACd, FL, FR, M; forloop_iter, N_in, N_out, size_out, inner_etype)
    end
end

function Mmap_parallel(AC, ACd, FL, FR; ifparallel, forloop_iter) 
    N_in1 = (2, 3)
    N_in2 = (4, 3)
    D1 = size(FL, 2)
    D2 = size(ACd, 2)
    D3 = size(FR, 2)
    D4 = size(AC, 2)
    size_out = (D1,D2,D3,D4)
    if ifparallel
        return parallel_sum(Mmap, AC, ACd, FL, FR; forloop_iter, N_in1, N_in2, size_out)
    else
        return forloop_sum(Mmap, AC, ACd, FL, FR; forloop_iter, N_in1, N_in2, size_out)
    end
end

function Mumap_parallel(AC, ACd, FL, FR, Mu; ifparallel, forloop_iter) 
    N_in1 = (2, 4)
    N_in2 = (4, 4)
    D1 = size(FL, 3)
    D2 = size(ACd, 3)
    D3 = size(FR, 3)
    D4 = size(AC, 3)
    
    d = size(Mu, 5)
    size_out = (D1,D2,D3,D4,d)
    if ifparallel
        return parallel_sum(Mumap, AC, ACd, FL, FR, Mu; forloop_iter, N_in1, N_in2, size_out)
    else
        return forloop_sum(Mumap, AC, ACd, FL, FR, Mu; forloop_iter, N_in1, N_in2, size_out)
    end
end

function Mdmap_parallel(AC, ACd, FL, FR, Md; ifparallel, forloop_iter) 
    N_in1 = (2, 4)
    N_in2 = (4, 4)
    D1 = size(FL, 2)
    D2 = size(ACd, 2)
    D3 = size(FR, 2)
    D4 = size(AC, 2)
    d = size(Md, 5)
    size_out = (D1,D2,D3,D4,d)
    if ifparallel
        return parallel_sum(Mdmap, AC, ACd, FL, FR, Md; forloop_iter, N_in1, N_in2, size_out)
    else
        return forloop_sum(Mdmap, AC, ACd, FL, FR, Md; forloop_iter, N_in1, N_in2, size_out)
    end
end