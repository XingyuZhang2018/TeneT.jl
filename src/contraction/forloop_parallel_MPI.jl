include("nccl_wrapper.jl")

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
const _comm_hostbuf = Ref{Any}(nothing)   # CPU staging buf for host-staged Phase 2
const _comm_local = Ref{Any}(nothing)     # node-local communicator
const _comm_leaders = Ref{Any}(nothing)   # inter-node leaders communicator
const _comm_siblings = Ref{Any}(nothing)  # inter-node comm split by local_rank

# MPI tag namespace for internal p2p collectives. Keep well above any tag the
# caller might use (current callers use tag = rank ∈ 0..nprocs-1).
const _TAG_BASE = 1000

# Opt-in host staging for cross-node Phase 2 Allreduce. On systems where the
# UCX CUDA-IB rendezvous is slower than the pinned-host path (e.g. Sofia H200:
# 40 ms vs 11 ms for a 15 MB slice), set `TENET_MPI_HOST_STAGE=1` in the env
# to route Phase 2 through D2H → system MPI.Allreduce! on host → H2D. Intra-
# node Phase 1/3 stay on-device (cuda_ipc over NVLink). Default is off so
# GDR-capable systems (JSC GH200, BSC H100) keep the GPU-resident path.
# Read per-call (not precompile-const) so ENV changes apply in fresh processes.
_host_stage_rndv() = get(ENV, "TENET_MPI_HOST_STAGE", "0") == "1"

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

    # NCCL fast path: if all per-rank counts are equal, one ncclAllGather
    # replaces the 3-phase flow. Variable-length layouts fall through to
    # the p2p ring since NCCL has no native Allgatherv. Stream FIFO handles
    # ordering — copyto! + ncclAllGather + downstream ops all chain on
    # CUDA.stream() without explicit syncs.
    if _use_nccl() && buf isa CuArray && all(==(counts[1]), counts)
        c = counts[1]
        my_off = c * rank
        sendbuf = _ensure_buf!(_comm_sendbuf, buf, c)
        copyto!(sendbuf, view(buf, my_off+1 : my_off+c))
        return _nccl_allgather_equal!(sendbuf, buf, comm)
    end

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

Bandwidth-optimal hierarchical allreduce via three phases:

1. Intra-node ring reduce-scatter: walks a ring of `local_size` steps,
   accumulating slices of `buf` so each local rank ends up owning the
   intra-node sum on a unique slice.
2. Cross-node ring allreduce per local_rank sibling communicator: the slice
   owned by each rank gets summed across nodes. Degenerates to a no-op when
   `n_nodes == 1`.
3. Intra-node ring allgather in the opposite ring direction: rotates the
   final slice ownership around so every rank ends up with the full reduced
   buffer.

Reduction order differs from a binary tree — machine-eps drift on complex
sums is expected and within the 1e-10 relative tolerance the codebase assumes.
"""
function allreduce_p2p!(buf, ::typeof(+), comm)
    rank = MPI.Comm_rank(comm)
    P    = MPI.Comm_size(comm)
    P == 1 && return buf

    # NCCL fast path: single ncclAllReduce handles intra+inter node hierarchy
    # optimally (GDR over IB + NVLink). On Sofia H200: 1.5 ms vs 44 ms 3-phase.
    # Stream FIFO on CUDA.stream() handles the before-after ordering so no
    # CUDA.synchronize is required around the call.
    if _use_nccl() && buf isa CuArray
        return _nccl_allreduce!(buf, comm)
    end

    synchronize(buf)
    N = length(buf)

    local_comm, _ = _get_local_comm(comm)
    local_rank = MPI.Comm_rank(local_comm)
    local_size = MPI.Comm_size(local_comm)
    n_nodes    = P ÷ local_size
    @assert P == n_nodes * local_size "allreduce_p2p! requires symmetric node layout; got P=$P, local_size=$local_size"

    # Slice buf into `local_size` ~equal pieces. The N-mod-local_size "stragglers"
    # get 1 extra element each (via split_count), matching the existing helper.
    slice_counts = split_count(N, local_size)
    slice_displs = cumsum([0; slice_counts[1:end-1]])
    slice_range(i) = (slice_displs[i] + 1) : (slice_displs[i] + slice_counts[i])

    prev_l, next_l = _ring_neighbors(local_rank, local_size)

    # ── Phase 1: intra-node ring reduce-scatter (send to next, recv from prev) ──
    if local_size > 1
        _ensure_buf!(_comm_recvbuf, buf, maximum(slice_counts))
        for step in 1:local_size-1
            send_idx = mod(local_rank - step + 1, local_size) + 1
            recv_idx = mod(local_rank - step,     local_size) + 1
            send_sub = view(buf, slice_range(send_idx))
            recv_sub = view(_comm_recvbuf[], 1:slice_counts[recv_idx])
            synchronize(buf)
            req_send = MPI.Isend(send_sub,  local_comm; dest=next_l,   tag=_TAG_BASE + 300 + step)
            req_recv = MPI.Irecv!(recv_sub, local_comm; source=prev_l, tag=_TAG_BASE + 300 + step)
            MPI.Waitall([req_send, req_recv])
            view(buf, slice_range(recv_idx)) .+= recv_sub
        end
    end

    # After Phase 1, rank r owns the intra-node sum on slice my_slice.
    my_slice = mod(local_rank + 1, local_size) + 1

    # ── Phase 2: cross-node ring allreduce on the owned slice ──
    if n_nodes > 1
        sib_comm = _per_local_rank_comm(comm, local_comm)
        sib_size = MPI.Comm_size(sib_comm)
        if sib_size > 1
            sib_rank = MPI.Comm_rank(sib_comm)
            slice_view = view(buf, slice_range(my_slice))
            if _host_stage_rndv()
                _allreduce_host_staged!(slice_view, sib_comm)
            else
                _allreduce_ring_on_slice!(slice_view, sib_comm, sib_rank, sib_size)
            end
        end
    end

    # ── Phase 3: intra-node ring allgather (send to prev, recv from next) ──
    if local_size > 1
        for step in 1:local_size-1
            send_idx = mod(my_slice - 1 + step - 1, local_size) + 1
            recv_idx = mod(my_slice - 1 + step,     local_size) + 1
            send_sub = view(buf, slice_range(send_idx))
            recv_sub = view(buf, slice_range(recv_idx))
            synchronize(buf)
            req_send = MPI.Isend(send_sub,  local_comm; dest=prev_l,   tag=_TAG_BASE + 400 + step)
            req_recv = MPI.Irecv!(recv_sub, local_comm; source=next_l, tag=_TAG_BASE + 400 + step)
            MPI.Waitall([req_send, req_recv])
        end
    end

    return buf
end

# Phase 2 helper: ring allreduce on a contiguous slice that every rank in
# `comm` owns an independent copy of. RS direction: send next, recv prev;
# AG direction: send prev, recv next (same ring-flip trick as the main
# allreduce). For `size_ == 2` this degenerates to a single swap+sum.
function _allreduce_ring_on_slice!(slice, comm, rank, size_)
    size_ == 1 && return slice
    N = length(slice)
    sub_counts = split_count(N, size_)
    sub_displs = cumsum([0; sub_counts[1:end-1]])
    sub_range(i) = (sub_displs[i] + 1) : (sub_displs[i] + sub_counts[i])
    prev, next_ = _ring_neighbors(rank, size_)

    _ensure_buf!(_comm_recvbuf, slice, maximum(sub_counts))

    # Reduce-scatter
    for step in 1:size_-1
        send_idx = mod(rank - step + 1, size_) + 1
        recv_idx = mod(rank - step,     size_) + 1
        send_sub = view(slice, sub_range(send_idx))
        recv_sub = view(_comm_recvbuf[], 1:sub_counts[recv_idx])
        synchronize(slice)
        req_send = MPI.Isend(send_sub,  comm; dest=next_, tag=_TAG_BASE + 500 + step)
        req_recv = MPI.Irecv!(recv_sub, comm; source=prev, tag=_TAG_BASE + 500 + step)
        MPI.Waitall([req_send, req_recv])
        view(slice, sub_range(recv_idx)) .+= recv_sub
    end
    my_sub = mod(rank + 1, size_) + 1

    # Allgather (opposite ring direction)
    for step in 1:size_-1
        send_idx = mod(my_sub - 1 + step - 1, size_) + 1
        recv_idx = mod(my_sub - 1 + step,     size_) + 1
        send_sub = view(slice, sub_range(send_idx))
        recv_sub = view(slice, sub_range(recv_idx))
        synchronize(slice)
        req_send = MPI.Isend(send_sub,  comm; dest=prev,  tag=_TAG_BASE + 600 + step)
        req_recv = MPI.Irecv!(recv_sub, comm; source=next_, tag=_TAG_BASE + 600 + step)
        MPI.Waitall([req_send, req_recv])
    end

    return slice
end

# Host-staged Phase 2: D2H → system MPI.Allreduce on pinned host memory → H2D.
# The cross-node step goes through the IB driver's native host path, which on
# Sofia H200 is ~4× faster than UCX's GPU-direct rendezvous at ≥16 MB slice.
# Intra-node Phase 1/3 of allreduce_p2p! keep using the cuda_ipc ring.
function _allreduce_host_staged!(slice, comm)
    MPI.Comm_size(comm) == 1 && return slice
    N = length(slice)
    T = eltype(slice)
    # Exact-size reuse: CUDA.jl's copyto!(Vector, SubArray{CuArray}) isn't
    # defined, so we copy in positional form on the full Vector.
    if isnothing(_comm_hostbuf[]) || length(_comm_hostbuf[]) != N || eltype(_comm_hostbuf[]) != T
        _comm_hostbuf[] = Vector{T}(undef, N)
    end
    hbuf = _comm_hostbuf[]::Vector{T}
    synchronize(slice)                  # Phase 1 `.+=` done
    copyto!(hbuf, 1, slice, 1, N)       # D2H
    synchronize(slice)                  # ensure transfer visible to host
    MPI.Allreduce!(hbuf, +, comm)       # system MPI on host, IB-native
    copyto!(slice, 1, hbuf, 1, N)       # H2D
    synchronize(slice)
    return slice
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

# ─── 2D distributed VUMPS primitives ──────────────────────────────────────
#
# Phase 1 of the 2D distributed VUMPS port (`docs/2026-05-11-2d-distributed-
# vumps-runtime-design.md`). `allgather_dim` is the first primitive: along a
# Cartesian sub-communicator (row_comm or col_comm of a `Cart2DGrid`), it
# materialises the full extent of one tensor dimension that the caller knows
# is distributed equally across the ranks of `comm`.
#
# Design call: dim != ndims via local permutedims dance
# ----------------------------------------------------
# `allgatherv_p2p!` flattens its receive buffer to a linear-index view and
# treats each rank's contribution as a contiguous range `displs[k]+1 :
# displs[k]+counts[k]`. In Julia's column-major layout, a slab placed at
# `result[..., local_range]` (last-dim range) maps to one contiguous linear
# range exactly; a slab placed at `result[local_range, ...]` (first-dim
# range) does NOT — the bytes from each rank would be strided across the
# linear buffer and remote contributions would land interleaved.
#
# The sandbox `sandbox/2d_allgather_sanity.jl` confirmed this empirically by
# choosing a 3D recv buffer `(χ_local, χ_local, N1)` over the originally
# proposed 2D `(χ_local·N1, χ_local)` precisely so each rank's chunk stays
# contiguous (see file header rationale).
#
# Section 2 of the design doc, however, requires both first-dim (FLmap `i`,
# FRmap `d`) and last-dim (ACmap `d`, Cmap `e`/`f`) gathers, so a last-dim-
# only API would force ad-hoc dances into every caller. Instead we centralise
# the dance here: when `dim != ndims(tensor_local)`, we
#   1. permute `dim ↔ ndims` to a temporary,
#   2. allgather on the (now last) dim via the cheap direct path,
#   3. permute back.
# Cost: two extra contiguous reads of the local + full tensor. Acceptable
# because allgather is bandwidth-bound on `allgatherv_p2p!`, not allocation-
# bound, and the maps that wrap this primitive are dominated by the
# subsequent einsum.

"""
    allgather_dim(tensor_local, dim, comm) -> tensor_full

Collect `tensor_local`'s `dim`-th dimension across all ranks in `comm`,
producing a tensor with the full extent on that dimension. Other dimensions
are unchanged.

Equal-size partition only: every rank must contribute the same `size(tensor
_local, dim)`. (If uneven slices become needed downstream, add a `counts`-
taking overload that calls `split_ranges`; do not modify this method.)

Implementation notes:

* If `MPI.Comm_size(comm) == 1`, returns `copy(tensor_local)` — a fresh array,
  never an alias of the input. The `copy` matters for the rrule (Task 1.4):
  mutation of the local-shape forward output by downstream code must not be
  visible through the input tangent.
* When `dim == ndims(tensor_local)` the data is placed at the last-dim slab
  of `result` and handed to `allgatherv_p2p!` directly — one contiguous range
  per rank, optimal.
* When `dim != ndims(tensor_local)`, the function permutes `dim ↔ ndims`,
  runs the direct path on the transposed tensor, then permutes back. This
  costs two extra contiguous reads but keeps the API uniform.

Used by the 2D distributed maps (Phase 2+) to gather distributed boundary
tensors along col_comm / row_comm before local einsum contractions.
"""
function allgather_dim(tensor_local::AbstractArray{T,N}, dim::Int, comm) where {T,N}
    1 <= dim <= N || throw(ArgumentError("dim=$dim out of range for $(N)D tensor"))
    M = MPI.Comm_size(comm)
    M == 1 && return copy(tensor_local)

    # Non-last-dim case: permute `dim ↔ N`, gather, permute back.
    # See module-header note for rationale (column-major contiguity).
    if dim != N
        perm = ntuple(d -> d == dim ? N : (d == N ? dim : d), N)
        permuted = permutedims(tensor_local, perm)
        gathered = allgather_dim(permuted, N, comm)
        return permutedims(gathered, perm)
    end

    # Direct path: dim == N (last dim, contiguous slabs).
    rank   = MPI.Comm_rank(comm)
    chi_local = size(tensor_local, dim)
    chi_full  = M * chi_local
    full_shape = ntuple(d -> d == dim ? chi_full : size(tensor_local, d), N)

    result = similar(tensor_local, full_shape)
    local_range = (rank * chi_local + 1):((rank + 1) * chi_local)
    idx = ntuple(d -> d == dim ? local_range : Colon(), N)
    result[idx...] = tensor_local
    synchronize(result)  # ensure the write is visible before MPI inspects buf

    elem_count = prod(size(tensor_local))
    counts = Cint[elem_count for _ in 1:M]
    allgatherv_p2p!(result, counts, comm)
    return result
end