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

# ─── Hierarchical Allgatherv ──────────────────────────────────────────────

"""
    allgatherv_p2p!(buf, counts, comm)

Hierarchical allgatherv: intra-node p2p (NVLink) + inter-node p2p (IB),
using pre-allocated send buffer to avoid GPU registration accumulation.
"""
function allgatherv_p2p!(buf, counts, comm)
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)
    nprocs == 1 && return buf
    synchronize(buf)
    displs = cumsum([0; counts[1:end-1]])
    my_count = counts[rank+1]
    max_count = maximum(counts)

    sendbuf = _ensure_buf!(_comm_sendbuf, buf, my_count)
    copyto!(sendbuf, view(buf, displs[rank+1]+1 : displs[rank+1]+my_count))
    synchronize(buf)

    # Use pre-allocated recvbuf to avoid GPU registration accumulation on buf
    _ensure_buf!(_comm_recvbuf, buf, max_count)

    # Send to all peers (non-blocking from fixed sendbuf)
    send_reqs = MPI.Request[]
    for r in 0:nprocs-1
        r == rank && continue
        push!(send_reqs, MPI.Isend(sendbuf, comm; dest=r, tag=rank))
    end

    # Receive from each peer into pre-allocated recvbuf, then copy to buf
    for r in 0:nprocs-1
        r == rank && continue
        rc = counts[r+1]
        recvview = view(_comm_recvbuf[], 1:rc)
        MPI.Recv!(recvview, comm; source=r, tag=r)
        copyto!(view(buf, displs[r+1]+1 : displs[r+1]+rc), recvview)
    end

    MPI.Waitall(send_reqs)
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

function forloop(f, args...; forloop_iter, N_in, N_out, size_out)
    if forloop_iter == 1
        return f(args...)
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

        return result
    end
end

function parallel(f, args...; forloop_iter, N_in, N_out, size_out)
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

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

    return result
end


function forloop_sum(f, args...; forloop_iter, N_in1, N_in2, size_out)
    if forloop_iter == 1
        return f(args...)
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

        return result
    end
end

function parallel_sum(f, args...; forloop_iter, N_in1, N_in2, size_out)
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

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
        synchronize(args[1])
    end

    allreduce_p2p!(result, +, comm)

    return result
end

function FLmap_parallel(FL, ALu, ALd, M; ifparallel, forloop_iter) 
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
    if ifparallel
        return parallel(FLmap, FL, ALu, ALd, M; forloop_iter, N_in, N_out, size_out)
    else
        return forloop(FLmap, FL, ALu, ALd, M; forloop_iter, N_in, N_out, size_out)
    end
end

function FRmap_parallel(FR, ARu, ARd, M; ifparallel, forloop_iter) 
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
        return parallel(FRmap, FR, ARu, ARd, M; forloop_iter, N_in, N_out, size_out)
    else
        return forloop(FRmap, FR, ARu, ARd, M; forloop_iter, N_in, N_out, size_out)
    end
end

function ACmap_parallel(AC, FL, FR, M; ifparallel, forloop_iter) 
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
        return parallel(ACmap, AC, FL, FR, M; forloop_iter, N_in, N_out, size_out)
    else
        return forloop(ACmap, AC, FL, FR, M; forloop_iter, N_in, N_out, size_out)
    end
end

function ACdmap_parallel(ACd, FL, FR, M; ifparallel, forloop_iter) 
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
        return parallel(ACdmap, ACd, FL, FR, M; forloop_iter, N_in, N_out, size_out)
    else
        return forloop(ACdmap, ACd, FL, FR, M; forloop_iter, N_in, N_out, size_out)
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