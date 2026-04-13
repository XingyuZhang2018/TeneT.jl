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

"""
    allgatherv_p2p!(buf, counts, comm)

Drop-in replacement for `MPI.Allgatherv!(VBuffer(buf, counts), comm)` using
non-blocking point-to-point (Isend/Irecv).  Works around the poor GPU-collective
performance of Open MPI 4.x Allgatherv on CUDA buffers.
"""
function allgatherv_p2p!(buf, counts, comm)
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)
    nprocs == 1 && return buf
    synchronize(buf)
    displs = cumsum([0; counts[1:end-1]])
    reqs = MPI.Request[]
    for r in 0:nprocs-1
        r == rank && continue
        my_view   = view(buf, displs[rank+1]+1 : displs[rank+1]+counts[rank+1])
        recv_view = view(buf, displs[r+1]+1    : displs[r+1]+counts[r+1])
        push!(reqs, MPI.Isend(my_view, comm; dest=r, tag=rank))
        push!(reqs, MPI.Irecv!(recv_view, comm; source=r, tag=r))
    end
    MPI.Waitall(reqs)
    return buf
end

# ─── NCCL-based Allreduce ─────────────────────────────────────────────────

using Libdl

const _nccl_state = Ref{Any}(nothing)  # (nccl_comm, libnccl_handle)

const _nccl_dtype = Dict(
    Float64 => Cint(8),  # ncclFloat64
    Float32 => Cint(7),  # ncclFloat32
    Float16 => Cint(6),  # ncclFloat16
)

function _init_nccl(mpi_comm)
    isnothing(_nccl_state[]) || return _nccl_state[]

    # Find NCCL library
    nccl_path = get(ENV, "NCCL_LIB_PATH", "")
    if isempty(nccl_path)
        for p in ["/apps/ACC/NCCL/2.24.3-1/lib/libnccl.so",
                  "/apps/ACC/NCCL/2.20.5/lib/libnccl.so",
                  "libnccl.so"]
            isfile(p) && (nccl_path = p; break)
        end
    end
    lib = dlopen(nccl_path)

    rank = MPI.Comm_rank(mpi_comm)
    nprocs = MPI.Comm_size(mpi_comm)

    # Get unique ID and broadcast
    uid = Ref{NTuple{128, UInt8}}(ntuple(_ -> UInt8(0), 128))
    rank == 0 && ccall(dlsym(lib, :ncclGetUniqueId), Cint, (Ref{NTuple{128,UInt8}},), uid)
    MPI.Bcast!(uid, 0, mpi_comm)

    # Init communicator
    nccl_comm = Ref{Ptr{Cvoid}}(C_NULL)
    ret = ccall(dlsym(lib, :ncclCommInitRank), Cint,
        (Ref{Ptr{Cvoid}}, Cint, NTuple{128,UInt8}, Cint),
        nccl_comm, nprocs, uid[], rank)
    ret != 0 && error("ncclCommInitRank failed: ret=$ret")

    _nccl_state[] = (comm=nccl_comm[], lib=lib)
    return _nccl_state[]
end

"""
    allreduce_p2p!(buf, +, mpi_comm)

GPU-native Allreduce using NCCL. Falls back to CPU staging if NCCL
is unavailable or buffer type is not supported.
"""
function allreduce_p2p!(buf, ::typeof(+), mpi_comm)
    P = MPI.Comm_size(mpi_comm)
    P == 1 && return buf

    T = eltype(buf)
    if haskey(_nccl_dtype, T)
        nccl = try _init_nccl(mpi_comm) catch; nothing end
        if !isnothing(nccl)
            dtype = _nccl_dtype[T]
            synchronize(buf)  # ensure all pending GPU ops on buf complete before NCCL reads it
            buf_ptr = reinterpret(Ptr{Cvoid}, pointer(buf))
            ccall(dlsym(nccl.lib, :ncclAllReduce), Cint,
                (Ptr{Cvoid}, Ptr{Cvoid}, Csize_t, Cint, Cint, Ptr{Cvoid}, Ptr{Cvoid}),
                buf_ptr, buf_ptr, length(buf), dtype, Cint(0), nccl.comm, C_NULL)
            synchronize(buf)  # ensure NCCL write completes before downstream reads
            return buf
        end
    end

    # Fallback: CPU staging
    cpu_buf = Array(buf)
    MPI.Allreduce!(cpu_buf, +, mpi_comm)
    copyto!(buf, cpu_buf)
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

    MPI.Allreduce!(result, +, comm)

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