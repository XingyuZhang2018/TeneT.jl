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
            isempty(range) && continue
            cols_in = (j == N_in[2] ? range : (:) for j in 1:ndims(args[N_in[1]]))
            cols_out = (j == N_out ? range : (:) for j in 1: ndims(result))
            split_args = Tuple(j == N_in[1] ? @view(args[j][cols_in...]) : args[j] for j in 1:length(args))
            result[cols_out...] = f(split_args...)
        end
    end

    return do_cast ? T_orig.(result) : result
end

function parallel(f, args...; forloop_iter, N_in, N_out, size_out, inner_etype=nothing, comm=MPI.COMM_WORLD)
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
        isempty(D_split_ranges[ind]) && continue
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
            isempty(range) && continue
            cols_in1 = (j == N_in1[2] ? range : (:) for j in 1:ndims(args[N_in1[1]]))
            cols_in2 = (j == N_in2[2] ? range : (:) for j in 1:ndims(args[N_in2[1]]))
            split_args = (j == N_in1[1] ? @view(args[j][cols_in1...]) : (j == N_in2[1] ? @view(args[j][cols_in2...]) : args[j]) for j in 1:length(args))
            result .+= f(split_args...)
        end
    end

    return do_cast ? T_orig.(result) : result
end

function parallel_sum(f, args...; forloop_iter, N_in1, N_in2, size_out, inner_etype=nothing, comm=MPI.COMM_WORLD)
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
        isempty(D_split_ranges[ind]) && continue
        cols_in1 = (j == N_in1[2] ? D_split_ranges[ind] : (:) for j in 1:ndims(args[N_in1[1]]))
        cols_in2 = (j == N_in2[2] ? D_split_ranges[ind] : (:) for j in 1:ndims(args[N_in2[1]]))
        split_args = (j == N_in1[1] ? @view(args[j][cols_in1...]) : (j == N_in2[1] ? @view(args[j][cols_in2...]) : args[j]) for j in 1:length(args))
        result .+= f(split_args...)
    end

    allreduce_p2p!(result, +, comm)

    return do_cast ? T_orig.(result) : result
end

function FLmap_parallel(FL, ALu, ALd, M; ifparallel, forloop_iter, inner_etype=nothing, comm=MPI.COMM_WORLD)
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
        return parallel(FLmap, FL, ALu, ALd, M; forloop_iter, N_in, N_out, size_out, inner_etype, comm)
    else
        return forloop(FLmap, FL, ALu, ALd, M; forloop_iter, N_in, N_out, size_out, inner_etype)
    end
end

function FRmap_parallel(FR, ARu, ARd, M; ifparallel, forloop_iter, inner_etype=nothing, comm=MPI.COMM_WORLD)
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
        return parallel(FRmap, FR, ARu, ARd, M; forloop_iter, N_in, N_out, size_out, inner_etype, comm)
    else
        return forloop(FRmap, FR, ARu, ARd, M; forloop_iter, N_in, N_out, size_out, inner_etype)
    end
end

function ACmap_parallel(AC, FL, FR, M; ifparallel, forloop_iter, inner_etype=nothing, comm=MPI.COMM_WORLD)
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
        return parallel(ACmap, AC, FL, FR, M; forloop_iter, N_in, N_out, size_out, inner_etype, comm)
    else
        return forloop(ACmap, AC, FL, FR, M; forloop_iter, N_in, N_out, size_out, inner_etype)
    end
end

function ACdmap_parallel(ACd, FL, FR, M; ifparallel, forloop_iter, inner_etype=nothing, comm=MPI.COMM_WORLD)
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
        return parallel(ACdmap, ACd, FL, FR, M; forloop_iter, N_in, N_out, size_out, inner_etype, comm)
    else
        return forloop(ACdmap, ACd, FL, FR, M; forloop_iter, N_in, N_out, size_out, inner_etype)
    end
end

function Mmap_parallel(AC, ACd, FL, FR; ifparallel, forloop_iter, comm=MPI.COMM_WORLD)
    N_in1 = (2, 3)
    N_in2 = (4, 3)
    D1 = size(FL, 2)
    D2 = size(ACd, 2)
    D3 = size(FR, 2)
    D4 = size(AC, 2)
    size_out = (D1,D2,D3,D4)
    if ifparallel
        return parallel_sum(Mmap, AC, ACd, FL, FR; forloop_iter, N_in1, N_in2, size_out, comm)
    else
        return forloop_sum(Mmap, AC, ACd, FL, FR; forloop_iter, N_in1, N_in2, size_out)
    end
end

function Mumap_parallel(AC, ACd, FL, FR, Mu; ifparallel, forloop_iter, comm=MPI.COMM_WORLD)
    N_in1 = (2, 4)
    N_in2 = (4, 4)
    D1 = size(FL, 3)
    D2 = size(ACd, 3)
    D3 = size(FR, 3)
    D4 = size(AC, 3)

    d = size(Mu, 5)
    size_out = (D1,D2,D3,D4,d)
    if ifparallel
        return parallel_sum(Mumap, AC, ACd, FL, FR, Mu; forloop_iter, N_in1, N_in2, size_out, comm)
    else
        return forloop_sum(Mumap, AC, ACd, FL, FR, Mu; forloop_iter, N_in1, N_in2, size_out)
    end
end

function Mdmap_parallel(AC, ACd, FL, FR, Md; ifparallel, forloop_iter, comm=MPI.COMM_WORLD)
    N_in1 = (2, 4)
    N_in2 = (4, 4)
    D1 = size(FL, 2)
    D2 = size(ACd, 2)
    D3 = size(FR, 2)
    D4 = size(AC, 2)
    d = size(Md, 5)
    size_out = (D1,D2,D3,D4,d)
    if ifparallel
        return parallel_sum(Mdmap, AC, ACd, FL, FR, Md; forloop_iter, N_in1, N_in2, size_out, comm)
    else
        return forloop_sum(Mdmap, AC, ACd, FL, FR, Md; forloop_iter, N_in1, N_in2, size_out)
    end
end
