function forloop(f, args...; forloop_iter, N_in, N_out, size_out)
    if forloop_iter == 1
        return f(args...)
    else
        D_split = size(args[N_in[1]])[N_in[2]]
        result = similar(args[1], size_out)
        D_split_loop = cld(D_split, forloop_iter)
        D_split_ranges = [range(1 + (i-1)*D_split_loop, min(i*D_split_loop, D_split)) for i in 1:forloop_iter]

        # GPUArrays.@cached GPUArrays.AllocCache() begin
            for range in D_split_ranges
                cols_in = (j == N_in[2] ? range : (:) for j in 1:ndims(args[N_in[1]]))
                cols_out = (j == N_out ? range : (:) for j in 1: ndims(result))
                split_args = (j == N_in[1] ? args[j][cols_in...] : args[j] for j in 1:length(args))
                result[cols_out...] = f(split_args...)
            end
        # end
        return result
    end
end

function parallel(f, args...; forloop_iter, N_in, N_out, size_out)
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    D_split = size(args[N_in[1]])[N_in[2]]
    result = similar(args[1], size_out)
    D_split_loop = cld(D_split, nprocs*forloop_iter)
    D_split_ranges = [range(1 + (i-1)*D_split_loop, min(i*D_split_loop, D_split)) for i in 1:nprocs*forloop_iter]

    # GPUArrays.@cached GPUArrays.AllocCache() begin
        for i in 1:forloop_iter
            ind = forloop_iter * rank + i
            cols_in = (j == N_in[2] ? D_split_ranges[ind] : (:) for j in 1:ndims(args[N_in[1]]))
            cols_out = (j == N_out ? D_split_ranges[ind] : (:) for j in 1: ndims(result))
            split_args = (j == N_in[1] ? args[j][cols_in...] : args[j] for j in 1:length(args))
            result[cols_out...] = f(split_args...)
            synchronize(args[1])
        end

        element_size = prod(size_out) ÷ D_split
        counts = Cint[sum([length(D_split_ranges[(i-1)*forloop_iter+j]) for j in 1:forloop_iter]) * element_size for i in 1:nprocs]
        MPI.Allgatherv!(VBuffer(result, counts), comm)
    # end

    # for root in 0:(nprocs-1)
    #     if root == rank
    #         MPI.Gatherv!(MPI.IN_PLACE, VBuffer(result, counts), comm; root=root)
    #     else
    #         MPI.Gatherv!(result[cols..., D_split_ranges[rank+1]], nothing, comm; root=root)
    #     end
    #     MPI.Barrier(comm)
    # end
    return result
end


function forloop_sum(f, args...; forloop_iter, N_in1, N_in2, size_out)
    if forloop_iter == 1
        return f(args...)
    else
        D_split = size(args[N_in1[1]])[N_in1[2]]
        result = similar(args[1], size_out)
        result .= 0
        D_split_loop = cld(D_split, forloop_iter)
        D_split_ranges = [range(1 + (i-1)*D_split_loop, min(i*D_split_loop, D_split)) for i in 1:forloop_iter]

        # GPUArrays.@cached GPUArrays.AllocCache() begin
            for range in D_split_ranges
                cols_in1 = (j == N_in1[2] ? range : (:) for j in 1:ndims(args[N_in1[1]]))
                cols_in2 = (j == N_in2[2] ? range : (:) for j in 1:ndims(args[N_in2[1]]))
                split_args = (j == N_in1[1] ? args[j][cols_in1...] : (j == N_in2[1] ? args[j][cols_in2...] : args[j]) for j in 1:length(args))
                result += f(split_args...)
            end
        # end

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
    D_split_loop = cld(D_split, nprocs*forloop_iter)
    D_split_ranges = [range(1 + (i-1)*D_split_loop, min(i*D_split_loop, D_split)) for i in 1:nprocs*forloop_iter]

    # GPUArrays.@cached GPUArrays.AllocCache() begin
        for i in 1:forloop_iter
            ind = forloop_iter * rank + i
            cols_in1 = (j == N_in1[2] ? D_split_ranges[ind] : (:) for j in 1:ndims(args[N_in1[1]]))
            cols_in2 = (j == N_in2[2] ? D_split_ranges[ind] : (:) for j in 1:ndims(args[N_in2[1]]))
            split_args = (j == N_in1[1] ? args[j][cols_in1...] : (j == N_in2[1] ? args[j][cols_in2...] : args[j]) for j in 1:length(args))
            result += f(split_args...)
            synchronize(args[1])
        end

        MPI.Allreduce!(result, +, comm)
    # end

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