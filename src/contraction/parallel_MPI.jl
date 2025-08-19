function FLmap_parallel(FL, ALu, ALd, M; ifcheckpoint, forloop_iter, ifparallel)
    if ifparallel
        comm = MPI.COMM_WORLD
        rank = MPI.Comm_rank(comm)
        nprocs = MPI.Comm_size(comm)
        χ = size(FL, 1)
        χ_slice = cld(χ, nprocs*forloop_iter)
        χ_ranges = [range(1 + (i-1)*χ_slice, min(i*χ_slice, χ)) for i in 1:nprocs*forloop_iter]
        cols = fill(:, ndims(FL)-1)
        if length(M) == 2
            D1 = size(M[1], 3)
            D2 = size(M[2], 3)
            FLm = similar(FL, χ,D1,D2,χ)
        elseif ndims(M) == 5
            D = size(M, 3)
            FLm = similar(FL, χ,D,D,χ)
        else
            D = size(M, 3)
            FLm = similar(FL, χ,D,χ)
        end

        for i in 1:forloop_iter
            ind = forloop_iter * rank + i
            FLm[cols..., χ_ranges[ind]] = FLmap(FL, ALu, ALd[cols..., χ_ranges[ind]], M)
            synchronize(FL)
        end
        
        element_size = prod(size(FLm)[1:end-1])
        counts = Cint[sum([length(χ_ranges[(i-1)*forloop_iter+j]) for j in 1:forloop_iter]) * element_size for i in 1:nprocs]
        MPI.Allgatherv!(VBuffer(FLm, counts), comm)
        # for root in 0:(nprocs-1)
        #     if root == rank
        #         MPI.Gatherv!(MPI.IN_PLACE, VBuffer(FLm, counts), comm; root=root)
        #     else
        #         MPI.Gatherv!(FLm[cols..., χ_ranges[rank+1]], nothing, comm; root=root)
        #     end
        #     MPI.Barrier(comm)
        # end
        return FLm
    else
        return FLmap_forloop(FL, ALu, ALd, M; forloop_iter)
    end
end

function FRmap_parallel(FR, ARu, ARd, M; ifcheckpoint, forloop_iter, ifparallel)
    if ifparallel
        comm = MPI.COMM_WORLD
        rank = MPI.Comm_rank(comm)
        nprocs = MPI.Comm_size(comm)
        
        χ = size(FR, 1)
        χ_slice = cld(χ, nprocs*forloop_iter)
        χ_ranges = [range(1 + (i-1)*χ_slice, min(i*χ_slice, χ)) for i in 1:nprocs*forloop_iter]
        cols = fill(:, ndims(FR)-1)
        if length(M) == 2
            D1 = size(M[1], 1)
            D2 = size(M[2], 1)
            FRm = similar(FR, χ,D1,D2,χ)
        elseif ndims(M) == 5
            D = size(M, 1)
            FRm = similar(FR, χ,D,D,χ)
        else
            D = size(M, 1)
            FRm = similar(FR, χ,D,χ)
        end

        for i in 1:forloop_iter
            ind = forloop_iter * rank + i
            FRm[cols..., χ_ranges[ind]] = FRmap(FR, ARu, ARd[χ_ranges[ind], cols...], M)
            synchronize(FR)
        end
        

        element_size = prod(size(FRm)[1:end-1])
        counts = Cint[sum([length(χ_ranges[(i-1)*forloop_iter+j]) for j in 1:forloop_iter]) * element_size for i in 1:nprocs]
        MPI.Allgatherv!(VBuffer(FRm, counts), comm)

        return FRm
    else
        return FRmap_forloop(FR, ARu, ARd, M; forloop_iter)
    end
end

function ACmap_parallel(AC, FL, FR, M; ifcheckpoint, forloop_iter, ifparallel)
    if ifparallel
        comm = MPI.COMM_WORLD
        rank = MPI.Comm_rank(comm)
        nprocs = MPI.Comm_size(comm)
        # if rank == 0
        #     free_memory = Sys.free_memory() / 2^30
        #     total_memory = Sys.total_memory() / 2^30
        #     gc_live = Base.gc_live_bytes() / 2^30
        #     @show "forward",gc_live,free_memory,total_memory,free_memory/total_memory
        # end
        
        χ = size(AC, 1)
        χ_slice = cld(χ, nprocs*forloop_iter)
        χ_ranges = [range(1 + (i-1)*χ_slice, min(i*χ_slice, χ)) for i in 1:nprocs*forloop_iter]
        cols = fill(:, ndims(AC)-1)

        if length(M) == 2
            D1 = size(M[1], 2)
            D2 = size(M[2], 2)
            ACm = similar(AC, χ,D1,D2,χ)
        elseif ndims(M) == 5
            D = size(M, 2)
            ACm = similar(AC, χ,D,D,χ)
        else
            D = size(M, 2)
            ACm = similar(AC, χ,D,χ)
        end

        for i in 1:forloop_iter
            ind = forloop_iter * rank + i
            ACm[cols..., χ_ranges[ind]] = ACmap(AC, FL, FR[cols...,χ_ranges[ind]], M)
            synchronize(AC)
        end
        
        element_size = prod(size(ACm)[1:end-1])
        counts = Cint[sum([length(χ_ranges[(i-1)*forloop_iter+j]) for j in 1:forloop_iter]) * element_size for i in 1:nprocs]
        MPI.Allgatherv!(VBuffer(ACm, counts), comm)

        return ACm
    else
        return ACmap_forloop(AC, FL, FR, M; forloop_iter)
    end
end
