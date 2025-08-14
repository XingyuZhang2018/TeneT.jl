function FLmap_parallel(FL, ALu, ALd, M; forloop_iter, ifparallel)
    if ifparallel
        comm = MPI.COMM_WORLD
        rank = MPI.Comm_rank(comm)
        nprocs = MPI.Comm_size(comm)
        χ = size(FL, 1)
        χ_device = cld(χ, nprocs)
        χ_ranges = [range(1 + (i-1)*χ_device, min(i*χ_device, χ)) for i in 1:nprocs]
        cols = fill(:, ndims(FL)-1)
        FLm = zero(FL)

        FLm[cols..., χ_ranges[rank+1]] .= FLmap_forloop(FL, ALu, ALd[cols..., χ_ranges[rank+1]], M; forloop_iter)
        synchronize(FL)

        element_size = prod(size(FL)[1:end-1])
        counts = Cint[length(χ_ranges[i]) * element_size for i in 1:nprocs]
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

function FRmap_parallel(FR, ARu, ARd, M; forloop_iter, ifparallel)
    if ifparallel
        comm = MPI.COMM_WORLD
        rank = MPI.Comm_rank(comm)
        nprocs = MPI.Comm_size(comm)
        
        χ = size(FR, 1)
        χ_device = cld(χ, nprocs)
        χ_ranges = [range(1 + (i-1)*χ_device, min(i*χ_device, χ)) for i in 1:nprocs]
        cols = fill(:, ndims(FR)-1)
        FRm = zero(FR)

        FRm[cols..., χ_ranges[rank+1]] .= FRmap_forloop(FR, ARu, ARd[χ_ranges[rank+1], cols...], M; forloop_iter)
        synchronize(FR)

        element_size = prod(size(FR)[1:end-1])
        counts = Cint[length(χ_ranges[i]) * element_size for i in 1:nprocs]
        MPI.Allgatherv!(VBuffer(FRm, counts), comm)

        return FRm
    else
        return FRmap_forloop(FR, ARu, ARd, M; forloop_iter)
    end
end

function ACmap_parallel(AC, FL, FR, M; forloop_iter, ifparallel)
    if ifparallel
        comm = MPI.COMM_WORLD
        rank = MPI.Comm_rank(comm)
        nprocs = MPI.Comm_size(comm)
        
        χ = size(AC, 1)
        χ_device = cld(χ, nprocs)
        χ_ranges = [range(1 + (i-1)*χ_device, min(i*χ_device, χ)) for i in 1:nprocs]
        cols = fill(:, ndims(AC)-1)
        ACm = zero(AC)

        ACm[cols..., χ_ranges[rank+1]] .= ACmap_forloop(AC, FL, FR[cols...,χ_ranges[rank+1]], M; forloop_iter)
        synchronize(AC)

        element_size = prod(size(AC)[1:end-1])
        counts = Cint[length(χ_ranges[i]) * element_size for i in 1:nprocs]
        MPI.Allgatherv!(VBuffer(ACm, counts), comm)

        return ACm
    else
        return ACmap_forloop(AC, FL, FR, M; forloop_iter)
    end
end
