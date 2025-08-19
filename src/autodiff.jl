@non_differentiable VUMPSRuntime(M, χ::Int)
@non_differentiable VUMPSRuntime(M, χ::Int, alg::VUMPS)
@non_differentiable randSA(kwargs...)
@non_differentiable ISA(kwargs...)
@non_differentiable set_device_id!(kwargs...)
@non_differentiable get_device(kwargs...)
@non_differentiable get_device_id(kwargs...)

# patch since it's currently broken otherwise
function ChainRulesCore.rrule(::typeof(Base.typed_hvcat), ::Type{T}, rows::Tuple{Vararg{Int}}, xs::S...) where {T,S}
    y = Base.typed_hvcat(T, rows, xs...)
    function back(ȳ)
        return NoTangent(), NoTangent(), NoTangent(), permutedims(ȳ)...
    end
    return y, back
end

function ChainRulesCore.rrule(::typeof(Base.sqrt), A::AbstractArray)
    As = Base.sqrt(A)
    function back(dAs)
        dA =  As' \ dAs ./2 
        return NoTangent(), dA
    end
    return As, back
end

function ChainRulesCore.rrule(::typeof(atype_device!), atype, x, i::Int)
    id_old = get_device_id(atype)
    function back(dx)
        f = pullback(atype, x)[2]
        set_device_id!(atype, get_device_id(x))
        dx = atype(f(dx)[1])
        set_device_id!(atype, id_old)
        return NoTangent(), NoTangent(), dx, NoTangent()
    end
    return atype_device!(atype, x, i), back
end

# adjoint for QR factorization
# https://journals.aps.org/prx/abstract/10.1103/PhysRevX.9.031041 eq.(5)
function ChainRulesCore.rrule(::typeof(qrpos), A::AbstractArray{T,2}) where {T}
    Q, R = qrpos(A)
    function back((dQ, dR))
        M = R * dR' - dQ' * Q
        # dA = (pinv(R + I * 1e-12) * (dQ + Q * Hermitian(M, :L))' )'
        dA, _ = linsolve(x->x * (R + I * 1e-12)', dQ + Q * Hermitian(M, :L); verbosity=0, maxiter=1)
        return NoTangent(), dA
    end
    return (Q, R), back
end

function ChainRulesCore.rrule(::typeof(lqpos), A::AbstractArray{T,2}) where {T}
    L, Q = lqpos(A)
    function back((dL, dQ))
        M = L' * dL - dQ * Q'
        # dA = pinv(L' + I * 1e-12) * (dQ + Hermitian(M, :L) * Q)
        dA, _ = linsolve(x->(L + I * 1e-12)' * x, dQ + Hermitian(M, :L) * Q; verbosity=0, maxiter=1)
        return NoTangent(), dA
    end
    return (L, Q), back
end

function ChainRulesCore.rrule(::typeof(orth_for_ad), v)
    function back(dv)
        dv -= dot(v, dv) * v
        return NoTangent(), dv
    end
    return v, back
end

function ChainRulesCore.rrule(::Type{<:VUMPSRuntime}, AL, AR, C, FL, FR)
    rt = VUMPSRuntime(AL, AR, C, FL, FR)
    function back(∂rt)
        ∂AL, ∂AR, ∂C, ∂FL, ∂FR = ∂rt
        # project_AL!(∂AL, AL)
        # project_AR!(∂AR, AR)
        return NoTangent(), ∂AL, ∂AR, ∂C, ∂FL, ∂FR
    end
    return rt, back
end

function ChainRulesCore.rrule(::Type{StructArray}, data, pattern)
    S = StructArray(data, pattern)
    function back(dS)
        return NoTangent(), dS.data, dS.pattern
    end
    return S, back
end

function ChainRulesCore.rrule(::typeof(norm), S::StructArray)
    y = norm(S)
    function back(dy)
        data_grad = pullback(norm, S.data)[2](dy)[1]
        return NoTangent(), StructArray(data_grad, S.pattern)
    end
    return y, back
end

function ChainRulesCore.rrule(::typeof(FLmap_parallel), FL, ALu, ALd, M; ifcheckpoint, forloop_iter, ifparallel)
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
            if ifcheckpoint 
                FLm[cols..., χ_ranges[ind]] = FLmap(FL, ALu, ALd[cols...,χ_ranges[ind]], M)
            else
                FLm[cols..., χ_ranges[ind]], FLmap_back = Zygote._pullback(FLmap, FL, ALu, ALd[cols...,χ_ranges[ind]], M) # FLmap_back is not correct for ifcheckpoint=false
            end
            synchronize(FL)
        end
        
        element_size = prod(size(FLm)[1:end-1])
        counts = Cint[sum([length(χ_ranges[(i-1)*forloop_iter+j]) for j in 1:forloop_iter]) * element_size for i in 1:nprocs]
        MPI.Allgatherv!(VBuffer(FLm, counts), comm)

        function back(dFLm)  
            dALd = zero(ALd)
            dFL = zero(FL)
            dALu = zero(ALu)
            dM = zero.(M)
            for i in 1:forloop_iter
                ind = forloop_iter * rank + i
                if ifcheckpoint
                    FLmap_back = Zygote._pullback(FLmap, FL, ALu, ALd[cols...,χ_ranges[ind]], M)[2]
                    synchronize(dFLm)
                end
                _, dFLs, dALus, dALd[cols...,χ_ranges[ind]], dMs = FLmap_back(dFLm[cols...,χ_ranges[ind]])
                synchronize(dFLm)
                dFL += dFLs
                dALu += dALus
                if dM isa Tuple
                    dM[1] .+= dMs[1]
                    dM[2] .+= dMs[2]
                else
                    dM += dMs
                end
            end
            
            element_size = prod(size(dALd)[1:end-1])
            counts = Cint[sum([length(χ_ranges[(i-1)*forloop_iter+j]) for j in 1:forloop_iter]) * element_size for i in 1:nprocs]
            MPI.Allgatherv!(VBuffer(dALd, counts), comm)

            MPI.Allreduce!(dFL, +, comm)
            MPI.Allreduce!(dALu, +, comm)
            if dM isa Tuple
                MPI.Allreduce!(dM[1], +, comm)
                MPI.Allreduce!(dM[2], +, comm)
            else
                MPI.Allreduce!(dM, +, comm)
            end

            return NoTangent(), dFL, dALu, dALd, dM, NoTangent(), NoTangent()
        end
        return FLm, back
    else
        closure(FL, ALu, ALd, M) = FLmap_forloop(FL, ALu, ALd, M; forloop_iter)
        FLm, FLmap_back = Zygote._pullback(closure, FL, ALu, ALd, M)

        function back2(dFLm)
            _, dFL, dALu, dALd, dM = FLmap_back(dFLm)
            return NoTangent(), dFL, dALu, dALd, dM, NoTangent(), NoTangent()
        end
        return FLm, back2
    end
end

function ChainRulesCore.rrule(::typeof(FRmap_parallel), FR, ARu, ARd, M; ifcheckpoint, forloop_iter, ifparallel)
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
            if ifcheckpoint 
                FRm[cols..., χ_ranges[ind]] = FRmap(FR, ARu, ARd[χ_ranges[ind], cols...], M)
            else
                FRm[cols..., χ_ranges[ind]], FRmap_back = Zygote._pullback(FRmap, FR, ARu, ARd[χ_ranges[ind], cols...], M)
            end
            synchronize(FR)
        end
        
        element_size = prod(size(FRm)[1:end-1])
        counts = Cint[sum([length(χ_ranges[(i-1)*forloop_iter+j]) for j in 1:forloop_iter]) * element_size for i in 1:nprocs]
        MPI.Allgatherv!(VBuffer(FRm, counts), comm)

        function back(dFRm)
            dARd = zero(ARd)
            dFR = zero(FR)
            dARu = zero(ARu)
            dM = zero.(M)
            for i in 1:forloop_iter
                ind = forloop_iter * rank + i
                if ifcheckpoint
                    FRmap_back = Zygote._pullback(FRmap, FR, ARu, ARd[χ_ranges[ind], cols...], M)[2]
                    synchronize(dFRm)
                end
                _, dFRs, dARus, dARd[χ_ranges[ind], cols...], dMs = FRmap_back(dFRm[cols..., χ_ranges[ind]])
                synchronize(dFRm)
                dFR += dFRs
                dARu += dARus
                if dM isa Tuple
                    dM[1] .+= dMs[1]
                    dM[2] .+= dMs[2]
                else
                    dM += dMs
                end
            end
            N = ndims(dARd)
            dARd = permutedims(dARd, (2:N..., 1))
            synchronize(dFRm)

            element_size = prod(size(dARd)[1:end-1])
            counts = Cint[sum([length(χ_ranges[(i-1)*forloop_iter+j]) for j in 1:forloop_iter]) * element_size for i in 1:nprocs]
            MPI.Allgatherv!(VBuffer(dARd, counts), comm)

            MPI.Allreduce!(dFR, +, comm)
            MPI.Allreduce!(dARu, +, comm)
            if dM isa Tuple
                MPI.Allreduce!(dM[1], +, comm)
                MPI.Allreduce!(dM[2], +, comm)
            else
                MPI.Allreduce!(dM, +, comm)
            end

            return NoTangent(), dFR, dARu, permutedims(dARd, (N, 1:N-1...)), dM, NoTangent(), NoTangent()
        end
        return FRm, back
    else
        closure(FR, ARu, ARd, M) = FRmap_forloop(FR, ARu, ARd, M; forloop_iter)
        FRm, FRmap_back = Zygote._pullback(closure, FR, ARu, ARd, M)

        function back2(dFRm)
            _, dFR, dARu, dARd, dM = FRmap_back(dFRm)
            return NoTangent(), dFR, dARu, dARd, dM, NoTangent(), NoTangent()
        end
        return FRm, back2
    end
end

function ChainRulesCore.rrule(::typeof(ACmap_parallel), AC, FL, FR, M; ifcheckpoint, forloop_iter, ifparallel)
    if ifparallel
        comm = MPI.COMM_WORLD
        rank = MPI.Comm_rank(comm)
        nprocs = MPI.Comm_size(comm)
        
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
            if ifcheckpoint 
                ACm[cols..., χ_ranges[ind]] = ACmap(AC, FL, FR[cols..., χ_ranges[ind]], M)
            else
                ACm[cols..., χ_ranges[ind]], ACmap_back = Zygote._pullback(ACmap, AC, FL, FR[cols..., χ_ranges[ind]], M)
            end
            synchronize(AC)
        end
        

        element_size = prod(size(ACm)[1:end-1])
        counts = Cint[sum([length(χ_ranges[(i-1)*forloop_iter+j]) for j in 1:forloop_iter]) * element_size for i in 1:nprocs]
        MPI.Allgatherv!(VBuffer(ACm, counts), comm)

        function back(dACm)
            # if rank == 0
            #     free_memory = Sys.free_memory() / 2^30
            #     total_memory = Sys.total_memory() / 2^30
            #     gc_live = Base.gc_live_bytes() / 2^30
            #     @show "back",gc_live,free_memory,total_memory,free_memory/total_memory
            # end
            dFR = zero(FR)
            dAC = zero(AC)
            dFL = zero(FL)
            dM = zero.(M)
            for i in 1:forloop_iter
                ind = forloop_iter * rank + i
                if ifcheckpoint
                    ACmap_back = Zygote._pullback(ACmap, AC, FL, FR[cols..., χ_ranges[ind]], M)[2]
                end
                _, dACs, dFLs, dFR[cols..., χ_ranges[ind]], dMs = ACmap_back(dACm[cols..., χ_ranges[ind]])
                synchronize(dACm)
                dAC += dACs
                dFL += dFLs
                if dM isa Tuple
                    dM[1] .+= dMs[1]
                    dM[2] .+= dMs[2]
                else
                    dM += dMs
                end
            end
            
            element_size = prod(size(dFR)[1:end-1])
            counts = Cint[sum([length(χ_ranges[(i-1)*forloop_iter+j]) for j in 1:forloop_iter]) * element_size for i in 1:nprocs]
            MPI.Allgatherv!(VBuffer(dFR, counts), comm)

            MPI.Allreduce!(dAC, +, comm)
            MPI.Allreduce!(dFL, +, comm)
            if dM isa Tuple
                MPI.Allreduce!(dM[1], +, comm)
                MPI.Allreduce!(dM[2], +, comm)
            else
                MPI.Allreduce!(dM, +, comm)
            end

            return NoTangent(), dAC, dFL, dFR, dM, NoTangent(), NoTangent()
        end
        return ACm, back
    else
        closure(AC, FL, FR, M) = ACmap_forloop(AC, FL, FR, M; forloop_iter)
        ACm, ACmap_back = Zygote._pullback(closure, AC, FL, FR, M)

        function back2(dACm)
            _, dAC, dFL, dFR, dM = ACmap_back(dACm)
            return NoTangent(), dAC, dFL, dFR, dM, NoTangent(), NoTangent()
        end
        return ACm, back2
    end
end