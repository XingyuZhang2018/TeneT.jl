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
        dA = (UpperTriangular(R + I * 1e-12) \ (dQ + Q * Hermitian(M, :L))' )'
        return NoTangent(), dA
    end
    return (Q, R), back
end

function ChainRulesCore.rrule(::typeof(lqpos), A::AbstractArray{T,2}) where {T}
    L, Q = lqpos(A)
    function back((dL, dQ))
        M = L' * dL - dQ * Q'
        dA = LowerTriangular(L + I * 1e-12)' \ (dQ + Hermitian(M, :L) * Q)
        return NoTangent(), dA
    end
    return (L, Q), back
end

function ChainRulesCore.rrule(::typeof(orth_for_ad), v)
    function back(dv)
        dv .-= dot(v, dv) * v
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

function ChainRulesCore.rrule(::typeof(FLmap_parallel), FL, ALu, ALd, M; forloop_iter, ifparallel)
    closure(FL, ALu, ALd, M) = FLmap_forloop(FL, ALu, ALd, M; forloop_iter)
    if ifparallel
        comm = MPI.COMM_WORLD
        rank = MPI.Comm_rank(comm)
        nprocs = MPI.Comm_size(comm)
        χ = size(FL, 1)
        χ_device = cld(χ, nprocs)
        χ_ranges = [range(1 + (i-1)*χ_device, min(i*χ_device, χ)) for i in 1:nprocs]
        cols = fill(:, ndims(FL)-1)
        FLm = zero(FL)

        FLm[cols..., χ_ranges[rank+1]], FLmap_back = pullback(closure, FL, ALu, ALd[cols...,χ_ranges[rank+1]], M)
        synchronize(FL)

        element_size = prod(size(FL)[1:end-1])
        counts = Cint[length(χ_ranges[i]) * element_size for i in 1:nprocs]
        MPI.Allgatherv!(VBuffer(FLm, counts), comm)

        function back(dFLm)  
            dALd = zero(ALd)
            dFL, dALu, dALd[cols...,χ_ranges[rank+1]], dM = FLmap_back(dFLm[cols...,χ_ranges[rank+1]])
            synchronize(dFLm)

            MPI.Allgatherv!(VBuffer(dALd, counts), comm)
            MPI.Allreduce!(dFL, +, comm)
            MPI.Allreduce!(dALu, +, comm)
            MPI.Allreduce!(dM, +, comm)

            return NoTangent(), dFL, dALu, dALd, dM, NoTangent(), NoTangent()
        end
        return FLm, back
    else
        FLm, FLmap_back = pullback(closure, FL, ALu, ALd, M)

        function back2(dFLm)
            dFL, dALu, dALd, dM = FLmap_back(dFLm)
            return NoTangent(), dFL, dALu, dALd, dM, NoTangent(), NoTangent()
        end
        return FLm, back2
    end
end

function ChainRulesCore.rrule(::typeof(FRmap_parallel), FR, ARu, ARd, M; forloop_iter, ifparallel)
    closure(FR, ARu, ARd, M) = FRmap_forloop(FR, ARu, ARd, M; forloop_iter)
    if ifparallel
        comm = MPI.COMM_WORLD
        rank = MPI.Comm_rank(comm)
        nprocs = MPI.Comm_size(comm)
        
        χ = size(FR, 1)
        χ_device = cld(χ, nprocs)
        χ_ranges = [range(1 + (i-1)*χ_device, min(i*χ_device, χ)) for i in 1:nprocs]
        cols = fill(:, ndims(FR)-1)
        FRm = zero(FR)

        FRm[cols..., χ_ranges[rank+1]], FRmap_back = pullback(closure, FR, ARu, ARd[χ_ranges[rank+1], cols...], M)
        synchronize(FR)

        element_size = prod(size(FR)[1:end-1])
        counts = Cint[length(χ_ranges[i]) * element_size for i in 1:nprocs]
        MPI.Allgatherv!(VBuffer(FRm, counts), comm)

        function back(dFRm)
            s = size(ARd)
            dARd = similar(ARd, s[2:end]..., s[1])
            dFR, dARu, dARd[cols..., χ_ranges[rank+1]], dM = FRmap_back(dFRm[cols..., χ_ranges[rank+1]])
            synchronize(dFRm)

            MPI.Allgatherv!(VBuffer(dARd, counts), comm)
            MPI.Allreduce!(dFR, +, comm)
            MPI.Allreduce!(dARu, +, comm)
            MPI.Allreduce!(dM, +, comm)

            N = ndims(dARd)
            return NoTangent(), dFR, dARu, permutedims(dARd, (N, 1:N-1...)), dM, NoTangent(), NoTangent()
        end
        return FRm, back
    else
        FRm, FRmap_back = pullback(closure, FR, ARu, ARd, M)

        function back2(dFRm)
            dFR, dARu, dARd, dM = FRmap_back(dFRm)
            return NoTangent(), dFR, dARu, dARd, dM, NoTangent(), NoTangent()
        end
        return FRm, back2
    end
end

function ChainRulesCore.rrule(::typeof(ACmap_parallel), AC, FL, FR, M; forloop_iter, ifparallel)
    closure(AC, FL, FR, M) = ACmap_forloop(AC, FL, FR, M; forloop_iter)
    if ifparallel
        comm = MPI.COMM_WORLD
        rank = MPI.Comm_rank(comm)
        nprocs = MPI.Comm_size(comm)
        
        χ = size(AC, 1)
        χ_device = cld(χ, nprocs)
        χ_ranges = [range(1 + (i-1)*χ_device, min(i*χ_device, χ)) for i in 1:nprocs]
        cols = fill(:, ndims(AC)-1)
        ACm = zero(AC)

        ACm[cols..., χ_ranges[rank+1]], ACmap_back = pullback(closure, AC, FL, FR[cols..., χ_ranges[rank+1]], M)
        synchronize(AC)

        element_size = prod(size(AC)[1:end-1])
        counts = Cint[length(χ_ranges[i]) * element_size for i in 1:nprocs]
        MPI.Allgatherv!(VBuffer(ACm, counts), comm)

        function back(dACm)
            dFR = zero(FR)
            dAC, dFL, dFR[cols..., χ_ranges[rank+1]], dM = ACmap_back(dACm[cols..., χ_ranges[rank+1]])
            synchronize(dACm)
            
            MPI.Allgatherv!(VBuffer(dFR, counts), comm)
            MPI.Allreduce!(dAC, +, comm)
            MPI.Allreduce!(dFL, +, comm)
            MPI.Allreduce!(dM, +, comm)

            return NoTangent(), dAC, dFL, dFR, dM, NoTangent(), NoTangent()
        end
        return ACm, back
    else
        ACm, ACmap_back = pullback(closure, AC, FL, FR, M)

        function back2(dACm)
            dAC, dFL, dFR, dM = ACmap_back(dACm)
            return NoTangent(), dAC, dFL, dFR, dM, NoTangent(), NoTangent()
        end
        return ACm, back2
    end
end