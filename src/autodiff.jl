export num_grad

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

function sum_device!(x)
    atype = _arraytype(x[1])
    set_device_id!(atype, 1)
    N_device = device_count(atype)
    @sync begin
        for i in 2:N_device
            @async begin
                x[i] = atype(x[i])
            end
        end
    end
    return sum(x)
end

function ChainRulesCore.rrule(::typeof(FLmap_parallel), FL, ALu, ALd, M)
    atype = _arraytype(FL)
    N_device = device_count(atype)
    χ = size(FL, 1)
    χ_device = cld(χ, N_device)
    χ_ranges = [range(1 + (i-1)*χ_device, min(i*χ_device, χ)) for i in 1:N_device]
    results = Vector{Any}(undef, N_device)
    cols = fill(:,ndims(FL)-1)
    FLmap_backs = Vector{Any}(undef, N_device)

    set_device_id!(atype, 1)
    FLm = similar(FL)
    FLs = to_N_device(FL)
    ALus = to_N_device(ALu)
    ALds = to_N_device(ALd)
    Ms = to_N_device(M)

    @sync begin
        for i in 1:N_device
            @async begin
                set_device_id!(atype, i)
                results[i], FLmap_backs[i] = pullback(FLmap, FLs[i], ALus[i], ALds[i][cols...,χ_ranges[i]], Ms[i])
            end
        end
    end

    set_device_id!(atype, 1)
    @sync begin
        for i in 1:N_device
            @async begin
                FLm[cols...,χ_ranges[i]] .= atype(results[i])
            end
        end
    end

    function back(dFLm)
        dFLms = to_N_device(dFLm)
        dFLs = Vector{Any}(undef, N_device)
        dALus = Vector{Any}(undef, N_device)
        dALds = device_similar(ALds)
        dMs = Vector{Any}(undef, N_device)
        
        @sync begin
            for i in 1:N_device
                @async begin
                    set_device_id!(atype, i)
                    dFLs[i], dALus[i], dALds[i][cols...,χ_ranges[i]], dMs[i] = FLmap_backs[i](dFLms[i][cols...,χ_ranges[i]])
                end
            end
        end

        set_device_id!(atype, 1)

        local dFL, dALu, dM
        @sync begin       
            @async dFL = sum_device!(dFLs)
            @async dALu = sum_device!(dALus)
            @async dM = sum_device!(dMs)     
            for i in 2:N_device
                @async begin
                    dALds[1][cols...,χ_ranges[i]] .= atype(dALds[i])[cols...,χ_ranges[i]]
                end
            end
        end

        return NoTangent(), dFL, dALu, dALds[1], dM
    end
    
    return FLm, back
end

function ChainRulesCore.rrule(::typeof(FRmap_parallel), FR, ARu, ARd, M)
    atype = _arraytype(FR)
    N_device = device_count(atype)
    χ = size(FR, 1)
    χ_device = cld(χ, N_device)
    χ_ranges = [range(1 + (i-1)*χ_device, min(i*χ_device, χ)) for i in 1:N_device]
    results = Vector{Any}(undef, N_device)
    cols = fill(:,ndims(FR)-1)
    FRmap_backs = Vector{Any}(undef, N_device)

    set_device_id!(atype, 1)
    FRm = similar(FR)
    FRs = to_N_device(FR)
    ARus = to_N_device(ARu)
    ARds = to_N_device(ARd)
    Ms = to_N_device(M)

    @sync begin
        for i in 1:N_device
            @async begin
                set_device_id!(atype, i)
                results[i], FRmap_backs[i] = pullback(FRmap, FRs[i], ARus[i][χ_ranges[i],cols...], ARds[i], Ms[i])
            end
        end
    end

    set_device_id!(atype, 1)
    @sync begin
        for i in 1:N_device
            @async begin
                FRm[χ_ranges[i],cols...] .= atype(results[i])
            end
        end
    end

    function back(dFRm)
        dFRms = to_N_device(dFRm)
        dFRs = Vector{Any}(undef, N_device)
        dARus = device_similar(ARds)
        dARds = Vector{Any}(undef, N_device)
        dMs = Vector{Any}(undef, N_device)
        
        @sync begin
            for i in 1:N_device
                @async begin
                    set_device_id!(atype, i)
                    dFRs[i], dARus[i][χ_ranges[i],cols...], dARds[i], dMs[i] = FRmap_backs[i](dFRms[i][χ_ranges[i],cols...])
                end
            end
        end

        set_device_id!(atype, 1)
        local dFR, dARd, dM
        @sync begin
            @async dFR = sum_device!(dFRs)
            @async dARd = sum_device!(dARds)
            @async dM = sum_device!(dMs)
            for i in 2:N_device
                @async begin
                    dARus[1][χ_ranges[i],cols...] .= atype(dARus[i])[χ_ranges[i],cols...]
                end
            end
        end

        return NoTangent(), dFR, dARus[1], dARd, dM
    end
    
    return FRm, back
end

function ChainRulesCore.rrule(::typeof(ACmap_parallel), AC, FL, FR, M)
    atype = _arraytype(AC)
    N_device = device_count(atype)
    χ = size(AC, 1)
    χ_device = cld(χ, N_device)
    χ_ranges = [range(1 + (i-1)*χ_device, min(i*χ_device, χ)) for i in 1:N_device]
    results = Vector{Any}(undef, N_device)
    cols = fill(:,ndims(AC)-1)
    ACmap_backs = Vector{Any}(undef, N_device)

    set_device_id!(atype, 1)
    if ndims(M) == 4
        D = size(M,2)
        ACm = similar(AC, χ, D, χ)
    else
        D = size(M,3)
        ACm = similar(AC, χ, D, D, χ)
    end
    ACs = to_N_device(AC)
    FLs = to_N_device(FL)
    FRs = to_N_device(FR)
    Ms = to_N_device(M)

    @sync begin
        for i in 1:N_device
            @async begin
                set_device_id!(atype, i)
                results[i], ACmap_backs[i] = pullback(ACmap, ACs[i], FLs[i], FRs[i][cols...,χ_ranges[i]], Ms[i])
            end
        end
    end

    set_device_id!(atype, 1)
    @sync begin
        for i in 1:N_device
            @async begin
                ACm[cols...,χ_ranges[i]] .= atype(results[i])
            end
        end
    end

    function back(dACm)
        dACms = to_N_device(dACm)
        dACs = Vector{Any}(undef, N_device)
        dFLs = Vector{Any}(undef, N_device)
        dFRs = device_similar(FRs)
        dMs = Vector{Any}(undef, N_device)
        
        @sync begin
            for i in 1:N_device
                @async begin
                    set_device_id!(atype, i)
                    dACs[i], dFLs[i], dFRs[i][cols...,χ_ranges[i]], dMs[i] = ACmap_backs[i](dACms[i][cols...,χ_ranges[i]])
                end
            end
        end

        set_device_id!(atype, 1)
        local dAC, dFL, dM
        @sync begin
            @async dAC = sum_device!(dACs)
            @async dFL = sum_device!(dFLs)
            @async dM = sum_device!(dMs)
            for i in 2:N_device
                @async begin
                    dFRs[1][cols...,χ_ranges[i]] .= atype(dFRs[i])[cols...,χ_ranges[i]]
                end
            end
        end

        return NoTangent(), dAC, dFL, dFRs[1], dM
    end
    
    return ACm, back
end