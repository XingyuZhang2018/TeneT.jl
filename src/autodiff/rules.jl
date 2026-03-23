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
    function back(ȳ)
        return NoTangent(), NoTangent(), NoTangent(), permutedims(ȳ)...
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
        # dA, _ = linsolve(x->x * (R + I * 1e-12)', dQ + Q * Hermitian(M, :L); verbosity=0, maxiter = 1)
        dA = (UpperTriangular(R + I * 1e-12) \ (dQ + Q * Hermitian(M, :L))' )'
        return NoTangent(), dA
    end
    return (Q, R), back
end

function ChainRulesCore.rrule(::typeof(lqpos), A::AbstractArray{T,2}) where {T}
    L, Q = lqpos(A)
    function back((dL, dQ))
        M = L' * dL - dQ * Q'
        # dA, _ = linsolve(x->(L + I * 1e-12)' * x, dQ + Hermitian(M, :L) * Q; verbosity=0, maxiter = 1)
        dA = LowerTriangular(L + I * 1e-12)' \ (dQ + Hermitian(M, :L) * Q)
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

function ChainRulesCore.rrule(::typeof(FLmap_parallel), FL, ALu, ALd, M; kwarg...)
    function back(dFLm)
        dFLm = conj(dFLm)
        dFL = conj!(FRmap_parallel(dFLm, ALu, ALd, M; kwarg...))
        dALu = conj!(ACdmap_parallel(ALd, FL, dFLm, M; kwarg...))
        dALd = conj!(ACmap_parallel(ALu, FL, dFLm, M; kwarg...))
        if M isa Tuple
            dMu = Mdmap_parallel(ALu, ALd, FL, dFLm, M[2]; kwarg...)
            dMd = Mumap_parallel(ALu, ALd, FL, dFLm, M[1]; kwarg...)
            dM = (conj!(dMu), conj!(dMd))
        elseif ndims(M) == 5
            dMu = Mdmap_parallel(ALu, ALd, FL, dFLm, conj(M); kwarg...)
            dMd = Mumap_parallel(ALu, ALd, FL, dFLm, M; kwarg...)
            dM = conj!(dMu) + dMd
        else
            dM = conj!(Mmap_parallel(ALu, ALd, FL, dFLm; kwarg...))
        end
        return NoTangent(), dFL, dALu, dALd, dM
    end
    return FLmap_parallel(FL, ALu, ALd, M; kwarg...), back
end

function ChainRulesCore.rrule(::typeof(FRmap_parallel), FR, ARu, ARd, M; kwarg...)
    function back(dFRm)
        dFRm = conj(dFRm)
        dFR = conj!(FLmap_parallel(dFRm, ARu, ARd, M; kwarg...))
        dARu = conj!(ACdmap_parallel(ARd, dFRm, FR, M; kwarg...))
        dARd = conj!(ACmap_parallel(ARu, dFRm, FR, M; kwarg...))
        if M isa Tuple
            dMu = Mdmap_parallel(ARu, ARd, dFRm, FR, M[2]; kwarg...)
            dMd = Mumap_parallel(ARu, ARd, dFRm, FR, M[1]; kwarg...)
            dM = (conj!(dMu), conj!(dMd))
        elseif ndims(M) == 5
            dMu = Mdmap_parallel(ARu, ARd, dFRm, FR, conj(M); kwarg...)
            dMd = Mumap_parallel(ARu, ARd, dFRm, FR, M; kwarg...)
            dM = conj!(dMu) + dMd
        else
            dM = conj!(Mmap_parallel(ARu, ARd, dFRm, FR; kwarg...))
        end
        return NoTangent(), dFR, dARu, dARd, dM
    end
    return FRmap_parallel(FR, ARu, ARd, M; kwarg...), back
end

function ChainRulesCore.rrule(::typeof(ACmap_parallel), AC, FL, FR, M; kwarg...)
    function back(dACm)
        dACm = conj(dACm)
        dAC = conj!(ACdmap_parallel(dACm, FL, FR, M; kwarg...))
        dFL = conj!(FRmap_parallel(FR, AC, dACm, M; kwarg...))
        dFR = conj!(FLmap_parallel(FL, AC, dACm, M; kwarg...))
        if M isa Tuple
            dMu = Mdmap_parallel(AC, dACm, FL, FR, M[2]; kwarg...)
            dMd = Mumap_parallel(AC, dACm, FL, FR, M[1]; kwarg...)
            dM = (conj!(dMu), conj!(dMd))
        elseif ndims(M) == 5
            dMu = Mdmap_parallel(AC, dACm, FL, FR, conj(M); kwarg...)
            dMd = Mumap_parallel(AC, dACm, FL, FR, M; kwarg...)
            dM = conj!(dMu) + dMd
        else
            dM = conj!(Mmap_parallel(AC, dACm, FL, FR; kwarg...))
        end
        return NoTangent(), dAC, dFL, dFR, dM
    end
    return ACmap_parallel(AC, FL, FR, M; kwarg...), back
end

# function ChainRulesCore.rrule(::typeof(vumps_itr), rt::VUMPSRuntime, M, alg::VUMPS)
#     rt, err = vumps_itr(rt, M, alg)
#     pattern = rt.AL.pattern
#     function back((∂rt, ∂err))
#         AL, AR = rt.AL, rt.AR
#         ∂AL, ∂AR, ∂C, ∂FL, ∂FR = ∂rt
#         # ∂AL = project_AL(∂AL.data, AL.data)
#         # ∂AR = project_AR(∂AR.data, AR.data)
#         ∂AL isa AbstractZero || project_AL!(∂AL.data, AL.data)
#         ∂AR isa AbstractZero || project_AR!(∂AR.data, AR.data)
#         # ∂rt0 = [∂AL, ∂AR, ∂C, ∂FL, ∂FR]

#         ∂rt0 = [∂AL isa AbstractZero ? ∂AL : StructArray(∂AL.data, pattern),
#                 ∂AR isa AbstractZero ? ∂AR : StructArray(∂AR.data, pattern),
#                  ∂C isa AbstractZero ? ∂C : StructArray(∂C.data, pattern),
#                 ∂FL isa AbstractZero ? ∂FL : StructArray(∂FL.data, pattern),
#                 ∂FR isa AbstractZero ? ∂FR : StructArray(∂FR.data, pattern)]

#         # _, vumps_itr_vjp = pullback(fix_gauge_vumps_step, rt, M, alg)
#         _, vumps_itr_vjp = pullback(vumps_step_Hermitian, rt, M, alg)
#         function vjp_rt_rt(∂rt)
#             ∂AL, ∂AR, ∂C, ∂FL, ∂FR = ∂rt
#             ∂AL isa AbstractZero || project_AL!(∂AL.data, AL.data)
#             ∂AR isa AbstractZero || project_AR!(∂AR.data, AR.data)
#             ∂rt = [∂AL isa AbstractZero ? ∂AL : StructArray(∂AL.data, pattern),
#                    ∂AR isa AbstractZero ? ∂AR : StructArray(∂AR.data, pattern),
#                     ∂C isa AbstractZero ? ∂C : StructArray(∂C.data, pattern),
#                    ∂FL isa AbstractZero ? ∂FL : StructArray(∂FL.data, pattern),
#                    ∂FR isa AbstractZero ? ∂FR : StructArray(∂FR.data, pattern)]

#             ∂rt = vumps_itr_vjp((∂rt, NoTangent()))[1]

#             ∂AL, ∂AR, ∂C, ∂FL, ∂FR = ∂rt
#             ∂AL isa AbstractZero || project_AL!(∂AL.data, AL.data)
#             ∂AR isa AbstractZero || project_AR!(∂AR.data, AR.data)
#             ∂rt = [∂AL isa AbstractZero ? ∂AL : StructArray(∂AL.data, pattern),
#                    ∂AR isa AbstractZero ? ∂AR : StructArray(∂AR.data, pattern),
#                     ∂C isa AbstractZero ? ∂C : StructArray(∂C.data, pattern),
#                    ∂FL isa AbstractZero ? ∂FL : StructArray(∂FL.data, pattern),
#                    ∂FR isa AbstractZero ? ∂FR : StructArray(∂FR.data, pattern)]

#             return ∂rt
#         end

#         # ∂rt = vjp_rt_rt(∂rt0)
#         # f_map(∂rt) = ∂rt - vjp_rt_rt(∂rt)
#         # ∂rtsum, info = linsolve(f_map, ∂rt, ∂rt; tol = 1e-10, maxiter = 1)
#         # alg.verbosity >= 1 && info.converged == 0 && @warn "AD linsolve doesn't converge"
#         # ∂rtsum = [∂rt0[1:2]+∂rtsum..., ∂C, ∂FL, ∂FR]

#         ∂rtsum = deepcopy(∂rt0)
#         ∂rt = vjp_rt_rt(∂rt0)
#         ∂rtsum += ∂rt
#         ϵ = Inf
#         for ix in 1:5
#             ∂rt = vjp_rt_rt(∂rt)
#             ∂rtsum += ∂rt
#             ϵ = norm(∂rt)
#             println("INFO vumps_pushback: $(ix) ϵ = ", ϵ)
#             (ϵ < 1e-12) && break
#         end

#         vjp_rt_M(∂rt) = vumps_itr_vjp((∂rt, NoTangent()))[2]
#         ∂M = vjp_rt_M(∂rtsum)

#         return NoTangent(), NoTangent(), ∂M, NoTangent()
#     end
#     return (rt, err), back
# end

# ─── SVD adjoint ──────────────────────────────────────────────────────────────

struct ZeroAdder end
Base.:+(a, zero::ZeroAdder) = a
Base.:+(zero::ZeroAdder, a) = a
Base.:-(a, zero::ZeroAdder) = a
Base.:-(zero::ZeroAdder, a) = -a
Base.:-(zero::ZeroAdder) = zero

"""
    svd_back(U, S, V, dU, dS, dV)

adjoint for SVD decomposition.

References:
    https://j-towns.github.io/papers/svd-derivative.pdf
    https://giggleliu.github.io/2019/04/02/einsumbp.html
"""
function svd_back(U::AbstractArray, S::AbstractArray{T}, V, dU, dS, dV; η::Real=1e-40) where T
    all(x -> x isa Nothing, (dU, dS, dV)) && return nothing
    η = T(η)
    NS = length(S)
    S2 = S .^ 2
    Sinv = @. S/(S2+η)
    F = S2' .- S2
    F ./= (F.^ 2 .+ η)

    res = ZeroAdder()
    if !(dU isa Nothing)
        UdU = U'*dU
        J = F.*(UdU)
        res += (J+J')*LinearAlgebra.Diagonal(S) + LinearAlgebra.Diagonal(1im*imag(LinearAlgebra.diag(UdU)) .* Sinv)
    end
    if !(dV isa Nothing)
        VdV = V'*dV
        K = F.*(VdV)
        res += LinearAlgebra.Diagonal(S) * (K+K')
    end
    if !(dS isa Nothing)
        res += LinearAlgebra.Diagonal(dS)
    end

    res = U*res*V'

    if !(dU isa Nothing) && size(U, 1) != size(U, 2)
        res += (dU - U* (U'*dU)) * LinearAlgebra.Diagonal(Sinv) * V'
    end

    if !(dV isa Nothing) && size(V, 1) != size(V, 2)
        res = res + U * LinearAlgebra.Diagonal(Sinv) * (dV' - (dV'*V)*V')
    end
    res
end

Zygote.@adjoint function LinearAlgebra.svd(A)
    res = LinearAlgebra.svd(A)
    res, function (dy)
        dU, dS, dVt = dy
        return (svd_back(res.U, res.S, res.V, dU, dS, dVt === nothing ? nothing : dVt'),)
    end
end
