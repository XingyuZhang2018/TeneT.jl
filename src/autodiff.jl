export num_grad

@non_differentiable VUMPSRuntime(M, χ::Int)
@non_differentiable VUMPSRuntime(M, χ::Int, alg::VUMPS)
@non_differentiable randSA(kwargs...)
@non_differentiable ISA(kwargs...)

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

# adjoint for QR factorization
# https://journals.aps.org/prx/abstract/10.1103/PhysRevX.9.031041 eq.(5)
function ChainRulesCore.rrule(::typeof(qrpos), A::AbstractArray{T,2}) where {T}
    Q, R = qrpos(A)
    function back((dQ, dR))
        M = Array(R * dR' - dQ' * Q)
        dA = (UpperTriangular(R + I * 1e-12) \ (dQ + Q * _arraytype(Q)(Hermitian(M, :L)))' )'
        return NoTangent(), _arraytype(Q)(dA)
    end
    return (Q, R), back
end

function ChainRulesCore.rrule(::typeof(lqpos), A::AbstractArray{T,2}) where {T}
    L, Q = lqpos(A)
    function back((dL, dQ))
        M = Array(L' * dL - dQ * Q')
        dA = LowerTriangular(L + I * 1e-12)' \ (dQ + _arraytype(Q)(Hermitian(M, :L)) * Q)
        return NoTangent(), _arraytype(Q)(dA)
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

function ChainRulesCore.rrule(::typeof(to_Array), x)
    function back(dx)
        return NoTangent(), to_CuArray(dx)
    end
    return to_Array(x), back
end

function ChainRulesCore.rrule(::typeof(to_CuArray), x)
    function back(dx)
        return NoTangent(), to_Array(dx)
    end
    return to_CuArray(x), back
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

# function ChainRulesCore.rrule(::typeof(vumps_itr), rt::VUMPSRuntime, M, alg::VUMPS)
#     rt = vumps_itr(rt, M, alg)
#     function back(∂rt)
#         AL, AR = rt.AL, rt.AR
#         ∂AL, ∂AR, ∂C, ∂FL, ∂FR = ∂rt.AL, ∂rt.AR, ∂rt.C, ∂rt.FL, ∂rt.FR
#         ∂AL = project_AL(∂AL, AL)
#         ∂AR = project_AR(∂AR, AR)
#         ∂rt0 = [∂AL, ∂AR, ∂C, ∂FL, ∂FR]

#         _, vumps_itr_vjp = pullback(fix_gauge_vumps_step, rt, M, alg)
#         # _, vumps_itr_vjp = pullback(vumps_step_Hermitian, rt, M, alg)
#         function vjp_rt_rt(∂rt)
#             if length(∂rt) == 2
#                 ∂AL, ∂AR = ∂rt
#                 ∂AL = project_AL(∂AL, AL)
#                 ∂AR = project_AR(∂AR, AR)
#                 ∂rt = [∂AL, ∂AR, NoTangent(), NoTangent(), NoTangent()]
#             else
#                 ∂AL, ∂AR, ∂C, ∂FL, ∂FR = ∂rt
#                 ∂AL = project_AL(∂AL, AL)
#                 ∂AR = project_AR(∂AR, AR)
#                 ∂rt = [∂AL, ∂AR, ∂C, ∂FL, ∂FR]
#             end
#             ∂rt = vumps_itr_vjp((∂rt, NoTangent()))[1]
#             ∂AL = project_AL(∂rt.AL, AL)
#             ∂AR = project_AR(∂rt.AR, AR)
#             ∂rt = [∂AL, ∂AR]
#             return ∂rt
#         end
        
#         ∂rt = vjp_rt_rt(∂rt0)
#         f_map(∂rt) = ∂rt - vjp_rt_rt(∂rt)
#         ∂rtsum, info = linsolve(f_map, ∂rt, ∂rt; tol = 1e-10, maxiter = 1) 
#         alg.verbosity >= 1 && info.converged == 0 && @warn "AD linsolve doesn't converge"
#         ∂rtsum = [∂rt0[1:2]+∂rtsum..., ∂C, ∂FL, ∂FR]

#         # ∂rtsum = deepcopy(∂rt0)
#         # ∂rt = vjp_rt_rt(∂rt0)
#         # ∂rtsum += ∂rt
#         # ϵ = Inf
#         # for ix in 1:100
#         #     ∂rt = vjp_rt_rt(∂rt)
#         #     ∂rtsum += ∂rt
#         #     ϵ = norm(∂rt)
#         #     println("INFO vumps_pushback: $(ix) ϵ = ", ϵ)
#         #     (ϵ < 1e-12) && break 
#         # end
#         # ∂rtsum = [∂rtsum..., ∂C, ∂FL, ∂FR]

#         vjp_rt_M(∂rt) = vumps_itr_vjp((∂rt, NoTangent()))[2]
#         ∂M = vjp_rt_M(∂rtsum)

#         return NoTangent(), NoTangent(), ∂M, NoTangent()
#     end
#     return rt, back
# end


# """
#     dAMmap!(Au, Ad, M, L, R, i)

# ```
#                ┌──  Auᵢⱼ ──┐ 
#                │     │     │ 
# dMᵢⱼ    =  -   L ──     ── R
#                │     │     │ 
#                └── Adᵢ₊₁ⱼ──┘ 

#                ┌──       ──┐ 
#                │     │     │ 
# dAuᵢⱼ   =  -   L ── Mᵢⱼ  ──R
#                │     │     │ 
#                └── Adᵢ₊₁ⱼ──┘ 

#                ┌──  Auᵢⱼ ──┐       a ────┬──── c     
#                │     │     │       │     b     │    
# dAdᵢ₊₁ⱼ =  -   L ─── Mᵢⱼ ──R       ├─ d ─┼─ e ─┤     
#                │     │     │       │     g     │  
#                └──       ──┘       f ────┴──── h  

# ```
# """
# function dAMmap!(dAui, dAdir, dMi, Aui, Adir, Mi, L, R)
#     Nj = size(dAui, 1)
#     for j in 1:Nj
#         dAui[j]  .= -conj!(ein"((adf,fgh),dgeb),ceh -> abc"(L[j], Adir[j], Mi[j],   R[j]))
#         dAdir[j] .= -conj!(ein"((adf,abc),dgeb),ceh -> fgh"(L[j], Aui[j],  Mi[j],   R[j]))
#         dMi[j]   .= -conj!(ein"(adf,abc),(fgh,ceh) -> dgeb"(L[j], Aui[j],  Adir[j], R[j]))
#     end
# end

# """
#     ξLm = ξLmap(ARu, ARd, M, FR, i)

# ```
#     ── ALuᵢⱼ  ──┐          ──┐          a ────┬──── c 
#         │       │            │          │     b     │ 
#     ── Mᵢⱼ   ──ξLᵢⱼ₊₁  =   ──ξLᵢⱼ       ├─ d ─┼─ e ─┤ 
#         │       │            │          │     g     │ 
#     ── ALdᵢ₊₁ⱼ ─┘          ──┘          f ────┴──── h 
# ```
# """
# function ξLmap(ALui, ALdir, Mi, ξLi)
#     ξLijr = circshift(ξLi, -1)
#     return [ein"((ceh,abc),dgeb),fgh -> adf"(ξL,ALu,M,ALd) for (ξL,ALu,M,ALd) in zip(ξLijr,ALui,Mi,ALdir)]
# end

# function ChainRulesCore.rrule(::typeof(leftenv), ALu, ALd, M, FL; ifobs = false, kwargs...)
#     λL, FL = leftenv(ALu, ALd, M, FL; ifobs, kwargs...)
#     Ni = size(M, 1)
#     function back((dλL, dFL))
#         dALu = zero(ALu)
#         dALd = zero(ALd)
#         dM   = zero(M)
#         @inbounds for i = 1:Ni
#             ir = ifobs ? Ni+1-i : mod1(i+1, Ni)
#             dFL[i,:] .-= dot(FL[i,:], dFL[i,:]) * FL[i,:]
#             ξL, info = linsolve(X -> ξLmap(ALu[i,:], ALd[ir,:], M[i,:], X), conj(dFL[i,:]), -λL[i], 1; maxiter = 1)
#             info.converged == 0 && @warn "ad's linsolve not converge"
#             ξL .= circshift(ξL, -1)
#             dAMmap!(view(dALu,i,:), view(dALd,ir,:), view(dM,i,:), ALu[i,:], ALd[ir,:], M[i,:], FL[i,:], ξL)
#         end
#         return NoTangent(), dALu, dALd, dM, NoTangent()
#     end
#     return (λL, FL), back
# end

# """
#     ξRm = ξRmap(ARu, ARd, M, FL)

# ```
#   ┌──       ┌──  ARuᵢⱼ  ──                     a ────┬──── c 
#   │         │     │                            │     b     │ 
# ξRᵢⱼ   =  ξRᵢⱼ₋₁─ Mᵢⱼ   ──                     ├─ d ─┼─ e ─┤ 
#   │         │     │                            │     g     │ 
#   └──       └──  ARdᵢ₊₁ⱼ ─                     f ────┴──── h 
# ```
# """

# function ξRmap(ARui, ARdir, Mi, ξRi)
#     ξRijr = circshift(ξRi, 1)
#     return [ein"((adf,abc),dgeb),fgh -> ceh"(ξR,ARu,M,ARd) for (ξR,ARu,M,ARd) in zip(ξRijr,ARui,Mi,ARdir)]
# end

# function ChainRulesCore.rrule(::typeof(rightenv), ARu, ARd, M, FR; ifobs = false, kwargs...)
#     λR, FR = rightenv(ARu, ARd, M, FR; ifobs, kwargs...)
#     Ni = size(M, 1)
#     function back((dλ, dFR))
#         dARu = zero(ARu)
#         dARd = zero(ARd)
#         dM   = zero(M)
#         @inbounds for i = 1:Ni
#             ir = ifobs ? Ni+1-i : mod1(i+1, Ni)
#             dFR[i,:] .-= dot(FR[i,:], dFR[i,:]) * FR[i,:]
#             ξR, info = linsolve(X -> ξRmap(ARu[i,:], ARd[ir,:], M[i,:], X), conj(dFR[i,:]), -λR[i], 1; maxiter = 1)
#             info.converged == 0 && @warn "ad's linsolve not converge"
#             ξR = circshift(ξR, 1)
#             dAMmap!(view(dARu,i,:), view(dARd,ir,:), view(dM,i,:), ARu[i,:], ARd[ir,:], M[i,:], ξR, FR[i,:])
#         end
#         return NoTangent(), dARu, dARd, dM, NoTangent()
#     end
#     return (λR, FR), back
# end

# """
#     ξACmap(ξAC, FL, FR, M, j)

# ```
#                                                             a ────┬──── c 
#                                                             │     b     │
# │        │         │          │        │         │          ├─ d ─┼─ e ─┤ 
# └───────ξACᵢⱼ ─────┘    =     FLᵢⱼ ─── Mᵢⱼ ───── FRᵢⱼ       │     g     │ 
#                               │        │         │          f ────┴──── h     
#                               └──────ξACᵢ₊₁ⱼ ────┘                                    
                         
# ```
# """
# function ξACmap(ξACj, FLj, FRj, Mj)
#     ξACirj = circshift(ξACj, -1)
#     return [ein"((adf,fgh),dgeb),ceh -> abc"(FL,ξAC,M,FR) for (FL,ξAC,M,FR) in zip(FLj,ξACirj,Mj,FRj)] 
# end

# """
#     ACdFMmap(FLj, Mi, FRj, AC, ACd, i, II)

# ```

#                ┌──── ACu  ───┐ 
#                │      │      │ 
# dFLᵢⱼ    =          ──Mᵢⱼ ── FRᵢⱼ
#                │      │      │
#                └───  ACd ────┘ 

#                ┌──── ACu  ───┐ 
#                │      │      │ 
# dMᵢⱼ    =      FLᵢⱼ ─   ─── FRᵢⱼ
#                │      │      │
#                └───  ACd ────┘ 

#                ┌──── ACu  ───┐          a ────┬──── c    
#                │      │      │          │     b     │  
# dFRᵢⱼ    =     FLᵢⱼ ──Mᵢⱼ ──            ├─ d ─┼─ e ─┤ 
#                │      │      │          │     g     │  
#                └───  ACd ────┘          f ────┴──── h   
# ```
# """
# function dFMmap!(dFLj, dMj, dFRj, FLj, Mj, FRj, ACuj, ACdj)
#     Ni = size(dFLj, 1)
#     for i in 1:Ni
#         dFLj[i] .= -conj!(ein"((abc,ceh),dgeb),fgh -> adf"(ACuj[i], FRj[i], Mj[i], ACdj[i]))
#         dMj[i]  .= -conj!(ein"(abc,adf),(ceh,fgh) -> dgeb"(ACuj[i], FLj[i], FRj[i], ACdj[i]))
#         dFRj[i] .= -conj!(ein"((abc,adf),dgeb),fgh -> ceh"(ACuj[i], FLj[i], Mj[i], ACdj[i]))
#     end
# end

# function ChainRulesCore.rrule(::typeof(ACenv), AC, FL, M, FR; kwargs...)
#     λAC, AC = ACenv(AC, FL, M, FR; kwargs...)
#     Nj = size(M, 2)
#     function back((dλ, dAC))
#         dFL = zero(FL)
#         dM  = zero(M)
#         dFR = zero(FR)
#         @inbounds for j = 1:Nj
#             dAC[:,j] .-= dot(AC[:,j], dAC[:,j]) * AC[:,j]
#             ξAC, info = linsolve(X -> ξACmap(X, FL[:,j], FR[:,j], M[:,j]), conj(dAC[:,j]), -λAC[j], 1; maxiter = 1)
#             info.converged == 0 && @warn "ad's linsolve not converge"
#             ξAC = circshift(ξAC, -1)
#             dFMmap!(view(dFL,:,j), view(dM,:,j), view(dFR,:,j), FL[:,j], M[:,j], FR[:,j], AC[:,j], ξAC)
#         end
#         return NoTangent(), NoTangent(), dFL, dM, dFR
#     end
#     return (λAC, AC), back
# end

# """
#     ξCmap(ξC, FL, FR, j)

# ```               
#                                                     a ─── b
#                         │                │          │     │       
# │                │      FLᵢⱼ₊₁ ───────  FRᵢⱼ        ├─ c ─┤   
# └────── Cᵢⱼ ─────┘  =   │                │          │     │     
#                         └───── Cᵢ₊₁ⱼ ────┘          d ─── e  
# ```
# """
# function ξCmap(ξCj, FLjr, FRj)
#     ξCirj = circshift(ξCj, -1)
#     return [ein"(acd,de),bce -> ab"(FL,ξC,FR) for (FL,ξC,FR) in zip(FLjr,ξCirj,FRj)]
# end

# """
#     CdFMmap(FLj, FRj, C, Cd, i, II)

# ```
#                ┌──── Cu ────┐ 
#                │            │ 
# dFLᵢⱼ₊₁ =  -        ────── FRᵢⱼ
#                │            │
#                └──── Cd ────┘ 
#                ┌──── Cu ────┐          a ─── b   
#                │            │          │     │  
# dFRᵢⱼ =    -   FLᵢⱼ₊₁──────            ├─ c ─┤ 
#                │            │          │     │  
#                └──── Cd ────┘          d ─── e  

# ```
# """
# function dFMmap!(dFLjr, dFRj, FLjr, FRj, Cu, Cd)
#     Ni = size(dFLjr, 1)
#     for i in 1:Ni
#         dFLjr[i] .= -conj!(ein"(ab,bce),de -> acd"(Cu[i], FRj[i],  Cd[i]))
#         dFRj[i]  .= -conj!(ein"(ab,acd),de -> bce"(Cu[i], FLjr[i], Cd[i]))
#     end
# end

# function ChainRulesCore.rrule(::typeof(Cenv), C, FL, FR; kwargs...)
#     λC, C = Cenv(C, FL, FR; kwargs...)
#     Nj = size(C, 2)
#     function back((dλ, dC))
#         dFL = zero(FL)
#         dFR = zero(FR)
#         for j = 1:Nj
#             jr = mod1(j + 1, Nj)
#             dC[:,j] .-= dot(C[:,j], dC[:,j]) * C[:,j]
#             ξC, info = linsolve(X -> ξCmap(X, FL[:,jr], FR[:,j]), conj(dC[:,j]), -λC[j], 1; maxiter = 1)
#             info.converged == 0 && @warn "ad's linsolve not converge"
#             ξC = circshift(ξC, -1)
#             dFMmap!(view(dFL,:,jr), view(dFR,:,j), FL[:,jr], FR[:,j], C[:,j], ξC)
#         end
#         return NoTangent(), NoTangent(), dFL, dFR
#     end
#     return (λC, C), back
# end