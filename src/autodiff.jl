using ChainRulesCore
using KrylovKit
using LinearAlgebra
using Random
using Zygote
using Zygote: @adjoint

import Base: reshape
import LinearAlgebra: norm
export num_grad

Zygote.@nograd StopFunction
Zygote.@nograd error
Zygote.@nograd save
Zygote.@nograd load
Zygote.@nograd Random.seed!
Zygote.@nograd show_every_count
Zygote.@nograd _initializect_square
Zygote.@nograd U1reshapeinfo
Zygote.@nograd FLint
Zygote.@nograd FRint

@doc raw"
    num_grad(f, K::Real; [δ = 1e-5])

return the numerical gradient of `f` at `K` calculated with
`(f(K+δ/2) - f(K-δ/2))/δ`

# example

```jldoctest; setup = :(using TensorNetworkAD)
julia> TensorNetworkAD.num_grad(x -> x * x, 3) ≈ 6
true
```
"
function num_grad(f, K; δ::Real=1e-5)
    if eltype(K) == ComplexF64
        (f(K + δ / 2) - f(K - δ / 2)) / δ + 
            (f(K + δ / 2 * 1.0im) - f(K - δ / 2 * 1.0im)) / δ * 1.0im
    else
        (f(K + δ / 2) - f(K - δ / 2)) / δ
    end
end

@doc raw"
    num_grad(f, K::AbstractArray; [δ = 1e-5])
    
return the numerical gradient of `f` for each element of `K`.

# example

```jldoctest; setup = :(using TensorNetworkAD, LinearAlgebra)
julia> TensorNetworkAD.num_grad(tr, rand(2,2)) ≈ I
true
```
"
function num_grad(f, a::AbstractArray; δ::Real=1e-5)
    b = Array(copy(a))
    df = map(CartesianIndices(b)) do i
        foo = x -> (ac = copy(b); ac[i] = x; f(_arraytype(a)(ac)))
        num_grad(foo, b[i], δ=δ)
    end
    return _arraytype(a)(df)
end


# function num_grad(f, a::U1Array, sitetype; δ::Real=1e-5)
#     b = Array(copy(a))
#     intype = _arraytype(a.tensor)
#     df = copy(a)
#     Adir = getdir(a)
#     for i in CartesianIndices(b)
#         qn = map(i->indextoqn(sitetype, i), i.I)
#         qnsum = A.ifZ2 ? sum(qn) % 2 : sum(qn .* Adir)
#         if qnsum == 0
#             foo = x -> (ac = copy(b); ac[i] = x; f(intype(ac)))
#             df[i] = num_grad(foo, b[i], δ=δ)
#         end
#     end
#     return intype(deletezeroblock(df))
# end

# patch since it's currently broken otherwise
function ChainRulesCore.rrule(::typeof(Base.typed_hvcat), ::Type{T}, rows::Tuple{Vararg{Int}}, xs::S...) where {T,S}
    y = Base.typed_hvcat(T, rows, xs...)
    function back(ȳ)
        return NoTangent(), NoTangent(), NoTangent(), permutedims(ȳ)...
    end
    return y, back
end

# improves performance compared to default implementation, also avoids errors
# with some complex arrays
function ChainRulesCore.rrule(::typeof(norm), A::AbstractArray{<:Number})
    n = norm(A)
    function back(Δ)
        return NoTangent(), Δ * A / (n + eps(0f0))
    end
    return n, back
end

function ChainRulesCore.rrule(::typeof(Base.sqrt), A::AbstractArray)
    As = Base.sqrt(A)
    function back(dAs)
        dA =  As' \ dAs ./2 
        return NoTangent(), dA
    end
    return As, back
end


@adjoint function reshape(A::U1Array, a::Int...)
    function back(dAr)
        @assert A.qn == dAr.qn
        # s = map(size, A.tensor) 
        # dAtensor = map((x, y) -> reshape(x, y), dAr.tensor, s)
        return U1Array(A.qn, A.dir, dAr.tensor, A.size, A.dims, A.division, A.ifZ2), a...
    end
    return reshape(A, a...), back
end

@adjoint *(A::AbstractSymmetricArray, B::AbstractSymmetricArray) = A * B, dC -> (dC * B', A' * dC)

@adjoint adjoint(A::AbstractSymmetricArray) = adjoint(A), djA -> (adjoint(djA), )

@adjoint conjM(A::AbstractArray) = conjM(A), dA -> (conjM(dA), )

ChainRulesCore.rrule(::typeof(asArray), sitetype::AbstractSiteType, A::AbstractSymmetricArray) = asArray(sitetype, A), dAt -> (NoTangent(), NoTangent(), asSymmetryArray(dAt, Val(getsymmetry(A)), sitetype; dir = getdir(A)))

ChainRulesCore.rrule(::typeof(asSymmetryArray), A::AbstractArray, symmetry, sitetype; kwarg...) = asSymmetryArray(A, symmetry, sitetype; kwarg...), dAt -> (NoTangent(), asArray(sitetype, dAt), NoTangent(), NoTangent()...)

function ChainRulesCore.rrule(::typeof(U1Array), qn::Vector{Vector{Int}}, dir::Vector{Int}, tensor::AbstractArray{T}, size::Tuple{Vararg{Int, N}}, dims::Vector{Vector{Int}}, division::Int, ifZ2::Bool) where {T,N}
    function back(dA)
        @assert qn == dA.qn
        return NoTangent(), NoTangent(), NoTangent(), dA.tensor, NoTangent(), NoTangent(), NoTangent(), NoTangent()...
    end
    U1Array(qn, dir, tensor, size, dims, division, ifZ2), back
end

# function ChainRulesCore.rrule(::typeof(symmetryreshape), A::AbstractArray, s...; kwarg...)
#     reA, choosesilces, chooseinds = symmetryreshape(A, s...; kwarg...)
#     back = dA -> (NoTangent(), symmetryreshape(dA, s; choosesilces = choosesilces, chooseinds = chooseinds, reqn = A.qn, redims = A.dims), NoTangent()...)
#     (reA, choosesilces, chooseinds), back
# end

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

# U1
function ChainRulesCore.rrule(::typeof(qrpos), A::U1Array)
    Q, R = qrpos(A)
    function back((dQ, dR))
        dA = copy(A)
        @assert Q.qn == dQ.qn
        # @assert R.qn == dR.qn
        Qqn, Qdir, Qdims, Qdiv = Q.qn, Q.dir, Q.dims, Q.division
        Rqn, Rdims = R.qn, R.dims
        Abdiv = blockdiv(A.dims)
        Qbdiv = blockdiv(Qdims)
        Qtensor = [reshape(@view(Q.tensor[Qbdiv[i]]), prod(Qdims[i][1:Qdiv]), prod(Qdims[i][Qdiv+1:end])) for i in 1:length(Qbdiv)]
        dQtensor = [reshape(@view(dQ.tensor[Qbdiv[i]]), prod(Qdims[i][1:Qdiv]), prod(Qdims[i][Qdiv+1:end])) for i in 1:length(Qbdiv)]

        Rbdiv = blockdiv(Rdims)
        Rtensor = [reshape(@view(R.tensor[Rbdiv[i]]), Rdims[i]...) for i in 1:length(Rbdiv)]
        if dR == ZeroTangent()
            dRtensor = ZeroTangent()
        else
            dRtensor = [reshape(@view(dR.tensor[Rbdiv[i]]), Rdims[i]...) for i in 1:length(Rbdiv)]
        end
        qs = A.ifZ2 ? map(x->sum(x[A.division+1:end]) % 2, A.qn) : map(x->sum(x[A.division+1:end] .* A.dir[A.division+1:end]), A.qn)
        for q in unique(qs)
            blockbackQR!(dA::U1Array, Abdiv, Qqn, Qdiv, Qdir, Qtensor, Rqn, Rtensor, dQtensor, dRtensor, q, A.ifZ2)
        end
        return NoTangent(), dA
    end
    return (Q, R), back
end

function blockbackQR!(dA::U1Array, Abdiv, Qqn, Qdiv, Qdir, Qtensor, Rqn, Rtensor, dQtensor, dRtensor, q, ifZ2)
    ind_A = ifZ2 ? findall(x->sum(x[Qdiv+1:end]) % 2 == q, Qqn) : findall(x->sum(x[Qdiv+1:end] .* Qdir[Qdiv+1:end]) == q, Qqn)
    m_j = unique(map(x->x[Qdiv+1:end], Qqn[ind_A]))
    m_i = unique(map(x->x[1:Qdiv], Qqn[ind_A]))

    ind = indexin([[i; m_j[1]] for i in m_i], Qqn)
    dQm = vcat(dQtensor[ind]...)
    Qm = vcat(Qtensor[ind]...)
    blockidims = [size(dQtensor[i],1) for i in ind]
    ind = indexin([[m_j[1]; m_j[1]]], Rqn)[1]
    dRm = dRtensor == ZeroTangent() ? ZeroTangent() : dRtensor[ind]
    Rm = Rtensor[ind]

    M = Array(Rm * dRm' - dQm' * Qm)
    dAm = (UpperTriangular(Rm + I * 1e-12) \ (dQm + Qm * _arraytype(Qm)(Hermitian(M, :L)))' )'

    for i in 1:length(m_i)
        ind = findfirst(x->x in [[m_i[i]; m_j[1]]], dA.qn)
        idim = sum(blockidims[1:i-1])+1:sum(blockidims[1:i])
        CUDA.@allowscalar dA.tensor[Abdiv[ind]] = vec(@view(dAm[idim, :]))
    end
end

function ChainRulesCore.rrule(::typeof(lqpos), A::U1Array)
    L, Q = lqpos(A)
    function back((dL, dQ))
        dA = copy(A)
        @assert Q.qn == dQ.qn
        # @assert L.qn == dL.qn
        Qqn, Qdir, Qdims, Qdiv = Q.qn, Q.dir, Q.dims, Q.division
        Lqn, Ldims = L.qn, L.dims
        Abdiv = blockdiv(A.dims)
        Qbdiv = blockdiv(Qdims)
        Qtensor = [reshape(@view(Q.tensor[Qbdiv[i]]), prod(Qdims[i][1:Qdiv]), prod(Qdims[i][Qdiv+1:end])) for i in 1:length(Qbdiv)]
        dQtensor = [reshape(@view(dQ.tensor[Qbdiv[i]]), prod(Qdims[i][1:Qdiv]), prod(Qdims[i][Qdiv+1:end])) for i in 1:length(Qbdiv)]

        Lbdiv = blockdiv(Ldims)
        Ltensor = [reshape(@view(L.tensor[Lbdiv[i]]), Ldims[i]...) for i in 1:length(Lbdiv)]
        if dL == ZeroTangent()
            dLtensor = ZeroTangent()
        else
            dLtensor = [reshape(@view(dL.tensor[Lbdiv[i]]), Ldims[i]...) for i in 1:length(Lbdiv)]
        end
        qs = A.ifZ2 ? map(x->x[1] % 2, A.qn) : map(x->x[1] * A.dir[1], A.qn)
        for q in unique(qs)
            blockbackLQ!(dA::U1Array, Abdiv, Qqn, Qdiv, Qdir, Qtensor, Lqn, Ltensor, dQtensor, dLtensor, q, A.ifZ2)
        end
        return NoTangent(), dA
    end
    return (L, Q), back
end

function blockbackLQ!(dA::U1Array, Abdiv, Qqn, Qdiv, Qdir, Qtensor, Lqn, Ltensor, dQtensor, dLtensor, q, ifZ2)
    ind_A = ifZ2 ? findall(x->x[1] % 2 == q, Qqn) : findall(x->x[1] * Qdir[1] == q, Qqn)
    m_j = unique(map(x->x[Qdiv+1:end], Qqn[ind_A]))
    m_i = unique(map(x->x[1], Qqn[ind_A]))

    ind = indexin([[m_i[1]; j] for j in m_j], Qqn)
    dQm = hcat(dQtensor[ind]...)
    Qm = hcat(Qtensor[ind]...)
    blockjdims = [size(dQtensor[i],2) for i in ind]
    ind = indexin([[m_i[1]; m_i[1]]], Lqn)[1]
    dLm = dLtensor == ZeroTangent() ? ZeroTangent() : dLtensor[ind]
    Lm = Ltensor[ind]
    
    M = Array(Lm' * dLm - dQm * Qm')
    dAm = LowerTriangular(Lm + I * 1e-12)' \ (dQm + _arraytype(Qm)(Hermitian(M, :L)) * Qm)

    for j in 1:length(m_j)
        ind = findfirst(x->x in [[m_i[1]; m_j[j]]], dA.qn)
        jdim = sum(blockjdims[1:j-1])+1:sum(blockjdims[1:j])
        CUDA.@allowscalar dA.tensor[Abdiv[ind]] = vec(@view(dAm[:, jdim]))
    end
end

"""
    dAMmap!(Au, Ad, M, L, R, i)

```
               ┌──  Auᵢⱼ ──┐ 
               │     │     │ 
dMᵢⱼ    =  -   L ──     ── R
               │     │     │ 
               └── Adᵢ₊₁ⱼ──┘ 

               ┌──       ──┐ 
               │     │     │ 
dAuᵢⱼ   =  -   L ── Mᵢⱼ  ──R
               │     │     │ 
               └── Adᵢ₊₁ⱼ──┘ 

               ┌──  Auᵢⱼ ──┐       a ────┬──── c     
               │     │     │       │     b     │    
dAdᵢ₊₁ⱼ =  -   L ─── Mᵢⱼ ──R       ├─ d ─┼─ e ─┤     
               │     │     │       │     g     │  
               └──       ──┘       f ────┴──── h  

```
"""
function dAMmap!(dAui, dAdir, dMi, Aui, Adir, Mi, L, R)
    Nj = size(dAui, 1)
    for j in 1:Nj
        dAui[j]  = -conj!(ein"((adf,fgh),dgeb),ceh -> abc"(L[j], Adir[j], Mi[j],   R[j]))
        dAdir[j] = -conj!(ein"((adf,abc),dgeb),ceh -> fgh"(L[j], Aui[j],  Mi[j],   R[j]))
         dMi[j]  = -conj!(ein"(adf,abc),(fgh,ceh) -> dgeb"(L[j], Aui[j],  Adir[j], R[j]))
    end
    return dAui, dAdir, dMi
end

"""
    ξLm = ξLmap(ARu, ARd, M, FR, i)

```
    ── ALuᵢⱼ  ──┐          ──┐          a ────┬──── c 
        │       │            │          │     b     │ 
    ── Mᵢⱼ   ──ξLᵢⱼ₊₁  =   ──ξLᵢⱼ       ├─ d ─┼─ e ─┤ 
        │       │            │          │     g     │ 
    ── ALdᵢ₊₁ⱼ ─┘          ──┘          f ────┴──── h 
```
"""
function ξLmap(ALui, ALdir, Mi, ξLi)
    Nj = size(ALui, 1)
    ξLijr = circshift(ξLi, -1)
    [ein"((ceh,abc),dgeb),fgh -> adf"(ξLijr[j],ALui[j],Mi[j],ALdir[j]) for j in 1:Nj]
end

similar(a::Vector{<:AbstractArray}) =similar.(a)

function ChainRulesCore.rrule(::typeof(leftenv), ALu, ALd, M, FL; ifobs = false, kwargs...)
    λL, FL = leftenv(ALu, ALd, M, FL; ifobs = ifobs, kwargs...)
    Ni,Nj = size(M)
    function back((dλL, dFL))
        dALu = zero.(ALu)
        dALd = zero.(ALd)
        dM   = zero.(M)
        @inbounds for i = 1:Ni
            ir = ifobs ? Ni+1-i : mod1(i+1, Ni)
            dFL[i,:] .-= sum([dFL[i,j] !== ZeroTangent() && Array(ein"abc,abc ->"(conj(FL[i,j]), dFL[i,j]))[] for j in 1:Nj]) * FL[i,:]
            ξL, info = linsolve(X -> ξLmap(ALu[i,:], ALd[ir,:], M[i,:], X), conj(dFL[i,:]), -λL[i], 1; maxiter = 100)
            info.converged == 0 && @warn "ad's linsolve not converge"
            ξL .= circshift(ξL, -1)
            dALu[i,:], dALd[ir,:], dM[i,:] = dAMmap!(dALu[i,:], dALd[ir,:], dM[i,:], ALu[i,:], ALd[ir,:], M[i,:], FL[i,:], ξL)
        end
        return NoTangent(), dALu, dALd, dM, NoTangent()
    end
    return (λL, FL), back
end

"""
    ξRm = ξRmap(ARu, ARd, M, FL)

```
  ┌──       ┌──  ARuᵢⱼ  ──                     a ────┬──── c 
  │         │     │                            │     b     │ 
ξRᵢⱼ   =  ξRᵢⱼ₋₁─ Mᵢⱼ   ──                     ├─ d ─┼─ e ─┤ 
  │         │     │                            │     g     │ 
  └──       └──  ARdᵢ₊₁ⱼ ─                     f ────┴──── h 
```
"""

function ξRmap(ARui, ARdir, Mi, ξRi)
    Nj = size(ARui, 1)
    ξRijr = circshift(ξRi, 1)
    [ein"((adf,abc),dgeb),fgh -> ceh"(ξRijr[j],ARui[j],Mi[j],ARdir[j]) for j in 1:Nj]
end

function ChainRulesCore.rrule(::typeof(rightenv), ARu, ARd, M, FR; ifobs = false, kwargs...)
    λR, FR = rightenv(ARu, ARd, M, FR; ifobs = ifobs, kwargs...)
    Ni,Nj = size(M)
    function back((dλ, dFR))
        dARu = zero.(ARu)
        dARd = zero.(ARd)
        dM   = zero.(M)
        @inbounds for i = 1:Ni
            ir = ifobs ? Ni+1-i : mod1(i+1, Ni)
            dFR[i,:] .-= sum([dFR[i,j] !== ZeroTangent() && Array(ein"abc,abc ->"(conj(FR[i,j]), dFR[i,j]))[] for j in 1:Nj]) * FR[i,:]
            ξR, info = linsolve(X -> ξRmap(ARu[i,:], ARd[ir,:], M[i,:], X), conj(dFR[i,:]), -λR[i], 1; maxiter = 100)
            info.converged == 0 && @warn "ad's linsolve not converge"
            ξR .= circshift(ξR, 1)
            dARu[i,:], dARd[ir,:], dM[i,:] = dAMmap!(dARu[i,:], dARd[ir,:], dM[i,:], ARu[i,:], ARd[ir,:], M[i,:], ξR, FR[i,:])
        end
        return NoTangent(), dARu, dARd, dM, NoTangent()
    end
    return (λR, FR), back
end

"""
    ξACmap(ξAC, FL, FR, M, j)

```
                                                            a ────┬──── c 
                                                            │     b     │
│        │         │          │        │         │          ├─ d ─┼─ e ─┤ 
└───────ξACᵢⱼ ─────┘    =     FLᵢⱼ ─── Mᵢⱼ ───── FRᵢⱼ       │     g     │ 
                              │        │         │          f ────┴──── h     
                              └──────ξACᵢ₊₁ⱼ ────┘                                    
                         
```
"""
function ξACmap(ξACj, FLj, FRj, Mj)
    Ni = size(FLj,1)
    ξACirj = circshift(ξACj, -1)
    [ein"((adf,fgh),dgeb),ceh -> abc"(FLj[i],ξACirj[i],Mj[i],FRj[i]) for i in 1:Ni]
end

"""
    ACdFMmap(FLj, Mi, FRj, AC, ACd, i, II)

```

               ┌──── ACu  ───┐ 
               │      │      │ 
dFLᵢⱼ    =          ──Mᵢⱼ ── FRᵢⱼ
               │      │      │
               └───  ACd ────┘ 

               ┌──── ACu  ───┐ 
               │      │      │ 
dMᵢⱼ    =      FLᵢⱼ ─   ─── FRᵢⱼ
               │      │      │
               └───  ACd ────┘ 

               ┌──── ACu  ───┐          a ────┬──── c    
               │      │      │          │     b     │  
dFRᵢⱼ    =     FLᵢⱼ ──Mᵢⱼ ──            ├─ d ─┼─ e ─┤ 
               │      │      │          │     g     │  
               └───  ACd ────┘          f ────┴──── h   
```
"""
function dFMmap!(dFLj, dMj, dFRj, FLj, Mj, FRj, ACuj, ACdj)
    Ni = size(FLj,1)
    for i in 1:Ni
        dFLj[i] = -conj!(ein"((abc,ceh),dgeb),fgh -> adf"(ACuj[i], FRj[i],  Mj[i], ACdj[i]))
         dMj[i] = -conj!(ein"(abc,adf),(ceh,fgh) -> dgeb"(ACuj[i], FLj[i], FRj[i], ACdj[i]))
        dFRj[i] = -conj!(ein"((abc,adf),dgeb),fgh -> ceh"(ACuj[i], FLj[i],  Mj[i], ACdj[i]))
    end
    return dFLj, dMj, dFRj
end

function ChainRulesCore.rrule(::typeof(ACenv), AC, FL, M, FR; kwargs...)
    λAC, AC = ACenv(AC, FL, M, FR)
    Ni, Nj = size(M)
    function back((dλ, dAC))
        dFL = zero.(FL)
        dM  = zero.(M)
        dFR = zero.(FR)
        @inbounds for j = 1:Nj
            dAC[:,j] .-= sum([dAC[i,j] !== ZeroTangent() && Array(ein"abc,abc ->"(conj(AC[i,j]), dAC[i,j]))[] for i in 1:Ni]) * AC[:,j]
            ξAC, info = linsolve(X -> ξACmap(X, FL[:,j], FR[:,j], M[:,j]), conj(dAC[:,j]), -λAC[j], 1; maxiter = 100)
            info.converged == 0 && @warn "ad's linsolve not converge"
            ξAC .= circshift(ξAC, -1)
            dFL[:,j], dM[:,j], dFR[:,j] = dFMmap!(dFL[:,j], dM[:,j], dFR[:,j], FL[:,j], M[:,j], FR[:,j], AC[:,j], ξAC)
        end
        return NoTangent(), NoTangent(), dFL, dM, dFR
    end
    return (λAC, AC), back
end

"""
    ξCmap(ξC, FL, FR, j)

```               
                                                    a ─── b
                        │                │          │     │       
│                │      FLᵢⱼ₊₁ ───────  FRᵢⱼ        ├─ c ─┤   
└────── Cᵢⱼ ─────┘  =   │                │          │     │     
                        └───── Cᵢ₊₁ⱼ ────┘          d ─── e  
```
"""
function ξCmap(ξCj, FLjr, FRj)
    Ni = size(FLjr,1)
    ξCirj = circshift(ξCj, -1)
    [ein"(acd,de),bce -> ab"(FLjr[i],ξCirj[i],FRj[i]) for i in 1:Ni]
end

"""
    CdFMmap(FLj, FRj, C, Cd, i, II)

```
               ┌──── Cu ────┐ 
               │            │ 
dFLᵢⱼ₊₁ =  -        ────── FRᵢⱼ
               │            │
               └──── Cd ────┘ 
               ┌──── Cu ────┐          a ─── b   
               │            │          │     │  
dFRᵢⱼ =    -   FLᵢⱼ₊₁──────            ├─ c ─┤ 
               │            │          │     │  
               └──── Cd ────┘          d ─── e  

```
"""
function dFMmap!(dFLjr, dFRj, FLjr, FRj, Cu, Cd)
    Ni = size(FLjr,1)
    for i in 1:Ni
        dFLjr[i] = -conj!(ein"(ab,bce),de -> acd"(Cu[i],  FRj[i], Cd[i]))
        dFRj[i]  = -conj!(ein"(ab,acd),de -> bce"(Cu[i], FLjr[i], Cd[i]))
    end
    return dFLjr, dFRj
end

function ChainRulesCore.rrule(::typeof(Cenv), C, FL, FR; kwargs...)
    λC, C = Cenv(C, FL, FR)
    Ni, Nj = size(C)
    function back((dλ, dC))
        dFL = zero.(FL)
        dFR = zero.(FR)
        for j = 1:Nj
            jr = mod1(j+1, Nj)
            dC[:,j] .-= sum([dC[i,j] !== ZeroTangent() && Array(ein"ab,ab ->"(conj(C[i,j]), dC[i,j]))[] for i in 1:Ni]) * C[:,j]
            ξC, info = linsolve(X -> ξCmap(X, FL[:,jr], FR[:,j]), conj(dC[:, j]), -λC[j], 1; maxiter = 100)
            info.converged == 0 && @warn "ad's linsolve not converge"
            ξC .= circshift(ξC, -1)
            dFL[:,jr], dFR[:,j] = dFMmap!(dFL[:,jr], dFR[:,j], FL[:,jr], FR[:,j], C[:, j], ξC)
        end
        return NoTangent(), NoTangent(), dFL, dFR
    end
    return (λC, C), back
end

function ChainRulesCore.rrule(::typeof(U1reshape), A::U1Array, s::Int...; kwarg...)
    reA, reinfo = U1reshape(A, s; kwarg...)
    function back((dAr,))
        return NoTangent(), U1reshape(dAr, size(A); reinfo = reinfo)[1], s...
    end
    return (reA, reinfo), back
end

@adjoint function tr(A::U1Array)
    function back(dtrA)
        dA = zero(A)
        atype = _arraytype(A.tensor)
        Abdiv = blockdiv(A.dims)
        for i in 1:length(Abdiv)
            dA.tensor[Abdiv[i]] = vec(atype(Matrix(I,dA.dims[i]...) * dtrA))
        end
        return (dA, )
    end
    tr(A), back
end


function ChainRulesCore.rrule(::typeof(dtr), A::U1Array{T,N}) where {T,N}
    function back(dtrA)
        atype = _arraytype(A.tensor)
        Aqn = A.qn
        Adims = A.dims
        dA = zero(A)
        Abdiv = blockdiv(Adims)
        for i in 1:length(Abdiv)
            if Aqn[i][1] == Aqn[i][3] && Aqn[i][2] == Aqn[i][4]
                d1 = Adims[i][1]
                d2 = Adims[i][2]
                dA.tensor[Abdiv[i]] = vec(atype(dtrA * ein"ab, cd -> acbd"(Matrix(I,d1,d1), Matrix(I,d2,d2))))
                # for j = 1:d1, k = 1:d2
                #     dA.tensor[i][j,k,j,k] = dtrA
                # end
            end
        end
        return NoTangent(), atype(deletezeroblock(dA))
    end
    dtr(A), back
end

function ChainRulesCore.rrule(::typeof(dtr), A::AbstractArray{T,N}) where {T,N}
    function back(dtrA)
        atype = _arraytype(A)
        s = size(A)
        dA = zeros(T, s...)
        for i = 1:s[1], j = 1:s[2]
            dA[i,j,i,j] = dtrA
        end
        return NoTangent(), atype(dA)
    end
    dtr(A), back
end