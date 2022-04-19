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
Zygote.@nograd FLint
Zygote.@nograd FRint
Zygote.@nograd leftorth
Zygote.@nograd rightorth
Zygote.@nograd ALCtoAC
Zygote.@nograd LRtoC
Zygote.@nograd initialA
Zygote.@nograd save
Zygote.@nograd load
Zygote.@nograd Random.seed!
Zygote.@nograd randinitial
Zygote.@nograd show_every_count
Zygote.@nograd _initializect_square
Zygote.@nograd U1reshapeinfo

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

function num_grad(f, a::Z2Array; δ::Real=1e-5)
    b = Array(copy(a))
    intype = _arraytype(a.tensor[1])
    df = copy(a)
    bits = map(x -> ceil(Int, log2(x)), size(a))
    for i in CartesianIndices(b)
        parity = collect(sum.(bitarray.(i.I .- 1, bits))) .% 2
        if sum(parity) % 2 == 0
            foo = x -> (ac = copy(b); ac[i] = x; f(intype(ac)))
            df[i] = num_grad(foo, b[i], δ=δ)
        end
    end
    return intype(df)
end

function num_grad(f, a::U1Array; δ::Real=1e-5)
    b = Array(copy(a))
    intype = _arraytype(a.tensor[1])
    df = copy(a)
    bits = map(x -> ceil(Int, log2(x)), size(a))
    Adir = getdir(a)
    for i in CartesianIndices(b)
        qn = collect(sum.(bitarray.(i.I .- 1, bits))) 
        if sum(qn.*Adir) == 0
            foo = x -> (ac = copy(b); ac[i] = x; f(intype(ac)))
            df[i] = num_grad(foo, b[i], δ=δ)
        end
    end
    return intype(df)
end

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

# function ChainRulesCore.rrule(::typeof(reshape), A::Z2Array{T,N}, a::Int...)
#     function back(dAr)
#         exchangeind = indexin(A.parity, dAr.parity)
#         s = size.(A.tensor)
#         dAtensor = reshape.(dAr.tensor[exchangeind], s)
#         return Z2Array(A.parity, dAtensor, N, A.division), a...
#     end
#     return reshape(A, a...), back
# end
    
@adjoint function reshape(A::Z2Array, a::Int...)
    function back(dAr)
        exchangeind = indexin(A.parity, dAr.parity)
        s = map(size, A.tensor) 
        dAtensor = map((x, y) -> reshape(x, y), dAr.tensor[exchangeind], s)
        return Z2Array(A.parity, dAtensor, A.size, A.dims, A.division), a...
    end
    return reshape(A, a...), back
end

@adjoint function reshape(A::U1Array, a::Int...)
    function back(dAr)
        exchangeind = indexin(A.qn, dAr.qn)
        s = map(size, A.tensor) 
        dAtensor = map((x, y) -> reshape(x, y), dAr.tensor[exchangeind], s)
        return U1Array(A.qn, A.dir, dAtensor, A.size, A.dims, A.division), a...
    end
    return reshape(A, a...), back
end

@adjoint *(A::AbstractSymmetricArray, B::AbstractSymmetricArray) = A * B, dC -> (dC * B', A' * dC)

@adjoint adjoint(A::AbstractSymmetricArray) = adjoint(A), djA -> (adjoint(djA), )

ChainRulesCore.rrule(::typeof(asArray), A::AbstractSymmetricArray) = asArray(A), dAt -> (NoTangent(), asSymmetryArray(dAt, Val(getsymmetry(A)); dir = getdir(A)))

ChainRulesCore.rrule(::typeof(asSymmetryArray), A::AbstractArray, symmetry; kwarg...) = asSymmetryArray(A, symmetry; kwarg...), dAt -> (NoTangent(), asArray(dAt), NoTangent()...)

# function ChainRulesCore.rrule(::typeof(symmetryreshape), A::AbstractArray, s...; kwarg...)
#     reA, choosesilces, chooseinds = symmetryreshape(A, s...; kwarg...)
#     back = dA -> (NoTangent(), symmetryreshape(dA, s; choosesilces = choosesilces, chooseinds = chooseinds, reqn = A.qn, redims = A.dims), NoTangent()...)
#     (reA, choosesilces, chooseinds), back
# end


# function ChainRulesCore.rrule(::typeof(Z2Array), parity::Vector{<:Vector{Int}}, tensor::Vector{<:AbstractArray{T}}, N::Tuple{Vararg}, division::Int) where {T}
#     function back(dy) 
#         @show dy.parity parity
#         Z2Array(dy.parity, dy.tensor, N, dy.division)
#     end
#     Z2Array(parity, tensor, N, division),  back
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

# Z2
function ChainRulesCore.rrule(::typeof(qrpos), A::Z2Array)
    Q, R = qrpos(A)
    function back((dQ, dR))
        dA = copy(A)
        @assert Q.parity == dQ.parity
        bulkbackQR!(A, dA, Q, R, dQ, dR, 0)
        bulkbackQR!(A, dA, Q, R, dQ, dR, 1)
        return NoTangent(), dA
    end
    return (Q, R), back
end

function bulkbackQR!(A::Z2Array, dA, Q, R, dQ, dR, p)
    div = dQ.division
    ind_A = findall(x->sum(x[div+1:end]) % 2 == p, dQ.parity)
    m_j = unique(map(x->x[div+1:end], dQ.parity[ind_A]))
    m_i = unique(map(x->x[1:div], dQ.parity[ind_A]))

    ind = [findfirst(x->x in [[i; m_j[1]]], dQ.parity) for i in m_i]
    dQm = vcat(dQ.tensor[ind]...)
    Qm = vcat(Q.tensor[ind]...)
    bulkidims = [size(dQ.tensor[i],1) for i in ind]
    bulkjdims = [size(dQm, 2)]
    ind = findfirst(x->x in [[m_j[1]; m_j[1]]], R.parity)
    dRm = dR == ZeroTangent() ? ZeroTangent() : dR.tensor[ind]
    Rm = R.tensor[ind]
    
    M = Array(Rm * dRm' - dQm' * Qm)
    dAm = (UpperTriangular(Rm + I * 1e-12) \ (dQm + Qm * _arraytype(Qm)(Hermitian(M, :L)))' )'

    for i in 1:length(m_i), j in 1:length(m_j)
        ind = findfirst(x->x in [[m_i[i]; m_j[j]]], A.parity)
        idim, jdim = sum(bulkidims[1:i-1])+1:sum(bulkidims[1:i]), sum(bulkjdims[1:j-1])+1:sum(bulkjdims[1:j])
        CUDA.@allowscalar dA.tensor[ind] = dAm[idim, jdim]
    end
end

function ChainRulesCore.rrule(::typeof(lqpos), A::Z2Array)
    L, Q = lqpos(A)
    function back((dL, dQ))
        dA = copy(A)
        @assert Q.parity == dQ.parity
        bulkbackLQ!(A, dA, L, Q, dL, dQ, 0)
        bulkbackLQ!(A, dA, L, Q, dL, dQ, 1)
        return NoTangent(), dA
    end
    return (L, Q), back
end

function bulkbackLQ!(A::Z2Array, dA, L, Q, dL, dQ, p)
    div = dQ.division
    ind_A = findall(x->sum(x[div+1:end]) % 2 == p, dQ.parity)
    m_j = unique(map(x->x[div+1:end], dQ.parity[ind_A]))
    m_i = unique(map(x->x[1:div], dQ.parity[ind_A]))

    ind = [findfirst(x->x in [[m_i[1]; j]], dQ.parity) for j in m_j]
    dQm = hcat(dQ.tensor[ind]...)
    Qm = hcat(Q.tensor[ind]...)
    bulkidims = [size(dQm, 1)]
    bulkjdims = [size(dQ.tensor[i],2) for i in ind]
    ind = findfirst(x->x in [[m_i[1]; m_i[1]]], L.parity)
    dLm = dL == ZeroTangent() ? ZeroTangent() : dL.tensor[ind]
    Lm = L.tensor[ind]
    
    M = Array(Lm' * dLm - dQm * Qm')
    dAm = LowerTriangular(Lm + I * 1e-12)' \ (dQm + _arraytype(Qm)(Hermitian(M, :L)) * Qm)

    for i in 1:length(m_i), j in 1:length(m_j)
        ind = findfirst(x->x in [[m_i[i]; m_j[j]]], A.parity)
        idim, jdim = sum(bulkidims[1:i-1])+1:sum(bulkidims[1:i]), sum(bulkjdims[1:j-1])+1:sum(bulkjdims[1:j])
        CUDA.@allowscalar dA.tensor[ind] = dAm[idim, jdim]
    end
end

# U1
function ChainRulesCore.rrule(::typeof(qrpos), A::U1Array)
    Q, R = qrpos(A)
    function back((dQ, dR))
        dA = copy(A)
        @assert Q.qn == dQ.qn
        for q in unique(map(x->sum(x[A.division+1:end] .* A.dir[A.division+1:end]), A.qn))
            bulkbackQR!(A::U1Array, dA, Q, R, dQ, dR, q)
        end
        return NoTangent(), dA
    end
    return (Q, R), back
end

function bulkbackQR!(A::U1Array, dA, Q, R, dQ, dR, q)
    div = dQ.division
    ind_A = findall(x->sum(x[div+1:end].* dQ.dir[dQ.division+1:end]) == q, dQ.qn)
    m_j = unique(map(x->x[div+1:end], dQ.qn[ind_A]))
    m_i = unique(map(x->x[1:div], dQ.qn[ind_A]))

    ind = [findfirst(x->x in [[i; m_j[1]]], dQ.qn) for i in m_i]
    dQm = vcat(dQ.tensor[ind]...)
    Qm = vcat(Q.tensor[ind]...)
    bulkidims = [size(dQ.tensor[i],1) for i in ind]
    bulkjdims = [size(dQm, 2)]
    ind = findfirst(x->x in [[m_j[1]; m_j[1]]], R.qn)
    dRm = dR == ZeroTangent() ? ZeroTangent() : dR.tensor[ind]
    Rm = R.tensor[ind]
    
    M = Array(Rm * dRm' - dQm' * Qm)
    dAm = (UpperTriangular(Rm + I * 1e-12) \ (dQm + Qm * _arraytype(Qm)(Hermitian(M, :L)))' )'

    for i in 1:length(m_i), j in 1:length(m_j)
        ind = findfirst(x->x in [[m_i[i]; m_j[j]]], A.qn)
        idim, jdim = sum(bulkidims[1:i-1])+1:sum(bulkidims[1:i]), sum(bulkjdims[1:j-1])+1:sum(bulkjdims[1:j])
        CUDA.@allowscalar dA.tensor[ind] = dAm[idim, jdim]
    end
end

function ChainRulesCore.rrule(::typeof(lqpos), A::U1Array)
    L, Q = lqpos(A)
    function back((dL, dQ))
        dA = copy(A)
        @assert Q.qn == Q.qn
        for q in unique(map(x->x[1] * A.dir[1], A.qn))
            bulkbackLQ!(A, dA, L, Q, dL, dQ, q)
        end
        return NoTangent(), dA
    end
    return (L, Q), back
end

function bulkbackLQ!(A::U1Array, dA, L, Q, dL, dQ, q)
    div = dQ.division
    ind_A = findall(x->x[1] * dQ.dir[1] == q, dQ.qn)
    m_j = unique(map(x->x[div+1:end], dQ.qn[ind_A]))
    m_i = unique(map(x->x[1], dQ.qn[ind_A]))

    ind = [findfirst(x->x in [[m_i[1]; j]], dQ.qn) for j in m_j]
    dQm = hcat(dQ.tensor[ind]...)
    Qm = hcat(Q.tensor[ind]...)
    bulkidims = [size(dQm, 1)]
    bulkjdims = [size(dQ.tensor[i],2) for i in ind]
    ind = findfirst(x->x in [[m_i[1]; m_i[1]]], L.qn)
    dLm = dL == ZeroTangent() ? ZeroTangent() : dL.tensor[ind]
    Lm = L.tensor[ind]
    
    M = Array(Lm' * dLm - dQm * Qm')
    dAm = LowerTriangular(Lm + I * 1e-12)' \ (dQm + _arraytype(Qm)(Hermitian(M, :L)) * Qm)

    for i in 1:length(m_i), j in 1:length(m_j)
        ind = findfirst(x->x in [[m_i[i]; m_j[j]]], A.qn)
        idim, jdim = sum(bulkidims[1:i-1])+1:sum(bulkidims[1:i]), sum(bulkjdims[1:j-1])+1:sum(bulkjdims[1:j])
        CUDA.@allowscalar dA.tensor[ind] = dAm[idim, jdim]
    end
end

"""
    dAMmap(Ai, Aip, Mi, L, R, j, J)

Aip means Aᵢ₊₁
```
               ┌──  Aᵢⱼ  ── ... ── AᵢJ   ──   ...  ──┐ 
               │     │              │                │ 
dMᵢJ    =     Lᵢⱼ ─ Mᵢⱼ  ── ... ──     ────── ...  ──Rᵢⱼ
               │     │              │                │ 
               └──  Aᵢ₊₁ⱼ  ─... ── Aᵢ₊₁J  ──  ...  ──┘ 

               ┌──  Aᵢⱼ  ── ... ──     ────── ...  ──┐ 
               │     │              │                │ 
dAᵢJ    =     Lᵢⱼ ─ Mᵢⱼ  ── ... ── dMᵢJ  ──── ...  ──Rᵢⱼ
               │     │              │                │ 
               └──  Aᵢ₊₁ⱼ  ─... ── Aᵢ₊₁J  ─── ...  ──┘ 

               ┌──  Aᵢⱼ  ── ... ── AᵢJ  ────  ...  ──┐        a ────┬──── c     
               │     │              │                │        │     b     │    
dAᵢ₊₁J   =     Lᵢⱼ ─ Mᵢⱼ  ── ... ── dMᵢJ ────  ...  ──Rᵢⱼ     ├─ d ─┼─ e ─┤     
               │     │              │                │        │     g     │  
               └──  Aᵢ₊₁ⱼ  ─... ──     ─────  ...  ──┘        f ────┴──── h  

```
"""
function dAMmap(Ai, Aip, Mi, L, R, j, J; ifconj = false)
    Nj = size(Ai, 1)
        NL = (J - j + (J - j < 0) * Nj)
    NR = Nj - NL - 1
    L = copy(L)
    R = copy(R)
    for jj = 1:NL
        jr = j + jj - 1 - (j + jj - 1 > Nj) * Nj
        L = ein"((adf,abc),dgeb),fgh -> ceh"(L, Ai[jr], Mi[jr], Aip[jr])
    end
    for jj = 1:NR
        jr = j - jj + (j - jj < 1) * Nj
        R = ein"((ceh,abc),dgeb),fgh -> adf"(R, Ai[jr], Mi[jr], Aip[jr])
    end
    dAiJ = -ein"((adf,fgh),dgeb),ceh -> abc"(L, Aip[J], Mi[J], R)
    dAipJ = -ein"((adf,abc),dgeb),ceh -> fgh"(L, Ai[J], Mi[J], R)
    dMiJ = -ein"(adf,abc),(fgh,ceh) -> dgeb"(L, Ai[J], Aip[J], R)
    return conj(dAiJ), ifconj ? dAipJ : conj(dAipJ), conj(dMiJ)
end

function ChainRulesCore.rrule(::typeof(leftenv), ALu, ALd, M, FL; kwargs...)
    λL, FL = leftenv(ALu, ALd, M, FL)
    Ni, Nj = size(ALu)
    function back((dλL, dFL))
        dALu = [zero(x) for x in ALu]
        dALd = [zero(x) for x in ALd]
        dM = [zero(x) for x in M]
        for j = 1:Nj, i = 1:Ni
            ir = i + 1 - Ni * (i == Ni)
            jr = j - 1 + Nj * (j == 1)
            dFL[i,j] -= Array(ein"abc,abc ->"(conj(FL[i,j]), dFL[i,j]))[] * FL[i,j]
            ξl, info = linsolve(FR -> FRmap(ALu[i,:], conj(ALd[ir,:]), M[i,:], FR, jr), conj(dFL[i,j]), -λL[i,j], 1; maxiter = 1)
            info.converged == 0 && @warn "ad's linsolve not converge"
            for J = 1:Nj
                dAiJ, dAipJ, dMiJ = dAMmap(ALu[i,:], conj(ALd[ir,:]), M[i,:], FL[i,j], ξl, j, J; ifconj = true)
                dALu[i,J] += dAiJ
                dALd[ir,J] += dAipJ
                dM[i,J] += dMiJ
            end
        end
        return NoTangent(), dALu, dALd, dM, NoTangent()
    end
    return (λL, FL), back
end

function ChainRulesCore.rrule(::typeof(rightenv), ARu, ARd, M, FR; kwargs...)
    λR, FR = rightenv(ARu, ARd, M, FR)
    Ni, Nj = size(ARu)
    function back((dλ, dFR))
        dARu = [zero(x) for x in ARu]
        dARd = [zero(x) for x in ARd]
        dM = [zero(x) for x in M]
        for j = 1:Nj, i = 1:Ni
            ir = i + 1 - Ni * (i == Ni)
            jr = j - 1 + Nj * (j == 1)
            dFR[i,jr] -= Array(ein"abc,abc ->"(conj(FR[i,jr]), dFR[i,jr]))[] * FR[i,jr]
            ξr, info = linsolve(FL -> FLmap(ARu[i,:], conj(ARd[ir,:]), M[i,:], FL, j), conj(dFR[i,jr]), -λR[i,jr], 1; maxiter = 1)
            info.converged == 0 && @warn "ad's linsolve not converge"
            for J = 1:Nj
                dAiJ, dAipJ, dMiJ = dAMmap(ARu[i,:], conj(ARd[ir,:]), M[i,:], ξr, FR[i,jr], j, J; ifconj = true)
                dARu[i,J] += dAiJ
                dARd[ir,J] += dAipJ
                dM[i,J] += dMiJ
            end
        end
        return NoTangent(), dARu, dARd, dM, NoTangent()
    end
    return (λR, FR), back
end

"""
    ACdmap(ACij, FLj, FRj, Mj, II)

```
.        .         .
.        .         .
.        .         .
│        │         │          a ────┬──── c 
FLᵢ₋₁ⱼ ─ Mᵢ₋₁ⱼ ──  FRᵢ₋₁ⱼ     │     b     │
│        │         │          ├─ d ─┼─ e ─┤ 
FLᵢⱼ ─── Mᵢⱼ ───── FRᵢⱼ       │     g     │ 
│        │         │          f ────┴──── h     
└─────── ACᵢⱼ ─────┘
```
"""
function ACdmap(ACij, FLj, FRj, Mj, II)
    Ni = size(FLj,1)
    ACdm = copy(ACij)
    for i=1:Ni
        ir = II-(i-1) + (II-(i-1) < 1)*Ni
        ACdm = ein"((adf,fgh),dgeb),ceh -> abc"(FLj[ir],ACdm,Mj[ir],FRj[ir])
    end
    return ACdm
end

"""
    ACdFMmap(FLj, Mi, FRj, AC, ACd, i, II)

```
               ┌─────  ACᵢⱼ ─────┐ 
               │        │        │ 
              FLᵢⱼ ─── Mᵢⱼ ──── FRᵢⱼ
               │        │        │ 
               ⋮         ⋮        ⋮
               │        │        │
dMIⱼ    =     FLIⱼ ───     ──── FRIⱼ 
               │        │        │
               ⋮         ⋮        ⋮
               │        │        │             
               └─────  ACdᵢ₋₁ⱼ ──┘ 

               ┌─────  ACᵢⱼ ─────┐ 
               │        │        │ 
              FLᵢⱼ ─── Mᵢⱼ  ─── FRᵢⱼ
               │        │        │ 
               ⋮         ⋮        ⋮
               │        │        │
dFLIⱼ   =        ───── MIⱼ ───  FRIⱼ 
               │        │        │
               ⋮         ⋮        ⋮
               │        │        │             
               └─────  ACdᵢ₋₁ⱼ ──┘

               ┌─────  ACᵢⱼ ─────┐ 
               │        │        │ 
              FLᵢⱼ ─── Mᵢⱼ  ─── FRᵢⱼ
               │        │        │ 
               ⋮         ⋮        ⋮
               │        │        │    
dFRIⱼ   =     FLIⱼ ─── MIⱼ ─────       a ────┬──── c   
               │        │        │     │     b     │ 
               ⋮         ⋮        ⋮     ├─ d ─┼─ e ─┤       
               │        │        │     │     g     │          
               └─────  ACdᵢ₋₁ⱼ ──┘     f ────┴──── h  
```
"""
function ACdFMmap(FLj, Mj, FRj, AC, ACd, i, II)
    Ni = size(FLj, 1)
    Nu = (II - i + (II - i < 0) * Ni)
    Nd = Ni - Nu - 1
    AC = copy(AC)
    ACd = copy(ACd)
    for ii = 1:Nu
        ir = i + ii - 1 - (i + ii - 1 > Ni) * Ni
        AC = ein"((adf,abc),dgeb),ceh -> fgh"(FLj[ir], AC, Mj[ir], FRj[ir])
    end
    for ii = 1:Nd
        ir = i - ii + (i - ii < 1) * Ni
        ACd = ein"((adf,fgh),dgeb),ceh -> abc"(FLj[ir], ACd, Mj[ir], FRj[ir])
    end
    dFLIj = -ein"((abc,ceh),dgeb),fgh -> adf"(AC, FRj[II], Mj[II], ACd)
    dMIj = -ein"(adf,abc),(fgh,ceh) -> dgeb"(FLj[II], AC, ACd, FRj[II])
    dFRIj = -ein"((abc,adf),dgeb),fgh -> ceh"(AC, FLj[II], Mj[II], ACd)
    return conj(dFLIj), conj(dMIj), conj(dFRIj)
end

function ChainRulesCore.rrule(::typeof(ACenv), AC, FL, M, FR; kwargs...)
    λAC, AC = ACenv(AC, FL, M, FR)
    Ni, Nj = size(AC)
    function back((dλ, dAC))
        dFL = [zero(x) for x in FL]
        dM = [zero(x) for x in M]
        dFR = [zero(x) for x in FR]
        for j = 1:Nj, i = 1:Ni
            if dAC[i,j] !== nothing
                ir = i - 1 + Ni * (i == 1)
                dAC[i,j] -= Array(ein"abc,abc ->"(conj(AC[i,j]), dAC[i,j]))[] * AC[i,j]
                ξAC, info = linsolve(ACd -> ACdmap(ACd, FL[:,j], FR[:,j], M[:,j], ir), conj(dAC[i,j]), -λAC[i,j], 1; maxiter = 1)
                info.converged == 0 && @warn "ad's linsolve not converge"
                # errAC = ein"abc,abc ->"(AC[i,j], ξAC)[]
                # abs(errAC) > 1e-1 && throw("AC and ξ aren't orthometric. $(errAC) $(info)")
                # @show info ein"abc,abc ->"(AC[i,j], ξAC)[] ein"abc,abc -> "(AC[i,j], dAC[i,j])[]
                for II = 1:Ni
                    dFLIj, dMIj, dFRIj = ACdFMmap(FL[:,j], M[:,j], FR[:,j], AC[i,j], ξAC, i, II)
                    dFL[II,j] += dFLIj
                    dM[II,j] += dMIj
                    dFR[II,j] += dFRIj
                end
            end
        end
        return NoTangent(), NoTangent(), dFL, dM, dFR
    end
    return (λAC, AC), back
end

"""
    Cdmap(Cij, FLj, FRj, II)

```
.                .
.                .
.                .
│                │          
FLᵢ₋₁ⱼ₊₁ ─────  FRᵢ₋₁ⱼ      a ─── b
│                │          │     │       
FLᵢⱼ₊₁ ───────  FRᵢⱼ        ├─ c ─┤   
│                │          │     │     
└────── Cᵢⱼ ─────┘          d ─── e  
```
"""
function Cdmap(Cij, FLjp, FRj, II)
    Ni = size(FLjp,1)
    Cdm = copy(Cij)
    for i=1:Ni
        ir = II-(i-1) + (II-(i-1) < 1)*Ni
        Cdm = ein"(acd,de),bce -> ab"(FLjp[ir],Cdm,FRj[ir])
    end
    return Cdm
end

"""
    CdFMmap(FLj, FRj, C, Cd, i, II)

```
               ┌────  Cᵢⱼ ────┐ 
               │              │ 
              FLᵢⱼ₊₁───────  FRᵢⱼ
               │              │ 
               ⋮               ⋮
               │              │
dFLIⱼ₊₁ =        ──────────  FRIⱼ
               │              │
               ⋮               ⋮
               │              │             
               └──── Cdᵢⱼ ────┘ 

               ┌────  Cᵢⱼ ────┐ 
               │              │ 
              FLᵢⱼ₊₁ ──────  FRᵢⱼ
               │              │ 
               ⋮               ⋮
               │              │
dFRIⱼ   =     FLᵢ₊Iⱼ₊₁ ──────           a ─── b     
               │              │         │     │   
               ⋮               ⋮         ├─ c ─┤    
               │              │         │     │                
               └──── Cdᵢⱼ ────┘         d ─── e  
```
"""
function CdFMmap(FLjp, FRj, C, Cd, i, II)
    Ni = size(FLjp, 1)
    Nu = (II - i + (II - i < 0) * Ni)
    Nd = Ni - Nu - 1
    C = copy(C)
    Cd = copy(Cd)
    for ii = 1:Nu
        ir = i + ii - 1 - (i + ii - 1 > Ni) * Ni
        C = ein"(acd,ab),bce -> de"(FLjp[ir], C, FRj[ir])
    end
    for ii = 1:Nd
        ir = i - ii + (i - ii < 1) * Ni
        Cd = ein"(acd,de),bce -> ab"(FLjp[ir], Cd, FRj[ir])
    end
    dFLIjp = -ein"(ab,bce),de -> acd"(C, FRj[II], Cd)
    dFRIj = -ein"(ab,acd),de -> bce"(C, FLjp[II], Cd)
    return conj(dFLIjp), conj(dFRIj)
end

function ChainRulesCore.rrule(::typeof(Cenv), C, FL, FR; kwargs...)
    λC, C = Cenv(C, FL, FR)
    Ni, Nj = size(C)
    function back((dλ, dC))
        dFL = [zero(x) for x in FL]
        dFR = [zero(x) for x in FR]
        for j = 1:Nj, i = 1:Ni
            if dC[i,j] !== nothing
                ir = i - 1 + Ni * (i == 1)
                jr = j + 1 - (j==Nj) * Nj
                dC[i,j] -= Array(ein"ab,ab ->"(conj(C[i,j]), dC[i,j]))[] * C[i,j]
                ξC, info = linsolve(Cd -> Cdmap(Cd, FL[:,jr], FR[:,j], ir), conj(dC[i,j]), -λC[i,j], 1; maxiter = 1)
                info.converged == 0 && @warn "ad's linsolve not converge"
                # errC = ein"ab,ab ->"(C[i,j], ξC)[]
                # abs(errC) > 1e-1 && throw("C and ξ aren't orthometric. $(errC) $(info)")
                # @show info ein"ab,ab ->"(C[i,j], ξC)[] ein"ab,ab -> "(C[i,j], dC[i,j])[]
                for II = 1:Ni
                    dFLIjp, dFRIj = CdFMmap(FL[:,jr], FR[:,j], C[i,j], ξC, i, II)
                    dFL[II,jr] += dFLIjp
                    dFR[II,j] += dFRIj
                end
            end
        end
        return NoTangent(), NoTangent(), dFL, dFR
    end
    return (λC, C), back
end

function ChainRulesCore.rrule(::typeof(obs_FL), ALu, ALd, M, FL; kwargs...)
    λL, FL = obs_FL(ALu, ALd, M, FL)
    Ni, Nj = size(ALu)
    function back((dλL, dFL))
        dALu = [zero(x) for x in ALu]
        dALd = [zero(x) for x in ALd]
        dM = [zero(x) for x in M]
        for j = 1:Nj, i = 1:Ni
            ir = Ni + 1 - i
            jr = j - 1 + Nj * (j == 1)
            dFL[i,j] -= Array(ein"abc,abc ->"(conj(FL[i,j]), dFL[i,j]))[] * FL[i,j]
            ξl, info = linsolve(FR -> FRmap(ALu[i,:], ALd[ir,:], M[i,:], FR, jr), conj(dFL[i,j]), -λL[i,j], 1; maxiter = 1)
            info.converged == 0 && @warn "ad's linsolve not converge"
            for J = 1:Nj
                dAiJ, dAipJ, dMiJ = dAMmap(ALu[i,:], ALd[ir,:], M[i,:], FL[i,j], ξl, j, J)
                dALu[i,J] += dAiJ
                dALd[ir,J] += dAipJ
                dM[i,J] += dMiJ
            end
        end
        return NoTangent(), dALu, dALd, dM, NoTangent()
    end
    return (λL, FL), back
end

function ChainRulesCore.rrule(::typeof(obs_FR), ARu, ARd, M, FR; kwargs...)
    λR, FR = obs_FR(ARu, ARd, M, FR)
    Ni, Nj = size(ARu)
    function back((dλ, dFR))
        dARu = [zero(x) for x in ARu]
        dARd = [zero(x) for x in ARd]
        dM = [zero(x) for x in M]
        for j = 1:Nj, i = 1:Ni
            ir = Ni + 1 - i
            jr = j - 1 + Nj * (j == 1)
            dFR[i,jr] -= Array(ein"abc,abc ->"(conj(FR[i,jr]), dFR[i,jr]))[] * FR[i,jr]
            ξr, info = linsolve(FL -> FLmap(ARu[i,:], ARd[ir,:], M[i,:], FL, j), conj(dFR[i,jr]), -λR[i,jr], 1; maxiter = 1)
            info.converged == 0 && @warn "ad's linsolve not converge"
            for J = 1:Nj
                dAiJ, dAipJ, dMiJ = dAMmap(ARu[i,:], ARd[ir,:], M[i,:], ξr, FR[i,jr], j, J)
                dARu[i,J] += dAiJ
                dARd[ir,J] += dAipJ
                dM[i,J] += dMiJ
            end
        end
        return NoTangent(), dARu, dARd, dM, NoTangent()
    end
    return (λR, FR), back
end

ChainRulesCore.rrule(::typeof(parity_conserving),T::Union{Array,CuArray}) = parity_conserving(T), dT -> (NoTangent(), parity_conserving(dT))

ChainRulesCore.rrule(::typeof(Z2reshape), A::Z2Array, s::Int...) = Z2reshape(A, s), dAr -> (NoTangent(), Z2reshape(dAr, size(A)), s...)

function ChainRulesCore.rrule(::typeof(U1reshape), A::U1Array, s::Int...; kwarg...)
    reA, reinfo = U1reshape(A, s; kwarg...)
    function back((dAr,))
        return NoTangent(), U1reshape(dAr, size(A); reinfo = reinfo)[1], s...
    end
    return (reA, reinfo), back
end

@adjoint function tr(A::AbstractSymmetricArray)
    function back(dtrA)
        dA = similar(A)
        atype = _arraytype(A.tensor[1])
        for i in 1:length(A.tensor)
            dA.tensor[i] = atype(Matrix(I,dA.dims[i]...)*dtrA)
        end
        return (dA, )
    end
    tr(A), back
end

function ChainRulesCore.rrule(::typeof(dtr), A::AbstractSymmetricArray{T,N}) where {T,N}
    function back(dtrA)
        dir = getdir(A)
        dA = zerosinitial(A, size(A)...; dir = dir)
        s = size(A)
        for i = 1:s[1], j = 1:s[2]
            dA[i,j,i,j] = dtrA # getindex is slow here, need to optimize
        end
        # dA = similar(A)
        # for i in 1:length(A.tensor)
        #     d1 = dA.dims[i][1]
        #     d2 = dA.dims[i][2]
        #     for j = 1:d1, k = 1:d2
        #         dA.tensor[i][j,k,j,k] = dtrA
        #     end
        # end
        return NoTangent(), dA
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