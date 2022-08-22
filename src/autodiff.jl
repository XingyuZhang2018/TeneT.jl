using ChainRulesCore
using KrylovKit
using LinearAlgebra
using Random
using Zygote

export num_grad

Zygote.@nograd StopFunction
Zygote.@nograd error
Zygote.@nograd FLint
Zygote.@nograd FRint
Zygote.@nograd leftorth
Zygote.@nograd rightorth
Zygote.@nograd LRtoC
Zygote.@nograd initialA
Zygote.@nograd save
Zygote.@nograd load
Zygote.@nograd Random.seed!

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
function ChainRulesCore.rrule(::typeof(LinearAlgebra.norm), A::AbstractArray)
    n = norm(A)
    function back(Δ)
        return NoTangent(), Δ .* A ./ (n + eps(0f0)), NoTangent()
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

"""
    dAMmap(Au, Ad, M, L, R, i)

```
               ┌──  Auᵢⱼ ──┐ 
               │     │     │ 
dMᵢⱼ    =  -   L ── Mᵢⱼ ── R
               │     │     │ 
               └── Adᵢ₊₁ⱼ──┘ 

               ┌──  Auᵢⱼ ──┐ 
               │     │     │ 
dAuᵢⱼ   =  -   L ── Mᵢⱼ  ──R
               │     │     │ 
               └── Adᵢ₊₁ⱼ──┘ 

               ┌──  Auᵢⱼ ──┐       a ────┬──── c     
               │     │     │       │     b     │    
dAdᵢ₊₁ⱼ =  -   L ─── Mᵢⱼ ──R       ├─ d ─┼─ e ─┤     
               │     │     │       │     g     │  
               └── Adᵢ₊₁ⱼ──┘       f ────┴──── h  

```
"""
function dAMmap!(dAui, dAdir, dMi, Aui, Adir, Mi, L, R)
    dAui  .= -conj(ein"((adfi,fghi),dgebi),cehi -> abci"(L, Adir, Mi,   R))
    dAdir .= -conj(ein"((adfi,abci),dgebi),cehi -> fghi"(L, Aui,  Mi,   R))
     dMi  .= -conj(ein"(adfi,abci),(fghi,cehi) -> dgebi"(L, Aui,  Adir, R))
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
    ξLijr = circshift(ξLi, (0,0,0,-1))
    ein"((cehi,abci),dgebi),fghi -> adfi"(ξLijr,ALui,Mi,ALdir)
end

function ChainRulesCore.rrule(::typeof(leftenv), ALu, ALd, M, FL; ifobs = false, kwargs...)
    λL, FL = leftenv(ALu, ALd, M, FL; ifobs = ifobs, kwargs...)
    Ni = size(M, 5)
    function back((dλL, dFL))
        dALu = zero(ALu)
        dALd = zero(ALd)
        dM   = zero(M)
        @inbounds for i = 1:Ni
            ir = ifobs ? Ni+1-i : i+1-Ni*(i==Ni)
            @views dFL[:,:,:,i,:] -= Array(ein"abcd,abcd ->"(conj(FL[:,:,:,i,:]), dFL[:,:,:,i,:]))[] * FL[:,:,:,i,:]
            ξL, info = linsolve(X -> ξLmap(ALu[:,:,:,i,:], ALd[:,:,:,ir,:], M[:,:,:,:,i,:], X), conj(dFL[:,:,:,i,:]), -λL[i], 1; maxiter = 1)
            info.converged == 0 && @warn "ad's linsolve not converge"
            ξL = circshift(ξL, (0,0,0,-1))
            @views dAMmap!(dALu[:,:,:,i,:], dALd[:,:,:,ir,:], dM[:,:,:,:,i,:], ALu[:,:,:,i,:], ALd[:,:,:,ir,:], M[:,:,:,:,i,:], FL[:,:,:,i,:], ξL)
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
    ξRijr = circshift(ξRi, (0,0,0,1))
    ein"((adfi,abci),dgebi),fghi -> cehi"(ξRijr,ARui,Mi,ARdir)
end

function ChainRulesCore.rrule(::typeof(rightenv), ARu, ARd, M, FR; ifobs = false, kwargs...)
    λR, FR = rightenv(ARu, ARd, M, FR; ifobs = ifobs, kwargs...)
    Ni = size(M, 5)
    function back((dλ, dFR))
        dARu = zero(ARu)
        dARd = zero(ARd)
        dM   = zero(M)
        @inbounds for i = 1:Ni
            ir = ifobs ? Ni+1-i : i+1-Ni*(i==Ni)
            @views dFR[:,:,:,i,:] -= Array(ein"abcd,abcd ->"(conj(FR[:,:,:,i,:]), dFR[:,:,:,i,:]))[] * FR[:,:,:,i,:]
            ξR, info = linsolve(X -> ξRmap(ARu[:,:,:,i,:], ARd[:,:,:,ir,:], M[:,:,:,:,i,:], X), conj(dFR[:,:,:,i,:]), -λR[i], 1; maxiter = 1)
            info.converged == 0 && @warn "ad's linsolve not converge"
            ξR = circshift(ξR, (0,0,0,1))
            @views dAMmap!(dARu[:,:,:,i,:], dARd[:,:,:,ir,:], dM[:,:,:,:,i,:], ARu[:,:,:,i,:], ARd[:,:,:,ir,:], M[:,:,:,:,i,:], ξR, FR[:,:,:,i,:])
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
    ξACirj = circshift(ξACj, (0,0,0,-1))
    ein"((adfj,fghj),dgebj),cehj -> abcj"(FLj,ξACirj,Mj,FRj)
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
    dFLj .= -conj(ein"((abcj,cehj),dgebj),fghj -> adfj"(ACuj, FRj, Mj, ACdj))
     dMj .= -conj(ein"(abcj,adfj),(cehj,fghj) -> dgebj"(ACuj, FLj,FRj, ACdj))
    dFRj .= -conj(ein"((abcj,adfj),dgebj),fghj -> cehj"(ACuj, FLj, Mj, ACdj))
end

function ChainRulesCore.rrule(::typeof(ACenv), AC, FL, M, FR; kwargs...)
    λAC, AC = ACenv(AC, FL, M, FR)
    Nj = size(M, 5)
    function back((dλ, dAC))
        dFL = zero(FL)
        dM  = zero(M)
        dFR = zero(FR)
        @inbounds for j = 1:Nj
            @views dAC[:,:,:,:,j] -= Array(ein"abcd,abcd ->"(conj(AC[:,:,:,:,j]), dAC[:,:,:,:,j]))[] * AC[:,:,:,:,j]
            ξAC, info = linsolve(X -> ξACmap(X, FL[:,:,:,:,j], FR[:,:,:,:,j], M[:,:,:,:,:,j]), conj(dAC[:,:,:,:,j]), -λAC[j], 1; maxiter = 1)
            info.converged == 0 && @warn "ad's linsolve not converge"
            ξAC = circshift(ξAC, (0,0,0,-1))
            @views dFMmap!(dFL[:,:,:,:,j], dM[:,:,:,:,:,j], dFR[:,:,:,:,j], FL[:,:,:,:,j], M[:,:,:,:,:,j], FR[:,:,:,:,j], AC[:,:,:,:,j], ξAC)
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
    ξCirj = circshift(ξCj, (0,0,-1))
    ein"(acdj,dej),bcej -> abj"(FLjr,ξCirj,FRj)
end

"""
    CdFMmap(FLj, FRj, C, Cd, i, II)

```
               ┌──── Cu ────┐ 
               │            │ 
dFLᵢⱼ₊₁ =           ────── FRᵢⱼ
               │            │
               └──── Cd ────┘ 
               ┌──── Cu ────┐          a ─── b   
               │            │          │     │  
dFRᵢⱼ =        FLᵢⱼ₊₁──────            ├─ c ─┤ 
               │            │          │     │  
               └──── Cd ────┘          d ─── e  

```
"""
function dFMmap!(dFLjr, dFRj, FLjr, FRj, Cu, Cd)
    dFLjr .= -conj(ein"(abj,bcej),dej -> acdj"(Cu, FRj,  Cd))
    dFRj  .= -conj(ein"(abj,acdj),dej -> bcej"(Cu, FLjr, Cd))
end

function ChainRulesCore.rrule(::typeof(Cenv), C, FL, FR; kwargs...)
    λC, C = Cenv(C, FL, FR)
    Nj = size(C, 4)
    function back((dλ, dC))
        dFL = zero(FL)
        dFR = zero(FR)
        for j = 1:Nj
            jr = j + 1 -  Nj * (j==Nj)
            dC[:,:,:,j] -= Array(ein"abc,abc ->"(conj(C[:,:,:,j]), dC[:,:,:,j]))[] * C[:,:,:,j]
            ξC, info = linsolve(X -> ξCmap(X, FL[:,:,:,:,jr], FR[:,:,:,:,j]), conj(dC[:,:,:,j]), -λC[j], 1; maxiter = 1)
            info.converged == 0 && @warn "ad's linsolve not converge"
            ξC = circshift(ξC, (0,0,-1))
            @views dFMmap!(dFL[:,:,:,:,jr], dFR[:,:,:,:,j], FL[:,:,:,:,jr], FR[:,:,:,:,j], C[:,:,:,j], ξC)
        end
        return NoTangent(), NoTangent(), dFL, dFR
    end
    return (λC, C), back
end