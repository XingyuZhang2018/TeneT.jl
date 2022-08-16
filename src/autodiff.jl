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
function dAMmap!(dAu, dAd, dM, Au, Ad, M, L, R, i, ir, j)
    dAu[:,:,:,  i, j] = -conj(ein"((adf,fgh),dgeb),ceh -> abc"(L, Ad[:,:,:,ir,j], M[:,:,:,:,i,j], R))
    dAd[:,:,:,  ir,j] = -conj(ein"((adf,abc),dgeb),ceh -> fgh"(L, Au[:,:,:,i ,j], M[:,:,:,:,i,j], R))
     dM[:,:,:,:,i, j] = -conj(ein"(adf,abc),(fgh,ceh) -> dgeb"(L, Au[:,:,:,i ,j], Ad[:,:,:,ir,j], R))
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
function ξLmap(ALu, ALd, M, ξL, i, ir)
    Nj = size(M, 6)
    ξLm = copy(ξL)
    @inbounds @views for j in 1:Nj
        jr = j + 1 - Nj * (j==Nj)
        ξLm[:,:,:,j] .= ein"((ceh,abc),dgeb),fgh -> adf"(ξL[:,:,:,jr],ALu[:,:,:,i,j],M[:,:,:,:,i,j],ALd[:,:,:,ir,j])
    end
    return ξLm
end

function ChainRulesCore.rrule(::typeof(leftenv), ALu, ALd, M, FL; ifobs = false, kwargs...)
    λL, FL = leftenv(ALu, ALd, M, FL; ifobs = ifobs, kwargs...)
    Ni,Nj = size(M)[end-1:end]
    function back((dλL, dFL))
        dALu = zero(ALu)
        dALd = zero(ALd)
        dM   = zero(M)
        @inbounds for i = 1:Ni
            ir = ifobs ? Ni+1-i : i+1-Ni*(i==Ni)
            dFL[:,:,:,i,:] -= Array(ein"abcd,abcd ->"(conj(FL[:,:,:,i,:]), dFL[:,:,:,i,:]))[] * FL[:,:,:,i,:]
            ξL, info = linsolve(X -> ξLmap(ALu, ALd, M, X, i, ir), conj(dFL[:,:,:,i,:]), -λL[i], 1; maxiter = 1)
            info.converged == 0 && @warn "ad's linsolve not converge"
            @views for j in 1:Nj
                jr = j + 1 - Nj * (j==Nj)
                dAMmap!(dALu, dALd, dM, ALu, ALd, M, FL[:,:,:,i,j], ξL[:,:,:,jr], i, ir, j)
            end
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

function ξRmap(ARu, ARd, M, ξR, i, ir)
    Nj = size(M, 6)
    ξRm = copy(ξR)
    @inbounds @views for j in 1:Nj
        jr = j - 1 + Nj * (j==1)
        ξRm[:,:,:,j] .= ein"((adf,abc),dgeb),fgh -> ceh"(ξR[:,:,:,jr],ARu[:,:,:,i,j],M[:,:,:,:,i,j],ARd[:,:,:,ir,j])
    end
    return ξRm
end

function ChainRulesCore.rrule(::typeof(rightenv), ARu, ARd, M, FR; ifobs = false, kwargs...)
    λR, FR = rightenv(ARu, ARd, M, FR; ifobs = ifobs, kwargs...)
    Ni,Nj = size(M)[end-1:end]
    function back((dλ, dFR))
        dARu = zero(ARu)
        dARd = zero(ARd)
        dM   = zero(M)
        for i = 1:Ni
            ir = ifobs ? Ni+1-i : i+1-Ni*(i==Ni)
            dFR[:,:,:,i,:] -= Array(ein"abcd,abcd ->"(conj(FR[:,:,:,i,:]), dFR[:,:,:,i,:]))[] * FR[:,:,:,i,:]
            ξR, info = linsolve(X -> ξRmap(ARu, ARd, M, X, i, ir), conj(dFR[:,:,:,i,:]), -λR[i], 1; maxiter = 1)
            info.converged == 0 && @warn "ad's linsolve not converge"
            for j = 1:Nj
                jr = j - 1 + Nj * (j==1)
                dAMmap!(dARu, dARd, dM, ARu, ARd, M,  ξR[:,:,:,jr], FR[:,:,:,i,j], i, ir, j)
            end
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
function ξACmap(ξAC, FL, FR, M, j)
    Ni = size(M, 5)
    ξACm = copy(ξAC)
    @inbounds @views for i in 1:Ni
        ir = i + 1 - Ni * (i==Ni)
        ξACm[:,:,:,i] .= ein"((adf,fgh),dgeb),ceh -> abc"(FL[:,:,:,i,j],ξAC[:,:,:,ir],M[:,:,:,:,i,j],FR[:,:,:,i,j])
    end
    return ξACm
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
function dFMmap!(dFL, dM, dFR, FL, M, FR, ACu, ACd, i, j)
    dFL[:,:,:,  i,j] = -conj(ein"((abc,ceh),dgeb),fgh -> adf"(ACu, FR[:,:,:,i,j], M[:,:,:,:,i,j], ACd))
     dM[:,:,:,:,i,j] = -conj(ein"(abc,adf),(ceh,fgh) -> dgeb"(ACu, FL[:,:,:,i,j],FR[:,:,:,  i,j], ACd))
    dFR[:,:,:,  i,j] = -conj(ein"((abc,adf),dgeb),fgh -> ceh"(ACu, FL[:,:,:,i,j], M[:,:,:,:,i,j], ACd))
end

function ChainRulesCore.rrule(::typeof(ACenv), AC, FL, M, FR; kwargs...)
    λAC, AC = ACenv(AC, FL, M, FR)
    Ni, Nj = size(M)[end-1:end]
    function back((dλ, dAC))
        dFL = zero(FL)
        dM  = zero(M)
        dFR = zero(FR)
        @inbounds for j = 1:Nj
            dAC[:,:,:,:,j] -= Array(ein"abcd,abcd ->"(conj(AC[:,:,:,:,j]), dAC[:,:,:,:,j]))[] * AC[:,:,:,:,j]
            ξAC, info = linsolve(X -> ξACmap(X, FL, FR, M, j), conj(dAC[:,:,:,:,j]), -λAC[j], 1; maxiter = 1)
            info.converged == 0 && @warn "ad's linsolve not converge"
            @views for i = 1:Ni
                ir = i + 1 - Ni * (i==Ni)
                dFMmap!(dFL, dM, dFR, FL, M, FR, AC[:,:,:,i,j], ξAC[:,:,:,ir], i, j)
            end
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
function ξCmap(ξC, FL, FR, j)
    Ni,Nj = size(FL)[[4,5]]
    ξCm = copy(ξC)
    jr = j + 1 - Nj * (j==Nj)
    @inbounds @views for i in 1:Ni
        ir = i + 1 - Ni * (i==Ni)
        ξCm[:,:,i] .= ein"(acd,de),bce -> ab"(FL[:,:,:,i,jr],ξC[:,:,ir],FR[:,:,:,i,j])
    end
    return ξCm
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
function dFMmap!(dFL, dFR, FL, FR, Cu, Cd, i, j)
    Nj = size(FL, 5)
    jr = j + 1 -  Nj * (j==Nj)
    dFL[:,:,:,i,jr] = -conj(ein"(ab,bce),de -> acd"(Cu, FR[:,:,:,i,j ], Cd))
    dFR[:,:,:,i,j ] = -conj(ein"(ab,acd),de -> bce"(Cu, FL[:,:,:,i,jr], Cd))
end

function ChainRulesCore.rrule(::typeof(Cenv), C, FL, FR; kwargs...)
    λC, C = Cenv(C, FL, FR)
    Ni, Nj = size(C)[[3,4]]
    function back((dλ, dC))
        dFL = zero(FL)
        dFR = zero(FR)
        for j = 1:Nj
            dC[:,:,:,j] -= Array(ein"abc,abc ->"(conj(C[:,:,:,j]), dC[:,:,:,j]))[] * C[:,:,:,j]
            ξC, info = linsolve(X -> ξCmap(X, FL, FR, j), conj(dC[:,:,:,j]), -λC[j], 1; maxiter = 1)
            info.converged == 0 && @warn "ad's linsolve not converge"
            @views for i = 1:Ni
                ir = i + 1 - Ni * (i==Ni)
                dFMmap!(dFL, dFR, FL, FR, C[:,:,i,j], ξC[:,:,ir], i, j)
            end
        end
        return NoTangent(), NoTangent(), dFL, dFR
    end
    return (λC, C), back
end