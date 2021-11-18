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
Zygote.@nograd BgFLint
Zygote.@nograd BgFRint
Zygote.@nograd leftorth
Zygote.@nograd rightorth
Zygote.@nograd ALCtoAC
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
    T = eltype(ALu[1,1])
    atype = _arraytype(M[1,1])
    function back((dλL, dFL))
        dALu = fill!(similar(ALu, atype), atype(zeros(T,size(ALu[1,1]))))
        dALd = fill!(similar(ALd, atype), atype(zeros(T,size(ALd[1,1]))))
        dM = fill!(similar(M, atype), atype(zeros(T,size(M[1,1]))))
        for j = 1:Nj, i = 1:Ni
            ir = i + 1 - Ni * (i == Ni)
            jr = j - 1 + Nj * (j == 1)
            dFL[i,j] -= Array(ein"abc,abc ->"(conj(FL[i,j]), dFL[i,j]))[] * FL[i,j]
            ξl, info = linsolve(FR -> FRmap(ALu[i,:], conj(ALd[ir,:]), M[i,:], FR, jr), conj(dFL[i,j]), -λL[i,j], 1; maxiter = 1)
            # @assert info.converged == 1
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
    T = eltype(ARu[1,1])
    atype = _arraytype(M[1,1])
    function back((dλ, dFR))
        dARu = fill!(similar(ARu, atype), atype(zeros(T,size(ARu[1,1]))))
        dARd = fill!(similar(ARd, atype), atype(zeros(T,size(ARd[1,1]))))
        dM = fill!(similar(M, atype), atype(zeros(T,size(M[1,1]))))
        for j = 1:Nj, i = 1:Ni
            ir = i + 1 - Ni * (i == Ni)
            jr = j - 1 + Nj * (j == 1)
            dFR[i,jr] -= Array(ein"abc,abc ->"(conj(FR[i,jr]), dFR[i,jr]))[] * FR[i,jr]
            ξr, info = linsolve(FL -> FLmap(ARu[i,:], conj(ARd[ir,:]), M[i,:], FL, j), conj(dFR[i,jr]), -λR[i,jr], 1; maxiter = 1)
            # @assert info.converged == 1
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
    T = eltype(AC[1,1])
    atype = _arraytype(M[1,1])
    function back((dλ, dAC))
        dFL = fill!(similar(FL, atype), atype(zeros(T,size(FL[1,1]))))
        dM = fill!(similar(M, atype), atype(zeros(T,size(M[1,1]))))
        dFR = fill!(similar(FR, atype), atype(zeros(T,size(FR[1,1]))))
        for j = 1:Nj, i = 1:Ni
            if dAC[i,j] !== nothing
                ir = i - 1 + Ni * (i == 1)
                dAC[i,j] -= Array(ein"abc,abc ->"(conj(AC[i,j]), dAC[i,j]))[] * AC[i,j]
                ξAC, info = linsolve(ACd -> ACdmap(ACd, FL[:,j], FR[:,j], M[:,j], ir), conj(dAC[i,j]), -λAC[i,j], 1; maxiter = 1)
                # @assert info.converged == 1
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
    T = eltype(C[1,1])
    atype = _arraytype(FL[1,1])
    function back((dλ, dC))
        dFL = fill!(similar(FL, atype), atype(zeros(T,size(FL[1,1]))))
        dFR = fill!(similar(FR, atype), atype(zeros(T,size(FR[1,1]))))
        for j = 1:Nj, i = 1:Ni
            if dC[i,j] !== nothing
                ir = i - 1 + Ni * (i == 1)
                jr = j + 1 - (j==Nj) * Nj
                dC[i,j] -= Array(ein"ab,ab ->"(conj(C[i,j]), dC[i,j]))[] * C[i,j]
                ξC, info = linsolve(Cd -> Cdmap(Cd, FL[:,jr], FR[:,j], ir), conj(dC[i,j]), -λC[i,j], 1; maxiter = 1)
                # @assert info.converged == 1
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
    T = eltype(ALu[1,1])
    atype = _arraytype(M[1,1])
    function back((dλL, dFL))
        dALu = fill!(similar(ALu, atype), atype(zeros(T,size(ALu[1,1]))))
        dALd = fill!(similar(ALd, atype), atype(zeros(T,size(ALd[1,1]))))
        dM = fill!(similar(M, atype), atype(zeros(T,size(M[1,1]))))
        for j = 1:Nj, i = 1:Ni
            ir = Ni + 1 - i
            jr = j - 1 + Nj * (j == 1)
            dFL[i,j] -= Array(ein"abc,abc ->"(conj(FL[i,j]), dFL[i,j]))[] * FL[i,j]
            ξl, info = linsolve(FR -> FRmap(ALu[i,:], ALd[ir,:], M[i,:], FR, jr), conj(dFL[i,j]), -λL[i,j], 1; maxiter = 1)
            # @assert info.converged == 1
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
    T = eltype(ARu[1,1])
    atype = _arraytype(M[1,1])
    function back((dλ, dFR))
        dARu = fill!(similar(ARu, atype), atype(zeros(T,size(ARu[1,1]))))
        dARd = fill!(similar(ARd, atype), atype(zeros(T,size(ARd[1,1]))))
        dM = fill!(similar(M, atype), atype(zeros(T,size(M[1,1]))))
        for j = 1:Nj, i = 1:Ni
            ir = Ni + 1 - i
            jr = j - 1 + Nj * (j == 1)
            dFR[i,jr] -= Array(ein"abc,abc ->"(conj(FR[i,jr]), dFR[i,jr]))[] * FR[i,jr]
            ξr, info = linsolve(FL -> FLmap(ARu[i,:], ARd[ir,:], M[i,:], FL, j), conj(dFR[i,jr]), -λR[i,jr], 1; maxiter = 1)
            # @assert info.converged == 1
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

"""
    dBgAMmap(Ai, Aip, Mi, Mip, L, R, j, J)
Aip means Aᵢ₊₁
```
               ┌──  Aᵢⱼ  ── ... ── AᵢJ   ──  ...   ──┐ 
               │     │              │                │ 
               │─   Mᵢⱼ  ── ...  ──   ────── ...   ──│ 
dMᵢJ    =     Lᵢⱼ    │              │                Rᵢⱼ 
               │─   Mᵢ₊₁ⱼ ──... ──Mᵢ₊₁J ───  ...   ──│ 
               │     │              │                │
               └──  Aᵢₚⱼ  ─  ...── AᵢₚJ  ──── ...   ──┘ 
               ┌──  Aᵢⱼ  ── ... ── AᵢJ   ──  ...   ──┐ 
               │     │              │                │ 
               │─   Mᵢⱼ  ── ...  ──MᵢJ   ─── ...   ──│ 
dMᵢₚJ    =    Lᵢⱼ    │              │               Rᵢⱼ 
               │─   Mᵢ₊₁ⱼ ──... ──     ───   ...   ──│ 
               │     │              │                │
               └──  Aᵢₚⱼ  ─  ...── AᵢₚJ  ──── ...   ──┘ 
               ┌──  Aᵢⱼ  ── ... ──       ──  ...   ──┐ 
               │     │              │                │ 
               │─   Mᵢⱼ  ── ... ── MᵢJ ────  ...   ──│ 
dAᵢJ    =     Lᵢⱼ    │              │                Rᵢⱼ 
               │─   Mᵢ₊₁ⱼ ──... ── Mᵢ₊₁J ─── ...   ──│ 
               │     │              │                │
               └──  Aᵢₚⱼ  ─  ...── AᵢₚJ  ──── ...   ──┘ 
               ┌──  Aᵢⱼ  ── ... ── AᵢJ   ──  ...   ──┐      a ────┬──── c 
               │     │              │                │      │     b     │  
               │─   Mᵢⱼ  ── ... ── MᵢJ ────  ...   ──│      ├─ d ─┼─ e ─┤ 
dAᵢₚJ    =     Lᵢⱼ    │              │               Rᵢⱼ    │     f     │   
               │─   Mᵢ₊₁ⱼ ──... ── Mᵢ₊₁J ─── ...   ──│      ├─ g ─┼─ h ─┤ 
               │     │              │                │      │     j     │ 
               └──  Aᵢₚⱼ  ─ ... ──      ──── ...   ──┘      i ────┴──── k  
```
"""
function dBgAMmap(Ai, Aip, Mi, Mip, L, R, j, J)
    Nj = size(Ai, 1)
        NL = (J - j + (J - j < 0) * Nj)
    NR = Nj - NL - 1
    L = copy(L)
    R = copy(R)
    for jj = 1:NL
        jr = j + jj - 1 - (j + jj - 1 > Nj) * Nj
        L = ein"(((adgi,abc),dfeb),gjhf),ijk -> cehk"(L, Ai[jr], Mi[jr], Mip[jr], Aip[jr])
    end
    for jj = 1:NR
        jr = j - jj + (j - jj < 1) * Nj
        R = ein"(((cehk,abc),dfeb),gjhf),ijk -> adgi"(R, Ai[jr], Mi[jr], Mip[jr], Aip[jr])
    end
    dAiJ = -ein"(((adgi,ijk),gjhf),dfeb), cehk -> abc"(L, Aip[J], Mip[J], Mi[J], R)
    dAipJ = -ein"(((adgi,abc),dfeb),gjhf), cehk -> ijk"(L, Ai[J], Mi[J], Mip[J], R)
    dMiJ = -ein"(adgi,abc),(gjhf,(ijk,cehk)) -> dfeb"(L, Ai[J], Mip[J], Aip[J], R)
    dMipJ = -ein"((adgi,abc),dfeb),(ijk,cehk)-> gjhf"(L, Ai[J], Mi[J], Aip[J], R)
    return conj(dAiJ), conj(dAipJ), conj(dMiJ), conj(dMipJ)
end

function ChainRulesCore.rrule(::typeof(bigleftenv), ALu, ALd, M, BgFL; kwargs...)
    λL, BgFL = bigleftenv(ALu, ALd, M, BgFL)
    Ni, Nj = size(ALu)
    T = eltype(ALu[1,1])
    atype = _arraytype(M[1,1])
    function back((dλL, dBgFL))
        dALu = fill!(similar(ALu, atype), atype(zeros(T,size(ALu[1,1]))))
        dM = fill!(similar(M, atype), atype(zeros(T,size(M[1,1]))))
        dALd = fill!(similar(ALd, atype), atype(zeros(T,size(ALd[1,1]))))
        for j = 1:Nj, i = 1:Ni
            if dBgFL[i,j] !== nothing
                ir = i + 1 - Ni * (i==Ni)
                irr = i + 2 - Ni * (i + 2 > Ni)
                jr = j - 1 + Nj * (j == 1)
                dBgFL[i,j] -= Array(ein"abcd,abcd ->"(conj(BgFL[i,j]), dBgFL[i,j]))[] * BgFL[i,j]
                ξl, info = linsolve(BgFR -> BgFRmap(ALu[i,:], ALd[irr,:], M[i,:], M[ir,:], BgFR, jr), conj(dBgFL[i,j]), -λL[i,j], 1; maxiter = 1)
                # @assert info.converged == 1
                # errL = ein"abc,cba ->"(FL[i,j], ξl)[]
                # abs(errL) > 1e-1 && throw("FL and ξl aren't orthometric. $(errL) $(info)")
                # @show info ein"abc,cba ->"(FL[i,j], ξl)[] ein"abc,abc -> "(FL[i,j], dFL[i,j])[]
                for J = 1:Nj
                    dAiJ, dAipJ, dMiJ, dMipJ = dBgAMmap(ALu[i,:], ALd[irr,:], M[i,:], M[ir,:], BgFL[i,j], ξl, j, J)
                    dALu[i,J] += dAiJ
                    dALd[i,J] += dAipJ
                    dM[i,J] += dMiJ
                    dM[ir,J] += dMipJ
                end
            end
        end
        return NoTangent(), dALu, dALd, dM, NoTangent()
    end
    return (λL, BgFL), back
end

function ChainRulesCore.rrule(::typeof(bigrightenv), ARu, ARd, M, BgFR; kwargs...)
    λR, BgFR = bigrightenv(ARu, ARd, M, BgFR)
    Ni, Nj = size(ARu)
    T = eltype(ARu[1,1])
    atype = _arraytype(M[1,1])
    function back((dλ, dBgFR))
        dARu = fill!(similar(ARu, atype), atype(zeros(T,size(ARu[1,1]))))
        dM = fill!(similar(M, atype), atype(zeros(T,size(M[1,1]))))
        dARd = fill!(similar(ARd, atype), atype(zeros(T,size(ARd[1,1]))))
        for j = 1:Nj, i = 1:Ni
            ir = i + 1 - Ni * (i==Ni)
            irr = i + 2 - Ni * (i + 2 > Ni)
            jr = j - 1 + Nj * (j == 1)
            if dBgFR[i,jr] !== nothing
                dBgFR[i,jr] -= Array(ein"abcd,abcd ->"(conj(BgFR[i,jr]), dBgFR[i,jr]))[] * BgFR[i,jr]
                ξr, info = linsolve(BgFL -> BgFLmap(ARu[i,:], ARd[irr,:], M[i,:], M[ir,:], BgFL, j), conj(dBgFR[i,jr]), -λR[i,jr], 1; maxiter = 1)
                # @assert info.converged == 1
                # errR = ein"abc,cba ->"(ξr, FR[i,jr])[]
                # abs(errR) > 1e-1 && throw("FR and ξr aren't orthometric. $(errR) $(info)")
                # @show info ein"abc,cba ->"(ξr, FR[i,jr])[] ein"abc,abc -> "(FR[i,jr], dFR[i,jr])[]
                for J = 1:Nj
                    dAiJ, dAipJ, dMiJ, dMipJ = dBgAMmap(ARu[i,:], ARd[irr,:], M[i,:], M[ir,:], ξr, BgFR[i,jr], j, J)
                    dARu[i,J] += dAiJ
                    dARd[i,J] += dAipJ
                    dM[i,J] += dMiJ
                    dM[ir,J] += dMipJ
                end
            end
        end
        return NoTangent(), dARu, dARd, dM, NoTangent()
    end
    return (λR, BgFR), back
end

