using Base.Threads
using LinearAlgebra
using KrylovKit
using OMEinsum
using Random
using Zygote

"""
tensor order graph: from left to right, top to bottom.
```
a ────┬──── c    a──────┬──────c     a─────b
│     b     │    │      │      │     │     │
├─ d ─┼─ e ─┤    │      b      │     ├──c──┤           
│     g     │    │      │      │     │     │
f ────┴──── h    d──────┴──────e     d─────e
```
"""

"""
    i, j = ktoij(k,Ni,Nj)
    LinearIndices -> CartesianIndices
"""
ktoij(k,Ni,Nj) = CartesianIndices((1:Ni,1:Nj))[k].I

safesign(x::Number) = iszero(x) ? one(x) : sign(x)

"""
    qrpos(A)

Returns a QR decomposition, i.e. an isometric `Q` and upper triangular `R` matrix, where `R`
is guaranteed to have positive diagonal elements.
"""
qrpos(A) = qrpos!(copy(A))
function qrpos!(A)
    mattype = _mattype(A)
    F = qr!(mattype(A))
    Q = mattype(F.Q)
    R = F.R
    phases = safesign.(diag(R))
    rmul!(Q, Diagonal(phases))
    lmul!(Diagonal(conj!(phases)), R)
    return Q, R
end

"""
    lqpos(A)

Returns a LQ decomposition, i.e. a lower triangular `L` and isometric `Q` matrix, where `L`
is guaranteed to have positive diagonal elements.
"""
lqpos(A) = lqpos!(copy(A))
function lqpos!(A)
    mattype = _mattype(A)
    F = qr!(mattype(A'))
    Q = mattype(mattype(F.Q)')
    L = mattype(F.R')
    phases = safesign.(diag(L))
    lmul!(Diagonal(phases), Q)
    rmul!(L, Diagonal(conj!(phases)))
    return L, Q
end

function ρmap(ρ,Ai,J)
    Nj = size(Ai,1)
    for j = 1:Nj
        jr = J+j-1 - (J+j-1 > Nj)*Nj
        ρ = ein"(dc,csb),dsa -> ab"(ρ,Ai[jr],conj(Ai[jr]))
    end
    return ρ
end

function initialA(M, D)
    Ni, Nj = size(M)
    atype = _arraytype(M[1,1])
    A = Array{atype{ComplexF64, 3}, 2}(undef, Ni, Nj)
    # dir = nothing
    # typeof(M[1]) <: U1Array && (getdir(M[1])[4] == -1 ? (dir = [-1, 1, 1]) : (dir = [1, -1, -1]))
    # typeof(M[1]) <: U1Array && (direction == "up" ? (dir = [-1, -1, 1, 1]) : (dir = [1, 1, -1, -1]))
    @threads for k = 1:Ni*Nj
        i, j = ktoij(k, Ni, Nj)
        d = size(M[i,j], 4)
        # A[i,j] = randinitial(M[1,1], D,d,D; dir = [-1, 1, 1]) # ordinary random initial
        A[i,j] = symmetryreshape(randinitial(M[1,1], D,Int(sqrt(d)),Int(sqrt(d)),D; dir = [-1, -1, 1, 1]), D,d,D)[1] # for double-layer ipeps
    end
    return A
end

function cellones(A)
    Ni, Nj = size(A)
    D = size(A[1,1],1)
    atype = _arraytype(A[1,1])
    Cell = Array{atype, 2}(undef, Ni, Nj)
    dir = nothing
    atype <: U1Array && (dir = getdir(A[1,1])[[1,3]])
    @threads for k = 1:Ni*Nj
        i, j = ktoij(k, Ni, Nj)
        Cell[i,j] = Iinitial(A[1,1], D; dir = dir)
    end
    return Cell
end

function sysvd!(ρ::AbstractArray)
    F = svd!(ρ)
    F.U, F.S, F.Vt
end

"""
    getL!(A,L; kwargs...)

````
┌ A1─A2─    ┌      L ─
ρ │  │    = ρ   =  │
┕ A1─A2─    ┕      L'─
````

ρ=L'*L, return L, where `L`is guaranteed to have positive diagonal elements.
L = cholesky!(ρ).U
If ρ is not exactly positive definite, cholesky will fail
"""
function getL!(A,L; kwargs...) 
    Ni,Nj = size(A)
    @threads for k = 1:Ni*Nj
        i, j = ktoij(k, Ni, Nj)
        λ,ρs,info = eigsolve(ρ->ρmap(ρ,A[i,:],j), L[i,j]'*L[i,j], 1, :LM; ishermitian = false, maxiter = 1, kwargs...)
        @debug "getL eigsolve" λ info sort(abs.(λ))
        info.converged == 0 && @warn "getL not converged"
        ρ = ρs[1] + ρs[1]'
        ρ /= tr(ρ)
        _, S, Vt = sysvd!(ρ)
        Lo = lmul!(Diagonal(sqrt.(S)), Vt)
        _, L[i,j] = qrpos!(Lo)
    end
    return L
end

"""
    getAL(A,L)

Given an MPS tensor `A` and `L` ，return a left-canonical MPS tensor `AL`, a gauge transform `R` and
a scalar factor `λ` such that ``λ AR R = L A``
"""
function getAL(A,L)
    Ni,Nj = size(A)
    atype = _arraytype(A[1,1])
    AL = Array{atype{ComplexF64, 3}, 2}(undef, Ni, Nj)
    Le = Array{atype{ComplexF64, 2}, 2}(undef, Ni, Nj)
    λ = zeros(Ni,Nj)
    @threads for k = 1:Ni*Nj
        i, j = ktoij(k, Ni, Nj)
        D, d, = size(A[i,j])
        Q, R = qrpos!(reshape(ein"ab,bcd -> acd"(L[i,j], A[i,j]), D*d, D))
        AL[i,j] = reshape(Q, D, d, D)
        λ[i,j] = norm(R)
        Le[i,j] = rmul!(R, 1/λ[i,j])
    end
    return AL, Le, λ
end

function getLsped(Le, A, AL; kwargs...)
    Ni,Nj = size(A)
    L = Array{_arraytype(A[1,1]){ComplexF64, 2}, 2}(undef, Ni, Nj)
    @threads for k = 1:Ni*Nj
        i, j = ktoij(k, Ni, Nj)
        λ , Ls, info = eigsolve(X -> ein"(dc,csb),dsa -> ab"(X,A[i,j],conj(AL[i,j])), Le[i,j], 1, :LM; ishermitian = false, kwargs...)
        @debug "getLsped eigsolve" λ info sort(abs.(λ))
        info.converged == 0 && @warn "getLsped not converged"
        _, L[i,j] = qrpos!(Ls[1])
    end
    return L
end

"""
    leftorth(A,L=cellones(size(A,1),size(A,2),size(A[1,1],1)); tol = 1e-12, maxiter = 100, kwargs...)

Given an MPS tensor `A`, return a left-canonical MPS tensor `AL`, a gauge transform `L` and
a scalar factor `λ` such that ``λ AL L = L A``, where an initial guess for `L` can be
provided.
"""
function leftorth(A,L=cellones(A); tol = 1e-12, maxiter = 100, kwargs...)
    L = getL!(A, L; kwargs...)
    AL, Le, λ = getAL(A, L;kwargs...)
    numiter = 1
    while norm(L.-Le) > tol && numiter < maxiter
        L = getLsped(Le, A, AL; kwargs...)
        AL, Le, λ = getAL(A, L; kwargs...)
        numiter += 1
    end
    L = Le
    return AL, L, λ
end


"""
    rightorth(A,L=cellones(size(A,1),size(A,2),size(A[1,1],1)); tol = 1e-12, maxiter = 100, kwargs...)

Given an MPS tensor `A`, return a gauge transform R, a right-canonical MPS tensor `AR`, and
a scalar factor `λ` such that ``λ R AR^s = A^s R``, where an initial guess for `R` can be
provided.
"""
function rightorth(A,L=cellones(A); tol = 1e-12, maxiter = 100, kwargs...)
    Ar, Lr = map(x->permutedims(x,(3,2,1)), A), map(x->permutedims(x,(2,1)), L)
    AL, L, λ = leftorth(Ar,Lr; tol = tol, maxiter = maxiter, kwargs...)
    R, AR = map(x->permutedims(x,(2,1)), L), map(x->permutedims(x,(3,2,1)), AL)
    AR = typeof(AL)(AR)
    return R, AR, λ
end

"""
    LRtoC(L,R)

```
 ── Cᵢⱼ ──  =  ── Lᵢⱼ ── Rᵢⱼ₊₁ ──
```
"""
function LRtoC(L, R)
    Ni, Nj = size(L)
    # atype = _arraytype(L[1,1])
    # C = Array{atype{ComplexF64, 2}, 2}(undef, Ni, Nj)
    # for j in 1:Nj,i in 1:Ni
    #     jr = j + 1 - (j + 1 > Nj) * Nj
    #     C[i,j] = L[i,j] * R[i,jr]
    # end
    # return C
    [L[i,j] * R[i, j+1 - (j+1 > Nj) * Nj] for i=1:Ni, j=1:Nj]
end

"""
    FLm = FLmap(ALi, ALip, Mi, FL, J)

ALip means ALᵢ₊₁
```
  ┌──        ┌──  ALᵢⱼ  ── ALᵢⱼ₊₁   ──   ...          a ────┬──── c 
  │          │     │        │                         │     b     │ 
 FLm   =   FLᵢⱼ ─ Mᵢⱼ   ── Mᵢⱼ₊₁    ──   ...          ├─ d ─┼─ e ─┤ 
  │          │     │        │                         │     g     │ 
  ┕──        ┕──  ALᵢ₊₁ⱼ ─ ALᵢ₊₁ⱼ₊₁ ──   ...          f ────┴──── h 
```
"""
function FLmap(ALi, ALip, Mi, FL, J)
    Nj = size(ALi,1)
    FLm = copy(FL)
    for j=1:Nj
        jr = J+j-1 - (J+j-1 > Nj)*Nj
        FLm = ein"((adf,abc),dgeb),fgh -> ceh"(FLm,ALi[jr],Mi[jr],ALip[jr])
    end
    return FLm
end

"""
    FRm = FRmap(ARi, ARip, Mi, FR, J)

ARip means ARᵢ₊₁
```
 ──┐       ... ─── ARᵢⱼ₋₁  ── ARᵢⱼ  ──┐            a ────┬──── c 
   │                │          │      │            │     b     │ 
──FRm  =   ... ──── Mᵢⱼ₋₁  ── Mᵢⱼ  ──FRᵢⱼ          ├─ d ─┼─ e ─┤ 
   │                │          │      │            │     g     │ 
 ──┘       ... ─ ARᵢ₊₁ⱼ₋₁ ─ ARᵢ₊₁ⱼ  ──┘            f ────┴──── h 
```
"""
function FRmap(ARi, ARip, Mi, FR, J)
    Nj = size(ARi,1)
    FRm = copy(FR)
    for j=1:Nj
        jr = J-(j-1) + (J-(j-1) < 1)*Nj
        FRm = ein"((ceh,abc),dgeb),fgh -> adf"(FRm,ARi[jr],Mi[jr],ARip[jr])
    end
    return FRm
end

function FLint(AL, M)
    Ni,Nj = size(AL)
    D, d = size(AL[1],1), size(M[1],1)
    # typeof(AL[1]) <: U1Array && (dir = [-sign(sum(AL[1].qn)[1]), -sign(sum(M[1].qn)[1]), sign(sum(AL[1].qn)[1])])
    dir = nothing
    typeof(AL[1]) <: U1Array && (dir = [1, getdir(M[1])[1], -getdir(M[1])[1], -1])
    # [randinitial(AL[i,j], D,d,D; dir = [1,1,-1]) for i=1:Ni, j=1:Nj]
    [symmetryreshape(randinitial(AL[i,j], D,Int(sqrt(d)),Int(sqrt(d)),D; dir = dir, q = [0]), D,d,D)[1] for i=1:Ni, j=1:Nj]
end

function FRint(AR, M)
    Ni,Nj = size(AR)
    D, d = size(AR[1],3), size(M[1],3)
    dir = nothing
    typeof(AR[1]) <: U1Array && (dir = [-1, getdir(M[1])[3], -getdir(M[1])[3], 1])
    # [randinitial(AR[i,j], D,d,D; dir = [-1,-1,1]) for i=1:Ni, j=1:Nj]
    [symmetryreshape(randinitial(AR[i,j], D,Int(sqrt(d)),Int(sqrt(d)),D; dir = dir, q = [0]), D,d,D)[1] for i=1:Ni, j=1:Nj]
end

"""
    λL, FL = leftenv(ALu, ALd, M, FL = FLint(ALu,M); kwargs...)

Compute the left environment tensor for MPS A and MPO M, by finding the left fixed point
of ALu - M - ALd contracted along the physical dimension.
```
 ┌──  ALuᵢⱼ  ── ALuᵢⱼ₊₁   ──   ...         ┌── 
 │     │        │                          │   
FLᵢⱼ ─ Mᵢⱼ   ── Mᵢⱼ₊₁     ──   ...  = λLᵢⱼ FLᵢⱼ 
 │     │        │                          │   
 ┕──  ALdᵢᵣⱼ  ─ ALdᵢᵣⱼ₊₁  ──   ...         ┕── 
```
"""
leftenv(ALu, ALd, M, FL = FLint(ALu,M); kwargs...) = leftenv!(ALu, ALd, M, copy(FL); kwargs...) 
function leftenv!(ALu, ALd, M, FL; kwargs...) 
    Ni,Nj = size(ALu)
    λL = zeros(eltype(FL[1,1]),Ni,Nj)
    @threads for k = 1:Ni*Nj
        i, j = ktoij(k, Ni, Nj)
        ir = i + 1 - Ni * (i==Ni)
        λLs, FL1s, info= eigsolve(X->FLmap(ALu[i,:], conj(ALd[ir,:]), M[i,:], X, j), FL[i,j], 1, :LM; maxiter=1000 , ishermitian = false, kwargs...)
        @debug "leftenv! eigsolve" λLs info sort(abs.(λLs))
        info.converged == 0 && @warn "leftenv not converged"
        if length(λLs) > 1 && norm(abs(λLs[1]) - abs(λLs[2])) < 1e-12
            @warn "leftenv may have multiple eigenvalues"
            @show λLs
            if real(λLs[1]) > 0
                FL[i,j] = FL1s[1]
                λL[i,j] = λLs[1]
            else
                FL[i,j] = FL1s[2]
                λL[i,j] = λLs[2]
            end
        else
            FL[i,j] = FL1s[1]
            λL[i,j] = λLs[1]
        end
    end
    return λL, FL
end

"""
    λR, FR = rightenv(ARu, ARd, M, FR = FRint(ARu,M); kwargs...)

Compute the right environment tensor for MPS A and MPO M, by finding the left fixed point
of AR - M - conj(AR) contracted along the physical dimension.
```
   ... ─── ARuᵢⱼ₋₁ ── ARuᵢⱼ  ──┐          ──┐   
            │          │       │            │  
   ... ──── Mᵢⱼ₋₁  ── Mᵢⱼ   ──FRᵢⱼ  = λRᵢⱼ──FRᵢⱼ
            │          │       │            │  
   ... ─   ARdᵢᵣⱼ₋₁ ─ ARdᵢᵣⱼ ──┘          ──┘  
```
"""
rightenv(ARu, ARd, M, FR = FRint(ARu,M); kwargs...) = rightenv!(ARu, ARd, M, copy(FR); kwargs...) 
function rightenv!(ARu, ARd, M, FR; kwargs...) 
    Ni,Nj = size(ARu)
    λR = zeros(eltype(FR[1,1]),Ni,Nj)
    @threads for k = 1:Ni*Nj
        i, j = ktoij(k, Ni, Nj)
        ir = i + 1 - Ni * (i==Ni)
        λRs, FR1s, info= eigsolve(X->FRmap(ARu[i,:], conj(ARd[ir,:]), M[i,:], X, j), FR[i,j], 1, :LM;maxiter=1000 , ishermitian = false, kwargs...)
        @debug "rightenv! eigsolve" λRs info sort(abs.(λRs))
        info.converged == 0 && @warn "rightenv not converged"
        if length(λRs) > 1 && norm(abs(λRs[1]) - abs(λRs[2])) < 1e-12
            @warn "rightenv may have multiple eigenvalues"
            @show λRs
            if real(λRs[1]) > 0
                FR[i,j] = FR1s[1]
                λR[i,j] = λRs[1]
            else
                FR[i,j] = FR1s[2]
                λR[i,j] = λRs[2]
            end
        else
            FR[i,j] = FR1s[1]
            λR[i,j] = λRs[1]
        end
    end
    return λR, FR
end

"""
    ACm = ACmap(ACij, FLj, FRj, Mj, II)

```
                                ┌─────── ACᵢⱼ ─────┐
                                │        │         │             a ────┬──── c   
┌─────── ACm  ─────┐      =     FLᵢⱼ ─── Mᵢⱼ ───── FRᵢⱼ          │     b     │ 
│        │         │            │        │         │             ├─ d ─┼─ e ─┤ 
                                FLᵢ₊₁ⱼ ─ Mᵢ₊₁ⱼ ──  FRᵢ₊₁ⱼ        │     g     │ 
                                │        │         │             f ────┴──── h 
                                .        .         .
                                .        .         .
                                .        .         .
```
"""
function ACmap(ACij, FLj, FRj, Mj, II)
    Ni = size(FLj,1)
    ACm = copy(ACij)
    for i=1:Ni
        ir = II+i-1 - (II+i-1 > Ni)*Ni
        ACm = ein"((adf,abc),dgeb),ceh -> fgh"(FLj[ir],ACm,Mj[ir],FRj[ir])
    end
    return ACm
end

"""
    Cmap(Cij, FLjp, FRj, II)

```
                    ┌────Cᵢⱼ ───┐         
                    │           │            a ─── b 
┌──── Cm ───┐   =   FLᵢⱼ₊₁ ──── FRᵢⱼ         │     │
│           │       │           │            ├─ c ─┤
                    FLᵢ₊₁ⱼ₊₁ ── FRᵢ₊₁ⱼ       │     │
                    │           │            d ─── e
                    .           .     
                    .           .     
                    .           .     
```
"""
function Cmap(Cij, FLjp, FRj, II)
    Ni = size(FLjp,1)
    Cm = copy(Cij)
    for i=1:Ni
        ir = II+i-1 - (II+i-1 > Ni)*Ni
        Cm = ein"(acd,ab),bce -> de"(FLjp[ir],Cm,FRj[ir])
    end
    return Cm
end

"""
    ACenv(AC, FL, M, FR;kwargs...)

Compute the up environment tensor for MPS `FL`,`FR` and MPO `M`, by finding the up fixed point
        of `FL - M - FR` contracted along the physical dimension.
```
┌─────── ACᵢⱼ ─────┐
│        │         │          
FLᵢⱼ ─── Mᵢⱼ ───── FRᵢⱼ
│        │         │   
FLᵢ₊₁ⱼ ─ Mᵢ₊₁ⱼ ──  FRᵢ₊₁ⱼ  =  λACᵢⱼ ┌──── ACᵢⱼ ───┐
│        │         │                │      │      │  
.        .         .
.        .         .
.        .         .
```
"""
ACenv(AC, FL, M, FR; kwargs...) = ACenv!(copy(AC), FL, M, FR; kwargs...)
function ACenv!(AC, FL, M, FR; kwargs...)
    Ni,Nj = size(AC)
    λAC = zeros(eltype(AC[1,1]),Ni,Nj)
    @threads for k = 1:Ni*Nj
        i, j = ktoij(k, Ni, Nj)
        λACs, ACs, info = eigsolve(X->ACmap(X, FL[:,j], FR[:,j], M[:,j], i), AC[i,j], 1, :LM; maxiter=1000 ,ishermitian = false, kwargs...)
        @debug "ACenv! eigsolve" λACs info sort(abs.(λACs))
        info.converged == 0 && @warn "ACenv Not converged"
        if length(λACs) > 1 && norm(abs(λACs[1]) - abs(λACs[2])) < 1e-12
            @warn "ACenv may have multiple eigenvalues"
            @show λACs
            if real(λACs[1]) > 0
                AC[i,j] = ACs[1]
                λAC[i,j] = λACs[1]
            else
                AC[i,j] = ACs[2]
                λAC[i,j] = λACs[2]
            end
        else
            AC[i,j] = ACs[1]
            λAC[i,j] = λACs[1]
        end
    end
    return λAC, AC
end

"""
    Cenv(C, FL, FR;kwargs...)

Compute the up environment tensor for MPS `FL` and `FR`, by finding the up fixed point
    of `FL - FR` contracted along the physical dimension.
```
┌────Cᵢⱼ ───┐
│           │          
FLᵢⱼ₊₁ ──── FRᵢⱼ
│           │   
FLᵢ₊₁ⱼ₊₁ ── FRᵢ₊₁ⱼ   =  λCᵢⱼ ┌──Cᵢⱼ ─┐
│           │                │       │  
.           .     
.           .     
.           .     
```
"""
Cenv(C, FL, FR; kwargs...) = Cenv!(copy(C), FL, FR; kwargs...)
function Cenv!(C, FL, FR; kwargs...)
    Ni,Nj = size(C)
    λC = zeros(eltype(C[1,1]),Ni,Nj)
    @threads for k = 1:Ni*Nj
        i, j = ktoij(k, Ni, Nj)
        jr = j + 1 - (j==Nj) * Nj
        λCs, Cs, info = eigsolve(X->Cmap(X, FL[:,jr], FR[:,j], i), C[i,j], 1, :LM; maxiter=1000 ,ishermitian = false, kwargs...)
        @debug "Cenv! eigsolve" λCs info sort(abs.(λCs))
        info.converged == 0 && @warn "Cenv Not converged"
        if length(λCs) > 1 && norm(abs(λCs[1]) - abs(λCs[2])) < 1e-12
            @warn "Cenv may have multiple eigenvalues"
            @show λCs
            if real(λCs[1]) > 0
                C[i,j] = Cs[1]
                λC[i,j] = λCs[1]
            else
                C[i,j] = Cs[2]
                λC[i,j] = λCs[2]
            end
        else
            C[i,j] = Cs[1]
            λC[i,j] = λCs[1]
        end
    end
    return λC, C
end

function ACCtoAL(ACij,Cij)
    D,d, = size(ACij)
    QAC, RAC = qrpos(reshape(ACij, D*d, D))
    QC, RC = qrpos(Cij)
    errL = norm(RAC-RC)
    # @show errL
    reshape(QAC*QC', D, d, D), errL
end

function ACCtoAR(ACij,Cijr)
    D,d, = size(ACij)
    LAC, QAC = lqpos(reshape(ACij, D, d*D))
    LC, QC = lqpos(Cijr)
    errR = norm(LAC-LC)
    # @show errR
    reshape(QC'*QAC, D, d, D), errR
end

"""
    itoir(i,Ni,Nj)

````
i -> (i,j) -> (i,jr) -> ir
````
"""
function itoir(i,Ni,Nj)
    Liner = LinearIndices((1:Ni,1:Nj))
    Cart = CartesianIndices((1:Ni,1:Nj))
    Index = Cart[i]
    i,j = Index[1],Index[2]
    jr = j - 1 + (j==1)*Nj
    Liner[i,jr]
end

function ALCtoAC(AL,C)
    Ni,Nj = size(AL)
    reshape([ein"asc,cb -> asb"(AL[i],C[i]) for i=1:Ni*Nj], Ni,Nj)
end

"""
    AL, AR = ACCtoALAR(AC, C)

QR factorization to get `AL` and `AR` from `AC` and `C`

````
──ALᵢⱼ──Cᵢⱼ──  =  ──ACᵢⱼ──  = ──Cᵢ₋₁ⱼ ──ARᵢⱼ──
  │                  │                  │   
````
"""
function ACCtoALAR(AC, C)
    Ni,Nj = size(AC)
    ALijerrL = [ACCtoAL(AC[i],C[i]) for i=1:Ni*Nj]
    AL = reshape([ALijerrL[i][1] for i=1:Ni*Nj],Ni,Nj)
    errL = Zygote.@ignore sum([ALijerrL[i][2] for i=1:Ni*Nj])
    ARijerrR = [ACCtoAR(AC[i],C[itoir(i,Ni,Nj)]) for i=1:Ni*Nj]
    AR = reshape([ARijerrR[i][1] for i=1:Ni*Nj],Ni,Nj)
    errR = Zygote.@ignore sum([ARijerrR[i][2] for i=1:Ni*Nj])
    return AL, AR, errL, errR
end

"""
Compute the error through all environment `AL,C,AR,FL,M,FR`

````
        ┌── AC──┐         
        │   │   │           ┌── AC──┐ 
MAC1 =  FL─ M ──FR  =  λAC  │   │   │ 
        │   │   │         

        ┌── AC──┐         
        │   │   │           ┌──C──┐ 
MAC2 =  FL─ M ──FR  =  λAC  │     │ 
        │   │   │         
         ── AR──┘   
        
── MAC1 ──    ≈    ── MAC2 ── AR ── 
    │                         │
````
"""
function error(AL,C,AR,FL,M,FR)
    Ni,Nj = size(AL)
    AC = ALCtoAC(AL, C)
    err = [0.0 for _ = 1:nthreads()]
    @threads for k = 1:Ni*Nj
        i, j = ktoij(k, Ni, Nj)
        MAC = ACmap(AC[i,j], FL[:,j], FR[:,j], M[:,j], i)
        MAC -= ein"(apc,dpc),dsb -> asb"(MAC,conj(AR[i,j]),AR[i,j])
        err[threadid()] += norm(MAC)
    end
    return reduce(+, err)
end

"""
    λL, FL = obs_FL(ALu, ALd, M, FL = FLint(AL,M); kwargs...)

Compute the observable left environment tensor for MPS A and MPO M, by finding the left fixed point
of AL - M - conj(AL) contracted along the physical dimension.
```
 ┌──  ALuᵢⱼ  ── ALuᵢⱼ₊₁   ──   ...         ┌── 
 │     │        │                          │   
FLᵢⱼ ─ Mᵢⱼ   ── Mᵢⱼ₊₁     ──   ...  = λLᵢⱼ FLᵢⱼ 
 │     │        │                          │   
 ┕──  ALdᵢᵣⱼ  ─ ALdᵢᵣⱼ₊₁  ──   ...         ┕── 
```
"""
obs_FL(ALu, ALd, M, FL = FLint(ALu,M); kwargs...) = obs_FL!(ALu, ALd, M, copy(FL); kwargs...) 
function obs_FL!(ALu, ALd, M, FL; kwargs...) 
    Ni,Nj = size(ALu)
    λL = zeros(eltype(FL[1,1]),Ni,Nj)
    @threads for k = 1:Ni*Nj
        i, j = ktoij(k, Ni, Nj)
        ir = Ni + 1 - i
        λLs, FL1s, info= eigsolve(X->FLmap(ALu[i,:], ALd[ir,:], M[i,:], X, j), FL[i,j], 1, :LM; ishermitian = false, kwargs...)
        @debug "obs_FL eigsolve" λLs info sort(abs.(λLs))
        info.converged == 0 && @warn "obs_FL Not converged"
        if length(λLs) > 1 && norm(abs(λLs[1]) - abs(λLs[2])) < 1e-12
            @warn "obs_FL may have multiple eigenvalues"
            @show λLs
            if real(λLs[1]) > 0
                FL[i,j] = FL1s[1]
                λL[i,j] = λLs[1]
            else
                FL[i,j] = FL1s[2]
                λL[i,j] = λLs[2]
            end
        else
            FL[i,j] = FL1s[1]
            λL[i,j] = λLs[1]
        end
    end
    return λL, FL
end

"""
    λR, FR = obs_FR(ARu, ARd, M, FR = FRint(ARu,M); kwargs...)

Compute the observable right environment tensor for MPS A and MPO M, by finding the left fixed point
of AR - M - conj(AR) contracted along the physical dimension.
```
   ... ─── ARuᵢⱼ₋₁ ── ARuᵢⱼ  ──┐          ──┐   
            │          │       │            │  
   ... ──── Mᵢⱼ₋₁  ── Mᵢⱼ   ──FRᵢⱼ  = λRᵢⱼ──FRᵢⱼ
            │          │       │            │  
   ... ─   ARdᵢᵣⱼ₋₁ ─ ARdᵢᵣⱼ ──┘          ──┘  
```
"""
obs_FR(ARu, ARd, M, FR = FRint(ARu,M); kwargs...) = obs_FR!(ARu, ARd, M, copy(FR); kwargs...) 
function obs_FR!(ARu, ARd, M, FR; kwargs...) 
    Ni,Nj = size(ARu)
    λR = zeros(eltype(FR[1,1]),Ni,Nj)
    @threads for k = 1:Ni*Nj
        i, j = ktoij(k, Ni, Nj)
        ir = Ni + 1 - i
        λRs, FR1s, info= eigsolve(X->FRmap(ARu[i,:], ARd[ir,:], M[i,:], X, j), FR[i,j], 1, :LM; ishermitian = false, kwargs...)
        @debug "obs_FR! eigsolve" λRs info sort(abs.(λRs))
        info.converged == 0 && @warn "obs_FR! Not converged"
        if length(λRs) > 1 && norm(abs(λRs[1]) - abs(λRs[2])) < 1e-12
            @warn "obs_FR may have multiple eigenvalues"
            @show λRs
            if real(λRs[1]) > 0
                FR[i,j] = FR1s[1]
                λR[i,j] = λRs[1]
            else
                FR[i,j] = FR1s[2]
                λR[i,j] = λRs[2]
            end
        else
            FR[i,j] = FR1s[1]
            λR[i,j] = λRs[1]
        end
    end
    return λR, FR
end

function norm_FLint(AL)
    Ni,Nj = size(AL)
    arraytype = _arraytype(AL[1,1])
    norm_FL = Array{arraytype,2}(undef, Ni, Nj)
    @threads for k = 1:Ni*Nj
        i, j = ktoij(k, Ni, Nj)
        D = size(AL[i,j],1)
        norm_FL[i,j] = randinitial(AL[1,1], D, D; dir = [1,1])
    end
    return norm_FL
end

"""
    FL_normm = norm_FLmap(ALui, ALdi, FL_norm, J)

```
   ┌──        ┌──  ALuᵢⱼ ── ALuᵢⱼ₊₁ ──  ...     a───┬───c   
  FLm   =    FLm     │        │                 │   b   │ 
   ┕──        ┕──  ALdᵢⱼ ── ALdᵢⱼ₊₁ ──  ...     d───┴───e  
```
"""
function norm_FLmap(ALui, ALdi, FL_norm, J)
    Nj = size(ALui,1)
    FL_normm = copy(FL_norm)
    for j=1:Nj
        jr = J+j-1 - (J+j-1 > Nj)*Nj
        FL_normm = ein"(ad,abc),dbe -> ce"(FL_normm,ALui[jr],ALdi[jr])
    end
    return FL_normm
end

norm_FL(ALu, ALd, FL_norm = norm_FLint(ALu); kwargs...) = norm_FL!(ALu, ALd, FL_norm; kwargs...)
function norm_FL!(ALu, ALd, FL_norm; kwargs...)
    Ni,Nj = size(ALu)
    λL = zeros(eltype(FL_norm[1,1]),Ni,Nj)
    @threads for k = 1:Ni*Nj
        i, j = ktoij(k, Ni, Nj)
        λLs, FL_norms, info= eigsolve(X->norm_FLmap(ALu[i,:], ALd[i,:], X, j), FL_norm[i,j], 1, :LM; ishermitian = false, kwargs...)
        @debug "norm_FL eigsolve" λLs info sort(abs.(λLs))
        if length(λLs) > 1 && norm(abs(λLs[1]) - abs(λLs[2])) < 1e-12
            @show λLs
            if real(λLs[1]) > 0
                FL_norm[i,j] = FL_norms[1]
                λL[i,j] = λLs[1]
            else
                FL_norm[i,j] = FL_norms[2]
                λL[i,j] = λLs[2]
            end
        else
            FL_norm[i,j] = FL_norms[1]
            λL[i,j] = λLs[1]
        end
    end
    return λL, FL_norm
end

function norm_FRint(AR)
    Ni,Nj = size(AR)
    arraytype = _arraytype(AR[1,1])
    norm_FR = Array{arraytype,2}(undef, Ni, Nj)
    @threads for k = 1:Ni*Nj
        i, j = ktoij(k, Ni, Nj)
        D = size(AR[i,j],1)
        norm_FR[i,j] = randinitial(AR[1,1], D, D; dir = [-1,-1])
    end
    return norm_FR
end

"""
    FR_normm = norm_FRmap(ARui, ARdi, FR_norm, J)

```
──┐       ... ─── ARᵢⱼ₋₁  ── ARᵢⱼ  ──┐      a───┬───c   
 FRm  =            │          │     FRm     │   b   │ 
──┘       ... ─── ARᵢⱼ₋₁ ─── ARᵢⱼ  ──┘      d───┴───e  
```
"""
function norm_FRmap(ARui, ARdi, FR_norm, J)
    Nj = size(ARui,1)
    FR_normm = copy(FR_norm)
    for j=1:Nj
        jr = J-(j-1) + (J-(j-1) < 1)*Nj
        FR_normm = ein"(ce,abc),dbe -> ad"(FR_normm,ARui[jr],ARdi[jr])
    end
    return FR_normm
end

norm_FR(ARu, ARd, FR_norm = norm_FRint(ARu); kwargs...) = norm_FR!(ARu, ARd, FR_norm; kwargs...)
function norm_FR!(ARu, ARd, FR_norm; kwargs...)
    Ni,Nj = size(ARu)
    λL = zeros(eltype(FR_norm[1,1]),Ni,Nj)
    @threads for k = 1:Ni*Nj
        i, j = ktoij(k, Ni, Nj)
        λLs, FR_norms, info= eigsolve(X->norm_FRmap(ARu[i,:], ARd[i,:], X, j), FR_norm[i,j], 1, :LM; ishermitian = false, kwargs...)
        @debug "norm_FR! eigsolve" λLs info sort(abs.(λLs))
        if length(λLs) > 1 && norm(abs(λLs[1]) - abs(λLs[2])) < 1e-12
            @show λLs
            if real(λLs[1]) > 0
                FR_norm[i,j] = FR_norms[1]
                λL[i,j] = λLs[1]
            else
                FR_norm[i,j] = FR_norms[2]
                λL[i,j] = λLs[2]
            end
        else
            FR_norm[i,j] = FR_norms[1]
            λL[i,j] = λLs[1]
        end
    end
    return λL, FR_norm
end