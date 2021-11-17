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
function ktoij(k,Ni,Nj)
    Cart = CartesianIndices((1:Ni,1:Nj))
    Index = Cart[k]
    Index[1],Index[2]
end

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

function cellones(A)
    Ni, Nj = size(A)
    D = size(A[1,1],1)
    Cell = Array{_arraytype(A[1,1]){ComplexF64,2},2}(undef, Ni, Nj)
    for j = 1:Nj, i = 1:Ni
        Cell[i,j] = _mattype(A){ComplexF64}(I, D, D)
    end
    return Cell
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
    arraytype = _arraytype(M[1,1])
    A = Array{arraytype{ComplexF64,3},2}(undef, Ni, Nj)
    for j = 1:Nj, i = 1:Ni
        d = size(M[i,j], 4)
        A[i,j] = arraytype(rand(ComplexF64, D, d, D))
    end
    return A
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
    for j = 1:Nj, i = 1:Ni
        _,ρs,_ = eigsolve(ρ->ρmap(ρ,A[i,:],j), L[i,j]'*L[i,j], 1, :LM; ishermitian = false, maxiter = 1, kwargs...)
        ρ = ρs[1] + ρs[1]'
        ρ ./= tr(ρ)
        F = svd!(ρ)
        Lo = lmul!(Diagonal(sqrt.(F.S)), F.Vt)
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
    arraytype = _arraytype(A[1,1])
    AL = Array{arraytype{ComplexF64,3},2}(undef, Ni, Nj)
    Le = Array{arraytype{ComplexF64,2},2}(undef, Ni, Nj)
    λ = zeros(Ni,Nj)
    for j = 1:Nj, i = 1:Ni
        D, d, = size(A[i,j])
        Q, R = qrpos!(reshape(L[i,j]*reshape(A[i,j], D, d*D), D*d, D))
        AL[i,j] = reshape(Q, D, d, D)
        λ[i,j] = norm(R)
        Le[i,j] = rmul!(R, 1/λ[i,j])
    end
    return AL, Le, λ
end

function getLsped(Le, A, AL; kwargs...)
    Ni,Nj = size(A)
    L = Array{_arraytype(A[1,1]){ComplexF64,2},2}(undef, Ni, Nj)
    for j = 1:Nj, i = 1:Ni
        _, Ls, _ = eigsolve(X -> ein"(dc,csb),dsa -> ab"(X,A[i,j],conj(AL[i,j])), Le[i,j], 1, :LM; ishermitian = false, kwargs...)
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
    L = getL!(A,L; kwargs...)
    AL, Le, λ = getAL(A,L;kwargs...)
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
    Ni,Nj = size(A)
    arraytype = _arraytype(A[1,1])
    Ar = Array{arraytype{ComplexF64,3},2}(undef, Ni, Nj)
    Lr = Array{arraytype{ComplexF64,2},2}(undef, Ni, Nj)
    for j = 1:Nj, i = 1:Ni
        Ar[i,j] = permutedims(A[i,j],(3,2,1))
        Lr[i,j] = permutedims(L[i,j],(2,1))
    end
    AL, L, λ = leftorth(Ar,Lr; tol = tol, maxiter = maxiter, kwargs...)
    R = Array{arraytype{ComplexF64,2},2}(undef, Ni, Nj)
    AR = Array{arraytype{ComplexF64,3},2}(undef, Ni, Nj)
    for j = 1:Nj, i = 1:Ni
        R[i,j] = permutedims(L[i,j],(2,1))
        AR[i,j] = permutedims(AL[i,j],(3,2,1))
    end
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
    arraytype = _arraytype(L[1,1])
    C = Array{arraytype{ComplexF64,2},2}(undef, Ni, Nj)
    for j in 1:Nj,i in 1:Ni
        jr = j + 1 - (j + 1 > Nj) * Nj
        C[i,j] = L[i,j] * R[i,jr]
    end
    return C
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
    arraytype = _arraytype(AL[1,1])
    FL = Array{arraytype{ComplexF64,3},2}(undef, Ni, Nj)
    for j = 1:Nj, i = 1:Ni
        D = size(AL[i,j],1)
        dL = size(M[i,j],1)
        FL[i,j] = arraytype(rand(ComplexF64, D, dL, D))
    end
    return FL
end

function FRint(AR, M)
    Ni,Nj = size(AR)
    arraytype = _arraytype(AR[1,1])
    FR = Array{arraytype{ComplexF64,3},2}(undef, Ni, Nj)
    for j = 1:Nj, i = 1:Ni
        D = size(AR[i,j],1)
        dR = size(M[i,j],3)
        FR[i,j] = arraytype(rand(ComplexF64, D, dR, D))
    end
    return FR
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
    for j = 1:Nj, i = 1:Ni
        ir = i + 1 - Ni * (i==Ni)
        λLs, FL1s, _= eigsolve(X->FLmap(ALu[i,:], conj(ALd[ir,:]), M[i,:], X, j), FL[i,j], 1, :LM; ishermitian = false, kwargs...)
        if length(λLs) > 1 && norm(abs(λLs[1]) - abs(λLs[2])) < 1e-12
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
    for j = 1:Nj, i = 1:Ni
        ir = i + 1 - Ni * (i==Ni)
        λRs, FR1s, _= eigsolve(X->FRmap(ARu[i,:], conj(ARd[ir,:]), M[i,:], X, j), FR[i,j], 1, :LM; ishermitian = false, kwargs...)
        if length(λRs) > 1 && norm(abs(λRs[1]) - abs(λRs[2])) < 1e-12
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
    for j = 1:Nj, i = 1:Ni
        λACs, ACs, = eigsolve(X->ACmap(X, FL[:,j], FR[:,j], M[:,j], i), AC[i,j], 1, :LM; ishermitian = false, kwargs...)
        if length(λACs) > 1 && norm(abs(λACs[1]) - abs(λACs[2])) < 1e-12
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
    for j = 1:Nj, i = 1:Ni
        jr = j + 1 - (j==Nj) * Nj
        λCs, Cs, = eigsolve(X->Cmap(X, FL[:,jr], FR[:,j], i), C[i,j], 1, :LM; ishermitian = false, kwargs...)
        if length(λCs) > 1 && norm(abs(λCs[1]) - abs(λCs[2])) < 1e-12
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
    QAC, RAC = qrpos(reshape(ACij,(D*d, D)))
    QC, RC = qrpos(Cij)
    errL = norm(RAC-RC)
    # @show errL
    reshape(QAC*QC', (D, d, D)), errL
end

function ACCtoAR(ACij,Cijr)
    D,d, = size(ACij)
    LAC, QAC = lqpos(reshape(ACij,(D, d*D)))
    LC, QC = lqpos(Cijr)
    errR = norm(LAC-LC)
    # @show errR
    reshape(QC'*QAC, (D, d, D)), errR
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
    ACij = [ein"asc,cb -> asb"(AL[i],C[i]) for i=1:Ni*Nj]
    reshape(ACij,Ni,Nj)
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
    err = 0
    for j = 1:Nj, i = 1:Ni
        MAC = ACmap(AC[i,j], FL[:,j], FR[:,j], M[:,j], i)
        MAC -= ein"(apc,dpc),dsb -> asb"(MAC,conj(AR[i,j]),AR[i,j])
        err += norm(MAC)
    end
    return err
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
    for j = 1:Nj, i = 1:Ni
        ir = Ni + 1 - i
        λLs, FL1s, _= eigsolve(X->FLmap(ALu[i,:], ALd[ir,:], M[i,:], X, j), FL[i,j], 1, :LM; ishermitian = false, kwargs...)
        if length(λLs) > 1 && norm(abs(λLs[1]) - abs(λLs[2])) < 1e-12
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
    for j = 1:Nj, i = 1:Ni
        ir = Ni + 1 - i
        λRs, FR1s, _= eigsolve(X->FRmap(ARu[i,:], ARd[ir,:], M[i,:], X, j), FR[i,j], 1, :LM; ishermitian = false, kwargs...)
        if length(λRs) > 1 && norm(abs(λRs[1]) - abs(λRs[2])) < 1e-12
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

function BgFLint(AL, M)
    Ni,Nj = size(AL)
    arraytype = _arraytype(AL[1,1])
    BgFL = Array{arraytype{ComplexF64,4},2}(undef, Ni, Nj)
    for j = 1:Nj, i = 1:Ni
        ir = i + 1 - Ni * (i==Ni)
        irr = i + 2 - Ni * (i + 2 > Ni)
        D1 = size(AL[i,j],1)
        D2 = size(AL[irr,j],1)
        dL1 = size(M[i,j],1)
        dL2 = size(M[ir,j],1)
        BgFL[i,j] = arraytype(rand(ComplexF64, D1, dL1, dL2, D2))
    end
    return BgFL
end

"""
    BgFLm = BgFLmap(ALi, ALip, Mi, Mip, BgFLij, J)

ALip means ALᵢ₊₂
Mip means Mᵢ₊₁
```
  ┌──        ┌──  ALᵢⱼ  ── ALᵢⱼ₊₁    ──   ...       a ────┬──── c
  │          │     │        │                       │     b     │
  │          │  ─ Mᵢⱼ   ── Mᵢⱼ₊₁     ──   ...       ├─ d ─┼─ e ─┤
BgFLm   =  BgFLᵢⱼ  │        │                       │     f     │
  │          │  ─ Mᵢ₊₁ⱼ ── Mᵢ₊₁ⱼ₊₁   ──   ...       ├─ g ─┼─ h ─┤
  │          │     │        │                       │     j     │
  ┕──        ┕──  ALᵢ₊₂ⱼ ─ ALᵢ₊₂ⱼ₊₁  ──   ...       i ────┴──── k 
```
"""
function BgFLmap(ALi, ALip, Mi, Mip, BgFLij, J)
    Nj = size(ALi,1)
    BgFLm = copy(BgFLij)
    for j=1:Nj
        jr = J+j-1 - (J+j-1 > Nj)*Nj
        BgFLm = ein"(((adgi,abc),dfeb),gjhf),ijk -> cehk"(BgFLm,ALi[jr],Mi[jr],Mip[jr],ALip[jr])
    end
    return BgFLm
end

"""
    λL, BgFL = bigleftenv(ALu, ALd, M, BgFL = BgFLint(AL,M); kwargs...)

Compute the up and down left environment tensor for MPS A and MPO M, by finding the left fixed point
of AL - M - M - conj(AL) contracted along the physical dimension.
```
   ┌──  ALuᵢⱼ ── ALuᵢⱼ₊₁   ──   ...           ┌── 
   │     │        │                           │   
   │  ─ Mᵢⱼ   ── Mᵢⱼ₊₁     ──   ...           │   
 BgFLᵢⱼ  │        │                   = λLᵢⱼ BgFLᵢⱼ
   │  ─ Mᵢ₊₁ⱼ ── Mᵢ₊₁ⱼ₊₁   ──   ...           │   
   │     │        │                           │   
   ┕── ALdᵢ₊₂ⱼ ─ ALdᵢ₊₂ⱼ₊₁ ──   ...           ┕── 
```
"""
bigleftenv(ALu, ALd, M, BgFL = BgFLint(ALu,M); kwargs...) = bigleftenv!(ALu, ALd, M, copy(BgFL); kwargs...)
function bigleftenv!(ALu, ALd, M, BgFL; kwargs...)
    Ni,Nj = size(ALu)
    λL = zeros(eltype(BgFL[1,1]),Ni,Nj)
    for j = 1:Nj, i = 1:Ni
        ir = i + 1 - Ni * (i==Ni)
        irr = i + 2 - Ni * (i + 2 > Ni)
        λLs, BgFL1s, _= eigsolve(X->BgFLmap(ALu[i,:], ALd[irr,:], M[i,:], M[ir,:], X, j), BgFL[i,j], 1, :LM; ishermitian = false, kwargs...)
        if length(λLs) > 1 && norm(abs(λLs[1]) - abs(λLs[2])) < 1e-12
            @show λLs
            if real(λLs[1]) > 0
                BgFL[i,j] = BgFL1s[1]
                λL[i,j] = λLs[1]
            else
                BgFL[i,j] = BgFL1s[2]
                λL[i,j] = λLs[2]
            end
        else
            BgFL[i,j] = BgFL1s[1]
            λL[i,j] = λLs[1]
        end
    end
    return λL, BgFL
end

function BgFRint(AR, M)
    Ni,Nj = size(AR)
    arraytype = _arraytype(AR[1,1])
    BgFR = Array{arraytype{ComplexF64,4},2}(undef, Ni, Nj)
    for j = 1:Nj, i = 1:Ni
        ir = i + 1 - Ni * (i==Ni)
        irr = i + 2 - Ni * (i + 2 > Ni)
        D1 = size(AR[i,j],3)
        D2 = size(AR[irr,j],3)
        dR1 = size(M[i,j],3)
        dR2 = size(M[ir,j],3)
        BgFR[i,j] = arraytype(rand(ComplexF64, D1, dR1, dR2, D2))
    end
    return BgFR
end

"""
    FRm = FRmap(ARi, ARip, Mi, FR, J)

ARip means ARᵢ₊₁
```
 ──┐          ...  ─── ARᵢⱼ₋₁  ── ARᵢⱼ  ──┐        a ────┬──── c
   │                    │          │      │        │     b     │
 ──│          ... ──── Mᵢⱼ₋₁  ──  Mᵢⱼ   ──│        ├─ d ─┼─ e ─┤
  BgFRm   =             │          │     BgFRm     │     f     │
 ──│          ... ──── Mᵢ₊₁ⱼ₋₁ ── Mᵢ₊₁ⱼ ──│        ├─ g ─┼─ h ─┤
   │                    │          │      │        │     j     │
 ──┘          ...  ─ ARᵢ₊₂ⱼ₋₁ ─── ARᵢ₊₂ⱼ──┘        i ────┴──── k 
```
"""
function BgFRmap(ARi, ARip, Mi, Mip, BgFR, J)
    Nj = size(ARi,1)
    BgFRm = copy(BgFR)
    for j=1:Nj
        jr = J-(j-1) + (J-(j-1) < 1)*Nj
        BgFRm = ein"(((cehk,abc),dfeb),gjhf),ijk -> adgi"(BgFRm,ARi[jr],Mi[jr],Mip[jr],ARip[jr])
    end
    return BgFRm
end

"""
    λR, BgFR = bigrightenv(ARu, ARd, M, BgFR = BgFRint(ARu,M); kwargs...)

Compute the up and down right environment tensor for MPS A and MPO M, by finding the left fixed point
of AR - M - M - conj(AR) contracted along the physical dimension.
```
     ──┐          ...  ─── ARᵢⱼ₋₁  ── ARᵢⱼ  ──┐ 
       │                    │          │      │ 
     ──│          ... ──── Mᵢⱼ₋₁  ──  Mᵢⱼ   ──│
λRᵢⱼ BgFRᵢⱼ   =             │          │     BgFRᵢⱼ
     ──│          ... ──── Mᵢ₊₁ⱼ₋₁ ── Mᵢ₊₁ⱼ ──│
       │                    │          │      │     
     ──┘          ...  ─ ARᵢ₊₂ⱼ₋₁ ─── ARᵢ₊₂ⱼ──┘ 
```
"""
bigrightenv(ARu, ARd, M, BgFR = BgFRint(ARu,M); kwargs...) = bigrightenv!(ARu, ARd, M, copy(BgFR); kwargs...)
function bigrightenv!(ARu, ARd, M, BgFR; kwargs...)
    Ni,Nj = size(ARu)
    λR = zeros(eltype(BgFR[1,1]),Ni,Nj)
    for j = 1:Nj, i = 1:Ni
        ir = i + 1 - Ni * (i==Ni)
        irr = i + 2 - Ni * (i + 2 > Ni)
        λRs, BgFR1s, _= eigsolve(X->BgFRmap(ARu[i,:], ARd[irr,:], M[i,:], M[ir,:], X, j), BgFR[i,j], 1, :LM; ishermitian = false, kwargs...)
        if length(λRs) > 1 && norm(abs(λRs[1]) - abs(λRs[2])) < 1e-12
            @show λRs
            if real(λRs[1]) > 0
                BgFR[i,j] = BgFR1s[1]
                λR[i,j] = λRs[1]
            else
                BgFR[i,j] = BgFR1s[2]
                λR[i,j] = λRs[2]
            end
        else
            BgFR[i,j] = BgFR1s[1]
            λR[i,j] = λRs[1]
        end
    end
    return λR, BgFR
end

function norm_FLint(AL)
    Ni,Nj = size(AL)
    arraytype = _arraytype(AL[1,1])
    norm_FL = Array{arraytype{ComplexF64,2},2}(undef, Ni, Nj)
    for j = 1:Nj, i = 1:Ni
        D = size(AL[i,j],1)
        norm_FL[i,j] = arraytype(rand(ComplexF64, D, D))
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
    for j = 1:Nj, i = 1:Ni
        λLs, FL_norms, _= eigsolve(X->norm_FLmap(ALu[i,:], ALd[i,:], X, j), FL_norm[i,j], 1, :LM; ishermitian = false, kwargs...)
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
    norm_FR = Array{arraytype{ComplexF64,2},2}(undef, Ni, Nj)
    for j = 1:Nj, i = 1:Ni
        D = size(AR[i,j],1)
        norm_FR[i,j] = arraytype(rand(ComplexF64, D, D))
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
    for j = 1:Nj, i = 1:Ni
        λLs, FR_norms, _= eigsolve(X->norm_FRmap(ARu[i,:], ARd[i,:], X, j), FR_norm[i,j], 1, :LM; ishermitian = false, kwargs...)
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