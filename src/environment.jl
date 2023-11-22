using Base.Threads
using LinearAlgebra
using KrylovKit
using OMEinsum
using Random
using Zygote
import LinearAlgebra: mul!
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

function env_norm(F)
    Ni,Nj = size(F)
    reshape([F[i]/norm(F[i]) for i in 1:Ni*Nj], Ni,Nj)
end

"""
    λs[1], Fs[1] = selectpos(λs, Fs)

Select the max positive one of λs and corresponding Fs.
"""
function selectpos(λs, Fs, N)
    if length(λs) > 1 && norm(abs(λs[1]) - abs(λs[2])) < 1e-12
        # @show "selectpos: λs are degeneracy"
        N = min(N, length(λs))
        p = argmax(real(λs[1:N]))  
        # @show λs p abs.(λs)
        return λs[1:N][p], Fs[1:N][p]
    else
        return λs[1], Fs[1]
    end
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

function initialA(M, χ; info = nothing)
    Ni, Nj = size(M)
    atype = _arraytype(M[1,1])
    A = Array{atype{ComplexF64, 3}, 2}(undef, Ni, Nj)
    if info === nothing
        for j in 1:Nj, i in 1:Ni
            D = size(M[i,j], 4)
            A[i,j] = atype == Array ? rand(ComplexF64, χ,D,χ) : CUDA.rand(ComplexF64, χ,D,χ)
        end
    else
        indD, indχ, dimsD, dimsχ = info
        for j in 1:Nj, i in 1:Ni
            D = size(M[i,j], 4)
            # A[i,j] = randinitial(M[1,1], D,D,D; dir = [-1, 1, 1]) # ordinary random initial
            indqn = [indχ, indD, indD, indχ]
            indims = [dimsχ, dimsD, dimsD, dimsχ]
            A[i,j] = symmetryreshape(randinitial(M[1,1], χ, Int(sqrt(D)), Int(sqrt(D)), χ; 
            dir = [-1, -1, 1, 1], indqn = indqn, indims = indims
            ), 
            χ, D, χ; reinfo = (nothing, nothing, nothing, indqn, indims, nothing, nothing))[1] # for double-layer ipeps
        end
    end
    return A
end

function cellones(A; info = nothing)
    Ni, Nj = size(A)
    χ = size(A[1,1],1)
    atype = _arraytype(A[1,1])
    Cell = Array{atype, 2}(undef, Ni, Nj)
    if info === nothing
        for j = 1:Nj, i = 1:Ni
            Cell[i,j] = atype{ComplexF64}(I, χ, χ)
        end
    else
        _, indχ, _, dimsχ = info
        dir = getdir(A[1,1])[[1,3]]
        for k = 1:Ni*Nj
            i, j = ktoij(k, Ni, Nj)
            Cell[i,j] = Iinitial(A[1,1], χ; dir = dir, indqn = [indχ, indχ], indims = [dimsχ, dimsχ])
        end
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
    for k = 1:Ni*Nj
        i, j = ktoij(k, Ni, Nj)
        λ,ρs,info = eigsolve(ρ->ρmap(ρ,A[i,:],j), L[i,j]'*L[i,j], 1, :LM; ishermitian = false, maxiter = 1, kwargs...)
        @debug "getL eigsolve" λ info sort(abs.(λ))
        info.converged == 0 && @warn "getL not converged"
        ρ = ρs[1] + ρs[1]'
        ρ /= tr(ρ)
        _, S, Vt = sysvd!(ρ)
        Lo = Diagonal(sqrt.(S)) * Vt
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
    for k = 1:Ni*Nj
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
    for k = 1:Ni*Nj
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
function leftorth(A, L = cellones(A); tol = 1e-12, maxiter = 100, kwargs...)
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
function rightorth(A, L=cellones(A); tol = 1e-12, maxiter = 100, kwargs...)
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
    [L[i,j] * R[i, mod1(j+1, Nj)] for i=1:Ni, j=1:Nj]
end

"""
    FLm = FLmap(ALu, ALd, M, FL)

```
  ┌──       ┌──  ALuᵢⱼ  ──                     a ────┬──── c 
  │         │     │                            │     b     │ 
FLᵢⱼ₊₁ =   FLᵢⱼ ─ Mᵢⱼ   ──                     ├─ d ─┼─ e ─┤ 
  │         │     │                            │     g     │ 
  └──       └──  ALdᵢᵣⱼ  ─                     f ────┴──── h 
```
"""

function FLmap(ALui, ALdir, Mi, FLi)
    Nj = size(ALui,1)
    FLij = [ein"((adf,abc),dgeb),fgh -> ceh"(FLi[j],ALui[j],Mi[j],ALdir[j]) for j = 1:Nj]
    circshift(FLij, 1)
end

"""
    FRm = FRmap(ARu, ARd, M, FR, i)

```
    ── ARuᵢⱼ  ──┐          ──┐          a ────┬──── c 
        │       │            │          │     b     │ 
    ── Mᵢⱼ   ──FRᵢⱼ  =    ──FRᵢⱼ₋₁      ├─ d ─┼─ e ─┤ 
        │       │            │          │     g     │ 
    ── ARdᵢᵣⱼ ──┘          ──┘          f ────┴──── h 
```
"""
function FRmap(ARui, ARdir, Mi, FRi)
    Nj = size(ARui,1)
    FRij = [ein"((ceh,abc),dgeb),fgh -> adf"(FRi[j],ARui[j],Mi[j],ARdir[j]) for j in 1:Nj]
    circshift(FRij, -1)
end

function FLint(AL, M; info = nothing)
    Ni,Nj = size(AL)
    χ = size(AL[1],1)
    atype = _arraytype(AL[1])
    FL = Array{atype{ComplexF64, 3}, 2}(undef, Ni, Nj)
    if info === nothing
        for j in 1:Nj, i in 1:Ni
            D = size(M[i,j], 1)
            FL[i,j] = atype == Array ? rand(ComplexF64, χ, D, χ) : CUDA.rand(ComplexF64, χ, D, χ)
        end
    else
        indD, indχ, dimsD, dimsχ = info
        dir = [1, getdir(M[1])[1], -getdir(M[1])[1], -1]
        indqn = [indχ, indD, indD, indχ]
        indims = [dimsχ, dimsD, dimsD, dimsχ]
        for j in 1:Nj, i in 1:Ni
            D = size(M[i,j], 1)
            FL[i,j] = symmetryreshape(randinitial(AL[i,j], χ, Int(sqrt(D)), Int(sqrt(D)), χ; 
            dir = dir, indqn = indqn, indims = indims
            ), χ, D, χ; 
            reinfo = (nothing, nothing, nothing, indqn, indims, nothing, nothing))[1]
        end
    end
    return FL
end

function FRint(AR, M; info = nothing)
    Ni,Nj = size(AR)
    χ = size(AR[1],3)
    atype = _arraytype(AR[1])
    FR = Array{atype{ComplexF64, 3}, 2}(undef, Ni, Nj)

    if info === nothing
        for j in 1:Nj, i in 1:Ni
            D = size(M[i,j], 3)
            FR[i,j] = atype == Array ? rand(ComplexF64, χ, D, χ) : CUDA.rand(ComplexF64, χ, D, χ)
        end
    else
        indD, indχ, dimsD, dimsχ = info
        dir = nothing
        typeof(AR[1]) <: U1Array && (dir = [-1, getdir(M[1])[3], -getdir(M[1])[3], 1])
        # [randinitial(AR[i,j], χ,D,χ; dir = [-1,-1,1]) for i=1:Ni, j=1:Nj]
        indqn = [indχ, indD, indD, indχ]
        indims = [dimsχ, dimsD, dimsD, dimsχ]
        for j in 1:Nj, i in 1:Ni
            D = size(M[i,j], 3)
            FR[i,j] = symmetryreshape(randinitial(AR[i,j], χ, Int(sqrt(D)), Int(sqrt(D)), χ; 
            dir = dir, indqn = indqn, indims = indims
            ), χ, D, χ; 
            reinfo = (nothing, nothing, nothing, indqn, indims, nothing, nothing))[1]
        end
    end
    return FR
end

mul!(Y::Vector{<:AbstractArray}, A::Vector{<:AbstractArray}, B::Number) = (map((x,a)->mul!(x,a,B), Y, A); Y)

"""
    λL, FL = leftenv(ALu, ALd, M, FL = FLint(ALu,M); kwargs...)

Compute the left environment tensor for MPS A and MPO M, by finding the left fixed point
of ALu - M - ALd contracted along the physical dimension.
```
 ┌──  ALuᵢⱼ  ──          ┌── 
 │     │                 │   
FLᵢⱼ ─ Mᵢⱼ   ──   = λLᵢⱼ FLᵢⱼ₊₁   
 │     │                 │   
 └──  ALdᵢᵣⱼ  ─          └── 
```
"""
leftenv(ALu, ALd, M, FL = FLint(ALu,M); kwargs...) = leftenv!(ALu, ALd, M, copy(FL); kwargs...) 
function leftenv!(ALu, ALd, M, FL; ifobs=false, kwargs...) 
    Ni,Nj = size(M)
    λL = zeros(ComplexF64,Ni)
    for i in 1:Ni
        ir = ifobs ? Ni+1-i : mod1(i+1, Ni)
        λLs, FL1s, info = eigsolve(X->FLmap(ALu[i,:], ALd[ir,:], M[i,:], X), FL[i,:], 1, :LM; maxiter=100, ishermitian = false, kwargs...)
        @debug "leftenv! eigsolve" λLs info sort(abs.(λLs))
        info.converged == 0 && @warn "leftenv not converged"
        λL[i], FL[i,:] = selectpos(λLs, FL1s, Nj)
    end
    return λL, FL
end

"""
    λR, FR = rightenv(ARu, ARd, M, FR = FRint(ARu,M); kwargs...)

Compute the right environment tensor for MPS A and MPO M, by finding the left fixed point
of AR - M - conj(AR) contracted along the physical dimension.
```
    ── ARuᵢⱼ  ──┐          ──┐   
        │       │            │  
    ── Mᵢⱼ   ──FRᵢⱼ  = λRᵢⱼ──FRᵢⱼ₋₁
        │       │            │  
    ── ARdᵢᵣⱼ ──┘          ──┘  
```
"""
rightenv(ARu, ARd, M, FR = FRint(ARu,M); kwargs...) = rightenv!(ARu, ARd, M, copy(FR); kwargs...) 
function rightenv!(ARu, ARd, M, FR; ifobs=false, kwargs...) 
    Ni,Nj = size(M)
    λR = zeros(ComplexF64,Ni)
    for i in 1:Ni
        ir = ifobs ? Ni+1-i : mod1(i+1, Ni)
        λRs, FR1s, info= eigsolve(X->FRmap(ARu[i,:], ARd[ir,:], M[i,:], X), FR[i,:], 1, :LM; maxiter=100, ishermitian = false, kwargs...)
        @debug "rightenv! eigsolve" λRs info sort(abs.(λRs))
        info.converged == 0 && @warn "rightenv not converged"
        λR[i], FR[i,:] = selectpos(λRs, FR1s, Nj)
    end
    return λR, FR
end

"""
    ACm = ACmap(ACij, FLj, FRj, Mj, II)

```
                                ┌─────── ACᵢⱼ ─────┐              a ────┬──── c  
┌───── ACᵢ₊₁ⱼ ─────┐            │        │         │              │     b     │ 
│        │         │      =     FLᵢⱼ ─── Mᵢⱼ ───── FRᵢⱼ           ├─ d ─┼─ e ─┤ 
                                │        │         │              │     g     │ 
                                                                  f ────┴──── h 
                                                               
```
"""
function ACmap(ACj, FLj, FRj, Mj)
    Ni = size(ACj,1)
    ACij = [ein"((adf,abc),dgeb),ceh -> fgh"(FLj[i],ACj[i],Mj[i],FRj[i]) for i = 1:Ni]
    circshift(ACij, 1)
end

"""
    Cmap(Cij, FLjp, FRj, II)

```
                    ┌────Cᵢⱼ ───┐            a ─── b
┌── Cᵢ₊₁ⱼ ──┐       │           │            │     │
│           │  =   FLᵢⱼ₊₁ ──── FRᵢⱼ          ├─ c ─┤
                    │           │            │     │
                                             d ─── e                                    
```
"""
function Cmap(Cj, FLjr, FRj)
    Ni = size(Cj, 1)
    Cij = [ein"(acd,ab),bce -> de"(FLjr[i],Cj[i],FRj[i]) for i = 1:Ni]
    circshift(Cij, 1)
end

"""
    ACenv(AC, FL, M, FR;kwargs...)

Compute the up environment tensor for MPS `FL`,`FR` and MPO `M`, by finding the up fixed point
        of `FL - M - FR` contracted along the physical dimension.
```
┌─────── ACᵢⱼ ─────┐         
│        │         │         =  λACᵢⱼ ┌─── ACᵢ₊₁ⱼ ──┐
FLᵢⱼ ─── Mᵢⱼ ───── FRᵢⱼ               │      │      │   
│        │         │   
```
"""
ACenv(AC, FL, M, FR; kwargs...) = ACenv!(copy(AC), FL, M, FR; kwargs...)
function ACenv!(AC, FL, M, FR; kwargs...)
    Ni,Nj = size(M)
    λAC = zeros(ComplexF64,Nj)
    for j in 1:Nj
        λACs, ACs, info = eigsolve(X->ACmap(X, FL[:,j], FR[:,j], M[:,j]), AC[:,j], 1, :LM; maxiter=100, ishermitian = false, kwargs...)
        info.converged == 0 && @warn "ACenv Not converged"
        λAC[j], AC[:,j] = selectpos(λACs, ACs, Ni)
    end
    return λAC, AC
end

"""
    Cenv(C, FL, FR;kwargs...)

Compute the up environment tensor for MPS `FL` and `FR`, by finding the up fixed point
    of `FL - FR` contracted along the physical dimension.
```
┌────Cᵢⱼ ───┐
│           │       =  λCᵢⱼ ┌──Cᵢⱼ ─┐
FLᵢⱼ₊₁ ──── FRᵢⱼ            │       │
│           │   
```
"""
Cenv(C, FL, FR; kwargs...) = Cenv!(copy(C), FL, FR; kwargs...)
function Cenv!(C, FL, FR; kwargs...)
    Ni,Nj = size(C)
    λC = zeros(ComplexF64, Nj)
    for j in 1:Nj
        jr = mod1(j+1, Nj)
        λCs, Cs, info = eigsolve(X->Cmap(X, FL[:,jr], FR[:,j]), C[:,j], 1, :LM; maxiter=100, ishermitian = false, kwargs...)
        info.converged == 0 && @warn "Cenv Not converged"
        λC[j], C[:,j] = selectpos(λCs, Cs, Ni)
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
    AC = env_norm(AC)
     C = env_norm(C)
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
    for _ in 1:Ni, j in 1:Nj
        AC[:,j] = ACmap(AC[:,j], FL[:,j], FR[:,j], M[:,j])
    end   
    # MAC = AC
    @inbounds @views for j = 1:Nj, i = 1:Ni
        AC[i,j] -= ein"(apc,dpc),dsb -> asb"(AC[i,j],conj(AR[i,j]),AR[i,j])
        err += norm(AC[i,j])
    end
    return err
end