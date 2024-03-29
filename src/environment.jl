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

function env_norm(F::AbstractArray{T,5}) where T
    Ni,Nj = size(F)[end-1:end]
    buf = Zygote.Buffer(F)
    @inbounds @views for j in 1:Nj, i in 1:Ni
        buf[:,:,:,i,j] = F[:,:,:,i,j]/norm(F[:,:,:,i,j])
    end
    return copy(buf)
end

function env_norm(F::AbstractArray{T,4}) where T
    Ni,Nj = size(F)[[3,4]]
    buf = Zygote.Buffer(F)
    @inbounds @views for j in 1:Nj, i in 1:Ni
        buf[:,:,i,j] = F[:,:,i,j]/norm(F[:,:,i,j])
    end
    return copy(buf)
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

function cellones(A)
    χ, Ni, Nj = size(A)[[1,4,5]]
    atype = _arraytype(A)
    Cell = atype == Array ? zeros(ComplexF64, χ,χ,Ni,Nj) : CUDA.zeros(ComplexF64, χ,χ,Ni,Nj)
    for j = 1:Nj, i = 1:Ni
        Cell[:,:,i,j] = atype{ComplexF64}(I, χ, χ)
    end
    return Cell
end

function ρmap(ρ,A)
    Ni, Nj = size(A)[end-1:end]
    ρ = copy(ρ)
    @inbounds @views for j in 1:Nj, i in 1:Ni
        jr = j + 1 - Nj * (j==Nj)
        ρ[:,:,i,jr] .= ein"(dc,csb),dsa -> ab"(ρ[:,:,i,j], A[:,:,:,i,j], conj(A[:,:,:,i,j]))
    end
    return ρ
end

function initialA(M, χ)
    D, Ni, Nj = size(M)[[4,5,6]]
    atype = _arraytype(M)
    A = atype == Array ? rand(ComplexF64, χ,D,χ,Ni,Nj) : CUDA.rand(ComplexF64, χ,D,χ,Ni,Nj)
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
    Ni,Nj = size(A)[end-1:end]
    λs, ρs, info = eigsolve(ρ->ρmap(ρ,A), L, 1, :LM; ishermitian = false, maxiter = 1, kwargs...)
    @debug "getL eigsolve" λs info sort(abs.(λs))
    info.converged == 0 && @warn "getL not converged"
    _, ρs1 = selectpos(λs, ρs, Nj)
    @inbounds @views for j = 1:Nj, i = 1:Ni
        ρ = ρs1[:,:,i,j] + ρs1[:,:,i,j]'
        ρ ./= tr(ρ)
        F = svd!(ρ)
        Lo = lmul!(Diagonal(sqrt.(F.S)), F.Vt)
        _, R = qrpos!(Lo)
        L[:,:,i,j] = R
    end
    return L
end

"""
    getAL(A,L)

Given an MPS tensor `A` and `L` ，return a left-canonical MPS tensor `AL`, a gauge transform `R` and
a scalar factor `λ` such that ``λ AR R = L A``
"""
function getAL(A,L)
    χ, D, Ni,Nj = size(A)[[1,2,4,5]]
    AL = similar(A)
    Le = similar(L)
    λ = zeros(Ni,Nj)
    @inbounds @views for j = 1:Nj, i = 1:Ni
        Q, R = qrpos!(reshape(L[:,:,i,j]*reshape(A[:,:,:,i,j], χ, D*χ), D*χ, χ))
        AL[:,:,:,i,j] = reshape(Q, χ, D, χ)
        λ[i,j] = norm(R)
        Le[:,:,i,j] = rmul!(R, 1/λ[i,j])
    end
    return AL, Le, λ
end

function getLsped(Le, A, AL; kwargs...)
    Ni,Nj = size(A)[end-1:end]
    L = similar(Le)
    @inbounds @views for j = 1:Nj, i = 1:Ni
        λs, Ls, info = eigsolve(X -> ein"(dc,csb),dsa -> ab"(X,A[:,:,:,i,j],conj(AL[:,:,:,i,j])), Le[:,:,i,j], 1, :LM; ishermitian = false, kwargs...)
        @debug "getLsped eigsolve" λs info sort(abs.(λs))
        info.converged == 0 && @warn "getLsped not converged"
        _, Ls1 = selectpos(λs, Ls, Nj)
        _, R = qrpos!(Ls1)
        L[:,:,i,j] = R
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
    Ni,Nj = size(A)[end-1:end]
    Ar = similar(A)
    Lr = similar(L)
    @inbounds @views for j = 1:Nj, i = 1:Ni
        Ar[:,:,:,i,j] = permutedims(A[:,:,:,i,j],(3,2,1))
        Lr[:,:,  i,j] = permutedims(L[:,:,  i,j],(2,1))
    end
    AL, L, λ = leftorth(Ar,Lr; tol = tol, maxiter = maxiter, kwargs...)
    R  = similar(L)
    AR = similar(AL)
    @inbounds @views for j = 1:Nj, i = 1:Ni
         R[:,:,  i,j] = permutedims( L[:,:,  i,j],(2,1))
        AR[:,:,:,i,j] = permutedims(AL[:,:,:,i,j],(3,2,1))
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
    Rijr = circshift(R, (0,0,0,-1))
    ein"abij,bcij -> acij"(L, Rijr)
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
    FLij = ein"((adfi,abci),dgebi),fghi -> cehi"(FLi,ALui,Mi,ALdir)
    circshift(FLij, (0,0,0,1))
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
    FRij = ein"((cehi,abci),dgebi),fghi -> adfi"(FRi,ARui,Mi,ARdir)
    circshift(FRij, (0,0,0,-1))
end

function FLint(AL, M)
    χ, Ni, Nj = size(AL)[[1,4,5]]
    D = size(M, 1)
    atype = _arraytype(AL)
    FL = atype == Array ? rand(ComplexF64, χ, D, χ, Ni, Nj) : CUDA.rand(ComplexF64, χ, D, χ, Ni, Nj)
    return FL
end

function FRint(AR, M)
    χ, Ni, Nj = size(AR)[[1,4,5]]
    D = size(M, 3)
    atype = _arraytype(AR)
    FR = atype == Array ? rand(ComplexF64, χ, D, χ, Ni, Nj) : CUDA.rand(ComplexF64, χ, D, χ, Ni, Nj)
    return FR
end

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
    Ni,Nj = size(M)[[5,6]]
    λL = zeros(eltype(FL),Ni)
    for i in 1:Ni
        ir = ifobs ? Ni+1-i : i+1-Ni*(i==Ni)
        λLs, FL1s, info = eigsolve(X->FLmap(ALu[:,:,:,i,:], ALd[:,:,:,ir,:], M[:,:,:,:,i,:], X), FL[:,:,:,i,:], 1, :LM; maxiter=100, ishermitian = false, kwargs...)
        @debug "leftenv! eigsolve" λLs info sort(abs.(λLs))
        info.converged == 0 && @warn "leftenv not converged"
        λL[i], FL[:,:,:,i,:] = selectpos(λLs, FL1s, Nj)
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
    Ni,Nj = size(M)[[5,6]]
    λR = zeros(eltype(FR),Ni)
    for i in 1:Ni
        ir = ifobs ? Ni+1-i : i+1-Ni*(i==Ni)
        λRs, FR1s, info= eigsolve(X->FRmap(ARu[:,:,:,i,:], ARd[:,:,:,ir,:], M[:,:,:,:,i,:], X), FR[:,:,:,i,:], 1, :LM; maxiter=100, ishermitian = false, kwargs...)
        @debug "rightenv! eigsolve" λRs info sort(abs.(λRs))
        info.converged == 0 && @warn "rightenv not converged"
        λR[i], FR[:,:,:,i,:] = selectpos(λRs, FR1s, Nj)
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
    ACij = ein"((adfj,abcj),dgebj),cehj -> fghj"(FLj,ACj,Mj,FRj)
    circshift(ACij, (0,0,0,1))
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
    Cij = ein"(acdj,abj),bcej -> dej"(FLjr,Cj,FRj)
    circshift(Cij, (0,0,1))
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
    Ni,Nj = size(M)[[5,6]]
    λAC = zeros(eltype(AC),Nj)
    for j in 1:Nj
        λACs, ACs, info = eigsolve(X->ACmap(X, FL[:,:,:,:,j], FR[:,:,:,:,j], M[:,:,:,:,:,j]), AC[:,:,:,:,j], 1, :LM; maxiter=100, ishermitian = false, kwargs...)
        @debug "ACenv! eigsolve" λACs info sort(abs.(λACs))
        info.converged == 0 && @warn "ACenv Not converged"
        λAC[j], AC[:,:,:,:,j] = selectpos(λACs, ACs, Ni)
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
    Ni,Nj = size(C)[[3,4]]
    λC = zeros(eltype(C),Nj)
    for j in 1:Nj
        jr = j + 1 - Nj * (j==Nj)
        λCs, Cs, info = eigsolve(X->Cmap(X, FL[:,:,:,:,jr], FR[:,:,:,:,j]), C[:,:,:,j], 1, :LM; maxiter=100, ishermitian = false, kwargs...)
        @debug "Cenv! eigsolve" λCs info sort(abs.(λCs))
        info.converged == 0 && @warn "Cenv Not converged"
        λC[j], C[:,:,:,j] = selectpos(λCs, Cs, Ni)
    end
    return λC, C
end

function ACCtoAL(AC, C)
    χ, D, Ni, Nj = size(AC)[[1,2,4,5]]
    errL = 0.0
    AL = Zygote.Buffer(AC)
    @inbounds @views for j in 1:Nj, i in 1:Ni
        QAC, RAC = qrpos(reshape(AC[:,:,:,i,j],(χ*D, χ)))
         QC, RC  = qrpos(C[:,:,i,j])
        errL += norm(RAC-RC)
        AL[:,:,:,i,j] = reshape(QAC*QC', (χ, D, χ))
    end
    return copy(AL), errL
end

function ACCtoAR(AC, C)
    χ, D, Ni, Nj = size(AC)[[1,2,4,5]]
    errR = 0.0
    AR = Zygote.Buffer(AC)
    @inbounds @views for j in 1:Nj, i in 1:Ni
        jr = j - 1 + (j==1)*Nj
        LAC, QAC = lqpos(reshape(AC[:,:,:,i,j],(χ, D*χ)))
         LC, QC  = lqpos(C[:,:,i,jr])
        errR += norm(LAC-LC)
        AR[:,:,:,i,j] = reshape(QC'*QAC, (χ, D, χ))
    end
    return copy(AR), errR
end

function ALCtoAC(AL,C)
    ein"ascij,cbij -> asbij"(AL, C)
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
    AC = env_norm(AC)
     C = env_norm( C)
    AL, errL = ACCtoAL(AC, C)
    AR, errR = ACCtoAR(AC, C)
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
    Ni,Nj = size(AL)[end-1:end]
    AC = ALCtoAC(AL, C)
    err = 0
    for _ in 1:Ni, j in 1:Nj
        AC[:,:,:,:,j] = ACmap(AC[:,:,:,:,j], FL[:,:,:,:,j], FR[:,:,:,:,j], M[:,:,:,:,:,j])
    end   
    # MAC = AC
    AC .-= ein"(apcij,dpcij),dsbij -> asbij"(AC,conj(AR),AR)
    @inbounds @views for j = 1:Nj, i = 1:Ni
        err += norm(AC[:,:,:,i,j])
    end
    return err
end