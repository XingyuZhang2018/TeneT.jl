"""
    VUMPSEnv{T<:Number, S<:IndexSpace,
             OT<:AbstractArray{S, 2, 2},
             ET<:AbstractArray{S, 2, 1},
             CT<:AbstractArray{S, 1, 1}}

A struct that contains the environment of the VUMPS algorithm for calculate observables.
    
For a `Ni` x `Nj` unitcell, each is a Matrix, containing

- `AC`: The mixed canonical environment tensor.
- `AR`: The right canonical environment tensor.
- `Lu`: The left upper environment tensor.
- `Ru`: The right upper environment tensor.
- `Lo`: The left mixed environment tensor.
- `Ro`: The right mixed environment tensor.
"""
struct VUMPSEnv{T<:Number,
                ET<:AbstractArray{T, 3}}
    ACu::Matrix{ET}
    ARu::Matrix{ET}
    ACd::Matrix{ET}
    ARd::Matrix{ET}
    FLu::Matrix{ET}
    FRu::Matrix{ET}
    FLo::Matrix{ET}
    FRo::Matrix{ET}
    function VUMPSEnv(ACu::Matrix{ET},
                      ARu::Matrix{ET},
                      ACd::Matrix{ET},
                      ARd::Matrix{ET},
                      FLu::Matrix{ET},
                      FRu::Matrix{ET},
                      FLo::Matrix{ET},
                      FRo::Matrix{ET}) where {ET}
        T = eltype(ACu[1])
        new{T, ET}(ACu, ARu, ACd, ARd, FLu, FRu, FLo, FRo)
    end
end

"""
    VUMPSRuntime{T<:Number, S<:IndexSpace,
                 OT<:AbstractArray{S, 2, 2},
                 ET<:AbstractArray{S, 2, 1},
                 CT<:AbstractArray{S, 1, 1}}

A struct that contains the environment of the VUMPS algorithm for runtime calculations.
    
For a `Ni` x `Nj` unitcell, each is a Matrix, containing

- `O`: The center transfer matrix PEPO tensor.
- `AL`: The left canonical environment tensor.
- `AR`: The right canonical environment tensor.
- `C`: The canonical environment tensor.
- `L`: The left environment tensor.
- `R`: The right environment tensor.
"""
struct VUMPSRuntime{T<:Number, 
                    ET<:AbstractArray{T, 3},
                    CT<:AbstractArray{T, 2}}
    AL::Matrix{ET}
    AR::Matrix{ET}
    C::Matrix{CT}
    FL::Matrix{ET}
    FR::Matrix{ET}
    function VUMPSRuntime(AL::Matrix{ET},
                          AR::Matrix{ET},
                          C::Matrix{CT},
                          FL::Matrix{ET},
                          FR::Matrix{ET}) where {ET, CT}
        T = eltype(AL[1])
        new{T, ET, CT}(AL, AR, C, FL, FR)
    end
end

# In-place update of environment
function update!(env::VUMPSRuntime, env´::VUMPSRuntime) 
    env.AL .= env´.AL
    env.AR .= env´.AR
    env.C .= env´.C
    env.FL .= env´.FL
    env.FR .= env´.FR
    return env
end

function update!(env::Tuple{VUMPSRuntime, VUMPSRuntime}, env´::Tuple{VUMPSRuntime, VUMPSRuntime}) 
    update!(env[1], env´[1]) 
    update!(env[2], env´[2])
    return env
end

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
    Q .= Q * Diagonal(phases)
    R .= Diagonal(conj.(phases)) * R
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
    Q .= Diagonal(phases) * Q
    L .= L * Diagonal(conj!(phases))
    return L, Q
end

function env_norm(F::Matrix)
    Ni,Nj = size(F)
    buf = Zygote.Buffer(F)
    @inbounds @views for j in 1:Nj, i in 1:Ni
        buf[i,j] = F[i,j]/norm(F[i,j])
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
    Ni, Nj = size(A)
    χ = size(A[1], 1)
    atype = _arraytype(A[1])
    return [atype{ComplexF64}(I, χ, χ) for i = 1:Ni, j = 1:Nj]
end

function ρmap(ρ,A)
    Ni, Nj = size(A)
    ρ = copy(ρ)
    @inbounds @views for j in 1:Nj, i in 1:Ni
        jr = mod1(j + 1, Nj)
        ρ[i,jr] = ein"(dc,csb),dsa -> ab"(ρ[i,j], A[i,j], conj(A[i,j]))
    end
    return ρ
end

function initial_A(M, χ)
    Ni, Nj = size(M)
    atype = _arraytype(M[1])
    return [(D = size(M[i,j], 4); atype(rand(ComplexF64, χ,D,χ))) for i = 1:Ni, j = 1:Nj]
end

"""
    getL!(A,L; kwargs...)

````
┌─ Aᵢⱼ ─ Aᵢⱼ₊₁─     ┌─      L ─
ρᵢⱼ │      │     =  ρᵢⱼ  =  │
└─ Aᵢⱼ─  Aᵢⱼ₊₁─     └─      L'─
````

ρ=L'*L, return L, where `L`is guaranteed to have positive diagonal elements.
L = cholesky!(ρ).U
If ρ is not exactly positive definite, cholesky will fail
"""
function getL!(A, L; kwargs...)
    Ni,Nj = size(A)
    λs, ρs, info = eigsolve(ρ->ρmap(ρ,A), L, 1, :LM; ishermitian = false, maxiter = 1, kwargs...)
    info.converged == 0 && @warn "getL not converged"
    _, ρs1 = selectpos(λs, ρs, Nj)
    @inbounds for j = 1:Nj, i = 1:Ni
        ρ = ρs1[i,j] + ρs1[i,j]'
        ρ ./= tr(ρ)
        F = svd!(ρ)
        Lo = Diagonal(sqrt.(F.S)) * F.Vt
        _, R = qrpos!(Lo)
        L[i,j] = R
    end
    return L
end

"""
    getAL(A,L)

Given an MPS tensor `A` and `L` ，return a left-canonical MPS tensor `AL`, a gauge transform `R` and
a scalar factor `λ` such that ``λ AR R = L A``
"""
function getAL(A, L)
    Ni,Nj = size(A)
    AL = similar(A)
    Le = similar(L)
    λ = zeros(Ni,Nj)
    @inbounds @views for j = 1:Nj, i = 1:Ni
        χ, D, _ = size(A[i,j])
        Q, R = qrpos!(reshape(L[i,j]*reshape(A[i,j], χ, D*χ), D*χ, χ))
        AL[i,j] = reshape(Q, χ, D, χ)
        λ[i,j] = norm(R)
        Le[i,j] = rmul!(R, 1/λ[i,j])
    end
    return AL, Le, λ
end

function getLsped(Le, A, AL; kwargs...)
    Ni,Nj = size(A)
    L = similar(Le)
    @inbounds @views for j = 1:Nj, i = 1:Ni
        λs, Ls, info = eigsolve(X -> ein"(dc,csb),dsa -> ab"(X,A[i,j],conj(AL[i,j])), Le[i,j], 1, :LM; ishermitian = false, kwargs...)
        @debug "getLsped eigsolve" λs info sort(abs.(λs))
        info.converged == 0 && @warn "getLsped not converged"
        _, Ls1 = selectpos(λs, Ls, Nj)
        _, R = qrpos!(Ls1)
        L[i,j] = R
    end
    return L
end

"""
    left_canonical(A,L=cellones(size(A,1),size(A,2),size(A[1,1],1)); tol = 1e-12, maxiter = 100, kwargs...)

Given an MPS tensor `A`, return a left-canonical MPS tensor `AL`, a gauge transform `L` and
a scalar factor `λ` such that ``λ AL L = L A``, where an initial guess for `L` can be
provided.
"""
function left_canonical(A,L=cellones(A); tol = 1e-12, maxiter = 100, kwargs...)
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
    right_canonical(A,L=cellones(size(A,1),size(A,2),size(A[1,1],1)); tol = 1e-12, maxiter = 100, kwargs...)

Given an MPS tensor `A`, return a gauge transform R, a right-canonical MPS tensor `AR`, and
a scalar factor `λ` such that ``λ R AR^s = A^s R``, where an initial guess for `R` can be
provided.
"""
function right_canonical(A,L=cellones(A); tol = 1e-12, maxiter = 100, kwargs...)
    Ni,Nj = size(A)
    Ar = similar(A)
    Lr = similar(L)
    @inbounds for j = 1:Nj, i = 1:Ni
        Ar[i,j] = permutedims(A[i,j],(3,2,1))
        Lr[i,j] = permutedims(L[i,j],(2,1))
    end
    AL, L, λ = left_canonical(Ar,Lr; tol = tol, maxiter = maxiter, kwargs...)
    R  = similar(L)
    AR = similar(AL)
    @inbounds for j = 1:Nj, i = 1:Ni
         R[i,j] = permutedims( L[i,j],(2,1))
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
    Rijr = circshift(R, (0,-1))
    return [L * R for (L, R) in zip(L, Rijr)]
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
    FLm = [ein"((adf,fgh),dgeb),abc -> ceh"(FL, ALd, M, ALu) for (FL,ALd,M,ALu) in zip(FLi,ALdir,Mi,ALui)]
    return circshift(FLm, 1)
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
    FRm = [ein"((abc,ceh),dgeb),fgh -> adf"(ARu, FR, M, ARd) for (ARu,FR,M,ARd) in zip(ARui,FRi,Mi,ARdir)]
    circshift(FRm, -1)
end

function FLint(AL, M)
    Ni, Nj = size(AL)
    χ = size(AL[1], 1)
    atype = _arraytype(AL[1])
    return [(D = size(M[i, j], 1); atype(rand(ComplexF64, χ, D, χ))) for i = 1:Ni, j = 1:Nj]
end

function FRint(AR, M)
    Ni, Nj = size(AR)
    χ = size(AR[1], 1)
    atype = _arraytype(AR[1])
    return [(D = size(M[i, j], 3); atype(rand(ComplexF64, χ, D, χ))) for i = 1:Ni, j = 1:Nj]
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
function leftenv(ALu, ALd, M, FL=FLint(ALu,M); ifobs=false, verbosity=Defaults.verbosity, kwargs...) 
    Ni, Nj = size(M)
    λL = Zygote.Buffer(zeros(ComplexF64, Ni))
    FL′ = Zygote.Buffer(FL)
    for i in 1:Ni
        ir = ifobs ? Ni+1-i : mod1(i+1, Ni)
        λLs, FL1s, info = eigsolve(X->FLmap(ALu[i,:], ALd[ir,:], M[i,:], X), FL[i,:], 1, :LM; 
                                   alg_rrule=KrylovKit.GMRES(), maxiter=100, ishermitian = false, kwargs...)
        verbosity >= 1 && info.converged == 0 && @warn "leftenv not converged"
        λL[i], FL′[i,:] = selectpos(λLs, FL1s, Nj)
    end
    return copy(λL), copy(FL′)
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
function rightenv(ARu, ARd, M, FR=FRint(ARu,M); ifobs=false, verbosity=Defaults.verbosity, kwargs...) 
    Ni,Nj = size(M)
    λR = Zygote.Buffer(zeros(ComplexF64, Ni))
    FR′ = Zygote.Buffer(FR)
    for i in 1:Ni
        ir = ifobs ? Ni+1-i : mod1(i+1, Ni)
        λRs, FR1s, info= eigsolve(X->FRmap(ARu[i,:], ARd[ir,:], M[i,:], X), FR[i,:], 1, :LM;
                                  alg_rrule=KrylovKit.GMRES(), maxiter=100, ishermitian = false, kwargs...)
        verbosity >= 1 && info.converged == 0 && @warn "rightenv not converged"
        λR[i], FR′[i,:] = selectpos(λRs, FR1s, Nj)
    end
    return copy(λR), copy(FR′)
end

"""
    Rm = Rmap(FRi::Vector{<:AbstractTensorMap}, 
                ARui::Vector{<:AbstractTensorMap}, 
                ARdir::Vector{<:AbstractTensorMap}, 
                )

```
    ── ARuᵢⱼ  ──┐          ──┐           a──────┬──────c    
        │       Rᵢⱼ  =       Rᵢⱼ₋₁       │      │      │ 
    ── ARdᵢᵣⱼ ──┘          ──┘           │      b      │    
                                         │      │      │      
                                         d──────┴──────e   
```
"""
function Rmap(Ri, ARui, ARdir)
    Rm = [ein"(abc,ce),dbe->ad"(ARu, R, ARd) for (R, ARu, ARd) in zip(Ri, ARui, ARdir)]
    return circshift(Rm, -1)
end

"""
    λR, FR = rightCenv(ARu::Matrix{<:AbstractTensorMap}, 
                       ARd::Matrix{<:AbstractTensorMap}, 
                       R::Matrix{<:AbstractTensorMap} = initial_C(ARu); 
                       kwargs...) 

Compute the right environment tensor for MPS A by finding the left fixed point
of AR - conj(AR) contracted along the physical dimension.
```
    ── ARuᵢⱼ  ──┐          ──┐   
        |       Rᵢⱼ  = λRᵢⱼ  Rᵢⱼ₋₁
    ── ARdᵢᵣⱼ ──┘          ──┘  
```
"""
function rightCenv(ARu, ARd, R=cellones(ARu); 
                   ifobs=false, verbosity=Defaults.verbosity, kwargs...) 

    Ni, Nj = size(ARu)
    λR = Zygote.Buffer(zeros(eltype(ARu[1]), Ni))
    R′ = Zygote.Buffer(R)
    for i in 1:Ni
        ir = ifobs ? mod1(Ni - i + 2, Ni) : i
        λRs, R1s, info = eigsolve(R -> Rmap(R, ARu[i,:], ARd[ir,:]), R[i,:], 1, :LM; 
                                  alg_rrule=KrylovKit.GMRES(), maxiter=100, ishermitian = false, kwargs...)
        verbosity >= 1 && info.converged == 0 && @warn "rightenv not converged"
        λR[i], R′[i,:] = selectpos(λRs, R1s, Nj)
    end
    return copy(λR), copy(R′)
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
    ACm = [ein"((abc,ceh),dgeb),adf -> fgh"(AC,FR,M,FL) for (AC,FR,M,FL) in zip(ACj,FRj,Mj,FLj)]
    return circshift(ACm, 1)
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
    Cm = [ein"(ab,bce),acd -> de"(C,FR,FL) for (C,FR,FL) in zip(Cj,FRj,FLjr)]
    return circshift(Cm, 1)
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
function ACenv(AC, FL, M, FR; verbosity=Defaults.verbosity, kwargs...)
    Ni, Nj = size(M)
    λAC = Zygote.Buffer(zeros(ComplexF64, Nj))
    AC′ = Zygote.Buffer(AC)
    for j in 1:Nj
        λACs, ACs, info = eigsolve(X->ACmap(X, FL[:,j], FR[:,j], M[:,j]), AC[:,j], 1, :LM; 
                                   alg_rrule=KrylovKit.GMRES(), maxiter=100, ishermitian = false, kwargs...)
        verbosity >= 1 && info.converged == 0 && @warn "ACenv Not converged"
        λAC[j], AC′[:,j] = selectpos(λACs, ACs, Ni)
    end
    return copy(λAC), copy(AC′)
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
function Cenv(C, FL, FR; verbosity=Defaults.verbosity, kwargs...)
    Ni, Nj = size(C)
    λC = Zygote.Buffer(zeros(ComplexF64, Nj))
    C′ = Zygote.Buffer(C)
    for j in 1:Nj
        jr = mod1(j + 1, Nj)
        λCs, Cs, info = eigsolve(X->Cmap(X, FL[:,jr], FR[:,j]), C[:,j], 1, :LM; 
                                 alg_rrule=KrylovKit.GMRES(), maxiter=100, ishermitian = false, kwargs...)
        verbosity >= 1 && info.converged == 0 && @warn "Cenv Not converged"
        λC[j], C′[:,j] = selectpos(λCs, Cs, Ni)
    end
    return copy(λC), copy(C′)
end

function ACCtoAL(AC, C)
    Ni, Nj = size(AC)
    errL = 0.0
    AL = Zygote.Buffer(AC)
    @inbounds for j in 1:Nj, i in 1:Ni
        χ, D = size(AC[i,j])[[1,2]]
        QAC, RAC = qrpos(reshape(AC[i,j],(χ*D, χ)))
         QC, RC  = qrpos(C[i,j])
        errL += norm(RAC-RC)
        AL[i,j] = reshape(QAC*QC', (χ, D, χ))
    end
    return copy(AL), errL
end

function ACCtoAR(AC, C)
    Ni, Nj = size(AC)
    errR = 0.0
    AR = Zygote.Buffer(AC)
    @inbounds for j in 1:Nj, i in 1:Ni
        χ, D = size(AC[i,j])[[1,2]]
        jr = mod1(j - 1, Nj)
        LAC, QAC = lqpos(reshape(AC[i,j],(χ, D*χ)))
         LC, QC  = lqpos(C[i,jr])
        errR += norm(LAC-LC)
        AR[i,j] = reshape(QC'*QAC, (χ, D, χ))
    end
    return copy(AR), errR
end

function ALCtoAC(AL,C)
    return [ein"asc,cb -> asb"(AL, C) for (AL, C) in zip(AL, C)]
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