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
struct VUMPSEnv
    ACu::StructArray
    ARu::StructArray
    ACd::StructArray
    ARd::StructArray
    FLu::StructArray
    FRu::StructArray
    FLo::StructArray
    FRo::StructArray
end

"""
    VUMPSRuntime{T<:Number, S<:IndexSpace,
                 OT<:AbstractArray{S, 2, 2},
                 ET<:AbstractArray{S, 2, 1},
                 CT<:AbstractArray{S, 1, 1}}

A struct that contains the environment of the VUMPS algorithm for runtime calculations.
    
For a `Ni` x `Nj` unitcell, each is a Matrix, containing

- `AL`: The left canonical environment tensor.
- `AR`: The right canonical environment tensor.
- `C`: The canonical environment tensor.
- `L`: The left environment tensor.
- `R`: The right environment tensor.
"""
struct VUMPSRuntime             
    AL::StructArray
    AR::StructArray
    C::StructArray
    FL::StructArray
    FR::StructArray
end

# In-place update of environment
function update!(env::VUMPSRuntime, env´::VUMPSRuntime) 
    env.AL.data .= env´.AL.data
    env.AR.data .= env´.AR.data
    env.C.data .= env´.C.data
    env.FL.data .= env´.FL.data
    env.FR.data .= env´.FR.data
    return env
end

function update!(env::Tuple{VUMPSRuntime, VUMPSRuntime}, env´::Tuple{VUMPSRuntime, VUMPSRuntime}) 
    update!(env[1], env´[1]) 
    update!(env[2], env´[2])
    return env
end

function update!(env::VUMPSRuntime, env´::Tuple{VUMPSRuntime, VUMPSRuntime}) 
    update!(env, env´[1])
    return env
end

Array(rt::VUMPSRuntime) = VUMPSRuntime(Array(rt.AL), Array(rt.AR), Array(rt.C), Array(rt.FL), Array(rt.FR))
Array(rt::Tuple{VUMPSRuntime, VUMPSRuntime}) = Array.(rt)
CuArray(rt::VUMPSRuntime) = VUMPSRuntime(CuArray(rt.AL), CuArray(rt.AR), CuArray(rt.C), CuArray(rt.FL), CuArray(rt.FR))
CuArray(rt::Tuple{VUMPSRuntime, VUMPSRuntime}) = CuArray.(rt)
ROCArray(rt::VUMPSRuntime) = VUMPSRuntime(ROCArray(rt.AL), ROCArray(rt.AR), ROCArray(rt.C), ROCArray(rt.FL), ROCArray(rt.FR))
ROCArray(rt::Tuple{VUMPSRuntime, VUMPSRuntime}) = ROCArray.(rt)

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

"""
    λs[1], Fs[1] = selectpos(λs, Fs)

Select the max positive one of λs and corresponding Fs.
"""
function selectpos(λs, Fs, N)
    if length(λs) > 1 && norm(abs(λs[1]) - abs(λs[2])) < 1e-12
        # @show "selectpos: λs are degeneracy"
        N = max(N, length(λs))
        p = argmax(real(λs[1:N]))  
        # @show λs p abs.(λs)
        return λs[1:N][p], Fs[1:N][p]
    else
        return λs[1], Fs[1]
    end
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
    @inbounds for j = 1:Nj, i = 1:Ni
        # _, ρs, info = eigsolve(ρ->ρmap(ρ,A[i,:],j), L[i,j]'*L[i,j], 1, :LM; ishermitian = false, maxiter = 1, kwargs...)
        # info.converged == 0 && @warn "getL not converged"
        # ρ = real(ρs[1] + ρs[1]')
        _, ρ = simple_eig(ρ->ρmap(ρ,A[i,:],j), L[i,j]'*L[i,j]; kwargs...)
        ρ = real(ρ + ρ') 
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
    AL = similar(A)
    Le = similar(L)
    λ = randSA(Array, AL.pattern)
    for i in 1:length(A)
        Q, R = qrpos!(_to_tail(L[i]*_to_front(A[i])))
        AL[i] = reshape(Q, size(A[i]))
        λ[i] = norm(R)
        Le[i] = rmul!(R, 1/λ[i])
    end
    
    return AL, Le, λ
end

function getLsped(Le, A, AL; kwargs...)
    L = similar(Le)
    for i in 1:length(A)
        _, Ls1 = simple_eig(X -> ρmap(X,A[i],conj(AL[i])), Le[i]; power_iter=5, kwargs...)
        _, R = qrpos!(Ls1)
        L[i] = R
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
    # L = getL!(A,L; kwargs...) # seems not necessary
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
function right_canonical(A, L=cellones(A); tol = 1e-12, maxiter = 100, kwargs...)
    Ar = similar(A)
    Lr = similar(L)
    @inbounds for i in 1:length(A)
        Ar[i] = permute_fronttail(A[i])
        Lr[i] = permutedims(L[i],(2,1))
    end

    AL, L, λ = left_canonical(Ar,Lr; tol = tol, maxiter = maxiter, kwargs...)
    R  = similar(L)
    AR = similar(AL)
    @inbounds for i in 1:length(AL)
        R[i] = permutedims(L[i],(2,1))
        AR[i] = permute_fronttail(AL[i])
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
    C = similar(L)
    @inbounds for i in 1:length(L)
        C[i] = L[i] * Rijr[i]
    end
    return C
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
function leftenv(ALu, ALd, M, FL=FLint(ALu,M); ifobs=false, ifvalue=false, alg, kwargs...) 
    λL = Zygote.Buffer(randSA(Array, M.pattern))
    FL′ = Zygote.Buffer(FL)
    Ni, Nj = size(M)
    processed_indices = Set{Int}()
    power_iter = ifobs ? alg.power_iter_obs : alg.power_iter
    forloop_iter = alg.forloop_iter
    ifcheckpoint = alg.ifcheckpoint
    for i in 1:Ni
        ir = ifobs ? Ni + 1 - i : mod1(i + 1, Ni)
        p = FL.pattern[i,1]
        if p ∉ processed_indices
            if alg.ifsimple_eig
                if ifcheckpoint
                    λL[i,1], FL′[i,1] = checkpoint(simple_eig, FLij -> FLmap(1, FLij, ALu[i,:], ALd[ir,:], M[i, :]; ifcheckpoint, forloop_iter), FL[i,1]; ifvalue, power_iter)
                else
                    λL[i,1], FL′[i,1] = simple_eig(FLij -> FLmap(1, FLij, ALu[i,:], ALd[ir,:], M[i, :]; forloop_iter), FL[i,1]; ifvalue, power_iter)
                end
            else
                λLs, FLi1s, info = eigsolve(FLij -> FLmap(1, FLij, ALu[i,:], ALd[ir,:], M[i, :]; 
                                                          ifcheckpoint, forloop_iter), 
                                            FL[i,1], 1, :LM; alg_rrule=GMRES(verbosity=-1), maxiter=100, ishermitian=false, kwargs...)
                alg.verbosity >= 1 && info.converged == 0 && @warn "leftenv not converged"
                λL[i,1], FL′[i,1] = selectpos(λLs, FLi1s, Nj)
            end
            push!(processed_indices, p)
            if length(processed_indices) == length(FL.data)
                break
            end
        end
        for j in 2:Nj
            p = FL.pattern[i,j]
            if p ∉ processed_indices
                FL′[i,j] = FLmap_forloop(FL′[i,j-1], ALu[i,j-1], ALd[ir,j-1],  M[i,j-1]; forloop_iter)
                λL[i,j] = λL[i,1]
                push!(processed_indices, p)
                if length(processed_indices) == length(FL.data)
                    break
                end
            end
        end
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
function rightenv(ARu, ARd, M, FR=FRint(ARu,M); ifobs=false, ifvalue=false, alg, kwargs...) 
    Ni,Nj = size(M)
    λR = Zygote.Buffer(randSA(Array, M.pattern))
    FR′ = Zygote.Buffer(FR)
    processed_indices = Set{Int}()
    power_iter = ifobs ? alg.power_iter_obs : alg.power_iter
    forloop_iter = alg.forloop_iter
    ifcheckpoint = alg.ifcheckpoint
    for i in 1:Ni
        ir = ifobs ? Ni + 1 - i : mod1(i + 1, Ni)
        p = FR.pattern[i,Nj]
        if p ∉ processed_indices
            if alg.ifsimple_eig
                if ifcheckpoint
                    λR[i,Nj], FR′[i,Nj] = checkpoint(simple_eig, FRiNj -> FRmap(Nj, FRiNj, ARu[i,:], ARd[ir,:], M[i,:]; ifcheckpoint, forloop_iter), FR[i,Nj]; ifvalue, power_iter)
                else
                    λR[i,Nj], FR′[i,Nj] = simple_eig(FRiNj -> FRmap(Nj, FRiNj, ARu[i,:], ARd[ir,:], M[i,:]; forloop_iter), FR[i,Nj]; ifvalue, power_iter)
                end
            else
                λRs, FR1s, info = eigsolve(FRiNj -> FRmap(Nj, FRiNj, ARu[i,:], ARd[ir,:], M[i,:]; 
                                                          ifcheckpoint, forloop_iter), 
                                        FR[i,Nj], 1, :LM; alg_rrule=GMRES(verbosity=-1), maxiter=100, ishermitian = false, kwargs...)
                alg.verbosity >= 1 && info.converged == 0 && @warn "rightenv not converged"
                λR[i,Nj], FR′[i,Nj] = selectpos(λRs, FR1s, Nj)
            end
            push!(processed_indices, p)
            if length(processed_indices) == length(FR.data)
                break
            end
        end
        for j in Nj-1:-1:1
            p = FR.pattern[i,j]
            if p ∉ processed_indices
                FR′[i,j] = FRmap_forloop(FR′[i,j+1], ARu[i,j+1], ARd[ir,j+1], M[i,j+1]; forloop_iter)
                λR[i,j] = λR[i,Nj]
                push!(processed_indices, p)
                if length(processed_indices) == length(FR.data)
                    break
                end
            end
        end
    end
    return copy(λR), copy(FR′)
end


"""
        leftCenv(ALu::Matrix{<:AbstractTensorMap}, 
                    ALd::Matrix{<:AbstractTensorMap}, 
                    L::Matrix{<:AbstractTensorMap} = cellones(ALu); 
                    ifobs=false, verbosity = Defaults.verbosity, kwargs...) 

Compute the left environment tensor for MPS A, by finding the left fixed point
of ALu - ALd contracted along the physical dimension.
```
   ┌── ALuᵢⱼ  ──          ┌──  
   Lᵢⱼ   |        = λLᵢⱼ  Lᵢⱼ₊₁
   └── ALdᵢᵣⱼ ──          └──  
```
"""
function leftCenv(ALu::StructArray, 
                  ALd::StructArray, 
                  L::StructArray = cellones(ALu); 
                  ifobs=false, ifvalue=false, alg, kwargs...) 

    Ni, Nj = size(L)
    λL = Zygote.Buffer(randSA(Array, ALu.pattern))
    L′ = Zygote.Buffer(L)
    power_iter = ifobs ? alg.power_iter_obs : alg.power_iter
    processed_indices = Set{Int}()
    for i in 1:Ni
        ir = ifobs ? Ni + 2 - i : i
        p = L.pattern[i,1]
        if p ∉ processed_indices
            f(Lij) = Lmap(1, Lij, ALu[i,:], ALd[ir,:])
            if alg.ifsimple_eig
                if alg.ifcheckpoint
                    λL[i,1], L′[i,1] = checkpoint(simple_eig, f, L[i,1]; ifvalue, power_iter)
                else
                    λL[i,1], L′[i,1] = simple_eig(f, L[i,1]; ifvalue, power_iter)
                end
            else
                λLs, Li1s, info = eigsolve(f, L[i,1], 1, :LM; maxiter=100, ishermitian = false, kwargs...)
                alg.verbosity >= 1 && info.converged == 0 && @warn "leftenv not converged"
                λL[i,1], L′[i,1] = selectpos(λLs, Li1s, Nj)
            end
            push!(processed_indices, p)
            if length(processed_indices) == length(L.data)
                break
            end
        end
        for j in 2:Nj
            p = L.pattern[i,j]
            if p ∉ processed_indices
                Lij = Lmap(L′[i,j-1], ALu[i,j-1], ALd[ir,j-1])
                L′[i,j] = Lij / norm(Lij)
                λL[i,j] = λL[i,1]
                push!(processed_indices, p)
                if length(processed_indices) == length(L.data)
                    break
                end
            end
        end
    end

    return copy(λL), copy(L′)
end

"""
        rightCenv(ARu::Matrix{<:AbstractTensorMap}, 
                    ARd::Matrix{<:AbstractTensorMap}, 
                    L::Matrix{<:AbstractTensorMap} = cellones(ARu); 
                    ifobs=false, verbosity = Defaults.verbosity, kwargs...) 

Compute the left environment tensor for MPS A, by finding the left fixed point
of ARu - ARd contracted along the physical dimension.
```
    ── ARuᵢⱼ  ──┐          ──┐    
        │       Rᵢⱼ  =       Rᵢⱼ₋₁ 
    ── ARdᵢᵣⱼ ──┘          ──┘     
```
"""
function rightCenv(ARu::StructArray, 
                  ARd::StructArray, 
                  R::StructArray = cellones(ARu); 
                  ifobs=false, ifvalue=false, alg, kwargs...) 

    λR = Zygote.Buffer(randSA(Array, ARu.pattern))
    R′ = Zygote.Buffer(R)
    power_iter = ifobs ? alg.power_iter_obs : alg.power_iter
    processed_indices = Set{Int}()
    for i in 1:Ni
        ir = ifobs ? Ni + 2 - i : i
        p = R.pattern[i,Nj]
        if p ∉ processed_indices
            f(RiNj) = Rmap(Ni, RiNj, ARu[i,:], ARd[ir,:])
            if alg.ifsimple_eig
                if alg.ifcheckpoint
                    λR[i,Nj], R′[i,Nj] = checkpoint(simple_eig, f, R[i,Nj]; ifvalue, power_iter)
                else
                    λR[i,Nj], R′[i,Nj] = simple_eig(f, R[i,Nj]; ifvalue, power_iter)
                end
            else
                λLs, Li1s, info = eigsolve(f, R[i,Nj], 1, :LM; maxiter=100, ishermitian = false, kwargs...)
                alg.verbosity >= Nj && info.converged == 0 && @warn "leftenv not converged"
                λR[i,Nj], R′[i,Nj] = selectpos(λLs, Li1s, Nj)
            end
            push!(processed_indices, p)
            if length(processed_indices) == length(R.data)
                break
            end
        end
        for j in Nj-1:-1:1
            p = R.pattern[i,j]
            if p ∉ processed_indices
                Rij = Rmap(R′[i,j+1], ARu[i,j+1], ARd[ir,j+1])
                R′[i,j] = Rij / norm(Rij)
                λR[i,j] = λR[i,Nj]
                push!(processed_indices, p)
                if length(processed_indices) == length(R.data)
                    break
                end
            end
        end
    end

    return copy(λR), copy(R′)
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
function ACenv(AC, FL, M, FR; ifvalue=false, alg, kwargs...)
    Ni, Nj = size(M)
    λAC = Zygote.Buffer(randSA(Array, M.pattern))
    AC′ = Zygote.Buffer(AC)
    processed_indices = Set{Int}()
    power_iter = alg.power_iter
    forloop_iter = alg.forloop_iter
    ifcheckpoint = alg.ifcheckpoint
    for j in 1:Nj
        p = AC.pattern[1,j]
        if p ∉ processed_indices
            if alg.ifsimple_eig
                if ifcheckpoint
                    λAC[1,j], AC′[1,j] = checkpoint(simple_eig, AC1j -> ACmap(1, AC1j, FL[:,j], FR[:,j], M[:,j]; ifcheckpoint, forloop_iter), AC[1,j]; ifvalue, power_iter)
                else
                    λAC[1,j], AC′[1,j] = simple_eig(AC1j -> ACmap(1, AC1j, FL[:,j], FR[:,j], M[:,j]; forloop_iter), AC[1,j]; ifvalue, power_iter)
                end
            else
                λACs, ACs, info = eigsolve(AC1j -> ACmap(1, AC1j, FL[:,j], FR[:,j], M[:,j]; 
                                                         ifcheckpoint, forloop_iter), 
                                        AC[1,j], 1, :LM; alg_rrule=GMRES(verbosity=-1), maxiter=100, ishermitian = false, kwargs...)
                alg.verbosity >= 1 && info.converged == 0 && @warn "ACenv Not converged"
                λAC[1,j], AC′[1,j] = selectpos(λACs, ACs, Ni)
            end
            push!(processed_indices, p)
            if length(processed_indices) == length(AC.data)
                break
            end
        end
        for i in 2:Ni
            p = AC.pattern[i,j]
            if p ∉ processed_indices
                ACij = ACmap_forloop(AC′[i-1,j], FL[i-1,j], FR[i-1,j], M[i-1,j]; forloop_iter)
                AC′[i,j] = ACij/norm(ACij)
                λAC[i,j] = λAC[1,j]
                push!(processed_indices, p)
                if length(processed_indices) == length(AC.data)
                    break
                end
            end
        end
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
function Cenv(C, FL, FR; alg, ifvalue=false, kwargs...)
    Ni, Nj = size(C)
    λC = Zygote.Buffer(randSA(Array, C.pattern))
    C′ = Zygote.Buffer(C)
    processed_indices = Set{Int}()
    power_iter = alg.power_iter
    ifcheckpoint = alg.ifcheckpoint
    for j in 1:Nj
        jr = mod1(j + 1, Nj)
        p = C.pattern[1,j]
        if p ∉ processed_indices
            if alg.ifsimple_eig
                if ifcheckpoint
                    λC[1,j], C′[1,j] = checkpoint(simple_eig, C1j -> Cmap(1, C1j, FL[:,jr], FR[:,j]), C[1,j]; ifvalue, power_iter)
                else
                    λC[1,j], C′[1,j] = simple_eig(C1j -> Cmap(1, C1j, FL[:,jr], FR[:,j]), C[1,j]; ifvalue, power_iter)
                end
            else
                λCs, Cs, info = eigsolve(C1j -> Cmap(1, C1j, FL[:,jr], FR[:,j]), 
                                        C[1,j], 1, :LM; alg_rrule=GMRES(verbosity=-1), maxiter=100, ishermitian = false, kwargs...)
                alg.verbosity >= 1 && info.converged == 0 && @warn "Cenv Not converged"
                λC[1,j], C′[1,j] = selectpos(λCs, Cs, Ni)
            end
            push!(processed_indices, p)
            if length(processed_indices) == length(C.data)
                break
            end
        end
        for i in 2:Ni
            p = C.pattern[i,j]
            if p ∉ processed_indices
                Cij = Cmap(C′[i-1,j], FL[i-1,jr], FR[i-1,j])
                C′[i,j] = Cij/norm(Cij)
                λC[i,j] = λC[1,j]
                push!(processed_indices, p)
                if length(processed_indices) == length(C.data)
                    break
                end
            end
        end
    end
    return copy(λC), copy(C′)
end

function ACCtoAL(AC, C)
    errL = 0.0
    AL = Zygote.Buffer(AC)
    @inbounds for i in 1:length(AC)
        QAC, RAC = qrpos(_to_tail(AC[i]))
         QC, RC  = qrpos(C[i])
        errL += norm(RAC-RC)
        AL[i] = reshape(QAC*QC', size(AC[i]))
    end
    return copy(AL), errL
end

function ACCtoAR(AC, C)
    errR = 0.0
    AR = Zygote.Buffer(AC)
    Nj = size(AC, 2)
    @inbounds for p in 1:length(AC.data)
        i, j = Tuple(findfirst(==(p), AC.pattern))
        jr = mod1(j - 1, Nj)
        LAC, QAC = lqpos(_to_front(AC[i,j]))
         LC, QC  = lqpos(C[i,jr])
        errR += norm(LAC-LC)
        AR[i,j] = reshape(QC'*QAC, size(AC[i,j]))
    end
    return copy(AR), errR
end

function ALCtoAC(AL::leg3, C)
    AC = Zygote.Buffer(AL)
    @inbounds for i in 1:length(AL)
        AC[i] = ein"asc,cb -> asb"(AL[i], C[i])
    end
    return copy(AC)
end

function ALCtoAC(AL::leg4, C)
    AC = Zygote.Buffer(AL)
    @inbounds for i in 1:length(AL)
        AC[i] = ein"astc,cb -> astb"(AL[i], C[i])
    end
    return copy(AC)
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
    AL, errL = ACCtoAL(AC, C)
    AR, errR = ACCtoAR(AC, C)
    return AL, AR, errL, errR
end