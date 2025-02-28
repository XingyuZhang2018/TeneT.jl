"""
    VUMPSEnv

A struct representing the environment tensors used in the Variational Uniform Matrix Product State (VUMPS) algorithm for calculating physical observables.

For a unit cell of size `Ni` x `Nj`, each field stores a matrix containing specific environment tensors:

- `ACu`: Upper mixed canonical environment tensor
- `ARu`: Upper right canonical environment tensor  
- `ACd`: Lower mixed canonical environment tensor
- `ARd`: Lower right canonical environment tensor
- `FLu`: Upper left fixed-point environment tensor
- `FRu`: Upper right fixed-point environment tensor
- `FLo`: Lower left fixed-point environment tensor 
- `FRo`: Lower right fixed-point environment tensor

These tensors are stored in StructArrays for efficient memory layout and GPU compatibility.
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
    L = getL!(A::Matrix{<:AbstractTensorMap}, L::Matrix{<:AbstractTensorMap}; verbosity = Defaults.verbosity, kwargs...)

````
     ┌─ Aᵢⱼ ─ Aᵢⱼ₊₁─     ┌─      L ─
     ρᵢⱼ │      │     =  ρᵢⱼ  =  │
     └─ Aᵢⱼ─  Aᵢⱼ₊₁─     └─      L'─
````

ρ=L'*L, return L, where `L`is guaranteed to have positive diagonal elements.

"""
function getL!(A::StructArray, L::StructArray)
    Ni, Nj = size(A)
    processed_indices = Set{Int}()
    @inbounds for j = 1:Nj, i = 1:Ni
        p = A.pattern[i,j]
        if p ∉ processed_indices
            _, ρs = simple_eig(ρ -> ρmap(ρ, A[i,:], j), L[i,j]'*L[i,j])
            ρ = real(ρs + ρs')
            ρ /= tr(ρ)
            _, S, Vt = tsvd!(ρ)
            Lo = sqrt(S) * Vt
            _, R = leftorth!(Lo)
            L[i,j] = R
            push!(processed_indices, p)
            if length(processed_indices) == length(A.data)
                break
            end
        end
    end
    return L
end

"""
    AL, Le, λ = getAL(A::Matrix{<:AbstractTensorMap}, L::Matrix{<:AbstractTensorMap})

Given an MPS tensor `A` and `L` ，return a left-canonical MPS tensor `AL`, a gauge transform `R` and
a scalar factor `λ` such that ``λ * AL * Le = L * A``
"""
function getAL(A::StructArray, L::StructArray)
    Ni, Nj = size(A)
    AL = similar(A)
    Le = similar(L)
    λ = rand(eltype(A.data[1]), A.pattern)
    processed_indices = Set{Int}()
    @inbounds for j = 1:Nj, i = 1:Ni
        p = A.pattern[i,j]
        if p ∉ processed_indices
            Q, R = leftorth!(_to_front(L[i,j] * _to_tail(A[i,j])))
            AL[i,j] = Q
            λ[i,j] = norm(R)
            Le[i,j] = rmul!(R, 1/λ[i,j])
            push!(processed_indices, p)
            if length(processed_indices) == length(A.data)
                break
            end
        end
    end
    return AL, Le, λ
end

function getLsped(Le::StructArray, A::StructArray, AL::StructArray)
    Ni, Nj = size(A)
    L = similar(Le)
    processed_indices = Set{Int}()
    @inbounds for j = 1:Nj, i = 1:Ni
        p = Le.pattern[i,j]
        if p ∉ processed_indices
            _, Ls = simple_eig(X -> ρmap(X, A[i,j], adjoint(AL[i,j])), Le[i,j])
            _, R = leftorth!(Ls)
            L[i,j] = R
            push!(processed_indices, p)
            if length(processed_indices) == length(Le.data)
                break
            end
        end
    end
    return L
end

"""
    AL, L, λ = left_canonical(A::Matrix{<:AbstractTensorMap}, L::Matrix{<:AbstractTensorMap} = initial_C(A); kwargs...)

Given an MPS tensor `A`, return a left-canonical MPS tensor `AL`, a gauge transform `L` and
a scalar factor `λ` such that ``λ*AL*L = L*A``, where an initial guess for `L` can be
provided.
"""
function left_canonical(A::StructArray, L::StructArray = initial_C(A); tol = 1e-12, maxiter = 100, kwargs...)
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
    R, AR, λ = right_canonical(A::Matrix{<:AbstractTensorMap}, L::Matrix{<:AbstractTensorMap} = initial_C(A); tol = 1e-12, maxiter = 100, kwargs...)

Given an MPS tensor `A`, return a gauge transform R, a right-canonical MPS tensor `AR`, and
a scalar factor `λ` such that ``λ * R * AR = A * R``, where an initial guess for `R` can be
provided.
"""
function right_canonical(A::StructArray, L::StructArray = initial_C(A); tol = 1e-12, maxiter = 100, kwargs...)
    Ar = similar(A)
    Lr = similar(L)
    @inbounds for p in 1:length(A.data)
        i, j = Tuple(findfirst(==(p), A.pattern))
        Ar[i,j] = permute_fronttail(A[i,j])
        Lr[i,j] = permute(L[i,j], ((2,), (1,)))
    end
    
    AL, L, λ = left_canonical(Ar, Lr; tol, maxiter, kwargs...)

    R  = similar(L)
    AR = similar(AL)
    @inbounds for p in 1:length(AL.data)
        i, j = Tuple(findfirst(==(p), AL.pattern))
         R[i,j] = permute(L[i,j], ((2,), (1,)))
        AR[i,j] = permute_fronttail(AL[i,j])
    end
    return R, AR, λ
end


"""
    λL, FL = leftenv(ALu, ALd, O, FL = initial_FL(ALu,O); kwargs...)

Compute the left environment tensor for MPS A and MPO O, by finding the left fixed point
of ALu - O - ALd contracted along the physical dimension.
```
 ┌──  ALuᵢⱼ  ──          ┌── 
 │     │                 │   
FLᵢⱼ ─ Oᵢⱼ   ──   = λLᵢⱼ FLᵢⱼ₊₁   
 │     │                 │   
 └──  ALdᵢᵣⱼ  ─          └── 
```
"""
function leftenv(ALu::StructArray, 
                 ALd::StructArray, 
                 M::StructArray, 
                 FL::StructArray = initial_FL(ALu,M); 
                 ifobs=false, ifvalue=false, alg, kwargs...) 

    Ni, Nj = size(M)
    λL = Zygote.Buffer(rand(eltype(M.data[1]), M.pattern))
    FL′ = Zygote.Buffer(FL)
    processed_indices = Set{Int}()
    for i in 1:Ni
        ir = ifobs ? Ni + 1 - i : mod1(i + 1, Ni)
        p = FL.pattern[i,1]
        if p ∉ processed_indices
            f(FLij) = FLmap(1, FLij, ALu[i,:], ALd[ir,:], M[i,:])
            if alg.ifsimple_eig
                if alg.ifcheckpoint
                    λL[i,1], FL′[i,1] = checkpoint(simple_eig, f, FL[i,1]; ifvalue=ifvalue)
                else
                    λL[i,1], FL′[i,1] = simple_eig(f, FL[i,1]; ifvalue=ifvalue)
                end
            else
                λLs, FLi1s, info = eigsolve(f, FL[i,1], 1, :LM; maxiter=100, ishermitian = false, kwargs...)
                alg.verbosity >= 1 && info.converged == 0 && @warn "leftenv not converged"
                λL[i,1], FL′[i,1] = λLs[1], FLi1s[1]
            end
            push!(processed_indices, p)
            if length(processed_indices) == length(FL.data)
                break
            end
        end
        for j in 2:Nj
            p = FL.pattern[i,j]
            if p ∉ processed_indices
                FL′[i,j] = FLmap(FL′[i,j-1], ALu[i,j-1], ALd[ir,j-1], M[i,j-1])
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
function rightenv(ARu::StructArray, 
                  ARd::StructArray, 
                  M::StructArray, 
                  FR::StructArray=initial_FR(ARu,M);
                  ifobs=false, ifvalue=false, alg, kwargs...) 

    Ni,Nj = size(M)
    λR = Zygote.Buffer(rand(eltype(M.data[1]), M.pattern))
    FR′ = Zygote.Buffer(FR)
    processed_indices = Set{Int}()
    for i in 1:Ni
        ir = ifobs ? Ni + 1 - i : mod1(i + 1, Ni)
        p = FR.pattern[i,Nj]
        if p ∉ processed_indices
            f(FRiNj) = FRmap(Nj, FRiNj, ARu[i,:], ARd[ir,:], M[i,:])
            if alg.ifsimple_eig
                if alg.ifcheckpoint
                    λR[i,Nj], FR′[i,Nj] = checkpoint(simple_eig, f, FR[i,Nj]; ifvalue=ifvalue)
                else
                    λR[i,Nj], FR′[i,Nj] = simple_eig(f, FR[i,Nj]; ifvalue=ifvalue)
                end
            else
                λRs, FR1s, info = eigsolve(f, FR[i,Nj], 1, :LM; alg_rrule=GMRES(verbosity=-1), maxiter=100, ishermitian = false, kwargs...)
                alg.verbosity >= 1 && info.converged == 0 && @warn "rightenv not converged"
                λR[i,Nj], FR′[i,Nj] = λRs[1], FR1s[1]
            end
            push!(processed_indices, p)
            if length(processed_indices) == length(FR.data)
                break
            end
        end
        for j in Nj-1:-1:1
            p = FR.pattern[i,j]
            if p ∉ processed_indices
                FR′[i,j] = FRmap(FR′[i,j+1], ARu[i,j+1], ARd[ir,j+1], M[i,j+1])
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
function ACenv(AC::StructArray, 
               FL::StructArray, 
               FR::StructArray,
               M::StructArray; ifvalue=false, alg, kwargs...)

    Ni, Nj = size(M)
    λAC = Zygote.Buffer(rand(eltype(M.data[1]), M.pattern))
    AC′ = Zygote.Buffer(AC)
    processed_indices = Set{Int}()
    for j in 1:Nj
        p = AC.pattern[1,j]
        if p ∉ processed_indices
            f(AC1j) = ACmap(1, AC1j, FL[:,j], FR[:,j], M[:,j])
            if alg.ifsimple_eig
                if alg.ifcheckpoint
                    λAC[1,j], AC′[1,j] = checkpoint(simple_eig, f, AC[1,j]; ifvalue=ifvalue)
                else
                    λAC[1,j], AC′[1,j] = simple_eig(f, AC[1,j]; ifvalue=ifvalue)
                end
            else
                λACs, ACs, info = eigsolve(f, AC[1,j], 1, :LM; alg_rrule=GMRES(verbosity=-1), maxiter=100, ishermitian = false, kwargs...)
                alg.verbosity >= 1 && info.converged == 0 && @warn "ACenv Not converged"
                λAC[1,j], AC′[1,j] = λACs[1], ACs[1]
            end
            push!(processed_indices, p)
            if length(processed_indices) == length(AC.data)
                break
            end
        end
        for i in 2:Ni
            p = AC.pattern[i,j]
            if p ∉ processed_indices
                ACij = ACmap(AC′[i-1,j], FL[i-1,j], FR[i-1,j], M[i-1,j])
                AC′[i,j] = ACij / norm(ACij)
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
    λC = Zygote.Buffer(rand(eltype(C.data[1]), C.pattern))
    C′ = Zygote.Buffer(C)
    processed_indices = Set{Int}()
    for j in 1:Nj
        jr = mod1(j + 1, Nj)
        p = C.pattern[1,j]
        if p ∉ processed_indices
            f(C1j) = Cmap(1, C1j, FL[:,jr], FR[:,j])
            if alg.ifsimple_eig
                if alg.ifcheckpoint
                    λC[1,j], C′[1,j] = checkpoint(simple_eig, f, C[1,j]; ifvalue=ifvalue)
                else
                    λC[1,j], C′[1,j] = simple_eig(f, C[1,j]; ifvalue=ifvalue)
                end
            else
                λCs, Cs, info = eigsolve(f, C[1,j], 1, :LM; alg_rrule=GMRES(verbosity=-1), maxiter=100, ishermitian = false, kwargs...)
                alg.verbosity >= 1 && info.converged == 0 && @warn "Cenv Not converged"
                λC[1,j], C′[1,j] = λCs[1], Cs[1]
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
                C′[i,j] = Cij / norm(Cij)
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

"""
    AL, AR = ACCtoALAR(AC, C)

QR factorization to get `AL` and `AR` from `AC` and `C`

````
──ALᵢⱼ──Cᵢⱼ──  =  ──ACᵢⱼ──  = ──Cᵢ₋₁ⱼ ──ARᵢⱼ──
  │                  │                  │   
````
"""
function ACCtoALAR(AC::StructArray, C::StructArray)
    AL, errL = ACCtoAL(AC, C)
    AR, errR = ACCtoAR(AC, C)
    return AL, AR, errL, errR
end

function ACCtoAL(AC::StructArray, C::StructArray)
    Ni, Nj = size(AC)
    errL = 0.0
    AL = Zygote.Buffer(AC)
    @inbounds for j in 1:Nj, i in 1:Ni
        QAC, RAC = leftorth(AC[i,j])
         QC, RC  = leftorth( C[i,j])
        errL += norm(RAC - RC)
        AL[i,j] = QAC * QC'
    end
    return copy(AL), errL
end

function ACCtoAR(AC::StructArray, C::StructArray)
    Ni, Nj = size(AC)
    errR = 0.0
    AR = Zygote.Buffer(AC)
    @inbounds for j in 1:Nj, i in 1:Ni
        jr = mod1(j - 1, Nj)
        LAC, QAC = rightorth(_to_tail(AC[i,j]))
         LC, QC  = rightorth(C[i,jr])
        errR += norm(LAC - LC)
        AR[i,j] = _to_front(QC' * QAC)
    end
    return copy(AR), errR
end