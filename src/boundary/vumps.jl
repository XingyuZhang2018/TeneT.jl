# VUMPS boundary algorithm implementation
# Uses struct definitions from boundary/algorithm.jl (VUMPS{M}) and
# boundary/environment.jl (VUMPSRuntime, VUMPSEnv)

# ── Helpers ──────────────────────────────────────────────────────────

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
    λs[1], Fs[1] = selectpos(λs, Fs, N)

Select the max positive one of λs and corresponding Fs.
"""
function selectpos(λs, Fs, N)
    if length(λs) > 1 && norm(abs(λs[1]) - abs(λs[2])) < 1e-12
        N = max(N, length(λs))
        p = argmax(real(λs[1:N]))
        return λs[1:N][p], Fs[1:N][p]
    else
        return λs[1], Fs[1]
    end
end

"""
    ρmap(ρ, A, j)

Density matrix map for the transfer matrix of an MPS row `A` at column `j`.
"""
ρmap(ρ, Au::leg3, Ad::leg3) = ein"(dc,csb),dsa -> ab"(ρ, Au, Ad)
ρmap(ρ, Au::leg4, Ad::leg4) = ein"(dc,cstb),dsta -> ab"(ρ, Au, Ad)

"""
    getL!(A, L; kwargs...)

Compute the gauge transform `L` from the transfer matrix density.
ρ = L' * L, returns `L` with positive diagonal elements.
"""
function getL!(A, L; kwargs...)
    Ni, Nj = size(A)
    @inbounds for j = 1:Nj, i = 1:Ni
        _, ρ = simple_eig(ρ -> ρmap(ρ, A[i, :], j), L[i, j]' * L[i, j]; kwargs...)
        ρ = real(ρ + ρ')
        ρ ./= tr(ρ)
        F = svd!(ρ)
        Lo = Diagonal(sqrt.(F.S)) * F.Vt
        _, R = qrpos!(Lo)
        L[i, j] = R
    end
    return L
end

"""
    getAL(A, L)

Given an MPS tensor `A` and `L`, return a left-canonical MPS tensor `AL`, a gauge transform `R` and
a scalar factor `λ` such that `λ AR R = L A`.
"""
function getAL(A, L)
    AL = similar(A)
    Le = similar(L)
    λ = randSA(Array, AL.pattern)
    for i in 1:length(A)
        Q, R = qrpos!(_to_tail(L[i] * _to_front(A[i])))
        AL[i] = reshape(Q, size(A[i]))
        λ[i] = norm(R)
        Le[i] = rmul!(R, 1 / λ[i])
    end
    return AL, Le, λ
end

function getLsped(Le, A, AL; kwargs...)
    L = similar(Le)
    for i in 1:length(A)
        _, Ls1 = simple_eig(X -> ρmap(X, A[i], conj(AL[i])), Le[i]; power_iter=5, kwargs...)
        _, R = qrpos!(Ls1[1])
        L[i] = R
    end
    return L
end

# ── Canonical forms ──────────────────────────────────────────────────

"""
    left_canonical(A, L=cellones(A); tol=1e-12, maxiter=100, kwargs...)

Given an MPS tensor `A`, return a left-canonical MPS tensor `AL`, a gauge transform `L` and
a scalar factor `λ` such that `λ AL L = L A`.
"""
function left_canonical(A, L=cellones(A); tol=1e-12, maxiter=100, kwargs...)
    AL, Le, λ = getAL(A, L; kwargs...)
    numiter = 1
    while norm(L .- Le) > tol && numiter < maxiter
        L = getLsped(Le, A, AL; kwargs...)
        AL, Le, λ = getAL(A, L; kwargs...)
        numiter += 1
    end
    L = Le
    return AL, L, λ
end

"""
    right_canonical(A, L=cellones(A); tol=1e-12, maxiter=100, kwargs...)

Given an MPS tensor `A`, return a gauge transform `R`, a right-canonical MPS tensor `AR`, and
a scalar factor `λ` such that `λ R AR = A R`.
"""
function right_canonical(A, L=cellones(A); tol=1e-12, maxiter=100, kwargs...)
    Ar = similar(A)
    Lr = similar(L)
    @inbounds for i in 1:length(A)
        Ar[i] = permute_fronttail(A[i])
        Lr[i] = permutedims(L[i], (2, 1))
    end

    AL, L, λ = left_canonical(Ar, Lr; tol=tol, maxiter=maxiter, kwargs...)
    R = similar(L)
    AR = similar(AL)
    @inbounds for i in 1:length(AL)
        R[i] = permutedims(L[i], (2, 1))
        AR[i] = permute_fronttail(AL[i])
    end
    return R, AR, λ
end

"""
    LRtoC(L, R)

Compute center matrix `C` from gauge transforms: `Cᵢⱼ = Lᵢⱼ * Rᵢⱼ₊₁`.
"""
function LRtoC(L, R)
    Rijr = circshift(R, (0, -1))
    C = similar(L)
    @inbounds for i in 1:length(L)
        C[i] = L[i] * Rijr[i]
    end
    return C
end

# ── Mixed canonical tensor ──────────────────────────────────────────

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

# ── Fixed-point environments ────────────────────────────────────────

function FLmap(J::Int, FLij, ALui, ALdir, Mi; ifcheckpoint, ifparallel, forloop_iter)
    Nj = length(ALui)
    for j in J:(J + Nj - 1)
        jr = mod1(j, Nj)
        FLij = ifcheckpoint ? checkpoint(FLmap_parallel, FLij, ALui[jr], ALdir[jr], Mi[jr]; ifparallel, forloop_iter) : FLmap_parallel(FLij, ALui[jr], ALdir[jr], Mi[jr]; ifparallel, forloop_iter)
    end
    return FLij
end

"""
    λL, FL = leftenv(ALu, ALd, M, FL=FLint(ALu,M); kwargs...)

Compute the left environment tensor for MPS `ALu`, `ALd` and MPO `M`, by finding the left fixed point
of ALu - M - ALd contracted along the physical dimension.
"""
function leftenv(ALu, ALd, M, FL=FLint(ALu, M); ifobs=false, ifvalue=false, alg, kwargs...)
    λL = Zygote.Buffer(randSA(Array, M.pattern))
    FL′ = Zygote.Buffer(FL)
    Ni, Nj = size(M)
    processed_indices = Set{Int}()
    power_iter = ifobs ? alg.power_iter_obs : alg.power_iter
    forloop_iter = alg.forloop_iter
    ifcheckpoint = alg.ifcheckpoint
    ifparallel = alg.ifparallel
    for i in 1:Ni
        ir = ifobs ? Ni + 1 - i : mod1(i + 1, Ni)
        p = FL.pattern[i, 1]
        if p ∉ processed_indices
            f(FLij) = ifcheckpoint ? checkpoint(FLmap, 1, FLij, ALu[i, :], ALd[ir, :], M[i, :]; ifcheckpoint, ifparallel, forloop_iter) : FLmap(1, FLij, ALu[i, :], ALd[ir, :], M[i, :]; ifcheckpoint, ifparallel, forloop_iter)
            if alg.ifsimple_eig
                if alg.iflinear_ad
                    if ifcheckpoint
                        λLs, FLi1s = checkpoint(simple_eig_linear_ad, f, FL[i, 1]; ifvalue, power_iter)
                    else
                        λLs, FLi1s = simple_eig_linear_ad(f, FL[i, 1]; ifvalue, power_iter)
                    end
                else
                    if ifcheckpoint
                        λLs, FLi1s = checkpoint(simple_eig, f, FL[i, 1]; ifvalue, power_iter)
                    else
                        λLs, FLi1s = simple_eig(f, FL[i, 1]; ifvalue, power_iter)
                    end
                end
            else
                λLs, FLi1s, info = eigsolve(f, FL[i, 1], 1, :LM; alg_rrule=GMRES(verbosity=-1), maxiter=100, ishermitian=false, kwargs...)
                alg.verbosity >= 1 && info.converged == 0 && @warn "leftenv not converged"
            end
            λL[i, 1], FL′[i, 1] = selectpos(λLs, FLi1s, Nj)

            push!(processed_indices, p)
            if length(processed_indices) == length(FL.data)
                break
            end
        end
        for j in 2:Nj
            p = FL.pattern[i, j]
            if p ∉ processed_indices
                FL′[i, j] = ifcheckpoint ? checkpoint(FLmap_parallel, FL′[i, j-1], ALu[i, j-1], ALd[ir, j-1], M[i, j-1]; ifparallel, forloop_iter) : FLmap_parallel(FL′[i, j-1], ALu[i, j-1], ALd[ir, j-1], M[i, j-1]; ifparallel, forloop_iter)
                λL[i, j] = λL[i, 1]
                push!(processed_indices, p)
                if length(processed_indices) == length(FL.data)
                    break
                end
            end
        end
    end

    return copy(λL), copy(FL′)
end

function FRmap(J::Int, FRij, ARui, ARdir, Mi; ifcheckpoint, ifparallel, forloop_iter)
    Nj = length(ARui)
    for j in J:-1:(J - Nj + 1)
        jr = mod1(j, Nj)
        FRij = ifcheckpoint ? checkpoint(FRmap_parallel, FRij, ARui[jr], ARdir[jr], Mi[jr]; ifparallel, forloop_iter) : FRmap_parallel(FRij, ARui[jr], ARdir[jr], Mi[jr]; ifparallel, forloop_iter)
    end
    return FRij
end

"""
    λR, FR = rightenv(ARu, ARd, M, FR=FRint(ARu,M); kwargs...)

Compute the right environment tensor for MPS `ARu`, `ARd` and MPO `M`, by finding the right fixed point
of AR - M - conj(AR) contracted along the physical dimension.
"""
function rightenv(ARu, ARd, M, FR=FRint(ARu, M); ifobs=false, ifvalue=false, alg, kwargs...)
    Ni, Nj = size(M)
    λR = Zygote.Buffer(randSA(Array, M.pattern))
    FR′ = Zygote.Buffer(FR)
    processed_indices = Set{Int}()
    power_iter = ifobs ? alg.power_iter_obs : alg.power_iter
    forloop_iter = alg.forloop_iter
    ifcheckpoint = alg.ifcheckpoint
    ifparallel = alg.ifparallel
    for i in 1:Ni
        ir = ifobs ? Ni + 1 - i : mod1(i + 1, Ni)
        p = FR.pattern[i, Nj]
        if p ∉ processed_indices
            f(FRiNj) = ifcheckpoint ? checkpoint(FRmap, Nj, FRiNj, ARu[i, :], ARd[ir, :], M[i, :]; ifcheckpoint, ifparallel, forloop_iter) : FRmap(Nj, FRiNj, ARu[i, :], ARd[ir, :], M[i, :]; ifcheckpoint, ifparallel, forloop_iter)
            if alg.ifsimple_eig
                if alg.iflinear_ad
                    if ifcheckpoint
                        λRs, FR1s = checkpoint(simple_eig_linear_ad, f, FR[i, Nj]; ifvalue, power_iter)
                    else
                        λRs, FR1s = simple_eig_linear_ad(f, FR[i, Nj]; ifvalue, power_iter)
                    end
                else
                    if ifcheckpoint
                        λRs, FR1s = checkpoint(simple_eig, f, FR[i, Nj]; ifvalue, power_iter)
                    else
                        λRs, FR1s = simple_eig(f, FR[i, Nj]; ifvalue, power_iter)
                    end
                end
            else
                λRs, FR1s, info = eigsolve(f, FR[i, Nj], 1, :LM; alg_rrule=GMRES(verbosity=-1), maxiter=100, ishermitian=false, kwargs...)
                alg.verbosity >= 1 && info.converged == 0 && @warn "rightenv not converged"
            end
            λR[i, Nj], FR′[i, Nj] = selectpos(λRs, FR1s, Nj)

            push!(processed_indices, p)
            if length(processed_indices) == length(FR.data)
                break
            end
        end
        for j in Nj-1:-1:1
            p = FR.pattern[i, j]
            if p ∉ processed_indices
                FR′[i, j] = ifcheckpoint ? checkpoint(FRmap_parallel, FR′[i, j+1], ARu[i, j+1], ARd[ir, j+1], M[i, j+1]; ifparallel, forloop_iter) : FRmap_parallel(FR′[i, j+1], ARu[i, j+1], ARd[ir, j+1], M[i, j+1]; ifparallel, forloop_iter)
                λR[i, j] = λR[i, Nj]
                push!(processed_indices, p)
                if length(processed_indices) == length(FR.data)
                    break
                end
            end
        end
    end
    return copy(λR), copy(FR′)
end

# ── Left/Right C environments (no MPO) ──────────────────────────────

function Lmap(J::Int, Lij, ALui, ALdir)
    Nj = length(ALui)
    for j in J:(J + Nj - 1)
        jr = mod1(j, Nj)
        Lij = Lmap(Lij, ALui[jr], ALdir[jr])
    end
    return Lij
end

"""
    leftCenv(ALu, ALd, L=cellones(ALu); kwargs...)

Compute the left environment tensor for MPS `ALu` and `ALd` (no MPO), by finding the left fixed point.
"""
function leftCenv(ALu::StructArray,
                  ALd::StructArray,
                  L::StructArray=cellones(ALu);
                  ifobs=false, ifvalue=false, alg, kwargs...)

    Ni, Nj = size(L)
    λL = Zygote.Buffer(randSA(Array, ALu.pattern))
    L′ = Zygote.Buffer(L)
    power_iter = ifobs ? alg.power_iter_obs : alg.power_iter
    processed_indices = Set{Int}()
    for i in 1:Ni
        ir = ifobs ? mod1(Ni + 2 - i, Ni) : i
        p = L.pattern[i, 1]
        if p ∉ processed_indices
            f(Lij) = Lmap(1, Lij, ALu[i, :], ALd[ir, :])
            if alg.ifsimple_eig
                if alg.ifcheckpoint
                    λLs, Li1s = checkpoint(simple_eig, f, L[i, 1]; ifvalue, power_iter)
                else
                    λLs, Li1s = simple_eig(f, L[i, 1]; ifvalue, power_iter)
                end
            else
                λLs, Li1s, info = eigsolve(f, L[i, 1], 1, :LM; maxiter=100, ishermitian=false, kwargs...)
                alg.verbosity >= 1 && info.converged == 0 && @warn "leftCenv not converged"
            end
            λL[i, 1], L′[i, 1] = selectpos(λLs, Li1s, Nj)

            push!(processed_indices, p)
            if length(processed_indices) == length(L.data)
                break
            end
        end
        for j in 2:Nj
            p = L.pattern[i, j]
            if p ∉ processed_indices
                Lij = Lmap(L′[i, j-1], ALu[i, j-1], ALd[ir, j-1])
                L′[i, j] = Lij / norm(Lij)
                λL[i, j] = λL[i, 1]
                push!(processed_indices, p)
                if length(processed_indices) == length(L.data)
                    break
                end
            end
        end
    end

    return copy(λL), copy(L′)
end

function Rmap(J::Int, Rij, ARui, ARdir)
    Nj = length(ARui)
    for j in J:-1:(J - Nj + 1)
        jr = mod1(j, Nj)
        Rij = Rmap(Rij, ARui[jr], ARdir[jr])
    end
    return Rij
end

"""
    rightCenv(ARu, ARd, R=cellones(ARu); kwargs...)

Compute the right environment tensor for MPS `ARu` and `ARd` (no MPO), by finding the right fixed point.
"""
function rightCenv(ARu::StructArray,
                   ARd::StructArray,
                   R::StructArray=cellones(ARu);
                   ifobs=false, ifvalue=false, alg, kwargs...)

    λR = Zygote.Buffer(randSA(Array, ARu.pattern))
    R′ = Zygote.Buffer(R)
    power_iter = ifobs ? alg.power_iter_obs : alg.power_iter
    processed_indices = Set{Int}()
    Ni, Nj = size(R)
    for i in 1:Ni
        ir = ifobs ? mod1(Ni + 2 - i, Ni) : i
        p = R.pattern[i, Nj]
        if p ∉ processed_indices
            f(RiNj) = Rmap(Ni, RiNj, ARu[i, :], ARd[ir, :])
            if alg.ifsimple_eig
                if alg.ifcheckpoint
                    λLs, Li1s = checkpoint(simple_eig, f, R[i, Nj]; ifvalue, power_iter)
                else
                    λLs, Li1s = simple_eig(f, R[i, Nj]; ifvalue, power_iter)
                end
            else
                λLs, Li1s, info = eigsolve(f, R[i, Nj], 1, :LM; maxiter=100, ishermitian=false, kwargs...)
                alg.verbosity >= Nj && info.converged == 0 && @warn "rightCenv not converged"
            end
            λR[i, Nj], R′[i, Nj] = selectpos(λLs, Li1s, Nj)

            push!(processed_indices, p)
            if length(processed_indices) == length(R.data)
                break
            end
        end
        for j in Nj-1:-1:1
            p = R.pattern[i, j]
            if p ∉ processed_indices
                Rij = Rmap(R′[i, j+1], ARu[i, j+1], ARd[ir, j+1])
                R′[i, j] = Rij / norm(Rij)
                λR[i, j] = λR[i, Nj]
                push!(processed_indices, p)
                if length(processed_indices) == length(R.data)
                    break
                end
            end
        end
    end

    return copy(λR), copy(R′)
end

# ── AC and C environment updates ────────────────────────────────────

function ACmap(I::Int, ACij, FLj, FRj, Mj; ifcheckpoint, ifparallel, forloop_iter)
    Ni = length(FLj)
    for i in I:(I + Ni - 1)
        ir = mod1(i, Ni)
        ACij = ifcheckpoint ? checkpoint(ACmap_parallel, ACij, FLj[ir], FRj[ir], Mj[ir]; ifparallel, forloop_iter) : ACmap_parallel(ACij, FLj[ir], FRj[ir], Mj[ir]; ifparallel, forloop_iter)
    end
    return ACij
end

"""
    ACenv(AC, FL, M, FR; kwargs...)

Compute the up environment tensor for MPS `FL`, `FR` and MPO `M`, by finding the up fixed point
of `FL - M - FR` contracted along the physical dimension.
"""
function ACenv(AC, FL, M, FR; ifvalue=false, alg, kwargs...)
    Ni, Nj = size(M)
    λAC = Zygote.Buffer(randSA(Array, M.pattern))
    AC′ = Zygote.Buffer(AC)
    processed_indices = Set{Int}()
    power_iter = alg.power_iter
    forloop_iter = alg.forloop_iter
    ifcheckpoint = alg.ifcheckpoint
    ifparallel = alg.ifparallel
    for j in 1:Nj
        p = AC.pattern[1, j]
        if p ∉ processed_indices
            f(AC1j) = ifcheckpoint ? checkpoint(ACmap, 1, AC1j, FL[:, j], FR[:, j], M[:, j]; ifcheckpoint, ifparallel, forloop_iter) : ACmap(1, AC1j, FL[:, j], FR[:, j], M[:, j]; ifcheckpoint, ifparallel, forloop_iter)
            if alg.ifsimple_eig
                if alg.iflinear_ad
                    if ifcheckpoint
                        λACs, ACs = checkpoint(simple_eig_linear_ad, f, AC[1, j]; ifvalue, power_iter)
                    else
                        λACs, ACs = simple_eig_linear_ad(f, AC[1, j]; ifvalue, power_iter)
                    end
                else
                    if ifcheckpoint
                        λACs, ACs = checkpoint(simple_eig, f, AC[1, j]; ifvalue, power_iter)
                    else
                        λACs, ACs = simple_eig(f, AC[1, j]; ifvalue, power_iter)
                    end
                end
            else
                λACs, ACs, info = eigsolve(f, AC[1, j], 1, :LM; alg_rrule=GMRES(verbosity=-1), maxiter=100, ishermitian=false, kwargs...)
                alg.verbosity >= 1 && info.converged == 0 && @warn "ACenv Not converged"
            end
            λAC[1, j], AC′[1, j] = selectpos(λACs, ACs, Ni)

            push!(processed_indices, p)
            if length(processed_indices) == length(AC.data)
                break
            end
        end
        for i in 2:Ni
            p = AC.pattern[i, j]
            if p ∉ processed_indices
                ACij = ifcheckpoint ? checkpoint(ACmap_parallel, AC′[i-1, j], FL[i-1, j], FR[i-1, j], M[i-1, j]; ifparallel, forloop_iter) : ACmap_parallel(AC′[i-1, j], FL[i-1, j], FR[i-1, j], M[i-1, j]; ifparallel, forloop_iter)
                AC′[i, j] = ACij / norm(ACij)
                λAC[i, j] = λAC[1, j]
                push!(processed_indices, p)
                if length(processed_indices) == length(AC.data)
                    break
                end
            end
        end
    end
    return copy(λAC), copy(AC′)
end

function Cmap(I, Cij, FLjr, FRj)
    Ni = length(FLjr)
    for i in I:(I + Ni - 1)
        ir = mod1(i, Ni)
        Cij = Cmap(Cij, FLjr[ir], FRj[ir])
    end
    return Cij
end

"""
    Cenv(C, FL, FR; kwargs...)

Compute the up environment tensor for MPS `FL` and `FR`, by finding the up fixed point
of `FL - FR` contracted along the physical dimension.
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
        p = C.pattern[1, j]
        if p ∉ processed_indices
            f(C1j) = Cmap(1, C1j, FL[:, jr], FR[:, j])
            if alg.ifsimple_eig
                if alg.iflinear_ad
                    if ifcheckpoint
                        λCs, Cs = checkpoint(simple_eig_linear_ad, f, C[1, j]; ifvalue, power_iter)
                    else
                        λCs, Cs = simple_eig_linear_ad(f, C[1, j]; ifvalue, power_iter)
                    end
                else
                    if ifcheckpoint
                        λCs, Cs = checkpoint(simple_eig, f, C[1, j]; ifvalue, power_iter)
                    else
                        λCs, Cs = simple_eig(f, C[1, j]; ifvalue, power_iter)
                    end
                end
            else
                λCs, Cs, info = eigsolve(f, C[1, j], 1, :LM; alg_rrule=GMRES(verbosity=-1), maxiter=100, ishermitian=false, kwargs...)
                alg.verbosity >= 1 && info.converged == 0 && @warn "Cenv Not converged"
            end
            λC[1, j], C′[1, j] = selectpos(λCs, Cs, Ni)

            push!(processed_indices, p)
            if length(processed_indices) == length(C.data)
                break
            end
        end
        for i in 2:Ni
            p = C.pattern[i, j]
            if p ∉ processed_indices
                Cij = Cmap(C′[i-1, j], FL[i-1, jr], FR[i-1, j])
                C′[i, j] = Cij / norm(Cij)
                λC[i, j] = λC[1, j]
                push!(processed_indices, p)
                if length(processed_indices) == length(C.data)
                    break
                end
            end
        end
    end
    return copy(λC), copy(C′)
end

# ── AC/C to AL/AR conversions ───────────────────────────────────────

function ACCtoAL(AC, C)
    errL = 0.0
    AL = Zygote.Buffer(AC)
    @inbounds for i in 1:length(AC)
        QAC, RAC = qrpos(_to_tail(AC[i]))
        QC, RC = qrpos(C[i])
        errL += norm(RAC - RC)
        AL[i] = reshape(QAC * QC', size(AC[i]))
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
        LAC, QAC = lqpos(_to_front(AC[i, j]))
        LC, QC = lqpos(C[i, jr])
        errR += norm(LAC - LC)
        AR[i, j] = reshape(QC' * QAC, size(AC[i, j]))
    end
    return copy(AR), errR
end

"""
    AL, AR, errL, errR = ACCtoALAR(AC, C)

QR factorization to get `AL` and `AR` from `AC` and `C`.
"""
function ACCtoALAR(AC, C)
    AL, errL = ACCtoAL(AC, C)
    AR, errR = ACCtoAR(AC, C)
    return AL, AR, errL, errR
end

# ── Down environment helpers ────────────────────────────────────────

_down_m(m::leg4) = permutedims(conj(m), (1, 4, 3, 2))
_down_m(m::leg5) = permutedims(conj(m), (1, 4, 3, 2, 5))
_down_m(m::leg8) = permutedims(conj(m), (1, 2, 7, 8, 5, 6, 3, 4))

function _down_M(M::StructArray)
    Ni, Nj = size(M)
    pattern_d = copy(M.pattern)
    Zygote.@ignore begin
        @inbounds for i in 1:Ni, j in 1:Nj
            ir = Ni + 1 - i
            pattern_d[i, j] = M.pattern[ir, j]
        end
    end
    data_d = [_down_m(data) for data in M.data]
    Md = StructArray(data_d, pattern_d)
    return Md
end

function _down_init_from_up(rtup::VUMPSRuntime, Md::StructArray)
    @unpack AL, AR, C, FL, FR = rtup
    ALd = StructArray(AL.data, Md.pattern)
    ARd = StructArray(AR.data, Md.pattern)
    Cd = StructArray(C.data, Md.pattern)
    FLd = StructArray(FL.data, Md.pattern)
    FRd = StructArray(FR.data, Md.pattern)
    return VUMPSRuntime(ALd, ARd, Cd, FLd, FRd)
end

# ── VUMPS step functions ────────────────────────────────────────────

"""
    vumps_step(rt, M, alg::VUMPS{General})

One step of the VUMPS algorithm with the standard (General) contraction mode.
Uses the power-method variant: update environments first, then re-solve AC/C.
"""
function vumps_step(rt::VUMPSRuntime, M::StructArray, alg::VUMPS{General})
    @unpack AL, C, AR, FL, FR = rt
    AC = ALCtoAC(AL, C)
    _, ACp = ACenv(AC, FL, M, FR; alg)
    _, Cp = Cenv(C, FL, FR; alg)
    ALp, ARp, _, _ = ACCtoALAR(ACp, Cp)
    _, FL = leftenv(AL, conj(ALp), M, FL; alg)
    _, FR = rightenv(AR, conj(ARp), M, FR; alg)
    _, ACp = ACenv(ACp, FL, M, FR; alg)
    _, Cp = Cenv(Cp, FL, FR; alg)
    ALp, ARp, errL, errR = ACCtoALAR(ACp, Cp)
    err = errL + errR
    alg.verbosity >= 4 && err > 1e-8 && println("errL=$errL, errR=$errR")
    Cp = for_gc(Cp)
    return VUMPSRuntime(ALp, ARp, Cp, FL, FR), err
end

"""
    vumps_step(rt, M, alg::VUMPS{Plaquette})

One step of the VUMPS algorithm with the 2x2 Plaquette contraction mode.
TODO: Implement plaquette-specific contraction logic.
Currently falls back to the General step.
"""
function vumps_step(rt::VUMPSRuntime, M::StructArray, alg::VUMPS{Plaquette})
    # TODO: Implement plaquette-specific VUMPS step with 2x2 unit cell handling
    # For now, delegate to the same logic as General
    @unpack AL, C, AR, FL, FR = rt
    AC = ALCtoAC(AL, C)
    _, ACp = ACenv(AC, FL, M, FR; alg)
    _, Cp = Cenv(C, FL, FR; alg)
    ALp, ARp, _, _ = ACCtoALAR(ACp, Cp)
    _, FL = leftenv(AL, conj(ALp), M, FL; alg)
    _, FR = rightenv(AR, conj(ARp), M, FR; alg)
    _, ACp = ACenv(ACp, FL, M, FR; alg)
    _, Cp = Cenv(Cp, FL, FR; alg)
    ALp, ARp, errL, errR = ACCtoALAR(ACp, Cp)
    err = errL + errR
    alg.verbosity >= 4 && err > 1e-8 && println("errL=$errL, errR=$errR")
    Cp = for_gc(Cp)
    return VUMPSRuntime(ALp, ARp, Cp, FL, FR), err
end

# ── VUMPS iteration loop ────────────────────────────────────────────

"""
    vumps_itr(rt, M, alg::VUMPS)

Run the VUMPS iteration loop: first without AD tracking (warm-up), then with AD.
Returns the converged runtime and final error.
"""
function vumps_itr(rt::VUMPSRuntime, M::StructArray, alg::VUMPS)
    t = Zygote.@ignore time()

    err = Inf
    Zygote.@ignore alg.verbosity >= 2 && @info "Start VUMPS iteration without AD..."
    Zygote.@ignore for i in 1:alg.maxiter
        rt, err = vumps_step(rt, M, alg)
        alg.verbosity >= 3 && i % alg.show_every == 0 && Zygote.@ignore @info @sprintf("VUMPS@step: %4d\terr = %.3e\ttime = %.3f sec", i, err, time() - t)
        if err < alg.tol && i >= alg.miniter
            alg.verbosity >= 2 && Zygote.@ignore @info @sprintf("VUMPS conv@step: %4d\terr = %.3e\ttime = %.3f sec", i, err, time() - t)
            break
        end
        if i == alg.maxiter
            alg.verbosity >= 2 && Zygote.@ignore @warn @sprintf("VUMPS cancel@step: %4d\terr = %.3e\ttime = %.3f sec", i, err, time() - t)
        end
    end

    if err < 1e-7
        alg.iflinear_ad = true
    else
        alg.iflinear_ad = false
    end
    Zygote.@ignore alg.verbosity >= 2 && @info "Start VUMPS iteration with AD..."
    for i in 1:alg.maxiter_ad
        rt, err = alg.ifcheckpoint ? checkpoint(vumps_step, rt, M, alg) : vumps_step(rt, M, alg)
        alg.verbosity >= 3 && i % alg.show_every == 0 && Zygote.@ignore @info @sprintf("VUMPS@step: %4d\terr = %.3e\ttime = %.3f sec", i, err, time() - t)
        if err < alg.tol && i >= alg.miniter_ad
            alg.verbosity >= 2 && Zygote.@ignore @info @sprintf("VUMPS conv@step: %4d\terr = %.3e\ttime = %.3f sec", i, err, time() - t)
            break
        end
        if i == alg.maxiter_ad
            alg.verbosity >= 2 && Zygote.@ignore @warn @sprintf("VUMPS cancel@step: %4d\terr = %.3e\ttime = %.3f sec", i, err, time() - t)
        end
    end

    return rt, err
end

# ── Public API ───────────────────────────────────────────────────────

"""
    leading_boundary(rt::VUMPSRuntime, M, alg::VUMPS)

Run the VUMPS boundary contraction for a single (up) environment.
Returns the converged runtime and error.
"""
function leading_boundary(rt::VUMPSRuntime, M::StructArray, alg::VUMPS)
    rt, err = vumps_itr(rt, M, alg)
    return rt, err
end

"""
    leading_boundary(rt::Tuple{VUMPSRuntime, VUMPSRuntime}, M, alg::VUMPS)

Run the VUMPS boundary contraction for both up and down environments.
Returns the converged runtimes and errors.
"""
function leading_boundary(rt::Tuple{VUMPSRuntime,VUMPSRuntime}, M::StructArray, alg::VUMPS)
    rtup, rtdown = rt

    rtup, errup = vumps_itr(rtup, M, alg)

    Md = _down_M(M)
    rtdown, errdown = vumps_itr(rtdown, Md, alg)
    return (rtup, rtdown), (errup, errdown)
end

# ── Initialization ──────────────────────────────────────────────────

"""
    init_VUMPSRuntime(M, alg::VUMPS; χ)

Initialize a `VUMPSRuntime` from an MPO `M` and bond dimension `χ`.
Computes initial canonical forms and fixed-point environments.
"""
function init_VUMPSRuntime(M::StructArray, alg::VUMPS; χ::Int)
    A = initial_A(M, χ)
    AL, L, _ = left_canonical(A)
    R, AR, _ = right_canonical(AL)
    C = LRtoC(L, R)
    if alg.ifparallel
        AL = MPI.bcast(AL, 0, MPI.COMM_WORLD)
        AR = MPI.bcast(AR, 0, MPI.COMM_WORLD)
        C = MPI.bcast(C, 0, MPI.COMM_WORLD)
    end
    _, FL = leftenv(AL, conj(AL), M; alg)
    _, FR = rightenv(AR, conj(AR), M; alg)
    return VUMPSRuntime(AL, AR, C, FL, FR)
end

"""
    init_VUMPSRuntime(M, alg::VUMPS; χ, ifupdown=alg.ifupdown)

Initialize one or two `VUMPSRuntime`s (up and optionally down) from an MPO `M`.
"""
function init_VUMPSRuntime(M::StructArray, alg::VUMPS; χ::Int, ifupdown::Bool=alg.ifupdown)
    Ni, Nj = size(M)

    rtup = init_VUMPSRuntime(M, alg; χ)
    alg.verbosity >= 2 && Zygote.@ignore @info "VUMPS init: cell=($(Ni)x$(Nj)) χ=$(χ) up environment"

    if ifupdown
        Md = _down_M(M)
        if alg.ifdownfromup
            rtdown = _down_init_from_up(rtup, Md)
            alg.verbosity >= 2 && Zygote.@ignore @info "VUMPS init: cell=($(Ni)x$(Nj)) χ=$(χ) down from up environment"
            return rtup, rtdown
        else
            rtdown = init_VUMPSRuntime(Md, alg; χ)
            alg.verbosity >= 2 && Zygote.@ignore @info "VUMPS init: cell=($(Ni)x$(Nj)) χ=$(χ) down environment"
            return rtup, rtdown
        end
    else
        return rtup
    end
end

# ── Observation environment construction ─────────────────────────────

"""
    VUMPSEnv(rt::VUMPSRuntime, M, alg)

Construct a `VUMPSEnv` observation environment from a single VUMPS runtime.
Uses the same runtime for both up and down directions.
"""
function VUMPSEnv(rt::VUMPSRuntime, M::StructArray, alg::VUMPS, Fo=[rt.FL, rt.FR])
    @unpack AL, AR, C, FL, FR = rt
    AC = ALCtoAC(AL, C)
    return VUMPSEnv(AC, AR, AC, AR, FL, FR, FL, FR)
end

"""
    VUMPSEnv(rt::Tuple{VUMPSRuntime, VUMPSRuntime}, M, alg)

Construct a `VUMPSEnv` observation environment from up and down VUMPS runtimes.
Computes mixed (observation) left and right environments.
"""
function VUMPSEnv(rt::Tuple{VUMPSRuntime,VUMPSRuntime}, M::StructArray, alg::VUMPS, Fo=[rt[1].FL, rt[1].FR])
    rtup, rtdown = rt

    ALu, ARu, Cu, FLu, FRu = rtup.AL, rtup.AR, rtup.C, rtup.FL, rtup.FR
    ACu = ALCtoAC(ALu, Cu)

    ALd, ARd, Cd = rtdown.AL, rtdown.AR, rtdown.C
    ACd = ALCtoAC(ALd, Cd)

    _, FLo = leftenv(ALu, conj(ALd), M, Fo[1]; ifobs=true, alg)
    _, FRo = rightenv(ARu, conj(ARd), M, Fo[2]; ifobs=true, alg)
    return VUMPSEnv(ACu, ARu, ACd, ARd, FLu, FRu, FLo, FRo)
end

# ── GPU/CPU array conversions ────────────────────────────────────────

Array(rt::VUMPSRuntime) = VUMPSRuntime(Array(rt.AL), Array(rt.AR), Array(rt.C), Array(rt.FL), Array(rt.FR))
Array(rt::Tuple{VUMPSRuntime,VUMPSRuntime}) = Array.(rt)
CuArray(rt::VUMPSRuntime) = VUMPSRuntime(CuArray(rt.AL), CuArray(rt.AR), CuArray(rt.C), CuArray(rt.FL), CuArray(rt.FR))
CuArray(rt::Tuple{VUMPSRuntime,VUMPSRuntime}) = CuArray.(rt)
ROCArray(rt::VUMPSRuntime) = VUMPSRuntime(ROCArray(rt.AL), ROCArray(rt.AR), ROCArray(rt.C), ROCArray(rt.FL), ROCArray(rt.FR))
ROCArray(rt::Tuple{VUMPSRuntime,VUMPSRuntime}) = ROCArray.(rt)
