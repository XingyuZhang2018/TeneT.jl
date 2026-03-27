# VUMPS boundary algorithm implementation
# Uses struct definitions from boundary/algorithm.jl (VUMPS{M}) and
# boundary/environment.jl (VUMPSRuntime, VUMPSEnv)

# ─── Reshape helpers ──────────────────────────────────────────────────────────

function _to_tail(t)
    χ = size(t)[end]
    return reshape(t, χ, Int(prod(size(t))/χ))
end

function _to_front(t)
    χ = size(t, 1)
    return reshape(t, Int(prod(size(t))/χ), χ)
end

# ─── Permutation helpers ─────────────────────────────────────────────────────

permute_fronttail(t::leg3) = permutedims(t, (3,2,1))
permute_fronttail(t::leg4) = permutedims(t, (4,2,3,1))
permute_fronttail(t::InnerProductVec) = RealVec(permute_fronttail(t.vec))
permute_fronttail(t::AbstractZero) = t

# ── Helpers ──────────────────────────────────────────────────────────
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
    getL!(A, L; kwargs...)

Compute the gauge transform `L` from the transfer matrix density.
ρ = L' * L, returns `L` with positive diagonal elements.
"""
function getL!(A, L; kwargs...)
    Ni, Nj = size(A)
    @inbounds for j = 1:Nj, i = 1:Ni
        _, ρ = simple_eig(x -> ρmap(x, A[i, :], j), L[i, j]' * L[i, j]; kwargs...)
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
        Q, R = qrpos!(_to_front(L[i] * _to_tail(A[i])))
        AL[i] = reshape(Q, size(A[i]))
        λ[i] = norm(R)
        Le[i] = rmul!(R, 1 / λ[i])
    end
    return AL, Le, λ
end

function getLsped(Le, A, AL; kwargs...)
    L = similar(Le)
    for i in 1:length(A)
        _, Ls1 = simple_eig(X -> Lmap(X, conj(AL[i]), A[i]), Le[i]; power_iter=5, kwargs...)
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

function ALCtoAC(AL, C)
    AC = Zygote.Buffer(AL)
    @inbounds for i in 1:length(AL)
        AC[i] = ALCtoAC_map(AL[i], C[i])
    end
    return copy(AC)
end

# ── Int FL FR environments ────────────────────────────────────────

function FLint(AL, M::leg4)
    χ = size(AL[1], 1)
    return randSA(M, [(D = size(m, 1); (χ, D, χ)) for m in M.data])
end
function FLint(AL, M::leg5)
    χ = size(AL[1], 1)
    return randSA(M, [(D = size(m, 1); (χ, D, D, χ)) for m in M.data])
end
function FLint(AL, M::leg8)
    χ = size(AL[1], 1)
    return randSA(M, [(D = size(m, 1); (χ, D, D, χ)) for m in M.data])
end

function FRint(AR, M::leg4)
    χ = size(AR[1], 1)
    return randSA(M, [(D = size(m, 3); (χ, D, χ)) for m in M.data])
end
function FRint(AR, M::leg5)
    χ = size(AR[1], 1)
    return randSA(M, [(D = size(m, 3); (χ, D, D, χ)) for m in M.data])
end
function FRint(AR, M::leg8)
    χ = size(AR[1], 1)
    return randSA(M, [(D = size(m, 5); (χ, D, D, χ)) for m in M.data])
end

# ── Fixed-point environments ────────────────────────────────────────

function FLmap(J::Int, FLij, ALui, ALdir, Mi; ifparallel, forloop_iter)
    Nj = length(ALui)
    for j in J:(J + Nj - 1)
        jr = mod1(j, Nj)
        FLij = FLmap_parallel(FLij, ALui[jr], ALdir[jr], Mi[jr]; ifparallel, forloop_iter)
    end
    return FLij
end

"""
    λL, FL = leftenv(ALu, ALd, M, FL=FLint(ALu,M); kwargs...)

Compute the left environment tensor for MPS `ALu`, `ALd` and MPO `M`, by finding the left fixed point
of ALu - M - ALd contracted along the physical dimension.
"""
function leftenv(ALu, ALd, M, FL=FLint(ALu, M); ifobs=false, alg, kwargs...)
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
            f(FLij) = FLmap(1, FLij, ALu[i, :], ALd[ir, :], M[i, :]; ifparallel, forloop_iter)
            if alg.ifsimple_eig
                if ifcheckpoint
                    λLs, FLi1s = checkpoint(simple_eig, f, FL[i, 1]; power_iter)
                else
                    λLs, FLi1s = simple_eig(f, FL[i, 1]; power_iter)
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
                FL′[i, j] = FLmap_parallel(FL′[i, j-1], ALu[i, j-1], ALd[ir, j-1], M[i, j-1]; ifparallel, forloop_iter)
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

function FRmap(J::Int, FRij, ARui, ARdir, Mi; ifparallel, forloop_iter)
    Nj = length(ARui)
    for j in J:-1:(J - Nj + 1)
        jr = mod1(j, Nj)
        FRij = FRmap_parallel(FRij, ARui[jr], ARdir[jr], Mi[jr]; ifparallel, forloop_iter)
    end
    return FRij
end

"""
    λR, FR = rightenv(ARu, ARd, M, FR=FRint(ARu,M); kwargs...)

Compute the right environment tensor for MPS `ARu`, `ARd` and MPO `M`, by finding the right fixed point
of AR - M - conj(AR) contracted along the physical dimension.
"""
function rightenv(ARu, ARd, M, FR=FRint(ARu, M); ifobs=false, alg, kwargs...)
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
            f(FRiNj) = FRmap(Nj, FRiNj, ARu[i, :], ARd[ir, :], M[i, :]; ifparallel, forloop_iter)
            if alg.ifsimple_eig
                if ifcheckpoint
                    λRs, FR1s = checkpoint(simple_eig, f, FR[i, Nj]; power_iter)
                else
                    λRs, FR1s = simple_eig(f, FR[i, Nj]; power_iter)
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
                FR′[i, j] = FRmap_parallel(FR′[i, j+1], ARu[i, j+1], ARd[ir, j+1], M[i, j+1]; ifparallel, forloop_iter)
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
                  ifobs=false, alg, kwargs...)

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
                λLs, Li1s = simple_eig(f, L[i, 1]; power_iter)
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
                   ifobs=false, alg, kwargs...)

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
                λLs, Li1s = simple_eig(f, R[i, Nj]; power_iter)
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

function ACmap(I::Int, ACij, FLj, FRj, Mj; ifparallel, forloop_iter)
    Ni = length(FLj)
    for i in I:(I + Ni - 1)
        ir = mod1(i, Ni)
        ACij = ACmap_parallel(ACij, FLj[ir], FRj[ir], Mj[ir]; ifparallel, forloop_iter)
    end
    return ACij
end

"""
    ACenv(AC, FL, M, FR; kwargs...)

Compute the up environment tensor for MPS `FL`, `FR` and MPO `M`, by finding the up fixed point
of `FL - M - FR` contracted along the physical dimension.
"""
function ACenv(AC, FL, M, FR; alg, kwargs...)
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
            f(AC1j) = ACmap(1, AC1j, FL[:, j], FR[:, j], M[:, j]; ifparallel, forloop_iter)
            if alg.ifsimple_eig
                if ifcheckpoint
                    λACs, ACs = checkpoint(simple_eig, f, AC[1, j]; power_iter)
                else
                    λACs, ACs = simple_eig(f, AC[1, j]; power_iter)
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
                ACij = ACmap_parallel(AC′[i-1, j], FL[i-1, j], FR[i-1, j], M[i-1, j]; ifparallel, forloop_iter)
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
function Cenv(C, FL, FR; alg, kwargs...)
    Ni, Nj = size(C)
    λC = Zygote.Buffer(randSA(Array, C.pattern))
    C′ = Zygote.Buffer(C)
    processed_indices = Set{Int}()
    power_iter = alg.power_iter
    for j in 1:Nj
        jr = mod1(j + 1, Nj)
        p = C.pattern[1, j]
        if p ∉ processed_indices
            f(C1j) = Cmap(1, C1j, FL[:, jr], FR[:, j])
            if alg.ifsimple_eig
                λCs, Cs = simple_eig(f, C[1, j]; power_iter)
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
        QAC, RAC = qrpos(_to_front(AC[i]))
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
        LAC, QAC = lqpos(_to_tail(AC[i, j]))
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

_down_m(m::leg4) = permutedims(m, (1, 4, 3, 2))
_down_m(m::leg5) = permutedims(m, (1, 4, 3, 2, 5))
_down_m(m::leg8) = permutedims(m, (1, 2, 7, 8, 5, 6, 3, 4))

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

# ── Initialization ──────────────────────────────────────────────────

function cellones(A)
    χ = size(A[1], 1)
    return ISA(A, [(χ, χ) for _ in 1:length(A.data)])
end

function initial_A(M::leg4, χ::Int)
    return randSA(M, [(D = size(m, 4); (χ, D, χ)) for m in M.data])
end
function initial_A(M::leg5, χ::Int)
    return randSA(M, [(D = size(m, 4); (χ, D, D, χ)) for m in M.data])
end
function initial_A(M::leg8, χ::Int)
    return randSA(M, [(D = size(m, 7); (χ, D, D, χ)) for m in M.data])
end

"""
    init_VUMPSRuntime(M, alg; χ)

Create a single VUMPSRuntime: canonical forms + fixed-point environments.
"""
function init_VUMPSRuntime(M::StructArray,  χ::Int, alg::VUMPS{:General})
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
    init_env(M, χ, alg)

Initialize one or two `VUMPSRuntime`s (up and optionally down) from an MPO `M`
and bond dimension `χ`.
"""
function init_env(M::StructArray, χ::Int, alg::VUMPS{:General})
    alg.ifparallelupdown && alg.ifparallel && throw(ArgumentError("Parallel up/down only works for two GPUs in one thread. ifparallel = true is supported by MPI-based multi-process parallelism."))
    
    Ni, Nj = size(M)

    if alg.ifupdown && alg.ifparallelupdown
        atype = _arraytype(M)
        @sync begin
            if alg.ifdownfromup
                set_device_id!(atype, 1)
                rtup = init_VUMPSRuntime(M, χ, alg)
                alg.verbosity >= 2 && Zygote.@ignore @info "VUMPS init at device $(get_device(atype)): cell=($(Ni)×$(Nj)) χ = $(χ) up(↑) environment"
                set_device_id!(atype, 2)
                Md = _down_M(atype(M))
                rtdown = _down_init_from_up(atype(rtup), Md)
                alg.verbosity >= 2 && Zygote.@ignore @info "VUMPS init: cell=($(Ni)×$(Nj)) χ = $(χ) down(↓) from up(↑) environment"
            else
                @async begin
                    set_device_id!(atype, 1)
                    rtup = init_VUMPSRuntime(M, χ, alg)
                    alg.verbosity >= 2 && Zygote.@ignore @info "VUMPS init at device $(get_device(atype)): cell=($(Ni)×$(Nj)) χ = $(χ) up(↑) environment"
                end
                @async begin
                    set_device_id!(atype, 2)
                    Md = _down_M(atype(M))
                    rtdown = init_VUMPSRuntime(Md, χ, alg)
                    alg.verbosity >= 2 && Zygote.@ignore @info "VUMPS init at device $(get_device(atype)): cell=($(Ni)×$(Nj)) χ = $(χ) down(↓) environment"
                end
            end
        end
        return rtup, rtdown
    end

    rtup = init_VUMPSRuntime(M, χ, alg)
    alg.verbosity >= 2 && Zygote.@ignore @info "VUMPS init: cell=($(Ni)×$(Nj)) χ = $(χ) up(↑) environment"

    if alg.ifupdown    
        Md = _down_M(M) 
        if alg.ifdownfromup
            rtdown = _down_init_from_up(rtup, Md)
            alg.verbosity >= 2 && Zygote.@ignore @info "VUMPS init: cell=($(Ni)×$(Nj)) χ = $(χ) down(↓) from up(↑) environment"
            return rtup, rtdown
        else
            rtdown = init_VUMPSRuntime(Md, χ, alg)
            alg.verbosity >= 2 && Zygote.@ignore @info "VUMPS init: cell=($(Ni)×$(Nj)) χ = $(χ) down(↓) environment"
            return rtup, rtdown
        end
    else
        return rtup
    end
end

# ── VUMPS step functions ────────────────────────────────────────────

"""
    vumps_step_power(rt, M, alg::VUMPS{:General}{General})

One step of the VUMPS algorithm with the standard (General) contraction mode.
Uses the power-method variant: update environments first, then re-solve AC/C.
"""
function vumps_step_power(rt::VUMPSRuntime, M::StructArray, alg::VUMPS{:General})
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

function vumps_step(rt::VUMPSRuntime, M::StructArray, alg::VUMPS{:General})
    @unpack AL, C, AR, FL, FR = rt
    AC = ALCtoAC(AL,C)
    _, FL =  leftenv(AL, conj(AL), M, FL; alg)
    _, FR = rightenv(AR, conj(AR), M, FR; alg)
    _, AC = ACenv(AC, FL, M, FR; alg)
    _,  C =  Cenv( C, FL, FR; alg)
    AL, AR, errL, errR = ACCtoALAR(AC, C)
    err = errL + errR
    alg.verbosity >= 4 && err > 1e-8 && println("errL=$errL, errR=$errR")
    C = for_gc(C)
    return VUMPSRuntime(AL, AR, C, FL, FR), err
end

# ── VUMPS iteration loop ────────────────────────────────────────────

"""
    vumps_itr(rt, M, alg::VUMPS{:General})

Run the VUMPS iteration loop: first without AD tracking (warm-up), then with AD.
Returns the converged runtime and final error.
"""
function vumps_itr(rt::VUMPSRuntime, M::StructArray, alg::VUMPS{:General})
    t = Zygote.@ignore time()

    atype = _arraytype(M)
    id = get_device_id(atype)
    local err
    Zygote.@ignore alg.verbosity >= 2 && @info "Start VUMPS iteration at $(get_device(atype)) without AD..."
    Zygote.@ignore for i in 1:alg.maxiter
        rt, err = vumps_step(rt, M, alg)
        alg.verbosity >= 3 && i % alg.show_every == 0 && Zygote.@ignore @info @sprintf("VUMPS@step device-%d: %4d\terr = %.3e\ttime = %.3f sec", id, i, err, time()-t)
        if err < alg.tol && i >= alg.miniter
            alg.verbosity >= 2 && Zygote.@ignore @info @sprintf("VUMPS conv@step device-%d: %4d\terr = %.3e\ttime = %.3f sec", id, i, err, time()-t)
            break
        end
        if i == alg.maxiter
            alg.verbosity >= 2 && Zygote.@ignore @warn @sprintf("VUMPS cancel@step device-%d: %4d\terr = %.3e\ttime = %.3f sec", id, i, err, time()-t)
        end
    end

    Zygote.@ignore alg.verbosity >= 2 && @info "Start VUMPS iteration at $(get_device(atype)) with AD..."
    for i in 1:alg.maxiter_ad
        power_iter_backup = alg.power_iter
        alg.power_iter = alg.power_iter_ad
        rt, err = alg.ifcheckpoint ? checkpoint(vumps_step, rt, M, alg) : vumps_step(rt, M, alg)
        alg.power_iter = power_iter_backup
        alg.verbosity >= 3 && i % alg.show_every == 0 && Zygote.@ignore @info @sprintf("VUMPS@step device-%d: %4d\terr = %.3e\ttime = %.3f sec", id, i, err, time()-t)
        if err < alg.tol && i >= alg.miniter_ad
            alg.verbosity >= 2 && Zygote.@ignore @info @sprintf("VUMPS conv@step device-%d: %4d\terr = %.3e\ttime = %.3f sec", id, i, err, time()-t)
            break
        end
        if i == alg.maxiter_ad
            alg.verbosity >= 2 && Zygote.@ignore @warn @sprintf("VUMPS cancel@step device-%d: %4d\terr = %.3e\ttime = %.3f sec", id, i, err, time()-t)
        end
    end

    return rt, err
end

"""
    leading_boundary(rt::VUMPSRuntime, M, alg::VUMPS{:General})

Run the VUMPS boundary contraction for a single (up) environment.
Returns the converged runtime and error.
"""
function leading_boundary(rt::VUMPSRuntime, M::StructArray, alg::VUMPS{:General})
    rt, err = vumps_itr(rt, M, alg)
    return rt, err
end

"""
    leading_boundary(rt::Tuple{VUMPSRuntime, VUMPSRuntime}, M, alg::VUMPS{:General})

Run the VUMPS boundary contraction for both up and down environments.
Returns the converged runtimes and errors.
"""
function leading_boundary(rt::Tuple{VUMPSRuntime, VUMPSRuntime}, M::StructArray, alg::VUMPS{:General})
    rtup, rtdown = rt
    
    if alg.ifupdown && alg.ifparallelupdown
        atype = _arraytype(M)
        @sync begin
            @async begin
                set_device_id!(atype, 1)
                rtup, errup = vumps_itr(rtup, M, alg)
            end
            @async begin
                set_device_id!(atype, 2)
                Md = _down_M(atype(M))
                rtdown, errdown = vumps_itr(rtdown, Md, alg)
            end
        end
        return (rtup, rtdown), (errup, errdown)
    end

    rtup, errup = vumps_itr(rtup, M, alg)

    Md = _down_M(M)
    rtdown, errdown = vumps_itr(rtdown, Md, alg)
    return (rtup, rtdown), (errup, errdown)
end

# ── Observation environment construction ─────────────────────────────

"""
    VUMPSEnv(rt::VUMPSRuntime, M, alg)

Construct a `VUMPSEnv` observation environment from a single VUMPS runtime.
"""
function ObsEnv(rt::VUMPSRuntime, M::StructArray, alg::VUMPS{:General}, Fo=[rt.FL, rt.FR])
    @unpack AL, AR, C, FL, FR = rt
    AC = ALCtoAC(AL, C)
    _, FLo =  leftenv(AL, AL, M, Fo[1]; ifobs = true, alg)
    _, FRo = rightenv(AR, AR, M, Fo[2]; ifobs = true, alg)
    return VUMPSEnv(AC, AR, AC, AR, FL, FR, FLo, FRo)
end

"""
    VUMPSEnv(rt::Tuple{VUMPSRuntime, VUMPSRuntime}, M, alg)

Construct a `VUMPSEnv` observation environment from up and down VUMPS runtimes.
Computes mixed (observation) left and right environments.
"""
function ObsEnv(rt::Tuple{VUMPSRuntime, VUMPSRuntime}, M::StructArray, alg::VUMPS{:General}, Fo=[rt[1].FL, rt[1].FR])
    atype = _arraytype(M)
    set_device_id!(atype, 1)
    rtup, rtdown = rt

    ALu, ARu, Cu, FLu, FRu = rtup.AL, rtup.AR, rtup.C, rtup.FL, rtup.FR
    ACu = ALCtoAC(ALu, Cu)

    ALd, ARd, Cd = rtdown.AL, rtdown.AR, rtdown.C
    ALd, ARd, Cd = map(x->atype_device!(atype, x, 1), [ALd, ARd, Cd]) # transfer device 2 data to 1
    ACd = ALCtoAC(ALd, Cd)

    _, FLo =  leftenv(ALu, ALd, M, Fo[1]; ifobs = true, alg)
    _, FRo = rightenv(ARu, ARd, M, Fo[2]; ifobs = true, alg)
    return VUMPSEnv(ACu, ARu, ACd, ARd, FLu, FRu, FLo, FRo)
end