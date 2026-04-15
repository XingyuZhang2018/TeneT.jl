# ============================================================================
# Plaquette VUMPS — specialized for 2×2 and 2x6 plaquette contractions
#
# Key differences from General VUMPS:
#   - Only left-canonical form (no AR, FR)
#   - ACenv uses FL on both sides (not FL and FR)
#   - Cenv uses FL on both sides: Cmap(C, FL[:,jr], FL[:,jr])
#   - PlaquetteVUMPSRuntime(AL, C, FL) — 3 fields
#   - PlaquetteVUMPSEnv(AL, C, FLu, FLo) — 4 fields for observables
# ============================================================================

# ── Plaquette-specific environment solvers ────────────────────────────
"""
    ACenv_plaq(AC, FL, M; kwargs...)

AC environment for plaquette mode.  Uses `FL` on both sides of the
column transfer matrix (no FR).
"""
function ACenv_plaq(AC, FL, M; alg::VUMPS{L}, kwargs...) where L <: Plaquette
    Ni, Nj = size(M)
    λAC = Zygote.Buffer(randSA(Array, M.pattern))
    AC′ = Zygote.Buffer(AC)
    processed_indices = Set{Int}()
    power_iter = alg.power_iter
    ifparallel = alg.ifparallel
    forloop_iter = alg.forloop_iter
    ifcheckpoint = alg.ifcheckpoint
    for j in 1:Nj
        p = AC.pattern[1,j]
        if L <: Plaquette{Square}
            jr = mod1(j + 1, Nj)
        elseif L <: Plaquette{Honeycomb{:brickwall}}
            jr = mod1(Nj - j , Nj)
        else
            error("Unsupported lattice for Plaquette VUMPS: $(L). Only Square and Honeycomb are supported.")
        end
        
        if p ∉ processed_indices
            f(AC1j) = ifcheckpoint ? checkpoint(ACmap, 1, AC1j, FL[:,j], FL[:,jr], M[:,j]; ifparallel, forloop_iter) : ACmap(1, AC1j, FL[:,j], FL[:,jr], M[:,j]; ifparallel, forloop_iter)
            if alg.ifsimple_eig
                λACs, ACs = simple_eig(f, AC[1,j]; power_iter)
            else
                λACs, ACs, info = eigsolve(f, AC[1,j], 1, :LM; alg_rrule=GMRES(verbosity=-1), maxiter=100,shermitian=false, kwargs...)
                alg.verbosity >= 1 && info.converged == 0 && @warn "ACenv_plaq not converged"
            end
            λAC[1,j], AC′[1,j] = selectpos(λACs, ACs, Ni)
            push!(processed_indices, p)
            length(processed_indices) == length(AC.data) && break
        end
        for i in 2:Ni
            p = AC.pattern[i,j]
            if p ∉ processed_indices
                ACij = ACmap_parallel(AC′[i-1,j], FL[i-1,j], FL[i-1,jr], M[i-1,j]; ifparallel, forloop_iter)
                AC′[i,j] = ACij / norm(ACij)
                λAC[i,j] = λAC[1,j]
                push!(processed_indices, p)
                length(processed_indices) == length(AC.data) && break
            end
        end
    end
    return copy(λAC), copy(AC′)
end

"""
    Cenv_plaq(C, FL; kwargs...)

C environment for plaquette mode.  Uses `FL` on both sides:
`Cmap(C, FL[:,jr], FL[:,jr])`.
"""
function Cenv_plaq(C, FL; alg::VUMPS{L}, kwargs...) where L <: Plaquette
    Ni, Nj = size(C)
    λC = Zygote.Buffer(randSA(Array, C.pattern))
    C′ = Zygote.Buffer(C)
    processed_indices = Set{Int}()
    power_iter = alg.power_iter
    ifcheckpoint = alg.ifcheckpoint
    for j in 1:Nj
        jl = mod1(j + 1, Nj)
        if L <: Plaquette{Square}
            jr = mod1(j + 1, Nj)
        elseif L <: Plaquette{Honeycomb{:brickwall}}
            jr = mod1(Nj - j , Nj)
        else
            error("Unsupported lattice for Plaquette VUMPS: $(L). Only Square and Honeycomb are supported.")
        end
        p = C.pattern[1,j]
        if p ∉ processed_indices
            f(C1j) = ifcheckpoint ? checkpoint(Cmap, 1, C1j, FL[:,jl], FL[:,jr]) : Cmap(1, C1j, FL[:,jl], FL[:,jr])
            if alg.ifsimple_eig
                λCs, Cs = simple_eig(f, C[1,j]; power_iter)
            else
                λCs, Cs, info = eigsolve(f, C[1,j], 1, :LM; alg_rrule=GMRES(verbosity=-1), maxiter=100,ishermitian=false, kwargs...)
                alg.verbosity >= 1 && info.converged == 0 && @warn "Cenv_plaq not converged"
            end
            λC[1,j], C′[1,j] = selectpos(λCs, Cs, Ni)
            push!(processed_indices, p)
            length(processed_indices) == length(C.data) && break
        end
        for i in 2:Ni
            p = C.pattern[i,j]
            if p ∉ processed_indices
                Cij = Cmap(C′[i-1,j], FL[i-1,jl], FL[i-1,jr])
                C′[i,j] = Cij / norm(Cij)
                λC[i,j] = λC[1,j]
                push!(processed_indices, p)
                length(processed_indices) == length(C.data) && break
            end
        end
    end
    return copy(λC), copy(C′)
end

# ACCtoAL is already defined in vumps.jl and shared with plaquette mode.

# ── Plaquette VUMPS step ─────────────────────────────────────────────

"""
    vumps_step(rt::PlaquetteVUMPSRuntime, M, alg::VUMPS{<:Plaquette})

One step of the plaquette VUMPS: leftenv → ACenv → Cenv → ACCtoAL.
Only uses left environments (no right canonical / right environment).
"""
function vumps_step(rt::PlaquetteVUMPSRuntime, M::StructArray, alg::VUMPS{<:Plaquette})
    @unpack AL, C, FL = rt
    AC = ALCtoAC(AL, C)
    _, FL = leftenv(AL, conj(AL), M, FL; alg)
    _, AC = ACenv_plaq(AC, FL, M; alg)
    _, C  = Cenv_plaq(C, FL; alg)
    AL, err = ACCtoAL(AC, C)
    C = for_gc(C)
    return PlaquetteVUMPSRuntime(AL, C, FL), err
end

# ── Plaquette initialization ─────────────────────────────────────────

function init_env(M::StructArray, χ::Int, alg::VUMPS{<:Plaquette})
    size(M.pattern) == (2,2) || size(M.pattern) == (2,6) || error("Plaquette VUMPS only supports 2×2 and 2×6 patterns. Got pattern of size $(size(M.pattern)).")
    A = initial_A(M, χ)
    AL, L, _ = left_canonical(A)
    C = LRtoC(L, L)   # use L on both sides (no right canonical)
    _, FL = leftenv(AL, conj(AL), M; alg)
    return PlaquetteVUMPSRuntime(AL, C, FL)
end

# ── Plaquette iteration + boundary ───────────────────────────────────

function vumps_itr(rt::PlaquetteVUMPSRuntime, M::StructArray, alg::VUMPS{<:Plaquette})
    t = ignore_derivatives(() -> time())
    local err
    ignore_derivatives(() -> alg.verbosity >= 2 && @info "Start Plaquette VUMPS iteration without AD...")
    ignore_derivatives() do
        for i in 1:alg.maxiter
        rt, err = vumps_step(rt, M, alg)
        alg.verbosity >= 3 && i % alg.show_every == 0 &&
            ignore_derivatives(() -> @info @sprintf("PlaqVUMPS@step: %4d\terr = %.3e\ttime = %.3f sec", i, err, time()-t))
        if err < alg.tol && i >= alg.miniter
            alg.verbosity >= 2 &&
                ignore_derivatives(() -> @info @sprintf("PlaqVUMPS conv@step: %4d\terr = %.3e\ttime = %.3f sec", i, err, time()-t))
            break
        end
        if i == alg.maxiter
            alg.verbosity >= 2 && ignore_derivatives(() -> @warn @sprintf("PlaqVUMPS cancel@step: %4d\terr = %.3e\ttime = %.3f sec", i, err, time()-t))
        end
    end
    end

    ignore_derivatives(() -> alg.verbosity >= 2 && @info "Start Plaquette VUMPS iteration with AD...")
    alg_ad = deepcopy(alg)
    alg_ad.power_iter = alg.power_iter_ad
    for i in 1:alg.maxiter_ad
        rt, err = alg.ifcheckpoint ? checkpoint(vumps_step, rt, M, alg_ad) : vumps_step(rt, M, alg_ad)
        alg.verbosity >= 3 && i % alg.show_every == 0 && ignore_derivatives(() -> @info @sprintf("PlaqVUMPS@step: %4d\terr = %.3e\ttime = %.3f sec", i, err, time()-t))
        if err < alg.tol && i >= alg.miniter_ad
            alg.verbosity >= 2 && ignore_derivatives(() -> @info @sprintf("PlaqVUMPS conv@step: %4d\terr = %.3e\ttime = %.3f sec", i, err, time()-t))
            break
        end
        if i == alg.maxiter_ad
            alg.verbosity >= 2 && ignore_derivatives(() -> @warn @sprintf("PlaqVUMPS cancel@step: %4d\terr = %.3e\ttime = %.3f sec", i, err, time()-t))
        end
    end
    return rt, err
end

function leading_boundary(rt::PlaquetteVUMPSRuntime, M::StructArray, alg::VUMPS{<:Plaquette})
    return vumps_itr(rt, M, alg)
end

# ── Plaquette observation environment ────────────────────────────────
"""
    PlaquetteVUMPSEnv(rt::PlaquetteVUMPSRuntime, M, alg)

Construct a `PlaquetteVUMPSEnv` from a plaquette runtime.
Computes the observation left environment `FLo` using `ifobs=true`.
"""
function ObsEnv(rt::PlaquetteVUMPSRuntime, M::StructArray, alg::VUMPS{<:Plaquette})
    @unpack AL, C, FL = rt
    _, FLo = leftenv(AL, AL, M, FL; ifobs=true, alg)
    return PlaquetteVUMPSEnv(AL, C, FL, FLo)
end