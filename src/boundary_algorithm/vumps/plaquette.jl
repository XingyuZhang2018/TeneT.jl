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
    inner_etype = alg.inner_etype
    simple_eig_polish_steps = alg.simple_eig_polish_steps
    # Fine polish: last `simple_eig_polish_steps` power iters use Float64 (inner_etype=nothing)
    polish_fine = inner_etype !== nothing && simple_eig_polish_steps > 0
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
            f(AC1j) = ifcheckpoint ? checkpoint(ACmap, 1, AC1j, FL[:,j], FL[:,jr], M[:,j]; ifparallel, forloop_iter, inner_etype) : ACmap(1, AC1j, FL[:,j], FL[:,jr], M[:,j]; ifparallel, forloop_iter, inner_etype)
            f_polish(AC1j) = ifcheckpoint ? checkpoint(ACmap, 1, AC1j, FL[:,j], FL[:,jr], M[:,j]; ifparallel, forloop_iter, inner_etype=nothing) : ACmap(1, AC1j, FL[:,j], FL[:,jr], M[:,j]; ifparallel, forloop_iter, inner_etype=nothing)
            if alg.ifsimple_eig
                λACs, ACs = polish_fine ?
                    simple_eig(f, AC[1,j]; power_iter, f_final=f_polish, final_polish_steps=simple_eig_polish_steps) :
                    simple_eig(f, AC[1,j]; power_iter)
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
                ACij = ACmap_parallel(AC′[i-1,j], FL[i-1,j], FL[i-1,jr], M[i-1,j]; ifparallel, forloop_iter, inner_etype)
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

    # Whole-VUMPS precision mode (alternative to `inner_etype`):
    # pre-cast rt and M to alg.whole_vumps_etype; run whole VUMPS (FLmap, QR,
    # norm, eigsolve, ...) in that precision; polish iters cast back to original.
    # Original eltype is taken from the first data array of rt.AL (StructArrays
    # have eltype = Any, so inspect the underlying data tensor).
    T_orig = eltype(rt.AL.data[1])
    want_whole = alg.whole_vumps_etype !== nothing && alg.whole_vumps_etype != real(T_orig)
    if want_whole
        W = alg.whole_vumps_etype
        rt = PlaquetteVUMPSRuntime(_downcast_eltype(W, rt.AL),
                                   _downcast_eltype(W, rt.C),
                                   _downcast_eltype(W, rt.FL))
        M  = _downcast_eltype(W, M)
    end
    # For whole-VUMPS mode we pass alg with inner_etype=nothing (FLmap etc.
    # should run natively on the already-downcasted tensors, no per-call conversion).
    alg_wholemode = alg
    if want_whole
        alg_wholemode = deepcopy(alg)
        alg_wholemode.inner_etype = nothing
        alg_wholemode.simple_eig_polish_steps = 0
    end

    ignore_derivatives(() -> alg.verbosity >= 2 && @info "Start Plaquette VUMPS iteration without AD...")
    ignore_derivatives() do
        for i in 1:alg.maxiter
        rt, err = vumps_step(rt, M, alg_wholemode)
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
    alg_ad = deepcopy(alg_wholemode)
    alg_ad.power_iter = alg.power_iter_ad
    alg_ad.simple_eig_polish_steps = 0   # fine polish fires only on the LAST AD iter via alg_ad_fine
    # Mixed-precision polish variants (activate only when mixed-precision mode is on):
    #  - Coarse: final N AD iters run in ORIGINAL precision (Float64).
    #  - Fine:   only the LAST AD iter sets simple_eig_polish_steps > 0.
    # If both set, coarse takes precedence.
    alg_ad_coarse = deepcopy(alg_ad)
    alg_ad_coarse.inner_etype = nothing
    alg_ad_fine = deepcopy(alg_ad)
    alg_ad_fine.simple_eig_polish_steps = alg.simple_eig_polish_steps
    # Is ANY mixed-precision mode active that the polish machinery should react to?
    mixed_active = (alg.inner_etype !== nothing) || want_whole
    for i in 1:alg.maxiter_ad
        alg_this_iter = alg_ad
        in_polish = alg.inner_etype_final_steps > 0 &&
                    i > alg.maxiter_ad - alg.inner_etype_final_steps
        if mixed_active
            if in_polish
                alg_this_iter = alg_ad_coarse
            elseif alg.simple_eig_polish_steps > 0 && i == alg.maxiter_ad
                alg_this_iter = alg_ad_fine
            end
        end
        # For whole-VUMPS mode: on the FIRST polish iter, cast rt and M back to T_orig.
        if want_whole && in_polish && eltype(rt.AL.data[1]) != T_orig
            rt = PlaquetteVUMPSRuntime(_downcast_eltype(real(T_orig), rt.AL),
                                       _downcast_eltype(real(T_orig), rt.C),
                                       _downcast_eltype(real(T_orig), rt.FL))
            M = _downcast_eltype(real(T_orig), M)
        end
        rt, err = alg.ifcheckpoint ? checkpoint(vumps_step, rt, M, alg_this_iter) : vumps_step(rt, M, alg_this_iter)
        alg.verbosity >= 3 && i % alg.show_every == 0 && ignore_derivatives(() -> @info @sprintf("PlaqVUMPS@step: %4d\terr = %.3e\ttime = %.3f sec", i, err, time()-t))
        if err < alg.tol && i >= alg.miniter_ad
            alg.verbosity >= 2 && ignore_derivatives(() -> @info @sprintf("PlaqVUMPS conv@step: %4d\terr = %.3e\ttime = %.3f sec", i, err, time()-t))
            break
        end
        if i == alg.maxiter_ad
            alg.verbosity >= 2 && ignore_derivatives(() -> @warn @sprintf("PlaqVUMPS cancel@step: %4d\terr = %.3e\ttime = %.3f sec", i, err, time()-t))
        end
    end
    # Exit guard: if whole-VUMPS mode is active and we never hit polish (e.g.
    # inner_etype_final_steps==0 or early break before polish started), cast
    # rt back to original precision so downstream AD flows correctly.
    if want_whole && eltype(rt.AL.data[1]) != T_orig
        rt = PlaquetteVUMPSRuntime(_downcast_eltype(real(T_orig), rt.AL),
                                   _downcast_eltype(real(T_orig), rt.C),
                                   _downcast_eltype(real(T_orig), rt.FL))
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