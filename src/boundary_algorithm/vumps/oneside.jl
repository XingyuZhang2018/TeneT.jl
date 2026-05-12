# ============================================================================
# Oneside VUMPS — for models with up-down (but NOT left-right) hermiticity
#
# Key differences from General VUMPS:
#   - Only ONE VUMPS iteration (no separate down VUMPS); down env derived via
#     model trait `_oneside_down_index(::Type{<:Model}, i, Ni)`
#   - ObsEnv builds FLo/FRo via leftenv_oneside / rightenv_oneside (which
#     use the model trait, not hardcoded `ir = Ni+1-i` of `ifobs=true`)
#   - Reuses VUMPSRuntime (5 fields); introduces OnesideVUMPSEnv (6 fields)
#
# Model opt-in: implicit via existence of `energy_value(::Model, A,
# env::OnesideVUMPSEnv, params)`. No explicit trait.
# ============================================================================

# Note: vumps_step, vumps_itr, leftenv_oneside, rightenv_oneside, ObsEnv,
# imag_error all live in this file (added in subsequent Stage 2 tasks).

# ── Initialization ──────────────────────────────────────────────────

"""
    init_env(M, χ, alg::VUMPS{<:Oneside})

Initialize a single VUMPSRuntime for Oneside mode. No down VUMPS is created
since the down env is derived from the up env via `_oneside_down_index` at
ObsEnv time.
"""
function init_env(M::StructArray, χ::Int, alg::VUMPS{<:Oneside})
    alg.ifparallelupdown && error("Oneside is incompatible with ifparallelupdown=true")
    alg.ifupdown && @warn "Oneside ignores ifupdown=true; running single-side VUMPS only" maxlog=1
    Ni, Nj = size(M)
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
    alg.verbosity >= 2 && ChainRulesCore.ignore_derivatives(() ->
        @info "Oneside VUMPS init: cell=($(Ni)×$(Nj)) χ = $(χ)")
    return VUMPSRuntime(AL, AR, C, FL, FR)
end

# ── leading_boundary entry ───────────────────────────────────────────

"""
    leading_boundary(rt::VUMPSRuntime, M, alg::VUMPS{<:Oneside})

Run the Oneside VUMPS iteration. Returns the converged runtime and the
final error. Note: `vumps_itr` for Oneside is added in Task 6 — this entry
will MethodError until then. Stage 2 Task 6 lands the iteration body.
"""
function leading_boundary(rt::VUMPSRuntime, M::StructArray, alg::VUMPS{<:Oneside})
    return vumps_itr(rt, M, alg)
end

# ── Oneside VUMPS step ───────────────────────────────────────────────

"""
    vumps_step(rt::VUMPSRuntime, M, alg::VUMPS{<:Oneside})

One step of Oneside VUMPS: leftenv → rightenv → ACenv → Cenv → ACCtoALAR.
Structurally identical to `vumps_step(...; alg::VUMPS{General})` but without
the separate down-VUMPS half — the down env is derived from up at ObsEnv time
via the model trait `_oneside_down_index`.
"""
function vumps_step(rt::VUMPSRuntime, M::StructArray, alg::VUMPS{<:Oneside})
    @unpack AL, C, AR, FL, FR = rt
    sub = alg.subop_checkpoint
    AC = ALCtoAC(AL, C)
    _, FL = checkpoint(sub, (a, b, m, fl) -> leftenv(a, b, m, fl; alg), AL, conj(AL), M, FL)
    _, FR = checkpoint(sub, (a, b, m, fr) -> rightenv(a, b, m, fr; alg), AR, conj(AR), M, FR)
    _, AC = checkpoint(sub, (ac, fl, m, fr) -> ACenv(ac, fl, m, fr; alg), AC, FL, M, FR)
    _, C  = Cenv(C, FL, FR; alg)
    AL, AR, errL, errR = checkpoint(sub, ACCtoALAR, AC, C)
    err = errL + errR
    alg.verbosity >= 4 && err > 1e-8 && println("errL=$errL, errR=$errR")
    C = for_gc(C)
    return VUMPSRuntime(AL, AR, C, FL, FR), err
end

# ── Oneside VUMPS iteration ──────────────────────────────────────────

"""
    vumps_itr(rt, M, alg::VUMPS{<:Oneside})

Run the Oneside VUMPS iteration: first without AD (warm-up via `maxiter`),
then with AD (`maxiter_ad`). Mirrors the General `vumps_itr` body verbatim
including mixed-precision / polish / whole_vumps_etype plumbing; only the
dispatch type and log labels differ.
"""
function vumps_itr(rt::VUMPSRuntime, M::StructArray, alg::VUMPS{<:Oneside})
    t = ChainRulesCore.ignore_derivatives(() -> time())
    atype = _arraytype(M)
    id = get_device_id(atype)
    local err

    # Whole-VUMPS precision mode (same as General).
    T_orig = eltype(rt.AL.data[1])
    want_whole = alg.whole_vumps_etype !== nothing && alg.whole_vumps_etype != real(T_orig)
    if want_whole
        W = alg.whole_vumps_etype
        rt = VUMPSRuntime(_downcast_eltype(W, rt.AL),
                          _downcast_eltype(W, rt.AR),
                          _downcast_eltype(W, rt.C),
                          _downcast_eltype(W, rt.FL),
                          _downcast_eltype(W, rt.FR))
        M  = _downcast_eltype(W, M)
    end
    alg_wholemode = alg
    if want_whole
        alg_wholemode = deepcopy(alg)
        alg_wholemode.inner_etype = nothing
        alg_wholemode.simple_eig_polish_steps = 0
    end

    ChainRulesCore.ignore_derivatives(() -> alg.verbosity >= 2 && @info "Start Oneside VUMPS iteration at $(get_device(atype)) without AD...")
    ChainRulesCore.ignore_derivatives() do
        for i in 1:alg.maxiter
            rt, err = vumps_step(rt, M, alg_wholemode)
            alg.verbosity >= 3 && i % alg.show_every == 0 &&
                ChainRulesCore.ignore_derivatives(() -> @info @sprintf("OnesideVUMPS@step device-%d: %4d\terr = %.3e\ttime = %.3f sec", id, i, err, time()-t))
            if err < alg.tol && i >= alg.miniter
                alg.verbosity >= 2 &&
                    ChainRulesCore.ignore_derivatives(() -> @info @sprintf("OnesideVUMPS conv@step device-%d: %4d\terr = %.3e\ttime = %.3f sec", id, i, err, time()-t))
                break
            end
            if i == alg.maxiter
                alg.verbosity >= 2 &&
                    ChainRulesCore.ignore_derivatives(() -> @warn @sprintf("OnesideVUMPS cancel@step device-%d: %4d\terr = %.3e\ttime = %.3f sec", id, i, err, time()-t))
            end
        end
    end

    ChainRulesCore.ignore_derivatives(() -> alg.verbosity >= 2 && @info "Start Oneside VUMPS iteration at $(get_device(atype)) with AD...")
    alg_ad = deepcopy(alg_wholemode)
    alg_ad.power_iter = alg.power_iter_ad
    alg_ad.simple_eig_polish_steps = 0
    alg_ad_coarse = deepcopy(alg_ad); alg_ad_coarse.inner_etype = nothing
    alg_ad_fine = deepcopy(alg_ad); alg_ad_fine.simple_eig_polish_steps = alg.simple_eig_polish_steps
    mixed_active = (alg.inner_etype !== nothing) || want_whole
    for i in 1:alg.maxiter_ad
        alg_this_iter = alg_ad
        in_polish = alg.inner_etype_final_steps > 0 && i > alg.maxiter_ad - alg.inner_etype_final_steps
        if mixed_active
            if in_polish
                alg_this_iter = alg_ad_coarse
            elseif alg.simple_eig_polish_steps > 0 && i == alg.maxiter_ad
                alg_this_iter = alg_ad_fine
            end
        end
        if want_whole && in_polish && eltype(rt.AL.data[1]) != T_orig
            rt = VUMPSRuntime(_downcast_eltype(real(T_orig), rt.AL),
                              _downcast_eltype(real(T_orig), rt.AR),
                              _downcast_eltype(real(T_orig), rt.C),
                              _downcast_eltype(real(T_orig), rt.FL),
                              _downcast_eltype(real(T_orig), rt.FR))
            M = _downcast_eltype(real(T_orig), M)
        end
        rt, err = checkpoint(alg.step_checkpoint, vumps_step, rt, M, alg_this_iter)
        alg.verbosity >= 3 && i % alg.show_every == 0 &&
            ChainRulesCore.ignore_derivatives(() -> @info @sprintf("OnesideVUMPS@step device-%d: %4d\terr = %.3e\ttime = %.3f sec", id, i, err, time()-t))
        if err < alg.tol && i >= alg.miniter_ad
            alg.verbosity >= 2 &&
                ChainRulesCore.ignore_derivatives(() -> @info @sprintf("OnesideVUMPS conv@step device-%d: %4d\terr = %.3e\ttime = %.3f sec", id, i, err, time()-t))
            break
        end
        if i == alg.maxiter_ad
            alg.verbosity >= 2 &&
                ChainRulesCore.ignore_derivatives(() -> @warn @sprintf("OnesideVUMPS cancel@step device-%d: %4d\terr = %.3e\ttime = %.3f sec", id, i, err, time()-t))
        end
    end
    if want_whole && eltype(rt.AL.data[1]) != T_orig
        rt = VUMPSRuntime(_downcast_eltype(real(T_orig), rt.AL),
                          _downcast_eltype(real(T_orig), rt.AR),
                          _downcast_eltype(real(T_orig), rt.C),
                          _downcast_eltype(real(T_orig), rt.FL),
                          _downcast_eltype(real(T_orig), rt.FR))
    end
    return rt, err
end
