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

# ── Oneside left/right observation envs ──────────────────────────────

"""
    leftenv_oneside(AL, M, FL=FLint(AL, M); alg::VUMPS{Oneside{Model}}, kwargs...) where Model

Left observation environment for Oneside mode. Row pairing is determined by
the model trait `_oneside_down_index(Model, i, Ni)` instead of the hardcoded
`ir = Ni+1-i` used by `leftenv(...; ifobs=true)`. AL is used twice (no
separate ALd) because under U-D hermiticity ALu == ALd at the corresponding
row (where "corresponding" is defined by the trait).

Body mirrors `leftenv` exactly except for the `ir` computation.
"""
function leftenv_oneside(AL, M, FL=FLint(AL, M);
                         alg::VUMPS{Oneside{Model}}, kwargs...) where Model
    @unpack inner_etype, forloop_iter, ifparallel,
            segment_checkpoint, inner_checkpoint, eig_checkpoint, ifsimple_eig, verbosity = alg
    # Env-level boundary cast (mirror leftenv)
    T_orig = AL isa StructArray ? eltype(AL.data[1]) : eltype(AL)
    do_env_cast = inner_etype !== nothing && inner_etype != real(T_orig)
    if do_env_cast
        AL = _downcast_eltype(inner_etype, AL)
        M  = _downcast_eltype(inner_etype, M)
        FL = _downcast_eltype(inner_etype, FL)
    end
    inner_etype_pass = do_env_cast ? nothing : inner_etype

    λL = Zygote.Buffer(randSA(Array, M.pattern))
    FL′ = Zygote.Buffer(FL)
    Ni, Nj = size(M)
    processed_indices = Set{Int}()
    power_iter = alg.power_iter_obs
    _assert_inner_method(inner_checkpoint)
    simple_eig_polish_steps = do_env_cast ? 0 : alg.simple_eig_polish_steps
    polish_fine = inner_etype_pass !== nothing && simple_eig_polish_steps > 0
    for i in 1:Ni
        ir = _oneside_down_index(Model, i, Ni)   # ← key difference from leftenv
        p = FL.pattern[i, 1]
        if p ∉ processed_indices
            f(FLij) = checkpoint(inner_checkpoint, FLmap, 1, FLij,
                                 AL[i, :], AL[ir, :], M[i, :];
                                 ifparallel, forloop_iter, inner_etype=inner_etype_pass)
            if ifsimple_eig
                λLs, FLi1s = checkpoint(eig_checkpoint, _simple_eig_FLmap,
                                         FL[i, 1], AL[i, :], AL[ir, :], M[i, :];
                                         power_iter, ifparallel, forloop_iter,
                                         inner_checkpoint,
                                         inner_etype=inner_etype_pass,
                                         segment_checkpoint,
                                         final_polish_steps = polish_fine ? simple_eig_polish_steps : 0)
            else
                λLs, FLi1s, info = eigsolve(f, FL[i, 1], 1, :LM;
                                            alg_rrule=GMRES(verbosity=-1), maxiter=100, ishermitian=false, kwargs...)
                verbosity >= 1 && info.converged == 0 && @warn "leftenv_oneside not converged"
            end
            λL[i, 1], FL′[i, 1] = selectpos(λLs, FLi1s, Nj)
            push!(processed_indices, p)
            length(processed_indices) == length(FL.data) && break
        end
        for j in 2:Nj
            p = FL.pattern[i, j]
            if p ∉ processed_indices
                FL′[i, j] = FLmap_parallel(FL′[i, j-1], AL[i, j-1], AL[ir, j-1], M[i, j-1];
                                            ifparallel, forloop_iter, inner_etype=inner_etype_pass)
                λL[i, j] = λL[i, 1]
                push!(processed_indices, p)
                length(processed_indices) == length(FL.data) && break
            end
        end
    end

    if do_env_cast
        return copy(λL), _downcast_eltype(real(T_orig), copy(FL′))
    end
    return copy(λL), copy(FL′)
end

"""
    rightenv_oneside(AR, M, FR=FRint(AR, M); alg::VUMPS{Oneside{Model}}, kwargs...) where Model

Right observation environment for Oneside mode. See `leftenv_oneside`.
"""
function rightenv_oneside(AR, M, FR=FRint(AR, M);
                          alg::VUMPS{Oneside{Model}}, kwargs...) where Model
    @unpack inner_etype, forloop_iter, ifparallel,
            segment_checkpoint, inner_checkpoint, eig_checkpoint, ifsimple_eig, verbosity = alg
    T_orig = AR isa StructArray ? eltype(AR.data[1]) : eltype(AR)
    do_env_cast = inner_etype !== nothing && inner_etype != real(T_orig)
    if do_env_cast
        AR = _downcast_eltype(inner_etype, AR)
        M  = _downcast_eltype(inner_etype, M)
        FR = _downcast_eltype(inner_etype, FR)
    end
    inner_etype_pass = do_env_cast ? nothing : inner_etype

    Ni, Nj = size(M)
    λR = Zygote.Buffer(randSA(Array, M.pattern))
    FR′ = Zygote.Buffer(FR)
    processed_indices = Set{Int}()
    power_iter = alg.power_iter_obs
    _assert_inner_method(inner_checkpoint)
    simple_eig_polish_steps = do_env_cast ? 0 : alg.simple_eig_polish_steps
    polish_fine = inner_etype_pass !== nothing && simple_eig_polish_steps > 0
    for i in 1:Ni
        ir = _oneside_down_index(Model, i, Ni)
        p = FR.pattern[i, Nj]
        if p ∉ processed_indices
            f(FRiNj) = checkpoint(inner_checkpoint, FRmap, Nj, FRiNj,
                                  AR[i, :], AR[ir, :], M[i, :];
                                  ifparallel, forloop_iter, inner_etype=inner_etype_pass)
            if ifsimple_eig
                λRs, FR1s = checkpoint(eig_checkpoint, _simple_eig_FRmap,
                                        FR[i, Nj], AR[i, :], AR[ir, :], M[i, :], Nj;
                                        power_iter, ifparallel, forloop_iter,
                                        inner_checkpoint,
                                        inner_etype=inner_etype_pass,
                                        segment_checkpoint,
                                        final_polish_steps = polish_fine ? simple_eig_polish_steps : 0)
            else
                λRs, FR1s, info = eigsolve(f, FR[i, Nj], 1, :LM;
                                           alg_rrule=GMRES(verbosity=-1), maxiter=100, ishermitian=false, kwargs...)
                verbosity >= 1 && info.converged == 0 && @warn "rightenv_oneside not converged"
            end
            λR[i, Nj], FR′[i, Nj] = selectpos(λRs, FR1s, Nj)
            push!(processed_indices, p)
            length(processed_indices) == length(FR.data) && break
        end
        for j in Nj-1:-1:1
            p = FR.pattern[i, j]
            if p ∉ processed_indices
                FR′[i, j] = FRmap_parallel(FR′[i, j+1], AR[i, j+1], AR[ir, j+1], M[i, j+1];
                                            ifparallel, forloop_iter, inner_etype=inner_etype_pass)
                λR[i, j] = λR[i, Nj]
                push!(processed_indices, p)
                length(processed_indices) == length(FR.data) && break
            end
        end
    end

    if do_env_cast
        return copy(λR), _downcast_eltype(real(T_orig), copy(FR′))
    end
    return copy(λR), copy(FR′)
end
