function leftenv_c4v(ALu, ALd, M, FL; alg, kwargs...)
    @unpack power_iter, ifparallel, forloop_iter, ifcheckpoint, inner_etype, simple_eig_polish_steps = alg
    f(FL) = ifcheckpoint ? checkpoint(FLmap_parallel, FL, ALu, ALd, M; ifparallel, forloop_iter, inner_etype) : FLmap_parallel(FL, ALu, ALd, M; ifparallel, forloop_iter, inner_etype)
    # Fine polish: last `simple_eig_polish_steps` power iters use Float64 (inner_etype=nothing)
    polish_fine = inner_etype !== nothing && simple_eig_polish_steps > 0
    f_polish(FL) = ifcheckpoint ? checkpoint(FLmap_parallel, FL, ALu, ALd, M; ifparallel, forloop_iter, inner_etype=nothing) : FLmap_parallel(FL, ALu, ALd, M; ifparallel, forloop_iter, inner_etype=nothing)
    if alg.ifsimple_eig
        λFLs, FLs = polish_fine ?
            simple_eig(f, FL; power_iter, f_final=f_polish, final_polish_steps=simple_eig_polish_steps) :
            simple_eig(f, FL; power_iter)
    else
        λFLs, FLs, info = eigsolve(f, FL, 1, :LM; alg_rrule=GMRES(verbosity=-1), maxiter=100,shermitian=false, kwargs...)
        alg.verbosity >= 1 && info.converged == 0 && @warn "FLenv_c4v not converged"
    end

    return λFLs[1], FLs[1]
end

function ACenv_c4v(AC, FL, M; alg, kwargs...)
    @unpack power_iter, ifparallel, forloop_iter, ifcheckpoint, inner_etype, simple_eig_polish_steps = alg
    f(AC) = ifcheckpoint ? checkpoint(ACmap_parallel, AC, FL, FL, M; ifparallel, forloop_iter, inner_etype) : ACmap_parallel(AC, FL, FL, M; ifparallel, forloop_iter, inner_etype)
    polish_fine = inner_etype !== nothing && simple_eig_polish_steps > 0
    f_polish(AC) = ifcheckpoint ? checkpoint(ACmap_parallel, AC, FL, FL, M; ifparallel, forloop_iter, inner_etype=nothing) : ACmap_parallel(AC, FL, FL, M; ifparallel, forloop_iter, inner_etype=nothing)
    if alg.ifsimple_eig
        λACs, ACs = polish_fine ?
            simple_eig(f, AC; power_iter, f_final=f_polish, final_polish_steps=simple_eig_polish_steps) :
            simple_eig(f, AC; power_iter)
    else
        λACs, ACs, info = eigsolve(f, AC, 1, :LM; alg_rrule=GMRES(verbosity=-1), maxiter=100,shermitian=false, kwargs...)
        alg.verbosity >= 1 && info.converged == 0 && @warn "ACenv_c4v not converged"
    end

    return λACs[1], ACs[1]
end

function Cenv_c4v(C, FL; alg, kwargs...)
    @unpack power_iter, ifcheckpoint = alg
    f(C) = ifcheckpoint ? checkpoint(Cmap, C, FL, FL) : Cmap(C, FL, FL)
    if alg.ifsimple_eig
        λCs, Cs = simple_eig(f, C; power_iter)
    else
        λCs, Cs, info = eigsolve(f,C, 1, :LM; alg_rrule=GMRES(verbosity=-1), maxiter=100,ishermitian=false, kwargs...)
        alg.verbosity >= 1 && info.converged == 0 && @warn "Cenv_plaq not converged"
    end

    return λCs[1], Cs[1]
end


# ── initialization ─────────────────────────────────────────

function init_env(M::StructArray, χ::Int, alg::VUMPS{C4v})
    M = M[1][:,:,:,:,:,1]
    D = size(M, 1)  
    if M isa leg4
        FL = rand!(similar(M,χ,D,χ))
        FL += conj(permutedims(FL, (3,2,1)))
    else
        FL = rand!(similar(M,χ,D,D,χ))
        FL += conj(permutedims(FL, (4,2,3,1)))
    end
    AL, C = qr(_to_front(FL))
    AL = reshape(_arraytype(FL)(AL), size(FL))

    return C4vVUMPSEnv(AL, C, FL)
end

"""
    vumps_step(rt::C4vVUMPSEnv, M, alg::VUMPS{C4v})

One step of the plaquette VUMPS: leftenv → ACenv → Cenv → ACCtoAL.
Only uses left environments (no right canonical / right environment).
"""
function vumps_step(rt::C4vVUMPSEnv, M::AbstractArray, alg::VUMPS{C4v})
    @unpack AL, C, FL = rt
    AC = ALCtoAC_map(AL, C)
    _, FL = leftenv_c4v(AL, conj(AL), M, FL; alg)
    _, AC = ACenv_c4v(AC, FL, M; alg)
    _, C  = Cenv_c4v(C, FL; alg)

    QAC, RAC = qrpos(_to_front(AC))
    QC, RC = qrpos(C)
    AL = reshape(QAC*QC', size(AC))
    err = ignore_derivatives(() -> norm(RAC - RC))

    return C4vVUMPSEnv(AL, C, FL), err
end

# ── Plaquette iteration + boundary ───────────────────────────────────

function leading_boundary(rt::C4vVUMPSEnv, M::StructArray, alg::VUMPS{C4v})
    t = ignore_derivatives(() -> time())
    M = M[1]
    local err

    # Whole-VUMPS precision mode (alternative to `inner_etype`):
    # pre-cast rt and M to alg.whole_vumps_etype; run whole VUMPS (FLmap, QR,
    # norm, eigsolve, ...) in that precision; polish iters cast back to original.
    T_orig = eltype(rt.AL)
    want_whole = alg.whole_vumps_etype !== nothing && alg.whole_vumps_etype != real(T_orig)
    if want_whole
        W = alg.whole_vumps_etype
        rt = C4vVUMPSEnv(_downcast_eltype(W, rt.AL),
                         _downcast_eltype(W, rt.C),
                         _downcast_eltype(W, rt.FL))
        M  = _downcast_eltype(W, M)
    end
    # For whole-VUMPS mode we pass alg with inner_etype=nothing (FLmap should
    # run natively on the already-downcasted tensors, no per-call conversion).
    alg_wholemode = alg
    if want_whole
        alg_wholemode = deepcopy(alg)
        alg_wholemode.inner_etype = nothing
        alg_wholemode.simple_eig_polish_steps = 0
    end

    ignore_derivatives(() -> alg.verbosity >= 2 && @info "Start C4v VUMPS iteration without AD...")
    ignore_derivatives() do
        for i in 1:alg.maxiter
        rt, err = vumps_step(rt, M, alg_wholemode)
        alg.verbosity >= 3 && i % alg.show_every == 0 &&
            ignore_derivatives(() -> @info @sprintf("C4vVUMPS@step: %4d\terr = %.3e\ttime = %.3f sec", i, err, time()-t))
        if err < alg.tol && i >= alg.miniter
            alg.verbosity >= 2 &&
                ignore_derivatives(() -> @info @sprintf("C4vVUMPS conv@step: %4d\terr = %.3e\ttime = %.3f sec", i, err, time()-t))
            break
        end
        if i == alg.maxiter
            alg.verbosity >= 2 && ignore_derivatives(() -> @warn @sprintf("C4vVUMPS cancel@step: %4d\terr = %.3e\ttime = %.3f sec", i, err, time()-t))
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
        if want_whole && in_polish && eltype(rt.AL) != T_orig
            rt = C4vVUMPSEnv(_downcast_eltype(real(T_orig), rt.AL),
                             _downcast_eltype(real(T_orig), rt.C),
                             _downcast_eltype(real(T_orig), rt.FL))
            M = _downcast_eltype(real(T_orig), M)
        end
        rt, err = alg.ifcheckpoint ? checkpoint(vumps_step, rt, M, alg_this_iter) : vumps_step(rt, M, alg_this_iter)
        alg.verbosity >= 3 && i % alg.show_every == 0 && ignore_derivatives(() -> @info @sprintf("PlaqVUMPS@step: %4d\terr = %.3e\ttime = %.3f sec", i, err, time()-t))
        if err < alg.tol && i >= alg.miniter_ad
            alg.verbosity >= 2 && ignore_derivatives(() -> @info @sprintf("C4vVUMPS conv@step: %4d\terr = %.3e\ttime = %.3f sec", i, err, time()-t))
            break
        end
        if i == alg.maxiter_ad
            alg.verbosity >= 2 && ignore_derivatives(() -> @warn @sprintf("C4vVUMPS cancel@step: %4d\terr = %.3e\ttime = %.3f sec", i, err, time()-t))
        end
    end
    # Exit guard: if whole-VUMPS mode is active and we never hit polish (e.g.
    # inner_etype_final_steps==0 or early break before polish started), cast
    # rt back to original precision so downstream AD flows correctly.
    if want_whole && eltype(rt.AL) != T_orig
        rt = C4vVUMPSEnv(_downcast_eltype(real(T_orig), rt.AL),
                         _downcast_eltype(real(T_orig), rt.C),
                         _downcast_eltype(real(T_orig), rt.FL))
    end
    return rt, err
end

ObsEnv(rt::C4vVUMPSEnv, M::StructArray, ::VUMPS{C4v}) = rt
