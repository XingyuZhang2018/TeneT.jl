# QRCTM boundary algorithm
#
# QR-based Corner Transfer Matrix method.
# Uses QR decomposition of the combined C*T tensor to obtain the projector U,
# then applies a single transfer-matrix step followed by a QR-based corner update.

# ── initialization ─────────────────────────────────────────

# function init_env(M::StructArray, χ::Int, alg::QRCTM)
#     M = M[1][:,:,:,:,:,1]
#     return init_env(M, χ, alg)
# end

function init_env(M::StructArray, χ::Int, alg::QRCTM)
    M = M[1]
    eltype(M) <: Complex && throw(ArgumentError("QRCTM only supports real-valued tensors for now."))

    D = size(M, 1)
    if ndims(M) == 4
        T = rand!(similar(M,χ,D,χ))
        T += conj(permutedims(T, (3,2,1)))
    else
        T = rand!(similar(M,χ,D,D,χ))
        T += conj(permutedims(T, (4,2,3,1)))
    end
    C = rand!(similar(M,χ,χ))
    C += C'

    return CTMEnv(C, T)
end

"""
    qrctm_step(env::CTMEnv, M::AbstractArray, alg::QRCTM)

One CTM left-move step for the QRCTM algorithm.
"""
function qrctm_step(env::CTMEnv, M::AbstractArray, alg::QRCTM)
    C = env.C
    T = env.T

    CT = _to_front(CTtoT(C, T))
    U, R = qr_for_ad(CT)
    U = reshape(U, size(T))

    T = FLmap_parallel(T, U, U, M; ifparallel=alg.ifparallel, forloop_iter=alg.forloop_iter,
                       inner_etype=alg.inner_etype)
    C_new = Cmap(R, T, U)

    T /= ignore_derivatives(() -> norm(T))
    C_new /= ignore_derivatives(() -> norm(C_new))
    err = ignore_derivatives(() -> norm(C_new - C))

    return CTMEnv(C_new, T), err
end

# ── iteration + boundary ───────────────────────────────────

# Core implementation operating on plain tensors (avoids StructArray overhead in AD)
function leading_boundary(env::CTMEnv, M::StructArray, alg::QRCTM)
    M = M[1]
    t = ignore_derivatives(() -> time())
    local err

    # Whole-VUMPS precision mode: pre-cast env (C, T) and M at entry, run
    # whole qrctm_step (FLmap + QR + norm) in it, cast back for polish iters.
    T_orig = eltype(env.T)
    want_whole = alg.whole_vumps_etype !== nothing && alg.whole_vumps_etype != real(T_orig)
    if want_whole
        W = alg.whole_vumps_etype
        env = CTMEnv(_downcast_eltype(W, env.C), _downcast_eltype(W, env.T))
        M   = _downcast_eltype(W, M)
    end
    # For whole-VUMPS mode, pass alg with inner_etype=nothing so FLmap runs
    # natively on the already-downcasted tensors (no per-call conversion).
    alg_wholemode = alg
    if want_whole
        alg_wholemode = deepcopy(alg)
        alg_wholemode.inner_etype = nothing
    end

    ignore_derivatives(() -> alg.verbosity >= 2 && @info "Start QRCTM iteration without AD...")
    ignore_derivatives() do
        for i in 1:alg.maxiter
        env, err = qrctm_step(env, M, alg_wholemode)
        alg.verbosity >= 3 && i % alg.show_every == 0 &&
            ignore_derivatives(() -> @info @sprintf("QRCTM@step: %4d\terr = %.3e\ttime = %.3f sec", i, err, time()-t))
        if err < alg.tol && i >= alg.miniter
            alg.verbosity >= 2 &&
                ignore_derivatives(() -> @info @sprintf("QRCTM conv@step: %4d\terr = %.3e\ttime = %.3f sec", i, err, time()-t))
            break
        end
        if i == alg.maxiter
            alg.verbosity >= 2 && ignore_derivatives(() -> @warn @sprintf("QRCTM cancel@step: %4d\terr = %.3e\ttime = %.3f sec", i, err, time()-t))
        end
    end
    end

    ignore_derivatives(() -> alg.verbosity >= 2 && @info "Start QRCTM iteration with AD...")
    # Coarse polish: final inner_etype_final_steps AD iters use full original precision.
    alg_ad = alg_wholemode
    alg_ad_coarse = alg
    if want_whole || alg.inner_etype !== nothing
        alg_ad_coarse = deepcopy(alg_wholemode)
        alg_ad_coarse.inner_etype = nothing
    end
    mixed_active = (alg.inner_etype !== nothing) || want_whole
    for i in 1:alg.maxiter_ad
        alg_this_iter = alg_ad
        in_polish = alg.inner_etype_final_steps > 0 &&
                    i > alg.maxiter_ad - alg.inner_etype_final_steps
        if mixed_active && in_polish
            alg_this_iter = alg_ad_coarse
        end
        # For whole-VUMPS mode: on the first polish iter, cast env and M back.
        if want_whole && in_polish && eltype(env.T) != T_orig
            env = CTMEnv(_downcast_eltype(real(T_orig), env.C),
                         _downcast_eltype(real(T_orig), env.T))
            M = _downcast_eltype(real(T_orig), M)
        end
        env, err = checkpoint(alg.step_checkpoint, qrctm_step, env, M, alg_this_iter)
        alg.verbosity >= 3 && i % alg.show_every == 0 && ignore_derivatives(() -> @info @sprintf("QRCTM@step: %4d\terr = %.3e\ttime = %.3f sec", i, err, time()-t))
        if err < alg.tol && i >= alg.miniter_ad
            alg.verbosity >= 2 && ignore_derivatives(() -> @info @sprintf("QRCTM conv@step: %4d\terr = %.3e\ttime = %.3f sec", i, err, time()-t))
            break
        end
        if i == alg.maxiter_ad
            alg.verbosity >= 2 && ignore_derivatives(() -> @warn @sprintf("QRCTM cancel@step: %4d\terr = %.3e\ttime = %.3f sec", i, err, time()-t))
        end
    end
    # Exit guard: if whole-VUMPS mode active and we never hit polish, cast back.
    if want_whole && eltype(env.T) != T_orig
        env = CTMEnv(_downcast_eltype(real(T_orig), env.C),
                     _downcast_eltype(real(T_orig), env.T))
    end
    return env, err
end

ObsEnv(env::CTMEnv, M::StructArray, ::QRCTM, model=nothing) = env

# Imaginary-error indicator (|⟨iSy⟩|) for real-valued energies.
# See docstring on `imag_error` in src/ipeps_optimize/optimize.jl.
function imag_error(env::CTMEnv, A, iSy, params::iPEPSOptimize)
    @unpack C, T = env
    @unpack forloop_iter, ifparallel = params.boundary_alg
    To = CTCtoT(C, T)
    A1 = A[1]
    My = contract_o_11(To, T, A1, T, To, iSy; ifparallel, forloop_iter)
    n  = contract_n_11(To, T, A1, T, To; ifparallel, forloop_iter)
    return abs(My / n)
end
