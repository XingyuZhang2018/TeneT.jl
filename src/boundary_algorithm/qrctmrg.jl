# QRCTMRG boundary algorithm
#
# QR-based Corner Transfer Matrix method.
# Uses QR decomposition of the combined C*T tensor to obtain the projector U,
# then applies a single transfer-matrix step followed by a QR-based corner update.

# ── initialization ─────────────────────────────────────────

# function init_env(M::StructArray, χ::Int, alg::QRCTMRG{C4v})
#     M = M[1][:,:,:,:,:,1]
#     return init_env(M, χ, alg)
# end

function init_env(M::StructArray, χ::Int, alg::QRCTMRG)
    M = M[1]
    eltype(M) <: Complex && throw(ArgumentError("QRCTMRG only supports real-valued tensors for now."))

    D = size(M, 1)
    # if ndims(M) == 4 && size(M, 4) == D
    #     T = rand!(similar(M,χ,D,χ))
    #     T += conj(permutedims(T, (3,2,1)))
    # else
        T = rand!(similar(M,χ,D,D,χ))
        T += conj(permutedims(T, (4,2,3,1)))
    # end
    C = rand!(similar(M,χ,χ))
    C += C'

    return CTMEnv(C, T)
end

"""
    qrctmrg_step(env::CTMEnv, M::AbstractArray, alg::QRCTMRG{C4v})

One CTM left-move step for the QRCTMRG{C4v} algorithm.
"""
function qrctmrg_step(env::CTMEnv, M::AbstractArray, alg::QRCTMRG{C4v})
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

function qrctmrg_step(env::CTMEnv, M::AbstractArray, alg::QRCTMRG{C3v})
    C = env.C
    T = env.T

    CT = _to_front(CTtoT(C, T))
    U, R = qr_for_ad(CT)
    U = reshape(U, size(T))

    T = FLmap_C3v(T, U, U, M)
    # @tensor Mr[1,4,5,3,6,7] := M[1,2,3,6] * M[4,5,2,7]
    # D,d = size(M)[[1,4]]
    # Mr = reshape(Mr, D,D,D,D,d^2)
    # T = FLmap_parallel(T, U, U, Mr; ifparallel=alg.ifparallel, forloop_iter=alg.forloop_iter,
    #                    inner_etype=alg.inner_etype)
    C_new = Cmap(R, T, U)

    T /= ignore_derivatives(() -> norm(T))
    C_new /= ignore_derivatives(() -> norm(C_new))
    err = ignore_derivatives(() -> norm(C_new - C))

    return CTMEnv(C_new, T), err
end

# Core implementation operating on plain tensors (avoids StructArray overhead in AD)
function leading_boundary(env::CTMEnv, M::StructArray, alg::QRCTMRG)
    M = M[1]
    t = ignore_derivatives(() -> time())
    local err

    # Whole-VUMPS precision mode: pre-cast env (C, T) and M at entry, run
    # whole qrctmrg_step (FLmap + QR + norm) in it, cast back for polish iters.
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

    ignore_derivatives(() -> alg.verbosity >= 2 && @info "Start QRCTMRG iteration without AD...")
    ignore_derivatives() do
        for i in 1:alg.maxiter
        env, err = qrctmrg_step(env, M, alg_wholemode)
        alg.verbosity >= 3 && i % alg.show_every == 0 &&
            ignore_derivatives(() -> @info @sprintf("QRCTMRG@step: %4d\terr = %.3e\ttime = %.3f sec", i, err, time()-t))
        if err < alg.tol && i >= alg.miniter
            alg.verbosity >= 2 &&
                ignore_derivatives(() -> @info @sprintf("QRCTMRG conv@step: %4d\terr = %.3e\ttime = %.3f sec", i, err, time()-t))
            break
        end
        if i == alg.maxiter
            alg.verbosity >= 2 && ignore_derivatives(() -> @warn @sprintf("QRCTMRG cancel@step: %4d\terr = %.3e\ttime = %.3f sec", i, err, time()-t))
        end
    end
    end

    ignore_derivatives(() -> alg.verbosity >= 2 && @info "Start QRCTMRG iteration with AD...")
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
        env, err = checkpoint(alg.step_checkpoint, qrctmrg_step, env, M, alg_this_iter)
        alg.verbosity >= 3 && i % alg.show_every == 0 && ignore_derivatives(() -> @info @sprintf("QRCTMRG@step: %4d\terr = %.3e\ttime = %.3f sec", i, err, time()-t))
        if err < alg.tol && i >= alg.miniter_ad
            alg.verbosity >= 2 && ignore_derivatives(() -> @info @sprintf("QRCTMRG conv@step: %4d\terr = %.3e\ttime = %.3f sec", i, err, time()-t))
            break
        end
        if i == alg.maxiter_ad
            alg.verbosity >= 2 && ignore_derivatives(() -> @warn @sprintf("QRCTMRG cancel@step: %4d\terr = %.3e\ttime = %.3f sec", i, err, time()-t))
        end
    end
    # Exit guard: if whole-VUMPS mode active and we never hit polish, cast back.
    if want_whole && eltype(env.T) != T_orig
        env = CTMEnv(_downcast_eltype(real(T_orig), env.C),
                     _downcast_eltype(real(T_orig), env.T))
    end
    return env, err
end

ObsEnv(env::CTMEnv, M::StructArray, ::QRCTMRG, model=nothing) = env

# ── C3v honeycomb two-site QRCTMRG ────────────────────────

function _c3v_two_site_tensors(M::StructArray)
    length(M.data) == 2 ||
        throw(ArgumentError("QRCTMRG{C3vTwoSite} expects exactly two site tensors."))
    return _c3v_site_tensor(M.data[1]), _c3v_site_tensor(M.data[2])
end

function init_env(M::StructArray, χ::Int, alg::QRCTMRG{C3vTwoSite})
    MA, MB = _c3v_two_site_tensors(M)
    envA = init_env(StructArray([MA], [1;;]), χ, QRCTMRG{C3v}(; _qrctmrg_c3v_kwargs(alg)...))
    envB = init_env(StructArray([MB], [1;;]), χ, QRCTMRG{C3v}(; _qrctmrg_c3v_kwargs(alg)...))
    return C3vTwoSiteCTMEnv(envA.C, envA.T, envB.C, envB.T)
end

function _qrctmrg_c3v_kwargs(alg)
    return (
        tol=alg.tol,
        maxiter=alg.maxiter,
        miniter=alg.miniter,
        maxiter_ad=alg.maxiter_ad,
        miniter_ad=alg.miniter_ad,
        show_every=alg.show_every,
        verbosity=alg.verbosity,
        maxiter_power=alg.maxiter_power,
        ifsimple_eig=alg.ifsimple_eig,
        ifparallel=alg.ifparallel,
        step_checkpoint=alg.step_checkpoint,
        forloop_iter=alg.forloop_iter,
        inner_etype=alg.inner_etype,
        inner_etype_final_steps=alg.inner_etype_final_steps,
        simple_eig_polish_steps=alg.simple_eig_polish_steps,
        whole_vumps_etype=alg.whole_vumps_etype,
    )
end

function _c3v_step_sector(C, R, M1, M2, alg::QRCTMRG{C3vTwoSite})
    V, Rfac = _c3v_qr(C, R)
    Rnew = _c3v_update_edge(V, R, M1, M2; inner_etype=alg.inner_etype)
    Cnew = _c3v_update_corner(Rnew, Rfac, V)
    Rnew /= ignore_derivatives(() -> norm(Rnew))
    Cnew /= ignore_derivatives(() -> norm(Cnew))
    err = ignore_derivatives(() -> norm(Cnew - C))
    return Cnew, Rnew, err
end

function qrctmrg_step(env::C3vTwoSiteCTMEnv, M::Tuple, alg::QRCTMRG{C3vTwoSite})
    MA, MB = M
    CA, TA, errA = _c3v_step_sector(env.CA, env.TA, MA, MB, alg)
    CB, TB, errB = _c3v_step_sector(env.CB, env.TB, MB, MA, alg)
    err = ignore_derivatives(() -> max(errA, errB))
    return C3vTwoSiteCTMEnv(CA, TA, CB, TB), err
end

function leading_boundary(env::C3vTwoSiteCTMEnv, M::StructArray, alg::QRCTMRG{C3vTwoSite})
    MA, MB = _c3v_two_site_tensors(M)
    t = ignore_derivatives(() -> time())
    local err

    T_orig = eltype(env.TA)
    want_whole = alg.whole_vumps_etype !== nothing && alg.whole_vumps_etype != real(T_orig)
    if want_whole
        W = alg.whole_vumps_etype
        env = C3vTwoSiteCTMEnv(_downcast_eltype(W, env.CA), _downcast_eltype(W, env.TA),
                               _downcast_eltype(W, env.CB), _downcast_eltype(W, env.TB))
        MA = _downcast_eltype(W, MA)
        MB = _downcast_eltype(W, MB)
    end
    alg_wholemode = alg
    if want_whole
        alg_wholemode = deepcopy(alg)
        alg_wholemode.inner_etype = nothing
    end

    ignore_derivatives(() -> alg.verbosity >= 2 && @info "Start QRCTMRG{C3vTwoSite} iteration without AD...")
    ignore_derivatives() do
        for i in 1:alg.maxiter
            env, err = qrctmrg_step(env, (MA, MB), alg_wholemode)
            alg.verbosity >= 3 && i % alg.show_every == 0 &&
                ignore_derivatives(() -> @info @sprintf("QRCTMRG{C3vTwoSite}@step: %4d\terr = %.3e\ttime = %.3f sec", i, err, time()-t))
            if err < alg.tol && i >= alg.miniter
                alg.verbosity >= 2 &&
                    ignore_derivatives(() -> @info @sprintf("QRCTMRG{C3vTwoSite} conv@step: %4d\terr = %.3e\ttime = %.3f sec", i, err, time()-t))
                break
            end
        end
    end

    ignore_derivatives(() -> alg.verbosity >= 2 && @info "Start QRCTMRG{C3vTwoSite} iteration with AD...")
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
        if want_whole && in_polish && eltype(env.TA) != T_orig
            env = C3vTwoSiteCTMEnv(_downcast_eltype(real(T_orig), env.CA),
                                   _downcast_eltype(real(T_orig), env.TA),
                                   _downcast_eltype(real(T_orig), env.CB),
                                   _downcast_eltype(real(T_orig), env.TB))
            MA = _downcast_eltype(real(T_orig), MA)
            MB = _downcast_eltype(real(T_orig), MB)
        end
        env, err = checkpoint(alg.step_checkpoint, qrctmrg_step, env, (MA, MB), alg_this_iter)
        alg.verbosity >= 3 && i % alg.show_every == 0 &&
            ignore_derivatives(() -> @info @sprintf("QRCTMRG{C3vTwoSite}@step: %4d\terr = %.3e\ttime = %.3f sec", i, err, time()-t))
        if err < alg.tol && i >= alg.miniter_ad
            alg.verbosity >= 2 &&
                ignore_derivatives(() -> @info @sprintf("QRCTMRG{C3vTwoSite} conv@step: %4d\terr = %.3e\ttime = %.3f sec", i, err, time()-t))
            break
        end
    end
    if want_whole && eltype(env.TA) != T_orig
        env = C3vTwoSiteCTMEnv(_downcast_eltype(real(T_orig), env.CA),
                               _downcast_eltype(real(T_orig), env.TA),
                               _downcast_eltype(real(T_orig), env.CB),
                               _downcast_eltype(real(T_orig), env.TB))
    end
    return env, err
end

ObsEnv(env::C3vTwoSiteCTMEnv, M::StructArray, ::QRCTMRG{C3vTwoSite}, model=nothing) = env

# Imaginary-error indicator (|⟨iSy⟩|) for real-valued energies.
# See docstring on `imag_error` in src/ipeps_optimize/optimize.jl.
function imag_error(env::CTMEnv, A, iSy, params::iPEPSOptimize)
    @unpack C, T = env
    @unpack forloop_iter, ifparallel = params.boundary_alg
    if params.model.lattice isa Honeycomb{:c3v}
        A1 = A[1]
        n = _contract_one(_contract_c3v_one_site_norm, (C, T, A1), params)
        My = checkpoint(params.bond_checkpoint, _contract_c3v_one_site, C, T, A1, iSy; ifparallel, forloop_iter) / n
        return abs(My)
    end
    To = CTCtoT(C, T)
    A1 = A[1]
    My = contract_o_11(To, T, A1, T, To, iSy; ifparallel, forloop_iter)
    n  = contract_n_11(To, T, A1, T, To; ifparallel, forloop_iter)
    return abs(My / n)
end
