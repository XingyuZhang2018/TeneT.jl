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

    T = FLmap_parallel(T, U, U, M; ifparallel=alg.ifparallel, forloop_iter=alg.forloop_iter)
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

    ignore_derivatives(() -> alg.verbosity >= 2 && @info "Start QRCTM iteration without AD...")
    ignore_derivatives() do
        for i in 1:alg.maxiter
        env, err = qrctm_step(env, M, alg)
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
    for i in 1:alg.maxiter_ad
        env, err = alg.ifcheckpoint ? checkpoint(qrctm_step, env, M, alg) : qrctm_step(env, M, alg)
        alg.verbosity >= 3 && i % alg.show_every == 0 && ignore_derivatives(() -> @info @sprintf("QRCTM@step: %4d\terr = %.3e\ttime = %.3f sec", i, err, time()-t))
        if err < alg.tol && i >= alg.miniter_ad
            alg.verbosity >= 2 && ignore_derivatives(() -> @info @sprintf("QRCTM conv@step: %4d\terr = %.3e\ttime = %.3f sec", i, err, time()-t))
            break
        end
        if i == alg.maxiter_ad
            alg.verbosity >= 2 && ignore_derivatives(() -> @warn @sprintf("QRCTM cancel@step: %4d\terr = %.3e\ttime = %.3f sec", i, err, time()-t))
        end
    end
    return env, err
end

ObsEnv(env::CTMEnv, M::StructArray, ::QRCTM) = env
