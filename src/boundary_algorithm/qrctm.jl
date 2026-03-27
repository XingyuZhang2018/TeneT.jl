# QRCTM boundary algorithm
#
# QR-based Corner Transfer Matrix method.
# Uses QR decomposition of the combined C*T tensor to obtain the projector U,
# then applies a single transfer-matrix step followed by a QR-based corner update.

# ── initialization ─────────────────────────────────────────

function init_env(M::StructArray, χ::Int, alg::QRCTM)
    M = M[1][:,:,:,:,:,1]
    D = size(M, 1)  
    if M isa leg4
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
    leftmove(M, env::CTMEnv, alg::QRCTM)

One CTM left-move step for the QRCTM algorithm.
"""
function qrctm_step(env::CTMEnv, M::StructArray, alg::QRCTM)
    M = M[1][:,:,:,:,:,1]
    C = env.C
    T = env.T

    CT = _to_front(CTtoT(C, T))
    U, R = qrpos(CT)
    U = reshape(U, size(T))

    T = FLmap_parallel(T, U, conj(U), M;
                       ifparallel=alg.ifparallel,
                       forloop_iter=alg.forloop_iter)
    C_new = Cmap(R, T, conj(U))

    T /= Zygote.@ignore norm(T)
    C_new /= Zygote.@ignore norm(C_new)
    err = Zygote.@ignore norm(C_new - C)

    return CTMEnv(C_new, T), err
end

# ── Plaquette iteration + boundary ───────────────────────────────────

function qrctm_itr(env::CTMEnv, M::StructArray, alg::QRCTM)
    t = Zygote.@ignore time()
    local err

    Zygote.@ignore alg.verbosity >= 2 && @info "Start QRCTM iteration without AD..."
    Zygote.@ignore for i in 1:alg.maxiter
        env, err = qrctm_step(env, M, alg)
        alg.verbosity >= 3 && i % alg.show_every == 0 &&
            Zygote.@ignore @info @sprintf("QRCTM@step: %4d\terr = %.3e\ttime = %.3f sec", i, err, time()-t)
        if err < alg.tol && i >= alg.miniter
            alg.verbosity >= 2 &&
                Zygote.@ignore @info @sprintf("QRCTM conv@step: %4d\terr = %.3e\ttime = %.3f sec", i, err, time()-t)
            break
        end
        if i == alg.maxiter
            alg.verbosity >= 2 && Zygote.@ignore @warn @sprintf("QRCTM cancel@step: %4d\terr = %.3e\ttime = %.3f sec", i, err, time()-t)
        end
    end

    Zygote.@ignore alg.verbosity >= 2 && @info "Start QRCTM iteration with AD..."
    for i in 1:alg.maxiter_ad
        env, err = alg.ifcheckpoint ? checkpoint(qrctm_step, env, M, alg) : qrctm_step(env, M, alg)
        alg.verbosity >= 3 && i % alg.show_every == 0 && Zygote.@ignore @info @sprintf("QRCTM@step: %4d\terr = %.3e\ttime = %.3f sec", i, err, time()-t)
        if err < alg.tol && i >= alg.miniter_ad
            alg.verbosity >= 2 && Zygote.@ignore @info @sprintf("QRCTM conv@step: %4d\terr = %.3e\ttime = %.3f sec", i, err, time()-t)
            break
        end
        if i == alg.maxiter_ad
            alg.verbosity >= 2 && Zygote.@ignore @warn @sprintf("QRCTM cancel@step: %4d\terr = %.3e\ttime = %.3f sec", i, err, time()-t)
        end
    end
    return env, err
end

function leading_boundary(env::CTMEnv, M::StructArray, alg::QRCTM)
    return qrctm_itr(env, M, alg)
end

ObsEnv(env::CTMEnv, M::StructArray, ::QRCTM) = env
