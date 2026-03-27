# PT boundary algorithm
#
# Power Transfer method.
# Uses polar decomposition of C*T to obtain an isometric form AL,
# then updates the edge tensor via eigenvector solve and Hermitianizes.

# ── PT leftmove ─────────────────────────────────────────────────────────

"""
    leftmove(M, env::CTMEnv, alg::PT)

One CTM left-move step for the PT (Power Transfer) algorithm.

1. Polar-decompose C*T to get isometric AL and updated corner C
2. Update edge T via Eenv_ctm eigsolve using AL as the projector
3. Hermitianize T and C
4. Normalize and return
"""
function leftmove(M, env::CTMEnv, alg::PT)
    C = env.C
    T = env.T

    # Polar decomposition of combined C*T
    Tu = CTtoT(C, T)
    # _to_tail: (chi, D, chi) -> (D*chi, chi) for leg3
    #           (chi, D, D, chi) -> (D*D*chi, chi) for leg4
    AL, C = polar_decomposition(_to_tail(Tu))
    C /= norm(C)
    AL = reshape(AL, size(Tu))

    # Edge tensor update via transfer matrix eigsolve
    _, T = Eenv_ctm(AL, conj(AL), M, T; alg)

    # Hermitianize edge and corner
    if T isa AbstractArray{<:Number,3}
        T += conj(permutedims(T, (3,2,1)))
    else
        T += conj(permutedims(T, (4,2,3,1)))
    end
    C += C'
    T /= Zygote.@ignore norm(T)
    C /= Zygote.@ignore norm(C)

    return CTMEnv(C, T)
end

# Override leading_boundary for PT since leftmove does not return err
function leading_boundary(M, env::CTMEnv, alg::PT)
    t0 = Zygote.@ignore time()
    err = Inf

    # Phase 1: non-differentiable warm-up
    Zygote.@ignore for i = 1:alg.maxiter
        env_new = leftmove(M, env, alg)
        err = norm(env_new.C / norm(env_new.C) - env.C / norm(env.C))
        env = env_new
        alg.verbosity >= 3 && i % alg.output_interval == 0 &&
            @info @sprintf("i = %5d,\tt = %.2fs\terr = %.3e\n", i, time()-t0, err)
        if err < alg.tol && i >= alg.miniter
            alg.verbosity >= 2 &&
                @info @sprintf("i = %5d,\tt = %.2fs\terr = %.3e\n", i, time()-t0, err)
            break
        end
        if i == alg.maxiter
            alg.verbosity >= 2 &&
                @warn @sprintf("i = %5d,\tt = %.2fs\terr = %.3e\n", i, time()-t0, err)
        end
    end

    # Phase 2: differentiable iterations
    ifcheckpoint = get(alg, :ifcheckpoint, false)
    for i = 1:alg.maxiter_ad
        env_new = ifcheckpoint ?
            checkpoint(leftmove, M, env, alg) : leftmove(M, env, alg)
        err = Zygote.@ignore norm(env_new.C / norm(env_new.C) - env.C / norm(env.C))
        env = env_new
        alg.verbosity >= 3 && i % alg.output_interval == 0 &&
            Zygote.@ignore @info @sprintf("i = %5d,\tt = %.2fs\terr = %.3e\n", i, time()-t0, err)
        if i > alg.miniter_ad && err < alg.tol
            alg.verbosity >= 2 &&
                Zygote.@ignore @info @sprintf("i = %5d,\tt = %.2fs\terr = %.3e\n", i, time()-t0, err)
            break
        end
        if i == alg.maxiter_ad
            alg.verbosity >= 2 &&
                Zygote.@ignore @warn @sprintf("i = %5d,\tt = %.2fs\terr = %.3e\n", i, time()-t0, err)
        end
    end

    return env, err
end
