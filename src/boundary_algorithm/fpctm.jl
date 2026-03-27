# FPCTM boundary algorithm
#
# Fixed-Point Corner Transfer Matrix method.
# Iteratively refines the projector U via polar decomposition
# until the fixed-point condition is satisfied, then uses eigenvector
# solves for corner and edge updates.

# ── FPCTM-specific getU ────────────────────────────────────────────────

"""
    getU(env::CTMEnv, ::FPCTM; ifsimple_eig)

Compute the isometric projector U and corner C for the FPCTM algorithm.

Iteratively solves for U using polar decomposition of the combined C*T tensor,
refining C via the corner eigenvector equation until convergence.
"""
function getU(env::CTMEnv, ::FPCTM; ifsimple_eig)
    C = env.C
    T = env.T

    # Initial corner eigenvector from C*C'
    _, Cl = Cenv_ctm(T, conj(T), C * C'; alg=_make_eig_alg(ifsimple_eig))
    C = _arraytype(Cl)(sqrt(Array(Cl)))

    chi, D, _ = size(T)
    # _to_tail: (chi, D, chi) -> (D*chi, chi)
    U, P = polar_decomposition(_to_tail(CTtoT(C, T)))
    U = reshape(U, chi, D, chi)
    error_val = norm(transpose(P) / sqrt(1.0) - C)
    C = transpose(P)

    i = 0
    while error_val > 1e-12 && i < 20
        _, C = Cenv_ctm(T, conj(U), C; alg=_make_eig_alg(ifsimple_eig))
        _, C = polar_decomposition(C)
        U, P = polar_decomposition(_to_tail(CTtoT(C, T)))
        U = reshape(U, chi, D, chi)
        temp = transpose(P) ./ C
        error_val = norm(temp[1] - temp[2])
        C = transpose(P)
        i += 1
    end
    return C, U
end

# Helper to build a minimal algorithm-like object for Cenv_ctm calls
struct _EigAlg <: Algorithm
    ifsimple_eig::Bool
    maxiter_power::Int
    verbosity::Int
    tol::Float64
    maxiter::Int
    miniter::Int
    maxiter_ad::Int
    miniter_ad::Int
    output_interval::Int
end

function _make_eig_alg(ifsimple_eig; maxiter_power=10)
    return _EigAlg(ifsimple_eig, maxiter_power, 0, 1e-10, 100, 1, 10, 1, 1)
end

# ── FPCTM leftmove ─────────────────────────────────────────────────────

"""
    leftmove(M, env::CTMEnv, alg::FPCTM)

One CTM left-move step for the FPCTM algorithm.

1. Compute projector U via iterated polar decomposition (non-differentiable)
2. Update corner C via Cenv_ctm eigsolve with projector
3. Update edge T via Eenv_ctm eigsolve
4. Hermitianize and normalize
"""
function leftmove(M, env::CTMEnv, alg::FPCTM)
    C = env.C
    T = env.T
    ifsimple_eig = alg.ifsimple_eig

    C, U = Zygote.@ignore getU(env, alg; ifsimple_eig)

    _, C = Cenv_ctm(T, conj(U), C; alg)
    _, T = Eenv_ctm(U, conj(U), M, T; alg)

    T += conj(permutedims(T, (3,2,1)))
    C += C'
    T /= Zygote.@ignore norm(T)
    C /= Zygote.@ignore norm(C)

    return CTMEnv(C, T)
end

# Override leading_boundary for FPCTM since leftmove does not return err
function leading_boundary(M, env::CTMEnv, alg::FPCTM)
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
    for i = 1:alg.maxiter_ad
        env_new = leftmove(M, env, alg)
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
