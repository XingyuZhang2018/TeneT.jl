# CTMRG boundary algorithm and shared CTM helpers
#
# Shared helpers (CTtoT, CTCtoT, CTMmap, CTMenv, Cenv_ctm, Eenv_ctm, ACenv_ctm,
# qr_for_ad, polar_decomposition, initialize_ctm_env, leading_boundary for CTMEnv)
# are defined here so that later files (qrctm.jl, fpctm.jl, pt.jl) can use them.

# ── Basic contraction maps for single-site CTM ──────────────────────────

"""
    CTtoT(C, T)

Contract corner matrix `C` into edge tensor `T`:
```
C[a,b] * T[b,c,d] -> [a,c,d]      (leg3)
C[a,b] * T[b,c,d,e] -> [a,c,d,e]  (leg4)
```
"""
CTtoT(C, T::AbstractArray{<:Number,3}) = ein"ab,bcd->acd"(C, T)
CTtoT(C, T::AbstractArray{<:Number,4}) = ein"ab,bcde->acde"(C, T)

"""
    CTCtoT(C, T)

Contract corner matrices on both sides of an edge tensor:
```
C[a,b] * T[b,c,d] * C[d,e] -> [a,c,e]          (leg3)
C[a,b] * T[b,c,d,e] * C[e,f] -> [a,c,d,f]      (leg4)
```
"""
CTCtoT(C, T::AbstractArray{<:Number,3}) = ein"(ab,bcd),de->ace"(C,T,C)
CTCtoT(C, T::AbstractArray{<:Number,4}) = ein"(ab,bcde),ef->acdf"(C,T,C)

"""
    ALCtoAC_ctm(AL, C)

Single-site version: contract left-canonical tensor with center matrix.
"""
ALCtoAC_ctm(AL::AbstractArray{<:Number,3}, C) = ein"asc,cb -> asb"(AL, C)
ALCtoAC_ctm(AL::AbstractArray{<:Number,4}, C) = ein"astc,cb -> astb"(AL, C)

"""
    CTMmap(C, Tu, Tl, Td, Tr, M)

Full CTM corner map for a single plaquette.
"""
CTMmap(C, Tu::AbstractArray{<:Number,3}, Tl, Td, Tr, M) =
    ein"(((((adb,ac),cei),degf),igj),bfh)->hj"(Tu, C, Tl, M, Td, Tr)
CTMmap(C, Tu::AbstractArray{<:Number,4}, Tl, Td, Tr, M) =
    ein"(((((bcde,ab),afhj),fkgcp),hlidp),jklm),egin->mn"(Tu, C, Tl, M, M, Td, Tr)

# ── QR / polar decomposition helpers ────────────────────────────────────

"""
    qr_for_ad(A)

QR factorization that materializes Q as a dense array of the same type as A.
Compatible with AD frameworks.
"""
function qr_for_ad(A::AbstractMatrix{T}) where {T}
    Q, R = qr(A)
    Q = _arraytype(A)(Q)
    return Q, R
end

"""
    polar_decomposition(A)

Polar decomposition A = U * P via truncated SVD.
Returns the unitary factor U and the positive-semidefinite factor P.
"""
function polar_decomposition(A::AbstractMatrix{<:Number})
    F = svd(A)
    trunc = sum(F.S .> 1e-10)
    U_part = F.U[:, 1:trunc] * F.Vt[1:trunc, :]
    P = F.V[:, 1:trunc] * Diagonal(F.S[1:trunc]) * F.Vt[1:trunc, :]
    return U_part, P
end

# ── Single-site eigenvector helpers ─────────────────────────────────────

"""
    CTMenv(Tu, Tl, Td, Tr, M, Cul)

Find the leading eigenvector of the full CTM corner map.
"""
function CTMenv(Tu, Tl, Td, Tr, M, Cul)
    lambda, cul, info = eigsolve(x -> CTMmap(x, Tu, Tl, Td, Tr, M), Cul, 1, :LM)
    info.converged == 0 && error("eigsolve did not converge")
    return lambda[1], cul[1]
end

"""
    Cenv_ctm(Tu, Td, Cint; alg, ifvalue=false)

Leading eigenvector of the corner map `C -> Cmap(C, Tu, Td)`.
Uses either power iteration (`ifsimple_eig`) or Krylov eigsolve.
"""
function Cenv_ctm(Tu, Td, Cint; alg, ifvalue=false)
    f(C) = Cmap(C, Tu, Td)
    if alg.ifsimple_eig
        lambda, v = simple_eig(f, Cint; power_iter=get(alg, :maxiter_power, 10), ifvalue)
        return lambda[1], v[1]
    else
        lambdas, Cs, info = eigsolve(f, Cint, 1, :LM, KrylovKit.Arnoldi(verbosity=0))
        return lambdas[1], Cs[1]
    end
end

# Helper: extract maxiter_power from algorithm struct, with fallback
function Base.get(alg::Algorithm, field::Symbol, default)
    hasfield(typeof(alg), field) ? getfield(alg, field) : default
end

"""
    Eenv_ctm(Tu, Td, M, Tint; alg, ifvalue=false)

Leading eigenvector of the edge transfer map `E -> FLmap_parallel(E, Tu, Td, M; ...)`.
"""
function Eenv_ctm(Tu, Td, M, Tint; alg, ifvalue=false)
    ifparallel = get(alg, :ifparallel, false)
    forloop_iter = get(alg, :forloop_iter, 1)
    f(E) = FLmap_parallel(E, Tu, Td, M; ifparallel, forloop_iter)
    if alg.ifsimple_eig
        power_iter = get(alg, :maxiter_power, 10)
        lambda, v = simple_eig(f, Tint; power_iter, ifvalue)
        return lambda[1], v[1]
    else
        lambdas, Es, info = eigsolve(f, Tint, 1, :LM, KrylovKit.Arnoldi(verbosity=0))
        return lambdas[1], Es[1]
    end
end

"""
    ACenv_ctm(Tl, Tr, M, Tint; alg, ifvalue=false)

Leading eigenvector of the AC transfer map `AC -> ACmap_parallel(AC, Tl, Tr, M; ...)`.
"""
function ACenv_ctm(Tl, Tr, M, Tint; alg, ifvalue=false)
    ifparallel = get(alg, :ifparallel, false)
    forloop_iter = get(alg, :forloop_iter, 1)
    f(AC) = ACmap_parallel(AC, Tl, Tr, M; ifparallel, forloop_iter)
    if alg.ifsimple_eig
        power_iter = get(alg, :maxiter_power, 10)
        lambda, v = simple_eig(f, Tint; power_iter, ifvalue)
        return lambda[1], v[1]
    else
        lambdas, Es, info = eigsolve(f, Tint, 1, :LM, KrylovKit.Arnoldi(verbosity=0))
        return lambdas[1], Es[1]
    end
end

# ── CTM environment initialization ──────────────────────────────────────

"""
    initialize_ctm_env(M, chi, alg::Algorithm; file="env.jld2")

Create (or load) an initial CTMEnv for a single-site CTM algorithm.
"""
function initialize_ctm_env(M, chi, alg::Algorithm; file::String="env.jld2")
    if get(alg, :ifload_env, false) && ispath(file)
        env = load_env(file, _arraytype(M))
        alg.verbosity >= 2 && @info "load CTM env from $file"
        return env
    else
        D = size(M, 1)
        if M isa AbstractArray{<:Number,4}
            T = rand!(similar(M, chi, D, chi))
            T += conj(permutedims(T, (3,2,1)))
        else
            T = rand!(similar(M, chi, D, D, chi))
            T += conj(permutedims(T, (4,2,3,1)))
        end
        C = rand!(similar(M, chi, chi))
        C += C'
        _, C = Cenv_ctm(T, conj(T), C; alg, ifvalue=false)
        if alg.verbosity >= 4
            printstyled("start $alg random initial environment->  \n"; bold=true, color=:green)
        elseif alg.verbosity >= 2
            printstyled("start random initial chi$(chi) CTM environment->  \n"; bold=true, color=:green)
        end
        return CTMEnv(C, T)
    end
end

# ── Convergence loop (shared by CTMRG, QRCTM, FPCTM, PT) ──────────────

"""
    leading_boundary(M, env::CTMEnv, alg::Algorithm)

Run the CTM convergence loop:
1. Non-differentiable warm-up phase (up to `alg.maxiter` steps)
2. Differentiable phase (up to `alg.maxiter_ad` steps, optionally checkpointed)

Returns `(env, err)`.
"""
function leading_boundary(M, env::CTMEnv, alg::Algorithm)
    t0 = Zygote.@ignore time()
    err = Inf

    # Phase 1: non-differentiable warm-up
    Zygote.@ignore for i = 1:alg.maxiter
        env, err = leftmove(M, env, alg)
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

    # Phase 2: differentiable (AD-compatible) iterations
    for i = 1:alg.maxiter_ad
        env, err = get(alg, :ifcheckpoint, false) ?
            checkpoint(leftmove, M, env, alg) : leftmove(M, env, alg)
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

# ── logZ for CTMEnv ─────────────────────────────────────────────────────

"""
    logZ(M, env::CTMEnv)

Compute log of the partition function ratio from a converged CTMEnv.
"""
function logZ(M::AbstractArray{<:Number,4}, env::CTMEnv)
    C = env.C
    T = env.T

    E = CTCtoT(C, T)
    E2 = CTCtoT(C, T)
    lambdaM = ein"abc,cba->"(FLmap(E, T, T, M), E2)[] / ein"abc,cba->"(E, E2)[]

    Csq = C * C
    Csq2 = Csq
    lambdaN = ein"ab,ba->"(Lmap(Csq, T, T), Csq2)[] / ein"ab,ba->"(Csq, Csq2)[]

    return log(abs(lambdaM / lambdaN))
end

# ── CTMRG leftmove ──────────────────────────────────────────────────────
# NOTE: CTMRG in ADC4PEPS did not have an explicit leftmove dispatch;
# it used the full CTMmap-based eigsolve. We port the CTMenv-based approach.

"""
    leftmove(M, env::CTMEnv, alg::CTMRG)

One CTM left-move step for the CTMRG algorithm.
Uses the full plaquette CTM map to update the corner matrix,
then updates the edge tensor via the transfer matrix eigsolve.
"""
function leftmove(M, env::CTMEnv, alg::CTMRG)
    C = env.C
    T = env.T

    # Update corner via full CTM map eigsolve
    _, C = CTMenv(T, T, conj(T), conj(T), M, C)

    # Update edge via transfer matrix eigsolve
    _, T = Eenv_ctm(T, conj(T), M, T; alg)

    # Hermitianize
    if T isa AbstractArray{<:Number,3}
        T += conj(permutedims(T, (3,2,1)))
    else
        T += conj(permutedims(T, (4,2,3,1)))
    end
    C += C'

    T /= Zygote.@ignore norm(T)
    C /= Zygote.@ignore norm(C)

    err = Zygote.@ignore norm(C - env.C / norm(env.C))

    return CTMEnv(C, T), err
end
