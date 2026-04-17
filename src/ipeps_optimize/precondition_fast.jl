# Preconditioners for iPEPS gradient optimization (optimized version).
# Changes vs precondition.jl:
#   1. Precompute normalization `n` outside the linsolve loop (it's independent of x)
#   2. Add `precond_every` parameter to reuse a stale preconditioned gradient

# ─── Stale-preconditioner cache ───────────────────────────────────────────────
# Stores the last preconditioned direction so it can be reused for `precond_every` steps.
const _precond_cache = Ref{Any}(nothing)   # (iter_last::Int, gradnew::Any)

"""
    precondition_fast(A, grad, rt, params, restriction_ipeps, fδEi, iter_precond; precond_every=1)

Drop-in replacement for `precondition_invese_single_envir` with two speed-ups:
  1. `contract_n_11` (normalization) is hoisted out of the linsolve closure.
  2. When `precond_every > 1`, the expensive preconditioner is recomputed only
     every `precond_every` optimisation steps; in between, the raw gradient is returned.
"""
function precondition_fast end   # forward declaration for dispatch

# ──────────────────────────────────────────────────────────────────────────────
# Dispatch 1: VUMPSRuntime / Tuple{VUMPSRuntime,VUMPSRuntime}
# ──────────────────────────────────────────────────────────────────────────────
function precondition_fast(A, grad, rt::Union{VUMPSRuntime, Tuple{VUMPSRuntime,VUMPSRuntime}}, params, restriction_ipeps, fδEi, iter_precond; precond_every::Int=1)
    if fδEi[3] <= iter_precond
        return grad
    end
    # Lazy reuse: skip expensive recomputation on intermediate steps
    iter = Int(fδEi[3])
    if precond_every > 1 && _precond_cache[] !== nothing
        iter_last, cached = _precond_cache[]
        if iter - iter_last < precond_every
            return cached
        end
    end

    t0 = time()
    δ = fδEi[2]
    build_restricted_A(x) = build_A(restriction_ipeps(x), params)

    _G_cache[] = nothing
    A_prime = build_restricted_A(A)

    env = ObsEnv(rt, A_prime, params.boundary_alg)
    @unpack ACu, ARu, ACd, ARd, FLu, FRu, FLo, FRo = env

    Ni, Nj = size(A_prime)
    @unpack forloop_iter = params
    @unpack ifparallel = params.boundary_alg

    # ── optimisation 1: precompute normalizations (independent of x) ──
    n_map = [begin
        ir = Ni + 1 - i
        contract_n_11(FLo[i,j], ACu[i,j], A_prime[i,j], ACd[ir,j], FRo[i,j]; forloop_iter, ifparallel)
    end for (i,j) in eachindex(A_prime)]

    gradnew = deepcopy(grad)
    gradnew, _ = linsolve(grad; isposdef = true, maxiter=1, verbosity=0) do x
        ε_fd = sqrt(eps(real(eltype(A))))
        B_plus  = build_restricted_A(A + ε_fd * x)
        B_minus = build_restricted_A(A - ε_fd * x)

        idx = 0
        T_x_data = [begin
            idx += 1
            A_prime_x_q = (B_plus[i,j] - B_minus[i,j]) / (2ε_fd)
            ir = Ni + 1 - i
            n = n_map[idx]
            Mumap_parallel(ACu[i,j], ACd[ir,j], FLo[i,j], FRo[i,j], A_prime_x_q; forloop_iter, ifparallel) / n
        end for (i,j) in eachindex(A_prime)]
        T_x = StructArray(T_x_data, A_prime.pattern)

        function overlap_vjp(y)
            total = zero(real(eltype(y)))
            Ad = build_restricted_A(y)
            for (i,j) in eachindex(A_prime)
                total += real(dot(Ad[i,j], T_x[i,j]))
            end
            return total
        end
        gN = Zygote.gradient(overlap_vjp, A)[1]

        return δ * x + gN
    end

    _precond_cache[] = (iter, gradnew)
    params.verbosity >= 3 && printstyled("Preconditioner took $(round(time() - t0, digits = 2)) s\n"; bold=true, color=:green)
    return gradnew
end

# ──────────────────────────────────────────────────────────────────────────────
# Dispatch 2: PlaquetteVUMPSRuntime
# ──────────────────────────────────────────────────────────────────────────────
function precondition_fast(A, grad, rt::PlaquetteVUMPSRuntime, params, restriction_ipeps, fδEi, iter_precond; precond_every::Int=1)
    if fδEi[3] <= iter_precond
        return grad
    end
    iter = Int(fδEi[3])
    if precond_every > 1 && _precond_cache[] !== nothing
        iter_last, cached = _precond_cache[]
        if iter - iter_last < precond_every
            return cached
        end
    end

    t0 = time()
    δ = fδEi[2]
    build_restricted_A(x) = build_A(restriction_ipeps(x), params)

    _G_cache[] = nothing
    A_prime = build_restricted_A(A)

    env = ObsEnv(rt, A_prime, params.boundary_alg)
    @unpack AL, C, FLu, FLo = env
    AC = ALCtoAC(AL, C)

    Ni, Nj = size(A_prime)
    @unpack forloop_iter = params
    @unpack ifparallel = params.boundary_alg

    # ── optimisation 1: precompute normalizations ──
    n_map = [begin
        ir = Ni + 1 - i
        jr = mod1(j + 1, Nj)
        contract_n_11(FLo[i,j], AC[i,j], A_prime[i,j], AC[ir,j], FLo[i,jr]; ifparallel, forloop_iter)
    end for (i,j) in eachindex(A_prime)]

    gradnew = deepcopy(grad)
    gradnew, _ = linsolve(grad; isposdef=true, maxiter=1, verbosity=0) do x
        ε_fd = sqrt(eps(real(eltype(A))))
        B_plus  = build_restricted_A(A + ε_fd * x)
        B_minus = build_restricted_A(A - ε_fd * x)

        idx = 0
        T_x_data = [begin
            idx += 1
            A_prime_x_q = (B_plus[i,j] - B_minus[i,j]) / (2ε_fd)
            ir = Ni + 1 - i
            jr = mod1(j + 1, Nj)
            n = n_map[idx]
            Mumap_parallel(AC[i,j], AC[ir,j], FLo[i,j], FLo[i,jr], A_prime_x_q; forloop_iter, ifparallel) / n
        end for (i,j) in eachindex(A_prime)]
        T_x = StructArray(T_x_data, A_prime.pattern)

        function overlap_vjp(y)
            total = zero(real(eltype(y)))
            Ad = build_restricted_A(y)
            for (i,j) in eachindex(A_prime)
                total += real(dot(Ad[i,j], T_x[i,j]))
            end
            return total
        end
        gN = Zygote.gradient(overlap_vjp, A)[1]

        return δ * x + gN
    end

    _precond_cache[] = (iter, gradnew)
    params.verbosity >= 3 && printstyled("Preconditioner took $(round(time() - t0, digits = 2)) s\n"; bold=true, color=:green)
    return gradnew
end

# ──────────────────────────────────────────────────────────────────────────────
# Dispatch 3: C4vVUMPSEnv
# ──────────────────────────────────────────────────────────────────────────────
function precondition_fast(A, grad, env::C4vVUMPSEnv, params, restriction_ipeps, fδEi, iter_precond; precond_every::Int=1)
    if fδEi[3] <= iter_precond
        return grad
    end
    iter = Int(fδEi[3])
    if precond_every > 1 && _precond_cache[] !== nothing
        iter_last, cached = _precond_cache[]
        if iter - iter_last < precond_every
            return cached
        end
    end

    t0 = time()
    δ = fδEi[2]
    build_restricted_A(x) = build_A(restriction_ipeps(x), params)

    @unpack AL, C, FL = env
    AC = ALCtoAC_map(AL, C)

    _G_cache[] = nothing
    A_prime = build_restricted_A(A)

    @unpack forloop_iter = params
    @unpack ifparallel = params.boundary_alg

    # ── optimisation 1: precompute normalizations ──
    n_map = [begin
        contract_n_11(FL, AC, A_prime[i,j], AC, FL; ifparallel, forloop_iter)
    end for (i,j) in eachindex(A_prime)]

    gradnew = deepcopy(grad)
    gradnew, _ = linsolve(grad; isposdef=true, maxiter=1, verbosity=0) do x
        ε_fd = sqrt(eps(real(eltype(A))))
        B_plus  = build_restricted_A(A + ε_fd * x)
        B_minus = build_restricted_A(A - ε_fd * x)

        idx = 0
        T_x_data = [begin
            idx += 1
            A_prime_x_q = (B_plus[i,j] - B_minus[i,j]) / (2ε_fd)
            n = n_map[idx]
            Mumap_parallel(AC, AC, FL, FL, A_prime_x_q; forloop_iter, ifparallel) / n
        end for (i,j) in eachindex(A_prime)]
        T_x = StructArray(T_x_data, A_prime.pattern)

        function overlap_vjp(y)
            total = zero(real(eltype(y)))
            Ad = build_restricted_A(y)
            for (i,j) in eachindex(A_prime)
                total += real(dot(Ad[i,j], T_x[i,j]))
            end
            return total
        end
        gN = Zygote.gradient(overlap_vjp, A)[1]

        return δ * x + gN
    end

    _precond_cache[] = (iter, gradnew)
    params.verbosity >= 3 && printstyled("Preconditioner took $(round(time() - t0, digits = 2)) s\n"; bold=true, color=:green)
    return gradnew
end

# ──────────────────────────────────────────────────────────────────────────────
# Dispatch 4: CTMEnv
# ──────────────────────────────────────────────────────────────────────────────
function precondition_fast(A, grad, env::CTMEnv, params, restriction_ipeps, fδEi, iter_precond; precond_every::Int=1)
    if fδEi[3] <= iter_precond
        return grad
    end
    iter = Int(fδEi[3])
    if precond_every > 1 && _precond_cache[] !== nothing
        iter_last, cached = _precond_cache[]
        if iter - iter_last < precond_every
            return cached
        end
    end

    t0 = time()
    δ = fδEi[2]
    build_restricted_A(x) = build_A(restriction_ipeps(x), params)

    @unpack C, T = env
    To = CTCtoT(C, T)

    _G_cache[] = nothing
    A_prime = build_restricted_A(A)

    @unpack forloop_iter = params
    @unpack ifparallel = params.boundary_alg

    # ── optimisation 1: precompute normalization ──
    n_val = dot(conj(To), To)

    gradnew = deepcopy(grad)
    gradnew, _ = linsolve(grad; isposdef=true, maxiter=1, verbosity=0) do x
        ε_fd = sqrt(eps(real(eltype(A))))
        B_plus  = build_restricted_A(A + ε_fd * x)
        B_minus = build_restricted_A(A - ε_fd * x)

        T_x_data = [begin
            A_prime_x_q = (B_plus[i,j] - B_minus[i,j]) / (2ε_fd)
            Mumap_parallel(T, T, To, To, A_prime_x_q; forloop_iter, ifparallel) / n_val
        end for (i,j) in eachindex(A_prime)]
        T_x = StructArray(T_x_data, A_prime.pattern)

        function overlap_vjp(y)
            total = zero(real(eltype(y)))
            Ad = build_restricted_A(y)
            for (i,j) in eachindex(A_prime)
                total += real(dot(Ad[i,j], T_x[i,j]))
            end
            return total
        end
        gN = Zygote.gradient(overlap_vjp, A)[1]

        return δ * x + gN
    end

    _precond_cache[] = (iter, gradnew)
    params.verbosity >= 3 && printstyled("Preconditioner took $(round(time() - t0, digits = 2)) s\n"; bold=true, color=:green)
    return gradnew
end
