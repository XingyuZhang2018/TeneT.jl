# Preconditioners for iPEPS gradient optimization.
# Various strategies to precondition the gradient using the transfer matrix environment.

# Allow ForwardDiff to build Dual{T, ComplexF64, N} (needed for JVP through complex arrays)
ForwardDiff.can_dual(::Type{ComplexF64}) = true

# Correct conj for complex-valued Dual numbers:
# d/dt conj(z(t)) = conj(dz/dt), so the partial must be conjugated too.
# ForwardDiff's default conj(d::Dual) = d is wrong for V<:Complex.
Base.conj(d::ForwardDiff.Dual{T,V,N}) where {T,V<:Complex,N} =
    ForwardDiff.Dual{T}(conj(ForwardDiff.value(d)), conj.(ForwardDiff.partials(d).values))

"""
    precondition_invese_single_envir(A, grad, rt, params, restriction_ipeps, fδEi, iter_precond)

Precondition the gradient using the single-layer VUMPS environment (VUMPSEnv).
Solves `(δ + M_u) x = grad` where `M_u = J_R^† T J_R` correctly propagates through
arbitrary `build_A ∘ restriction_ipeps` (Jacobian J_R).

Split into three steps to avoid nested ForwardDiff which breaks for ComplexF64
(ForwardDiff's promote_rule for Dual requires V<:Real, so two differently-tagged
ComplexF64 Duals in the same @tensor give TC = Array{Union{},N}):
  1. JVP via ForwardDiff.derivative (real t): A_prime_x[p] = J_R_p * x
  2. Transfer matrix action: T_x[p] = Mumap_parallel(..., A_prime_x[p]) / n
  3. VJP via Zygote: gN = J_R^† * T_x
"""
function precondition_invese_single_envir(A, grad, rt::Union{VUMPSRuntime, Tuple{VUMPSRuntime,VUMPSRuntime}}, params, restriction_ipeps, fδEi, iter_precond)
    if fδEi[3] <= iter_precond
        return grad
    end
    t0 = time()
    δ = fδEi[2]
    build_restricted_A(x) = build_A(restriction_ipeps(x), params)

    _G_cache[] = nothing          # reset so the first plain call below computes fresh G
    A_prime = build_restricted_A(A)   # populates _G_cache; all JVP+VJP calls reuse it

    env = ObsEnv(rt, A_prime, params.boundary_alg)
    @unpack ACu, ARu, ACd, ARd, FLu, FRu, FLo, FRo = env

    gradnew = deepcopy(grad)
    Ni, Nj = size(A_prime)
    @unpack forloop_iter = params
    @unpack ifparallel = params.boundary_alg
    pattern = A_prime.pattern

    gradnew, _ = linsolve(grad; isposdef = true, maxiter=1, verbosity=0) do x
        # Step 1: JVP — perturb A by x, get site-tensor perturbation A_prime_x[p] = J_R_p * x
        # ForwardDiff.derivative: real t → no nested complex Dual issue
        A_prime_x = [ForwardDiff.derivative(
            t -> build_restricted_A(A + t * x)[p], 0.0
        ) for p in 1:length(A_prime)]

        # Step 2: transfer matrix action on each perturbed site tensor
        T_x = [begin
            i, j = Tuple(findfirst(==(p), pattern))
            ir = Ni + 1 - i
            n = contract_n1(FLo[i,j], ACu[i,j], A_prime[i,j], ACd[ir,j], FRo[i,j]; forloop_iter, ifparallel)
            Mumap_parallel(ACu[i,j], ACd[ir,j], FLo[i,j], FRo[i,j], A_prime_x[p]; forloop_iter, ifparallel) / n
        end for p in 1:length(A_prime)]

        # Step 3: VJP — pull T_x back through build_A ∘ restriction_ipeps via Zygote
        # Zygote gradient of real(dot(T_x[p], Ad)) w.r.t. Ad equals T_x[p]
        function overlap_vjp(y)
            total = zero(real(eltype(y)))
            Ad = build_restricted_A(y)
            for p in 1:length(A_prime)
                total += real(dot(Ad[p], T_x[p]))
            end
            return total
        end
        gN = Zygote.gradient(overlap_vjp, A)[1]

        return δ * x + gN
    end

    params.verbosity >= 3 && printstyled("Preconditioner took $(round(time() - t0, digits = 2)) s\n"; bold=true, color=:green)
    return gradnew
end

function precondition_invese_single_envir(A, grad, rt::PlaquetteVUMPSRuntime, params, restriction_ipeps, fδEi, iter_precond)
    if fδEi[3] <= iter_precond
        return grad
    end
    t0 = time()
    δ = fδEi[2]
    build_restricted_A(x) = build_A(restriction_ipeps(x), params)

    _G_cache[] = nothing          # reset so the first plain call below computes fresh G
    A_prime = build_restricted_A(A)   # populates _G_cache; all JVP+VJP calls reuse it

    env = ObsEnv(rt, A_prime, params.boundary_alg)
    @unpack AL, C, FLu, FLo = env
    AC = ALCtoAC(AL, C)

    gradnew = deepcopy(grad)
    Ni, Nj = size(A_prime)
    @unpack forloop_iter = params
    @unpack ifparallel = params.boundary_alg
    pattern = A_prime.pattern

    gradnew, _ = linsolve(grad; isposdef=true, maxiter=1, verbosity=0) do x
        A_prime_x = [ForwardDiff.derivative(
            t -> build_restricted_A(A + t * x)[p], 0.0
        ) for p in 1:length(A_prime)]

        T_x = [begin
            i, j = Tuple(findfirst(==(p), pattern))
            ir = Ni + 1 - i
            jr = mod1(j + 1, Nj)
            n = contract_n1(FLo[i,j], AC[i,j], A_prime[i,j], AC[ir,j], FLo[i,jr]; ifparallel, forloop_iter)
            Mumap_parallel(AC[i,j], AC[ir,j], FLo[i,j], FLo[i,jr], A_prime_x[p]; forloop_iter, ifparallel) / n
        end for p in 1:length(A_prime)]

        function overlap_vjp(y)
            total = zero(real(eltype(y)))
            Ad = build_restricted_A(y)
            for p in 1:length(A_prime)
                total += real(dot(Ad[p], T_x[p]))
            end
            return total
        end
        gN = Zygote.gradient(overlap_vjp, A)[1]

        return δ * x + gN
    end

    params.verbosity >= 3 && printstyled("Preconditioner took $(round(time() - t0, digits = 2)) s\n"; bold=true, color=:green)
    return gradnew
end

function precondition_invese_single_envir(A, grad, env::C4vVUMPSEnv, params, restriction_ipeps, fδEi, iter_precond)
    if fδEi[3] <= iter_precond
        return grad
    end
    t0 = time()
    δ = fδEi[2]
    build_restricted_A(x) = build_A(restriction_ipeps(x), params)

    @unpack AL, C, FL = env
    AC = ALCtoAC_map(AL, C)

    A_prime = build_restricted_A(A)
    gradnew = deepcopy(grad)

    @unpack forloop_iter = params
    @unpack ifparallel = params.boundary_alg

    gradnew, _ = linsolve(grad; isposdef=true, maxiter=1, verbosity=0) do x
        A_prime_x = [ForwardDiff.derivative(
            t -> build_restricted_A(A + t * x)[p], 0.0
        ) for p in 1:length(A_prime)]

        T_x = [begin
            n = contract_n1(FL, AC, A_prime[p], AC, FL; ifparallel, forloop_iter)
            Mumap_parallel(AC, AC, FL, FL, A_prime_x[p]; forloop_iter, ifparallel) / n
        end for p in 1:length(A_prime)]

        function overlap_vjp(y)
            total = zero(real(eltype(y)))
            Ad = build_restricted_A(y)
            for p in 1:length(A_prime)
                total += real(dot(Ad[p], T_x[p]))
            end
            return total
        end
        gN = Zygote.gradient(overlap_vjp, A)[1]

        return δ * x + gN
    end

    params.verbosity >= 3 && printstyled("Preconditioner took $(round(time() - t0, digits = 2)) s\n"; bold=true, color=:green)
    return gradnew
end

function precondition_invese_single_envir(A, grad, env::CTMEnv, params, restriction_ipeps, fδEi, iter_precond)
    if fδEi[3] <= iter_precond
        return grad
    end
    t0 = time()
    δ = fδEi[2]
    build_restricted_A(x) = build_A(restriction_ipeps(x), params)

    @unpack C, T = env
    To = CTCtoT(C, T)

    A_prime = build_restricted_A(A)
    gradnew = deepcopy(grad)

    @unpack forloop_iter = params
    @unpack ifparallel = params.boundary_alg

    gradnew, _ = linsolve(grad; isposdef=true, maxiter=1, verbosity=0) do x
        A_prime_x = [ForwardDiff.derivative(
            t -> build_restricted_A(A + t * x)[p], 0.0
        ) for p in 1:length(A_prime)]

        T_x = [begin
            n = dot(To, To)
            Mumap_parallel(T, T, To, To, A_prime_x[p]; forloop_iter, ifparallel) / n
        end for p in 1:length(A_prime)]

        function overlap_vjp(y)
            total = zero(real(eltype(y)))
            Ad = build_restricted_A(y)
            for p in 1:length(A_prime)
                total += real(dot(Ad[p], T_x[p]))
            end
            return total
        end
        gN = Zygote.gradient(overlap_vjp, A)[1]

        return δ * x + gN
    end

    params.verbosity >= 3 && printstyled("Preconditioner took $(round(time() - t0, digits = 2)) s\n"; bold=true, color=:green)
    return gradnew
end
