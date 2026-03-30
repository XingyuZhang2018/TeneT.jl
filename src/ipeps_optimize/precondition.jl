# Preconditioners for iPEPS gradient optimization.
# Various strategies to precondition the gradient using the transfer matrix environment.

"""
    precondition_invese_single_envir(A, grad, rt, params, restriction_ipeps, fδEi, iter_precond)

Precondition the gradient using the single-layer VUMPS environment (VUMPSEnv).
Solves `(δ + M_u) x = grad` where `M_u` is the transfer matrix environment map.

TeneT_demo version: uses gauge-transformed environment with `find_local_hermite_G`.
"""
function precondition_invese_single_envir(A, grad, rt::Union{VUMPSRuntime, Tuple{VUMPSRuntime,VUMPSRuntime}}, params, restriction_ipeps, fδEi, iter_precond)
    if fδEi[3] <= iter_precond
        return grad
    end
    δ = fδEi[2]
    if params.ifMCF
        Gh, Gv = find_local_min_norm_G(A, params)
        A_prime = gauge_transfer(A, [Gh, Gv], params)
    else
        A_prime = restriction_ipeps(A)
    end
    A_prime = build_A(A_prime, params)

    env = ObsEnv(rt, A_prime, params.boundary_alg)
    @unpack ACu, ARu, ACd, ARd, FLu, FRu, FLo, FRo = env

    gradnew = deepcopy(grad)
    Ni, Nj = size(A_prime)
    @unpack forloop_iter = params
    @unpack ifparallel = params.boundary_alg
    pattern = A_prime.pattern
    for p in 1:length(A_prime)
        i, j = Tuple(findfirst(==(p), pattern))
        ir = Ni + 1 - i
        n = contract_n1(FLo[i,j], ACu[i,j], A_prime[i,j], ACd[ir,j], FRo[i,j]; forloop_iter,ifparallel)
        gradnew[:,:,:,:,:,p], _ = linsolve(
            x -> δ * x + Mumap_parallel(ACu[i,j], ACd[ir,j], FLo[i,j], FRo[i,j], x; forloop_iter,ifparallel) / n,
            grad[:,:,:,:,:,p];
            isposdef=true, maxiter=1, verbosity=0
        )
        if params.ifMCF
            irr = mod1(i - 1, Ni)
            jr = mod1(j - 1, Nj)
            G_temp = [Gh[:,:,pattern[i,jr]], inv(Gv[:,:,p]), inv(Gh[:,:,p]), Gv[:,:,pattern[irr,j]]]
            gradnew[:,:,:,:,:,p] = local_gauge_contraction(gradnew[:,:,:,:,:,p], G_temp)
        end
    end

    return gradnew
end

function precondition_invese_single_envir(A, grad, rt::PlaquetteVUMPSRuntime, params, restriction_ipeps, fδEi, iter_precond)
    if fδEi[3] <= iter_precond
        return grad
    end
    δ = fδEi[2]

    A_prime = restriction_ipeps(A)
    A_prime = build_A(A_prime, params)

    env = ObsEnv(rt, A_prime, params.boundary_alg)
    @unpack AL, C, FLu, FLo = env
    AC = ALCtoAC(AL, C)

    gradnew = deepcopy(grad)
    Ni, Nj = size(A_prime)
    @unpack forloop_iter = params
    @unpack ifparallel = params.boundary_alg
    pattern = A_prime.pattern
    for p in 1:length(A_prime)
        i, j = Tuple(findfirst(==(p), pattern))
        ir = Ni + 1 - i
        jr = mod1(j + 1, Nj)
        n = contract_n1(FLo[i,j], AC[i,j], A_prime[i,j], AC[ir,j], FLo[i,jr]; ifparallel, forloop_iter)
        gradnew[:,:,:,:,:,p], _ = linsolve(
            x -> δ * x + Mumap_parallel(AC[i,j], AC[ir,j], FLo[i,j], FLo[i,jr], x; forloop_iter,ifparallel) / n,
            grad[:,:,:,:,:,p];
            isposdef=true, maxiter=1, verbosity=0
        )
    end

    return gradnew
end

function precondition_invese_single_envir(A, grad, env::C4vVUMPSEnv, params, restriction_ipeps, fδEi, iter_precond)
    if fδEi[3] <= iter_precond
        return grad
    end
    δ = fδEi[2]

    @unpack AL, C, FL = env
    AC = ALCtoAC_map(AL, C)

    A_prime = restriction_ipeps(A)
    A_prime = build_A(A_prime, params)
    gradnew = deepcopy(grad)

    @unpack forloop_iter = params
    @unpack ifparallel = params.boundary_alg

    n = contract_n1(FL, AC, A_prime[1], AC, FL; ifparallel, forloop_iter)
    gradnew, _ = linsolve(
        x -> δ * x + Mumap_parallel(AC, AC, FL, FL, x; forloop_iter, ifparallel)/n,
        grad[:,:,:,:,:,1];
        isposdef=true, maxiter=1, verbosity=0
    )

    return reshape(gradnew, size(grad))
end

function precondition_invese_single_envir(A, grad, env::CTMEnv, params, restriction_ipeps, fδEi, iter_precond)
    if fδEi[3] <= iter_precond
        return grad
    end
    δ = fδEi[2]

    @unpack C, T = env
    To = CTCtoT(C, T)

    gradnew = deepcopy(grad)

    @unpack forloop_iter = params
    @unpack ifparallel = params.boundary_alg

    gradnew, _ = linsolve(
        x -> δ * x + Mumap_parallel(T, T, To, To, x; forloop_iter, ifparallel),
        grad[:,:,:,:,:,1];
        isposdef=true, maxiter=1, verbosity=0
    )

    return reshape(gradnew, size(grad))
end
