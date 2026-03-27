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
        A_prime = guage_transfer(A, [Gh, Gv], params)
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

"""
    precondition_invese_single_envir(A, grad, env::CTMEnv, params, restriction_ipeps, fδEi, iter_precond)

Precondition the gradient using the CTM environment (norm-only).
"""
function precondition_invese_single_envir(A, grad, env::CTMEnv, params, restriction_ipeps, fδEi, iter_precond)
    t0 = time()
    if fδEi[3] <= iter_precond
        return grad
    end
    δ = fδEi[2]
    A = restriction_ipeps(A)

    @unpack C, T = env
    @unpack forloop_iter, ifparallel = params.boundary_alg
    To = CTCtoT(C, T)

    gradnew, _ = linsolve(grad; isposdef=true, maxiter=1, verbosity=0) do x
        return δ * x + Mumap_parallel(T, T, To, To, x; forloop_iter, ifparallel)
    end

    params.verbosity >= 2 && printstyled("precondition calculation took $(round(time() - t0, digits=2)) s\n"; bold=true, color=:green)
    return gradnew
end

"""
    precondition_invese_single_envir(A, grad, env::VUMPSEnv, params, restriction_ipeps, fδEi, iter_precond)

Precondition the gradient using the VUMPS environment directly.
"""
function precondition_invese_single_envir(A, grad, env::VUMPSEnv, params, restriction_ipeps, fδEi, iter_precond)
    if fδEi[3] <= iter_precond
        return grad
    end
    t0 = time()
    δ = fδEi[2]
    A = restriction_ipeps(A)
    h1, h2 = Zygote.@ignore _arraytype(A).(hamiltonian_trunc(model))
    D, d = size(A)[[1,5]]
    Dh = size(h1, 3)

    @unpack AL, C, T = env
    @unpack forloop_iter, ifparallel = params.boundary_alg
    AC = ALCtoAC(AL, C)

    function f(x)
        Mumap_parallel(AC, AC, T, T, x; forloop_iter, ifparallel)
    end
    gradnew, _ = linsolve(x -> δ * x + f(x), grad; isposdef=true, maxiter=1, verbosity=0)

    params.verbosity >= 2 && printstyled("precondition calculation took $(round(time() - t0, digits=2)) s\n"; bold=true, color=:green)
    return gradnew
end

"""
    precondition_invese_N(A, g, env::CTMEnv, params, restriction_ipeps, fδEi, iter_precond)

Precondition the gradient using the norm transfer matrix with ForwardDiff
for the Hessian-vector product. CTM environment version.
"""
function precondition_invese_N(A, g, env::CTMEnv, params, restriction_ipeps, fδEi, iter_precond)
    t0 = time()
    if fδEi[3] <= iter_precond
        return g
    end
    δ = fδEi[2]

    algr = deepcopy(params.boundary_alg)
    algr.verbosity = 0
    gnew, _ = linsolve(g; isposdef=true, maxiter=1, verbosity=0) do x
        function f(Au)
            @unpack C, T = env
            @unpack forloop_iter, ifparallel = algr
            To = CTCtoT(C, T)

            ForwardDiff.gradient(y -> (@tensor Mumap_parallel(T, T, To, To, restriction_ipeps(Au); forloop_iter, ifparallel)[a,b,c,d,p] * conj(restriction_ipeps(y))[a,b,c,d,p]), A)
        end
        gN = ForwardDiff.derivative(t -> f(A + t * x), 0.0)
        return δ * x + gN
    end

    params.verbosity >= 2 && printstyled("precondition calculation took $(round(time() - t0, digits=2)) s\n"; bold=true, color=:green)
    return gnew
end

"""
    precondition_invese_hessian(A, grad, rt, rt_prime, params, restriction_ipeps, fδEi, iter_precond)

Precondition the gradient using an approximate Hessian-vector product.
Constructs a two-layer norm network and computes the action of the Hessian.
"""
function precondition_invese_hessian(A, grad, rt, rt_prime, params, restriction_ipeps, fδEi, iter_precond)
    δ = fδEi[2]

    function contract_n1_hess(FLo, ACu, A, Ap, ACd, FRo; forloop_iter)
        D1, D2, D3, D4, _ = size(A)
        @tensor M[a,f,b,g,c,h,d,m] := A[a,b,c,d,e] * Ap[f,g,h,m,e]
        return sum(oc1_leg3(FLo, ACu, M, ACd, FRo; forloop_iter))
    end

    function build_M_hess(A, Ap, params)
        D = size(A[1], 1)
        len = length(unique(params.pattern))
        return StructArray([begin
            @tensor M[a,f,b,g,c,h,d,m] := A[i][a,b,c,d,e] * Ap[i][f,g,h,m,e]
            M
        end for i in 1:len], params.pattern)
    end

    function ipeps_norm_hess(A, Ap, rt, rt_prime, params::iPEPSOptimize)
        M = build_M_hess(A, Ap, params)
        rt, _ = leading_boundary(rt, M, params.boundary_alg)
        Zygote.@ignore update!(rt_prime, rt)
        env = VUMPSEnv(rt, M, params.boundary_alg)
        @unpack ACu, ACd, FLo, FRo = env
        n = contract_n1_hess(FLo[1], ACu[1], A[1], Ap[1], conj(ACd[1]), FRo[1]; params.forloop_iter)
        return real(n)
    end

    function f(A_in, Ap_in)
        A_in = restriction_ipeps(A_in)
        A_in = build_A(A_in, params)
        Ap_in = restriction_ipeps(Ap_in)
        Ap_in = build_A(Ap_in, params)
        return ipeps_norm_hess(A_in, Ap_in, rt, rt_prime, params)
    end

    @show dot(grad, Zygote.gradient(x1 -> f(x1, conj(A)), A)[1])
    @show ForwardDiff.derivative(t -> f(t * A, conj(A)), 0)[1]
    return grad
end

"""
    precondition_invese_BP_envir(A, grad, rt, params, restriction_ipeps, fδEi, iter_precond)

Precondition the gradient using a Belief Propagation (BP) environment approximation.
TeneT_demo version: uses the double-layer transfer matrix.
"""
function precondition_invese_BP_envir(A, grad, rt, params, restriction_ipeps, fδEi, iter_precond)
    if fδEi[2] > 0.01 || fδEi[3] <= 20
        return grad
    end
    δ = fδEi[2]
    A = restriction_ipeps(A)
    A = build_A(A, params)

    D = size(A[1], 1)
    # Construct double-layer tensor locally for BP contraction
    M = StructArray([begin
        @tensor T[a,f,b,g,c,h,d,m] := A[i][a,b,c,d,e] * conj(A[i])[f,g,h,m,e]
        reshape(T, D^2, D^2, D^2, D^2)
    end for i in 1:length(unique(params.pattern))], params.pattern)
    B = _arraytype(M[1])(randn(ComplexF64, D^2))
    error_val = 1.0
    Z = 1.0
    for i in 1:100
        @tensor B[a] := M[1][a,b,c,d] * B[d] * B[c] * B[b]
        Z_n = dot(B, B)
        normalize!(B)
        error_val = norm(Z_n - Z)
        if error_val < 1e-16
            break
        end
        Z = Z_n
    end
    println("================================")
    @show error_val, Z
    println("================================")
    gradnew = deepcopy(grad)
    Ni = size(M)[1]
    reB = reshape(B, D, D)
    for p in 1:length(M)
        i, j = Tuple(findfirst(==(p), M.pattern))
        ir = Ni + 1 - i
        n = @tensor M[1][a,b,c,d] * B[d] * B[c] * B[b] * B[a]
        gradnew[p], _ = linsolve(
            x -> δ * x + (@tensor xout[e,f,g,h,p] := x[a,b,c,d,p] * reB[a,e] * reB[b,f] * reB[c,g] * reB[d,h]) / n,
            grad[p];
            isposdef=true, maxiter=1
        )
    end

    return gradnew
end

"""
    precondition_invese_BP_envir(A, grad, params, restriction_ipeps, fδEi, iter_precond)

Precondition the gradient using a Belief Propagation environment.
ADC4PEPS version: uses the single-layer tensor directly.
"""
function precondition_invese_BP_envir(A, grad, params, restriction_ipeps, fδEi, iter_precond)
    if fδEi[3] <= iter_precond
        return grad
    end
    δ = fδEi[2]
    A = restriction_ipeps(A)
    D = size(A, 1)

    B = _arraytype(A)(rand(eltype(A), D, D))

    error_val = 1.0
    Z = 1.0
    for i in 1:100
        @tensor B_new[c,g] := A[a,b,c,d,p] * B[a,e] * B[b,f] * B[d,h] * conj(A[e,f,g,h,p])
        B = B_new
        Z_n = dot(B, B)
        normalize!(B)
        error_val = norm(Z_n - Z)
        if error_val < 1e-16
            break
        end
        Z = Z_n
    end
    gradnew = deepcopy(grad)

    n = @tensor A[a,b,c,d,p] * B[a,e] * B[b,f] * B[d,h] * B[c,g] * conj(A[e,f,g,h,p])
    gradnew, _ = linsolve(
        x -> δ * x + (@tensor out[e,f,g,h,p] := x[a,b,c,d,p] * B[a,e] * B[b,f] * B[c,g] * B[d,h]) / n,
        grad;
        isposdef=true, maxiter=1
    )

    return gradnew
end

"""
    precondition_invese_single_chi1_envir(A, grad, params, restriction_ipeps, fδEi, iter_precond)

Precondition the gradient using a χ=1 CTM environment approximation.
Cheaper than full environment preconditioning but still captures the leading
contribution of the transfer matrix.
"""
function precondition_invese_single_chi1_envir(A, grad, params, restriction_ipeps, fδEi, iter_precond)
    if fδEi[3] <= iter_precond
        return grad
    end
    δ = fδEi[2]
    A = restriction_ipeps(A)

    algr = deepcopy(params.boundary_alg)
    algr.chi = 1
    env, _ = environment(A, algr)

    @unpack C, T = env
    @unpack forloop_iter, ifparallel = params.boundary_alg
    To = CTCtoT(C, T)

    gradnew, _ = linsolve(grad; isposdef=true, maxiter=1, verbosity=0) do x
        return δ * x + Mumap_parallel(T, T, To, To, x; forloop_iter, ifparallel)
    end

    return gradnew
end

"""
    environment_FWAD(M, env::CTMEnv, alg::Algorithm)

Compute CTM environment using forward-mode AD (ForwardDiff).
First runs `maxiter` steps with extracted values (no AD), then `maxiter_ad` steps
with full forward-mode differentiation.
"""
function environment_FWAD(M, env::CTMEnv, alg::Algorithm)
    err = Inf
    for i = 1:alg.maxiter
        env = leftmove(ForwardDiff.value.(M), env, alg)
    end

    for i = 1:alg.maxiter_ad
        env = leftmove(M, env, alg)
    end

    return env, err
end
