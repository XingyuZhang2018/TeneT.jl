# iPEPS optimization: gradient-based (AD/LBFGS) and simple-update entry points
# Ported from TeneT_demo/src/optimise_ipeps.jl, optimise_patch.jl, SUFU.jl
# with ADC4PEPS additions (CTMEnv support, α/β preconditioning, save_interval)


# ============================================================================
# Energy wrapper for AD (forward pass inside Zygote pullback)
# ============================================================================

"""
    energy(A, rt, rt′, fδEierr, params::iPEPSOptimize)

Compute the energy of iPEPS tensors `A` using boundary environment `rt`.
This function is designed to be differentiated through by Zygote.
`rt′` is updated in-place (outside AD) with the converged environment for reuse.
`fδEierr` tracks [last_energy, delta_energy, iteration, imag_energy].
"""
function energy(A, rt, rt′, fδEierr, params::iPEPSOptimize)
    A = build_A(A, params)
    rt, err = leading_boundary(rt, A, params.boundary_alg)
    env = ObsEnv(rt, A, params.boundary_alg)
    e = energy_value(params.model, A, env, params)[1]

    ignore_derivatives() do
        update!(rt′, rt)
        if eltype(e) <: Complex
            fδEierr[4] = abs(imag(e))
        else
            iSy = real(1im * const_Sy(params.model.S))
            d = size(iSy, 1)
            Id = Matrix{Float64}(I, d, d)
            n = round(Int, log(d, size(A[1], 5)))
            iSy = _arraytype(A[1])(reduce(kron, fill(Id, n - 1); init = iSy))
            @unpack forloop_iter, ifparallel = params.boundary_alg
            i, j = 1, 1
            Ni,Nj = size(A)
            if env isa VUMPSEnv
                @unpack FLo, ACu, ACd, FRo = env
                id = Ni + 1 - i
                My = contract_o_11(FLo[i,j], ACu[i,j], A[i,j], ACd[id,j], FRo[i,j], iSy; forloop_iter, ifparallel)
                n  = contract_n_11(FLo[i,j], ACu[i,j], A[i,j], ACd[id,j], FRo[i,j]; forloop_iter, ifparallel)
                fδEierr[4] = abs(My/n)
            elseif env isa PlaquetteVUMPSEnv
                @unpack AL, C, FLu, FLo = env
                AC = ALCtoAC(AL, C)
                ir = 2
                jr = params.model.lattice isa Square ? mod1(j + 1, Nj) : mod1(Nj - j, Nj)
                My = contract_o_11(FLo[i,j],AC[i,j],A[i,j],AC[ir,j],FLo[i,jr], iSy; ifparallel, forloop_iter)
                n = contract_n_11(FLo[i,j],AC[i,j],A[i,j],AC[ir,j],FLo[i,jr]; ifparallel, forloop_iter)
                fδEierr[4] = abs(My/n)
            end
        end
    end

    return e
end

# ============================================================================
# LBFGS inner product
# ============================================================================

_inner(x, dx1, dx2) = real(dot(dx1, dx2))

# ============================================================================
# Finalize callback — called after each LBFGS iteration
# ============================================================================

# NOTE: Always called through a 4-arg closure adapter, never invoked directly.
# The full 11-arg signature is specific to iPEPS optimization and does not follow
# OptimKit's default finalize! convention (x, f, g, iter).  The closure in
# optimise_ipeps captures rt, rt′, D, χ, params, t0, and fδEierr so that
# optimize_reload only ever sees the standard 4-arg interface.
"""
    _finalize!(x, f, g, iter, rt, rt′, D, χ, params, t0, fδEierr)

LBFGS iteration callback.  Updates the boundary environment cache, saves
checkpoints and log files, and zeros the gradient when convergence stalls
or imaginary energy exceeds tolerance.
"""
function _finalize!(x, f, g, iter, rt, rt′, D, χ, params, t0, fδEierr)
    @unpack folder = params

    fδEierr[3] = iter
    fδEierr[2] = abs(fδEierr[1] - f)
    fδEierr[1] = f
    message = @sprintf("i = %5d\tt = %0.2f sec\tenergy_χ%d = %.15f\tgnorm = %.3e\tEimag = %.3e\n",
                        iter, time() - t0, χ, f, norm(g), fδEierr[4])

    folder0 = joinpath(folder, "D$(D)")
    !(ispath(folder0)) && mkpath(folder0)

    # Reuse environment from the last iteration to speed up convergence
    params.reuse_env && update!(rt, rt′)

    # Save environment to disk
    if params.ifsave_env
        folder1 = joinpath(folder, "D$(D)", "environment")
        save_rt(folder1, rt; file="χ$(χ).jld2")
    end

    # Print and log
    if params.verbosity >= 3 && iter % params.show_every == 0
        printstyled(message; bold=true, color=:red)
        flush(stdout)

        logfile = open(joinpath(folder0, "history.log"), "a")
        write(logfile, message)
        close(logfile)
    end

    # Save iPEPS checkpoint
    if params.save_every  != 0 && iter % params.save_every  == 0
        ipeps_dir = joinpath(folder0, "ipeps", "χ$χ")
        !ispath(ipeps_dir) && mkpath(ipeps_dir)
        save(joinpath(ipeps_dir, "No.$(iter).jld2"), "bcipeps", Array(x); iotype=IOStream)
    end

    if abs(fδEierr[2]) < 1e-12 || abs(fδEierr[4]) > 1e-8
        g .= 0
    end

    return x, f, g
end

# ============================================================================
# Main entry: gradient-based (AD) optimization
# ============================================================================

"""
    optimise_ipeps(A, χ, χshift, params::GradientOptimize; restriction_ipeps=_restriction_ipeps)

Unified gradient-based iPEPS optimization using automatic differentiation
and LBFGS. Supports VUMPS boundary with multi-cell patterns.

# Arguments
- `A`: raw iPEPS parameter array of shape `(D, D, D, D, d, Nsites)`
- `χ`: initial boundary bond dimension
- `χshift`: increment to `χ` after each restart
- `params`: `GradientOptimize` containing model, boundary algorithm, I/O options
- `restriction_ipeps`: symmetry restriction function applied to `A` before building tensors

After each LBFGS run completes (converges or hits `maxiter`), the bond dimension
is increased by `χshift` and the optimizer restarts from the current state.
This loop repeats up to `params.maxiter_restart` times.
"""
function optimise_ipeps(A, χ::Int, χshift::Int, params::GradientOptimize;
                        restriction_ipeps=_restriction_ipeps)
    D = size(A, 1)
    rt = initialize_env(A, D, χ, params; restriction_ipeps)
    rt′ = deepcopy(rt)
    fδEierr = [1.0, 1.0, 0.0, 0.0]

    params_obs = deepcopy(params)
    params_obs.boundary_alg.maxiter = params.boundary_alg.maxiter * 10

    function fenergy(A)
        _G_cache[] = nothing
        A = restriction_ipeps(A)
        return real(energy(A, rt, rt′, fδEierr, params))
    end

    function fg(x)
        t1 = time()
        e, vjp = pullback(fenergy, x)
        params.verbosity >= 2 && printstyled(" forward calculation took $(round(time() - t1, digits = 2)) s\n"; bold=true, color=:green)
        reclaim(x)
        t2 = time()
        g = vjp(1)[1]
        params.verbosity >= 2 && printstyled("backward calculation took $(round(time() - t2, digits = 2)) s\n"; bold=true, color=:green)
        reclaim(g)
        return e, g
    end

    alg = params.optimizer
    t0 = time()

    _precond(x, g) = params.ifprecondition ? precondition_invese_single_envir(x, g, rt, params, restriction_ipeps, fδEierr, params.iter_precond) : g

    state_path = joinpath(params.folder, "D$(D)", "lbfgs_checkpoint")

    local e, eg, fgnum, history
    for _ in 1:params.maxiter_restart
        A, e, eg, fgnum, history = optimize_reload(fg, A, alg;
            resume_from  = params.ifload_lbfgs ? joinpath(state_path, "χ$χ.jld2") : nothing,
            save_state_to = params.ifsave_lbfgs ? joinpath(state_path, "χ$χ.jld2") : nothing,
            save_every    = params.save_every,
            precondition  = _precond,
            inner         = _inner,
            finalize!     = (x, f, g, iter) -> _finalize!(x, f, g, iter, rt, rt′, D, χ, params, t0, fδEierr)
        )
        χ += χshift
        enew, = observable(A, χ, params_obs; restriction_ipeps)
        rt = initialize_env(A, D, χ, params; restriction_ipeps)
        rt′ = deepcopy(rt)
        if abs(real(enew[1]) - e) < 1e-7 && history[end-1] < 1e-5
            break
        end
    end
    return A, e, eg, fgnum, history
end