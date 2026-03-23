# iPEPS optimization: gradient-based (AD/LBFGS) and simple-update entry points
# Ported from TeneT_demo/src/optimise_ipeps.jl, optimise_patch.jl, SUFU.jl
# with ADC4PEPS additions (CTMEnv support, α/β preconditioning, save_interval)

using OptimKit: LBFGSInverseHessian, DefaultShouldStop, DefaultHasConverged,
                _precondition, _retract, _transport!, _scale!, _add!
import OptimKit: LBFGSInverseHessian

# ============================================================================
# LBFGS state for checkpointing
# ============================================================================

struct LBFGSState{T,S}
    x::T
    f::S
    g::T
    H::LBFGSInverseHessian
    numfg::Int
    numiter::Int
    fhistory::Vector{S}
    normgradhistory::Vector{S}
    t₀::Float64
end

# ── Array / GPU conversions for LBFGSInverseHessian ──────────────────────

function Base.Array(H::LBFGSInverseHessian)
    S = []
    Y = []
    ρ = []
    for i in 1:length(H.S)
        if isassigned(H.S, i)
            push!(S, Array(H.S[i]))
            push!(Y, Array(H.Y[i]))
            push!(ρ, H.ρ[i])
        end
    end
    return LBFGSInverseHessian(H.maxlength, S, Y, ρ)
end

function CUDA.CuArray(H::LBFGSInverseHessian)
    S = []
    Y = []
    ρ = []
    for i in 1:length(H.S)
        if isassigned(H.S, i)
            push!(S, CuArray(H.S[i]))
            push!(Y, CuArray(H.Y[i]))
            push!(ρ, H.ρ[i])
        end
    end
    return LBFGSInverseHessian(H.maxlength, S, Y, ρ)
end

function AMDGPU.ROCArray(H::LBFGSInverseHessian)
    S = []
    Y = []
    ρ = []
    for i in 1:length(H.S)
        if isassigned(H.S, i)
            push!(S, ROCArray(H.S[i]))
            push!(Y, ROCArray(H.Y[i]))
            push!(ρ, H.ρ[i])
        end
    end
    return LBFGSInverseHessian(H.maxlength, S, Y, ρ)
end

# ============================================================================
# LBFGS state save / load helpers
# ============================================================================

"""
    save_lbfgs_state(alg, state::LBFGSState, filename)

Save the LBFGS optimizer state to a JLD2 file for checkpointing.
"""
function save_lbfgs_state(alg, state::LBFGSState, filename::String="lbfgs_state.jld2")
    try
        dir = dirname(filename)
        !isempty(dir) && !ispath(dir) && mkpath(dir)
        save(filename, "state", state)
        alg.verbosity >= 2 && @info "LBFGS state saved to $filename"
        return true
    catch e
        @error "Failed to save LBFGS state: $e"
        return false
    end
end

"""
    load_lbfgs_state(alg, filename)

Load a previously saved LBFGS optimizer state from a JLD2 file.
Returns `nothing` on failure.
"""
function load_lbfgs_state(alg, filename::String="lbfgs_state.jld2")
    try
        state = load(filename, "state")
        alg.verbosity >= 2 && @info "LBFGS state loaded from $filename"
        return state
    catch e
        alg.verbosity >= 1 && @warn "Failed to load LBFGS state: $e"
        return nothing
    end
end

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
    M = build_M(A, params)
    rt, err = leading_boundary(rt, M, params.boundary_alg)
    env = VUMPSEnv(rt, M, params.boundary_alg)
    e = expectation_value(params.model, A, env, params)[1]

    Zygote.@ignore begin
        update!(rt′, rt)
        if eltype(e) <: Complex
            fδEierr[4] = abs(imag(e))
        else
            iSy = _arraytype(A[1])(real(1im * const_Sy(params.model.S)))
            i, j = 1, 1
            Ni = size(A, 1)
            id = Ni + 1 - i
            @unpack FLo, ACu, ACd, FRo = env
            My = contract_o1(FLo[i,j], ACu[i,j], A[i,j], ACd[id,j], FRo[i,j], iSy;
                             forloop_iter=params.boundary_alg.forloop_iter)
            n  = contract_n1(FLo[i,j], ACu[i,j], A[i,j], ACd[id,j], FRo[i,j];
                             forloop_iter=params.boundary_alg.forloop_iter)
            fδEierr[4] = abs(My/n)
        end
    end

    return e
end

"""
    energy(A, model, env, env′, fδEierr, params::iPEPSOptimize)

ADC4PEPS-style energy wrapper that accepts a generic environment (CTMEnv or VUMPSEnv)
and a model object. Used when the environment is managed externally (e.g. CTM-based).
"""
function energy(A, model, env, env′, fδEierr, params::iPEPSOptimize)
    A_built = build_A(A, params)
    M = build_M(A_built, params)
    env, err = leading_boundary(M, env, params.boundary_alg)
    e = expectation_value(model, A_built, env, params)[1]

    Zygote.@ignore begin
        update!(env′, env)
        if eltype(e) <: Complex
            fδEierr[4] = abs(imag(e))
        else
            fδEierr[4] = 0.0
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

"""
    _finalize!(x, f, g, iter, rt, rt′, D, χ, χshift, params, t0, fδEierr; restriction_ipeps)

LBFGS iteration callback.  Updates the boundary environment cache, saves
checkpoints and log files, and zeros the gradient when convergence stalls
or imaginary energy exceeds tolerance.
"""
function _finalize!(x, f, g, iter, rt, rt′, D, χ, χshift, params, t0, fδEierr;
                    restriction_ipeps=_restriction_ipeps)
    @unpack folder = params

    fδEierr[3] = iter
    fδEierr[2] = abs(fδEierr[1] - f)
    fδEierr[1] = f
    message = @sprintf("i = %5d\tt = %0.2f sec\tenergy_χ%d = %.15f\tgnorm = %.3e\tEimag = %.3e\n",
                        iter, time() - t0, χ, f, norm(g), fδEierr[4])

    folder0 = joinpath(folder, "D$(D)")
    !(ispath(folder0)) && mkpath(folder0)

    # Reuse environment from the converged boundary
    params.reuse_env && update!(rt, rt′)

    # Save environment to disk
    if hasproperty(params, :ifsave_env) && params.ifsave_env
        folder1 = joinpath(folder, "D$(D)", "VUMPS_rt_env")
        save_rt(folder1, rt; file="χ$(χ).jld2")
    end

    # Print and log
    show_every = hasproperty(params, :show_every) ? params.show_every : 1
    if params.verbosity >= 3 && iter % show_every == 0
        printstyled(message; bold=true, color=:red)
        flush(stdout)

        logfile = open(joinpath(folder0, "history.log"), "a")
        write(logfile, message)
        close(logfile)
    end

    # Save iPEPS checkpoint
    save_every = hasproperty(params, :save_every) ? params.save_every : 0
    if save_every != 0 && iter % save_every == 0
        ipeps_dir = joinpath(folder0, "ipeps", "χ$χ")
        !ispath(ipeps_dir) && mkpath(ipeps_dir)
        save(joinpath(ipeps_dir, "No.$(iter).jld2"), "bcipeps", Array(x))
    end

    # Zero gradient when energy change is negligible or imaginary energy is large
    if abs(fδEierr[2]) < 1e-12 || abs(fδEierr[4]) > 1e-8
        g .= 0
    end

    return x, f, g
end

"""
    _finalize!(x, f, g, iter, env, env′, D, χ, params, t0, fδEierr)

ADC4PEPS-style finalize callback for CTMEnv-based optimization.
"""
function _finalize!(x, f, g, iter, env, env′, D, χ, params, t0, fδEierr)
    @unpack folder = params

    fδEierr[3] = iter
    fδEierr[2] = abs(fδEierr[1] - f)
    fδEierr[1] = f
    message = @sprintf("i = %5d\tt = %0.2f sec\tenergy_χ%d = %.15f\tgnorm = %.3e\tEimag = %.3e\n",
                        iter, time() - t0, χ, f, norm(g), fδEierr[4])

    folder1 = joinpath(folder, "D$(D)")
    !(ispath(folder1)) && mkpath(folder1)
    params.reuse_env && update!(env, env′)

    if hasproperty(params.boundary_alg, :ifsave_env) && params.boundary_alg.ifsave_env
        file = joinpath(params.folder, "D$D", "env", "χ$χ.jld2")
        dir = dirname(file)
        !ispath(dir) && mkpath(dir)
        save_rt(dir, env; file="χ$χ.jld2")
        params.verbosity >= 2 && @info "Saved environment to $file"
    end

    show_every = hasproperty(params, :show_every) ? params.show_every : 1
    if params.verbosity >= 3 && iter % show_every == 0
        printstyled(message; bold=true, color=:red)
        flush(stdout)

        logfile = open(joinpath(folder1, "history.log"), "a")
        write(logfile, message)
        close(logfile)
    end

    save_every = hasproperty(params, :save_every) ? params.save_every : 0
    if save_every != 0 && iter % save_every == 0
        ipeps_dir = joinpath(folder1, "ipeps", "χ$(χ)")
        !ispath(ipeps_dir) && mkpath(ipeps_dir)
        save(joinpath(ipeps_dir, "No.$(iter).jld2"), "bcipeps", Array(x))
    end

    if fδEierr[2] < 1e-12 || fδEierr[4] > 1e-7
        g .= 0
    end

    return x, f, g
end

# ============================================================================
# optimize_reload — LBFGS with checkpointing (resume / save state)
# ============================================================================

"""
    optimize_reload(fg, x, alg::LBFGS; kwargs...) -> (x, f, g, numfg, history)

Run the LBFGS optimizer with optional checkpoint save/load.

# Keyword arguments
- `resume_from`: path to a JLD2 file or an `LBFGSState` to resume from
- `save_state_to`: path to periodically save the optimizer state
- `save_every`: save interval in iterations (default 100)
- `precondition`, `finalize!`, `inner`, `retract`, `transport!`, `scale!`, `add!`:
  OptimKit algorithm hooks
"""
function optimize_reload(fg, x, alg::LBFGS;
                         resume_from::Union{String,LBFGSState,Nothing}=nothing,
                         save_state_to::Union{String,Nothing}=nothing,
                         save_every::Int=100,
                         precondition=_precondition,
                         (finalize!)=_finalize!,
                         shouldstop=DefaultShouldStop(alg.maxiter),
                         hasconverged=DefaultHasConverged(alg.gradtol),
                         retract=_retract, inner=_inner, (transport!)=_transport!,
                         (scale!)=_scale!, (add!)=_add!,
                         isometrictransport=(transport! == _transport! && inner == _inner))

    # Try to restore from saved state
    initial_state = nothing
    if resume_from !== nothing
        if isa(resume_from, String)
            initial_state = load_lbfgs_state(alg, resume_from)
        elseif isa(resume_from, LBFGSState)
            initial_state = resume_from
        end
    end

    # Initialize variables
    if initial_state !== nothing
        TangentType = _arraytype(x)
        x = TangentType(initial_state.x)
        f = initial_state.f
        g = TangentType(initial_state.g)
        H = TangentType(initial_state.H)
        numfg = initial_state.numfg
        numiter = initial_state.numiter
        fhistory = copy(initial_state.fhistory)
        normgradhistory = copy(initial_state.normgradhistory)
        t₀ = initial_state.t₀

        innergg = inner(x, g, g)
        normgrad = sqrt(innergg)

        alg.verbosity >= 2 &&
            @info @sprintf("LBFGS: resuming from iteration %d with f = %.12f, ‖∇f‖ = %.4e",
                           numiter, f, normgrad)
    else
        t₀ = time()
        verbosity = alg.verbosity
        f, g = fg(x)
        numfg = 1
        numiter = 0
        innergg = inner(x, g, g)
        normgrad = sqrt(innergg)
        fhistory = [f]
        normgradhistory = [normgrad]

        TangentType = typeof(g)
        ScalarType = typeof(innergg)
        m = alg.m
        H = LBFGSInverseHessian(m, TangentType[], TangentType[], ScalarType[])

        verbosity >= 2 &&
            @info @sprintf("LBFGS: initializing with f = %.12f, ‖∇f‖ = %.4e", f, normgrad)
    end

    t = time() - t₀
    _hasconverged = hasconverged(x, f, g, normgrad)
    _shouldstop = shouldstop(x, f, g, numfg, numiter, t)
    verbosity = alg.verbosity

    while !(_hasconverged || _shouldstop)
        # Compute new search direction
        if length(H) > 0
            Hg = let x = x
                H(g, ξ -> precondition(x, ξ), (ξ1, ξ2) -> inner(x, ξ1, ξ2), add!, scale!)
            end
            η = scale!(Hg, -1)
        else
            Pg = precondition(x, deepcopy(g))
            normPg = sqrt(inner(x, Pg, Pg))
            η = scale!(Pg, -0.01 / normPg)  # initial guess: scale invariant
        end

        # Store current quantities as previous
        xprev = x
        gprev = g
        ηprev = η

        # Perform line search
        x, f, g, ξ, α, nfg = alg.linesearch(fg, x, η, (f, g);
                                              initialguess=one(f),
                                              acceptfirst=alg.acceptfirst,
                                              retract=retract, inner=inner)
        numfg += nfg
        numiter += 1
        x, f, g = finalize!(x, f, g, numiter)
        innergg = inner(x, g, g)
        normgrad = sqrt(innergg)
        push!(fhistory, f)
        push!(normgradhistory, normgrad)

        # Transport gprev, ηprev and Hessian approximation to new x
        gprev = transport!(gprev, xprev, ηprev, α, x)
        for k in 1:length(H)
            @inbounds s, y, ρ = H[k]
            s = transport!(s, xprev, ηprev, α, x)
            y = transport!(y, xprev, ηprev, α, x)
            H[k] = (s, y, ρ)
        end
        ηprev = transport!(deepcopy(ηprev), xprev, ηprev, α, x)

        if isometrictransport
            # Locking condition trick for Riemannian LBFGS
            normη = sqrt(inner(x, ηprev, ηprev))
            normξ = sqrt(inner(x, ξ, ξ))
            β = normη / normξ
            if !(inner(x, ξ, ηprev) ≈ normξ * normη)  # ξ and η not parallel
                ξ₁ = ηprev
                ξ₂ = scale!(ξ, β)
                ν₁ = add!(ξ₁, ξ₂, +1)
                ν₂ = scale!(deepcopy(ξ₂), -2)
                squarednormν₁ = inner(x, ν₁, ν₁)
                squarednormν₂ = inner(x, ν₂, ν₂)
                # Householder transforms
                gprev = add!(gprev, ν₁, -2 * inner(x, ν₁, gprev) / squarednormν₁)
                gprev = add!(gprev, ν₂, -2 * inner(x, ν₂, gprev) / squarednormν₂)
                for k in 1:length(H)
                    @inbounds s, y, ρ = H[k]
                    s = add!(s, ν₁, -2 * inner(x, ν₁, s) / squarednormν₁)
                    s = add!(s, ν₂, -2 * inner(x, ν₂, s) / squarednormν₂)
                    y = add!(y, ν₁, -2 * inner(x, ν₁, y) / squarednormν₁)
                    y = add!(y, ν₂, -2 * inner(x, ν₂, y) / squarednormν₂)
                    H[k] = (s, y, ρ)
                end
                ηprev = ξ₂
            end
        else
            # Cautious update
            β = one(normgrad)
        end

        # LBFGS update
        y = add!(scale!(deepcopy(g), 1 / β), gprev, -1)
        s = scale!(ηprev, α)
        innersy = inner(x, s, y)
        innerss = inner(x, s, s)

        if innersy / innerss > normgrad / 10000
            norms = sqrt(innerss)
            ρ = innerss / innersy
            push!(H, (scale!(s, 1 / norms), scale!(y, 1 / norms), ρ))
        end

        # Periodically save state
        if save_state_to !== nothing && numiter % save_every == 0
            current_state = LBFGSState(Array(x), f, Array(g), Array(H),
                                        numfg, numiter, copy(fhistory),
                                        copy(normgradhistory), t₀)
            save_lbfgs_state(alg, current_state, save_state_to)
        end

        t = time() - t₀
        _hasconverged = hasconverged(x, f, g, normgrad)
        _shouldstop = shouldstop(x, f, g, numfg, numiter, t)

        if _hasconverged || _shouldstop
            break
        end
        verbosity >= 3 &&
            @info @sprintf("LBFGS: iter %4d, time %7.2f s: f = %.12f, ‖∇f‖ = %.4e, α = %.2e, m = %d, nfg = %d",
                           numiter, t, f, normgrad, α, length(H), nfg)
    end

    if _hasconverged
        verbosity >= 2 &&
            @info @sprintf("LBFGS: converged after %d iterations and time %.2f s: f = %.12f, ‖∇f‖ = %.4e",
                           numiter, t, f, normgrad)
    else
        verbosity >= 1 &&
            @warn @sprintf("LBFGS: not converged to requested tol after %d iterations and time %.2f s: f = %.12f, ‖∇f‖ = %.4e",
                           numiter, t, f, normgrad)
    end

    history = [fhistory normgradhistory]
    return x, f, g, numfg, history
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

    _precond(x, g) = params.ifprecondition ?
        precondition_invese_single_envir(x, g, rt, params, restriction_ipeps, fδEierr, params.iter_precond) : g

    state_path = joinpath(params.folder, "D$(D)", "lbfgs_checkpoint")

    for _ in 1:params.maxiter_restart
        A, e, eg, fgnum, history = optimize_reload(fg, A, alg;
            resume_from  = params.ifload_lbfgs ? joinpath(state_path, "χ$χ.jld2") : nothing,
            save_state_to = params.ifsave_lbfgs ? joinpath(state_path, "χ$χ.jld2") : nothing,
            save_every    = params.save_every,
            precondition  = _precond,
            inner         = _inner,
            finalize!     = (x, f, g, iter) -> _finalize!(x, f, g, iter, rt, rt′, D, χ, χshift, params, t0, fδEierr; restriction_ipeps)
        )
        χ += χshift
        enew, = observable(A, χ, params_obs; restriction_ipeps)
        rt = initialize_env(A, D, χ, params; restriction_ipeps)
        rt′ = deepcopy(rt)
        if abs(real(enew[1]) - e) < 1e-7 && history[end-1] < 1e-5
            break
        end
    end
end

"""
    optimise_ipeps(A, χ, χshift, model, params::GradientOptimize; restriction_ipeps=_restriction_ipeps)

ADC4PEPS-style entry point with an explicit `model` argument and generic
environment (CTMEnv / VUMPSEnv).
"""
function optimise_ipeps(A, χ::Int, χshift::Int, model, params::GradientOptimize;
                        restriction_ipeps=_restriction_ipeps)
    D = size(A, 1)

    A′ = restriction_ipeps(A)
    env = initialize_env(A′, χ, params.boundary_alg;
                         file=joinpath(params.folder, "D$D", "env", "$χ.jld2"))
    env′ = deepcopy(env)
    fδEierr = [1.0, 1.0, 0.0, 0.0]

    function f(A)
        A = restriction_ipeps(A)
        return real(energy(A, model, env, env′, fδEierr, params))
    end

    function fg(x)
        t1 = time()
        e, vjp = pullback(f, x)
        params.verbosity >= 2 && printstyled(" forward calculation took $(round(time() - t1, digits = 2)) s\n"; bold=true, color=:green)
        reclaim(x)
        t2 = time()
        g = vjp(1)[1]
        params.verbosity >= 2 && printstyled("backward calculation took $(round(time() - t2, digits = 2)) s\n"; bold=true, color=:green)
        reclaim(g)
        return e, g
    end

    @unpack optimizer, iter_precond, α, β = params
    t0 = time()
    _precond(x, g) = params.ifprecondition ?
        precondition_invese_single_envir(x, g, env′, params, restriction_ipeps, fδEierr, params.iter_precond, model, α, β) : g

    state_path = joinpath(params.folder, "D$(D)", "lbfgs_checkpoint")

    for _ in 1:params.maxiter_restart
        resume_from   = params.ifload_lbfgs ? joinpath(state_path, "χ$χ.jld2") : nothing
        save_state_to = params.ifsave_lbfgs ? joinpath(state_path, "χ$χ.jld2") : nothing
        A, e, eg, numfg, history = optimize_reload(fg, A, optimizer;
            resume_from,
            save_state_to,
            save_every    = params.save_every,
            precondition  = _precond,
            inner         = _inner,
            finalize!     = (x, f, g, iter) -> _finalize!(x, f, g, iter, env, env′, D, χ, params, t0, fδEierr)
        )
        χ += χshift
        params_obs = deepcopy(params)
        params_obs.boundary_alg.maxiter = params.boundary_alg.maxiter * 10
        enew, = observable(A, χ, model, params_obs; restriction_ipeps)
        A′ = restriction_ipeps(A)
        env = initialize_env(A′, χ, params.boundary_alg;
                             file=joinpath(params.folder, "D$D", "env", "$χ.jld2"))
        env′ = deepcopy(env)
        if abs(real(enew[1]) - e) < 1e-7 && history[end-1] < 1e-5
            break
        end
    end
end

# ============================================================================
# Simple Update optimization
# ============================================================================

"""
    optimise_ipeps(A, χ, params::SUOptimize)

Simple Update (SU) optimization of iPEPS.  Iteratively applies imaginary-time
evolution gates via SVD truncation and monitors the energy via VUMPS.
"""
function optimise_ipeps(A, χ::Int, params::SUOptimize)
    D = size(A[1], 1)
    A = build_A(A, params)
    for i in 1:params.maxiter
        A = hv_SU_update(A, params)
        A = map(x -> x / norm(x), A)
        M = build_M(A, params)
        rt = VUMPSRuntime(M, χ, params.boundary_alg)
        rt, _ = leading_boundary(rt, M, params.boundary_alg)
        env = VUMPSEnv(rt, M, params.boundary_alg)
        e = real(expectation_value(params.model, A, env, params)[1])
        params.verbosity >= 3 && println("SU@$i: energy: $e")
    end
    return A
end

# ============================================================================
# Partition function (Z) helper
# ============================================================================

"""
    Z(M, rt, alg)

Compute the partition function ratio `λ_AC / λ_C` from the VUMPS environment.
"""
function Z(M, rt, alg)
    @unpack AL, AR, C, FL, FR = rt
    AC = ALCtoAC(AL, C)
    λAC, = ACenv(AC, FL, M, FR; ifvalue=true, alg)
    λC,  = Cenv(C, FL, FR; ifvalue=true, alg)
    return real(λAC[1] / λC[1])
end
