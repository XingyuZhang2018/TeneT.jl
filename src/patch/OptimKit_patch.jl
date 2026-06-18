using OptimKit: LBFGSInverseHessian, DefaultShouldStop, DefaultHasConverged, _precondition,_retract,_transport!, _scale!, _add!
import OptimKit: LBFGSInverseHessian

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

function Array(H::LBFGSInverseHessian)
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
    # return LBFGSInverseHessian(H.maxlength, H.length, H.first, S, Y, H.ρ, H.α)
end

function CuArray(H::LBFGSInverseHessian)
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

function ROCArray(H::LBFGSInverseHessian)
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

# Save LBFGS state to file
function save_lbfgs_state(alg, state::LBFGSState, filename::String="lbfgs_state.jld2")
    try
        save(filename, "state", state; iotype=IOStream)
        alg.verbosity >= 2 && @info "LBFGS state saved to $filename"
        return true
    catch e
        @error "Failed to save LBFGS state: $e"
        return false
    end
end

# Load LBFGS state from file
function load_lbfgs_state(alg, filename::String="lbfgs_state.jld2")
    try
        state = load(filename, "state"; iotype=IOStream)
        alg.verbosity >= 2 && @info "LBFGS state loaded from $filename"
        return state
    catch e
        alg.verbosity >= 1 && @warn "Failed to load LBFGS state: $e"
        return nothing
    end
end

function optimize_reload(fg, x, alg::LBFGS;
                         resume_from::Union{String,LBFGSState,Nothing}=nothing,
                         save_state_to::Union{String,Nothing}=nothing,
                         save_every::Int=100,
                         precondition=_precondition,
                         (finalize!)=_finalize!,  # default is OptimKit's own _finalize!; callers pass TeneT's closure adapter
                         shouldstop=DefaultShouldStop(alg.maxiter),
                         hasconverged=DefaultHasConverged(alg.gradtol),
                         retract=_retract, inner=_inner, (transport!)=_transport!,
                         (scale!)=_scale!, (add!)=_add!,
                         isometrictransport=(transport! == _transport! && inner == _inner))

    # Try to restore from state
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
        # Restore from saved state
        x = TangentType(initial_state.x)
        f = initial_state.f
        g = TangentType(initial_state.g)
        H = TangentType(initial_state.H)
        numfg = initial_state.numfg
        numiter = initial_state.numiter
        fhistory = copy(initial_state.fhistory)
        normgradhistory = copy(initial_state.normgradhistory)
        t₀ = initial_state.t₀  # Keep original start time for correct total time calculation

        # Recompute current state quantities
        innergg = inner(x, g, g)
        normgrad = sqrt(innergg)

        alg.verbosity >= 2 &&
            @info @sprintf("LBFGS: resuming from iteration %d with f = %.12f, ‖∇f‖ = %.4e",
                          numiter, f, normgrad)
    else
        # Start from scratch
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
        # compute new search direction
        if length(H) > 0
            Hg = let x = x
                H(g, ξ -> precondition(x, ξ), (ξ1, ξ2) -> inner(x, ξ1, ξ2), add!, scale!)
            end
            η = scale!(Hg, -1)
        else
            Pg = precondition(x, deepcopy(g))
            normPg = sqrt(inner(x, Pg, Pg))
            η = scale!(Pg, -0.01 / normPg) # initial guess: scale invariant
        end

        # store current quantities as previous quantities
        xprev = x
        gprev = g
        ηprev = η

        # perform line search
        x, f, g, ξ, α, nfg = alg.linesearch(fg, x, η, (f, g);
                                            initialguess=one(f),
                                            acceptfirst=alg.acceptfirst,
                                            # for some reason, line search seems to converge to solution alpha = 2 in most cases if acceptfirst = false. If acceptfirst = true, the initial value of alpha can immediately be accepted. This typically leads to a more erratic convergence of normgrad, but to less function evaluations in the end.
                                            retract=retract, inner=inner)
        numfg += nfg
        numiter += 1
        x, f, g = finalize!(x, f, g, numiter)
        innergg = inner(x, g, g)
        normgrad = sqrt(innergg)
        push!(fhistory, f)
        push!(normgradhistory, normgrad)

        # transport gprev, ηprev and vectors in Hessian approximation to x
        gprev = transport!(gprev, xprev, ηprev, α, x)
        for k in 1:length(H)
            @inbounds s, y, ρ = H[k]
            s = transport!(s, xprev, ηprev, α, x)
            y = transport!(y, xprev, ηprev, α, x)
            # QUESTION:
            # Do we need to recompute ρ = inv(inner(x, s, y)) if transport is not isometric?
            H[k] = (s, y, ρ)
        end
        ηprev = transport!(deepcopy(ηprev), xprev, ηprev, α, x)

        if isometrictransport
            # TRICK TO ENSURE LOCKING CONDITION IN THE CONTEXT OF LBFGS
            #-----------------------------------------------------------
            # (see A BROYDEN CLASS OF QUASI-NEWTON METHODS FOR RIEMANNIAN OPTIMIZATION)
            # define new isometric transport such that, applying it to transported ηprev,
            # it returns a vector proportional to ξ but with the norm of ηprev
            # still has norm normη because transport is isometric
            normη = sqrt(inner(x, ηprev, ηprev))
            normξ = sqrt(inner(x, ξ, ξ))
            β = normη / normξ
            if !(inner(x, ξ, ηprev) ≈ normξ * normη) # ξ and η are not parallel
                ξ₁ = ηprev
                ξ₂ = scale!(ξ, β)
                ν₁ = add!(ξ₁, ξ₂, +1)
                ν₂ = scale!(deepcopy(ξ₂), -2)
                squarednormν₁ = inner(x, ν₁, ν₁)
                squarednormν₂ = inner(x, ν₂, ν₂)
                # apply Householder transforms to gprev, ηprev and vectors in H
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
            # use cautious update below; see "A Riemannian BFGS Method without
            # Differentiated Retraction for Nonconvex Optimization Problems"
            β = one(normgrad)
        end

        # set up quantities for LBFGS update
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
        if save_state_to !== nothing && save_every != 0 && numiter % save_every == 0
            current_state = LBFGSState(Array(x), f, Array(g), Array(H),
                                        numfg, numiter, copy(fhistory), copy(normgradhistory), t₀)
            save_lbfgs_state(alg, current_state, save_state_to)
        end
        t = time() - t₀
        _hasconverged = hasconverged(x, f, g, normgrad)
        _shouldstop = shouldstop(x, f, g, numfg, numiter, t)

        # check stopping criteria and print info
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
