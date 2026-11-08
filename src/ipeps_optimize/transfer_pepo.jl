export partition_logdensity, transfer_pepo_density, transfer_pepo_observable
export optimise_transfer_pepo, optimize_transfer_pepo

function _transfer_pepo_boundary_alg(alg::VUMPS{General})
    alg = deepcopy(alg)
    _effective_grid(alg) === nothing ||
        throw(ArgumentError("transfer-PEPO optimization currently supports serial VUMPS{General}; Slice2D grids are not supported."))
    alg.ifupdown = false
    alg.ifparallelupdown = false
    return alg
end

function _transfer_pepo_boundary_alg(alg::VUMPS{C4v})
    alg = deepcopy(alg)
    _effective_grid(alg) === nothing ||
        throw(ArgumentError("transfer-PEPO optimization currently supports serial VUMPS{C4v}; Slice2D grids are not supported."))
    return alg
end

function _transfer_pepo_boundary_alg(alg)
    throw(ArgumentError("transfer-PEPO optimization requires VUMPS{General} or VUMPS{C4v}; got $(typeof(alg))"))
end

function _transfer_pepo_params(params::GradientOptimize)
    params = deepcopy(params)
    params.boundary_alg = _transfer_pepo_boundary_alg(params.boundary_alg)
    return params
end

function _transfer_pepo_params_for_chi(params::GradientOptimize, χ::Integer)
    params = _transfer_pepo_params(params)
    params.boundary_alg.forloop_iter = min(params.boundary_alg.forloop_iter, max(1, Int(χ)))
    return params
end

_transfer_pepo_vumps_chi(rt::VUMPSRuntime) = size(rt.C[1], 1)
_transfer_pepo_vumps_chi(rt::C4vVUMPSEnv) = size(rt.C, 1)

function _transfer_pepo_update_stop!(g, f, fδ, params::GradientOptimize, χ::Integer; stall_tol=1e-12)
    previous = fδ[1]
    f_real = real(f)
    f_abs = abs(f_real)
    δ = isfinite(previous) ? abs(real(previous) - f_real) :
        (isfinite(f_abs) ? max(f_abs, one(f_abs)) : one(f_abs))
    fδ[1] = real(f)
    fδ[2] = δ
    params.last_stop_χ = Int(χ)
    params.last_stop_eimag = 0.0

    if δ < stall_tol
        params.last_stop_reason = :objective_stall
        g .= zero(eltype(g))
    else
        params.last_stop_reason = :running
    end
    return δ
end

function _transfer_pepo_precondition(A, grad, rt_norm, params::GradientOptimize,
                                     restriction_ipeps, fδobjective)
    params.ifprecondition || return grad
    length(fδobjective) >= 2 || return grad
    isfinite(fδobjective[2]) && fδobjective[2] > 0 || return grad
    return precondition_invese_single_envir(A, grad, rt_norm, params,
                                            restriction_ipeps, fδobjective,
                                            params.iter_precond)
end

function _build_transfer_pepo_A(A, params::GradientOptimize; restriction_ipeps)
    _G_cache[] = nothing
    return build_A(restriction_ipeps(A), params)
end

function partition_logdensity(rt::VUMPSRuntime, M::StructArray, alg::VUMPS{General})
    AC = ALCtoAC(rt.AL, rt.C)
    lambda_ACs, _ = ACenv(AC, rt.FL, M, rt.FR; alg)
    lambda_Cs, _ = Cenv(rt.C, rt.FL, rt.FR; alg)
    lambda_AC = lambda_ACs[1]
    lambda_C = lambda_Cs[1]
    ratio = lambda_AC / lambda_C
    return (
        logdensity = real(log(abs(ratio))),
        lambda_AC = lambda_AC,
        lambda_C = lambda_C,
        eigenvalue_ratio = ratio,
    )
end

function partition_logdensity(rt::C4vVUMPSEnv, M::StructArray, alg::VUMPS{C4v})
    AC = ALCtoAC_map(rt.AL, rt.C)
    M_c4v = _c4v_local_tensor(M)
    lambda_AC, _ = ACenv_c4v(AC, rt.FL, M_c4v; alg)
    lambda_C, _ = Cenv_c4v(rt.C, rt.FL; alg)
    ratio = lambda_AC / lambda_C
    return (
        logdensity = real(log(abs(ratio))),
        lambda_AC = lambda_AC,
        lambda_C = lambda_C,
        eigenvalue_ratio = ratio,
    )
end

function _transfer_pepo_refresh_envs(rt_norm, rt_transfer,
                                     A_built::StructArray, params::GradientOptimize)
    ignore_derivatives() do
        rt_norm_new, err_norm = leading_boundary(rt_norm, A_built, params.boundary_alg)
        M_transfer = transfer_layer(params.model, A_built)
        rt_transfer_new, err_transfer = leading_boundary(rt_transfer, M_transfer, params.boundary_alg)
        return rt_norm_new, rt_transfer_new, real(err_norm), real(err_transfer)
    end
end

function _transfer_pepo_fixed_env_density(A_built::StructArray, rt_norm,
                                          rt_transfer, params::GradientOptimize)
    M_transfer = transfer_layer(params.model, A_built)
    norm_stat = partition_logdensity(rt_norm, A_built, params.boundary_alg)
    transfer_stat = partition_logdensity(rt_transfer, M_transfer, params.boundary_alg)
    log_density = real(transfer_stat.logdensity - norm_stat.logdensity)
    return (
        log_density = log_density,
        objective = -log_density,
        log_norm = norm_stat.logdensity,
        log_transfer = transfer_stat.logdensity,
        norm = norm_stat,
        transfer = transfer_stat,
    )
end

function transfer_pepo_density(A, rt_norm, rt_transfer,
                               params::GradientOptimize; restriction_ipeps=_restriction_ipeps)
    params = _transfer_pepo_params_for_chi(params, _transfer_pepo_vumps_chi(rt_norm))
    A_built = _build_transfer_pepo_A(A, params; restriction_ipeps)
    rt_norm_new, rt_transfer_new, err_norm, err_transfer =
        _transfer_pepo_refresh_envs(rt_norm, rt_transfer, A_built, params)
    obs = _transfer_pepo_fixed_env_density(A_built, rt_norm_new, rt_transfer_new, params)
    return merge(obs, (
        err_norm = err_norm,
        err_transfer = err_transfer,
        rt_norm = rt_norm_new,
        rt_transfer = rt_transfer_new,
    ))
end

function transfer_pepo_correlation_length(rt::VUMPSRuntime; howmany::Integer=5)
    howmany > 0 || throw(ArgumentError("howmany must be positive"))
    f = C -> Lmap(1, C, rt.AR[1, :], conj(rt.AR[1, :]))
    v_init = cellones(rt.AL)[1]
    nev = min(howmany, length(v_init))
    lambdas, _, info = eigsolve(f, v_init, nev, :LM; maxiter=100, ishermitian=false)
    info.converged == 0 && @warn "transfer_pepo_correlation_length did not converge"
    length(lambdas) < 2 && return Inf

    lambda2 = zero(eltype(lambdas))
    for i in 2:length(lambdas)
        if !(norm(lambdas[i]) ≈ norm(lambdas[1]))
            lambda2 = lambdas[i]
            break
        end
    end
    lambda2 == 0 && return Inf
    return real(-1 / log(abs(lambda2 / lambdas[1])))
end

transfer_pepo_history_label(model) = "density"

function _transfer_pepo_finalize!(x, f, g, iter, rt_norm, rt_norm_next,
                                  rt_transfer, rt_transfer_next, D, χ,
                                  params, t0, fδobjective)
    folder0 = joinpath(params.folder, "D$(D)")
    ispath(folder0) || mkpath(folder0)

    update!(rt_norm, rt_norm_next)
    update!(rt_transfer, rt_transfer_next)

    density = -real(f)
    label = transfer_pepo_history_label(params.model)
    message = @sprintf(
        "i = %5d\tt = %0.2f sec\tobjective_χ%d = %.15f\t%s_χ%d = %.15f\tgnorm = %.3e\n",
        iter, time() - t0, χ, real(f), label, χ, density, norm(g),
    )

    if _io_root(params.boundary_alg) && params.verbosity >= 3 && iter % params.show_every == 0
        printstyled(message; bold=true, color=:red)
        flush(stdout)
        open(joinpath(folder0, "history.log"), "a") do io
            write(io, message)
        end
    end

    if _io_root(params.boundary_alg) && params.save_every != 0 && iter % params.save_every == 0
        ipeps_dir = joinpath(folder0, "ipeps", "χ$χ")
        isdir(ipeps_dir) || mkpath(ipeps_dir)
        save(joinpath(ipeps_dir, "No.$(iter).jld2"), "bcipeps", Array(x); iotype=IOStream)
    end

    length(fδobjective) >= 3 && (fδobjective[3] = iter)
    δobjective = _transfer_pepo_update_stop!(g, f, fδobjective, params, χ)
    if params.last_stop_reason == :objective_stall && params.verbosity >= 1
        @warn "transfer-PEPO objective stalled; ending current χ and advancing" χ=χ delta=δobjective
    end

    gc(x)
    return x, f, g
end

function transfer_pepo_observable(A, χ::Integer, params::GradientOptimize;
                                  restriction_ipeps=_restriction_ipeps)
    return transfer_pepo_observable(params.model, A, χ, params; restriction_ipeps)
end

function transfer_pepo_observable(model, A, χ::Integer, params::GradientOptimize;
                                  restriction_ipeps=_restriction_ipeps)
    params = _transfer_pepo_params_for_chi(params, χ)
    A_built = _build_transfer_pepo_A(A, params; restriction_ipeps)
    rt_norm = init_env(A_built, Int(χ), params.boundary_alg)
    M_transfer = transfer_layer(model, A_built)
    rt_transfer = init_env(M_transfer, Int(χ), params.boundary_alg)
    return transfer_pepo_density(A, rt_norm, rt_transfer, params; restriction_ipeps)
end

function optimise_transfer_pepo(A, χlist::AbstractVector{<:Integer},
                                params::GradientOptimize;
                                restriction_ipeps=_restriction_ipeps)
    params = _transfer_pepo_params(params)
    χs = _normalize_χlist(χlist)
    D = _ipeps_bond_dimension(A)

    local rt_norm, rt_norm_next, rt_transfer, rt_transfer_next
    active_params = Ref(params)
    fδobjective = [NaN, Inf, 0.0]

    function fobjective(x)
        params = active_params[]
        A_built = _build_transfer_pepo_A(x, params; restriction_ipeps)
        rt_norm_new, rt_transfer_new, err_norm, err_transfer =
            _transfer_pepo_refresh_envs(rt_norm, rt_transfer, A_built, params)
        obs = _transfer_pepo_fixed_env_density(A_built, rt_norm_new, rt_transfer_new, params)

        ignore_derivatives() do
            update!(rt_norm_next, rt_norm_new)
            update!(rt_transfer_next, rt_transfer_new)
        end
        return obs.objective
    end

    function fg(x)
        params = active_params[]
        t1 = time()
        f, vjp = pullback(fobjective, x)
        params.verbosity >= 2 &&
            printstyled(" forward calculation took $(round(time() - t1, digits = 2)) s\n"; bold=true, color=:green)
        gc(x)
        t2 = time()
        g = vjp(1)[1]
        params.verbosity >= 2 &&
            printstyled("backward calculation took $(round(time() - t2, digits = 2)) s\n"; bold=true, color=:green)
        gc(x)
        return f, g
    end

    state_path = joinpath(params.folder, "D$(D)", "lbfgs_checkpoint")
    if params.ifsave_lbfgs && _io_root(params.boundary_alg)
        isdir(state_path) || mkpath(state_path)
    end

    alg = params.optimizer
    t0 = time()
    local objective, grad, fgnum, history

    for χ in χs
        params_χ = _transfer_pepo_params_for_chi(params, χ)
        active_params[] = params_χ
        fδobjective .= (NaN, Inf, 0.0)
        A_built = _build_transfer_pepo_A(A, params_χ; restriction_ipeps)
        rt_norm = init_env(A_built, χ, params_χ.boundary_alg)
        rt_norm_next = deepcopy(rt_norm)
        M_transfer = transfer_layer(params_χ.model, A_built)
        rt_transfer = init_env(M_transfer, χ, params_χ.boundary_alg)
        rt_transfer_next = deepcopy(rt_transfer)

        A, objective, grad, fgnum, history = optimize_reload(fg, A, alg;
            resume_from = params.ifload_lbfgs ? joinpath(state_path, "χ$χ.jld2") : nothing,
            save_state_to = (params.ifsave_lbfgs && _io_root(params.boundary_alg)) ? joinpath(state_path, "χ$χ.jld2") : nothing,
            save_every = params.save_every,
            precondition = (x, g) -> _transfer_pepo_precondition(
                x, g, rt_norm_next, params_χ, restriction_ipeps, fδobjective
            ),
            inner = _inner,
            finalize! = (x, f, g, iter) -> _transfer_pepo_finalize!(
                x, f, g, iter, rt_norm, rt_norm_next,
                rt_transfer, rt_transfer_next, D, χ, params_χ, t0, fδobjective,
            ),
        )
        transfer_pepo_observable(A, χ, params_χ; restriction_ipeps)
    end

    return A, -objective, grad, fgnum, history
end

optimize_transfer_pepo(args...; kwargs...) = optimise_transfer_pepo(args...; kwargs...)
