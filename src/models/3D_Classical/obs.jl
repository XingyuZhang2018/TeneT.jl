export residual_entropy, cubic_dimer_observable, cubic_dimer_correlation_length
export optimise_cubic_dimer, optimize_cubic_dimer

_cubic_dimer_boundary_alg(alg::VUMPS{General}) = _transfer_pepo_boundary_alg(alg)
_cubic_dimer_boundary_alg(alg) = _transfer_pepo_boundary_alg(alg)

function _cubic_dimer_params(params::GradientOptimize)
    params.model isa CubicDimer ||
        throw(ArgumentError("expected params.model isa CubicDimer; got $(typeof(params.model))"))
    return _transfer_pepo_params(params)
end

function _cubic_dimer_params_for_chi(params::GradientOptimize, χ::Integer)
    params.model isa CubicDimer ||
        throw(ArgumentError("expected params.model isa CubicDimer; got $(typeof(params.model))"))
    return _transfer_pepo_params_for_chi(params, χ)
end

_vumps_chi(rt::VUMPSRuntime) = _transfer_pepo_vumps_chi(rt)
_vumps_chi(rt::C4vVUMPSEnv) = _transfer_pepo_vumps_chi(rt)

_cubic_dimer_update_stop!(g, f, fδ, params::GradientOptimize, χ::Integer; kwargs...) =
    _transfer_pepo_update_stop!(g, f, fδ, params, χ; kwargs...)

_partition_logdensity(rt::VUMPSRuntime, M::StructArray, alg::VUMPS{General}) =
    partition_logdensity(rt, M, alg)
_partition_logdensity(rt::C4vVUMPSEnv, M::StructArray, alg::VUMPS{C4v}) =
    partition_logdensity(rt, M, alg)

_build_cubic_dimer_A(A, params::GradientOptimize; restriction_ipeps) =
    _build_transfer_pepo_A(A, params; restriction_ipeps)

transfer_pepo_history_label(::CubicDimer) = "entropy"

function _as_cubic_dimer_entropy(obs)
    return merge((entropy = obs.log_density,), obs)
end

"""
    residual_entropy(A, rt_norm, rt_transfer, params; restriction_ipeps=_restriction_ipeps)

Compute the variational cubic-dimer residual entropy density
`log(<A|T|A>) - log(<A|A>)` from a norm boundary runtime and a mixed
triple-layer boundary runtime.
"""
function residual_entropy(A, rt_norm, rt_transfer,
                          params::GradientOptimize; restriction_ipeps=_restriction_ipeps)
    params.model isa CubicDimer ||
        throw(ArgumentError("expected params.model isa CubicDimer; got $(typeof(params.model))"))
    return _as_cubic_dimer_entropy(
        transfer_pepo_density(A, rt_norm, rt_transfer, params; restriction_ipeps)
    )
end

"""
    cubic_dimer_correlation_length(rt; howmany=5)

Estimate the boundary-MPS correlation length from the double-layer norm
runtime, matching the Fig. 1 extraction from the double-layer boundary MPS.
"""
cubic_dimer_correlation_length(rt::VUMPSRuntime; howmany::Integer=5) =
    transfer_pepo_correlation_length(rt; howmany)

function cubic_dimer_correlation_length(rt::VUMPSRuntime, params::GradientOptimize,
                                        A_built::StructArray; howmany::Integer=5,
                                        method::Symbol=:mps)
    return cubic_dimer_correlation_length(rt; howmany)
end

function cubic_dimer_correlation_length(rt::C4vVUMPSEnv, params::GradientOptimize,
                                        A_built::StructArray; howmany::Integer=5,
                                        method::Symbol=:mps)
    return real(cor_len_value(rt, params, A_built; method))
end

function _write_cubic_dimer_observable(obs, χ::Int, folder::String)
    path = joinpath(folder, "observable")
    isdir(path) || mkpath(path)
    obs_log = joinpath(path, "χ$χ.log")
    open(obs_log, "w") do io
        @printf(io, "residual_entropy_per_site:\n%.15f\n", real(obs.entropy))
        @printf(io, "objective_minus_entropy:\n%.15f\n", real(obs.objective))
        @printf(io, "log_transfer:\n%.15f\n", real(obs.log_transfer))
        @printf(io, "log_norm:\n%.15f\n", real(obs.log_norm))
        @printf(io, "correlation_length:\n%.15f\n", real(obs.xi))
        @printf(io, "err_norm:\n%.15e\n", real(obs.err_norm))
        @printf(io, "err_transfer:\n%.15e\n", real(obs.err_transfer))
    end
    return obs_log
end

function transfer_pepo_observable(model::CubicDimer, A, χ::Integer,
                                  params::GradientOptimize;
                                  restriction_ipeps=_restriction_ipeps)
    params = _cubic_dimer_params_for_chi(params, χ)
    D = _ipeps_bond_dimension(A)
    A_built = _build_cubic_dimer_A(A, params; restriction_ipeps)
    rt_norm = init_env(A_built, Int(χ), params.boundary_alg)
    M_transfer = transfer_layer(model, A_built)
    rt_transfer = init_env(M_transfer, Int(χ), params.boundary_alg)

    obs = residual_entropy(A, rt_norm, rt_transfer, params; restriction_ipeps)
    xi = cubic_dimer_correlation_length(obs.rt_norm, params, A_built)
    full_obs = merge(obs, (xi = xi,))

    _io_root(params.boundary_alg) &&
        _write_cubic_dimer_observable(full_obs, Int(χ), joinpath(params.folder, "D$(D)"))
    return full_obs
end

"""
    cubic_dimer_observable(A, χ, params; restriction_ipeps=_restriction_ipeps)

Converge norm and mixed transfer boundaries, compute residual entropy and the
double-layer boundary-MPS correlation length, and write
`D*/observable/χ*.log`.
"""
cubic_dimer_observable(A, χ::Integer, params::GradientOptimize;
                       restriction_ipeps=_restriction_ipeps) =
    transfer_pepo_observable(params.model, A, χ, params; restriction_ipeps)

"""
    optimise_cubic_dimer(A, χlist, params; restriction_ipeps=_restriction_ipeps)

Optimize the cubic-dimer PEPS fixed point by minimizing the negative residual
entropy through the generic transfer-PEPO optimizer.
"""
function optimise_cubic_dimer(A, χlist::AbstractVector{<:Integer},
                              params::GradientOptimize;
                              restriction_ipeps=_restriction_ipeps)
    params.model isa CubicDimer ||
        throw(ArgumentError("expected params.model isa CubicDimer; got $(typeof(params.model))"))
    return optimise_transfer_pepo(A, χlist, params; restriction_ipeps)
end

optimize_cubic_dimer(args...; kwargs...) = optimise_cubic_dimer(args...; kwargs...)
