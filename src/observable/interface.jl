# Observable computation for iPEPS
# Magnetization, correlation length, and full observable wrapper

"""
    energy(A, env, params::iPEPSOptimize)

Main entry point for energy computation during iPEPS optimization.
Calls `energy_value` dispatching on `params.model` type.
"""
function energy(A, env, params::iPEPSOptimize)
    return energy_value(params.model, A, env, params)[1]
end

# ============================================================================
# Full observable computation
# ============================================================================

"""
    observable(A, χ, params::iPEPSOptimize; restriction_ipeps=_restriction_ipeps, cor_len_method=:mps)

Compute all observables (energy, magnetization, correlation length) for a
given iPEPS tensor `A` at bond dimension `χ`. Initializes a VUMPS runtime,
converges the boundary, and evaluates expectation values.

`cor_len_method` chooses the correlation-length estimator (see
[`cor_len_value`](@ref)): `:mps` (default, pure boundary-MPS transfer
matrix; cheap) or `:channel` (channel TM with bulk M; closer to the
physical ξ at finite χ).
"""
function observable(A, χ, params::iPEPSOptimize; restriction_ipeps=_restriction_ipeps,
                    cor_len_method::Symbol=:mps)
    D = _ipeps_bond_dimension(A)
    rt = initialize_env(A, D, χ, params; restriction_ipeps)

    _G_cache[] = nothing
    A = restriction_ipeps(A)
    A = build_A(A, params)

    rt, _ = leading_boundary(rt, A, params.boundary_alg)
    params.ifsave_env && save_rt(joinpath(params.folder, "D$(D)", "environment"), rt; file="χ$(χ).jld2")
    env = ObsEnv(rt, A, params.boundary_alg, params.model)
    e = energy_value(params.model, A, env, params)
    mag = magnetization_value(params.model, A, env, params)
    ξ = cor_len_value(env, params, A; method=cor_len_method)

    # For Kagome merge: compute per-bond energies and use them for logging/plotting
    if params.model.lattice isa Kagome{:merge}
        e_perbond = energy_value_perbond(params.model, A, env, params)
        e = (e[1], e_perbond)  # replace aggregate e_dict with per-bond e_dict
    end

    write_obs_log(e, mag, ξ, χ, joinpath(params.folder, "D$(D)"), params)

    # Visualization: read all logs and plot (includes history from previous runs)
    if params.ifplot
        obs_path = joinpath(params.folder, "D$(D)", "observable")
        plot_observables(obs_path, params.model.lattice, params.pattern;
                         save_format=params.plot_format, S=params.model.S)
    end

    # if params.model.lattice == Honeycomb(:brickwall_h)
    #     Wp_value(params.model, A, env, params)
    #     fwave_order(params.model, A, env, params)
    # end
    return e, mag, ξ
end
