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
physical ξ at finite χ). Use `:none` to skip the correlation-length
calculation and write `NaN` in the observable log.
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
    # magnetization_value / cor_len_value are serial (not slice2d-ized): on the distributed (block)
    # obs env they'd run ALCtoAC on a χ-block → crash. For the slice2d-Plaquette case gather the
    # block env to full just for them. energy_value above stays block-distributed (the expensive,
    # accuracy-critical part); mag/ξ are a cheap replicated post-measurement on the gathered env.
    env_obs = env
    mag = magnetization_value(params.model, A, env_obs, params)
    ξ = cor_len_method === :none ? nothing : cor_len_value(env_obs, params, A; method=cor_len_method)

    # For Kagome merge: compute per-bond energies and use them for logging/plotting
    if params.model.lattice isa Kagome{:merge}
        e_perbond = energy_value_perbond(params.model, A, env, params)
        e = (e[1], e_perbond)  # replace aggregate e_dict with per-bond e_dict
    end

    # Only the grid-root rank writes the obs log / plots. Under slice2d the env gather + mag/ξ above
    # run on ALL ranks (collective + replicated result), but all ranks racing the same obs-log file
    # hits the JLD2/IO write race that killed 1287372's checkpoint save. This gate is AFTER the
    # collective gather so control flow stays rank-uniform across every MPI collective.
    _obsroot = params.boundary_alg.grid === nothing || params.boundary_alg.grid.rank == 0
    _obsroot && write_obs_log(e, mag, ξ, χ, joinpath(params.folder, "D$(D)"), params)

    # Visualization: read all logs and plot (includes history from previous runs)
    if _obsroot && params.ifplot
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
