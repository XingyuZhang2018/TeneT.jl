# iPEPS optimization parameter structs
# Each struct configures a different optimization strategy for iPEPS tensors.

"""
    GradientOptimize <: iPEPSOptimize

Parameters for gradient-based (AD) optimization of iPEPS.
Uses automatic differentiation through the boundary contraction to compute
energy gradients, then applies a quasi-Newton optimizer (e.g. LBFGS).
"""
@kwdef mutable struct GradientOptimize <: iPEPSOptimize
    model::HamiltonianModel
    pattern::Matrix{Int}
    boundary_alg::Algorithm

    # Environment reuse / fixed-point
    reuse_env::Bool = true
    iffixedpoint::Bool = false

    # Verbosity and iteration control
    verbosity::Int = Defaults.VERBOSE_ITER
    maxiter_restart::Int = 1

    # Simple-update warm start
    SUτ::Real = 0.0
    ifSU::Bool = false

    # Optimizer (e.g. LBFGS from OptimKit)
    optimizer = LBFGS(200; maxiter=100, verbosity=4, gradtol=1e-7, 
                      linesearch=HagerZhangLineSearch(maxfg=5)) #GradientDescent()

    # I/O paths and intervals
    folder::String = joinpath(pkgdir(TeneT), "data")
    show_every::Int = 1
    save_every::Int = 1

    # Environment save/load
    ifsave_env::Bool = false
    save_env_tol::Real = 1e-4
    ifload_env::Bool = true

    # LBFGS state save/load
    ifsave_lbfgs::Bool = true
    ifload_lbfgs::Bool = false

    # Tensor layout
    forloop_iter::Int = 1

    # Checkpointing (Zygote checkpoint for memory saving)
    ifcheckpoint::Bool = false

    # ── AD checkpointing on the energy/gradient side (coarse → fine) ────
    # See src/utils/checkpoint.jl for Plain/Recompute/Offload semantics.
    # Note: a map-level option is intentionally absent — rrule(parallel) in
    # src/autodiff/rules.jl already does per-chunk Recompute, so wrapping
    # each *map_parallel in checkpoint() would be redundant.
    #
    #   obs_checkpoint  — wraps `energy_value(model, A, env, params)` as a
    #                      whole. Recompute alone does NOT reduce peak
    #                      (only defers tape construction to backward).
    #                      Offload is the useful mode: moves the ObsEnv
    #                      tape to host RAM so later phases have more VRAM.
    #   bond_checkpoint — wraps each bond term inside `_contract_barebones`
    #                      and each norm contraction (via `_contract_one`).
    #                      THE main lever for multi-bond models: peak drops
    #                      from (N_terms × bond_tape) to (1 × bond_tape).
    #                      Use Recompute() — Offload is overkill at this
    #                      granularity.
    obs_checkpoint::CheckpointMethod  = Plain()
    bond_checkpoint::CheckpointMethod = Plain()

    # Preconditioning
    ifprecondition::Bool = false
    iter_precond::Int = 20

    # Visualization (requires CairoMakie)
    ifplot::Bool = true
    plot_format::String = "png"
end

"""
    SUOptimize <: iPEPSOptimize

Parameters for Simple Update (SU) optimization of iPEPS.
Applies imaginary-time evolution via local SVD truncation on each bond,
without requiring a full environment contraction.
"""
@kwdef mutable struct SUOptimize <: iPEPSOptimize
    model::HamiltonianModel
    pattern::Matrix{Int}
    boundary_alg::Algorithm

    reuse_env::Bool = true
    verbosity::Int = Defaults.VERBOSE_ITER
    maxiter::Int = 1000
    tol::Real = 1e-10
    SUτ::Real = 0.01

    folder::String = joinpath(pwd(), "data", "ipeps")
    show_every::Int = 10
    save_every::Int = 100

    # Present so that `energy_value` / `_contract_barebones(..., params::iPEPSOptimize)`
    # can read them uniformly; Plain() is a no-op (checkpoint becomes identity).
    obs_checkpoint::CheckpointMethod  = Plain()
    bond_checkpoint::CheckpointMethod = Plain()
end

"""
    FUOptimize <: iPEPSOptimize

Parameters for Full Update (FU) optimization of iPEPS.
Uses the full boundary environment to perform a more accurate truncation
of the bond dimension after applying the imaginary-time gate.
"""
@kwdef mutable struct FUOptimize <: iPEPSOptimize
    model::HamiltonianModel
    pattern::Matrix{Int}
    boundary_alg::Algorithm

    reuse_env::Bool = true
    verbosity::Int = Defaults.VERBOSE_ITER
    maxiter::Int = 1000
    tol::Real = 1e-10
    SUτ::Real = 0.01

    folder::String = joinpath(pwd(), "data", "ipeps")
    show_every::Int = 10
    save_every::Int = 100
end
