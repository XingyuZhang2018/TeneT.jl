# iPEPS fixed-point eigenvalue iteration with MCF gauge fix.
# PoC for the alternative to gradient-based optimization.
# See docs/plans/2026-04-27-ipeps-fixedpoint-mcf-design.md.

export iPEPSFixedPointConfig, optimize_ipeps_fixedpoint

"""
    iPEPSFixedPointConfig

Configuration for the fixed-point eigenvalue iteration optimizer.

Fields:
- `env_mode`           — `:A` full VUMPS, `:B` warm restart, `:C` 1-step
- `env_warm_steps`     — for `:B`, number of warm-restart VUMPS steps
- `H_eff_mode`         — `:a` h on central bond only, `:b` all 7 bonds (v2)
- `decompose_method`   — `:X` symmetrize-first, `:Y` avg, `:Z` lazy SVD
- `geneig_*`           — KrylovKit.geneigsolve knobs
- `mcf_ifignore_gauge` — pass to `local_min_norm`; if `true`, MCF disabled (control)
- `outer_maxiter` / `outer_tol_λ` / `outer_tol_A` — outer-loop convergence
- `log_every` / `save_every` — logging cadence
"""
@kwdef mutable struct iPEPSFixedPointConfig
    env_mode::Symbol           = :A
    env_warm_steps::Int        = 5
    H_eff_mode::Symbol         = :a
    decompose_method::Symbol   = :X
    geneig_krylovdim::Int      = 20
    geneig_tol::Float64        = 1e-10
    geneig_maxiter::Int        = 100
    mcf_ifignore_gauge::Bool   = false
    outer_maxiter::Int         = 200
    outer_tol_λ::Float64       = 1e-8
    outer_tol_A::Float64       = 1e-7
    log_every::Int             = 1
    save_every::Int            = 10
end

# build_phi, make_N_op, make_H_op, decompose_phi, sweep_bond, optimize_ipeps_fixedpoint
# are added in subsequent tasks.
