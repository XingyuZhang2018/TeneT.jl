# Heisenberg PoC: fixed-point eigenvalue iteration with MCF.
#
# Usage: julia --project=. examples/Heisenberg/Heisenberg_Square_FixedPoint.jl <experiment>
# experiment ∈ {E1, E2, E3, E4, E5} (default: E1)
#
# Initial state: tries to load an existing LBFGS-converged jld2 first;
# falls back to small-perturbation random init.
#
# Reference baseline: docs/plans/baseline-energies.md
# (E_LBFGS = -0.660231 at D=2, χ=16 for rotated AFM Heisenberg)

using TeneT
using JLD2
using LinearAlgebra
using Random
using Printf

Random.seed!(42)

const D = 2
const d = 2
const χ = 8

# Rotated AFM Heisenberg defaults (matches stupefied-cartwright LBFGS data)
const model = Heisenberg(lattice = Square(),
                         S = 0.5,
                         Jx = -1.0, Jy = -1.0, Jz = 1.0,
                         ifrotate = true,
                         couplingtype = :uniform,
                         bondratio = 1.0)

const params = TeneT.make_default_params(; D=D, χ=χ)

# ---- Initial state ----------------------------------------------------------

const LBFGS_JLD2 = "D:/1 - research/1.26 - iPEPS_opt/TeneT.jl/.claude/worktrees/stupefied-cartwright/data/Heisenberg_Square(S=0.5,Jx=-1.0,Jy=-1.0,Jz=1.0,ifrotate=true,couplingtype=uniform)/[1;;]/VUMPS_General/Float64/seed42/D2/ipeps/χ16/No.20.jld2"

A_init = if isfile(LBFGS_JLD2)
    @info "Loading LBFGS-converged iPEPS as initial state from:\n  $LBFGS_JLD2"
    A_loaded = load(LBFGS_JLD2, "bcipeps"; iotype=IOStream)
    @info "Loaded shape: $(size(A_loaded))"
    A_loaded
else
    @warn "No LBFGS jld2 found at $LBFGS_JLD2 — using small-perturbation random init"
    A_rand = randn(D, D, D, D, d, 1) / sqrt(D^4 * d)
    A_rand
end

# ---- Experiment definitions -------------------------------------------------

const EXPERIMENTS = Dict(
    "E1" => iPEPSFixedPointConfig(env_mode=:A, H_eff_mode=:a, decompose_method=:X,
                                  mcf_ifignore_gauge=false,
                                  outer_maxiter=50, log_every=1),
    "E2" => iPEPSFixedPointConfig(env_mode=:A, H_eff_mode=:a, decompose_method=:X,
                                  mcf_ifignore_gauge=true,            # MCF off — control
                                  outer_maxiter=50, log_every=1),
    "E3" => iPEPSFixedPointConfig(env_mode=:C, H_eff_mode=:a, decompose_method=:X,
                                  mcf_ifignore_gauge=false,            # cheap env
                                  outer_maxiter=50, log_every=1),
    "E4" => iPEPSFixedPointConfig(env_mode=:A, H_eff_mode=:a, decompose_method=:Z,
                                  mcf_ifignore_gauge=false,            # lazy decomp
                                  outer_maxiter=50, log_every=1),
    "E5" => iPEPSFixedPointConfig(env_mode=:A, H_eff_mode=:a, decompose_method=:Y,
                                  mcf_ifignore_gauge=false,            # avg decomp
                                  outer_maxiter=50, log_every=1),
)

const EXP_NAME = length(ARGS) >= 1 ? ARGS[1] : "E1"
haskey(EXPERIMENTS, EXP_NAME) || error("Unknown experiment: $EXP_NAME (must be one of $(collect(keys(EXPERIMENTS))))")
const cfg = EXPERIMENTS[EXP_NAME]

println("="^60)
println("Heisenberg Square — fixed-point PoC, experiment $EXP_NAME")
println("="^60)
println("D = $D, χ = $χ, d = $d")
println("model = $model")
println("cfg   = $cfg")
println("="^60)

# ---- Run --------------------------------------------------------------------

const t_start = time()
history = optimize_ipeps_fixedpoint(A_init, χ, model, params, cfg)
const t_total = time() - t_start

println("="^60)
@printf("Run wall time: %.1f sec\n", t_total)
@printf("Iterations:    %d / %d\n", length(history), cfg.outer_maxiter)
if length(history) > 0
    last = history[end]
    @printf("Final λ:       %.10g\n", last.λ)
    @printf("Final E:       %.10g\n", last.E)
    @printf("Final dλ:      %.3e\n", last.dλ)
    @printf("Final dA:      %.3e\n", last.dA)
end
println("="^60)

# ---- Save log ---------------------------------------------------------------

const OUT_DIR = joinpath(@__DIR__, "..", "..", "data", "fixedpoint_logs")
mkpath(OUT_DIR)
const OUT_FILE = joinpath(OUT_DIR, "$(EXP_NAME)_D$(D)_chi$(χ).jld2")
save(OUT_FILE, "history", history, "cfg", cfg, "model", model, "exp_name", EXP_NAME)
println("Saved log to: $OUT_FILE")
