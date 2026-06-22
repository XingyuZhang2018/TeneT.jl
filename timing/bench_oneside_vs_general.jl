# Sanity benchmark: J1J2p :brickwall_v with VUMPS{General}(ifupdown=false)
# (one-sided env path) vs VUMPS{General}(ifupdown=true) (full two-sided path).
# After the Stage 2 simplification refactor, the one-sided path is selected
# by `ifupdown=false` + `params.model` reaching `ObsEnv` — there is no longer
# a separate VUMPS{Oneside{M}} algorithm.
#
# Verifies:
#   - VUMPS{General}(ifupdown=false) with the model passed → produces
#     OnesideVUMPSEnv from ObsEnv
#   - Energy agrees with VUMPS{General}(ifupdown=true) to within 1e-3
#   - One-sided path is faster (≥1.5× expected per plan)

using TeneT
using TeneT: ObsEnv, OnesideVUMPSEnv, build_A,
             leading_boundary, initialize_env, J1J2p, energy_value
using Random, LinearAlgebra, Printf, Zygote
using OptimKit

Random.seed!(42)

const D, χ = 2, 4
const J1, J2p = 1.0, 0.3
const pattern = [1 2; 2 1]
const d, N = 2, 2

const model = J1J2p(lattice=Honeycomb(:brickwall_v),
                    S=0.5, J1=J1, J2p=J2p,
                    ifrotate=false,
                    couplingtype=:uniform, bondratio=1.0)

# Restriction for :brickwall_v: parity-class mapping + U-D self-symmetry per
# tensor (required for the one-sided path; the two-sided path also accepts it).
function restriction_ipeps_v(A)
    B = Zygote.Buffer(A)
    B[:,:,:,:,:,1] = A[:,:,:,:,:,1]
    B[:,:,:,:,:,1] += permutedims(B[:,:,:,:,:,1], (1,4,3,2,5))
    B[:,:,:,:,:,2] = A[:,:,:,:,:,2]
    B[:,:,:,:,:,2] += permutedims(B[:,:,:,:,:,2], (1,4,3,2,5))
    Bc = copy(B)
    return Bc / norm(Bc)
end

A_raw = (rand(Float64, 1, D, D, D, d, N) .- 0.5)
A_raw /= norm(A_raw)

# Boundary algorithm: same params across both modes EXCEPT ifupdown.
make_alg(ifupdown_flag) = VUMPS{General}(; ifsimple_eig=true,
                                            ifupdown=ifupdown_flag,
                                            ifparallelupdown=false,
                                            ifparallel=false,
                                            forloop_iter=1,
                                            maxiter=30,
                                            miniter=0,
                                            maxiter_ad=0,
                                            miniter_ad=0,
                                            tol=1e-10,
                                            verbosity=0,
                                            show_every=10000)

function make_params(alg)
    folder = mktempdir()
    GradientOptimize(model=model, pattern=pattern,
                     boundary_alg=alg,
                     optimizer=LBFGS(10; maxiter=1, gradtol=1e-3, verbosity=0),
                     forloop_iter=1,
                     verbosity=0, folder=folder,
                     ifSU=false, SUτ=0.0, ifprecondition=false,
                     iter_precond=0,
                     reuse_env=true, ifsave_env=false, ifload_env=false,
                     ifsave_lbfgs=false, ifload_lbfgs=false)
end

function bench(alg, alg_label)
    params = make_params(alg)
    A_restricted = restriction_ipeps_v(A_raw)
    A = build_A(A_restricted, params)
    rt = initialize_env(A_raw, D, χ, params; restriction_ipeps=restriction_ipeps_v)

    # First, compile (warm-up) — one boundary + ObsEnv
    rt, _ = leading_boundary(rt, A, params.boundary_alg)
    env = ObsEnv(rt, A, params.boundary_alg, params.model)
    e_warmup, _ = energy_value(model, A, env, params)
    println("[$alg_label] warmup e = $(real(e_warmup))   env = $(typeof(env))")

    # Time the converged loop (5 leading_boundary + ObsEnv + energy_value)
    nrep = 5
    t = time()
    local e_inner
    for _ in 1:nrep
        rt, _ = leading_boundary(rt, A, params.boundary_alg)
        env = ObsEnv(rt, A, params.boundary_alg, params.model)
        e_inner, _ = energy_value(model, A, env, params)
    end
    elapsed = time() - t
    println("[$alg_label] $(nrep) reps elapsed = $(round(elapsed, digits=3)) s  ($(round(1000*elapsed/nrep, digits=1)) ms/rep)")
    println("[$alg_label] final e = $(real(e_inner))  env = $(typeof(env))")
    return real(e_inner), elapsed/nrep, typeof(env)
end

println("=" ^ 70)
println("J1J2p :brickwall_v   D=$D  χ=$χ  J1=$J1  J2p=$J2p  pattern=$pattern")
println("=" ^ 70)

# General two-sided baseline (ifupdown=true)
e_general, t_general, env_general_type = bench(make_alg(true), "General(ifupdown=true)")

println()

# One-sided (ifupdown=false) — should return OnesideVUMPSEnv when model passed
e_oneside, t_oneside, env_oneside_type = bench(make_alg(false), "General(ifupdown=false)")

println()
println("=" ^ 70)
println("Comparison")
println("=" ^ 70)
abs_diff = abs(e_oneside - e_general)
rel_diff = abs_diff / max(abs(e_general), 1e-12)
speedup = t_general / t_oneside
@printf("Energy (General, ifupdown=true ): % .10f\n", e_general)
@printf("Energy (General, ifupdown=false): % .10f\n", e_oneside)
@printf("|Δe|     = %.3e   (rel = %.3e)\n", abs_diff, rel_diff)
@printf("Per-rep  General  = %.1f ms\n", 1000*t_general)
@printf("Per-rep  Oneside  = %.1f ms\n", 1000*t_oneside)
@printf("Speedup           = %.2fx\n", speedup)

println()
@assert env_oneside_type === OnesideVUMPSEnv "Expected OnesideVUMPSEnv from ifupdown=false + model, got $(env_oneside_type)"
println("OK  env type for ifupdown=false+model is OnesideVUMPSEnv")

@assert rel_diff < 1e-3 "Energy mismatch: rel_diff=$(rel_diff) >= 1e-3"
println("OK  energies agree within rel_diff < 1e-3")

if speedup >= 1.5
    println("OK  speedup >= 1.5x")
else
    @warn "Speedup $(round(speedup, digits=2))x is below 1.5x target (still ran successfully)"
end
