# Benchmark C4v VUMPS at chi=128 for 2D classical Ising
# Compares simple_eig (power iteration, various power_iter) vs KrylovKit eigsolve
# Target: VUMPS tol=1e-6
#
# Usage: julia --project=. timing/bench_c4v_ising_chi128.jl

using TeneT
using CUDA
using LinearAlgebra
using TensorOperations
using KrylovKit
using Random
using Printf
using Dates

using TeneT: StructArray, C4vVUMPSEnv, init_env, leading_boundary, leg4

# ============================================================================
# Ising MPO builder (copy from test/runtests.jl)
# ============================================================================
function ising_mpo(beta; atype=Array)
    B = [exp(beta) exp(-beta); exp(-beta) exp(beta)]
    W = sqrt(B)
    d = size(W, 2)
    δ = zeros(d, d, d, d)
    for s in 1:d
        δ[s, s, s, s] = 1.0
    end
    @tensor M[a, b, c, d_] := δ[s1, s2, s3, s4] * W[s1, a] * W[s2, b] * W[s3, c] * W[s4, d_]
    M = atype(M)
    return StructArray([M], [1;;])
end

# Build C4v rank-5 MPO
function build_c4v_M(beta; atype=CuArray)
    M_4 = ising_mpo(beta; atype).data[1]
    D = size(M_4, 1)
    M5_data = reshape(M_4, D, D, D, D, 1)
    return StructArray([M5_data], [1;;])
end

# ============================================================================
# Benchmark one configuration
# ============================================================================
function bench_one(; chi, beta, ifsimple_eig, power_iter, tol, maxiter, verbosity, label)
    println("\n[", label, "]")
    @printf "  chi=%d  beta=%.3f  ifsimple_eig=%s  power_iter=%d  tol=%.0e  maxiter=%d\n" chi beta ifsimple_eig power_iter tol maxiter

    M_c4v = build_c4v_M(beta; atype=CuArray)

    alg = VUMPS{C4v}(;
        verbosity = verbosity,
        maxiter   = maxiter,
        miniter   = 1,
        miniter_ad = 1,
        maxiter_ad = 1,
        tol = tol,
        ifsimple_eig = ifsimple_eig,
        power_iter = power_iter,
        ifupdown = false,   # C4v doesn't need updown
    )

    Random.seed!(42)
    rt = init_env(M_c4v, chi, alg)
    CUDA.synchronize()

    # Warm up the kernels: do a single VUMPS step (will recompile cuTENSOR plans, etc.)
    # Actually leading_boundary itself runs maxiter iterations; let's just time it once
    # and call it the cold-start time. For the sweep, we can do 2 runs and report the 2nd.
    # For simplicity, do 1 timed run with maxiter=200 → tol=1e-6 stop early on convergence.

    t0 = time()
    rt, err = leading_boundary(rt, M_c4v, alg)
    CUDA.synchronize()
    t = time() - t0

    converged = err < tol
    @printf "  Time: %.2f s   err=%.2e  %s\n" t err (converged ? "✓ converged" : "✗ NOT converged")
    return (label=label, t=t, err=err, converged=converged, power_iter=power_iter, ifsimple_eig=ifsimple_eig)
end

# ============================================================================
# Sweep
# ============================================================================
const CHI = parse(Int, get(ENV, "BENCH_CHI", "128"))
const BETA = parse(Float64, get(ENV, "BENCH_BETA", "0.4"))   # T_c ~ 2.27 → β_c ~ 0.4407; β=0.4 is paramagnetic
const TOL = parse(Float64, get(ENV, "BENCH_TOL", "1e-6"))
const MAXITER = parse(Int, get(ENV, "BENCH_MAXITER", "500"))

println("=" ^ 78)
println("# C4v VUMPS bench: 2D classical Ising")
println("Date: ", string(now()))
println("Hardware: ", CUDA.name(CUDA.device()))
println("Settings: chi=", CHI, ", beta=", BETA, ", tol=", TOL, ", maxiter=", MAXITER)
println("=" ^ 78)

results = []

# Warmup: small chi run to compile cuTENSOR/cuBLAS plans before timing
println("\n[warmup] chi=8, simple_eig power_iter=5 (compile JIT)")
let
    M_w = build_c4v_M(BETA; atype=CuArray)
    alg_w = VUMPS{C4v}(verbosity=0, maxiter=5, miniter=1, miniter_ad=1, maxiter_ad=1,
                       tol=1e-3, ifsimple_eig=true, power_iter=5, ifupdown=false)
    Random.seed!(42)
    rt_w = init_env(M_w, 8, alg_w)
    leading_boundary(rt_w, M_w, alg_w)
    CUDA.synchronize()
    println("  warmup done")
end

# Second warmup at the actual chi=128 to compile size-specific kernels
println("[warmup2] chi=$CHI, simple_eig power_iter=5 (size-specific JIT)")
let
    M_w = build_c4v_M(BETA; atype=CuArray)
    alg_w = VUMPS{C4v}(verbosity=0, maxiter=5, miniter=1, miniter_ad=1, maxiter_ad=1,
                       tol=1e-3, ifsimple_eig=true, power_iter=5, ifupdown=false)
    Random.seed!(42)
    rt_w = init_env(M_w, CHI, alg_w)
    leading_boundary(rt_w, M_w, alg_w)
    CUDA.synchronize()
    println("  warmup2 done")
end

# Power iteration sweep
for pi in [5, 10, 20, 50, 100]
    label = "simple_eig power_iter=$pi"
    push!(results, bench_one(;
        chi=CHI, beta=BETA, ifsimple_eig=true,
        power_iter=pi, tol=TOL, maxiter=MAXITER,
        verbosity=2, label=label
    ))
end

# KrylovKit eigsolve baseline
push!(results, bench_one(;
    chi=CHI, beta=BETA, ifsimple_eig=false,
    power_iter=1, tol=TOL, maxiter=MAXITER,
    verbosity=2, label="KrylovKit eigsolve"
))

# Summary
println("\n", "=" ^ 78)
println("Summary")
println("=" ^ 78)
@printf "%-30s  %10s  %12s  %s\n" "Method" "Time (s)" "VUMPS err" "Converged"
for r in results
    @printf "%-30s  %10.2f  %12.2e  %s\n" r.label r.t r.err (r.converged ? "yes" : "NO")
end
