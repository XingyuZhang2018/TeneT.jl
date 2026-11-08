# Final benchmark: VUMPS{General} on 2D classical Ising at χ=128, tol=1e-6.
# Runs in TWO PHASES to avoid Base.delete_method pitfalls:
#   Phase 1: power_iter sweep + KrylovKit native (no type-piracy yet)
#   Phase 2: register GPUKrylov type-piracy, run GPUKrylov
#
# Usage: julia --project=. timing/bench_general_ising_chi128.jl

using TeneT, CUDA, LinearAlgebra, TensorOperations, KrylovKit, Random, Printf, Dates
using TeneT: StructArray, VUMPSRuntime, init_env, leading_boundary,
             ACenv, Cenv, ALCtoAC

# ============================================================================
# Corrected 2D Ising MPO (4 legs all through one spin)
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
    return StructArray([atype(M)], [1;;])
end

function ising_exact_free_energy(beta)
    npts = 5000
    s = 0.0
    dt = pi / npts
    for i in 1:npts
        t1 = (i - 0.5) * dt
        for j in 1:npts
            t2 = (j - 0.5) * dt
            s += log(cosh(2 * beta)^2 - sinh(2 * beta) * (cos(t1) + cos(t2)))
        end
    end
    s *= dt^2 / (2 * pi^2)
    return -(log(2) + s) / beta
end

function compute_f(rt::VUMPSRuntime, M::StructArray, alg::VUMPS, beta::Float64)
    AC = ALCtoAC(rt.AL, rt.C)
    λACs, _ = ACenv(AC, rt.FL, M, rt.FR; alg)
    λCs,  _ =  Cenv(rt.C, rt.FL, rt.FR; alg)
    Z = real(λACs[1] / λCs[1])
    return -log(Z) / beta
end

function bench_one(M, chi, beta, alg, label, f_exact)
    Random.seed!(42)
    rt = init_env(M, chi, alg)
    CUDA.synchronize()
    t0 = time()
    rt, err = leading_boundary(rt, M, alg)
    CUDA.synchronize()
    t = time() - t0
    f = compute_f(rt, M, alg, beta)
    Δf = abs(f - f_exact)
    status = err < alg.tol ? "✓conv" : "⚠no-conv"
    @printf("%-30s  t=%.2fs  err=%.2e  f=%.10f  Δf=%.2e  %s\n",
        label, t, err, f, Δf, status)
    return (label=label, t=t, err=err, f=f, Δf=Δf)
end

const CHI = parse(Int, get(ENV, "BENCH_CHI", "128"))
const BETA = parse(Float64, get(ENV, "BENCH_BETA", "0.43"))
const TOL = parse(Float64, get(ENV, "BENCH_TOL", "1e-6"))

println("=" ^ 100)
println("# VUMPS{General} 2D Ising bench at χ=", CHI, " β=", BETA, " tol=", TOL)
println("Hardware: ", CUDA.name(CUDA.device()))
println("Date: ", string(now()))
println("=" ^ 100)
f_exact = ising_exact_free_energy(BETA)
@printf("Onsager exact f = %.10f\n\n", f_exact)

M = ising_mpo(BETA; atype=CuArray)

# ============================================================================
# Phase 1: power_iter sweep + KrylovKit native
# ============================================================================
println("─" ^ 100)
println("Phase 1: power_iter sweep + KrylovKit native (no GPUKrylov override yet)")
println("─" ^ 100)

# Warmup with both backends (simple_eig + KrylovKit native)
println("[warmup chi=8]")
let alg_w = VUMPS{General}(verbosity=0, maxiter=2, miniter=1, miniter_ad=1, maxiter_ad=1,
                           tol=1e-3, ifsimple_eig=true, power_iter=5, ifupdown=false)
    Random.seed!(42); rt_w = init_env(M, 8, alg_w); leading_boundary(rt_w, M, alg_w)
    CUDA.synchronize()
end
println("[warmup chi=$CHI both backends]")
let
    alg_p = VUMPS{General}(verbosity=0, maxiter=2, miniter=1, miniter_ad=1, maxiter_ad=1,
                           tol=1e-3, ifsimple_eig=true, power_iter=5, ifupdown=false)
    Random.seed!(42); rt_p = init_env(M, CHI, alg_p); leading_boundary(rt_p, M, alg_p); CUDA.synchronize()
    alg_k = VUMPS{General}(verbosity=0, maxiter=2, miniter=1, miniter_ad=1, maxiter_ad=1,
                           tol=1e-3, ifsimple_eig=false, power_iter=1, ifupdown=false)
    Random.seed!(42); rt_k = init_env(M, CHI, alg_k); leading_boundary(rt_k, M, alg_k); CUDA.synchronize()
end

println("\nMethod                          Time     err          f               Δf       Status")
println("-" ^ 100)
results = []

for pi in [5, 10, 20, 50, 100]
    alg = VUMPS{General}(verbosity=0, maxiter=500, miniter=1, miniter_ad=1, maxiter_ad=1,
                         tol=TOL, ifsimple_eig=true, power_iter=pi, ifupdown=false)
    push!(results, bench_one(M, CHI, BETA, alg, "simple_eig pi=$pi", f_exact))
end

# KrylovKit native — type-piracy is NOT yet defined, so this goes through KK
let alg = VUMPS{General}(verbosity=0, maxiter=500, miniter=1, miniter_ad=1, maxiter_ad=1,
                         tol=TOL, ifsimple_eig=false, power_iter=1, ifupdown=false)
    push!(results, bench_one(M, CHI, BETA, alg, "KrylovKit (native)", f_exact))
end

# ============================================================================
# Phase 2: register GPUKrylov type-piracy, run GPUKrylov
# ============================================================================
println()
println("─" ^ 100)
println("Phase 2: register GPUKrylov type-piracy and bench it")
println("─" ^ 100)

import Pkg
gpukrylov_path = "D:/1 - research/1.19 - GPU/GPUKrylov.jl"
if !haskey(Pkg.project().dependencies, "GPUKrylov")
    Pkg.develop(path=gpukrylov_path)
end
@eval using GPUKrylov

# Define type-piracy AFTER Phase 1 finished — Phase 1 used native KK
@eval function KrylovKit.eigsolve(f::Function, x₀::CuArray{T,N}, howmany::Int, which::Symbol;
                                   tol::Real=1e-12, krylovdim::Int=30, maxiter::Int=100,
                                   ishermitian::Bool=false, alg_rrule=nothing, verbosity::Int=0,
                                   kwargs...) where {T, N}
    @assert which == :LM
    @assert howmany == 1
    opts = GPUKrylov.ArnoldiOpts(nev=1, krylov_dim=krylovdim, tol=tol, maxiter=maxiter, verbosity=0)
    f! = (out, vin) -> (copyto!(out, f(vin)); out)
    λs, vs, info = N == 1 ? GPUKrylov.eigsolve!(f!, x₀, opts) :
                            GPUKrylov.eigsolve_array!(f!, x₀, opts)
    info_compat = (
        converged=info.converged, numiter=info.numiter, numops=info.numops,
        residuals=info.residuals,
        normres=isempty(info.residuals) ? 0.0 : info.residuals[1],
    )
    return λs, vs, info_compat
end

# Re-warmup after type-piracy is in (different code path)
println("[warmup chi=$CHI with GPUKrylov type-piracy]")
let alg_g = VUMPS{General}(verbosity=0, maxiter=2, miniter=1, miniter_ad=1, maxiter_ad=1,
                           tol=1e-3, ifsimple_eig=false, power_iter=1, ifupdown=false)
    Random.seed!(42); rt_g = init_env(M, CHI, alg_g); leading_boundary(rt_g, M, alg_g); CUDA.synchronize()
end

println()
let alg = VUMPS{General}(verbosity=0, maxiter=500, miniter=1, miniter_ad=1, maxiter_ad=1,
                         tol=TOL, ifsimple_eig=false, power_iter=1, ifupdown=false)
    push!(results, bench_one(M, CHI, BETA, alg, "GPUKrylov", f_exact))
end

println("\n", "=" ^ 100)
println("Summary  (χ=$CHI, β=$BETA, tol=$TOL)")
println("=" ^ 100)
@printf("Onsager  f = %.10f\n", f_exact)
@printf("%-30s  %10s  %15s  %10s\n", "Method", "Time (s)", "f", "|Δf|")
for r in results
    @printf("%-30s  %10.2f  %15.10f  %10.2e\n", r.label, r.t, r.f, r.Δf)
end
