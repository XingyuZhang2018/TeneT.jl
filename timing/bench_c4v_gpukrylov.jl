# Run the C4v Ising VUMPS bench with GPUKrylov substituted for KrylovKit.eigsolve.
# This is "type piracy" — we add a method KrylovKit.eigsolve(f, ::CuArray, ...) that
# routes to GPUKrylov.eigsolve_array!. Julia dispatches to the more specific method.
#
# Usage: julia --project=. timing/bench_c4v_gpukrylov.jl

using TeneT, CUDA, LinearAlgebra, TensorOperations, KrylovKit, Random, Printf, Dates
using TeneT: StructArray, C4vVUMPSEnv, init_env, leading_boundary, leg4

# Make GPUKrylov available — it's at the GPUKrylov.jl path
import Pkg
gpukrylov_path = "D:/1 - research/1.19 - GPU/GPUKrylov.jl"
if !haskey(Pkg.project().dependencies, "GPUKrylov")
    Pkg.develop(path=gpukrylov_path)
end
using GPUKrylov

# ============================================================================
# Type piracy: override KrylovKit.eigsolve for CuArray inputs.
# ============================================================================
function KrylovKit.eigsolve(f::Function, x₀::CuArray{T,N}, howmany::Int, which::Symbol;
                            tol::Real=1e-12, krylovdim::Int=30, maxiter::Int=100,
                            ishermitian::Bool=false, alg_rrule=nothing, verbosity::Int=0,
                            kwargs...) where {T, N}
    @assert which == :LM "GPUKrylov supports only :LM, got $which"
    @assert howmany == 1 "GPUKrylov v1 supports nev=1 only, got howmany=$howmany"
    opts = GPUKrylov.ArnoldiOpts(nev=howmany, krylov_dim=krylovdim, tol=tol, maxiter=maxiter, verbosity=0)
    f! = (out, vin) -> (copyto!(out, f(vin)); out)
    λs, vs, info = if N == 1
        GPUKrylov.eigsolve!(f!, x₀, opts)
    else
        GPUKrylov.eigsolve_array!(f!, x₀, opts)
    end
    # Build a struct compatible with KrylovKit.ConvergenceInfo (TeneT only reads .converged)
    info_compat = (
        converged = info.converged,
        numiter = info.numiter,
        numops = info.numops,
        residuals = info.residuals,
        normres = isempty(info.residuals) ? 0.0 : info.residuals[1],
    )
    return λs, vs, info_compat
end

# ============================================================================
# Ising MPO builders (copy from test/runtests.jl)
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

function build_c4v_M(beta; atype=CuArray)
    M_4 = ising_mpo(beta; atype).data[1]
    D = size(M_4, 1)
    M5_data = reshape(M_4, D, D, D, D, 1)
    return StructArray([M5_data], [1;;])
end

# ============================================================================
# Bench harness
# ============================================================================
function bench_one(; chi, beta, ifsimple_eig, power_iter, tol, maxiter, verbosity, label)
    println("\n[", label, "]")
    @printf "  chi=%d  beta=%.3f  ifsimple_eig=%s  power_iter=%d  tol=%.0e\n" chi beta ifsimple_eig power_iter tol

    M_c4v = build_c4v_M(beta; atype=CuArray)
    alg = VUMPS{C4v}(;
        verbosity = verbosity, maxiter = maxiter,
        miniter = 1, miniter_ad = 1, maxiter_ad = 1,
        tol = tol, ifsimple_eig = ifsimple_eig,
        power_iter = power_iter, ifupdown = false,
    )

    Random.seed!(42)
    rt = init_env(M_c4v, chi, alg)
    CUDA.synchronize()
    t0 = time()
    rt, err = leading_boundary(rt, M_c4v, alg)
    CUDA.synchronize()
    t = time() - t0
    converged = err < tol
    @printf "  Time: %.2f s   err=%.2e  %s\n" t err (converged ? "✓" : "✗")
    return (label=label, t=t, err=err, converged=converged)
end

# ============================================================================
# Sweep
# ============================================================================
const CHI = parse(Int, get(ENV, "BENCH_CHI", "128"))
const BETA = parse(Float64, get(ENV, "BENCH_BETA", "0.43"))
const TOL = parse(Float64, get(ENV, "BENCH_TOL", "1e-6"))
const MAXITER = parse(Int, get(ENV, "BENCH_MAXITER", "500"))

println("=" ^ 78)
println("# C4v VUMPS bench (GPUKrylov substitute): 2D classical Ising")
println("Date: ", string(now()))
println("Hardware: ", CUDA.name(CUDA.device()))
println("Settings: chi=", CHI, ", beta=", BETA, ", tol=", TOL, ", maxiter=", MAXITER)
println("=" ^ 78)

# Warmup at small chi
println("\n[warmup chi=8]")
let
    M_w = build_c4v_M(BETA; atype=CuArray)
    alg_w = VUMPS{C4v}(verbosity=0, maxiter=5, miniter=1, miniter_ad=1, maxiter_ad=1,
                       tol=1e-3, ifsimple_eig=true, power_iter=5, ifupdown=false)
    Random.seed!(42)
    rt_w = init_env(M_w, 8, alg_w)
    leading_boundary(rt_w, M_w, alg_w)
    CUDA.synchronize()
end

# Warmup at full chi (size-specific JIT for cuTENSOR)
println("[warmup chi=$CHI, both eig backends]")
let
    M_w = build_c4v_M(BETA; atype=CuArray)
    # simple_eig warmup
    alg_p = VUMPS{C4v}(verbosity=0, maxiter=2, miniter=1, miniter_ad=1, maxiter_ad=1,
                       tol=1e-3, ifsimple_eig=true, power_iter=5, ifupdown=false)
    Random.seed!(42)
    rt_p = init_env(M_w, CHI, alg_p); leading_boundary(rt_p, M_w, alg_p); CUDA.synchronize()
    # GPUKrylov-routed eigsolve warmup
    alg_k = VUMPS{C4v}(verbosity=0, maxiter=2, miniter=1, miniter_ad=1, maxiter_ad=1,
                       tol=1e-3, ifsimple_eig=false, power_iter=1, ifupdown=false)
    Random.seed!(42)
    rt_k = init_env(M_w, CHI, alg_k); leading_boundary(rt_k, M_w, alg_k); CUDA.synchronize()
end

# Sweep
results = []
for pi in [5, 10, 20, 50, 100]
    push!(results, bench_one(; chi=CHI, beta=BETA, ifsimple_eig=true,
                             power_iter=pi, tol=TOL, maxiter=MAXITER,
                             verbosity=2, label="simple_eig power_iter=$pi"))
end

push!(results, bench_one(; chi=CHI, beta=BETA, ifsimple_eig=false, power_iter=1,
                         tol=TOL, maxiter=MAXITER, verbosity=2,
                         label="GPUKrylov eigsolve"))

println("\n", "=" ^ 78)
println("Summary  (chi=$CHI, beta=$BETA, tol=$TOL)")
println("=" ^ 78)
@printf "%-30s  %10s  %12s  %s\n" "Method" "Time (s)" "VUMPS err" "Converged"
for r in results
    @printf "%-30s  %10.2f  %12.2e  %s\n" r.label r.t r.err (r.converged ? "yes" : "NO")
end
