# Sanity: VUMPS{General} on 2D classical Ising vs Onsager exact f.
# Uses rank-4 M directly (no bra/ket doubling), so λ_AC/λ_C should give Z per site.
#
# Usage: julia --project=. timing/sanity_general_ising_freeenergy.jl

using TeneT, CUDA, LinearAlgebra, TensorOperations, KrylovKit, Random, Printf, Dates
using TeneT: StructArray, VUMPSRuntime, init_env, leading_boundary,
             leftenv, ACenv, Cenv, ALCtoAC

# GPUKrylov via type-piracy
import Pkg
gpukrylov_path = "D:/1 - research/1.19 - GPU/GPUKrylov.jl"
if !haskey(Pkg.project().dependencies, "GPUKrylov")
    Pkg.develop(path=gpukrylov_path)
end
using GPUKrylov

function KrylovKit.eigsolve(f::Function, x₀::CuArray{T,N}, howmany::Int, which::Symbol;
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

# ============================================================================
# Ising MPO + Onsager
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

# ============================================================================
# Compute f from converged General VUMPS state.
# Standard formula: log(Z_per_site) = log(λ_AC) - log(λ_C)
# where λ_AC = eigenvalue of M acting on AC, λ_C = eigenvalue of bond gauge.
# ============================================================================
function compute_free_energy_general(rt::VUMPSRuntime, M::StructArray, alg::VUMPS, beta::Float64)
    AL = rt.AL; AR = rt.AR; C = rt.C; FL = rt.FL; FR = rt.FR
    AC = ALCtoAC(AL, C)
    # Get λ_AC and λ_C from official eigsolves on the converged state
    λACs, _ = ACenv(AC, FL, M, FR; alg)
    λCs,  _ =  Cenv(C, FL, FR; alg)
    # AC is a StructArray (one per pattern site); for 1×1 take [1]
    λ_AC = real(λACs[1])
    λ_C  = real(λCs[1])
    Z_per_site = λ_AC / λ_C
    # Standard 2D Ising MPO: each M tensor sits at one spin site.
    # Z_per_site = exp(-β·f_per_spin), so f = -log(Z_per_site) / β.
    f = -log(Z_per_site) / beta
    return (λ_AC=λ_AC, λ_C=λ_C, Z_per_site=Z_per_site, f=f)
end

# ============================================================================
# Bench harness
# ============================================================================
function sanity_one(M, chi, beta, alg, label, f_exact)
    Random.seed!(42)
    rt = init_env(M, chi, alg)
    CUDA.synchronize()
    t0 = time()
    rt, err = leading_boundary(rt, M, alg)
    CUDA.synchronize()
    t = time() - t0

    fr = compute_free_energy_general(rt, M, alg, beta)
    @printf("%-30s  err=%.2e  λ_AC=%.4e  λ_C=%.4e  Z_per_site=%.6f  f=%.10f  Δf=%.2e  t=%.2fs\n",
        label, err, fr.λ_AC, fr.λ_C, fr.Z_per_site, fr.f, abs(fr.f - f_exact), t)
    return (label=label, err=err, fr=fr, t=t)
end

# ============================================================================
# Main
# ============================================================================
const CHI = parse(Int, get(ENV, "BENCH_CHI", "128"))
const BETA = parse(Float64, get(ENV, "BENCH_BETA", "0.43"))
const TOL = parse(Float64, get(ENV, "BENCH_TOL", "1e-8"))

println("=" ^ 100)
println("# Sanity: VUMPS{General} 2D Ising free energy vs Onsager")
println("Date: ", string(now()))
println("Hardware: ", CUDA.name(CUDA.device()))
println("Settings: χ=", CHI, " β=", BETA, " tol=", TOL)
println("=" ^ 100)

f_exact = ising_exact_free_energy(BETA)
@printf("Onsager f_exact at β=%.3f:  %.10f\n", BETA, f_exact)

M = ising_mpo(BETA; atype=CuArray)   # rank-4, no doubling

# Warmup
let alg_w = VUMPS{General}(verbosity=0, maxiter=2, miniter=1, miniter_ad=1, maxiter_ad=1,
                           tol=1e-3, ifsimple_eig=true, power_iter=5, ifupdown=false)
    Random.seed!(42); rt_w = init_env(M, 8, alg_w); leading_boundary(rt_w, M, alg_w)
    Random.seed!(42); rt_w = init_env(M, CHI, alg_w); leading_boundary(rt_w, M, alg_w)
    CUDA.synchronize()
end

println("\nMethod                          VUMPS_err  λ_AC          λ_C           Z_per_site  f          Δf       Time")
println("-" ^ 100)

for pi in [5, 10, 20, 50, 100]
    alg = VUMPS{General}(verbosity=0, maxiter=500, miniter=1, miniter_ad=1, maxiter_ad=1,
                         tol=TOL, ifsimple_eig=true, power_iter=pi, ifupdown=false)
    sanity_one(M, CHI, BETA, alg, "simple_eig pi=$pi", f_exact)
end

let alg = VUMPS{General}(verbosity=0, maxiter=500, miniter=1, miniter_ad=1, maxiter_ad=1,
                         tol=TOL, ifsimple_eig=false, power_iter=1, ifupdown=false)
    sanity_one(M, CHI, BETA, alg, "GPUKrylov", f_exact)
end

println()
println("=" ^ 100)
@printf("Onsager:  f = %.10f\n", f_exact)
println("Δf shows |f_VUMPS - f_Onsager|. Should be small for converged methods.")
