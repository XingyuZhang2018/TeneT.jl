# Sanity check: compute free energy from VUMPS{C4v} converged state and
# compare with Onsager's exact value. Tests whether the "1 VUMPS step to
# machine precision" claim from bench_c4v_gpukrylov is real or whether VUMPS
# err metric just happens to look small without the eigenvector being correct.
#
# Usage: julia --project=. timing/sanity_c4v_ising_freeenergy.jl

using TeneT, CUDA, LinearAlgebra, TensorOperations, KrylovKit, Random, Printf, Dates
using TeneT: StructArray, C4vVUMPSEnv, init_env, leading_boundary, leg4,
             ALCtoAC, ACmap_parallel

# Make GPUKrylov available
import Pkg
gpukrylov_path = "D:/1 - research/1.19 - GPU/GPUKrylov.jl"
if !haskey(Pkg.project().dependencies, "GPUKrylov")
    Pkg.develop(path=gpukrylov_path)
end
using GPUKrylov

# Type-piracy: route KrylovKit.eigsolve(::CuArray) → GPUKrylov
function KrylovKit.eigsolve(f::Function, x₀::CuArray{T,N}, howmany::Int, which::Symbol;
                            tol::Real=1e-12, krylovdim::Int=30, maxiter::Int=100,
                            ishermitian::Bool=false, alg_rrule=nothing, verbosity::Int=0,
                            kwargs...) where {T, N}
    @assert which == :LM "GPUKrylov supports only :LM"
    @assert howmany == 1 "GPUKrylov v1 supports nev=1 only"
    opts = GPUKrylov.ArnoldiOpts(nev=1, krylov_dim=krylovdim, tol=tol, maxiter=maxiter, verbosity=0)
    f! = (out, vin) -> (copyto!(out, f(vin)); out)
    λs, vs, info = if N == 1
        GPUKrylov.eigsolve!(f!, x₀, opts)
    else
        GPUKrylov.eigsolve_array!(f!, x₀, opts)
    end
    info_compat = (
        converged=info.converged, numiter=info.numiter, numops=info.numops,
        residuals=info.residuals,
        normres=isempty(info.residuals) ? 0.0 : info.residuals[1],
    )
    return λs, vs, info_compat
end

# Ising MPO + Onsager
function ising_mpo(beta; atype=Array)
    B = [exp(beta) exp(-beta); exp(-beta) exp(beta)]
    W = sqrt(B)
    @tensor T12[s1, s2, a, b] := W[s1, a] * W[s2, b]
    @tensor T34[s3, s4, a, b] := W[s3, a] * W[s4, b]
    d = size(W, 2)
    T12r = reshape(T12, size(T12,1), size(T12,2), d*d)
    T34r = reshape(T34, size(T34,1), size(T34,2), d*d)
    @tensor M[s1, s2, s3, s4] := T12r[s1, s2, σ] * T34r[s3, s4, σ]
    M = atype(M)
    return StructArray([M], [1;;])
end

function build_c4v_M(beta; atype=CuArray)
    M_4 = ising_mpo(beta; atype).data[1]
    D = size(M_4, 1)
    M5_data = reshape(M_4, D, D, D, D, 1)
    return StructArray([M5_data], [1;;])
end

function ising_exact_free_energy(beta)
    npts = 5000  # smaller for speed; result converges fast
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
# After VUMPS converges, compute λ_AC and free energy.
#
# For c4v Ising with M as rank-5 (D,D,D,D,1) and FL as rank-4 (χ,D,D,χ),
# AL is rank-4 too. ACmap_parallel applies M to AC (rank-4) yielding M_AC.
# Eigenvalue λ_AC = ⟨AC, M_AC⟩ / ⟨AC, AC⟩.
# ============================================================================
function compute_free_energy(rt::C4vVUMPSEnv, M_c4v::StructArray, alg::VUMPS{C4v}, beta::Float64)
    AL = rt.AL
    C  = rt.C
    FL = rt.FL
    M_full = M_c4v[1]                # rank-6: (D,D,D,D,1,1)
    M  = M_full[:,:,:,:,:,1]          # rank-5

    # Method A: official — call ACenv_c4v on the converged state to get λ_AC
    AC0 = TeneT.ALCtoAC_map(AL, C)
    λACs, _ = TeneT.ACenv_c4v(AC0, FL, M; alg)
    λ_AC_official = real(λACs)

    # Method B: Rayleigh quotient on AL·C with manual ACmap_parallel
    @tensor AC[a,b,c,d] := AL[a,b,c,e] * C[e,d]
    M_AC = ACmap_parallel(AC, FL, FL, M; ifparallel=false, forloop_iter=1, inner_etype=nothing)
    λ_AC_rayleigh = real(dot(AC, M_AC) / dot(AC, AC))

    # Method C: λ_FL — official from leftenv on converged state
    λFLs, _ = TeneT.leftenv_c4v(AL, conj(AL), M, FL; alg)
    λ_FL_official = real(λFLs)

    # Method D: λ_C — official from Cenv on converged state
    λCs, _ = TeneT.Cenv_c4v(C, FL; alg)
    λ_C_official = real(λCs)

    # Standard VUMPS for transfer matrix (positive λ): Z per site = λ_AC / λ_C
    Z_per_site_AC_over_C = λ_AC_official / λ_C_official
    f_AC_over_C = -log(Z_per_site_AC_over_C) / beta
    # If bra/ket doubled (rank-5 leg5 → ACmap uses M⊗M*): Z² per site = λ_AC/λ_C, so divide by 2β
    f_AC_over_C_2 = -log(Z_per_site_AC_over_C) / (2*beta)

    return (
        λ_AC = λ_AC_official, λ_AC_R = λ_AC_rayleigh,
        λ_FL = λ_FL_official, λ_C = λ_C_official,
        Z_AC_over_C = Z_per_site_AC_over_C,
        f_AC_over_C = f_AC_over_C, f_AC_over_C_2 = f_AC_over_C_2,
        f_AC_only = -log(λ_AC_official)/beta,
        f_AC_only_2 = -log(λ_AC_official)/(2*beta),
    )
end

# ============================================================================
# Run VUMPS with given config and report f_VUMPS vs f_exact
# ============================================================================
function sanity_one(M_c4v, chi, beta, alg, label, f_exact)
    Random.seed!(42)
    rt = init_env(M_c4v, chi, alg)
    CUDA.synchronize()
    t0 = time()
    rt, err = leading_boundary(rt, M_c4v, alg)
    CUDA.synchronize()
    t = time() - t0

    fr = compute_free_energy(rt, M_c4v, alg, beta)
    @printf("%-30s  err=%.2e  λ_AC=%.4e  λ_FL=%.4e  λ_C=%.4e  λ_AC/λ_C=%.4e\n",
        label, err, fr.λ_AC, fr.λ_FL, fr.λ_C, fr.Z_AC_over_C)
    @printf("    f(λ_AC)/β=%.6f  f(λ_AC)/2β=%.6f  f(λ_AC/λ_C)/β=%.6f  f(λ_AC/λ_C)/2β=%.6f  Δ_min=%.2e  t=%.2fs\n",
        fr.f_AC_only, fr.f_AC_only_2, fr.f_AC_over_C, fr.f_AC_over_C_2,
        min(abs(fr.f_AC_only - f_exact), abs(fr.f_AC_only_2 - f_exact),
            abs(fr.f_AC_over_C - f_exact), abs(fr.f_AC_over_C_2 - f_exact)), t)
    return (label=label, fr=fr, t=t, err=err)
end

# ============================================================================
# Main
# ============================================================================
const CHI = parse(Int, get(ENV, "BENCH_CHI", "128"))
const BETA = parse(Float64, get(ENV, "BENCH_BETA", "0.43"))
const TOL = parse(Float64, get(ENV, "BENCH_TOL", "1e-6"))

println("=" ^ 90)
println("# Sanity: VUMPS{C4v} 2D Ising free energy vs Onsager exact")
println("Date: ", string(now()))
println("Hardware: ", CUDA.name(CUDA.device()))
println("Settings: χ=", CHI, " β=", BETA, " tol=", TOL)
println("=" ^ 90)

f_exact = ising_exact_free_energy(BETA)
@printf("Onsager exact free energy at β=%.3f:  f = %.10f\n", BETA, f_exact)

M_c4v = build_c4v_M(BETA; atype=CuArray)

# Warmup
let alg_w = VUMPS{C4v}(verbosity=0, maxiter=2, miniter=1, miniter_ad=1, maxiter_ad=1,
                      tol=1e-3, ifsimple_eig=true, power_iter=5, ifupdown=false)
    Random.seed!(42); rt_w = init_env(M_c4v, 8, alg_w); leading_boundary(rt_w, M_c4v, alg_w)
    Random.seed!(42); rt_w = init_env(M_c4v, CHI, alg_w); leading_boundary(rt_w, M_c4v, alg_w)
    CUDA.synchronize()
end

println("\nMethod                        VUMPS_err  λ_AC      f(λ_AC)/β     f(λ_AC)/2β    |Δf v1| |Δf v2|  Time")
println("-" ^ 90)

results = []
for pi in [5, 10, 20, 50, 100]
    alg = VUMPS{C4v}(verbosity=0, maxiter=500, miniter=1, miniter_ad=1, maxiter_ad=1,
                    tol=TOL, ifsimple_eig=true, power_iter=pi, ifupdown=false)
    push!(results, sanity_one(M_c4v, CHI, BETA, alg, "simple_eig pi=$pi", f_exact))
end

let alg = VUMPS{C4v}(verbosity=0, maxiter=500, miniter=1, miniter_ad=1, maxiter_ad=1,
                    tol=TOL, ifsimple_eig=false, power_iter=1, ifupdown=false)
    push!(results, sanity_one(M_c4v, CHI, BETA, alg, "GPUKrylov", f_exact))
end

println("\n", "=" ^ 90)
@printf("Onsager: f_exact = %.10f\n", f_exact)
println("If a method is correct, |Δf| should be small.")
println("If method's |Δf v1| ≈ |Δf v2|·factor, see which is closer to 0 to identify the doubling convention.")
