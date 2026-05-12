# Bench: VUMPS forward + AD on a random rank-4 MPO at D=4, χ=64.
#
# Compares 3 boundary-eigsolve configurations:
#   A) simple_eig with power_iter_ad=5  (TeneT default)
#   B) simple_eig with power_iter_ad=40 (more accurate)
#   C) GPUKrylov via install_krylovkit_override!()
#
# Why this shape vs full iPEPS optimize: at D=4 χ=64 with ComplexF64, TeneT's
# C4v_restriction triggers a 6D broadcast that hits a known CUDA 13.0/13.2
# toolchain mismatch on this Windows machine (PTX compile fails). We sidestep
# by using a random MPO directly. State size (FL[χ, D², χ] = 65536) and the
# leftenv operator structure are identical to the iPEPS production case —
# we're measuring exactly the eigsolve+linsolve cost that dominates an iPEPS
# step.
#
# Methodology:
#   - Phase 1 (configs A, B): runs BEFORE install_krylovkit_override!()
#   - Phase 2 (config C): GPUKrylov override installed, run config C
#   - For each config: warmup eigsolve + warmup gradient call, then time both.
#   - The gradient is `Zygote.gradient(p -> compute_f(...), p0)` through a
#     parameterized M(p) = M0 + p·M1.

using TeneT, CUDA, LinearAlgebra, TensorOperations, KrylovKit, Random, Printf, Dates
using TeneT: StructArray, VUMPSRuntime, init_env, leading_boundary,
             ACenv, Cenv, ALCtoAC
using Zygote

# ============================================================================
# iPEPS-derived MPO at D=4: build random iPEPS tensor A[D,D,D,D,d] then
# contract A·conj(A) along physical leg → M[D², D², D², D²]. This is the actual
# iPEPS transfer matrix shape — well-conditioned for VUMPS (unlike a totally
# random MPO which has no clean dominant eigenvector).
# ============================================================================
function ipeps_mpo_pair(D::Int, d::Int=2; seed::Int=42, atype=CuArray, etype=ComplexF64)
    Random.seed!(seed)
    s = real(etype)(sqrt(D * D))
    A0 = randn(etype, D, D, D, D, d) ./ s
    A1 = randn(etype, D, D, D, D, d) ./ s
    function build_M(A)
        @tensor M[a, ap, b, bp, c, cp, d, dp] := A[a, b, c, d, σ] * conj(A)[ap, bp, cp, dp, σ]
        DM = D * D
        return reshape(M, DM, DM, DM, DM)
    end
    M0 = build_M(A0)
    M1 = build_M(A1)
    return atype(M0), atype(M1)
end

# Pattern matrix is constant; hoisted outside any Zygote-traced code path
# (the `[1;;]` literal expands to `hvncat` which mutates a Matrix{Int} buffer
# and Zygote can't AD through that).
const STRUCT_PATTERN = reshape([1], 1, 1)

function build_mpo_struct(M::CuArray{T,4}) where {T}
    return StructArray([M], STRUCT_PATTERN)
end

# ============================================================================
# Loss = real(λ_AC / λ_C) ≈ -log(Z) etc., used as gradient driver.
# We just want a scalar that depends on p through M(p) and λ.
# ============================================================================
function compute_loss(rt::VUMPSRuntime, M::StructArray, alg::VUMPS)
    AC = ALCtoAC(rt.AL, rt.C)
    λACs, _ = ACenv(AC, rt.FL, M, rt.FR; alg)
    λCs,  _ =  Cenv(rt.C, rt.FL, rt.FR; alg)
    return real(λACs[1] / λCs[1])
end

function full_loss(p::Float64, M0::CuArray, M1::CuArray, rt0::VUMPSRuntime, alg::VUMPS)
    M_p = M0 .+ p .* M1
    M_struct = build_mpo_struct(M_p)
    rt, err = leading_boundary(rt0, M_struct, alg)
    return compute_loss(rt, M_struct, alg)
end

# ============================================================================
# Bench one config
# ============================================================================
function bench_one(label::String, alg::VUMPS, M0::CuArray, M1::CuArray, chi::Int;
                   n_grad_warmup::Int=1, n_grad_samples::Int=1, skip_gradient::Bool=false)
    println("\n" * "─" ^ 80)
    println("[$label]")
    println("─" ^ 80)
    flush(stdout)

    # Build initial runtime at p=0 with the un-perturbed MPO
    M_struct0 = build_mpo_struct(M0)
    Random.seed!(42)
    rt0 = init_env(M_struct0, chi, alg)

    # ---- Forward: leading_boundary timing (warmup + 1 sample) ----
    Random.seed!(42); rt_w = init_env(M_struct0, chi, alg)
    leading_boundary(rt_w, M_struct0, alg); CUDA.synchronize()  # warmup

    Random.seed!(42); rt = init_env(M_struct0, chi, alg)
    CUDA.synchronize()
    t_fwd = @elapsed begin
        rt, err = leading_boundary(rt, M_struct0, alg)
        CUDA.synchronize()
    end
    # err is Float64 for ifupdown=false, or Tuple{Float64,Float64} for ifupdown=true
    err_max = err isa Tuple ? maximum(err) : err
    @printf "  forward (leading_boundary): t=%.3fs  err=%.2e\n" t_fwd err_max
    flush(stdout)

    # ---- Gradient: Zygote.gradient(p -> full_loss(p, ...), p0) ----
    if skip_gradient
        @printf "  gradient: SKIPPED (rrule not registered)\n"
        flush(stdout)
        return (label=label, t_fwd=t_fwd, err=err_max, t_grad=NaN, g=NaN)
    end

    p0 = 0.05
    # Warmup
    for _ in 1:n_grad_warmup
        Zygote.gradient(p -> full_loss(p, M0, M1, rt0, alg), p0)
        CUDA.synchronize()
    end
    # Timed sample(s)
    times = Float64[]
    for _ in 1:n_grad_samples
        t = @elapsed begin
            g, = Zygote.gradient(p -> full_loss(p, M0, M1, rt0, alg), p0)
            CUDA.synchronize()
            global last_g = g
        end
        push!(times, t)
    end
    t_grad = minimum(times)
    @printf "  gradient (Zygote): t=%.3fs  g=%.6f  finite=%s  grad/fwd=%.2fx\n" t_grad last_g isfinite(last_g) (t_grad / t_fwd)
    flush(stdout)

    return (label=label, t_fwd=t_fwd, err=err_max, t_grad=t_grad, g=last_g)
end

# ============================================================================
# Main
# ============================================================================
const CHI = 64
const D   = 4
const ETYPE = ComplexF64
const TOL = 1e-10

println("=" ^ 100)
println("# VUMPS forward + AD bench at D=$D, χ=$CHI, $ETYPE")
println("Hardware: ", CUDA.name(CUDA.device()))
println("CUDA: ", CUDA.runtime_version(), "  Julia: ", VERSION)
println("Date: ", string(now()))
println("State dim N = χ²·D² = ", CHI^2 * D^2)
println("=" ^ 100)
flush(stdout)

M0, M1 = ipeps_mpo_pair(D, 2; seed=42, atype=CuArray, etype=ETYPE)

# Common VUMPS settings — ifupdown=false to keep init_env / leading_boundary
# returning a single VUMPSRuntime (not a Tuple) for simpler bench plumbing.
# Workload semantics identical: same eigsolve calls, same matvec count.
const COMMON = (ifupdown=false,
                ifdownfromup=false,
                ifparallelupdown=false,
                ifparallel=false,
                forloop_iter=1,
                maxiter=30,
                miniter=0,
                maxiter_ad=4,
                miniter_ad=4,
                power_iter=1,
                power_iter_obs=40,
                show_every=10,
                tol=TOL,
                verbosity=0)

# Phase 1 — simple_eig configs (must run BEFORE override install)
results = []
println("\n## Phase 1: simple_eig configurations\n")
flush(stdout)

alg_A = VUMPS{General}(; COMMON..., ifsimple_eig=true, power_iter_ad=5)
push!(results, bench_one("A: simple_eig power_iter_ad=5", alg_A, M0, M1, CHI))

alg_B = VUMPS{General}(; COMMON..., ifsimple_eig=true, power_iter_ad=40)
push!(results, bench_one("B: simple_eig power_iter_ad=40", alg_B, M0, M1, CHI))

# Phase 2 — GPUKrylov (install override)
println("\n## Phase 2: GPUKrylov override\n")
flush(stdout)

import Pkg
gpukrylov_path = "D:/1 - research/1.19 - GPU/GPUKrylov.jl"
if !haskey(Pkg.project().dependencies, "GPUKrylov")
    Pkg.develop(path=gpukrylov_path)
end
@eval using ChainRulesCore
@eval using GPUKrylov

# The GPUKrylovChainRulesCoreExt extension can fail to auto-load when GPUKrylov
# is added via `Pkg.develop` at runtime (circular-dep precompile skip). Try
# multiple recovery approaches:
ext = Base.get_extension(GPUKrylov, :GPUKrylovChainRulesCoreExt)
if ext === nothing
    @info "Extension didn't auto-load. Trying Base.retry_load_extensions()..."
    Base.retry_load_extensions()
    ext = Base.get_extension(GPUKrylov, :GPUKrylovChainRulesCoreExt)
end
const HAS_GK_RRULE = ext !== nothing
if !HAS_GK_RRULE
    @warn "GPUKrylov ChainRulesCore extension did NOT load (Julia ext-graph " *
          "precompile-skip in this TeneT dep env). Forward GPUKrylov override " *
          "will work, but Zygote.gradient through it will fail with `llvmcall " *
          "requires the compiler` because the rrule is not registered. Will " *
          "skip GK gradient and report only GK forward time."
end
GPUKrylov.install_krylovkit_override!()

alg_C = VUMPS{General}(; COMMON..., ifsimple_eig=false, power_iter_ad=5)  # power_iter_ad ignored when ifsimple_eig=false
push!(results, bench_one("C: GPUKrylov (ifsimple_eig=false)", alg_C, M0, M1, CHI; skip_gradient = !HAS_GK_RRULE))

# ============================================================================
# Summary
# ============================================================================
println("\n" * "=" ^ 100)
println("# Summary  (D=$D, χ=$CHI, $ETYPE, tol=$TOL)")
println("=" ^ 100)
@printf "%-40s  %12s  %12s  %12s\n" "Config" "Forward (s)" "Gradient (s)" "grad/fwd"
println("-" ^ 100)
for r in results
    grad_str = isnan(r.t_grad) ? "  (skipped)" : @sprintf("%12.3f", r.t_grad)
    ratio_str = isnan(r.t_grad) ? "       --" : @sprintf("%12.2fx", r.t_grad / r.t_fwd)
    @printf "%-40s  %12.3f  %s  %s\n" r.label r.t_fwd grad_str ratio_str
end

# Speedup table — relative to slowest forward and slowest gradient
println()
@printf "## Speedup vs simple_eig power_iter_ad=5 baseline\n"
println()
base_fwd  = results[1].t_fwd
base_grad = results[1].t_grad
@printf "%-40s  %12s  %12s\n" "Config" "Forward x" "Gradient x"
println("-" ^ 70)
for r in results
    grad_x = isnan(r.t_grad) ? "       --" : @sprintf("%12.2fx", base_grad / r.t_grad)
    @printf "%-40s  %12.2fx  %s\n" r.label (base_fwd / r.t_fwd) grad_x
end
flush(stdout)
