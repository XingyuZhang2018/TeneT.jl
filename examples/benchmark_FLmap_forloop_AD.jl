# examples/benchmark_FLmap_forloop_AD.jl
#
# Phase 1 diagnostic: measure FLmap forloop AD forward/backward timing
# on a single GPU (local RTX 4090), sweeping forloop_iter to quantify
# Zygote's per-pullback overhead in rrule(forloop).
#
# Design: docs/plans/2026-04-16-flmap-forloop-ad-benchmark-design.md
# Usage:  julia --project=. examples/benchmark_FLmap_forloop_AD.jl

using CUDA, Zygote, Statistics, Printf, Random, LinearAlgebra, Dates, TeneT

const RESULTS_PATH = joinpath(@__DIR__, "MPI_parallel", "benchmarks",
                              "Local_4090_FLmap_AD.md")

function env_info()
    return (
        date     = string(today()),
        gpu      = CUDA.name(CUDA.device()),
        mem_gb   = round(CUDA.total_memory() / 1024^3, digits=1),
        julia    = string(VERSION),
        cuda     = string(CUDA.runtime_version()),
        cuda_jl  = string(pkgversion(CUDA)),
        zygote   = string(pkgversion(Zygote)),
        hostname = gethostname(),
    )
end

function print_env(env)
    println("=" ^ 70)
    println("Local 4090 FLmap forloop AD Benchmark")
    println("=" ^ 70)
    for (k, v) in pairs(env)
        @printf("  %-10s: %s\n", k, v)
    end
    println()
end

"""
    linfit(xs, ys) -> (α, β, R²)

Least-squares fit `y ≈ α·x + β`. Returns slope α, intercept β, and
coefficient of determination R².
"""
function linfit(xs::AbstractVector, ys::AbstractVector)
    n = length(xs)
    @assert length(ys) == n "xs and ys must have same length"
    @assert n >= 2 "need >=2 points for linear fit"
    x̄ = mean(xs); ȳ = mean(ys)
    Sxx = sum((x - x̄)^2 for x in xs)
    Sxy = sum((xs[i] - x̄) * (ys[i] - ȳ) for i in eachindex(xs))
    α = Sxy / Sxx
    β = ȳ - α * x̄
    ŷ = α .* xs .+ β
    SS_res = sum((ys .- ŷ).^2)
    SS_tot = sum((ys .- ȳ).^2)
    R² = SS_tot > 0 ? 1 - SS_res / SS_tot : 1.0
    return (α=α, β=β, R²=R²)
end

"""
    run_config(D, χ, forloop_iter; nrep=5, dtype=Float64, d=2)

Measure forward + backward timing of `FLmap_parallel(..., M; forloop_iter,
ifparallel=false)` with leg5 bilayer tensors. Uses 1 warmup + `nrep` reps,
GC + CUDA.reclaim between reps, median aggregation. Gradient wrt `FL` only.
Returns `(D, χ, forloop_iter, t_fwd, t_bwd, ratio)` where times are in
seconds.
"""
function run_config(D::Int, χ::Int, forloop_iter::Int;
                    nrep::Int=5, dtype::Type=Float64, d::Int=2)
    FL  = CUDA.rand(dtype, χ, D, D, χ)
    ALu = CUDA.rand(dtype, χ, D, D, χ)
    ALd = CUDA.rand(dtype, χ, D, D, χ)
    M   = CUDA.rand(dtype, D, D, D, D, d)

    fwd() = TeneT.FLmap_parallel(FL, ALu, ALd, M; ifparallel=false, forloop_iter)
    loss(x) = sum(TeneT.FLmap_parallel(x, ALu, ALd, M; ifparallel=false, forloop_iter))

    # forward warmup + measure
    _ = fwd(); CUDA.synchronize()
    t_fwds = Float64[]
    for _ in 1:nrep
        GC.gc(); CUDA.reclaim(); CUDA.synchronize()
        push!(t_fwds, @elapsed begin
            _ = fwd()
            CUDA.synchronize()
        end)
    end

    # backward warmup + measure
    _, bp = Zygote.pullback(loss, FL)
    _ = bp(one(dtype)); CUDA.synchronize()
    t_bwds = Float64[]
    for _ in 1:nrep
        GC.gc(); CUDA.reclaim(); CUDA.synchronize()
        push!(t_bwds, @elapsed begin
            _, bp = Zygote.pullback(loss, FL)
            _ = bp(one(dtype))
            CUDA.synchronize()
        end)
    end

    t_fwd = median(t_fwds); t_bwd = median(t_bwds)
    return (D=D, χ=χ, forloop_iter=forloop_iter,
            t_fwd=t_fwd, t_bwd=t_bwd, ratio=t_bwd/t_fwd)
end

"""
    run_raw(D, χ; nrep=5, dtype=Float64, d=2)

Measure forward + backward timing of a direct `FLmap(FL, ALu, ALd, M)` call
(no forloop wrapper, no parallel wrapper). Same tensor layout as
`run_config`. Baseline for isolating rrule(forloop) wrapping overhead.
"""
function run_raw(D::Int, χ::Int; nrep::Int=5, dtype::Type=Float64, d::Int=2)
    FL  = CUDA.rand(dtype, χ, D, D, χ)
    ALu = CUDA.rand(dtype, χ, D, D, χ)
    ALd = CUDA.rand(dtype, χ, D, D, χ)
    M   = CUDA.rand(dtype, D, D, D, D, d)

    fwd() = TeneT.FLmap(FL, ALu, ALd, M)  # leg5 → FLmap(..., M, conj(M)) internally
    loss(x) = sum(TeneT.FLmap(x, ALu, ALd, M))

    _ = fwd(); CUDA.synchronize()
    t_fwds = Float64[]
    for _ in 1:nrep
        GC.gc(); CUDA.reclaim(); CUDA.synchronize()
        push!(t_fwds, @elapsed begin
            _ = fwd(); CUDA.synchronize()
        end)
    end

    _, bp = Zygote.pullback(loss, FL); _ = bp(one(dtype)); CUDA.synchronize()
    t_bwds = Float64[]
    for _ in 1:nrep
        GC.gc(); CUDA.reclaim(); CUDA.synchronize()
        push!(t_bwds, @elapsed begin
            _, bp = Zygote.pullback(loss, FL)
            _ = bp(one(dtype))
            CUDA.synchronize()
        end)
    end

    t_fwd = median(t_fwds); t_bwd = median(t_bwds)
    return (D=D, χ=χ, t_fwd=t_fwd, t_bwd=t_bwd, ratio=t_bwd/t_fwd)
end

function main()
    CUDA.allowscalar(false)
    Random.seed!(42)

    env = env_info()
    print_env(env)

    # sanity check: linfit on synthetic y = 2x + 1
    f = linfit([1.0, 2.0, 3.0, 4.0], [3.0, 5.0, 7.0, 9.0])
    @assert abs(f.α - 2.0) < 1e-10 "linfit α failed: got $(f.α)"
    @assert abs(f.β - 1.0) < 1e-10 "linfit β failed: got $(f.β)"
    @assert abs(f.R² - 1.0) < 1e-10 "linfit R² failed: got $(f.R²)"
    println("linfit sanity check OK  (α=$(f.α), β=$(f.β), R²=$(f.R²))")

    # smoke test: tiny config to verify run_config wiring
    r = run_config(4, 16, 2; nrep=2)
    @assert r.t_fwd > 0 && isfinite(r.t_fwd) "t_fwd bad: $(r.t_fwd)"
    @assert r.t_bwd > 0 && isfinite(r.t_bwd) "t_bwd bad: $(r.t_bwd)"
    @printf("run_config smoke OK: D=%d χ=%d iter=%d  fwd=%.2fms  bwd=%.2fms  ratio=%.2fx\n",
            r.D, r.χ, r.forloop_iter, r.t_fwd*1000, r.t_bwd*1000, r.ratio)

    r_raw = run_raw(4, 16; nrep=2)
    @assert r_raw.t_fwd > 0 && isfinite(r_raw.t_fwd)
    @assert r_raw.t_bwd > 0 && isfinite(r_raw.t_bwd)
    @printf("run_raw  smoke OK: D=%d χ=%d          fwd=%.2fms  bwd=%.2fms  ratio=%.2fx\n",
            r_raw.D, r_raw.χ, r_raw.t_fwd*1000, r_raw.t_bwd*1000, r_raw.ratio)
end

main()
