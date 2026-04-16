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
end

main()
