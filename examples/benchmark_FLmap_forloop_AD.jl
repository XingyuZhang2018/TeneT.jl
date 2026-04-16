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

function main()
    CUDA.allowscalar(false)
    Random.seed!(42)

    env = env_info()
    print_env(env)

    println("TODO: implement sweep")
end

main()
