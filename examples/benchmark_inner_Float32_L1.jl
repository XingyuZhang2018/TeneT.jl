# examples/benchmark_inner_Float32_L1.jl
#
# L1 diagnostic for the inner_etype=Float32 experiment.
# Compares single FLmap_parallel calls in three modes and reports relative errors.
# Design: docs/2026-04-17-inner-float32-vumps-design.md §4.1
# Plan:   docs/2026-04-17-inner-float32-vumps-plan.md Task 10

using Random, Statistics, Printf, LinearAlgebra, Dates, TeneT

const RESULTS_PATH = joinpath(@__DIR__, "..", "docs", "benchmarks",
                              "CPU_inner_Float32.md")

function run_L1(D::Int, χ::Int, seed::Int; d::Int=D)
    Random.seed!(seed)
    FL  = randn(Float64, χ, D, D, χ)
    ALu = randn(Float64, χ, D, D, χ)
    ALd = randn(Float64, χ, D, D, χ)
    M   = randn(Float64, D, D, D, D, d)

    R0  = TeneT.FLmap_parallel(FL, ALu, ALd, M; ifparallel=false, forloop_iter=1,  inner_etype=nothing)
    R16 = TeneT.FLmap_parallel(FL, ALu, ALd, M; ifparallel=false, forloop_iter=16, inner_etype=nothing)
    R32 = TeneT.FLmap_parallel(FL, ALu, ALd, M; ifparallel=false, forloop_iter=1,  inner_etype=Float32)

    nrm = norm(R0)
    err_forloop = norm(R16 .- R0) / nrm
    err_float32 = norm(R32 .- R0) / nrm
    ratio = err_float32 / max(err_forloop, eps(Float64))
    return (D=D, χ=χ, seed=seed, err_forloop=err_forloop,
            err_float32=err_float32, ratio=ratio)
end

function main()
    configs = [(2, 16), (2, 32), (3, 16), (3, 32), (3, 64)]
    seeds = [42, 43, 44]

    rows = []
    for (D, χ) in configs, s in seeds
        push!(rows, run_L1(D, χ, s))
    end

    println("=" ^ 80)
    println("L1 results (single FLmap_parallel call)")
    println("=" ^ 80)
    @printf("%-3s %-4s %-5s %-14s %-14s %-10s\n",
            "D", "χ", "seed", "err_forloop", "err_float32", "ratio")
    for r in rows
        @printf("%-3d %-4d %-5d %-14.3e %-14.3e %-10.3f\n",
                r.D, r.χ, r.seed, r.err_forloop, r.err_float32, r.ratio)
    end

    # Append to Markdown report (create if missing)
    mkpath(dirname(RESULTS_PATH))
    open(RESULTS_PATH, "a") do io
        println(io, "\n## L1 — single FLmap_parallel call (", today(), ", CPU)\n")
        println(io, "| D | χ | seed | err_forloop | err_float32 | ratio |")
        println(io, "|---|---|------|-------------|-------------|-------|")
        for r in rows
            @printf(io, "| %d | %d | %d | %.3e | %.3e | %.3f |\n",
                    r.D, r.χ, r.seed, r.err_forloop, r.err_float32, r.ratio)
        end
    end
    @printf("\nReport appended to %s\n", RESULTS_PATH)

    # Gate decision
    max_ratio = maximum(r.ratio for r in rows)
    println("\nmax ratio = ", max_ratio)
    if max_ratio < 10
        println("GATE: PASS — proceed to L4.")
    elseif max_ratio < 100
        println("GATE: PASS WITH CAUTION — L4 may fail.")
    else
        println("GATE: STOP — reconsider precision strategy (see design §5.2).")
    end
end

main()
