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

"""
    write_md(out_path, rows_raw, rows_wrap1, rows_sweep, fits, env)

Emit benchmark results to a Markdown file matching the JSC/BSC benchmark
style. 4 tables: raw baseline, wrap1 baseline, main sweep, linear fit.
"""
function write_md(out_path::AbstractString,
                  rows_raw::AbstractVector,
                  rows_wrap1::AbstractVector,
                  rows_sweep::AbstractVector,
                  fits::AbstractVector,
                  env::NamedTuple)
    mkpath(dirname(out_path))
    open(out_path, "w") do io
        println(io, "# Local 4090 FLmap forloop AD Benchmark")
        println(io)
        println(io, "- **Date**: $(env.date)")
        println(io, "- **GPU**: $(env.gpu)  ($(env.mem_gb) GiB)")
        println(io, "- **Host**: $(env.hostname)")
        println(io, "- **Julia**: $(env.julia)   **CUDA runtime**: $(env.cuda)")
        println(io, "- **CUDA.jl**: $(env.cuda_jl)   **Zygote.jl**: $(env.zygote)")
        println(io, "- **Tensors**: leg5 bilayer, `FL,ALu,ALd: (χ,D,D,χ)`, `M: (D,D,D,D,2)`, Float64")
        println(io, "- **Protocol**: 1 warmup + 5 reps + median; `GC.gc()` + `CUDA.reclaim()` between reps")
        println(io, "- **Purpose**: quantify Zygote per-pullback overhead in `rrule(forloop)`")
        println(io)

        println(io, "## Table 1: Raw FLmap baseline (no wrapper)")
        println(io)
        println(io, "| D | χ | fwd (ms) | bwd (ms) | bwd/fwd |")
        println(io, "|---|---|----------|----------|---------|")
        for r in rows_raw
            @printf(io, "| %d | %d | %.2f | %.2f | %.2fx |\n",
                    r.D, r.χ, r.t_fwd*1000, r.t_bwd*1000, r.ratio)
        end
        println(io)

        println(io, "## Table 2: rrule wrap baseline (`forloop_iter=1`, early-exit path)")
        println(io)
        println(io, "| D | χ | fwd (ms) | bwd (ms) | bwd/fwd |")
        println(io, "|---|---|----------|----------|---------|")
        for r in rows_wrap1
            @printf(io, "| %d | %d | %.2f | %.2f | %.2fx |\n",
                    r.D, r.χ, r.t_fwd*1000, r.t_bwd*1000, r.ratio)
        end
        println(io)

        println(io, "## Table 3: Main sweep — `t_fwd`, `t_bwd` vs `forloop_iter`")
        println(io)
        println(io, "| D | χ | forloop_iter | fwd (ms) | bwd (ms) | bwd/fwd |")
        println(io, "|---|---|--------------|----------|----------|---------|")
        for r in rows_sweep
            @printf(io, "| %d | %d | %d | %.2f | %.2f | %.2fx |\n",
                    r.D, r.χ, r.forloop_iter, r.t_fwd*1000, r.t_bwd*1000, r.ratio)
        end
        println(io)

        println(io, "## Table 4: Linear fit  `t_bwd(n) ≈ α·n + β`")
        println(io)
        println(io, "| D | χ | α (ms/chunk) | β (ms) | R² | α·128 / (α·128+β) |")
        println(io, "|---|---|--------------|--------|----|--------------------|")
        for f in fits
            share = (f.α*128 + f.β) > 0 ? f.α*128 / (f.α*128 + f.β) : 0.0
            @printf(io, "| %d | %d | %.3f | %.2f | %.4f | %.1f%% |\n",
                    f.D, f.χ, f.α, f.β, f.R², share*100)
        end
        println(io)

        println(io, "## Diagnosis guide")
        println(io)
        println(io, "- If `α·128 ≫ β` (last column large, e.g. >80%): Zygote per-chunk")
        println(io, "  overhead dominates at high `forloop_iter`. Optimize `rrule(forloop)`")
        println(io, "  (cache pullback across chunks, preallocate grad buffers, or write a")
        println(io, "  hand-rolled FLmap rrule).")
        println(io, "- If `α·128 ≈ β` (last column small, e.g. <30%): most cost is fixed")
        println(io, "  per-call work. Likely `@tensor` AD itself — hand-roll FLmap rrule.")
        println(io, "- Compare Table 1 vs Table 2: difference quantifies rrule wrapping cost.")
        println(io, "- If neither α nor β is large yet full-fg `bwd/fwd` is still ~10x,")
        println(io, "  the bottleneck is elsewhere (leftenv/rightenv power-iter unrolling).")
    end
    return out_path
end

function main()
    CUDA.allowscalar(false)
    Random.seed!(42)

    env = env_info()
    print_env(env)

    # -------- sanity checks (kept in-script) ----------------------------
    f = linfit([1.0, 2.0, 3.0, 4.0], [3.0, 5.0, 7.0, 9.0])
    @assert abs(f.α - 2.0) < 1e-10 && abs(f.β - 1.0) < 1e-10
    println("linfit sanity OK")

    # -------- config --------------------------------------------------
    # Dry-run (tiny): uncomment to verify the driver end-to-end quickly.
    # sweep_plan  = [((4, 16), [1, 2, 4]), ((4, 32), [1, 2, 4])]
    # raw_configs = [(4, 16), (4, 32)]
    # nrep        = 2

    # Real benchmark:
    # Per-config iters: skip low forloop_iter on large (D, χ) where the
    # @tensor backward rank-7 intermediate (~χ²·D⁵·16 bytes) OOMs a 24 GB card.
    # At (10, 128) everything fits; at (10, 256) iter=1 OOMs; at (10, 512) and
    # (12, 256) iter<8 OOMs.
    sweep_plan  = [
        ((10, 128), [1, 2, 4, 8, 16, 32, 64, 128]),
        ((10, 256), [2, 4, 8, 16, 32, 64, 128]),
        ((10, 512), [8, 16, 32, 64, 128]),
        ((12, 256), [8, 16, 32, 64, 128]),
    ]
    raw_configs = [(10, 128)]  # only configs where direct FLmap bwd fits
    nrep        = 5

    # -------- sweep ---------------------------------------------------
    println("\n─── Raw baseline ───")
    rows_raw = NamedTuple[]
    for (D, χ) in raw_configs
        GC.gc(); CUDA.reclaim()
        r = run_raw(D, χ; nrep)
        push!(rows_raw, r)
        @printf("  D=%-2d χ=%-4d        fwd=%8.2fms  bwd=%9.2fms  ratio=%.2fx\n",
                r.D, r.χ, r.t_fwd*1000, r.t_bwd*1000, r.ratio)
    end

    println("\n─── Main sweep (wrap + forloop_iter) ───")
    rows_sweep = NamedTuple[]
    for ((D, χ), iters) in sweep_plan, n in iters
        GC.gc(); CUDA.reclaim()
        r = run_config(D, χ, n; nrep)
        push!(rows_sweep, r)
        @printf("  D=%-2d χ=%-4d iter=%-3d fwd=%8.2fms  bwd=%9.2fms  ratio=%.2fx\n",
                r.D, r.χ, r.forloop_iter, r.t_fwd*1000, r.t_bwd*1000, r.ratio)
    end

    # wrap1 rows = forloop_iter=1 slice of sweep (only available for small configs)
    rows_wrap1 = filter(r -> r.forloop_iter == 1, rows_sweep)

    # -------- fit -----------------------------------------------------
    println("\n─── Linear fits ───")
    fits = NamedTuple[]
    for ((D, χ), _) in sweep_plan
        rows = filter(r -> r.D == D && r.χ == χ, rows_sweep)
        xs = Float64[r.forloop_iter for r in rows]
        ys = Float64[r.t_bwd * 1000 for r in rows]  # ms
        f  = linfit(xs, ys)
        push!(fits, (D=D, χ=χ, α=f.α, β=f.β, R²=f.R²))
        @printf("  D=%-2d χ=%-4d  α=%7.3f ms/chunk  β=%7.2f ms  R²=%.4f\n",
                D, χ, f.α, f.β, f.R²)
    end

    # -------- write output --------------------------------------------
    write_md(RESULTS_PATH, rows_raw, rows_wrap1, rows_sweep, fits, env)
    println("\nResults saved to: $RESULTS_PATH")
end

main()
