# examples/benchmark_inner_Float32_L4.jl
#
# L4 comparison: full Heisenberg C4v iPEPS optimization under all-Float64
# baseline vs inner_etype=Float32 inside FLmap/FRmap/ACmap @tensor contractions.
# Design: docs/2026-04-17-inner-float32-vumps-design.md §4.2
# Plan:   docs/2026-04-17-inner-float32-vumps-plan.md Task 11
#
# Stage A (this file, D=2): (D=2, χ) × {Float64, Float32} × 3 seeds = 12 runs, ~2-4 min total CPU.
# Stage B (D=3) will be added as a follow-up task if D=2 passes.

using TeneT, OptimKit, LinearAlgebra, Random, Zygote, Printf, Dates, Statistics, CUDA

const RESULTS_PATH = joinpath(@__DIR__, "..", "docs", "benchmarks",
                              "CPU_inner_Float32.md")

function run_L4(D, χ, seed, inner_etype;
                polish_mode::String = "none",
                polish_steps::Int = 2)
    # polish_mode: "none" | "coarse" (inner_etype_final_steps=polish_steps) | "fine" (simple_eig_polish_steps=polish_steps)
    inner_etype_final_steps = polish_mode == "coarse" ? polish_steps : 0
    simple_eig_polish_steps = polish_mode == "fine"   ? polish_steps : 0

    Random.seed!(seed)
    atype = Array
    etype = Float64
    pattern = [1;;]
    model = Heisenberg(lattice=Square(), S=0.5, Jx=1.0, Jy=1.0, Jz=1.0,
                       ifrotate=true, couplingtype=:uniform, bondratio=1.0)
    tag = inner_etype === nothing ? "f64" : string(inner_etype) * "_" * polish_mode * string(polish_steps)
    folder = joinpath(pkgdir(TeneT), "data/bench_L4/$tag/D$(D)_chi$(χ)_s$(seed)/")

    boundary_alg = VUMPS{C4v}(; ifsimple_eig=true, ifparallel=false, ifcheckpoint=true,
                              forloop_iter=1, maxiter=3, miniter=0,
                              maxiter_ad=4, miniter_ad=4, power_iter=1,
                              power_iter_ad=5, power_iter_obs=40,
                              show_every=10, tol=1e-10, verbosity=0,
                              inner_etype=inner_etype,
                              inner_etype_final_steps=inner_etype_final_steps,
                              simple_eig_polish_steps=simple_eig_polish_steps)
    params = GradientOptimize(; model=model, pattern=pattern,
                              boundary_alg=boundary_alg,
                              optimizer=LBFGS(200; maxiter=200, verbosity=0,
                                              gradtol=1e-7,
                                              linesearch=HagerZhangLineSearch(maxfg=5)),
                              ifcheckpoint=false, forloop_iter=1,
                              maxiter_restart=1, verbosity=0, folder=folder,
                              ifSU=false, SUτ=0, ifprecondition=true,
                              iter_precond=0, reuse_env=true,
                              ifsave_env=true, ifload_env=false,
                              ifsave_lbfgs=true, ifload_lbfgs=false)

    A = init_ipeps(; atype, etype, No=0, D=D, χ=χ, params)
    restriction_ipeps(A) = C4v_restriction(A)

    GC.gc()
    rss_before = Sys.maxrss()
    t_start = time()
    result = optimise_ipeps(A, χ, 0, params; restriction_ipeps)
    wall = time() - t_start
    rss_after = Sys.maxrss()

    # optimise_ipeps returns (A, e, eg, fgnum, history)
    # history is an (numiter+1) x 2 matrix: [fhistory normgradhistory]
    E_final = result[2]
    n_steps = size(result[5], 1) - 1

    label = inner_etype === nothing ? "Float64" :
            string(inner_etype) * "/" * polish_mode * "=" * string(polish_steps)
    return (D=D, χ=χ, seed=seed,
            precision=label,
            E=E_final, n_steps=n_steps, wall=wall,
            rss_bytes=(rss_after - rss_before))
end

function main()
    BLAS.set_num_threads(4)
    println("BLAS threads: ", BLAS.get_num_threads())

    # Usage:
    #   julia benchmark_inner_Float32_L4.jl              -> fine polish N=2 (default)
    #   julia benchmark_inner_Float32_L4.jl coarse       -> coarse polish N=2
    #   julia benchmark_inner_Float32_L4.jl fine 3       -> fine polish N=3
    #   julia benchmark_inner_Float32_L4.jl fine_sweep   -> sweep fine polish N ∈ {2,3,4,5}
    mode_arg = length(ARGS) >= 1 ? ARGS[1] : "fine"
    sweep = mode_arg == "fine_sweep"
    equiv = mode_arg == "equiv"
    if sweep
        steps_list = [2, 3, 4, 5]
        polish_modes = ["fine", "fine", "fine", "fine"]
    elseif equiv
        # Sanity check: coarse=1 and fine=5 should be strictly equivalent at power_iter_ad=5
        # (both make the entire last AD iter run in Float64).
        steps_list  = [1,        5]
        polish_modes = ["coarse", "fine"]
    else
        @assert mode_arg in ("none", "coarse", "fine") "ARGS[1] must be none|coarse|fine|fine_sweep|equiv"
        step_arg = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 2
        steps_list = [step_arg]
        polish_modes = [mode_arg]
    end
    @printf("Sweep: %s  polish_modes=%s  steps_list=%s\n",
            sweep ? "fine_sweep" : (equiv ? "equiv" : mode_arg),
            string(polish_modes), string(steps_list))

    configs = [(2, 16), (2, 32)]
    seeds = [42, 43, 44]

    # Build arms: Float64 baseline always + one Float32 arm per polish step count
    arms = Tuple{Any,String,Int}[(nothing, "none", 0)]
    for (mode, n) in zip(polish_modes, steps_list)
        push!(arms, (Float32, mode, n))
    end

    rows = []
    for (D, χ) in configs, s in seeds, (p, mode, n) in arms
        @printf("Running D=%d χ=%d seed=%d precision=%s polish=%s N=%d ...\n",
                D, χ, s, p === nothing ? "Float64" : string(p), mode, n)
        r = run_L4(D, χ, s, p; polish_mode=mode, polish_steps=n)
        push!(rows, r)
        @printf("  → E=%.12f  n_steps=%d  wall=%.1fs\n", r.E, r.n_steps, r.wall)
    end

    mkpath(dirname(RESULTS_PATH))
    open(RESULTS_PATH, "a") do io
        label = sweep ? "fine polish sweep N ∈ $(steps_list)" :
                equiv ? "coarse=1 vs fine=5 equivalence check" :
                "polish mode `$(polish_modes[1])` N=$(steps_list[1])"
        println(io, "\n## L4 D=2 — $label (", today(), ", CPU)\n")
        println(io, "BLAS threads: ", BLAS.get_num_threads())
        println(io, "")
        println(io, "| D | χ | seed | precision | E_final | n_steps | wall (s) | ΔRSS (MB) |")
        println(io, "|---|---|------|-----------|---------|---------|----------|-----------|")
        for r in rows
            @printf(io, "| %d | %d | %d | %s | %.12f | %d | %.1f | %.0f |\n",
                    r.D, r.χ, r.seed, r.precision, r.E, r.n_steps, r.wall,
                    r.rss_bytes / 1e6)
        end
    end

    # Comparison: Float64 baseline vs each Float32 arm
    f32_labels = ["Float32/" * mode * "=" * string(n) for (mode, n) in zip(polish_modes, steps_list)]
    println("\n=== D=2 energy comparison per polish arm (median over seeds) ===")
    for (D, χ) in configs
        E_f64 = [r.E for r in rows if r.D==D && r.χ==χ && r.precision=="Float64"]
        for lbl in f32_labels
            E_f32 = [r.E for r in rows if r.D==D && r.χ==χ && r.precision==lbl]
            dE = abs(median(E_f64) - median(E_f32))
            pass = dE < 1e-7
            @printf("  (D=%d, χ=%d, %s): |ΔE| = %.3e  →  %s\n",
                    D, χ, lbl, dE, pass ? "PASS" : "FAIL")
        end
    end

    println("\nDetailed energies per (D, χ, precision):")
    all_labels = ["Float64"]; append!(all_labels, f32_labels)
    for (D, χ) in configs, p in all_labels
        es = [r.E for r in rows if r.D==D && r.χ==χ && r.precision==p]
        ns = [r.n_steps for r in rows if r.D==D && r.χ==χ && r.precision==p]
        ws = [r.wall for r in rows if r.D==D && r.χ==χ && r.precision==p]
        @printf("  (D=%d, χ=%d, %s): E=%s n_steps=%s wall=%s\n", D, χ, p,
                string(es), string(ns), string(round.(ws, digits=2)))
    end
end

main()
