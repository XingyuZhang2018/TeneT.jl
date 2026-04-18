# Clean-environment benchmark for D=4 χ=128: run exactly ONE (precision, forloop_iter)
# arm per process invocation to avoid CUDA pool fragmentation from cross-arm
# contamination.
#
# Usage:
#   julia examples/benchmark_D4chi128_verify_clean.jl <f64|f32|whole> <forloop_iter> [polish_steps]
# e.g.
#   julia examples/benchmark_D4chi128_verify_clean.jl f64 1
#   julia examples/benchmark_D4chi128_verify_clean.jl f32 2        # coarse=2 (default, inner_etype)
#   julia examples/benchmark_D4chi128_verify_clean.jl f32 1 1      # coarse=1
#   julia examples/benchmark_D4chi128_verify_clean.jl whole 1      # whole-VUMPS F32, coarse=2
#   julia examples/benchmark_D4chi128_verify_clean.jl whole 1 1    # whole-VUMPS F32, coarse=1

using TeneT, OptimKit, LinearAlgebra, Random, Zygote, Printf, CUDA, Dates

function gpu_used_bytes()
    CUDA.total_memory() - CUDA.available_memory()
end

function main()
    @assert length(ARGS) >= 2 "Usage: julia ... <f64|f32|whole> <forloop_iter> [polish_steps]"
    precision_arg = lowercase(ARGS[1])
    forloop_iter = parse(Int, ARGS[2])
    @assert precision_arg in ("f64", "f32", "whole") "precision must be f64|f32|whole"
    # Mode decoding:
    #   f64   → inner_etype=nothing, whole_vumps_etype=nothing (pure Float64)
    #   f32   → inner_etype=Float32, whole_vumps_etype=nothing (per-call downcast)
    #   whole → inner_etype=nothing, whole_vumps_etype=Float32 (state-level Float32)
    inner_etype       = precision_arg == "f32"   ? Float32 : nothing
    whole_vumps_etype = precision_arg == "whole" ? Float32 : nothing
    precision_active  = inner_etype !== nothing || whole_vumps_etype !== nothing
    polish_steps = if length(ARGS) >= 3
        parse(Int, ARGS[3])
    else
        precision_active ? 2 : 0
    end

    D, χ = 4, 128
    seed = 42
    maxiter_lbfgs = 20

    @printf("GPU: %s (total %.1f GB, available %.1f GB)\n",
            CUDA.name(CUDA.device()),
            CUDA.total_memory()/1e9,
            CUDA.available_memory()/1e9)
    mode_label = if whole_vumps_etype !== nothing
        "whole_vumps_" * string(whole_vumps_etype)
    elseif inner_etype !== nothing
        "inner_" * string(inner_etype)
    else
        "Float64"
    end
    @printf("Setup: D=%d χ=%d seed=%d forloop=%d maxiter=%d  mode=%s polish=%d\n",
            D, χ, seed, forloop_iter, maxiter_lbfgs, mode_label, polish_steps)

    Random.seed!(seed)
    CUDA.seed!(seed)
    atype = CuArray
    etype = Float64
    pattern = [1;;]
    model = Heisenberg(lattice=Square(), S=0.5, Jx=1.0, Jy=1.0, Jz=1.0,
                       ifrotate=true, couplingtype=:uniform, bondratio=1.0)
    inner_etype_final_steps = polish_steps
    tag = "clean_" * precision_arg *
          (precision_active ? "_c" * string(polish_steps) : "") *
          "_fl" * string(forloop_iter)
    folder = joinpath(pkgdir(TeneT), "data/bench_D4chi128/$tag/s$(seed)/")

    boundary_alg = VUMPS{C4v}(; ifsimple_eig=true, ifparallel=false, ifcheckpoint=true,
                              forloop_iter=forloop_iter, maxiter=3, miniter=0,
                              maxiter_ad=4, miniter_ad=4, power_iter=1,
                              power_iter_ad=5, power_iter_obs=40,
                              show_every=10, tol=1e-10, verbosity=1,
                              inner_etype=inner_etype,
                              inner_etype_final_steps=inner_etype_final_steps,
                              simple_eig_polish_steps=0,
                              whole_vumps_etype=whole_vumps_etype)
    params = GradientOptimize(; model=model, pattern=pattern,
                              boundary_alg=boundary_alg,
                              optimizer=LBFGS(200; maxiter=maxiter_lbfgs, verbosity=1,
                                              gradtol=1e-7,
                                              linesearch=HagerZhangLineSearch(maxfg=5)),
                              ifcheckpoint=false, forloop_iter=forloop_iter,
                              maxiter_restart=1, verbosity=1, folder=folder,
                              ifSU=false, SUτ=0, ifprecondition=true,
                              iter_precond=0, reuse_env=true,
                              ifsave_env=false, ifload_env=false,
                              ifsave_lbfgs=false, ifload_lbfgs=false)

    A = init_ipeps(; atype, etype, No=0, D=D, χ=χ, params)
    restriction_ipeps(A) = C4v_restriction(A)

    CUDA.synchronize(); GC.gc(); CUDA.reclaim()
    gpu_before = gpu_used_bytes()
    @printf("GPU used before run: %.0f MB\n", gpu_before/1e6)
    @printf(">>> Starting at %s\n", Dates.now())
    flush(stdout)

    t_start = time()
    result = optimise_ipeps(A, χ, 0, params; restriction_ipeps)
    CUDA.synchronize()
    wall = time() - t_start
    gpu_after = gpu_used_bytes()

    @printf("\n>>> Finished at %s\n", Dates.now())
    @printf("    E=%.12f  n_steps=%d  wall=%.1fs  ΔGPU=%.0fMB\n",
            result[2], size(result[5], 1) - 1, wall, (gpu_after - gpu_before)/1e6)

    # Append one-row result to clean summary file
    summary_path = joinpath(@__DIR__, "..", "docs", "benchmarks", "D4chi128_clean_sweep.md")
    mkpath(dirname(summary_path))
    if !isfile(summary_path)
        open(summary_path, "w") do io
            println(io, "# D=4 χ=128 GPU clean-environment sweep\n")
            println(io, "Each row is a SEPARATE Julia invocation (no cross-arm contamination).\n")
            println(io, "| precision | forloop | wall (s) | E_final | n_steps | ΔGPU (MB) |")
            println(io, "|-----------|---------|----------|---------|---------|-----------|")
        end
    end
    row_label = if whole_vumps_etype !== nothing
        "wholeF32/coarse=$polish_steps"
    elseif inner_etype !== nothing
        "innerF32/coarse=$polish_steps"
    else
        "Float64"
    end
    open(summary_path, "a") do io
        @printf(io, "| %s | %d | %.1f | %.12f | %d | %.0f |\n",
                row_label, forloop_iter, wall, result[2], size(result[5], 1) - 1,
                (gpu_after - gpu_before)/1e6)
    end
end

main()
