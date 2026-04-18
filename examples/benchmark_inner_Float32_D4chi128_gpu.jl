# examples/benchmark_inner_Float32_D4chi128_gpu.jl
#
# GPU scaling test: Heisenberg C4v at D=4, χ=128 on CuArray.
# Compares Float64 baseline vs inner_etype=Float32 with coarse=2 polish.
# Design: docs/2026-04-17-inner-float32-vumps-design.md (§4.2, GPU phase)
#
# Expected: at consumer-GPU (RTX 4090) F64 throughput is 1/64 of F32, so
# Float32 inner contractions should give multi-× wall-clock speedup while
# polish keeps final energy within ~1e-8 of Float64.

using TeneT, OptimKit, LinearAlgebra, Random, Zygote, Printf, Dates, Statistics, CUDA

const RESULTS_PATH = joinpath(@__DIR__, "..", "docs", "benchmarks",
                              "CPU_inner_Float32.md")

function gpu_used_bytes()
    CUDA.total_memory() - CUDA.available_memory()
end

function run_D4chi128_gpu(seed, inner_etype;
                         polish_steps::Int = 2,
                         maxiter_lbfgs::Int = 20,
                         forloop_iter::Int = 1,
                         D::Int = 4, χ::Int = 128)
    Random.seed!(seed)
    CUDA.seed!(seed)    # independent CUDA RNG seeding so runs are reproducible
    atype = CuArray
    etype = Float64
    pattern = [1;;]
    model = Heisenberg(lattice=Square(), S=0.5, Jx=1.0, Jy=1.0, Jz=1.0,
                       ifrotate=true, couplingtype=:uniform, bondratio=1.0)
    inner_etype_final_steps = (inner_etype !== nothing && polish_steps > 0) ? polish_steps : 0
    tag = (inner_etype === nothing ? "f64_gpu" :
           string(inner_etype) * "_coarse" * string(polish_steps) * "_gpu") *
          "_fl$(forloop_iter)"
    folder = joinpath(pkgdir(TeneT), "data/bench_D4chi128/$tag/s$(seed)/")

    boundary_alg = VUMPS{C4v}(; ifsimple_eig=true, ifparallel=false, ifcheckpoint=true,
                              forloop_iter=forloop_iter, maxiter=3, miniter=0,
                              maxiter_ad=4, miniter_ad=4, power_iter=1,
                              power_iter_ad=5, power_iter_obs=40,
                              show_every=10, tol=1e-10, verbosity=1,
                              inner_etype=inner_etype,
                              inner_etype_final_steps=inner_etype_final_steps,
                              simple_eig_polish_steps=0)
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

    t_start = time()
    result = optimise_ipeps(A, χ, 0, params; restriction_ipeps)
    CUDA.synchronize()
    wall = time() - t_start

    gpu_after = gpu_used_bytes()
    gpu_delta_mb = (gpu_after - gpu_before) / 1e6

    E_final = result[2]
    n_steps = size(result[5], 1) - 1

    label = inner_etype === nothing ? "Float64" :
            string(inner_etype) * "/coarse=" * string(polish_steps)
    return (; seed, label, E=E_final, n_steps, wall, gpu_delta_mb)
end

function main()
    D, χ = 4, 128
    @printf("================================================================================\n")
    @printf("GPU experiment: Heisenberg C4v at D=%d χ=%d\n", D, χ)
    @printf("GPU: %s, total %.1fGB, available %.1fGB\n",
            CUDA.name(CUDA.device()),
            CUDA.total_memory()/1e9,
            CUDA.available_memory()/1e9)
    @printf("================================================================================\n\n")

    seed = 42
    maxiter_lbfgs = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 20
    forloop_iter  = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 1
    @printf("Seed: %d, LBFGS maxiter: %d, forloop_iter: %d\n\n", seed, maxiter_lbfgs, forloop_iter)

    @printf(">>> Float64 baseline, seed=%d, maxiter=%d, forloop=%d\n", seed, maxiter_lbfgs, forloop_iter)
    flush(stdout)
    r64 = run_D4chi128_gpu(seed, nothing; maxiter_lbfgs, forloop_iter, D, χ)
    @printf("    E=%.12f  n_steps=%d  wall=%.1fs  ΔGPU=%.0fMB\n\n",
            r64.E, r64.n_steps, r64.wall, r64.gpu_delta_mb)
    flush(stdout)

    @printf(">>> Float32/coarse=2, seed=%d, maxiter=%d, forloop=%d\n", seed, maxiter_lbfgs, forloop_iter)
    flush(stdout)
    r32 = run_D4chi128_gpu(seed, Float32; polish_steps=2, maxiter_lbfgs, forloop_iter, D, χ)
    @printf("    E=%.12f  n_steps=%d  wall=%.1fs  ΔGPU=%.0fMB\n\n",
            r32.E, r32.n_steps, r32.wall, r32.gpu_delta_mb)
    flush(stdout)

    @printf("=== Comparison ===\n")
    @printf("|ΔE|              = %.3e\n", abs(r64.E - r32.E))
    @printf("speedup (F64/F32) = %.2fx (wall)\n", r64.wall / r32.wall)

    mkpath(dirname(RESULTS_PATH))
    open(RESULTS_PATH, "a") do io
        println(io, "\n## L4 D=4 χ=128 — GPU, forloop=$forloop_iter (", today(), ")\n")
        println(io, "Hardware: ", CUDA.name(CUDA.device()), " (", round(CUDA.total_memory()/1e9, digits=1), " GB)")
        println(io, "Seed: ", seed, ", LBFGS maxiter cap: ", maxiter_lbfgs,
                    ", gradtol: 1e-7, forloop_iter: ", forloop_iter)
        println(io, "")
        println(io, "| precision | E_final | n_steps | wall (s) | ΔGPU (MB) |")
        println(io, "|-----------|---------|---------|----------|-----------|")
        @printf(io, "| Float64 | %.12f | %d | %.1f | %.0f |\n", r64.E, r64.n_steps, r64.wall, r64.gpu_delta_mb)
        @printf(io, "| Float32/coarse=2 | %.12f | %d | %.1f | %.0f |\n", r32.E, r32.n_steps, r32.wall, r32.gpu_delta_mb)
        println(io, "")
        @printf(io, "- **Speedup (F64/F32 wall):** %.2fx\n", r64.wall / r32.wall)
        @printf(io, "- **|ΔE|:** %.3e\n", abs(r64.E - r32.E))
    end
    @printf("\nReport appended to %s\n", RESULTS_PATH)
end

main()
