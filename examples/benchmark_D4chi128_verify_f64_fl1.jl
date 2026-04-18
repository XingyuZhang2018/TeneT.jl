# Companion reproducer: JUST F64 at forloop=1, clean GPU state.
# Paired with benchmark_D4chi128_verify_forloop1.jl (F32/coarse=2 clean forloop=1)
# to enable a fair F64 vs F32 comparison at forloop=1 without cross-arm
# pool-fragmentation contamination.

using TeneT, OptimKit, LinearAlgebra, Random, Zygote, Printf, CUDA, Dates

function gpu_used_bytes()
    CUDA.total_memory() - CUDA.available_memory()
end

function main()
    D, χ = 4, 128
    seed = 42
    forloop_iter = 1
    maxiter_lbfgs = 20

    @printf("GPU: %s (total %.1f GB, available %.1f GB)\n",
            CUDA.name(CUDA.device()),
            CUDA.total_memory()/1e9,
            CUDA.available_memory()/1e9)
    @printf("Setup: D=%d χ=%d seed=%d forloop=%d maxiter=%d  precision=Float64\n",
            D, χ, seed, forloop_iter, maxiter_lbfgs)

    Random.seed!(seed)
    CUDA.seed!(seed)
    atype = CuArray
    etype = Float64
    pattern = [1;;]
    model = Heisenberg(lattice=Square(), S=0.5, Jx=1.0, Jy=1.0, Jz=1.0,
                       ifrotate=true, couplingtype=:uniform, bondratio=1.0)
    tag = "f64_gpu_fl1_verify"
    folder = joinpath(pkgdir(TeneT), "data/bench_D4chi128/$tag/s$(seed)/")

    boundary_alg = VUMPS{C4v}(; ifsimple_eig=true, ifparallel=false, ifcheckpoint=true,
                              forloop_iter=forloop_iter, maxiter=3, miniter=0,
                              maxiter_ad=4, miniter_ad=4, power_iter=1,
                              power_iter_ad=5, power_iter_obs=40,
                              show_every=10, tol=1e-10, verbosity=1,
                              inner_etype=nothing,
                              inner_etype_final_steps=0,
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
    @printf("GPU used before run: %.0f MB\n", gpu_before/1e6)

    @printf(">>> Starting F64 forloop=1 optimization at %s\n", Dates.now())
    flush(stdout)
    t_start = time()
    result = optimise_ipeps(A, χ, 0, params; restriction_ipeps)
    CUDA.synchronize()
    wall = time() - t_start
    gpu_after = gpu_used_bytes()

    @printf("\n>>> Finished at %s\n", Dates.now())
    @printf("    E=%.12f  n_steps=%d  wall=%.1fs  ΔGPU=%.0fMB\n",
            result[2], size(result[5], 1) - 1, wall, (gpu_after - gpu_before)/1e6)

    @printf("\n=== Comparison ===\n")
    @printf("Prior combined-script F64 forloop=1 wall: 772.6s (as first arm in dual-arm script)\n")
    @printf("This run clean F64 forloop=1 wall: %.1fs\n", wall)
    @printf("Verify F32/coarse=2 forloop=1 clean: 146.8s (prior verify run)\n")
    if wall > 400
        @printf("→ The 772s was reproducible; forloop=1 F64 is genuinely slow (not contaminated since F64 ran first).\n")
    else
        @printf("→ F64 also varies by environment; need more runs to determine baseline.\n")
    end
end

main()
