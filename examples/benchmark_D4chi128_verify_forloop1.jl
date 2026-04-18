# Minimal reproducer: JUST F32/coarse=2 at forloop=1, clean GPU state.
# Goal: verify whether the 1585s wall-clock observed in the forloop=1
# run was an artifact of environment (shared GPU, pool fragmentation after
# the F64 run in the same process) or intrinsic to the forloop=1 path.

using TeneT, OptimKit, LinearAlgebra, Random, Zygote, Printf, CUDA, Dates

function gpu_used_bytes()
    CUDA.total_memory() - CUDA.available_memory()
end

function main()
    D, χ = 4, 128
    seed = 42
    forloop_iter = 1
    polish_steps = 2
    maxiter_lbfgs = 20

    @printf("GPU: %s (total %.1f GB, available %.1f GB)\n",
            CUDA.name(CUDA.device()),
            CUDA.total_memory()/1e9,
            CUDA.available_memory()/1e9)
    @printf("Setup: D=%d χ=%d seed=%d forloop=%d polish_steps=%d maxiter=%d\n",
            D, χ, seed, forloop_iter, polish_steps, maxiter_lbfgs)

    Random.seed!(seed)
    CUDA.seed!(seed)
    atype = CuArray
    etype = Float64
    pattern = [1;;]
    model = Heisenberg(lattice=Square(), S=0.5, Jx=1.0, Jy=1.0, Jz=1.0,
                       ifrotate=true, couplingtype=:uniform, bondratio=1.0)
    inner_etype = Float32
    inner_etype_final_steps = polish_steps
    tag = "Float32_coarse2_gpu_fl1_verify"
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
    @printf("GPU used before run: %.0f MB\n", gpu_before/1e6)

    @printf(">>> Starting F32/coarse=2 forloop=1 optimization at %s\n", Dates.now())
    flush(stdout)
    t_start = time()
    result = optimise_ipeps(A, χ, 0, params; restriction_ipeps)
    CUDA.synchronize()
    wall = time() - t_start
    gpu_after = gpu_used_bytes()

    @printf("\n>>> Finished at %s\n", Dates.now())
    @printf("    E=%.12f  n_steps=%d  wall=%.1fs  ΔGPU=%.0fMB\n",
            result[2], size(result[5], 1) - 1, wall, (gpu_after - gpu_before)/1e6)

    # Compare to prior results
    @printf("\n=== Comparison ===\n")
    @printf("Prior forloop=1 F32/coarse=2 wall: 1585.2s (original observation)\n")
    @printf("Prior forloop=2 F32/coarse=2 wall:   67.7s (after forloop=2 pivot)\n")
    @printf("This run   forloop=1 F32/coarse=2 wall: %.1fs\n", wall)

    if wall > 1000
        @printf("→ Pattern: the original 1585s was NOT environment-related; forloop=1 IS pathological for F32.\n")
    elseif wall < 200
        @printf("→ Pattern: original 1585s was LIKELY environment-related (shared GPU / pool fragmentation).\n")
    else
        @printf("→ Pattern: mixed. forloop=1 is genuinely slower than forloop=2 but the original 1585s was partly environmental.\n")
    end
end

main()
