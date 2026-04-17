# examples/benchmark_FLmap_D4chi128_gpu.jl
#
# Pure-FLmap diagnostic at D=4 χ=128 on GPU.
# Goal: isolate per-call cost of FLmap (Float64) vs FLmap(inner_etype=Float32)
# WITHOUT VUMPS / simple_eig / LBFGS overhead — to see if the F32 slowdown
# observed in the full D=4 χ=128 L4 test is at the FLmap level or elsewhere.

using CUDA, TeneT, Random, Statistics, Printf, LinearAlgebra, Zygote

const RESULTS_PATH = joinpath(@__DIR__, "..", "docs", "benchmarks",
                              "CPU_inner_Float32.md")

# Warmup + timed reps (median) with GPU synchronize before/after each.
function time_fwd(f, args...; nwarmup=2, nreps=5)
    for _ in 1:nwarmup
        CUDA.synchronize()
        _ = f(args...)
        CUDA.synchronize()
    end
    ts = Float64[]
    for _ in 1:nreps
        GC.gc(); CUDA.reclaim(); CUDA.synchronize()
        t0 = time()
        _ = f(args...)
        CUDA.synchronize()
        push!(ts, time() - t0)
    end
    return median(ts)
end

function time_fwdback(f, args...; nwarmup=2, nreps=5)
    loss(FL, ALu, ALd, M) = sum(f(FL, ALu, ALd, M))
    for _ in 1:nwarmup
        _, bp = Zygote.pullback(loss, args...)
        _ = bp(1.0)
        CUDA.synchronize()
    end
    ts = Float64[]
    for _ in 1:nreps
        GC.gc(); CUDA.reclaim(); CUDA.synchronize()
        t0 = time()
        _, bp = Zygote.pullback(loss, args...)
        _ = bp(1.0)
        CUDA.synchronize()
        push!(ts, time() - t0)
    end
    return median(ts)
end

function main()
    D, χ, d = 4, 128, 4
    T = Float64
    @printf("================================================================================\n")
    @printf("FLmap diagnostic: D=%d χ=%d d=%d on %s\n", D, χ, d, CUDA.name(CUDA.device()))
    @printf("GPU total %.1f GB, available %.1f GB\n",
            CUDA.total_memory()/1e9, CUDA.available_memory()/1e9)
    @printf("================================================================================\n\n")

    Random.seed!(42); CUDA.seed!(42)
    # leg5 bilayer: FL/ALu/ALd are (χ, D, D, χ); M is (D, D, D, D, d)
    FL  = CUDA.rand(T, χ, D, D, χ)
    ALu = CUDA.rand(T, χ, D, D, χ)
    ALd = CUDA.rand(T, χ, D, D, χ)
    M   = CUDA.rand(T, D, D, D, D, d)
    @printf("Tensor sizes: FL=%s (%.1f MB); M=%s (%.3f MB)\n\n",
            size(FL), sizeof(FL)/1e6, size(M), sizeof(M)/1e6)

    # Define the three FLmap variants we care about
    f_raw_f64 = (FL, ALu, ALd, M) -> TeneT.FLmap(FL, ALu, ALd, M; inner_etype=nothing)
    f_raw_f32 = (FL, ALu, ALd, M) -> TeneT.FLmap(FL, ALu, ALd, M; inner_etype=Float32)
    f_par_f64 = (FL, ALu, ALd, M) -> TeneT.FLmap_parallel(FL, ALu, ALd, M; ifparallel=false, forloop_iter=1, inner_etype=nothing)
    f_par_f32 = (FL, ALu, ALd, M) -> TeneT.FLmap_parallel(FL, ALu, ALd, M; ifparallel=false, forloop_iter=1, inner_etype=Float32)

    println("--- Forward only (median over 5 reps after 2 warmups) ---")
    t_raw_f64 = time_fwd(f_raw_f64, FL, ALu, ALd, M)
    @printf("FLmap             F64     : %6.1f ms\n", t_raw_f64*1000)
    t_raw_f32 = time_fwd(f_raw_f32, FL, ALu, ALd, M)
    @printf("FLmap             F32-inner: %6.1f ms  (F32/F64 ratio = %.2f)\n",
            t_raw_f32*1000, t_raw_f32/t_raw_f64)
    t_par_f64 = time_fwd(f_par_f64, FL, ALu, ALd, M)
    @printf("FLmap_parallel    F64     : %6.1f ms\n", t_par_f64*1000)
    t_par_f32 = time_fwd(f_par_f32, FL, ALu, ALd, M)
    @printf("FLmap_parallel    F32-inner: %6.1f ms  (F32/F64 ratio = %.2f)\n",
            t_par_f32*1000, t_par_f32/t_par_f64)

    println("\n--- Forward + backward (Zygote pullback + cotangent) ---")
    tb_raw_f64 = time_fwdback(f_raw_f64, FL, ALu, ALd, M)
    @printf("FLmap             F64     : %6.1f ms\n", tb_raw_f64*1000)
    tb_raw_f32 = time_fwdback(f_raw_f32, FL, ALu, ALd, M)
    @printf("FLmap             F32-inner: %6.1f ms  (F32/F64 ratio = %.2f)\n",
            tb_raw_f32*1000, tb_raw_f32/tb_raw_f64)

    # Numerical sanity
    r_f64 = Array(f_raw_f64(FL, ALu, ALd, M))
    r_f32 = Array(f_raw_f32(FL, ALu, ALd, M))
    rel_err = maximum(abs, r_f32 .- r_f64) / maximum(abs, r_f64)
    @printf("\nRelative error F32 vs F64: %.3e\n", rel_err)

    open(RESULTS_PATH, "a") do io
        println(io, "\n## FLmap single-call diagnostic at D=4 χ=128 on GPU")
        println(io, "")
        println(io, "Hardware: ", CUDA.name(CUDA.device()), " (", round(CUDA.total_memory()/1e9, digits=1), " GB)")
        println(io, "")
        println(io, "| call | precision | ms (median of 5) | F32/F64 ratio |")
        println(io, "|------|-----------|------------------|---------------|")
        @printf(io, "| FLmap forward             | F64       | %.1f | 1.00 |\n", t_raw_f64*1000)
        @printf(io, "| FLmap forward             | F32-inner | %.1f | %.2f |\n", t_raw_f32*1000, t_raw_f32/t_raw_f64)
        @printf(io, "| FLmap_parallel forward    | F64       | %.1f | 1.00 |\n", t_par_f64*1000)
        @printf(io, "| FLmap_parallel forward    | F32-inner | %.1f | %.2f |\n", t_par_f32*1000, t_par_f32/t_par_f64)
        @printf(io, "| FLmap forward+backward    | F64       | %.1f | 1.00 |\n", tb_raw_f64*1000)
        @printf(io, "| FLmap forward+backward    | F32-inner | %.1f | %.2f |\n", tb_raw_f32*1000, tb_raw_f32/tb_raw_f64)
        println(io, "")
        @printf(io, "Single-call F32 rel_err vs F64: %.3e\n", rel_err)
    end
    @printf("\nAppended to %s\n", RESULTS_PATH)
end

main()
