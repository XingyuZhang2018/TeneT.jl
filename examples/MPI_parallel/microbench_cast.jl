# Micro-benchmark: FLmap_parallel at various forloop_iter values.
#
# FLmap_parallel splits only ALd's last leg (and the corresponding output leg);
# FL, ALu, M stay full-size across all forloop iterations. So each sub-kernel
# still does a big reduction over full chi; only the output slice is small.
#
# This is the RIGHT comparison for our production use: ifparallel=false,
# varying forloop_iter, with inner_etype=nothing vs inner_etype=Float32 (the
# new parallel-level boundary cast).
#
# Run: julia --project=../.. microbench_cast.jl [D] [chi] [N]

using CUDA, TeneT, Statistics, Printf

D   = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 10
chi = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 400
N   = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 20
CUDA.device!(0)

println("=== FLmap_parallel micro-benchmark  D=$D  chi=$chi  N=$N ===")
println("GPU: ", CUDA.name(CUDA.device()))
CUDA.synchronize()

D_phys = 2
FL  = CUDA.rand(ComplexF64, chi, D, D, chi)
ALu = CUDA.rand(ComplexF64, chi, D, D, chi)
ALd = CUDA.rand(ComplexF64, chi, D, D, chi)
M1  = CUDA.rand(ComplexF64, D, D, D, D, D_phys)
M2  = CUDA.rand(ComplexF64, D, D, D, D, D_phys)
M = (M1, M2)

# ────── Warmup (compile for every forloop_iter we will use) ──────
println("Warmup...")
warm_min = chi >= 400 ? 4 : 1   # avoid OOM at chi=400 forloop=1
for forloop_iter in [warm_min, 8, 32, 128]
    forloop_iter > chi && continue
    for prec in [nothing, Float32]
        TeneT.FLmap_parallel(FL, ALu, ALd, M; ifparallel=false, forloop_iter, inner_etype=prec)
        CUDA.synchronize()
    end
end

function time_call(f, N)
    CUDA.synchronize()
    ts = Float64[]
    for _ in 1:N
        t0 = time_ns()
        f()
        CUDA.synchronize()
        push!(ts, (time_ns() - t0) / 1e6)
    end
    return ts
end

m(v) = median(length(v) > 5 ? v[5:end] : v)

@printf("\n%-8s | %-10s %-10s %-10s %-10s %-10s\n",
        "forloop", "F64 ms", "F32 ms", "F32/F64", "ratio", "chi_chunk")
println("-"^70)

# chi=400 forloop_iter=1 OOMs (23 GiB intermediate); start from 4.
min_forloop = chi >= 400 ? 4 : 1
for forloop_iter in [1, 4, 8, 16, 32, 64, 128]
    # Skip if chi can't be divided this finely or would OOM
    forloop_iter < min_forloop && continue
    forloop_iter > chi && continue
    chi_chunk = chi ÷ forloop_iter

    t_f64 = time_call(
        () -> TeneT.FLmap_parallel(FL, ALu, ALd, M; ifparallel=false,
                                    forloop_iter, inner_etype=nothing),
        N)
    t_f32 = time_call(
        () -> TeneT.FLmap_parallel(FL, ALu, ALd, M; ifparallel=false,
                                    forloop_iter, inner_etype=Float32),
        N)

    m64, m32 = m(t_f64), m(t_f32)
    delta_pct = 100 * (m32/m64 - 1)
    label = m32 < m64 ? "faster" : "slower"
    @printf("%-8d | %-10.2f %-10.2f %-10.3f %+6.1f%% %s   chi/n=%d\n",
            forloop_iter, m64, m32, m32/m64, delta_pct, label, chi_chunk)
end

println()
