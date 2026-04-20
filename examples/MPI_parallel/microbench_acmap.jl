# MPI micro-benchmark: ACmap_parallel(ifparallel=true) at production config.
# Same methodology as microbench_mpi.jl but for ACmap instead of FLmap.
# Isolates whether ACmap has the same F32 speedup as FLmap.

using CUDA, MPI, TeneT, Statistics, Printf

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nprocs = MPI.Comm_size(comm)
CUDA.device!(0)

D = parse(Int, get(ENV, "D", "10"))
chi = parse(Int, get(ENV, "CHI", "400"))
forloop_iter = parse(Int, get(ENV, "FORLOOP_ITER", "32"))
N = parse(Int, get(ENV, "N", "20"))

rank == 0 && println("=== ACmap_parallel(ifparallel=true) MPI micro-benchmark ===")
rank == 0 && println("nprocs=$nprocs D=$D chi=$chi forloop_iter=$forloop_iter N=$N")
rank == 0 && println("GPU: ", CUDA.name(CUDA.device()))

D_phys = 2
AC = CUDA.rand(ComplexF64, chi, D, D, chi)
FL = CUDA.rand(ComplexF64, chi, D, D, chi)
FR = CUDA.rand(ComplexF64, chi, D, D, chi)
M1 = CUDA.rand(ComplexF64, D, D, D, D, D_phys)
M2 = CUDA.rand(ComplexF64, D, D, D, D, D_phys)
M = (M1, M2)

CUDA.synchronize()
MPI.Barrier(comm)

rank == 0 && println("Warmup...")
for prec in [nothing, Float32]
    TeneT.ACmap_parallel(AC, FL, FR, M; ifparallel=true, forloop_iter, inner_etype=prec)
    CUDA.synchronize()
    MPI.Barrier(comm)
end

function time_loop(prec, N, comm)
    ts = Float64[]
    for _ in 1:N
        CUDA.synchronize()
        MPI.Barrier(comm)
        t0 = time_ns()
        TeneT.ACmap_parallel(AC, FL, FR, M; ifparallel=true, forloop_iter, inner_etype=prec)
        CUDA.synchronize()
        MPI.Barrier(comm)
        push!(ts, (time_ns() - t0) / 1e6)
    end
    return ts
end

t_f64 = time_loop(nothing, N, comm)
t_f32 = time_loop(Float32, N, comm)

MPI.Barrier(comm)
CUDA.synchronize()

if rank == 0
    m(v) = median(length(v) > 5 ? v[5:end] : v)
    m64, m32 = m(t_f64), m(t_f32)
    @printf("\nPer ACmap_parallel call (median of %d, skip first 4):\n", N)
    @printf("  F64 (inner_etype=nothing): %7.2f ms\n", m64)
    @printf("  F32 (inner_etype=Float32): %7.2f ms   ratio F32/F64: %.3f (%+.1f%%)\n",
            m32, m32/m64, 100*(m32/m64 - 1))
    @printf("\nReference FLmap_parallel (same config, from job 380627):\n")
    @printf("  FLmap F64: 210.26 ms, F32: 163.08 ms, ratio 0.776 (F32 22%% faster)\n")
    @printf("\nACmap vs FLmap:\n")
    @printf("  F64: %.2fx FLmap (AC relative cost)\n", m64 / 210.26)
    @printf("  F32: %.2fx FLmap\n", m32 / 163.08)
    delta_kernel = m32 - m64
    expected_saving = m64 * (1 - 0.776)
    @printf("\nIf ACmap had FLmap's 22%% F32 speedup, F32 should be %.2f ms.\n", m64 * 0.776)
    @printf("Missing savings: %.2f ms per ACmap call.\n", m32 - m64 * 0.776)
end

MPI.Finalize()
