# MPI micro-benchmark: FLmap_parallel(ifparallel=true) at production config.
# Runs 4 MPI ranks, D=10 chi=400 forloop_iter=32, matching production 4-GPU setup.
# Times repeated FLmap_parallel calls (simulates simple_eig power iterations)
# in F64 vs F32, isolating parallel() overhead from the VUMPS outer loop.
#
# Run with: srun -n 4 julia --project=../.. microbench_mpi.jl

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

rank == 0 && println("=== FLmap_parallel(ifparallel=true) MPI micro-benchmark ===")
rank == 0 && println("nprocs=$nprocs D=$D chi=$chi forloop_iter=$forloop_iter N=$N")
rank == 0 && println("GPU: ", CUDA.name(CUDA.device()))

D_phys = 2
Random_seed = 42
# All ranks use same random state so inputs match (deterministic)
import Random
Random.seed!(Random_seed)
# Actually for MPI we need each rank to have its own CUDA array but matching shape;
# contents don't matter for timing. Just use CUDA.rand.
FL  = CUDA.rand(ComplexF64, chi, D, D, chi)
ALu = CUDA.rand(ComplexF64, chi, D, D, chi)
ALd = CUDA.rand(ComplexF64, chi, D, D, chi)
M1  = CUDA.rand(ComplexF64, D, D, D, D, D_phys)
M2  = CUDA.rand(ComplexF64, D, D, D, D, D_phys)
M = (M1, M2)

CUDA.synchronize()
MPI.Barrier(comm)

# ────── Warmup ──────
rank == 0 && println("Warmup...")
for prec in [nothing, Float32]
    TeneT.FLmap_parallel(FL, ALu, ALd, M; ifparallel=true, forloop_iter, inner_etype=prec)
    CUDA.synchronize()
    MPI.Barrier(comm)
end

# ────── Timing ──────
function time_loop(prec, N, comm)
    ts = Float64[]
    for _ in 1:N
        CUDA.synchronize()
        MPI.Barrier(comm)
        t0 = time_ns()
        TeneT.FLmap_parallel(FL, ALu, ALd, M; ifparallel=true, forloop_iter, inner_etype=prec)
        CUDA.synchronize()
        MPI.Barrier(comm)
        push!(ts, (time_ns() - t0) / 1e6)
    end
    return ts
end

t_f64 = time_loop(nothing, N, comm)
t_f32 = time_loop(Float32, N, comm)

# ────── Decomposition: just cast + allgather, no @tensor ──────
# How much does the boundary cast + allgather portion alone cost?
# (simulate: cast inputs, allocate F32 result of right size, filled with 1.0,
#  run allgatherv, cast result back)
function time_cast_plus_gather(N, comm)
    ts = Float64[]
    # The result buffer that FLmap_parallel would produce: (chi, D, D, chi)
    size_out = (chi, D, D, chi)
    for _ in 1:N
        CUDA.synchronize()
        MPI.Barrier(comm)
        t0 = time_ns()
        # cast 5 inputs
        FL_c  = ComplexF32.(FL)
        ALu_c = ComplexF32.(ALu)
        ALd_c = ComplexF32.(ALd)
        M1_c  = ComplexF32.(M1)
        M2_c  = ComplexF32.(M2)
        # allocate F32 result
        result_c = similar(FL_c, size_out)
        fill!(result_c, 1.0f0)
        # mimic allgatherv on chi's last dim (result is chi×D×D×chi; split last dim)
        D_split_ranges = TeneT.split_ranges(chi, nprocs*forloop_iter)
        element_size = prod(size_out) ÷ chi
        counts = Cint[sum([length(D_split_ranges[(i-1)*forloop_iter+j]) for j in 1:forloop_iter]) * element_size for i in 1:nprocs]
        TeneT.allgatherv_p2p!(result_c, counts, comm)
        # cast back
        result = ComplexF64.(result_c)
        CUDA.synchronize()
        MPI.Barrier(comm)
        push!(ts, (time_ns() - t0) / 1e6)
    end
    return ts
end

t_cast = time_cast_plus_gather(N, comm)

MPI.Barrier(comm)
CUDA.synchronize()

if rank == 0
    m(v) = median(length(v) > 5 ? v[5:end] : v)
    m64, m32, mc = m(t_f64), m(t_f32), m(t_cast)
    @printf("\nPer FLmap_parallel call (median of %d, skip first 4):\n", N)
    @printf("  F64 (inner_etype=nothing): %7.2f ms\n", m64)
    @printf("  F32 (inner_etype=Float32): %7.2f ms   ratio F32/F64: %.3f (%+.1f%%)\n",
            m32, m32/m64, 100*(m32/m64 - 1))
    @printf("  Cast+allgatherv only     : %7.2f ms   = %.1f%% of F64 time\n",
            mc, 100*mc/m64)

    # What if we subtract cast+allgather overhead from F32? Estimate pure-compute
    # F32 @tensor time would be ~ m32 - mc (rough, since F64 also has allgather)
    # Actually both have allgather; only the cast is F32-specific.
    # Rough cast-only estimate: cast ~ 40-50% of (cast+allgather) in production
    @printf("\nIf cast overhead alone were ~50%% of (cast+allgather): %.2f ms saved by removing cast\n",
            mc * 0.5)
    @printf("Implied F32 kernel+gather (no cast): %.2f ms vs F64 %.2f ms\n",
            m32 - mc*0.5, m64)
end

MPI.Finalize()
