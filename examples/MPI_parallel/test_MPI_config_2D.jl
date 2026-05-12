# MPI Multi-GPU Configuration Test — 2D extension
#
# Mirrors examples/MPI_parallel/test_MPI_config.jl EXACTLY for the FLmap
# Part 2 methodology (CUDA.rand per-rank tensors, single MPI.Barrier before
# the timing loop, CUDA.synchronize inside each iter, divide elapsed by nrep,
# GC+reclaim between configs). Adds a third measurement: FLmap_parallel_2D.
#
# Run with the same submit env as Sofia/submit_test.sh / submit_D16_*_sofia.sh:
#   mpirun -np $N bash -c 'export CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK; \
#                          export TENET_USE_NCCL=1; export TENET_NCCL_REGISTER=1; \
#                          ... ; exec julia --project=$WD/TeneT-2d-validation test_MPI_config_2D.jl'
#
# Goal: apples-to-apples comparison between 1D FLmap_parallel and 2D
# FLmap_parallel_2D under the proven production benchmark setup.

using CUDA, MPI, LinearAlgebra, Printf, Random, TeneT

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nprocs = MPI.Comm_size(comm)
CUDA.device!(0)

# 2D grid setup — pick N1 = floor(sqrt(N)), N2 = N/N1. For N=4 → (2,2);
# N=8 → (2,4); N=16 → (4,4).
N1 = max(1, Int(floor(sqrt(nprocs))))
while nprocs % N1 != 0
    N1 -= 1
end
N2 = nprocs ÷ N1
@assert N1 * N2 == nprocs
grid = Cart2DGrid(N1, N2)

rank == 0 && println("=" ^ 70)
rank == 0 && println("MPI 2D FLmap Bench — mirrors test_MPI_config.jl methodology")
rank == 0 && println("=" ^ 70)
rank == 0 && println("nprocs=$nprocs (N1=$N1 × N2=$N2)  device=$(CUDA.device())  hostname=$(gethostname())")
rank == 0 && println("CUDA_VISIBLE_DEVICES=$(get(ENV, "CUDA_VISIBLE_DEVICES", "<unset>"))")
rank == 0 && println("TENET_USE_NCCL=$(get(ENV, "TENET_USE_NCCL", "<unset>"))")
rank == 0 && println("GPU: $(CUDA.name(CUDA.device()))  Memory: $(round(CUDA.total_memory()/1024^3, digits=1)) GiB")
rank == 0 && println()
rank == 0 && @printf("%-30s %12s %12s %12s %10s\n",
    "Config", "1D fwd (ms)", "2D fwd (ms)", "ratio 2D/1D", "speedup")
rank == 0 && println("─" ^ 80)

# Limit to D=10 only (the production OOM regime from 1.25 Kagome 1003757).
# χ ∈ {512, 768, 1024} covers the bench range.
for (D, χ) in [(10, 512), (10, 768), (10, 1024)]
    total_splits = 128
    forloop_iter = total_splits ÷ nprocs
    d = 2

    # ─── 1D inputs: full tensors per rank (mirror test_MPI_config.jl exactly) ───
    FL  = CUDA.rand(Float64, χ, D, D, χ)
    ALu = CUDA.rand(Float64, χ, D, D, χ)
    ALd = CUDA.rand(Float64, χ, D, D, χ)
    M   = CUDA.rand(Float64, D, D, D, D, d)
    tensor_mb = χ^2 * D^2 * 8 / 1024^2

    # ─── 2D inputs: per-rank slices of shape (χ/N1, D, D, χ/N2) ───
    chi_per_N1 = χ ÷ N1
    chi_per_N2 = χ ÷ N2
    FL_2d  = CUDA.rand(Float64, chi_per_N1, D, D, chi_per_N2)
    ALu_2d = CUDA.rand(Float64, chi_per_N1, D, D, chi_per_N2)
    ALd_2d = CUDA.rand(Float64, chi_per_N1, D, D, chi_per_N2)

    # ─── 1D forward timing (mirror methodology) ───
    # Warmup: one call before timing.
    _ = TeneT.FLmap_parallel(FL, ALu, ALd, M; ifparallel=true, forloop_iter)
    CUDA.synchronize(); MPI.Barrier(comm)
    nrep = 3
    t_1d = @elapsed for _ in 1:nrep
        TeneT.FLmap_parallel(FL, ALu, ALd, M; ifparallel=true, forloop_iter)
        CUDA.synchronize()
    end
    t_1d /= nrep

    GC.gc(); CUDA.reclaim()

    # ─── 2D forward timing (same loop structure) ───
    _ = TeneT.FLmap_parallel_2D(FL_2d, ALu_2d, ALd_2d, M; grid)
    CUDA.synchronize(); MPI.Barrier(comm)
    t_2d = @elapsed for _ in 1:nrep
        TeneT.FLmap_parallel_2D(FL_2d, ALu_2d, ALd_2d, M; grid)
        CUDA.synchronize()
    end
    t_2d /= nrep

    rank == 0 && @printf("D=%-2d χ=%-4d (%6.1fMB)  %12.2f %12.2f %12.3fx %10s\n",
        D, χ, tensor_mb, t_1d*1000, t_2d*1000, t_2d/t_1d,
        t_2d < t_1d ? @sprintf("%.0f%%", 100*(1 - t_2d/t_1d)) : "—")
    flush(stdout)

    GC.gc(); CUDA.reclaim()
end

rank == 0 && println()
rank == 0 && println("=" ^ 70)
rank == 0 && println("Done")
rank == 0 && println("=" ^ 70)

GC.gc(); CUDA.reclaim()
MPI.Barrier(comm)
MPI.Finalize()
