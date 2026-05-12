# timing/bench_flmap_2d_vs_1d.jl
#
# Benchmark `FLmap_parallel_2D` (new, 2D-distributed) vs the existing 1D
# `FLmap_parallel` (full tensors, allgatherv at exit). Runs under MPI with
# N = N1 × N2 ranks.
#
# Usage:
#   mpirun -n 4 julia --project=. timing/bench_flmap_2d_vs_1d.jl
#
# Env knobs (all optional; reasonable defaults):
#   BENCH_D=4         physical-leg bond dimension (D in code)
#   BENCH_CHI=64      boundary bond dimension (must be divisible by N1, N2, and N for the 1D split)
#   BENCH_N1=2        grid dim along axis 1 (N2 = N / N1 is implied)
#   BENCH_REPEATS=5   number of timed runs per algorithm
#   BENCH_WARMUP=2    untimed warmup runs (lets JIT + MPI handshake settle)
#   BENCH_USE_GPU=0   set 1 to allocate CuArrays (requires CUDA-aware MPI)
#   BENCH_SEED=42     RNG seed used by rank 0 before bcast
#
# Output is printed by rank 0 only. Per-rank peak VRAM is reported via
# `CUDA.memory_status()`-style introspection when BENCH_USE_GPU=1.

using MPI
using Random
using Test
using Statistics
using LinearAlgebra
using Printf
using TeneT
using TeneT: FLmap_parallel

MPI.Initialized() || MPI.Init()
const world = MPI.COMM_WORLD
const N     = MPI.Comm_size(world)
const rank  = MPI.Comm_rank(world)

const USE_GPU = parse(Int, get(ENV, "BENCH_USE_GPU", "0")) == 1
if USE_GPU
    using CUDA
    # Production Sofia submit scripts set CUDA_VISIBLE_DEVICES per-rank to
    # OMPI_COMM_WORLD_LOCAL_RANK, so each rank sees exactly one GPU as device 0.
    # Don't call CUDA.device!(local_rank) — only device 0 is visible.
    # If CUDA_VISIBLE_DEVICES isn't set (debugging on a single-GPU host),
    # fall back to selecting by local rank for safety.
    if !haskey(ENV, "CUDA_VISIBLE_DEVICES")
        local_rank = parse(Int, get(ENV, "OMPI_COMM_WORLD_LOCAL_RANK", "$rank"))
        n_dev = length(CUDA.devices())
        if n_dev > 1
            CUDA.device!(local_rank % n_dev)
        end
    end
    # Otherwise: leave at default device (= the single visible GPU per rank)
end
const ATYPE = USE_GPU ? CuArray : Array

const D    = parse(Int, get(ENV, "BENCH_D",   "4"))
const CHI  = parse(Int, get(ENV, "BENCH_CHI", "64"))
# Default N1 = 2 when running under multi-rank MPI, else 1. The user can
# override via BENCH_N1 explicitly (e.g. BENCH_N1=4 for a 4x1 grid).
const N1   = parse(Int, get(ENV, "BENCH_N1",  (N == 1 ? "1" : "2")))
const N2   = N ÷ N1
const REPEATS = parse(Int, get(ENV, "BENCH_REPEATS", "5"))
const WARMUP  = parse(Int, get(ENV, "BENCH_WARMUP",  "2"))
const SEED    = parse(Int, get(ENV, "BENCH_SEED",    "42"))
# total_splits/nprocs to match production bench.jl methodology (default 128 splits).
# For nprocs=4 → forloop_iter=32. Production uses this same scheme so timings
# are comparable to examples/MPI_parallel/benchmarks/Sofia_VUB_H200.md.
const TOTAL_SPLITS = parse(Int, get(ENV, "BENCH_TOTAL_SPLITS", "128"))
const FORLOOP_ITER = max(1, TOTAL_SPLITS ÷ N)

@assert N == N1 * N2  "N1 ($N1) * N2 ($N2) must equal MPI world size ($N)"
@assert CHI % N1 == 0 "CHI ($CHI) must be divisible by N1 ($N1)"
@assert CHI % N2 == 0 "CHI ($CHI) must be divisible by N2 ($N2)"
@assert CHI % N  == 0 "CHI ($CHI) must be divisible by N=$N (1D split divides last dim by nprocs)"

# Use the 2D grid for the new map; 1D map uses MPI.COMM_WORLD internally.
const grid = Cart2DGrid(N1, N2)

# ─── Build test tensors ───────────────────────────────────────────────────
#
# Rank 0 generates random tensors with a fixed seed; broadcast to all ranks
# so each rank sees the SAME full tensors. Then carve out per-rank slices
# for the 2D map.
#
# Layout convention (per `docs/2026-05-12-flmap-2d-algorithm.md`-style):
#   FL_full[a (chi), e (D), f (D), i (chi)]
#   ALu_full[a (chi), b (D), c (D), d (chi)]
#   ALd_full[i (chi), j (D), k (D), l (chi)]
#   M    [e (D), j (D), g (D), b (D), p (2)]  (rank-5 / leg5)

function build_full_tensors(seed)
    rng = Random.MersenneTwister(seed)
    chi = CHI
    d = D
    p = 2  # physical leg width (Kagome/Plaquette double-layer uses p=2 for D=4 toy)
    FL  = randn(rng, Float64, chi, d, d, chi)
    ALu = randn(rng, Float64, chi, d, d, chi)
    ALd = randn(rng, Float64, chi, d, d, chi)
    M   = randn(rng, Float64, d, d, d, d, p)
    return FL, ALu, ALd, M
end

if rank == 0
    println("=" ^ 60)
    println("FLmap_parallel_2D vs FLmap_parallel (1D) benchmark")
    println("=" ^ 60)
    @printf("N=%d (N1=%d × N2=%d), D=%d, χ=%d, ATYPE=%s\n", N, N1, N2, D, CHI, ATYPE)
    @printf("eltype=Float64 (matches production bench.jl)\n")
    @printf("TOTAL_SPLITS=%d → FORLOOP_ITER=%d/rank (1D only; 2D doesn't use forloop)\n", TOTAL_SPLITS, FORLOOP_ITER)
    @printf("WARMUP=%d, REPEATS=%d, SEED=%d\n", WARMUP, REPEATS, SEED)
    flush(stdout)
end

# Build tensors on rank 0, then bcast.
local FL_full, ALu_full, ALd_full, M_full
if rank == 0
    FL_full, ALu_full, ALd_full, M_full = build_full_tensors(SEED)
else
    # Pre-allocate matching-size arrays for bcast.
    FL_full  = zeros(Float64, CHI, D, D, CHI)
    ALu_full = zeros(Float64, CHI, D, D, CHI)
    ALd_full = zeros(Float64, CHI, D, D, CHI)
    M_full   = zeros(Float64, D, D, D, D, 2)
end
MPI.Bcast!(FL_full,  0, world)
MPI.Bcast!(ALu_full, 0, world)
MPI.Bcast!(ALd_full, 0, world)
MPI.Bcast!(M_full,   0, world)

# Per-rank 2D slices: each rank owns (chi/N1, D, D, chi/N2). For FL, ALu:
# first χ along col_comm (slice_r1), last χ along row_comm (slice_r2). For
# ALd: first χ on col_comm (slice_r1), last χ on row_comm (slice_r2).
chi_per_N1 = CHI ÷ N1
chi_per_N2 = CHI ÷ N2
r1, r2 = grid.r1, grid.r2

slice_a = (r1 * chi_per_N1 + 1):((r1 + 1) * chi_per_N1)  # slice_r1 on first χ (a, i)
slice_l = (r2 * chi_per_N2 + 1):((r2 + 1) * chi_per_N2)  # slice_r2 on last  χ (i, d, l)

# 2D-distribution inputs. NB: the FLmap_parallel_2D contract expects:
#   FL[a∈slice_r1, e, f, i∈slice_r2]      → (chi/N1, D, D, chi/N2)
#   ALu[a∈slice_r1, b, c, d∈slice_r2]     → (chi/N1, D, D, chi/N2)
#   ALd[i∈slice_r1, j, k, l∈slice_r2]     → (chi/N1, D, D, chi/N2)
# But the implementation’s first AllGather along col_comm operates on dim=1
# of ALd to recover FULL i. So ALd's first dim must be the slice_r1 view of
# the global `i`-leg. We construct exactly that below.
FL_local  = ATYPE(FL_full[slice_a, :, :, slice_l])
ALu_local = ATYPE(ALu_full[slice_a, :, :, slice_l])
ALd_local = ATYPE(ALd_full[slice_a, :, :, slice_l])
M_local   = ATYPE(M_full)

# 1D-distribution inputs. FLmap_parallel(`ifparallel=true`, `forloop_iter=1`)
# splits the LAST dim of ALd across nprocs internally; callers pass the
# FULL tensors. So each rank passes the full FL/ALu/ALd/M.
FL_1d  = ATYPE(FL_full)
ALu_1d = ATYPE(ALu_full)
ALd_1d = ATYPE(ALd_full)
M_1d   = ATYPE(M_full)

# ─── Helper: timed runs with GPU sync ────────────────────────────────────

"Run `f()` `n` times, returning per-iteration wall times in seconds."
function timed_runs(f, n; sync=true)
    times = Vector{Float64}(undef, n)
    for i in 1:n
        MPI.Barrier(world)
        t0 = time()
        out = f()
        if sync && USE_GPU
            CUDA.synchronize()
        end
        MPI.Barrier(world)
        times[i] = time() - t0
    end
    return times
end

# ─── Warmup ──────────────────────────────────────────────────────────────

if rank == 0
    println("\n[warmup] $WARMUP runs each (untimed)")
    flush(stdout)
end
for _ in 1:WARMUP
    _ = FLmap_parallel(FL_1d, ALu_1d, ALd_1d, M_1d; ifparallel=true, forloop_iter=FORLOOP_ITER)
    _ = FLmap_parallel_2D(FL_local, ALu_local, ALd_local, M_local; grid)
end

# ─── Benchmark ───────────────────────────────────────────────────────────

if rank == 0
    println("[benchmark] 1D FLmap_parallel x $REPEATS …")
    flush(stdout)
end
times_1d = timed_runs(REPEATS) do
    FLmap_parallel(FL_1d, ALu_1d, ALd_1d, M_1d; ifparallel=true, forloop_iter=FORLOOP_ITER)
end

if rank == 0
    println("[benchmark] 2D FLmap_parallel_2D x $REPEATS …")
    flush(stdout)
end
times_2d = timed_runs(REPEATS) do
    FLmap_parallel_2D(FL_local, ALu_local, ALd_local, M_local; grid)
end

# ─── Correctness ─────────────────────────────────────────────────────────
#
# 1D output: full (χ, D, D, χ) on every rank.
# 2D output: (χ/N1, D, D, χ/N2) on each rank.
# To compare, AllGather the 2D output's first dim along col_comm (N1) and
# last dim along row_comm (N2). Then check elementwise against 1D output.

if rank == 0
    println("[correctness] running both maps once more and checking parity …")
    flush(stdout)
end
res_1d = FLmap_parallel(FL_1d, ALu_1d, ALd_1d, M_1d; ifparallel=true, forloop_iter=FORLOOP_ITER)
res_2d_local = FLmap_parallel_2D(FL_local, ALu_local, ALd_local, M_local; grid)

# Gather 2D back to full (χ, D, D, χ): first along col, then along row.
# IMPORTANT: use the direct variants because we are gathering over row_comm
# / col_comm (sub-communicators of world), and `allgather_dim` uses the
# unsafe-for-subcomm cache.
res_2d_full_d_only = allgather_dim_direct(res_2d_local, 1, grid.col_comm)
res_2d_full        = allgather_dim_direct(res_2d_full_d_only, ndims(res_2d_full_d_only), grid.row_comm)

# Move to host for comparison.
res_1d_h   = Array(res_1d)
res_2d_h   = Array(res_2d_full)
maxabs     = maximum(abs.(res_1d_h .- res_2d_h))
relerr     = norm(res_1d_h .- res_2d_h) / norm(res_1d_h)

# ─── Report on rank 0 ────────────────────────────────────────────────────

if rank == 0
    println("\n" * "=" ^ 60)
    println("Results (rank 0)")
    println("=" ^ 60)

    function describe(label, t)
        @printf("%-30s  median=%8.3f ms   min=%8.3f ms   mean=%8.3f ms\n",
                label, 1000*median(t), 1000*minimum(t), 1000*mean(t))
    end
    describe("1D  FLmap_parallel",     times_1d)
    describe("2D  FLmap_parallel_2D",  times_2d)
    @printf("\nratio (2D / 1D, median): %.2fx\n", median(times_2d) / median(times_1d))

    println("\nCorrectness")
    @printf("  max|res_1d - res_2d|  =  %.3e\n", maxabs)
    @printf("  rel L2 error           =  %.3e\n", relerr)
    if relerr < 1e-10
        println("  PASS (relerr < 1e-10)")
    elseif relerr < 1e-6
        println("  PASS (relerr < 1e-6 — Float32 noise tolerance)")
    else
        println("  FAIL — relerr above 1e-6, investigate.")
    end

    if USE_GPU
        println("\nGPU memory (rank 0)")
        # Best-effort: not all CUDA.jl versions surface peak alloc, so just
        # report current free/total.
        free, total = CUDA.Mem.info()
        @printf("  free  / total  =  %.2f / %.2f GB\n",
                free / 2^30, total / 2^30)
    end

    flush(stdout)
end

MPI.Barrier(world)
# Deliberately NOT calling MPI.Finalize() — bench script may be `include`d
# from a driver.
