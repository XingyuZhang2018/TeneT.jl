# Cannon (2×2) vs slice FLmap benchmark over the Part-2 D×χ matrix of
# benchmarks/Sofia_VUB_H200.md, same methodology as test_MPI_config.jl Part 2:
# Float64 leg5 tensors, single M, slice path with total_splits=128 sub-slicing,
# nrep=3, backward = Zygote pullback of a bare `sum` loss.
#
# TENET_USE_NCCL is read per-call by the collectives, so it is toggled
# in-job per section: one run yields ring AND NCCL columns for both paths.
# NCCL coverage note: slice fwd (allgatherv) + slice/cannon bwd (allreduce)
# take the NCCL fast path; the Cannon ring shifts and column reduce-scatter/
# allgather are MPI point-to-point (no NCCL path yet).
#
# The Cannon path sub-slices its local l range via `forloop_iter`: per cell
# the driver picks the smallest n whose BACKWARD per-chunk peak
# (4+2d)·|H|/n (pullback tape + adjoint chain) plus residents fits the
# budget, so every cell of the matrix runs — no skips.
#
# Output-placement asymmetry (deliberate, the honest map-level comparison):
# slice fwd INCLUDES the allgatherv that replicates the full result on every
# rank; cannon fwd ends with each rank holding only its block (an iterating
# map needs no gather since output distribution = input distribution).
# Likewise cannon bwd leaves dFL distributed while slice bwd allgathers it.
#
# Launched by Sofia/submit_benchmark_cannon.sh (mpirun -np 4).
using CUDA, MPI, LinearAlgebra, Zygote, Printf, TeneT
using TeneT: cannon_grid, cannon_scatter, FLmap_cannon, split_ranges

MPI.Init()
const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
const nprocs = MPI.Comm_size(comm)
@assert nprocs == 4
CUDA.device!(0)

const N1 = parse(Int, get(ENV, "TENET_CANNON_N1", "2"))
const N2 = parse(Int, get(ENV, "TENET_CANNON_N2", "2"))
const g = cannon_grid(N1, N2)
const total_splits = 128
const forloop_iter = total_splits ÷ nprocs
const nrep = 3
const d_phys = 2
const MEM_BUDGET = 110e9   # bytes; per-cell forloop_iter sizing budget

report(s...) = rank == 0 && println(s...)

H_bytes(D, χ) = χ^2 * D^4 / (N1 * N2) * 8
tensor_bytes(D, χ) = χ^2 * D^2 * 8
# backward per-chunk peak dominates: (4+2d)·|H|/n (bp2 tape + adjoint chain);
# residents ≈ input tensors + blocks row + partial/dpartial + full dALu/dALd.
const BWD_COEFF = 4 + 2 * d_phys
resident_bytes(D, χ) = 12 * tensor_bytes(D, χ)
function pick_n(D, χ)
    avail = MEM_BUDGET - resident_bytes(D, χ)
    @assert avail > 0
    return max(1, ceil(Int, BWD_COEFF * H_bytes(D, χ) / avail))
end

function timeit(f)
    f(); CUDA.synchronize(); MPI.Barrier(comm)   # warm (also builds NCCL comms on first use)
    GC.gc(); CUDA.reclaim()
    tot = 0.0
    for _ in 1:nrep
        MPI.Barrier(comm)
        t = MPI.Wtime()
        f(); CUDA.synchronize()
        tot += MPI.Wtime() - t
        # Per-rep cleanup OUTSIDE the timed window: dead pullback tapes
        # otherwise accumulate as live pool bytes across reps and OOM the
        # big-H cells (job 1265371 died at D=10 χ=768 with the pool at
        # 99.98% despite a fitting single-call peak).
        GC.gc(); CUDA.reclaim()
    end
    return MPI.Allreduce(tot / nrep, MPI.MAX, comm) * 1000   # ms, max over ranks
end

fmt(x) = isnan(x) ? @sprintf("%8s", "skip") : @sprintf("%8.1f", x)

report("=== Cannon $(N1)x$(N2) vs slice FLmap benchmark (4 GPU, ", CUDA.name(CUDA.device()), ") ===")
report("methodology: test_MPI_config.jl Part 2 (Float64 leg5, single M, total_splits=$total_splits, nrep=$nrep), cannon forloop_iter=n per cell (backward-peak formula)")
report("| D  | χ    | n  | sl fwd ring | sl fwd nccl | ca fwd ring | ca fwd nccl | sl bwd ring | sl bwd nccl | ca bwd ring | ca bwd nccl | parity |")
report("|----|------|----|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|--------|")

for (D, χ) in vec([(D, χ) for χ in 256:256:1024, D in 8:2:16])
    CUDA.seed!(42)   # deterministic per-process stream
    FL  = CUDA.rand(Float64, χ, D, D, χ)
    ALu = CUDA.rand(Float64, χ, D, D, χ)
    ALd = CUDA.rand(Float64, χ, D, D, χ)
    M   = CUDA.rand(Float64, D, D, D, D, d_phys)
    # Make cross-rank tensor identity unconditional (parity checks assume
    # replicated inputs); NVLink bcast is negligible vs the timed sections.
    for t in (FL, ALu, ALd, M)
        MPI.Bcast!(t, 0, comm)
    end
    blk = cannon_scatter(FL, g)

    n = pick_n(D, χ)

    slice_fwd_f  = () -> TeneT.FLmap_parallel(FL, ALu, ALd, M; ifparallel = true, forloop_iter)
    cannon_fwd_f = () -> FLmap_cannon(blk, ALu, ALd, M, g; forloop_iter = n)
    slice_bwd_f  = () -> Zygote.pullback(
        x -> sum(TeneT.FLmap_parallel(x, ALu, ALd, M; ifparallel = true, forloop_iter)), FL)[2](1.0)
    cannon_bwd_f = () -> Zygote.pullback(
        x -> sum(FLmap_cannon(x, ALu, ALd, M, g; forloop_iter = n)), blk)[2](1.0)

    t = Dict{String, Float64}()
    for (tag, on) in (("ring", "0"), ("nccl", "1"))
        ENV["TENET_USE_NCCL"] = on
        t["sf_$tag"] = timeit(slice_fwd_f)
        t["cf_$tag"] = timeit(cannon_fwd_f)
        t["sb_$tag"] = timeit(slice_bwd_f)
        t["cb_$tag"] = timeit(cannon_bwd_f)
        GC.gc(); CUDA.reclaim()
    end
    ENV["TENET_USE_NCCL"] = "0"

    # block-level parity (no gather needed): cannon block vs the matching
    # slice of the full slice-path result / dFL
    a_rs = split_ranges(χ, N1); i_rs = split_ranges(χ, N2)
    ref_blk = slice_fwd_f()[a_rs[g.r1 + 1], :, :, i_rs[g.r2 + 1]]
    err = norm(cannon_fwd_f() - ref_blk) / norm(ref_blk)
    p_fwd = MPI.Allreduce(err, MPI.MAX, comm) < 1e-10 ? "F✓" : "F✗"
    ref_blk = nothing
    g_sl = slice_bwd_f()[1][a_rs[g.r1 + 1], :, :, i_rs[g.r2 + 1]]
    err = norm(cannon_bwd_f()[1] - g_sl) / norm(g_sl)
    p_bwd = MPI.Allreduce(err, MPI.MAX, comm) < 1e-8 ? "B✓" : "B✗"
    g_sl = nothing

    rank == 0 && @printf("| %-2d | %-4d | %-2d | %s | %s | %s | %s | %s | %s | %s | %s | %s %s |\n",
        D, χ, n,
        fmt(t["sf_ring"]), fmt(t["sf_nccl"]), fmt(t["cf_ring"]), fmt(t["cf_nccl"]),
        fmt(t["sb_ring"]), fmt(t["sb_nccl"]), fmt(t["cb_ring"]), fmt(t["cb_nccl"]),
        p_fwd, p_bwd)
    flush(stdout)

    FL = ALu = ALd = M = blk = nothing
    GC.gc(); CUDA.reclaim()
end

report("=== done ===")
GC.gc(); CUDA.reclaim()
MPI.Barrier(comm)
MPI.Finalize()
