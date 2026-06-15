# Cannon MAP timing: FLmap / FRmap / ACmap / ACdmap distributed, over the
# Part-2 D×χ matrix. Answers "are the gather-class maps (FRmap/ACdmap, 2-level
# i/d chunk → ≈P·n smaller GEMMs) slower than the ring-class maps (FLmap/ACmap,
# single l-chunk → n GEMMs)?". Same methodology as benchmark_cannon_sofia.jl:
# Float64 leg5 single-M, nrep=3, cannon forloop_iter = pick_n(D,χ) per cell
# (the backward-peak formula — SAME n for all four maps, so the per-chunk peak is
# equal and the timing gap is purely chunk-count/granularity, the thing asked).
#
# RING ONLY: the cannon collectives (row/col allgather, reduce-scatter) are MPI
# point-to-point with no NCCL path (Part 5 fn.2 / Part 10), so ring IS the honest
# cannon number. The slice path is run UNTIMED, only as the parity reference
# (chunked *_parallel → no serial-ref OOM, unlike Part 9's validator).
#
# Output: per map, cannon fwd / cannon bwd (ms, ring) + block parity vs slice.
# Launched by Sofia/submit_benchmark_cannon_maps.sh (np 4) / _16gpu.sh (np 16).
using CUDA, MPI, LinearAlgebra, Zygote, Printf, TeneT
using TeneT: cannon_grid, cannon_scatter, split_ranges,
             FLmap_cannon_dist, FRmap_cannon_dist, ACmap_cannon_dist, ACdmap_cannon_dist

MPI.Init()
const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
const nprocs = MPI.Comm_size(comm)
CUDA.device!(0)

const N1 = parse(Int, get(ENV, "TENET_CANNON_N1", "2"))
const N2 = parse(Int, get(ENV, "TENET_CANNON_N2", "2"))
@assert nprocs == N1 * N2 "nprocs=$nprocs ≠ N1×N2=$(N1 * N2)"
@assert N1 == N2 "M3 maps require a square grid (N1==N2)"
const g = cannon_grid(N1, N2)
const total_splits = 128
const forloop_slice = total_splits ÷ nprocs           # slice path (parity ref) chunking
@assert total_splits % nprocs == 0
const nrep = 3
const d_phys = 2
const MEM_BUDGET = 105e9

report(s...) = rank == 0 && println(s...)

H_bytes(D, χ) = χ^2 * D^4 / (N1 * N2) * 8
tensor_bytes(D, χ) = χ^2 * D^2 * 8
const BWD_COEFF = 4 + 2 * d_phys
# 16× (vs FLmap-only 12×): FR/ACd hold 3 gathered cross-axis slices + grads.
resident_bytes(D, χ) = 16 * tensor_bytes(D, χ)
function pick_n(D, χ)
    avail = MEM_BUDGET - resident_bytes(D, χ)
    @assert avail > 0
    return max(1, ceil(Int, BWD_COEFF * H_bytes(D, χ) / avail))
end

function timeit(f)
    f(); CUDA.synchronize(); MPI.Barrier(comm)        # warm (builds plans/comms)
    GC.gc(); CUDA.reclaim()
    tot = 0.0
    for _ in 1:nrep
        MPI.Barrier(comm)
        t = MPI.Wtime()
        f(); CUDA.synchronize()
        tot += MPI.Wtime() - t
        GC.gc()                                        # finalize dead tapes outside timed window
    end
    return MPI.Allreduce(tot / nrep, MPI.MAX, comm) * 1000   # ms, max over ranks
end

fmt(x) = isnan(x) ? @sprintf("%10s", "skip") : @sprintf("%10.1f", x)

ENV["TENET_USE_NCCL"] = "0"   # ring only

report("=== Cannon $(N1)x$(N2) MAP timing: FL/FR/AC/ACd ($(nprocs) GPU, ", CUDA.name(CUDA.device()), ") — RING vs NCCL ===")
report("methodology: benchmark_cannon_sofia.jl (Float64 leg5 single-M, nrep=$nrep, cannon forloop_iter=n); slice untimed=parity ref; parity recomputed under NCCL (validates the new path)")
report("| map    | D  | χ    | n  | fwd ring | fwd nccl | bwd ring | bwd nccl | parity |")
report("|--------|----|------|----|----------|----------|----------|----------|--------|")

# (name, slice_parallel, cannon_dist) — arg order is identical for the pair.
const MAPS = (
    ("FLmap",  (a, b, c, M) -> TeneT.FLmap_parallel(a, b, c, M; ifparallel = true, forloop_iter = forloop_slice),
               (a, b, c, M, n) -> FLmap_cannon_dist(a, b, c, M, g; forloop_iter = n)),
    ("FRmap",  (a, b, c, M) -> TeneT.FRmap_parallel(a, b, c, M; ifparallel = true, forloop_iter = forloop_slice),
               (a, b, c, M, n) -> FRmap_cannon_dist(a, b, c, M, g; forloop_iter = n)),
    ("ACmap",  (a, b, c, M) -> TeneT.ACmap_parallel(a, b, c, M; ifparallel = true, forloop_iter = forloop_slice),
               (a, b, c, M, n) -> ACmap_cannon_dist(a, b, c, M, g; forloop_iter = n)),
    ("ACdmap", (a, b, c, M) -> TeneT.ACdmap_parallel(a, b, c, M; ifparallel = true, forloop_iter = forloop_slice),
               (a, b, c, M, n) -> ACdmap_cannon_dist(a, b, c, M, g; forloop_iter = n)),
)

for (D, χ) in vec([(D, χ) for χ in 256:256:1024, D in 8:2:16])
    n = pick_n(D, χ)
    a_rs = split_ranges(χ, N1); i_rs = split_ranges(χ, N2)
    CUDA.seed!(20260615)
    M = CUDA.rand(Float64, D, D, D, D, d_phys); MPI.Bcast!(M, 0, comm)

    for (name, slicef, cannonf) in MAPS
        CUDA.seed!(42)
        T1 = CUDA.rand(Float64, χ, D, D, χ)
        T2 = CUDA.rand(Float64, χ, D, D, χ)
        T3 = CUDA.rand(Float64, χ, D, D, χ)
        for t in (T1, T2, T3); MPI.Bcast!(t, 0, comm); end
        b1 = cannon_scatter(T1, g); b2 = cannon_scatter(T2, g); b3 = cannon_scatter(T3, g)

        cannon_fwd_f = () -> cannonf(b1, b2, b3, M, n)
        cannon_bwd_f = () -> Zygote.pullback(x -> sum(cannonf(x, b2, b3, M, n)), b1)[2](1.0)

        tt = Dict{String, Float64}()
        for (tag, on) in (("ring", "0"), ("nccl", "1"))
            ENV["TENET_USE_NCCL"] = on
            tt["f_$tag"] = timeit(cannon_fwd_f)
            tt["b_$tag"] = timeit(cannon_bwd_f)
        end

        # parity (UNTIMED) under NCCL — validates the NEW NCCL cannon path vs the
        # chunked *_parallel slice ref (ring already validated by test_cannon_m3 +
        # the M3.5 16-GPU run). No serial-ref OOM (slice is forloop-chunked).
        ENV["TENET_USE_NCCL"] = "1"
        ref = slicef(T1, T2, T3, M)[a_rs[g.r1 + 1], :, :, i_rs[g.r2 + 1]]
        ef  = norm(cannon_fwd_f() - ref) / norm(ref); ref = nothing
        pf  = MPI.Allreduce(ef, MPI.MAX, comm) < 1e-10 ? "F✓" : "F✗"
        gsl = Zygote.pullback(x -> sum(slicef(x, T2, T3, M)), T1)[2](1.0)[1][a_rs[g.r1 + 1], :, :, i_rs[g.r2 + 1]]
        eb  = norm(cannon_bwd_f()[1] - gsl) / norm(gsl); gsl = nothing
        pb  = MPI.Allreduce(eb, MPI.MAX, comm) < 1e-8 ? "B✓" : "B✗"

        ENV["TENET_USE_NCCL"] = "0"
        rank == 0 && @printf("| %-6s | %-2d | %-4d | %-2d | %s | %s | %s | %s | %s %s |\n",
            name, D, χ, n, fmt(tt["f_ring"]), fmt(tt["f_nccl"]), fmt(tt["b_ring"]), fmt(tt["b_nccl"]), pf, pb)
        flush(stdout)

        T1 = T2 = T3 = b1 = b2 = b3 = nothing
        GC.gc(); CUDA.reclaim()
    end
    M = nothing; GC.gc(); CUDA.reclaim()
end

report("=== done ===")
GC.gc(); CUDA.reclaim()
MPI.Barrier(comm); MPI.Finalize()
