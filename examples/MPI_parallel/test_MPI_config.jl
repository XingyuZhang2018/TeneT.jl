# MPI Multi-GPU Configuration Test
#
# Quick validation that MPI + CUDA + TeneT parallel setup is correct.
# Run with: mpirun -np $N bash -c 'export CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK; exec julia --project=../.. test_MPI_config.jl'
#
# Two test sections:
#   Part 1: MPI collective correctness & performance (allgatherv_p2p!, allreduce_p2p!)
#   Part 2: FLmap_parallel forward & backward correctness & performance

using CUDA, MPI, LinearAlgebra, Zygote, Printf, Random, Statistics, TeneT

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nprocs = MPI.Comm_size(comm)
CUDA.device!(0)

rank == 0 && println("=" ^ 70)
rank == 0 && println("MPI Multi-GPU Configuration Test")
rank == 0 && println("=" ^ 70)
rank == 0 && println("nprocs=$nprocs  device=$(CUDA.device())  hostname=$(gethostname())")
rank == 0 && println("CUDA_VISIBLE_DEVICES=$(get(ENV, "CUDA_VISIBLE_DEVICES", "<unset>"))")
rank == 0 && println("GPU: $(CUDA.name(CUDA.device()))  Memory: $(round(CUDA.total_memory()/1024^3, digits=1)) GiB")
rank == 0 && println()

all_pass = Ref(true)
function check(name, cond)
    if !cond
        all_pass[] = false
        rank == 0 && println("  FAIL: $name")
    end
    return cond
end

function timed_samples_ms(f, nrep, comm)
    samples = Float64[]
    for _ in 1:nrep
        MPI.Barrier(comm)
        local_sec = @elapsed begin
            f()
            CUDA.synchronize()
        end
        # Multi-rank wall time is the slowest rank for this repetition.
        max_sec = MPI.Allreduce(local_sec, max, comm)
        push!(samples, max_sec * 1000)
    end
    return samples
end

function fmt_samples(samples)
    return join((@sprintf("%.2f", x) for x in samples), ",")
end

function sample_stats(samples)
    return (best = minimum(samples), med = median(samples), avg = mean(samples))
end

# ═══════════════════════════════════════════════════════════════════════
# Part 1: MPI Collective Unit Tests
# ═══════════════════════════════════════════════════════════════════════
rank == 0 && println("─── Part 1: MPI Collectives ───")

for (label, N) in [("small 8KB", 1024), ("medium 8MB", 1_000_000), ("large 128MB", 16_000_000)]
    size_mb = N * 8 / 1024^2
    nrep = 10

    counts = TeneT.split_count(N, nprocs)

    # ── Allgatherv correctness (verify p2p result is consistent across runs) ──
    buf1 = CUDA.rand(Float64, N)
    buf2 = copy(buf1)
    TeneT.allgatherv_p2p!(buf1, Cint.(counts), comm); CUDA.synchronize()
    TeneT.allgatherv_p2p!(buf2, Cint.(counts), comm); CUDA.synchronize()
    ag_ok = isapprox(Array(buf1), Array(buf2))

    # ── Allgatherv timing ──
    buf = CUDA.rand(Float64, N)
    TeneT.allgatherv_p2p!(buf, Cint.(counts), comm); CUDA.synchronize(); MPI.Barrier(comm)
    ag_ms = timed_samples_ms(() -> TeneT.allgatherv_p2p!(buf, Cint.(counts), comm), nrep, comm)
    ag = sample_stats(ag_ms)

    # ── Allreduce correctness (verify sum equals nprocs) ──
    buf_ar = CUDA.ones(Float64, N)
    TeneT.allreduce_p2p!(buf_ar, +, comm); CUDA.synchronize()
    ar_ok = isapprox(Array(buf_ar)[1], Float64(nprocs))

    # ── Allreduce timing ──
    buf = CUDA.rand(Float64, N)
    TeneT.allreduce_p2p!(buf, +, comm); CUDA.synchronize(); MPI.Barrier(comm)
    ar_ms = timed_samples_ms(() -> TeneT.allreduce_p2p!(buf, +, comm), nrep, comm)
    ar = sample_stats(ar_ms)

    check("Allgatherv $label", ag_ok)
    check("Allreduce $label", ar_ok)
    bw_ag = size_mb / ag.best  # GB/s, using ms
    bw_ar = size_mb / ar.best  # GB/s, using ms
    rank == 0 && @printf("  AG_SAMPLES %-14s ms=%s\n", label, fmt_samples(ag_ms))
    rank == 0 && @printf("  AR_SAMPLES %-14s ms=%s\n", label, fmt_samples(ar_ms))
    rank == 0 && @printf("  %-14s  Allgatherv: %7.2fms (%5.1f GB/s) %s   Allreduce: %7.2fms (%5.1f GB/s) %s   avg/med ag: %.2f/%.2fms ar: %.2f/%.2fms\n",
        label, ag.best, bw_ag, ag_ok ? "✓" : "✗", ar.best, bw_ar, ar_ok ? "✓" : "✗",
        ag.avg, ag.med, ar.avg, ar.med)
end
rank == 0 && println()

# ═══════════════════════════════════════════════════════════════════════
# Part 2: FLmap_parallel Forward & Backward Test
# ═══════════════════════════════════════════════════════════════════════
rank == 0 && println("─── Part 2: FLmap_parallel Forward & Backward ───")

for (D, χ) in vec([(D, χ) for χ in 256:256:1024, D in 8:2:16])
    total_splits = 128
    forloop_iter = total_splits ÷ nprocs
    d = 2  # physical dimension

    # leg5 tensors: FL(χ,D,D,χ), AL(χ,D,D,χ), M(D,D,D,D,d)
    FL  = CUDA.rand(Float64, χ, D, D, χ)
    ALu = CUDA.rand(Float64, χ, D, D, χ)
    ALd = CUDA.rand(Float64, χ, D, D, χ)
    M   = CUDA.rand(Float64, D, D, D, D, d)

    tensor_mb = χ^2 * D^2 * 8 / 1024^2

    # ── Forward correctness: parallel vs serial ──
    result_par = TeneT.FLmap_parallel(FL, ALu, ALd, M; ifparallel=true, forloop_iter)
    CUDA.synchronize()
    result_seq = TeneT.FLmap_parallel(FL, ALu, ALd, M; ifparallel=false, forloop_iter=total_splits)
    CUDA.synchronize()
    fwd_ok = isapprox(Array(result_par), Array(result_seq); rtol=1e-4)
    check("FLmap forward D=$D χ=$χ", fwd_ok)

    # ── Forward timing ──
    CUDA.synchronize(); MPI.Barrier(comm)
    nrep = 3
    fwd_ms = timed_samples_ms(() -> TeneT.FLmap_parallel(FL, ALu, ALd, M; ifparallel=true, forloop_iter), nrep, comm)
    fwd = sample_stats(fwd_ms)

    # ── Backward correctness & timing ──
    bwd_ok = true
    bwd_ms = Float64[]
    try
        # warmup
        _, bp = Zygote.pullback(x -> sum(TeneT.FLmap_parallel(x, ALu, ALd, M; ifparallel=true, forloop_iter)), FL)
        g = bp(one(eltype(FL)))[1]
        CUDA.synchronize(); MPI.Barrier(comm)
        GC.gc(); CUDA.reclaim()

        bwd_ms = timed_samples_ms(() -> begin
            _, bp = Zygote.pullback(x -> sum(TeneT.FLmap_parallel(x, ALu, ALd, M; ifparallel=true, forloop_iter)), FL)
            g = bp(one(eltype(FL)))[1]
        end, nrep, comm)

        # Check gradient is finite and non-zero
        g_arr = Array(g)
        bwd_ok = all(isfinite, g_arr) && norm(g_arr) > 0
    catch e
        rank == 0 && println("  WARN: backward failed for D=$D χ=$χ: ", sprint(showerror, e))
        bwd_ok = false
    end
    check("FLmap backward D=$D χ=$χ", bwd_ok)
    bwd = isempty(bwd_ms) ? (best = NaN, med = NaN, avg = NaN) : sample_stats(bwd_ms)

    rank == 0 && @printf("  FWD_SAMPLES D=%d χ=%d ms=%s\n", D, χ, fmt_samples(fwd_ms))
    rank == 0 && @printf("  BWD_SAMPLES D=%d χ=%d ms=%s\n", D, χ, fmt_samples(bwd_ms))
    rank == 0 && @printf("  D=%-2d χ=%-4d (%6.1fMB)  fwd: %8.2fms %s   bwd: %8.2fms %s   bwd/fwd: %.1fx   avg/med fwd: %.2f/%.2fms bwd: %.2f/%.2fms\n",
        D, χ, tensor_mb, fwd.best, fwd_ok ? "✓" : "✗", bwd.best, bwd_ok ? "✓" : "✗",
        isnan(bwd.best) ? NaN : bwd.best / fwd.best,
        fwd.avg, fwd.med, bwd.avg, bwd.med)

    GC.gc(); CUDA.reclaim()
end

rank == 0 && println()
rank == 0 && println("=" ^ 70)
if all_pass[]
    rank == 0 && println("ALL TESTS PASSED ✓")
else
    rank == 0 && println("SOME TESTS FAILED ✗")
end
rank == 0 && println("=" ^ 70)

# Clean up GPU resources before MPI finalize to avoid UCX/CUDA teardown crashes
GC.gc(); CUDA.reclaim()
MPI.Barrier(comm)
MPI.Finalize()
