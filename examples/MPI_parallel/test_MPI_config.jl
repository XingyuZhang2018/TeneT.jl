# MPI Multi-GPU Configuration Test
#
# Quick validation that MPI + CUDA + TeneT parallel setup is correct.
# Run with: mpirun -np $N bash -c 'export CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK; exec julia --project=../.. test_MPI_config.jl'
#
# Two test sections:
#   Part 1: MPI collective correctness & performance (allgatherv_p2p!, allreduce_p2p!)
#   Part 2: FLmap_parallel forward & backward correctness & performance

using MPI, CUDA, LinearAlgebra, Zygote, Printf, Random, TeneT

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
    t_ag = @elapsed for _ in 1:nrep; TeneT.allgatherv_p2p!(buf, Cint.(counts), comm); CUDA.synchronize(); end
    t_ag /= nrep

    # ── Allreduce correctness (verify sum equals nprocs) ──
    buf_ar = CUDA.ones(Float64, N)
    TeneT.allreduce_p2p!(buf_ar, +, comm); CUDA.synchronize()
    ar_ok = isapprox(Array(buf_ar)[1], Float64(nprocs))

    # ── Allreduce timing ──
    buf = CUDA.rand(Float64, N)
    TeneT.allreduce_p2p!(buf, +, comm); CUDA.synchronize(); MPI.Barrier(comm)
    t_ar = @elapsed for _ in 1:nrep; TeneT.allreduce_p2p!(buf, +, comm); CUDA.synchronize(); end
    t_ar /= nrep

    check("Allgatherv $label", ag_ok)
    check("Allreduce $label", ar_ok)
    bw_ag = size_mb / (t_ag * 1000)  # GB/s
    bw_ar = size_mb / (t_ar * 1000)  # GB/s
    rank == 0 && @printf("  %-14s  Allgatherv: %7.2fms (%5.1f GB/s) %s   Allreduce: %7.2fms (%5.1f GB/s) %s\n",
        label, t_ag*1000, bw_ag, ag_ok ? "✓" : "✗", t_ar*1000, bw_ar, ar_ok ? "✓" : "✗")
end
rank == 0 && println()

# ═══════════════════════════════════════════════════════════════════════
# Part 2: FLmap_parallel Forward & Backward Test
# ═══════════════════════════════════════════════════════════════════════
rank == 0 && println("─── Part 2: FLmap_parallel Forward & Backward ───")

for (D, χ) in [(8,256),(8,512),(8,1024), (10,256),(10,512),(10,1024), (12,256),(12,512),(12,1024)]
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
    t_fwd = @elapsed for _ in 1:nrep
        TeneT.FLmap_parallel(FL, ALu, ALd, M; ifparallel=true, forloop_iter)
        CUDA.synchronize()
    end
    t_fwd /= nrep

    # ── Backward correctness & timing ──
    bwd_ok = true
    t_bwd = 0.0
    try
        # warmup
        _, bp = Zygote.pullback(x -> sum(TeneT.FLmap_parallel(x, ALu, ALd, M; ifparallel=true, forloop_iter)), FL)
        g = bp(one(eltype(FL)))[1]
        CUDA.synchronize(); MPI.Barrier(comm)
        GC.gc(); CUDA.reclaim()

        t_bwd = @elapsed for _ in 1:nrep
            _, bp = Zygote.pullback(x -> sum(TeneT.FLmap_parallel(x, ALu, ALd, M; ifparallel=true, forloop_iter)), FL)
            g = bp(one(eltype(FL)))[1]
            CUDA.synchronize()
        end
        t_bwd /= nrep

        # Check gradient is finite and non-zero
        g_arr = Array(g)
        bwd_ok = all(isfinite, g_arr) && norm(g_arr) > 0
    catch e
        rank == 0 && println("  WARN: backward failed for D=$D χ=$χ: ", sprint(showerror, e))
        bwd_ok = false
        t_bwd = NaN
    end
    check("FLmap backward D=$D χ=$χ", bwd_ok)

    rank == 0 && @printf("  D=%-2d χ=%-4d (%6.1fMB)  fwd: %8.2fms %s   bwd: %8.2fms %s   bwd/fwd: %.1fx\n",
        D, χ, tensor_mb, t_fwd*1000, fwd_ok ? "✓" : "✗", t_bwd*1000, bwd_ok ? "✓" : "✗",
        isnan(t_bwd) ? NaN : t_bwd/t_fwd)

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
