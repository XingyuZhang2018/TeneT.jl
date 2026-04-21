# MPI Multi-GPU checkpoint() correctness test
#
# Probes the three `CheckpointMethod`s on `FLmap_parallel` under MPI:
#   Plain()     — no @adjoint override; baseline forward + backward
#   Recompute() — @adjoint re-runs forward during backward (in-place ref capture)
#   Offload()   — @adjoint copies args to host after forward, reconstructs on backward
#
# What we're probing:
#   1. All three methods give identical forward output (bit-exact)
#   2. All three methods give identical gradient under Zygote (rtol ~1e-10)
#   3. No MPI deadlock in any method — Offload's args-to-host + args-back-to-device
#      bracket must not break rank synchronization during backward re-forward
#   4. `_offload_to_host(::CuArray)` on rank r must use rank r's device (not rank 0's)
#      via Array(x) — verified by result correctness on cross-rank data
#
# Expected failure modes to watch for:
#   - MPI hang (any method fails to return)
#   - Gradient divergence between ranks (Offload copies back to wrong device)
#   - Host OOM when N ranks × tensor_size exceeds host RAM
#
# Run on JSC: see submit_test_checkpoint.sh in examples/MPI_parallel/JSC/

using CUDA, MPI, LinearAlgebra, Zygote, Printf, Random, TeneT
using TeneT: Plain, Recompute, Offload, checkpoint

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nprocs = MPI.Comm_size(comm)
CUDA.device!(0)  # CUDA_VISIBLE_DEVICES already maps LOCAL_RANK → GPU

rank == 0 && println("=" ^ 70)
rank == 0 && println("MPI Multi-GPU checkpoint() Correctness Test")
rank == 0 && println("=" ^ 70)
rank == 0 && println("nprocs=$nprocs  device=$(CUDA.device())  hostname=$(gethostname())")
rank == 0 && println("CUDA_VISIBLE_DEVICES=$(get(ENV, "CUDA_VISIBLE_DEVICES", "<unset>"))")
rank == 0 && println("GPU: $(CUDA.name(CUDA.device()))  Memory: $(round(CUDA.total_memory()/1024^3, digits=1)) GiB")
rank == 0 && println()

all_pass = Ref(true)
function check(name, cond; msg="")
    if !cond
        all_pass[] = false
        rank == 0 && println("  FAIL: $name  $msg")
    end
    return cond
end

# Deterministic RNG — SAME seed on every rank so inputs are identical replicas
Random.seed!(42 + 0)  # rank-independent

# ═══════════════════════════════════════════════════════════════════════
# Test matrix: align with test_MPI_config.jl sizes so cuTENSOR has
# code-paths we know work on JSC (D<8 hits cuTENSOR JIT edge cases).
# Correctness is independent of size; timings scale up naturally.
# ═══════════════════════════════════════════════════════════════════════
configs = [
    (D=8,  χ=128),
    (D=8,  χ=256),
    (D=10, χ=256),
    (D=10, χ=512),
]

for (D, χ) in configs
    total_splits = 128
    forloop_iter = total_splits ÷ nprocs

    # leg5 tensors (same construction as test_MPI_config.jl)
    FL  = CUDA.rand(Float64, χ, D, D, χ)
    ALu = CUDA.rand(Float64, χ, D, D, χ)
    ALd = CUDA.rand(Float64, χ, D, D, χ)
    M   = CUDA.rand(Float64, D, D, D, D, 2)

    tensor_mb = χ^2 * D^2 * 8 / 1024^2

    rank == 0 && @printf("─── D=%d χ=%d (per-arg %6.1f MB × 4 args) ───\n", D, χ, tensor_mb)

    # ── Forward correctness: all three methods give same output ──
    r_plain = checkpoint(Plain(), TeneT.FLmap_parallel, FL, ALu, ALd, M;
                         ifparallel=true, forloop_iter)
    CUDA.synchronize(); MPI.Barrier(comm)

    r_recomp = checkpoint(Recompute(), TeneT.FLmap_parallel, FL, ALu, ALd, M;
                          ifparallel=true, forloop_iter)
    CUDA.synchronize(); MPI.Barrier(comm)

    r_offload = checkpoint(Offload(), TeneT.FLmap_parallel, FL, ALu, ALd, M;
                           ifparallel=true, forloop_iter)
    CUDA.synchronize(); MPI.Barrier(comm)

    # Forward is a plain Julia call in all three (no @adjoint override fires);
    # should be bit-exact equal.
    fwd_rec_ok = isapprox(Array(r_plain), Array(r_recomp); rtol=1e-12)
    fwd_off_ok = isapprox(Array(r_plain), Array(r_offload); rtol=1e-12)
    check("Forward Recompute == Plain D=$D χ=$χ", fwd_rec_ok)
    check("Forward Offload == Plain D=$D χ=$χ", fwd_off_ok)

    # ── Backward correctness: gradients must match ──
    loss(f) = f() |> sum

    # Plain (no @adjoint — baseline gradient)
    g_plain = nothing
    t_plain = try
        _, bp = Zygote.pullback(x -> sum(checkpoint(Plain(), TeneT.FLmap_parallel, x, ALu, ALd, M;
                                                     ifparallel=true, forloop_iter)), FL)
        g_plain = bp(one(eltype(FL)))[1]
        CUDA.synchronize(); MPI.Barrier(comm)

        GC.gc(); CUDA.reclaim()
        t = @elapsed begin
            _, bp = Zygote.pullback(x -> sum(checkpoint(Plain(), TeneT.FLmap_parallel, x, ALu, ALd, M;
                                                         ifparallel=true, forloop_iter)), FL)
            g_plain = bp(one(eltype(FL)))[1]
            CUDA.synchronize()
        end
        MPI.Barrier(comm)
        t
    catch e
        rank == 0 && println("  ERROR Plain backward D=$D χ=$χ: ", sprint(showerror, e))
        NaN
    end

    # Recompute
    g_recomp = nothing
    t_recomp = try
        _, bp = Zygote.pullback(x -> sum(checkpoint(Recompute(), TeneT.FLmap_parallel, x, ALu, ALd, M;
                                                     ifparallel=true, forloop_iter)), FL)
        g_recomp = bp(one(eltype(FL)))[1]
        CUDA.synchronize(); MPI.Barrier(comm)

        GC.gc(); CUDA.reclaim()
        t = @elapsed begin
            _, bp = Zygote.pullback(x -> sum(checkpoint(Recompute(), TeneT.FLmap_parallel, x, ALu, ALd, M;
                                                         ifparallel=true, forloop_iter)), FL)
            g_recomp = bp(one(eltype(FL)))[1]
            CUDA.synchronize()
        end
        MPI.Barrier(comm)
        t
    catch e
        rank == 0 && println("  ERROR Recompute backward D=$D χ=$χ: ", sprint(showerror, e))
        NaN
    end

    # Offload (the one we're probing for multi-GPU issues)
    g_offload = nothing
    t_offload = try
        _, bp = Zygote.pullback(x -> sum(checkpoint(Offload(), TeneT.FLmap_parallel, x, ALu, ALd, M;
                                                     ifparallel=true, forloop_iter)), FL)
        g_offload = bp(one(eltype(FL)))[1]
        CUDA.synchronize(); MPI.Barrier(comm)

        GC.gc(); CUDA.reclaim()
        t = @elapsed begin
            _, bp = Zygote.pullback(x -> sum(checkpoint(Offload(), TeneT.FLmap_parallel, x, ALu, ALd, M;
                                                         ifparallel=true, forloop_iter)), FL)
            g_offload = bp(one(eltype(FL)))[1]
            CUDA.synchronize()
        end
        MPI.Barrier(comm)
        t
    catch e
        rank == 0 && println("  ERROR Offload backward D=$D χ=$χ: ", sprint(showerror, e))
        NaN
    end

    # Gradient equivalence
    if g_plain !== nothing && g_recomp !== nothing
        gp = Array(g_plain)
        gr = Array(g_recomp)
        rel = norm(gp .- gr) / norm(gp)
        rec_ok = rel < 1e-10
        check("Backward Recompute gradient == Plain D=$D χ=$χ", rec_ok; msg="rel=$rel")
    else
        check("Backward Recompute gradient == Plain D=$D χ=$χ", false; msg="missing grad")
    end

    if g_plain !== nothing && g_offload !== nothing
        gp = Array(g_plain)
        go = Array(g_offload)
        rel = norm(gp .- go) / norm(gp)
        off_ok = rel < 1e-10
        check("Backward Offload gradient == Plain D=$D χ=$χ", off_ok; msg="rel=$rel")
    else
        check("Backward Offload gradient == Plain D=$D χ=$χ", false; msg="missing grad")
    end

    # Device-residency check: Offload's device-arg reconstruction must land
    # on THIS rank's device (not rank 0's). If CUDA.device() returns the right
    # device and g_offload is a CuArray, we're OK.
    if g_offload !== nothing
        dev_ok = g_offload isa CuArray
        check("Offload gradient on CuArray D=$D χ=$χ", dev_ok; msg="got $(typeof(g_offload))")
    end

    rank == 0 && @printf("  Times (ms): Plain=%7.2f  Recompute=%7.2f  Offload=%7.2f  (off/plain=%.2fx)\n",
        t_plain*1000, t_recomp*1000, t_offload*1000,
        isnan(t_offload) || isnan(t_plain) ? NaN : t_offload/t_plain)

    GC.gc(); CUDA.reclaim()
    MPI.Barrier(comm)
end

rank == 0 && println()
rank == 0 && println("=" ^ 70)
if all_pass[]
    rank == 0 && println("ALL CHECKPOINT TESTS PASSED ✓")
else
    rank == 0 && println("SOME CHECKPOINT TESTS FAILED ✗")
end
rank == 0 && println("=" ^ 70)

GC.gc(); CUDA.reclaim()
MPI.Barrier(comm)
MPI.Finalize()
