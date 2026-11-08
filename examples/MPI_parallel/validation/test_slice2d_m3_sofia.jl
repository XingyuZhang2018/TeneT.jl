# Sofia 4×H200 M3 Slice2D validation: distributed parity (forward + Zygote-sum
# gradient) vs the serial kernels for all four M3 slice2d-wrapper maps —
# Cmap_slice2d / FRmap_slice2d_dist / ACmap_slice2d_dist / ACdmap_slice2d_dist —
# plus per-map device-memory accounting. The NEW value over the 4-rank CPU
# parity gate (test/test_slice2d_m3.jl, already green) is the GPU MEMORY check at
# production scale: FRmap and ACdmap carry full-`i`×full-`d` chain intermediates
# and rely on the 2-level chunk to stay bounded (≈χ²D⁴/(P·forloop_iter), NOT a
# χ×χ plane) — a claim the small CPU test cannot probe.
#
# Launched by Sofia/submit_test_slice2d_m3.sh (mpirun -np 4, 2×2 Slice2D grid,
# CUDA_VISIBLE_DEVICES pins one GPU per rank). Mirrors test_slice2d_sofia.jl's
# mem_line / timed helpers verbatim.
#
# Local CPU smoke (no GPU, tiny dims): runs the SAME *_slice2d_dist code on plain
# Arrays under 4 ranks via TENET_BENCH_CPU=1 — confirms the driver parses + runs
# end-to-end before it ever reaches the cluster:
#   TENET_BENCH_CPU=1 julia --project=. examples/MPI_parallel/validation/run_test_slice2d_m3_sofia_cpu.jl
# (the GPU path requires CUDA.functional(); the CPU smoke uses Array and skips
# all CUDA calls.)

# atype switch: GPU (CuArray) on Sofia, Array for a local CPU syntax/smoke check
# (set TENET_BENCH_CPU=1; uses tiny dims, no CUDA needed). Same shim shape as
# bench_chain_gate_m2_sofia.jl.
const BENCH_CPU = get(ENV, "TENET_BENCH_CPU", "0") == "1"

using MPI, Zygote, LinearAlgebra, Random, Printf
using TeneT
using TeneT: slice2d_grid, Slice2DGrid, slice2d_scatter, slice2d_gather, split_ranges,
             Cmap, Cmap_slice2d,
             FRmap, FRmap_slice2d_dist,
             ACmap, ACmap_slice2d_dist,
             ACdmap, ACdmap_slice2d_dist
if !BENCH_CPU
    using CUDA
end

MPI.Init()
const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
const nprocs = MPI.Comm_size(comm)
@assert nprocs == 4 "test_slice2d_m3_sofia.jl expects exactly 4 ranks (2×2 grid)"

# Production-ish cells. The validation cell is intentionally smaller than the
# production cell so the gradient-parity allreduce never OOMs on the serial
# reference (which holds the FULL χ×χ tensors on every rank).
const CHI = parse(Int, get(ENV, "TENET_SLICE2D_CHI", "400"))   # production χ
const D   = parse(Int, get(ENV, "TENET_SLICE2D_D",   "10"))    # production D
const CHI_VALID = parse(Int, get(ENV, "TENET_SLICE2D_CHI_VALID", "256"))
const D_VALID   = parse(Int, get(ENV, "TENET_SLICE2D_D_VALID",   "8"))
const d_phys = 2

report(s...) = rank == 0 && println(s...)

# ── device/array shim (identical harness on CPU and GPU) ─────────────────────
if BENCH_CPU
    _arr(x)   = x                            # already an Array
    _sync()   = nothing
    _reclaim() = nothing
    mem_used_gb() = 0.0                       # no device-mem probe on CPU
    _devname() = "CPU"
else
    _arr(x)   = CuArray(x)
    _sync()   = CUDA.synchronize()
    _reclaim() = CUDA.reclaim()
    mem_used_gb() = (CUDA.total_memory() - CUDA.available_memory()) / 2^30
    _devname() = CUDA.name(CUDA.device())
end

# mem_line / timed: verbatim shape from test_slice2d_sofia.jl:17-31 (the device
# probe is no-op'd to 0 on CPU so the smoke runs without CUDA).
function mem_line(label)
    m = mem_used_gb()
    mmax = MPI.Allreduce(m, MPI.MAX, comm)
    rank == 0 && @printf("[mem]  %-40s rank0 %6.2f GB   max %6.2f GB\n", label, m, mmax)
    return mmax
end
function timed(f, label, n)
    f(); _sync(); MPI.Barrier(comm)          # warm
    t = MPI.Wtime()
    for _ in 1:n; f(); end
    _sync(); MPI.Barrier(comm)
    dt = (MPI.Wtime() - t) / n
    dtmax = MPI.Allreduce(dt, MPI.MAX, comm)
    rank == 0 && @printf("[time] %-40s %8.3f s/call (max over ranks)\n", label, dtmax)
    return dtmax
end

# Allreduced max relative error across ranks (a block-local mismatch on ONE rank
# must fail the whole gate).
relerr(a, b) = norm(a - b) / norm(b)
allreduce_max(x) = MPI.Allreduce(Float64(x), MPI.MAX, comm)

const FWD_RTOL  = 1e-10   # forward rel err (production GPU; CPU test gate is 1e-12)
const GRAD_RTOL = 1e-8    # gradient max rel err, allreduced (CPU test gate is 1e-10)

# forloop_iter for the gather-class maps: pick n so n_d=n_i ≥ 2N (= 4 for the
# 2×2 grid), the regime that forces the 2-level accumulate/assign chunk (and the
# bounded full-i×full-d intermediates). ACmap is single-l-chunk; the same n is a
# valid (≥1) chunk count there too.
const FLOOP = parse(Int, get(ENV, "TENET_SLICE2D_FLOOP", "4"))

# ── per-map validators ───────────────────────────────────────────────────────
# Each returns (fwd_ok::Bool, grad_ok::Bool); Cmap takes any grid (replicated
# output), the other three require the square 2×2 grid. The `mem_probe` flag
# prints mem_line immediately after the forward and after the fwd+bwd for the
# two memory-load-bearing maps (FRmap, ACdmap).

# leg5 input geometry: A/Bu/Bd boundary tensors (χ,D,D,χ); M1,M2 (D,D,D,D,d).
function make_leg5(χ, D; seed)
    Random.seed!(seed)   # identical tensors on every rank
    A  = _arr(rand(ComplexF64, χ, D, D, χ))
    Bu = _arr(rand(ComplexF64, χ, D, D, χ))
    Bd = _arr(rand(ComplexF64, χ, D, D, χ))
    M1 = _arr(rand(ComplexF64, D, D, D, D, d_phys))
    M2 = _arr(rand(ComplexF64, D, D, D, D, d_phys))
    W  = _arr(rand(ComplexF64, χ, D, D, χ))
    return A, Bu, Bd, M1, M2, W
end

# ── Cmap (replicated output; tuple-M not applicable — leg4 C/FL/FR) ───────────
function validate_Cmap(χ, D, g; tuple_M)
    Random.seed!(7000 + χ + D)
    FL = _arr(rand(ComplexF64, χ, D, D, χ))
    FR = _arr(rand(ComplexF64, χ, D, D, χ))
    C  = _arr(rand(ComplexF64, χ, χ))
    W  = _arr(rand(ComplexF64, χ, χ))
    FLb = slice2d_scatter(FL, g); FRb = slice2d_scatter(FR, g)

    ref = Cmap(C, FL, FR)                              # full χ×χ
    out = Cmap_slice2d(C, FLb, FRb, g)                  # full χ×χ replicated
    ef = allreduce_max(relerr(out, ref))

    loss_ref(C, FL, FR)    = real(sum(W .* Cmap(C, FL, FR)))
    loss_dist(C, FLb, FRb) = real(sum(W .* Cmap_slice2d(C, FLb, FRb, g)))
    g_ref  = Zygote.pullback(loss_ref,  C, FL, FR)[2](1.0)
    g_dist = Zygote.pullback(loss_dist, C, FLb, FRb)[2](1.0)
    a_rs = split_ranges(χ, g.N1); e_rs = split_ranges(χ, g.N2)
    blkof(x) = x[a_rs[g.r1 + 1], :, :, e_rs[g.r2 + 1]]
    eg = allreduce_max(maximum((
        relerr(g_dist[1], g_ref[1]),          # dC full (replicated)
        relerr(g_dist[2], blkof(g_ref[2])),   # dFL block (take-my-block)
        relerr(g_dist[3], blkof(g_ref[3])),   # dFR block
    )))
    report(@sprintf("[Cmap]   χ=%d D=%d  fwd rel=%.2e  grad max rel=%.2e", χ, D, ef, eg))
    return ef <= FWD_RTOL, eg <= GRAD_RTOL
end

# ── gather-class maps (FR/AC/ACd): block output, square grid, optional mem probe
# `mapfwd(A,Bu,Bd,M,g)` is the *_slice2d_dist; `serial(A,Bu,Bd,M)` is the kernel.
function validate_gather(name, mapfwd, serial, χ, D, g; tuple_M, mem_probe, floop)
    A, Bu, Bd, M1, M2, W = make_leg5(χ, D; seed = 7100 + χ + D + (tuple_M ? 0 : 1))
    Ab = slice2d_scatter(A, g); Bub = slice2d_scatter(Bu, g); Bdb = slice2d_scatter(Bd, g)
    Wb = slice2d_scatter(W, g)
    M  = tuple_M ? (M1, M2) : M1            # single-M: serial uses M2=conj(M1) internally

    # forward parity
    ref = serial(A, Bu, Bd, M)
    out_blk = mapfwd(Ab, Bub, Bdb, M, g; forloop_iter = floop)
    if mem_probe
        _sync(); mem_line("$name forward (peak)")
    end
    out = slice2d_gather(out_blk, g)
    ef = allreduce_max(relerr(out, ref))

    # gradient parity (Zygote-sum loss, block-of for [a,e,f,i]/[i,j,k,l] = [r1,:,:,r2])
    loss_ref(A, Bu, Bd, M1, M2)   = real(sum(W  .* serial(A, Bu, Bd, (M1, M2))))
    loss_dist(Ab, Bub, Bdb, M1, M2) =
        real(sum(Wb .* mapfwd(Ab, Bub, Bdb, (M1, M2), g; forloop_iter = floop)))
    g_ref  = Zygote.pullback(loss_ref,  A, Bu, Bd, M1, M2)[2](1.0)
    g_dist = Zygote.pullback(loss_dist, Ab, Bub, Bdb, M1, M2)[2](1.0)
    if mem_probe
        _sync(); mem_line("$name fwd+bwd (peak retained)")
    end
    p_rs = split_ranges(χ, g.N1)
    blkof(x) = x[p_rs[g.r1 + 1], :, :, p_rs[g.r2 + 1]]
    eg = allreduce_max(maximum((
        relerr(g_dist[1], blkof(g_ref[1])),   # dA block
        relerr(g_dist[2], blkof(g_ref[2])),   # dBu block
        relerr(g_dist[3], blkof(g_ref[3])),   # dBd block
        relerr(g_dist[4], g_ref[4]),          # dM1 replicated
        relerr(g_dist[5], g_ref[5]),          # dM2 replicated
    )))
    report(@sprintf("[%-7s] χ=%d D=%d n=%d  fwd rel=%.2e  grad max rel=%.2e",
                    name, χ, D, floop, ef, eg))
    GC.gc(); _reclaim()
    return ef <= FWD_RTOL, eg <= GRAD_RTOL
end

# ── driver ───────────────────────────────────────────────────────────────────
function main()
    g = slice2d_grid(2, 2)   # square 2×2 (required by FR/AC/ACd; Cmap accepts it too)
    cells = BENCH_CPU ? [(8, 2)] : [(CHI_VALID, D_VALID), (CHI, D)]

    report("=== M3 Slice2D Sofia validation: 2×2 grid, ", _devname(),
           ", CLB=", get(ENV, "CUDA_LAUNCH_BLOCKING", "0"),
           ", forloop_iter=", BENCH_CPU ? 2 : FLOOP, " ===")
    report("# parity gate: forward rel ≤ ", FWD_RTOL, ", gradient max rel (allreduced) ≤ ", GRAD_RTOL)
    floop = BENCH_CPU ? 2 : FLOOP

    all_pass = true
    function record(name, χ, D, fwd_ok, grad_ok)
        ok = fwd_ok && grad_ok
        all_pass &= ok
        report(@sprintf("=== RESULT %-7s χ=%d D=%d : %s  (fwd %s, grad %s) ===",
                        name, χ, D, ok ? "PASS" : "FAIL",
                        fwd_ok ? "✓" : "✗", grad_ok ? "✓" : "✗"))
    end

    for (χ, Dc) in cells
        report("")
        report("──── cell χ=$χ D=$Dc ────")
        mem_line("cell start")

        # Cmap: both single-M-N/A (leg4) — run once. Replicated output, any grid.
        for tuple_M in (true, false)
            tuple_M || continue   # Cmap leg4 has no M-tuple variant; one pass only
            f, gd = validate_Cmap(χ, Dc, g; tuple_M)
            record("Cmap", χ, Dc, f, gd)
        end

        # FRmap / ACmap / ACdmap: tuple-M AND single-M, both halves.
        for tuple_M in (true, false)
            tag = tuple_M ? "" : "/1M"
            f, gd = validate_gather("FRmap$tag", FRmap_slice2d_dist, FRmap, χ, Dc, g;
                                    tuple_M, mem_probe = true,  floop)   # MEM load-bearing
            record("FRmap$tag", χ, Dc, f, gd)

            f, gd = validate_gather("ACmap$tag", ACmap_slice2d_dist, ACmap, χ, Dc, g;
                                    tuple_M, mem_probe = false, floop)
            record("ACmap$tag", χ, Dc, f, gd)

            f, gd = validate_gather("ACdmap$tag", ACdmap_slice2d_dist, ACdmap, χ, Dc, g;
                                    tuple_M, mem_probe = true,  floop)   # MEM load-bearing
            record("ACdmap$tag", χ, Dc, f, gd)
        end

        GC.gc(); _reclaim(); mem_line("cell end (after reclaim)")
    end

    report("")
    report(all_pass ? "=== RESULT: PASS ===" : "=== RESULT: FAIL ===")
    return all_pass
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
