# Sofia GPU M5 validation: distributed `vumps_step_slice2d` (full VUMPS step assembled
# from the four M4 env solvers + the two QR gather seams) parity vs the SERIAL
# `vumps_step`, on GPU, at multi-card scale (16-GPU 4×4 = 2 nodes). The NEW value over
# the CPU gate (test/test_slice2d_m5.jl) is the WHOLE step on CuArray + MPI-CUDA: the
# QR-seam gather/scatter of AC/AL/AR, the entry-seam ALCtoAC gather/scatter, plus the
# M4 env collectives, all on-device and cross-node.
#
# Parity uses a RANDOM (non-canonical) rt fed identically to serial and slice2d — both run
# the same DETERMINISTIC finite computation, so convergence/canonicalization is irrelevant
# (and we skip the slow serial init_VUMPSRuntime). Grid AUTO-DERIVED: 16 ranks → 4×4.
# Design: docs/2026-06-15-m5-vumps-step-slice2d-assembly-design.md (§6).
#
# Local CPU smoke (tiny dims, 4 ranks, plain Arrays):
#   TENET_BENCH_CPU=1 julia --project=. examples/MPI_parallel/run_test_slice2d_m5_sofia_cpu.jl

const BENCH_CPU = get(ENV, "TENET_BENCH_CPU", "0") == "1"

using MPI, Zygote, LinearAlgebra, Random, Printf
using TeneT
using TeneT: slice2d_grid, Slice2DGrid, slice2d_scatter, slice2d_gather, split_ranges,
             VUMPS, General, StructArray, Recompute, VUMPSRuntime,
             vumps_step, vumps_step_slice2d, ALCtoAC, ALCtoAC_slice2d, ACCtoALAR,
             ACCtoALAR_slice2d_gather_ref, ACCtoALAR_dist_slice2d,
             leftenv, leftenv_slice2d, rightenv, rightenv_slice2d, ACenv, ACenv_slice2d, checkpoint
if !BENCH_CPU
    using CUDA
end

MPI.Init()
const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
const nprocs = MPI.Comm_size(comm)
const N = isqrt(nprocs)
@assert N * N == nprocs "test_slice2d_m5_sofia.jl needs a square nprocs (4→2×2, 16→4×4); got $nprocs"

const CHI = parse(Int, get(ENV, "TENET_SLICE2D_CHI", "256"))
const D   = parse(Int, get(ENV, "TENET_SLICE2D_D",   "8"))
const d_phys = 2
const POWER    = parse(Int, get(ENV, "TENET_SLICE2D_POWER", "20"))
const POWER_AD = 4

report(s...) = rank == 0 && (println(s...); flush(stdout))

# ── device/array shim ─────────────────────────────────────────────────────────
if BENCH_CPU
    _arr(x) = x; _sync() = nothing; _reclaim() = nothing
    mem_used_gb() = 0.0; _devname() = "CPU"
else
    _arr(x) = CuArray(x); _sync() = CUDA.synchronize(); _reclaim() = CUDA.reclaim()
    mem_used_gb() = (CUDA.total_memory() - CUDA.available_memory()) / 2^30
    _devname() = CUDA.name(CUDA.device())
end
function mem_line(label)
    m = mem_used_gb(); mmax = MPI.Allreduce(m, MPI.MAX, comm)
    rank == 0 && @printf("[mem]  %-28s rank0 %6.2f GB   max %6.2f GB\n", label, m, mmax)
end

relerr(a, b) = norm(a - b) / max(norm(b), eps())
allreduce_max(x) = MPI.Allreduce(Float64(x), MPI.MAX, comm)
# global-phase/scale-insensitive relerr; GPU-safe (dot+broadcast, NO scalar indexing).
phase_relerr(a, b) = (c = dot(b, a) / dot(b, b); relerr(a, b .* c))

const FWD_RTOL  = 1e-10    # FL/FR/C/err + seam-input AC (stable, gauge-robust quantities)
const SEAM_RTOL = 1e-10    # seam bit-parity / gradient (GPU cuSOLVER: ~1e-12, not exact 0)
const GRAD_RTOL = 1e-8

scatter_sa(SA, g) = StructArray([slice2d_scatter(t, g) for t in SA.data], SA.pattern)
gather_sa(SA, g)  = StructArray([slice2d_gather(t, g)  for t in SA.data], SA.pattern)
scatter_rt(rt, g) = VUMPSRuntime(scatter_sa(rt.AL, g), scatter_sa(rt.AR, g), rt.C,
                                 scatter_sa(rt.FL, g), scatter_sa(rt.FR, g))

# random (non-canonical) full rt + M, identical on every rank (fixed seed), on device.
function build_rt(χ, Dc; seed)
    Random.seed!(seed)
    pat = reshape([1], 1, 1)
    sa(dims) = StructArray([_arr(rand(ComplexF64, dims...))], pat)
    rt = VUMPSRuntime(sa((χ,Dc,Dc,χ)), sa((χ,Dc,Dc,χ)), sa((χ,χ)), sa((χ,Dc,Dc,χ)), sa((χ,Dc,Dc,χ)))
    M  = StructArray([_arr(rand(ComplexF64, Dc, Dc, Dc, Dc, d_phys))], pat)
    return rt, M
end
algf()  = VUMPS(General(); ifsimple_eig=true, power_iter=POWER,    forloop_iter=1, ifupdown=false, maxiter=1, maxiter_ad=1, verbosity=0)
algad() = VUMPS(General(); ifsimple_eig=true, power_iter=POWER_AD, forloop_iter=1, ifupdown=false, maxiter=1, maxiter_ad=1, verbosity=0)

# ── Gate M5-0: isolated seam parity (identical full input → slice2d == serial) ──
function val_seam(χ, Dc, g)
    Random.seed!(3001)
    AC = StructArray([_arr(rand(ComplexF64, χ, Dc, Dc, χ))], reshape([1],1,1))
    AL = StructArray([_arr(rand(ComplexF64, χ, Dc, Dc, χ))], reshape([1],1,1))
    C  = StructArray([_arr(rand(ComplexF64, χ, χ))],          reshape([1],1,1))
    ALs, ARs, eLs, eRs = ACCtoALAR(AC, C)
    ALc, ARc, eLc, eRc = ACCtoALAR_slice2d_gather_ref(scatter_sa(AC, g), C, g)
    _sync()
    eAL = allreduce_max(relerr(gather_sa(ALc, g).data[1], ALs.data[1]))
    eAR = allreduce_max(relerr(gather_sa(ARc, g).data[1], ARs.data[1]))
    ACs = ALCtoAC(AL, C); ACc = gather_sa(ALCtoAC_slice2d(scatter_sa(AL, g), C, g), g)
    eAC = allreduce_max(relerr(ACc.data[1], ACs.data[1]))
    eerr = allreduce_max(abs(eLc - eLs) + abs(eRc - eRs))
    report(@sprintf("[seam ] AL=%.2e AR=%.2e ALCtoAC=%.2e errdiff=%.2e", eAL, eAR, eAC, eerr))
    return eAL ≤ SEAM_RTOL && eAR ≤ SEAM_RTOL && eAC ≤ SEAM_RTOL && eerr ≤ SEAM_RTOL
end

# ── Gate M5-1: full-step forward parity (stable, conditioning-independent) ─────
# Compare the assembled step's FL/FR/C/err to serial, AND the pre-seam INTERMEDIATE AC
# (the seam's input) up to phase. With the bit-exact seam (val_seam), matching AC proves
# AL/AR. We do NOT compare AL/AR directly: AL = qrpos(AC)·qrpos(C)′ and qrpos amplifies
# the ~1e-16 slice2d-vs-serial residual by cond(C); a random rt's Cenv-output C is
# near-singular (cond ~1e15) → AL diverges O(1) (a fixture-conditioning artifact, not a
# defect — see test/test_slice2d_m5.jl Gate M5-1 + the cond(C) confirmation).
function val_step_fwd(χ, Dc, g)
    rt, M = build_rt(χ, Dc; seed=7001); alg_s = algf(); alg_c = algf(); alg_c.grid = g
    rt_s2, e_s = vumps_step(rt, M, alg_s)
    rt_c2, e_c = vumps_step_slice2d(scatter_rt(rt, g), M, g, alg_c)
    _sync()
    eFL = allreduce_max(phase_relerr(gather_sa(rt_c2.FL, g).data[1], rt_s2.FL.data[1]))
    eFR = allreduce_max(phase_relerr(gather_sa(rt_c2.FR, g).data[1], rt_s2.FR.data[1]))
    eC  = allreduce_max(phase_relerr(rt_c2.C.data[1], rt_s2.C.data[1]))
    eerr = allreduce_max(abs(e_c - e_s))
    # pre-seam intermediate AC (vumps_step's first 4 calls), serial vs slice2d
    rtb = scatter_rt(rt, g)
    AC0s = ALCtoAC(rt.AL, rt.C)
    _, FLs = leftenv(rt.AL, conj(rt.AL), M, rt.FL; alg=alg_s)
    _, FRs = rightenv(rt.AR, conj(rt.AR), M, rt.FR; alg=alg_s)
    _, ACs = ACenv(AC0s, FLs, M, FRs; alg=alg_s)
    AC0c = ALCtoAC_slice2d(rtb.AL, rtb.C, g)
    _, FLc = leftenv_slice2d(rtb.AL, conj(rtb.AL), M, rtb.FL, g; alg=alg_c)
    _, FRc = rightenv_slice2d(rtb.AR, conj(rtb.AR), M, rtb.FR, g; alg=alg_c)
    _, ACc = ACenv_slice2d(AC0c, FLc, M, FRc, g; alg=alg_c)
    _sync()
    eAC = allreduce_max(phase_relerr(gather_sa(ACc, g).data[1], ACs.data[1]))
    report(@sprintf("[fwd  ] FL=%.2e FR=%.2e C=%.2e | seam-input AC=%.2e | errdiff=%.2e",
                    eFL, eFR, eC, eAC, eerr))
    return eFL ≤ FWD_RTOL && eFR ≤ FWD_RTOL && eC ≤ FWD_RTOL && eAC ≤ FWD_RTOL && eerr ≤ FWD_RTOL
end

# ── Gate M5-2: seam gradient parity — the R-1 over-count catcher ───────────────
# Tests the NEW seam adjoints (QR-seam AC-gather take-my-block + AL/AR-scatter allreduce;
# entry-seam AL-gather + AC-scatter) with a WELL-CONDITIONED (orthogonal) C so the qrpos
# pullback is stable. Linear scattered-weight loss, per-BLOCK compare: an N1/N2 over-count
# (reduce-scatter gather adjoint instead of take-my-block) shows as a per-rank constant
# scale, caught here. + a Recompute wrap (R-6: seam collectives re-run on backward).
function val_step_grad(χ, Dc, g)
    Random.seed!(8001)
    AC = StructArray([_arr(rand(ComplexF64, χ, Dc, Dc, χ))], reshape([1],1,1))
    AL = StructArray([_arr(rand(ComplexF64, χ, Dc, Dc, χ))], reshape([1],1,1))
    Q, _ = qr(rand(ComplexF64, χ, χ)); C = StructArray([_arr(Matrix(Q))], reshape([1],1,1))  # cond≈1
    Random.seed!(8501)
    WAL=_arr(rand(ComplexF64,χ,Dc,Dc,χ)); WAR=_arr(rand(ComplexF64,χ,Dc,Dc,χ)); WAC=_arr(rand(ComplexF64,χ,Dc,Dc,χ))
    WALb=slice2d_scatter(WAL,g); WARb=slice2d_scatter(WAR,g); WACb=slice2d_scatter(WAC,g)
    rs = split_ranges(χ, g.N1); blkof(t) = t[rs[g.r1+1], :, :, rs[g.r2+1]]
    ACb = StructArray([slice2d_scatter(AC.data[1], g)], reshape([1],1,1))
    ALb = StructArray([slice2d_scatter(AL.data[1], g)], reshape([1],1,1))
    # ACCtoALAR seam
    lqr_r(ac,c) = let (al,ar,_,_)=ACCtoALAR(ac,c);        real(sum(conj(WAL).*al.data[1])+sum(conj(WAR).*ar.data[1])) end
    lqr_c(ac,c) = let (al,ar,_,_)=ACCtoALAR_dist_slice2d(ac,c,g); real(sum(conj(WALb).*al.data[1])+sum(conj(WARb).*ar.data[1])) end
    gr=Zygote.gradient(lqr_r,AC,C); gc=Zygote.gradient(lqr_c,ACb,C); _sync()
    eAC=allreduce_max(relerr(gc[1].data[1], blkof(gr[1].data[1])))
    eC1=(gr[2]===nothing||gc[2]===nothing) ? 0.0 : allreduce_max(relerr(gc[2].data[1], gr[2].data[1]))
    lqr_cr(ac,c)=let (al,ar,_,_)=checkpoint(Recompute(),ACCtoALAR_dist_slice2d,ac,c,g); real(sum(conj(WALb).*al.data[1])+sum(conj(WARb).*ar.data[1])) end
    gcr=Zygote.gradient(lqr_cr,ACb,C); _sync()
    eACr=allreduce_max(relerr(gcr[1].data[1], blkof(gr[1].data[1])))
    # ALCtoAC seam
    lal_r(al,c)=real(sum(conj(WAC) .*ALCtoAC(al,c).data[1]))
    lal_c(al,c)=real(sum(conj(WACb).*ALCtoAC_slice2d(al,c,g).data[1]))
    gr2=Zygote.gradient(lal_r,AL,C); gc2=Zygote.gradient(lal_c,ALb,C); _sync()
    eAL=allreduce_max(relerr(gc2[1].data[1], blkof(gr2[1].data[1])))
    report(@sprintf("[grad ] ACCtoALAR dAC=%.2e dC=%.2e (recompute %.2e) | ALCtoAC dAL=%.2e", eAC, eC1, eACr, eAL))
    return eAC ≤ GRAD_RTOL && eC1 ≤ GRAD_RTOL && eACr ≤ GRAD_RTOL && eAL ≤ GRAD_RTOL
end

function main()
    g = slice2d_grid(N, N)
    χ, Dc = BENCH_CPU ? (8, 2) : (CHI, D)
    report("=== M5 vumps_step_slice2d Sofia validation: ", N, "×", N, " grid (", nprocs, " ranks), ",
           _devname(), ", CLB=", get(ENV, "CUDA_LAUNCH_BLOCKING", "0"), ", power=", POWER, "/", POWER_AD, " ===")
    report("# gate: seam ≤ ", SEAM_RTOL, "; fwd FL/FR/C/err + seam-input AC ≤ ", FWD_RTOL, "; seam grad ≤ ", GRAD_RTOL)
    mem_line("start")
    all_pass = true
    rec(name, ok) = (all_pass &= ok; report(@sprintf("=== RESULT %-10s : %s ===", name, ok ? "PASS" : "FAIL")))
    rec("seam",     val_seam(χ, Dc, g));      mem_line("after seam")
    rec("fwd step", val_step_fwd(χ, Dc, g));  mem_line("after fwd")
    rec("grad step", val_step_grad(χ, Dc, g))
    GC.gc(); _reclaim(); mem_line("end (reclaim)")
    report("")
    report(all_pass ? "=== RESULT: PASS ===" : "=== RESULT: FAIL ===")
    return all_pass
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
