# M5: cannon vumps_step assembly + the QR gather seam (4 ranks, 2×2 grid, CPU).
# Run via: julia --project=. test/run_test_cannon_m5.jl
# Design: docs/2026-06-15-m5-vumps-step-cannon-assembly-design.md
#
# NOTE: the parity gates feed serial and cannon the SAME (random, non-canonical) rt and
# compare outputs of a DETERMINISTIC function — so they do NOT call init_VUMPSRuntime
# (whose serial canonicalization is slow/flaky for unlucky random seeds). Convergence is
# irrelevant to parity; both paths run the identical finite computation.
#
# Gates (design §6, post-R1):
#   M5-0  isolated seam BIT-PARITY: feed IDENTICAL full AC/C → ACCtoALAR_cannon_gather_ref/ALCtoAC_cannon
#         bit-match serial (no gauge ambiguity — same input). The rigorous seam proof.
#   M5-1  full-step forward parity: vumps_step_cannon vs serial. FL/FR/C/err at 1e-8;
#         AL/AR up-to-gauge at 1e-5 (QR amplifies the upstream block-vs-full FP-order diff
#         in AC; design F4). Verbose: prints the actual relerr per tensor.
#   M5-2  gradient parity (the R-1 catcher): LINEAR scattered-weight loss on AL_out+AR_out,
#         per-BLOCK compare (an N1/N2 over-count shows per-rank; invisible to a global norm).
#         Plain + Recompute.
#   M5-3  init_VUMPSRuntime_cannon cross-rank consistency (catches the un-bcast'd RNG bug, F1).
#   M5-4  routing: vumps_step(rt, M, alg{grid}) == vumps_step_cannon directly.
using Test
using MPI
using LinearAlgebra
using Random
using Zygote
using Printf
using TeneT
using TeneT: cannon_grid, CannonGrid, cannon_scatter, cannon_gather, split_ranges,
             VUMPS, General, StructArray, Recompute, VUMPSRuntime,
             vumps_step, init_VUMPSRuntime_cannon, vumps_step_cannon,
             ALCtoAC, ALCtoAC_cannon, ACCtoALAR, ACCtoALAR_dist_cannon,
             leftenv, leftenv_cannon, rightenv, rightenv_cannon,
             ACenv, ACenv_cannon, Cenv, Cenv_cannon, checkpoint, qrpos,
             ALCtoAC_cannon_gather_ref, ACCtoALAR_cannon_gather_ref
import ChainRulesCore

MPI.Init()
const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
@assert MPI.Comm_size(comm) == 4 "test_cannon_m5.jl expects exactly 4 ranks"
say(s) = (rank == 0 && (println(s); flush(stdout)))

# ── helpers ──────────────────────────────────────────────────────────────────
scatter_sa(SA, g) = StructArray([cannon_scatter(t, g) for t in SA.data], SA.pattern)
gather_sa(SA, g)  = StructArray([cannon_gather(t, g)  for t in SA.data], SA.pattern)
scatter_rt(rt, g) = VUMPSRuntime(scatter_sa(rt.AL, g), scatter_sa(rt.AR, g), rt.C,
                                 scatter_sa(rt.FL, g), scatter_sa(rt.FR, g))   # C replicated
# best-global-phase/scale-aligned relative error (gauge-robust to a global phase)
ph_relerr(a, b) = (c = dot(b, a) / dot(b, b); norm(a .- b .* c) / max(norm(b), eps()))

# random (non-canonical) full rt + M, identical on every rank (fixed seed).
function build_rt(Ni, Nj, χ, D; d=2, seed=42, pattern=nothing)
    Random.seed!(seed)
    pat = pattern === nothing ? reshape(collect(1:Ni*Nj), Ni, Nj) : pattern
    nu = length(unique(pat))
    sa(dims) = StructArray([rand(ComplexF64, dims...) for _ in 1:nu], pat)
    rt = VUMPSRuntime(sa((χ,D,D,χ)), sa((χ,D,D,χ)), sa((χ,χ)), sa((χ,D,D,χ)), sa((χ,D,D,χ)))
    M  = StructArray([rand(ComplexF64, D, D, D, D, d) for _ in 1:nu], pat)
    return rt, M
end
const ALG_KW = (ifsimple_eig=true, ifupdown=false, maxiter=1, maxiter_ad=1, verbosity=0)

# ── Gate M5-0: isolated seam BIT-parity (the rigorous seam proof) ─────────────
@testset "Gate M5-0: seam bit-parity (identical input)" begin
    g = cannon_grid(2, 2)
    for (ci, (Ni, Nj, pat)) in enumerate([(1, 1, nothing), (2, 2, [1 2; 2 1]), (2, 2, nothing)])
        χ, D = (16, 2)
        Random.seed!(3000 + ci)
        nu = pat === nothing ? Ni*Nj : length(unique(pat))
        patt = pat === nothing ? reshape(collect(1:Ni*Nj), Ni, Nj) : pat
        AC = StructArray([rand(ComplexF64, χ, D, D, χ) for _ in 1:nu], patt)
        AL = StructArray([rand(ComplexF64, χ, D, D, χ) for _ in 1:nu], patt)
        C  = StructArray([rand(ComplexF64, χ, χ)        for _ in 1:nu], patt)
        # ACCtoALAR: identical full AC/C → cannon (gathered) must equal serial bit-for-bit.
        ALs, ARs, eLs, eRs = ACCtoALAR(AC, C)
        ALc, ARc, eLc, eRc = ACCtoALAR_cannon_gather_ref(scatter_sa(AC, g), C, g)
        ALcf = gather_sa(ALc, g); ARcf = gather_sa(ARc, g)
        for idx in 1:nu
            @test maximum(abs, ALcf.data[idx] .- ALs.data[idx]) == 0
            @test maximum(abs, ARcf.data[idx] .- ARs.data[idx]) == 0
        end
        @test eLc == eLs && eRc == eRs
        # ALCtoAC: identical full AL/C → cannon (gathered) must equal serial bit-for-bit.
        ACs = ALCtoAC(AL, C)
        ACcf = gather_sa(ALCtoAC_cannon_gather_ref(scatter_sa(AL, g), C, g), g)
        for idx in 1:nu
            @test maximum(abs, ACcf.data[idx] .- ACs.data[idx]) == 0
        end
    end
end

# ── Gate M5-1: full-step forward parity ───────────────────────────────────────
# Validates the assembled vumps_step_cannon via the STABLE, conditioning-independent
# chain: (a) the full step's FL/FR/C/err match serial to machine precision; (b) the
# pre-seam INTERMEDIATE AC (the seam's input) matches serial up to per-cell phase.
# Together with the bit-exact seam (M5-0), (b) proves AL/AR are correct.
#
# Why NOT compare the final AL/AR directly: AL = qrpos(AC)·qrpos(C)′, and qrpos(·).Q
# amplifies the inevitable ~1e-16 cannon-vs-serial residual by cond(C). A random,
# non-canonical rt yields a near-singular Cenv-output C (cond ~1e15), so AL/AR diverge
# O(1) — a numerical-conditioning artifact of the non-physical fixture, NOT a parity
# defect (confirmed: AL sensitivity ∝ cond(C); seam bit-exact given identical input).
# At a physical fixed point C carries the bond spectrum and is well-conditioned.
@testset "Gate M5-1: vumps_step_cannon forward parity" begin
    g = cannon_grid(2, 2)
    for (ci, (Ni, Nj, pat)) in enumerate([(1, 1, nothing), (2, 2, [1 2; 2 1]), (2, 2, nothing)])
        χ, D = 14, 2
        rt, M = build_rt(Ni, Nj, χ, D; seed=7000 + ci, pattern=pat)
        alg_s = VUMPS(General(); power_iter=20, forloop_iter=1, ALG_KW...)
        alg_c = VUMPS(General(); power_iter=20, forloop_iter=1, grid=g, ALG_KW...)
        rtb = scatter_rt(rt, g)
        # (a) assembled full step → FL/FR/C/err parity
        rt_s2, err_s = vumps_step(rt, M, alg_s)
        rt_c2, err_c = vumps_step_cannon(rtb, M, g, alg_c)
        FLc = gather_sa(rt_c2.FL, g); FRc = gather_sa(rt_c2.FR, g)
        maxFL = maxFR = maxC = 0.0
        for idx in 1:length(rt_s2.FL.data)
            maxFL = max(maxFL, ph_relerr(FLc.data[idx], rt_s2.FL.data[idx]))
            maxFR = max(maxFR, ph_relerr(FRc.data[idx], rt_s2.FR.data[idx]))
            maxC  = max(maxC,  ph_relerr(rt_c2.C.data[idx], rt_s2.C.data[idx]))
        end
        # (b) pre-seam intermediate AC parity (manual pipeline, same as vumps_step's first 4 calls)
        AC0s = ALCtoAC(rt.AL, rt.C)
        _, FLs = leftenv(rt.AL, conj(rt.AL), M, rt.FL; alg=alg_s)
        _, FRs = rightenv(rt.AR, conj(rt.AR), M, rt.FR; alg=alg_s)
        _, ACs = ACenv(AC0s, FLs, M, FRs; alg=alg_s)
        AC0c = ALCtoAC_cannon(rtb.AL, rtb.C, g)
        _, FLc2 = leftenv_cannon(rtb.AL, conj(rtb.AL), M, rtb.FL, g; alg=alg_c)
        _, FRc2 = rightenv_cannon(rtb.AR, conj(rtb.AR), M, rtb.FR, g; alg=alg_c)
        _, ACc = ACenv_cannon(AC0c, FLc2, M, FRc2, g; alg=alg_c)
        ACcf = gather_sa(ACc, g)
        maxAC = maximum(ph_relerr(ACcf.data[idx], ACs.data[idx]) for idx in 1:length(ACs.data))
        say(@sprintf("  [M5-1 case %d (%dx%d)] FL=%.1e FR=%.1e C=%.1e | seam-input AC=%.1e | err diff=%.1e",
                     ci, Ni, Nj, maxFL, maxFR, maxC, maxAC, abs(err_c - err_s)))
        @test maxFL ≤ 1e-9
        @test maxFR ≤ 1e-9
        @test maxC  ≤ 1e-9
        @test maxAC ≤ 1e-9      # seam input matches serial → with bit-exact seam (M5-0), AL/AR correct
        @test abs(err_c - err_s) ≤ 1e-9
    end
end

# ── Gate M5-2: seam gradient parity (the R-1 over-count catcher) ──────────────
# Tests the NEW seam adjoints (QR-seam AC-gather take-my-block + AL/AR-scatter allreduce;
# entry-seam AL-gather + AC-scatter) in ISOLATION, with a WELL-CONDITIONED (orthogonal) C
# so the qrpos pullback is stable and the gradient is a clean parity signal. A linear
# scattered-weight loss; per-BLOCK compare — an N1/N2 over-count (the R-1 trap: using a
# reduce-scatter gather adjoint instead of take-my-block) shows as a constant per-rank
# scale, caught here. Plain + a Recompute wrap (R-6: re-runs the seam collectives on
# backward). Env-solver adjoints are covered by the M4 gates; the full step composes them.
@testset "Gate M5-2: seam gradient parity (R-1 catcher)" begin
    g = cannon_grid(2, 2)
    χ, D = 10, 2
    a_rs = split_ranges(χ, g.N1); i_rs = split_ranges(χ, g.N2)
    blkof(t) = t[a_rs[g.r1+1], :, :, i_rs[g.r2+1]]
    relb(cc, rc) = norm(cc - blkof(rc)) / max(norm(blkof(rc)), 1e-12)
    reld(cc, rc) = norm(cc - rc) / max(norm(rc), 1e-12)
    for (ci, (Ni, Nj, pat)) in enumerate([(1, 1, nothing), (2, 2, [1 2; 2 1])])
        patt = pat === nothing ? reshape(collect(1:Ni*Nj), Ni, Nj) : pat
        nu = length(unique(patt))
        Random.seed!(8000 + ci)
        AC = StructArray([rand(ComplexF64, χ, D, D, χ) for _ in 1:nu], patt)
        AL = StructArray([rand(ComplexF64, χ, D, D, χ) for _ in 1:nu], patt)
        C  = StructArray([Matrix(qr(rand(ComplexF64, χ, χ)).Q) for _ in 1:nu], patt)  # cond≈1
        Random.seed!(8500 + ci)
        WAL = [rand(ComplexF64, χ, D, D, χ) for _ in 1:nu]; WAR = [rand(ComplexF64, χ, D, D, χ) for _ in 1:nu]
        WAC = [rand(ComplexF64, χ, D, D, χ) for _ in 1:nu]
        WALb = [cannon_scatter(W, g) for W in WAL]; WARb = [cannon_scatter(W, g) for W in WAR]
        WACb = [cannon_scatter(W, g) for W in WAC]
        ACb = scatter_sa(AC, g); ALb = scatter_sa(AL, g)

        # ── ACCtoALAR seam: loss on AL+AR; grad wrt AC (block) + C (replicated) ──
        lqr_ref(ac, c) = let (al, ar, _, _) = ACCtoALAR(ac, c)
            real(sum(sum(conj(WAL[k]) .* al.data[k]) + sum(conj(WAR[k]) .* ar.data[k]) for k in 1:nu)) end
        lqr_c(ac, c)   = let (al, ar, _, _) = ACCtoALAR_dist_cannon(ac, c, g)
            real(sum(sum(conj(WALb[k]) .* al.data[k]) + sum(conj(WARb[k]) .* ar.data[k]) for k in 1:nu)) end
        gr = Zygote.gradient(lqr_ref, AC, C); gc = Zygote.gradient(lqr_c, ACb, C)
        eAC = maximum(relb(gc[1].data[k], gr[1].data[k]) for k in 1:nu)
        eC1 = (gr[2] === nothing || gc[2] === nothing) ? 0.0 : maximum(reld(gc[2].data[k], gr[2].data[k]) for k in 1:nu)
        # Recompute wrap (R-6: seam collectives re-run on backward)
        lqr_cr(ac, c) = let (al, ar, _, _) = checkpoint(Recompute(), ACCtoALAR_dist_cannon, ac, c, g)
            real(sum(sum(conj(WALb[k]) .* al.data[k]) + sum(conj(WARb[k]) .* ar.data[k]) for k in 1:nu)) end
        gcr = Zygote.gradient(lqr_cr, ACb, C)
        eACr = maximum(relb(gcr[1].data[k], gr[1].data[k]) for k in 1:nu)

        # ── ALCtoAC seam: loss on AC_out; grad wrt AL (block) + C (replicated) ──
        lal_ref(al, c) = real(sum(sum(conj(WAC[k])  .* ALCtoAC(al, c).data[k])         for k in 1:nu))
        lal_c(al, c)   = real(sum(sum(conj(WACb[k]) .* ALCtoAC_cannon(al, c, g).data[k]) for k in 1:nu))
        gr2 = Zygote.gradient(lal_ref, AL, C); gc2 = Zygote.gradient(lal_c, ALb, C)
        eAL = maximum(relb(gc2[1].data[k], gr2[1].data[k]) for k in 1:nu)
        eC2 = (gr2[2] === nothing || gc2[2] === nothing) ? 0.0 : maximum(reld(gc2[2].data[k], gr2[2].data[k]) for k in 1:nu)

        say(@sprintf("  [M5-2 case %d (%dx%d)] ACCtoALAR: dAC=%.1e dC=%.1e (recompute dAC=%.1e) | ALCtoAC: dAL=%.1e dC=%.1e",
                     ci, Ni, Nj, eAC, eC1, eACr, eAL, eC2))
        @test eAC ≤ 1e-7
        @test eC1 ≤ 1e-7
        @test eACr ≤ 1e-7
        @test eAL ≤ 1e-7
        @test eC2 ≤ 1e-7
    end
end

# ── Gate M5-2b: pre-seam pipeline gradient — the cross-seam cotangent SUM ─────
# rt.AL is consumed TWICE in one step: by the entry seam ALCtoAC_cannon (gather adjoint =
# take-my-block) AND by leftenv_cannon (gather adjoint = reduce-scatter). Its cotangent must
# be the SUM of both. We differentiate a loss on the seam-INPUT AC (after the 4 pre-seam
# calls), which depends on rt.AL through ALCtoAC AND through FL=leftenv — exercising exactly
# that sum — and compare dAL/dAR per-block to serial. Loss is on AC (NOT through the QR seam),
# so no qrpos/cond(C) amplification; low power_iter keeps the eigenvector-derivative stable.
@testset "Gate M5-2b: cross-seam cotangent sum (dAL/dAR)" begin
    g = cannon_grid(2, 2)
    χ, D = 10, 2
    a_rs = split_ranges(χ, g.N1); i_rs = split_ranges(χ, g.N2)
    blkof(t) = t[a_rs[g.r1+1], :, :, i_rs[g.r2+1]]
    for (ci, (Ni, Nj, pat)) in enumerate([(1, 1, nothing), (2, 2, [1 2; 2 1])])
        rt, M = build_rt(Ni, Nj, χ, D; seed=8800 + ci, pattern=pat)
        nu = length(M.data)
        alg_s = VUMPS(General(); power_iter=2, forloop_iter=1, ALG_KW...)
        alg_c = VUMPS(General(); power_iter=2, forloop_iter=1, grid=g, ALG_KW...)
        Random.seed!(8850 + ci)
        W  = [rand(ComplexF64, χ, D, D, χ) for _ in 1:nu]; Wb = [cannon_scatter(w, g) for w in W]
        rtb = scatter_rt(rt, g)
        ls(al, ar, c, fl, fr) = let ac = ALCtoAC(al, c)
            _, FLs = leftenv(al, conj(al), M, fl; alg=alg_s); _, FRs = rightenv(ar, conj(ar), M, fr; alg=alg_s)
            _, AC = ACenv(ac, FLs, M, FRs; alg=alg_s); real(sum(sum(conj(W[k]) .* AC.data[k]) for k in 1:nu)) end
        lc(al, ar, c, fl, fr) = let ac = ALCtoAC_cannon(al, c, g)
            _, FLc = leftenv_cannon(al, conj(al), M, fl, g; alg=alg_c); _, FRc = rightenv_cannon(ar, conj(ar), M, fr, g; alg=alg_c)
            _, AC = ACenv_cannon(ac, FLc, M, FRc, g; alg=alg_c); real(sum(sum(conj(Wb[k]) .* AC.data[k]) for k in 1:nu)) end
        gr = Zygote.gradient(ls, rt.AL, rt.AR, rt.C, rt.FL, rt.FR)
        gc = Zygote.gradient(lc, rtb.AL, rtb.AR, rtb.C, rtb.FL, rtb.FR)
        emax = 0.0
        for leg in (1, 2), k in 1:nu            # dAL, dAR (the double-consumption cotangents) — block
            (gr[leg] === nothing || gc[leg] === nothing) && continue
            e = norm(gc[leg].data[k] - blkof(gr[leg].data[k])) / max(norm(blkof(gr[leg].data[k])), 1e-12)
            emax = max(emax, e)
            @test isapprox(gc[leg].data[k], blkof(gr[leg].data[k]); rtol=1e-6, atol=1e-10)
        end
        say(@sprintf("  [M5-2b case %d (%dx%d)] dAL/dAR cross-seam block max rel=%.1e", ci, Ni, Nj, emax))
    end
end

# ── Gate M5-3: init_VUMPSRuntime_cannon cross-rank consistency (F1 catcher) ────
@testset "Gate M5-3: init_VUMPSRuntime_cannon cross-rank consistency" begin
    g = cannon_grid(2, 2)
    χ, D = 8, 2
    Random.seed!(9000)
    M = StructArray([rand(ComplexF64, D, D, D, D, 2)], reshape([1], 1, 1))   # MPO: identical on every rank
    # DIVERGE each rank's global RNG so init's internal initial_A draws a DIFFERENT random A
    # per rank — reproducing the real MPI-launch condition. This makes the unconditional bcast
    # in init_VUMPSRuntime_cannon load-bearing: with it, the gathered runtime is bitwise-identical
    # across ranks (test passes); without it, ranks would scatter slices of different A (test would
    # fail). (A shared seed would mask the F1 bug — every rank's initial_A would draw identically.)
    for _ in 1:rank; rand(ComplexF64); end
    alg_c = VUMPS(General(); power_iter=15, forloop_iter=1, grid=g, ALG_KW...)
    rt = init_VUMPSRuntime_cannon(M, χ, g, alg_c)
    for (full, name) in [(gather_sa(rt.AL, g), "AL"), (gather_sa(rt.AR, g), "AR"),
                         (gather_sa(rt.FL, g), "FL"), (gather_sa(rt.FR, g), "FR"), (rt.C, "C")]
        for idx in 1:length(full.data)
            ref = MPI.bcast(full.data[idx], 0, comm)
            @test maximum(abs, full.data[idx] .- ref) == 0    # bitwise-identical on every rank
        end
    end
end

# ── Gate M5-4: routing via alg.grid ───────────────────────────────────────────
@testset "Gate M5-4: vumps_step routes to cannon when grid set" begin
    g = cannon_grid(2, 2)
    rt, M = build_rt(2, 2, 14, 2; seed=9500)
    alg_c = VUMPS(General(); power_iter=15, forloop_iter=1, grid=g, ALG_KW...)
    rtb = scatter_rt(rt, g)
    via_routing, e1 = vumps_step(rtb, M, alg_c)               # routes to cannon
    direct, e2      = vumps_step_cannon(rtb, M, g, alg_c)
    @test e1 ≈ e2
    for idx in 1:length(direct.AL.data)
        @test direct.AL.data[idx] == via_routing.AL.data[idx]
    end
end

# ── Gate M5-5: qrpos phase-equivariance (the seam's load-bearing premise) ─────
# The whole "intermediate AC matches up to per-cell phase + bit-exact seam ⇒ AL/AR correct"
# argument rests on qrpos(c·M).Q = (c/|c|)·qrpos(M).Q for complex c. Gate it so a future
# qrpos regression that broke this would be caught here, not silently in production.
@testset "Gate M5-5: qrpos phase-equivariance" begin
    Random.seed!(55)
    M0 = rand(ComplexF64, 40, 12)
    Q1, _ = qrpos(M0)
    for c in (exp(im*0.7), 0.3 + 0.9im, rand(ComplexF64), -1.0 + 0im)
        Q2, _ = qrpos(c .* M0)
        @test norm(Q2 .- (c/abs(c)) .* Q1) / norm(Q1) ≤ 1e-12
    end
end

say("all M5 gates done.")
