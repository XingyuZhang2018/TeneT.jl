# M5-plaq: Plaquette-mode cannon vumps_step parity (4 ranks, 2×2 grid, CPU).
# Run via: julia --project=. test/run_test_cannon_m5_plaq.jl
#
# Plaquette cannon mirrors General (M5) with FR←FL[:,jr] and no AR/rightenv. Same
# stable-chain validation: seam bit-exact + FL/C/err + intermediate-AC parity + seam
# gradient (well-conditioned C). (Direct AL comparison is invalid for a random rt —
# qrpos amplifies the 1e-16 residual by cond(C); see test_cannon_m5.jl.)
using Test, MPI, LinearAlgebra, Random, Zygote, Printf
using TeneT
using TeneT: cannon_grid, CannonGrid, cannon_scatter, cannon_gather, split_ranges,
             VUMPS, Plaquette, Square, StructArray, PlaquetteVUMPSRuntime,
             vumps_step, init_VUMPSRuntime_cannon, vumps_step_cannon,
             ALCtoAC, ALCtoAC_cannon, ACCtoAL, ACCtoAL_cannon, ACCtoAL_cannon_gather_ref,
             leftenv, leftenv_cannon, ACenv_plaq, ACenv_plaq_cannon, Cenv_plaq, Cenv_plaq_cannon, qrpos
import ChainRulesCore

MPI.Init()
const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
@assert MPI.Comm_size(comm) == 4 "test_cannon_m5_plaq.jl expects exactly 4 ranks"
say(s) = (rank == 0 && (println(s); flush(stdout)))

scatter_sa(SA, g) = StructArray([cannon_scatter(t, g) for t in SA.data], SA.pattern)
gather_sa(SA, g)  = StructArray([cannon_gather(t, g)  for t in SA.data], SA.pattern)
scatter_rt(rt, g) = PlaquetteVUMPSRuntime(scatter_sa(rt.AL, g), rt.C, scatter_sa(rt.FL, g))   # C replicated
ph_relerr(a, b) = (c = dot(b, a) / dot(b, b); norm(a .- b .* c) / max(norm(b), eps()))

# random (non-canonical) Plaquette runtime + leg5 M; identical on every rank.
function build_rt(χ, D; seed, pat)
    Random.seed!(seed)
    nu = length(unique(pat))
    sa(dims) = StructArray([rand(ComplexF64, dims...) for _ in 1:nu], pat)
    rt = PlaquetteVUMPSRuntime(sa((χ,D,D,χ)), sa((χ,χ)), sa((χ,D,D,χ)))
    M  = StructArray([rand(ComplexF64, D, D, D, D, 2) for _ in 1:nu], pat)
    return rt, M
end
algp(g) = VUMPS{Plaquette{Square}}(grid=g, ifsimple_eig=true, ifupdown=false,
                                   power_iter=1, forloop_iter=1, maxiter=1, maxiter_ad=1, verbosity=0)
const PATS = [[1 3; 2 4], [1 2; 2 1]]   # 2×2 (Plaquette requires (2,2)); 4-unique + repeated

# ── Gate M5p-0: ACCtoAL_cannon seam bit-parity (identical input) ──────────────
@testset "Gate M5p-0: ACCtoAL_cannon bit-parity" begin
    g = cannon_grid(2, 2)
    for (ci, pat) in enumerate(PATS)
        χ, D = 16, 2; nu = length(unique(pat))
        Random.seed!(300 + ci)
        AC = StructArray([rand(ComplexF64, χ, D, D, χ) for _ in 1:nu], pat)
        C  = StructArray([rand(ComplexF64, χ, χ)        for _ in 1:nu], pat)
        ALs, eLs = ACCtoAL(AC, C)
        ALc, eLc = ACCtoAL_cannon_gather_ref(scatter_sa(AC, g), C, g)
        ALcf = gather_sa(ALc, g)
        for idx in 1:nu
            @test maximum(abs, ALcf.data[idx] .- ALs.data[idx]) == 0
        end
        @test eLc == eLs
    end
end

# ── Gate M5p-1: full Plaquette step parity (stable chain) ─────────────────────
@testset "Gate M5p-1: vumps_step_cannon(Plaquette) forward parity" begin
    g = cannon_grid(2, 2)
    for (ci, pat) in enumerate(PATS)
        χ, D = 14, 2
        rt, M = build_rt(χ, D; seed=700 + ci, pat=pat)
        alg_s = VUMPS{Plaquette{Square}}(grid=nothing, ifsimple_eig=true, ifupdown=false,
                                         power_iter=20, forloop_iter=1, maxiter=1, maxiter_ad=1, verbosity=0)
        alg_c = algp(g); alg_c.power_iter = 20
        rtb = scatter_rt(rt, g)
        rt_s2, err_s = vumps_step(rt, M, alg_s)               # serial (grid===nothing)
        rt_c2, err_c = vumps_step_cannon(rtb, M, g, alg_c)
        FLc = gather_sa(rt_c2.FL, g)
        maxFL = maximum(ph_relerr(FLc.data[i], rt_s2.FL.data[i]) for i in 1:length(rt_s2.FL.data))
        maxC  = maximum(ph_relerr(rt_c2.C.data[i], rt_s2.C.data[i]) for i in 1:length(rt_s2.C.data))
        # pre-seam intermediate AC (the seam input): ALCtoAC → leftenv → ACenv_plaq
        AC0s = ALCtoAC(rt.AL, rt.C); _, FLs2 = leftenv(rt.AL, conj(rt.AL), M, rt.FL; alg=alg_s)
        _, ACs = ACenv_plaq(AC0s, FLs2, M; alg=alg_s)
        AC0c = ALCtoAC_cannon(rtb.AL, rtb.C, g); _, FLc2 = leftenv_cannon(rtb.AL, conj(rtb.AL), M, rtb.FL, g; alg=alg_c)
        _, ACc = ACenv_plaq_cannon(AC0c, FLc2, M, g; alg=alg_c)
        ACcf = gather_sa(ACc, g)
        maxAC = maximum(ph_relerr(ACcf.data[i], ACs.data[i]) for i in 1:length(ACs.data))
        say(@sprintf("  [M5p-1 case %d %s] FL=%.1e C=%.1e | seam-input AC=%.1e | err diff=%.1e",
                     ci, string(pat), maxFL, maxC, maxAC, abs(err_c - err_s)))
        @test maxFL ≤ 1e-9
        @test maxC  ≤ 1e-9
        @test maxAC ≤ 1e-9
        @test abs(err_c - err_s) ≤ 1e-9
    end
end

# ── Gate M5p-2: seam gradient parity (well-conditioned C) — R-1 catcher ────────
@testset "Gate M5p-2: ACCtoAL_cannon gradient parity" begin
    g = cannon_grid(2, 2)
    χ, D = 10, 2
    a_rs = split_ranges(χ, g.N1); i_rs = split_ranges(χ, g.N2)
    blkof(t) = t[a_rs[g.r1+1], :, :, i_rs[g.r2+1]]
    for (ci, pat) in enumerate(PATS)
        nu = length(unique(pat))
        Random.seed!(800 + ci)
        AC = StructArray([rand(ComplexF64, χ, D, D, χ) for _ in 1:nu], pat)
        C  = StructArray([Matrix(qr(rand(ComplexF64, χ, χ)).Q) for _ in 1:nu], pat)   # cond≈1
        Random.seed!(850 + ci)
        WAL = [rand(ComplexF64, χ, D, D, χ) for _ in 1:nu]; WALb = [cannon_scatter(W, g) for W in WAL]
        ACb = scatter_sa(AC, g)
        lr(ac, c) = let (al, _) = ACCtoAL(ac, c);        real(sum(sum(conj(WAL[k])  .* al.data[k]) for k in 1:nu)) end
        lc(ac, c) = let (al, _) = ACCtoAL_cannon(ac, c, g); real(sum(sum(conj(WALb[k]) .* al.data[k]) for k in 1:nu)) end
        gr = Zygote.gradient(lr, AC, C); gc = Zygote.gradient(lc, ACb, C)
        eAC = maximum(norm(gc[1].data[k] - blkof(gr[1].data[k])) / max(norm(blkof(gr[1].data[k])), 1e-12) for k in 1:nu)
        eC  = (gr[2] === nothing || gc[2] === nothing) ? 0.0 :
              maximum(norm(gc[2].data[k] - gr[2].data[k]) / max(norm(gr[2].data[k]), 1e-12) for k in 1:nu)
        say(@sprintf("  [M5p-2 case %d %s] ACCtoAL dAC=%.1e dC=%.1e", ci, string(pat), eAC, eC))
        @test eAC ≤ 1e-7
        @test eC  ≤ 1e-7
    end
end

# ── Gate M5p-3: init_VUMPSRuntime_cannon(Plaquette) cross-rank consistency ─────
@testset "Gate M5p-3: init(Plaquette) cross-rank consistency" begin
    g = cannon_grid(2, 2)
    χ, D = 8, 2
    Random.seed!(9000)
    M = StructArray([rand(ComplexF64, D, D, D, D, 2) for _ in 1:4], [1 3; 2 4])   # 2×2, 4 unique
    for _ in 1:rank; rand(ComplexF64); end          # diverge per-rank RNG → bcast load-bearing
    rt = init_VUMPSRuntime_cannon(M, χ, g, algp(g))
    for (full, nm) in [(gather_sa(rt.AL, g), "AL"), (gather_sa(rt.FL, g), "FL"), (rt.C, "C")]
        for idx in 1:length(full.data)
            ref = MPI.bcast(full.data[idx], 0, comm)
            @test maximum(abs, full.data[idx] .- ref) == 0
        end
    end
end

# ── Gate M5p-4: routing via alg.grid ──────────────────────────────────────────
@testset "Gate M5p-4: vumps_step(Plaquette) routes to cannon" begin
    g = cannon_grid(2, 2)
    rt, M = build_rt(14, 2; seed=9500, pat=[1 3; 2 4])
    alg_c = algp(g); alg_c.power_iter = 15
    rtb = scatter_rt(rt, g)
    via, e1 = vumps_step(rtb, M, alg_c)                       # routes to cannon
    direct, e2 = vumps_step_cannon(rtb, M, g, alg_c)
    @test e1 ≈ e2
    for idx in 1:length(direct.AL.data)
        @test direct.AL.data[idx] == via.AL.data[idx]
    end
end

say("all M5-plaq gates done.")
