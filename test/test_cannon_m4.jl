# M4 Batches A–D: distributed leftenv/rightenv/ACenv/Cenv _cannon (gather hoisting).
# Run via: julia --project=. test/run_test_cannon_m4.jl   (spawns 4 ranks)
# CPU Arrays only (no CUDA-aware MPI locally); GPU = a Sofia driver (M4 Gate 5).
# Design: docs/2026-06-15-m4-env-cannon-integration-design.md
using Test
using MPI
using LinearAlgebra
using Random
using Zygote
using TeneT
using TeneT: cannon_grid, CannonGrid, cannon_scatter, cannon_gather,
             cannon_dot, cannon_norm, cannon_gather_row, cannon_gather_col,
             orth_for_ad, orth_for_ad_cannon, split_ranges,
             FLmap_cannon_dist, FLmap_cannon_sliced,
             FRmap_cannon_dist, FRmap_cannon_sliced,
             ACmap_cannon_dist, ACmap_cannon_sliced,
             leftenv, leftenv_cannon, rightenv, rightenv_cannon,
             ACenv, ACenv_cannon, Cenv, Cenv_cannon,
             VUMPS, General, StructArray, Recompute
import ChainRulesCore

MPI.Init()
const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
@assert MPI.Comm_size(comm) == 4 "test_cannon_m4.jl expects exactly 4 ranks"

# ── helpers ──────────────────────────────────────────────────────────────────
# Identical full tensors on every rank (fixed seed), then scatter to blocks.
# Distinguishes the UNIT cell (Ni×Nj, the .pattern) from the PROCESS grid (N1×N2).
function build_cell(Ni, Nj, χ, D; d=2, seed=42, pattern=nothing)
    Random.seed!(seed)
    pat = pattern === nothing ? reshape(collect(1:Ni*Nj), Ni, Nj) : pattern
    nuniq = length(unique(pat))
    ALu = StructArray([rand(ComplexF64, χ, D, D, χ) for _ in 1:nuniq], pat)
    ALd = StructArray([rand(ComplexF64, χ, D, D, χ) for _ in 1:nuniq], pat)
    M   = StructArray([rand(ComplexF64, D, D, D, D, d) for _ in 1:nuniq], pat)
    FL  = StructArray([rand(ComplexF64, χ, D, D, χ) for _ in 1:nuniq], pat)
    return ALu, ALd, M, FL
end
scatter_sa(SA, g) = StructArray([cannon_scatter(t, g) for t in SA.data], SA.pattern)
gather_sa(SA, g)  = StructArray([cannon_gather(t, g) for t in SA.data], SA.pattern)
function cannon_forward_sliced_body()
    src = read(joinpath(@__DIR__, "..", "src", "contraction", "cannon_2d.jl"), String)
    sig = "function _cannon_forward_sliced("
    start = findfirst(sig, src)
    @test start !== nothing
    tail = src[last(start):end]
    stop = findfirst("# Replicated-AL path", tail)
    @test stop !== nothing
    return tail[1:first(stop)-1]
end
function flmap_sliced_rrule_body()
    src = read(joinpath(@__DIR__, "..", "src", "autodiff", "rules.jl"), String)
    sig = "function ChainRulesCore.rrule(::typeof(FLmap_cannon_sliced)"
    start = findfirst(sig, src)
    @test start !== nothing
    tail = src[last(start):end]
    stop = findfirst("# M4 hoisted FRmap rrule", tail)
    @test stop !== nothing
    return tail[1:first(stop)-1]
end

# global-phase-insensitive eigenvector comparison (test_cannon.jl idiom)
function phase_ok(a_full, b_full; rtol=1e-6)
    an = a_full ./ norm(a_full)
    bn = b_full ./ norm(b_full)
    imax = argmax(abs.(bn))
    ph = an[imax] / bn[imax]
    return isapprox(abs(ph), 1; rtol) && isapprox(an, bn .* ph; rtol)
end

@testset "Gate 0: FLmap sliced uses row allgather for FL iterate" begin
    body = cannon_forward_sliced_body()
    @test occursin("FL_row = cannon_gather_row", body)
    @test occursin("ALd_chunk = view(ALd_col, :, :, :, ch)", body)
    @test occursin("Hc = _cannon_stage1(FL_row, ALd_chunk)", body)
    @test !occursin("_cannon_row_shift", body)
    @test !occursin("for t in 0:N2-1", body)

    rbody = flmap_sliced_rrule_body()
    @test occursin("ALd_chunk = view(ALd_col, :, :, :, ch)", rbody)
    @test occursin("dFL_row .+= tmp", rbody)
    @test occursin("view(dALd_col, :, :, :, ch) .+= tmp", rbody)
    @test !occursin("for t in 0:N2-1", rbody)
end

# ── Gate 1: env-level eigenpair parity (the headline M4 gate) ─────────────────
@testset "Gate 1: leftenv_cannon eigenpair parity" begin
    g = cannon_grid(2, 2)
    # (Ni, Nj, pattern): all-distinct cells + a REPEATED pattern (2 unique in a 2×2 cell)
    # so the processed_indices dedup + break + cross-row FL' reuse are exercised (R1-m9).
    # χ ∈ {16,18}: 18 forces UNEVEN blocks (d_rs=[5,4] on N=2) through the hoisted gather
    # wrappers + their reduce-scatter adjoints (the env layer's only uneven-block coverage).
    cases = [(1, 1, nothing), (1, 2, nothing), (2, 2, nothing), (2, 2, [1 2; 2 1])]
    for (ci, (Ni, Nj, pat)) in enumerate(cases), χ in (16, 18)
        D = 3
        ALu, ALd, M, FL = build_cell(Ni, Nj, χ, D; seed=4000 + 1000ci + χ, pattern=pat)
        alg = VUMPS(General(); ifsimple_eig=true, power_iter=40, forloop_iter=1, verbosity=0)
        λref, FLref = leftenv(ALu, ALd, M, FL; alg)                  # serial reference
        ALu_b, ALd_b, FL_b = scatter_sa(ALu, g), scatter_sa(ALd, g), scatter_sa(FL, g)
        λc, FLc = leftenv_cannon(ALu_b, ALd_b, M, FL_b, g; alg)      # distributed, hoisted
        FLc_full = gather_sa(FLc, g)
        for idx in 1:length(FLref.data)
            @test λc.data[idx] ≈ λref.data[idx] rtol=1e-8
            @test phase_ok(FLc_full.data[idx], FLref.data[idx]; rtol=1e-8)
        end
    end
end

# ── Gate 2: hoisting equivalence (sliced map == per-call dist map) ────────────
# Bitwise-ish: both call _cannon_forward_sliced with the SAME gathered slices, so
# the reduction order is identical (CPU/MPI same backend). Catches stale-slice /
# wrong-slice bugs the existing tests cannot. Includes a K-iteration loop with the
# hoisted slices held fixed, and an Nj>1-style reuse (a second column's slice).
@testset "Gate 2: hoisting equivalence (sliced == dist)" begin
    g = cannon_grid(2, 2)
    χ, D = 16, 3
    Random.seed!(7000)
    FL  = rand(ComplexF64, χ, D, D, χ)
    ALu = rand(ComplexF64, χ, D, D, χ)
    ALd = rand(ComplexF64, χ, D, D, χ)
    M   = rand(ComplexF64, D, D, D, D, 2)
    FLb, ALub, ALdb = cannon_scatter(FL, g), cannon_scatter(ALu, g), cannon_scatter(ALd, g)
    χg = MPI.Allreduce(size(ALub, 1), +, g.col_comm)
    a_rs = split_ranges(χg, g.N1); l_rs = split_ranges(χg, g.N2)
    ALu_row = cannon_gather_row(ALub, g, l_rs)
    ALd_col = cannon_gather_col(ALdb, g, a_rs)
    # single application: sliced (pre-gathered) == dist (gathers internally)
    w_dist   = FLmap_cannon_dist(FLb, ALub, ALdb, M, g)
    w_sliced = FLmap_cannon_sliced(FLb, ALu_row, ALd_col, M, g)
    @test w_sliced ≈ w_dist rtol=1e-13
    # K iterations with the hoisted slices held FIXED (the leftenv power loop)
    v_d, v_s = FLb, FLb
    for _ in 1:5
        v_d = FLmap_cannon_dist(v_d, ALub, ALdb, M, g)
        v_s = FLmap_cannon_sliced(v_s, ALu_row, ALd_col, M, g)
    end
    @test v_s ≈ v_d rtol=1e-12
    # AD equivalence (single application): differentiating the HOIST path w.r.t the BLOCK
    # inputs (gather wrappers compose their reduce-scatter adjoints) must match the per-call
    # dist gradient exactly — pins sliced-rrule + cannon_gather_row/col rrules == dist rrule
    # under AD, before Batch E introduces buffer reuse (R1-M8).
    Random.seed!(7001); Wb2 = cannon_scatter(rand(ComplexF64, χ, D, D, χ), g)
    ld(fl, au, ad) = real(sum(conj(Wb2) .* FLmap_cannon_dist(fl, au, ad, M, g)))
    lh(fl, au, ad) = real(sum(conj(Wb2) .* FLmap_cannon_sliced(fl,
                          cannon_gather_row(au, g, l_rs), cannon_gather_col(ad, g, a_rs), M, g)))
    gd = Zygote.gradient(ld, FLb, ALub, ALdb)
    gh = Zygote.gradient(lh, FLb, ALub, ALdb)
    for k in 1:3
        @test gh[k] ≈ gd[k] rtol=1e-10
    end
end

# ── Gate 3a: the new rrules vs the trusted serial/ChainRules rrule (COMPLEX) ──
# Direct rrule-API comparison with COMPLEX cotangents (real inputs would hide the
# conj / real-projection bugs). Compares cannon_dot/cannon_norm/orth_for_ad_cannon
# (on blocks) to dot/norm/orth_for_ad on the FULL tensor, block-restricted.
@testset "Gate 3a: cannon_dot/norm/orth rrules (complex)" begin
    g = cannon_grid(2, 2)
    χ, D = 16, 3
    Random.seed!(8000)
    x  = rand(ComplexF64, χ, D, D, χ); y  = rand(ComplexF64, χ, D, D, χ)
    dv = rand(ComplexF64, χ, D, D, χ)
    a_rs = split_ranges(χ, g.N1); i_rs = split_ranges(χ, g.N2)
    blkof(t) = t[a_rs[g.r1+1], :, :, i_rs[g.r2+1]]
    xb, yb, dvb = cannon_scatter(x, g), cannon_scatter(y, g), cannon_scatter(dv, g)

    # cannon_dot: dx = y·conj(ds), dy = x·ds  (ChainRules dot convention; ds complex)
    ds = 0.6 + 0.5im
    s, dot_back = ChainRulesCore.rrule(cannon_dot, xb, yb, g)
    @test s ≈ dot(x, y) rtol=1e-12
    _, dx_blk, dy_blk, _ = dot_back(ds)
    @test dx_blk ≈ blkof(y .* conj(ds)) rtol=1e-12
    @test dy_blk ≈ blkof(x .* ds)        rtol=1e-12

    # cannon_norm (normalization denominator): the incoming `dn` is each rank's PIECE
    # (from the broadcast v_blk/n division), so the rrule allreduces it → the true scalar
    # cotangent c̄ = Σ_r dn_r, then dx = real(c̄)·x/n (GLOBAL n, real-projected, complex c̄).
    # Pass DISTINCT per-rank pieces and check the SUM is used (validates the allreduce).
    dn_r = (rank + 1) * (0.3 + 0.8im)
    dn_sum = sum((r + 1) * (0.3 + 0.8im) for r in 0:3)
    n, norm_back = ChainRulesCore.rrule(cannon_norm, xb, g)
    @test n ≈ norm(x) rtol=1e-12
    _, dx_n, _ = norm_back(dn_r)
    @test dx_n ≈ blkof(x .* (real(dn_sum) / norm(x))) rtol=1e-12

    # orth_for_ad_cannon: dv - cannon_dot(v,dv)·v  == blkof of serial orth_for_ad
    _, oc_back = ChainRulesCore.rrule(orth_for_ad_cannon, xb, g)
    _, dvb_out, _ = oc_back(dvb)
    _, os_back = ChainRulesCore.rrule(orth_for_ad, x)
    _, dv_full_thunk = os_back(dv)
    dv_full = ChainRulesCore.unthunk(dv_full_thunk)
    @test dvb_out ≈ blkof(dv_full) rtol=1e-10
end

# ── Gate 3b: leftenv_cannon gradient parity vs serial (in-situ, all rrules) ───
# Per-rank weighted-sum loss on the block-distributed output (disjoint tiles), the
# established cannon-gradient pattern. Validates the WHOLE AD chain in composition:
# gather wrappers + sliced-map rrule + cannon_dot/norm/orth rrules. Small power_iter
# (gradient parity needs the SAME finite computation, not convergence).
@testset "Gate 3b: leftenv_cannon gradient parity" begin
    g = cannon_grid(2, 2)
    χ, D = 12, 2
    a_rs = split_ranges(χ, g.N1); i_rs = split_ranges(χ, g.N2)
    blkof(t) = t[a_rs[g.r1+1], :, :, i_rs[g.r2+1]]
    # configs: (Ni, Nj, pattern, mode). :recompute uses eig_checkpoint=Recompute() +
    # segment_checkpoint=Recompute() with power_iter=8 > checkpoint_every(5) so BOTH the
    # per-row eig recompute AND the segment-loop recompute fire — pinning R1-M2 (the hoisted
    # gathers, lexically OUTSIDE the checkpoint, fire once on the outer tape under recompute).
    # The (2,2,[1 2;2 1]) case differentiates the j=2:Nj cycle + ir down-partner + dedup.
    configs = [(1, 1, nothing, :plain),
               (1, 1, nothing, :recompute),
               (2, 2, [1 2; 2 1], :plain)]
    for (ci, (Ni, Nj, pat, mode)) in enumerate(configs)
        ALu, ALd, M, FL = build_cell(Ni, Nj, χ, D; seed=9100 + 100ci, pattern=pat)
        nu = length(M.data)
        Random.seed!(9200 + ci); Ws = [rand(ComplexF64, χ, D, D, χ) for _ in 1:nu]
        Wbs = [cannon_scatter(W, g) for W in Ws]
        alg = mode === :recompute ?
            VUMPS(General(); ifsimple_eig=true, eig_checkpoint=Recompute(),
                  segment_checkpoint=Recompute(), power_iter=8, forloop_iter=1, verbosity=0) :
            VUMPS(General(); ifsimple_eig=true, power_iter=4, forloop_iter=1, verbosity=0)
        ALu_b, ALd_b, FL_b = scatter_sa(ALu, g), scatter_sa(ALd, g), scatter_sa(FL, g)
        loss_ref(au, ad, fl) = real(sum(sum(conj(Ws[c])  .* leftenv(au, ad, M, fl; alg)[2].data[c])        for c in 1:nu))
        loss_c(au, ad, fl)   = real(sum(sum(conj(Wbs[c]) .* leftenv_cannon(au, ad, M, fl, g; alg)[2].data[c]) for c in 1:nu))
        gref = Zygote.gradient(loss_ref, ALu, ALd, FL)
        gc   = Zygote.gradient(loss_c, ALu_b, ALd_b, FL_b)
        for k in 1:3, c in 1:nu
            # A non-leading cell's FL input is unused (FL'[i,j] comes from FL'[i,j-1], not
            # FL[i,j]) → Zygote returns `nothing` for that .data entry on BOTH paths. atol
            # covers near-zero weakly-coupled gradients (rtol alone is noise at ~1e-8).
            rc = gref[k] === nothing ? nothing : gref[k].data[c]
            cc = gc[k]   === nothing ? nothing : gc[k].data[c]
            if rc === nothing
                @test cc === nothing || maximum(abs, cc) < 1e-9
            else
                @test isapprox(cc, blkof(rc); rtol=1e-8, atol=1e-10)
            end
        end
    end
end

# ── Batch B: rightenv_cannon ─────────────────────────────────────────────────
@testset "Gate B1: rightenv_cannon eigenpair parity" begin
    g = cannon_grid(2, 2)
    cases = [(1, 1, nothing), (1, 2, nothing), (2, 2, nothing), (2, 2, [1 2; 2 1])]
    for (ci, (Ni, Nj, pat)) in enumerate(cases), χ in (16, 18)
        D = 3
        ARu, ARd, M, FR = build_cell(Ni, Nj, χ, D; seed=5000 + 1000ci + χ, pattern=pat)
        alg = VUMPS(General(); ifsimple_eig=true, power_iter=40, forloop_iter=1, verbosity=0)
        λref, FRref = rightenv(ARu, ARd, M, FR; alg)
        ARu_b, ARd_b, FR_b = scatter_sa(ARu, g), scatter_sa(ARd, g), scatter_sa(FR, g)
        λc, FRc = rightenv_cannon(ARu_b, ARd_b, M, FR_b, g; alg)
        FRc_full = gather_sa(FRc, g)
        for idx in 1:length(FRref.data)
            @test λc.data[idx] ≈ λref.data[idx] rtol=1e-8
            @test phase_ok(FRc_full.data[idx], FRref.data[idx]; rtol=1e-8)
        end
    end
end

# sliced==dist hoisting equivalence (Gate B2/C2) — the cheapest pin of the per-call
# iterate-gather surgery (a NEW path FLmap lacks). Mirrors Gate 2.
@testset "Gate B2: FRmap hoisting equivalence (sliced == dist)" begin
    g = cannon_grid(2, 2); χ, D = 16, 3
    Random.seed!(5700)
    FR  = rand(ComplexF64, χ, D, D, χ); ARu = rand(ComplexF64, χ, D, D, χ); ARd = rand(ComplexF64, χ, D, D, χ)
    M   = rand(ComplexF64, D, D, D, D, 2)
    FRb, ARub, ARdb = cannon_scatter(FR, g), cannon_scatter(ARu, g), cannon_scatter(ARd, g)
    χg = MPI.Allreduce(size(ARdb, 1), +, g.col_comm); p_rs = split_ranges(χg, g.N1)
    ARu_g = cannon_gather_row(ARub, g, p_rs); ARd_g = cannon_gather_col(ARdb, g, p_rs)
    @test FRmap_cannon_sliced(FRb, ARu_g, ARd_g, M, g) ≈ FRmap_cannon_dist(FRb, ARub, ARdb, M, g) rtol=1e-13
    v_d, v_s = FRb, FRb
    for _ in 1:5
        v_d = FRmap_cannon_dist(v_d, ARub, ARdb, M, g)
        v_s = FRmap_cannon_sliced(v_s, ARu_g, ARd_g, M, g)
    end
    @test v_s ≈ v_d rtol=1e-12
    Random.seed!(5701); Wb = cannon_scatter(rand(ComplexF64, χ, D, D, χ), g)
    ld(fr, au, ad) = real(sum(conj(Wb) .* FRmap_cannon_dist(fr, au, ad, M, g)))
    lh(fr, au, ad) = real(sum(conj(Wb) .* FRmap_cannon_sliced(fr, cannon_gather_row(au, g, p_rs), cannon_gather_col(ad, g, p_rs), M, g)))
    gd = Zygote.gradient(ld, FRb, ARub, ARdb); gh = Zygote.gradient(lh, FRb, ARub, ARdb)
    for k in 1:3; @test gh[k] ≈ gd[k] rtol=1e-10; end
end

@testset "Gate B3: rightenv_cannon gradient parity" begin
    g = cannon_grid(2, 2); χ, D = 12, 2
    a_rs = split_ranges(χ, g.N1); i_rs = split_ranges(χ, g.N2)
    blkof(t) = t[a_rs[g.r1+1], :, :, i_rs[g.r2+1]]
    # Plain 1×1, Recompute() 1×1 (R1-M2: gather-once + iterate-gather survive recompute),
    # multi-cell 2×2 dedup (differentiate the j=Nj-1:-1:1 sliced cycle + ir + dedup under AD).
    configs = [(1, 1, nothing, :plain), (1, 1, nothing, :recompute), (2, 2, [1 2; 2 1], :plain)]
    for (ci, (Ni, Nj, pat, mode)) in enumerate(configs)
        ARu, ARd, M, FR = build_cell(Ni, Nj, χ, D; seed=5900 + 100ci, pattern=pat)
        nu = length(M.data)
        Random.seed!(5950 + ci); Ws = [rand(ComplexF64, χ, D, D, χ) for _ in 1:nu]
        Wbs = [cannon_scatter(W, g) for W in Ws]
        alg = mode === :recompute ?
            VUMPS(General(); ifsimple_eig=true, eig_checkpoint=Recompute(),
                  segment_checkpoint=Recompute(), power_iter=8, forloop_iter=1, verbosity=0) :
            VUMPS(General(); ifsimple_eig=true, power_iter=4, forloop_iter=1, verbosity=0)
        ARu_b, ARd_b, FR_b = scatter_sa(ARu, g), scatter_sa(ARd, g), scatter_sa(FR, g)
        loss_ref(au, ad, fr) = real(sum(sum(conj(Ws[c])  .* rightenv(au, ad, M, fr; alg)[2].data[c])        for c in 1:nu))
        loss_c(au, ad, fr)   = real(sum(sum(conj(Wbs[c]) .* rightenv_cannon(au, ad, M, fr, g; alg)[2].data[c]) for c in 1:nu))
        gref = Zygote.gradient(loss_ref, ARu, ARd, FR)
        gc   = Zygote.gradient(loss_c, ARu_b, ARd_b, FR_b)
        for k in 1:3, c in 1:nu
            rc = gref[k] === nothing ? nothing : gref[k].data[c]
            cc = gc[k]   === nothing ? nothing : gc[k].data[c]
            if rc === nothing
                @test cc === nothing || maximum(abs, cc) < 1e-9
            else
                @test isapprox(cc, blkof(rc); rtol=1e-8, atol=1e-10)
            end
        end
    end
end

# ── Batch C: ACenv_cannon (axis-transposed; non-leading cells normalize) ─────
@testset "Gate C1: ACenv_cannon eigenpair parity" begin
    g = cannon_grid(2, 2)
    # include Ni≠Nj (2×1, 1×2) to exercise the transposed cell-row chain + the
    # cannon_norm normalize on non-leading (i≥2) cells.
    cases = [(1, 1, nothing), (2, 1, nothing), (1, 2, nothing), (2, 2, nothing), (2, 2, [1 2; 2 1])]
    for (ci, (Ni, Nj, pat)) in enumerate(cases), χ in (16, 18)
        D = 3
        AC, FL, M, FR = build_cell(Ni, Nj, χ, D; seed=6000 + 1000ci + χ, pattern=pat)
        alg = VUMPS(General(); ifsimple_eig=true, power_iter=40, forloop_iter=1, verbosity=0)
        λref, ACref = ACenv(AC, FL, M, FR; alg)
        AC_b, FL_b, FR_b = scatter_sa(AC, g), scatter_sa(FL, g), scatter_sa(FR, g)
        λc, ACc = ACenv_cannon(AC_b, FL_b, M, FR_b, g; alg)
        ACc_full = gather_sa(ACc, g)
        for idx in 1:length(ACref.data)
            @test λc.data[idx] ≈ λref.data[idx] rtol=1e-8
            @test phase_ok(ACc_full.data[idx], ACref.data[idx]; rtol=1e-8)
        end
    end
end

@testset "Gate C2: ACmap hoisting equivalence (sliced == dist)" begin
    g = cannon_grid(2, 2); χ, D = 16, 3
    Random.seed!(6700)
    AC  = rand(ComplexF64, χ, D, D, χ); FL = rand(ComplexF64, χ, D, D, χ); FR = rand(ComplexF64, χ, D, D, χ)
    M   = rand(ComplexF64, D, D, D, D, 2)
    ACb, FLb, FRb = cannon_scatter(AC, g), cannon_scatter(FL, g), cannon_scatter(FR, g)
    χg = MPI.Allreduce(size(ACb, 1), +, g.col_comm); p_rs = split_ranges(χg, g.N1)
    FL_g = cannon_gather_row(FLb, g, p_rs); FR_g = cannon_gather_col(FRb, g, p_rs)
    @test ACmap_cannon_sliced(ACb, FL_g, FR_g, M, g) ≈ ACmap_cannon_dist(ACb, FLb, FRb, M, g) rtol=1e-13
    v_d, v_s = ACb, ACb
    for _ in 1:5
        v_d = ACmap_cannon_dist(v_d, FLb, FRb, M, g)
        v_s = ACmap_cannon_sliced(v_s, FL_g, FR_g, M, g)
    end
    @test v_s ≈ v_d rtol=1e-12
    Random.seed!(6701); Wb = cannon_scatter(rand(ComplexF64, χ, D, D, χ), g)
    ld(ac, fl, fr) = real(sum(conj(Wb) .* ACmap_cannon_dist(ac, fl, fr, M, g)))
    lh(ac, fl, fr) = real(sum(conj(Wb) .* ACmap_cannon_sliced(ac, cannon_gather_row(fl, g, p_rs), cannon_gather_col(fr, g, p_rs), M, g)))
    gd = Zygote.gradient(ld, ACb, FLb, FRb); gh = Zygote.gradient(lh, ACb, FLb, FRb)
    for k in 1:3; @test gh[k] ≈ gd[k] rtol=1e-10; end
end

@testset "Gate C3: ACenv_cannon gradient parity" begin
    g = cannon_grid(2, 2)
    χ, D = 12, 2
    a_rs = split_ranges(χ, g.N1); i_rs = split_ranges(χ, g.N2)
    blkof(t) = t[a_rs[g.r1+1], :, :, i_rs[g.r2+1]]
    # 1×1 (no normalize), 2×1 (i=2 cell uses cannon_norm — differentiate through it),
    # 1×1 Recompute() (iterate-gather + recompute interplay, R1-M2).
    for (ci, (Ni, Nj, mode)) in enumerate(((1, 1, :plain), (2, 1, :plain), (1, 1, :recompute)))
        AC, FL, M, FR = build_cell(Ni, Nj, χ, D; seed=6900 + 100ci)
        nu = Ni * Nj
        Random.seed!(6950 + ci); Ws = [rand(ComplexF64, χ, D, D, χ) for _ in 1:nu]
        Wbs = [cannon_scatter(W, g) for W in Ws]
        alg = mode === :recompute ?
            VUMPS(General(); ifsimple_eig=true, eig_checkpoint=Recompute(),
                  segment_checkpoint=Recompute(), power_iter=8, forloop_iter=1, verbosity=0) :
            VUMPS(General(); ifsimple_eig=true, power_iter=4, forloop_iter=1, verbosity=0)
        AC_b, FL_b, FR_b = scatter_sa(AC, g), scatter_sa(FL, g), scatter_sa(FR, g)
        loss_ref(ac, fl, fr) = real(sum(sum(conj(Ws[c])  .* ACenv(ac, fl, M, fr; alg)[2].data[c])        for c in 1:nu))
        loss_c(ac, fl, fr)   = real(sum(sum(conj(Wbs[c]) .* ACenv_cannon(ac, fl, M, fr, g; alg)[2].data[c]) for c in 1:nu))
        gref = Zygote.gradient(loss_ref, AC, FL, FR)
        gc   = Zygote.gradient(loss_c, AC_b, FL_b, FR_b)
        for k in 1:3, c in 1:nu
            rc = gref[k] === nothing ? nothing : gref[k].data[c]
            cc = gc[k]   === nothing ? nothing : gc[k].data[c]
            if rc === nothing
                @test cc === nothing || maximum(abs, cc) < 1e-9
            else
                @test isapprox(cc, blkof(rc); rtol=1e-8, atol=1e-10)
            end
        end
    end
end

# ── Batch D: Cenv_cannon (replicated-output; C full χ×χ, FL/FR gathered) ─────
# C is REPLICATED (not block-distributed) → compare it DIRECTLY (no cannon_gather,
# R1-M7). The gather adjoint for FL/FR is take-my-block (replicated downstream).
function build_C_cell(Ni, Nj, χ, D; seed, pattern=nothing)
    Random.seed!(seed)
    pat = pattern === nothing ? reshape(collect(1:Ni*Nj), Ni, Nj) : pattern
    nu = length(unique(pat))
    C  = StructArray([rand(ComplexF64, χ, χ) for _ in 1:nu], pat)
    FL = StructArray([rand(ComplexF64, χ, D, D, χ) for _ in 1:nu], pat)
    FR = StructArray([rand(ComplexF64, χ, D, D, χ) for _ in 1:nu], pat)
    return C, FL, FR
end

@testset "Gate D1: Cenv_cannon eigenpair parity" begin
    g = cannon_grid(2, 2)
    cases = [(1, 1, nothing), (1, 2, nothing), (2, 2, nothing), (2, 2, [1 2; 2 1])]
    for (ci, (Ni, Nj, pat)) in enumerate(cases), χ in (16, 18)
        D = 3
        C, FL, FR = build_C_cell(Ni, Nj, χ, D; seed=7000 + 1000ci + χ, pattern=pat)
        alg = VUMPS(General(); ifsimple_eig=true, power_iter=40, verbosity=0)
        λref, Cref = Cenv(C, FL, FR; alg)
        FL_b, FR_b = scatter_sa(FL, g), scatter_sa(FR, g)
        λc, Cc = Cenv_cannon(C, FL_b, FR_b, g; alg)    # C replicated (NOT scattered)
        for idx in 1:length(Cref.data)
            @test λc.data[idx] ≈ λref.data[idx] rtol=1e-8
            @test phase_ok(Cc.data[idx], Cref.data[idx]; rtol=1e-8)   # C replicated → no gather
        end
    end
end

@testset "Gate D3: Cenv_cannon gradient parity (take-my-block)" begin
    g = cannon_grid(2, 2)
    χ, D = 12, 2
    a_rs = split_ranges(χ, g.N1); i_rs = split_ranges(χ, g.N2)
    blkof(t) = t[a_rs[g.r1+1], :, :, i_rs[g.r2+1]]
    C, FL, FR = build_C_cell(1, 1, χ, D; seed=7900)
    Random.seed!(7901); W = rand(ComplexF64, χ, χ)         # FULL weight (replicated output)
    alg = VUMPS(General(); ifsimple_eig=true, power_iter=4, verbosity=0)
    FL_b, FR_b = scatter_sa(FL, g), scatter_sa(FR, g)
    loss_ref(c, fl, fr) = real(sum(conj(W) .* Cenv(c, fl, fr; alg)[2].data[1]))
    loss_c(c, fl, fr)   = real(sum(conj(W) .* Cenv_cannon(c, fl, fr, g; alg)[2].data[1]))
    gref = Zygote.gradient(loss_ref, C, FL, FR)
    gc   = Zygote.gradient(loss_c, C, FL_b, FR_b)
    @test isapprox(gc[1].data[1], gref[1].data[1]; rtol=1e-8, atol=1e-10)            # dC full (replicated)
    @test isapprox(gc[2].data[1], blkof(gref[2].data[1]); rtol=1e-8, atol=1e-10)     # dFL block (take-my-block)
    @test isapprox(gc[3].data[1], blkof(gref[3].data[1]); rtol=1e-8, atol=1e-10)     # dFR block
end

# ── Batch E: steady-state allocation (the "zero-allocation steady-state" gate) ──
# R1-M11: gross @allocated is meaningless on its own (it counts every transient,
# even freed+recycled ones). The MEANINGFUL gate isolates the LEVER (hoisting) and
# the STEADY STATE (no growth): (1) the hoisted sliced map allocates strictly LESS
# per power step than the per-call dist map — because the FIXED-operand gathers
# (χ²D²/N each) are pulled OUT of the loop; (2) the per-iteration allocation is FLAT
# across iterations (no upward growth → no leak/accumulation; the chain transients
# are eagerly freed + allocator-recycled). The remaining per-call comm buffers
# (iterate gather + output reduce-scatter + ring) are a marginal further target for
# the persistent-buffer pool (deferred — pool-recycled, comm ≈5-11% per benchmarks).
@testset "Gate E: hoisting reduces + flattens per-iteration allocation" begin
    g = cannon_grid(2, 2); χ, D = 32, 2
    Random.seed!(8500)
    FL  = rand(ComplexF64, χ, D, D, χ); ALu = rand(ComplexF64, χ, D, D, χ); ALd = rand(ComplexF64, χ, D, D, χ)
    M   = rand(ComplexF64, D, D, D, D, 2)
    FLb, ALub, ALdb = cannon_scatter(FL, g), cannon_scatter(ALu, g), cannon_scatter(ALd, g)
    χg = MPI.Allreduce(size(ALub, 1), +, g.col_comm); a_rs = split_ranges(χg, g.N1); l_rs = split_ranges(χg, g.N2)
    ALu_row = cannon_gather_row(ALub, g, l_rs); ALd_col = cannon_gather_col(ALdb, g, a_rs)
    fh(v) = FLmap_cannon_sliced(v, ALu_row, ALd_col, M, g)   # hoisted: gathers OUTSIDE the loop
    fd(v) = FLmap_cannon_dist(v, ALub, ALdb, M, g)           # per-call: re-gathers ALu/ALd each step
    fh(FLb); fd(FLb)                                         # warm up (compile)
    ah = minimum([@allocated(fh(FLb)) for _ in 1:5])
    ad = minimum([@allocated(fd(FLb)) for _ in 1:5])
    @test ah < ad                                           # hoisting eliminates the per-step gathers
    # steady state: per-iter allocation is flat (deterministic per call after warmup → no growth)
    iters = [@allocated(fh(FLb)) for _ in 1:6]
    @test maximum(iters) <= minimum(iters) + minimum(iters) ÷ 20   # within ~5% (no upward drift)
    rank == 0 && println("[Gate E] hoisted=$(ah÷1024)KB/iter  dist=$(ad÷1024)KB/iter  saved=$((ad-ah)÷1024)KB/iter (the hoisted gathers)")
end

MPI.Finalize()
