# Slice2D-style 2D distributed FLmap tests. Run via: julia --project=. test/run_test_slice2d.jl
# CPU Arrays only (no CUDA-aware MPI on local machines); GPU is covered by
# examples/MPI_parallel/validation/test_slice2d_sofia.jl.
using Test
using MPI
using LinearAlgebra
using Random
using Zygote
using TeneT
using TeneT: slice2d_grid, Slice2DGrid, slice2d_scatter, slice2d_gather, slice2d_dot, slice2d_norm, simple_eig, FLmap, FLmap_slice2d, FLmap_slice2d_dist, split_ranges, _slice2d_stage1, _slice2d_stage1_add!, _slice2d_fold, _slice2d_fold1, _slice2d_fold2, _slice2d_stage2, _slice2d_stage1_dFL, _slice2d_stage1_dALd, _slice2d_fold1_dH, _slice2d_fold1_dM1, _slice2d_fold2_dT, _slice2d_fold2_dM2, _slice2d_stage2_dG, _slice2d_stage2_dALu

MPI.Init()
const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
@assert MPI.Comm_size(comm) == 4 "test_slice2d.jl expects exactly 4 ranks"

# Identical tensors on every rank: fixed seed immediately before each rand group.
function make_leg5(χ, D; d=2, seed=42)
    Random.seed!(seed)
    FL  = rand(ComplexF64, χ, D, D, χ)
    ALu = rand(ComplexF64, χ, D, D, χ)
    ALd = rand(ComplexF64, χ, D, D, χ)
    M1  = rand(ComplexF64, D, D, D, D, d)
    M2  = rand(ComplexF64, D, D, D, D, d)
    W   = rand(ComplexF64, χ, D, D, χ)     # fixed weight for scalar losses
    return FL, ALu, ALd, M1, M2, W
end

@testset "slice2d_grid" begin
    for (N1, N2) in ((2, 2), (1, 4), (4, 1))
        g = slice2d_grid(N1, N2)
        @test g isa Slice2DGrid
        @test g.r1 * N2 + g.r2 == rank
        @test MPI.Comm_size(g.row_comm) == N2
        @test MPI.Comm_size(g.col_comm) == N1
        @test MPI.Comm_rank(g.row_comm) == g.r2
        @test MPI.Comm_rank(g.col_comm) == g.r1
    end
end

@testset "stage kernels == FLmap (local, no MPI)" begin
    χ, D = 8, 3
    FL, ALu, ALd, M1, M2, _ = make_leg5(χ, D; seed=101)
    H = _slice2d_stage1(FL, ALd)
    G = _slice2d_fold(H, M1, M2)
    P = _slice2d_stage2(G, ALu)
    ref = FLmap(FL, ALu, ALd, M1, M2)
    @test P ≈ ref rtol = 1e-12
    # accumulating variant: zero-init + two i-block adds == full contraction
    # (pins the += semantics and the linearity invariant the forward ring uses)
    H2 = zero(H)
    _slice2d_stage1_add!(H2, FL[:, :, :, 1:3], ALd[1:3, :, :, :])
    _slice2d_stage1_add!(H2, FL[:, :, :, 4:8], ALd[4:8, :, :, :])
    @test H2 ≈ H rtol = 1e-12
end

@testset "scatter/gather roundtrip" begin
    for (N1, N2) in ((2, 2), (1, 4), (4, 1)), χ in (16, 18)   # 18: uneven blocks
        Random.seed!(500 + χ + 10N1)
        g = slice2d_grid(N1, N2)
        FL = rand(ComplexF64, χ, 4, 4, χ)
        blk = slice2d_scatter(FL, g)
        a_rs = split_ranges(χ, N1); i_rs = split_ranges(χ, N2)
        @test size(blk) == (length(a_rs[g.r1 + 1]), 4, 4, length(i_rs[g.r2 + 1]))
        @test blk == FL[a_rs[g.r1 + 1], :, :, i_rs[g.r2 + 1]]
        FL2 = slice2d_gather(blk, g)
        @test FL2 ≈ FL
    end
end

@testset "FLmap_slice2d forward" begin
    for (N1, N2) in ((2, 2), (1, 4), (4, 1)), χ in (16, 18)
        D = 3
        FL, ALu, ALd, M1, M2, _ = make_leg5(χ, D; seed=600 + χ + 10N1)
        g = slice2d_grid(N1, N2)
        ref = FLmap(FL, ALu, ALd, M1, M2)
        out_blk = FLmap_slice2d(slice2d_scatter(FL, g), ALu, ALd, (M1, M2), g)
        a_rs = split_ranges(χ, N1); i_rs = split_ranges(χ, N2)
        @test size(out_blk) == (length(a_rs[g.r1 + 1]), D, D, length(i_rs[g.r2 + 1]))
        @test slice2d_gather(out_blk, g) ≈ ref rtol = 1e-12
        # single-M entry point (M2 = conj(M1) internally)
        out1 = slice2d_gather(FLmap_slice2d(slice2d_scatter(FL, g), ALu, ALd, M1, g), g)
        @test out1 ≈ FLmap(FL, ALu, ALd, M1) rtol = 1e-12
        # iterability: feed the output block straight back in
        out2_blk = FLmap_slice2d(out_blk, ALu, ALd, (M1, M2), g)
        ref2 = FLmap(ref, ALu, ALd, M1, M2)
        @test slice2d_gather(out2_blk, g) ≈ ref2 rtol = 1e-11
    end
end

@testset "scatter/gather rrules" begin
    for (N1, N2) in ((2, 2), (1, 4))
        Random.seed!(700 + 10N1)
        g = slice2d_grid(N1, N2)
        χ = 12
        FL = rand(ComplexF64, χ, 3, 3, χ)
        W  = rand(ComplexF64, χ, 3, 3, χ)
        # identity chain: gather(scatter(x)) == x, so dFL must equal the plain pullback of the loss
        loss(x) = real(sum(W .* slice2d_gather(slice2d_scatter(x, g), g)))
        l, back = Zygote.pullback(loss, FL)
        @test l ≈ real(sum(W .* FL))
        dFL = back(1.0)[1]
        l_ref, back_ref = Zygote.pullback(x -> real(sum(W .* x)), FL)
        @test dFL ≈ back_ref(1.0)[1]
    end
end

@testset "FLmap_slice2d gradient parity" begin
    for (N1, N2) in ((2, 2), (1, 4), (4, 1)), χ in (16, 18)
        D = 3
        FL, ALu, ALd, M1, M2, W = make_leg5(χ, D; seed=800 + χ + 10N1)
        g = slice2d_grid(N1, N2)

        loss_ref(FL, ALu, ALd, M1, M2) =
            real(sum(W .* FLmap(FL, ALu, ALd, M1, M2)))
        loss_can(FL, ALu, ALd, M1, M2) =
            real(sum(W .* slice2d_gather(
                FLmap_slice2d(slice2d_scatter(FL, g), ALu, ALd, (M1, M2), g), g)))

        l_ref, back_ref = Zygote.pullback(loss_ref, FL, ALu, ALd, M1, M2)
        l_can, back_can = Zygote.pullback(loss_can, FL, ALu, ALd, M1, M2)
        @test l_can ≈ l_ref rtol = 1e-12
        g_ref = back_ref(1.0)
        g_can = back_can(1.0)
        for (i, name) in enumerate(("dFL", "dALu", "dALd", "dM1", "dM2"))
            @test isapprox(g_can[i], g_ref[i]; rtol = 1e-10)
        end

        # single-M entry: checks the dM = dM1 + conj(dM2) composition
        loss1_ref(FL, M) = real(sum(W .* FLmap(FL, ALu, ALd, M)))
        loss1_can(FL, M) = real(sum(W .* slice2d_gather(
            FLmap_slice2d(slice2d_scatter(FL, g), ALu, ALd, M, g), g)))
        _, b1r = Zygote.pullback(loss1_ref, FL, M1)
        _, b1c = Zygote.pullback(loss1_can, FL, M1)
        gr, gc = b1r(1.0), b1c(1.0)
        @test isapprox(gc[1], gr[1]; rtol = 1e-10)   # dFL
        @test isapprox(gc[2], gr[2]; rtol = 1e-10)   # dM
    end
end

@testset "bare-sum loss (structured Fill cotangent)" begin
    χ, D = 16, 3
    FL, ALu, ALd, M1, M2, _ = make_leg5(χ, D; seed=950)
    g = slice2d_grid(2, 2)
    blk = slice2d_scatter(FL, g)
    # a bare `sum` loss produces a FillArrays cotangent — exercises the
    # densify guard in the FLmap_slice2d rrule (MPI needs a real device buffer)
    l_blk, back = Zygote.pullback(x -> real(sum(FLmap_slice2d(x, ALu, ALd, (M1, M2), g))), blk)
    dblk = back(1.0)[1]
    dFL_ref = Zygote.pullback(x -> real(sum(FLmap(x, ALu, ALd, M1, M2))), FL)[2](1.0)[1]
    a_rs = split_ranges(χ, 2); i_rs = split_ranges(χ, 2)
    @test dblk ≈ dFL_ref[a_rs[g.r1 + 1], :, :, i_rs[g.r2 + 1]] rtol = 1e-10
    # global loss = Σ over ranks of the local block sums
    l_glob = MPI.Allreduce(l_blk, +, comm)
    @test l_glob ≈ real(sum(FLmap(FL, ALu, ALd, M1, M2))) rtol = 1e-12
end

@testset "inner_etype Float32 boundary cast" begin
    χ, D = 16, 3
    FL, ALu, ALd, M1, M2, W = make_leg5(χ, D; seed=900)
    g = slice2d_grid(2, 2)
    ref = FLmap(FL, ALu, ALd, M1, M2)
    out = slice2d_gather(
        FLmap_slice2d(slice2d_scatter(FL, g), ALu, ALd, (M1, M2), g; inner_etype = Float32), g)
    @test eltype(out) == ComplexF64          # upcast at exit
    @test out ≈ ref rtol = 1e-4              # F32 accuracy

    loss(FL) = real(sum(W .* slice2d_gather(
        FLmap_slice2d(slice2d_scatter(FL, g), ALu, ALd, (M1, M2), g; inner_etype = Float32), g)))
    loss_ref(FL) = real(sum(W .* FLmap(FL, ALu, ALd, M1, M2)))
    dFL = Zygote.pullback(loss, FL)[2](1.0)[1]
    dFL_ref = Zygote.pullback(loss_ref, FL)[2](1.0)[1]
    @test eltype(dFL) == ComplexF64
    @test dFL ≈ dFL_ref rtol = 1e-3
end

@testset "hand adjoints == Zygote (local, no MPI)" begin
    χ, D = 8, 3
    FL, ALu, ALd, M1, M2, _ = make_leg5(χ, D; seed=102)
    H = _slice2d_stage1(FL, ALd)
    T = _slice2d_fold1(H, M1)
    G = _slice2d_fold2(T, M2)
    Random.seed!(103)
    dH = rand(ComplexF64, size(H)...)
    dT = rand(ComplexF64, size(T)...)
    dG = rand(ComplexF64, size(G)...)
    dP = rand(ComplexF64, size(_slice2d_stage2(G, ALu))...)

    _, bp1 = Zygote.pullback(_slice2d_stage1, FL, ALd)
    dFL_z, dALd_z = bp1(dH)
    @test _slice2d_stage1_dFL(dH, ALd) ≈ dFL_z rtol = 1e-12
    @test _slice2d_stage1_dALd(dH, FL) ≈ dALd_z rtol = 1e-12

    _, bpf1 = Zygote.pullback(_slice2d_fold1, H, M1)
    dH_z, dM1_z = bpf1(dT)
    @test _slice2d_fold1_dH(dT, M1) ≈ dH_z rtol = 1e-12
    @test _slice2d_fold1_dM1(dT, H) ≈ dM1_z rtol = 1e-12

    _, bpf2 = Zygote.pullback(_slice2d_fold2, T, M2)
    dT_z, dM2_z = bpf2(dG)
    @test _slice2d_fold2_dT(dG, M2) ≈ dT_z rtol = 1e-12
    @test _slice2d_fold2_dM2(dG, T) ≈ dM2_z rtol = 1e-12

    _, bps2 = Zygote.pullback(_slice2d_stage2, G, ALu)
    dG_z, dALu_z = bps2(dP)
    @test _slice2d_stage2_dG(dP, ALu) ≈ dG_z rtol = 1e-12
    @test _slice2d_stage2_dALu(dP, G) ≈ dALu_z rtol = 1e-12
end

@testset "forloop_iter chunking parity" begin
    for (N1, N2) in ((2, 2), (1, 4), (4, 1))
        χ, D = 18, 3   # uneven blocks AND uneven chunks
        local FL, ALu, ALd, M1, M2, ref
        for n in (2, 3)
            FL, ALu, ALd, M1, M2, W = make_leg5(χ, D; seed=1000 + 10N1 + n)
            g = slice2d_grid(N1, N2)
            ref = FLmap(FL, ALu, ALd, M1, M2)
            out = slice2d_gather(FLmap_slice2d(slice2d_scatter(FL, g), ALu, ALd, (M1, M2), g; forloop_iter = n), g)
            @test out ≈ ref rtol = 1e-12
            loss_ref(FL, ALu, ALd, M1, M2) = real(sum(W .* FLmap(FL, ALu, ALd, M1, M2)))
            loss_can(FL, ALu, ALd, M1, M2) = real(sum(W .* slice2d_gather(
                FLmap_slice2d(slice2d_scatter(FL, g), ALu, ALd, (M1, M2), g; forloop_iter = n), g)))
            g_ref = Zygote.pullback(loss_ref, FL, ALu, ALd, M1, M2)[2](1.0)
            g_can = Zygote.pullback(loss_can, FL, ALu, ALd, M1, M2)[2](1.0)
            for i in 1:5
                @test isapprox(g_can[i], g_ref[i]; rtol = 1e-10)
            end
        end
        # clamp path: forloop_iter beyond the local l extent
        g = slice2d_grid(N1, N2)
        out_cl = slice2d_gather(FLmap_slice2d(slice2d_scatter(FL, g), ALu, ALd, (M1, M2), g; forloop_iter = 99), g)
        @test out_cl ≈ ref rtol = 1e-12
    end
end

@testset "slice2d_dot / slice2d_norm" begin
    for (N1, N2) in ((2, 2), (1, 4), (4, 1)), χ in (16, 18)   # 18: uneven blocks
        Random.seed!(1100 + χ + 10N1)
        g = slice2d_grid(N1, N2)
        x = rand(ComplexF64, χ, 3, 3, χ)
        y = rand(ComplexF64, χ, 3, 3, χ)
        xb = slice2d_scatter(x, g); yb = slice2d_scatter(y, g)
        @test slice2d_dot(xb, yb, g) ≈ dot(x, y) rtol = 1e-12
        @test slice2d_norm(xb, g) ≈ norm(x) rtol = 1e-12
    end
end

@testset "distributed power iteration == simple_eig" begin
    χ, D = 16, 3
    FL, ALu, ALd, M1, M2, _ = make_leg5(χ, D; seed=1200)
    g = slice2d_grid(2, 2)
    # serial reference: simple_eig on the full map (returns ([λ], [v]))
    λ_refs, FL_refs = simple_eig(x -> FLmap(x, ALu, ALd, M1, M2), FL; power_iter = 200)
    λ_ref, FL_ref = λ_refs[1], FL_refs[1]
    # distributed: same algorithm on blocks with slice2d dot/norm.
    # simple_eig does (power_iter-1) normalizing iters, then v1 = f(v),
    # λ = dot(v, v1) — i.e. λ = dot(v_199, f(v_199)). The loop below computes
    # the identical quantities on blocks (f is linear, so the initial
    # normalization doesn't change the iterate direction).
    blk = slice2d_scatter(FL, g)
    v = blk ./ slice2d_norm(blk, g)
    local λ, w
    for _ in 1:200
        w = FLmap_slice2d(v, ALu, ALd, (M1, M2), g)
        λ = slice2d_dot(v, w, g)
        v = w ./ slice2d_norm(w, g)
    end
    @test λ ≈ λ_ref rtol = 1e-8
    # eigenvector parity up to global phase: fix phase via the largest |entry|
    v_full = slice2d_gather(v, g)
    FL_ref_n = FL_ref ./ norm(FL_ref)   # simple_eig already normalizes; harmless
    imax = argmax(abs.(FL_ref_n))
    phase = v_full[imax] / FL_ref_n[imax]
    @test abs(phase) ≈ 1 rtol = 1e-6
    @test v_full ≈ FL_ref_n .* phase rtol = 1e-6
    # _dist signature iterates too: one map application on blocks (with ALu/ALd
    # also distributed) matches the block of the serial map — input/output
    # distribution match, so the output feeds straight back in.
    ALub = slice2d_scatter(ALu, g); ALdb = slice2d_scatter(ALd, g)
    w_dist = FLmap_slice2d_dist(v, ALub, ALdb, (M1, M2), g)
    @test w_dist ≈ slice2d_scatter(FLmap(v_full, ALu, ALd, M1, M2), g) rtol = 1e-10
end

@testset "FLmap_slice2d_dist == FLmap (AL distributed)" begin
    for (N1, N2) in ((2, 2), (1, 4), (4, 1)), χ in (16, 18), n in (1, 3)
        D = 3
        FL, ALu, ALd, M1, M2, W = make_leg5(χ, D; seed=1400 + χ + 10N1 + n)
        g = slice2d_grid(N1, N2)
        ref = FLmap(FL, ALu, ALd, M1, M2)
        FLb = slice2d_scatter(FL, g); ALub = slice2d_scatter(ALu, g); ALdb = slice2d_scatter(ALd, g)
        out = slice2d_gather(FLmap_slice2d_dist(FLb, ALub, ALdb, (M1, M2), g; forloop_iter = n), g)
        @test out ≈ ref rtol = 1e-12

        # gradient parity vs serial — all five, with AL gradients compared block-wise.
        # Block-local weighted sum: the global loss is the sum over ranks (the
        # blocks tile W .* result disjointly), identical to the serial
        # full-tensor loss; each rank's cotangent is its W block.
        loss_ref(FL, ALu, ALd, M1, M2) = real(sum(W .* FLmap(FL, ALu, ALd, M1, M2)))
        Wb = slice2d_scatter(W, g)
        loss_dist(FLb, ALub, ALdb, M1, M2) = real(sum(Wb .* FLmap_slice2d_dist(FLb, ALub, ALdb, (M1, M2), g; forloop_iter = n)))
        g_ref = Zygote.pullback(loss_ref, FL, ALu, ALd, M1, M2)[2](1.0)
        g_dist = Zygote.pullback(loss_dist, FLb, ALub, ALdb, M1, M2)[2](1.0)
        a_rs = split_ranges(χ, N1); i_rs = split_ranges(χ, N2)
        blkof(x) = x[a_rs[g.r1 + 1], :, :, i_rs[g.r2 + 1]]
        @test g_dist[1] ≈ blkof(g_ref[1]) rtol = 1e-10   # dFL block
        @test g_dist[2] ≈ blkof(g_ref[2]) rtol = 1e-10   # dALu block
        @test g_dist[3] ≈ blkof(g_ref[3]) rtol = 1e-10   # dALd block
        @test g_dist[4] ≈ g_ref[4] rtol = 1e-10          # dM1 (replicated)
        @test g_dist[5] ≈ g_ref[5] rtol = 1e-10          # dM2
    end
end

@testset "simple_eig kwargs (default-equivalent)" begin
    χ, D = 12, 3
    FL, ALu, ALd, M1, M2, _ = make_leg5(χ, D; seed=1300)
    f = x -> FLmap(x, ALu, ALd, M1, M2)
    λ1, v1 = simple_eig(f, FL; power_iter = 100)
    λ2, v2 = simple_eig(f, FL; power_iter = 100, inner_product = dot, norm_fn = norm)
    @test λ1[1] ≈ λ2[1] rtol = 1e-12
    @test v1[1] ≈ v2[1] rtol = 1e-12
end

println("rank $rank: test_slice2d.jl done")
