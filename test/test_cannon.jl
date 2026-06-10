# Cannon-style 2D distributed FLmap tests. Run via: julia --project=. test/run_test_cannon.jl
# CPU Arrays only (no CUDA-aware MPI on local machines); GPU is covered by
# examples/MPI_parallel/test_cannon_sofia.jl.
using Test
using MPI
using LinearAlgebra
using Random
using Zygote
using TeneT
using TeneT: cannon_grid, CannonGrid, cannon_scatter, cannon_gather, FLmap, FLmap_cannon, split_ranges, _cannon_stage1, _cannon_stage1_add!, _cannon_fold, _cannon_stage2

MPI.Init()
const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
@assert MPI.Comm_size(comm) == 4 "test_cannon.jl expects exactly 4 ranks"

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

@testset "cannon_grid" begin
    for (N1, N2) in ((2, 2), (1, 4), (4, 1))
        g = cannon_grid(N1, N2)
        @test g isa CannonGrid
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
    H = _cannon_stage1(FL, ALd)
    G = _cannon_fold(H, M1, M2)
    P = _cannon_stage2(G, ALu)
    ref = FLmap(FL, ALu, ALd, M1, M2)
    @test P ≈ ref rtol = 1e-12
    # accumulating variant: zero-init + two i-block adds == full contraction
    # (pins the += semantics and the linearity invariant the forward ring uses)
    H2 = zero(H)
    _cannon_stage1_add!(H2, FL[:, :, :, 1:3], ALd[1:3, :, :, :])
    _cannon_stage1_add!(H2, FL[:, :, :, 4:8], ALd[4:8, :, :, :])
    @test H2 ≈ H rtol = 1e-12
end

@testset "scatter/gather roundtrip" begin
    for (N1, N2) in ((2, 2), (1, 4), (4, 1)), χ in (16, 18)   # 18: uneven blocks
        Random.seed!(500 + χ + 10N1)
        g = cannon_grid(N1, N2)
        FL = rand(ComplexF64, χ, 4, 4, χ)
        blk = cannon_scatter(FL, g)
        a_rs = split_ranges(χ, N1); i_rs = split_ranges(χ, N2)
        @test size(blk) == (length(a_rs[g.r1 + 1]), 4, 4, length(i_rs[g.r2 + 1]))
        @test blk == FL[a_rs[g.r1 + 1], :, :, i_rs[g.r2 + 1]]
        FL2 = cannon_gather(blk, g)
        @test FL2 ≈ FL
    end
end

@testset "FLmap_cannon forward" begin
    for (N1, N2) in ((2, 2), (1, 4), (4, 1)), χ in (16, 18)
        D = 3
        FL, ALu, ALd, M1, M2, _ = make_leg5(χ, D; seed=600 + χ + 10N1)
        g = cannon_grid(N1, N2)
        ref = FLmap(FL, ALu, ALd, M1, M2)
        out_blk = FLmap_cannon(cannon_scatter(FL, g), ALu, ALd, (M1, M2), g)
        a_rs = split_ranges(χ, N1); i_rs = split_ranges(χ, N2)
        @test size(out_blk) == (length(a_rs[g.r1 + 1]), D, D, length(i_rs[g.r2 + 1]))
        @test cannon_gather(out_blk, g) ≈ ref rtol = 1e-12
        # single-M entry point (M2 = conj(M1) internally)
        out1 = cannon_gather(FLmap_cannon(cannon_scatter(FL, g), ALu, ALd, M1, g), g)
        @test out1 ≈ FLmap(FL, ALu, ALd, M1) rtol = 1e-12
        # iterability: feed the output block straight back in
        out2_blk = FLmap_cannon(out_blk, ALu, ALd, (M1, M2), g)
        ref2 = FLmap(ref, ALu, ALd, M1, M2)
        @test cannon_gather(out2_blk, g) ≈ ref2 rtol = 1e-11
    end
end

@testset "scatter/gather rrules" begin
    for (N1, N2) in ((2, 2), (1, 4))
        Random.seed!(700 + 10N1)
        g = cannon_grid(N1, N2)
        χ = 12
        FL = rand(ComplexF64, χ, 3, 3, χ)
        W  = rand(ComplexF64, χ, 3, 3, χ)
        # identity chain: gather(scatter(x)) == x, so dFL must equal the plain pullback of the loss
        loss(x) = real(sum(W .* cannon_gather(cannon_scatter(x, g), g)))
        l, back = Zygote.pullback(loss, FL)
        @test l ≈ real(sum(W .* FL))
        dFL = back(1.0)[1]
        l_ref, back_ref = Zygote.pullback(x -> real(sum(W .* x)), FL)
        @test dFL ≈ back_ref(1.0)[1]
    end
end

@testset "FLmap_cannon gradient parity" begin
    for (N1, N2) in ((2, 2), (1, 4), (4, 1)), χ in (16, 18)
        D = 3
        FL, ALu, ALd, M1, M2, W = make_leg5(χ, D; seed=800 + χ + 10N1)
        g = cannon_grid(N1, N2)

        loss_ref(FL, ALu, ALd, M1, M2) =
            real(sum(W .* FLmap(FL, ALu, ALd, M1, M2)))
        loss_can(FL, ALu, ALd, M1, M2) =
            real(sum(W .* cannon_gather(
                FLmap_cannon(cannon_scatter(FL, g), ALu, ALd, (M1, M2), g), g)))

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
        loss1_can(FL, M) = real(sum(W .* cannon_gather(
            FLmap_cannon(cannon_scatter(FL, g), ALu, ALd, M, g), g)))
        _, b1r = Zygote.pullback(loss1_ref, FL, M1)
        _, b1c = Zygote.pullback(loss1_can, FL, M1)
        gr, gc = b1r(1.0), b1c(1.0)
        @test isapprox(gc[1], gr[1]; rtol = 1e-10)   # dFL
        @test isapprox(gc[2], gr[2]; rtol = 1e-10)   # dM
    end
end

@testset "inner_etype Float32 boundary cast" begin
    χ, D = 16, 3
    FL, ALu, ALd, M1, M2, W = make_leg5(χ, D; seed=900)
    g = cannon_grid(2, 2)
    ref = FLmap(FL, ALu, ALd, M1, M2)
    out = cannon_gather(
        FLmap_cannon(cannon_scatter(FL, g), ALu, ALd, (M1, M2), g; inner_etype = Float32), g)
    @test eltype(out) == ComplexF64          # upcast at exit
    @test out ≈ ref rtol = 1e-4              # F32 accuracy

    loss(FL) = real(sum(W .* cannon_gather(
        FLmap_cannon(cannon_scatter(FL, g), ALu, ALd, (M1, M2), g; inner_etype = Float32), g)))
    loss_ref(FL) = real(sum(W .* FLmap(FL, ALu, ALd, M1, M2)))
    dFL = Zygote.pullback(loss, FL)[2](1.0)[1]
    dFL_ref = Zygote.pullback(loss_ref, FL)[2](1.0)[1]
    @test eltype(dFL) == ComplexF64
    @test dFL ≈ dFL_ref rtol = 1e-3
end

println("rank $rank: test_cannon.jl done")
