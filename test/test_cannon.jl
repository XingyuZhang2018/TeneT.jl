# Cannon-style 2D distributed FLmap tests. Run via: julia --project=. test/run_test_cannon.jl
# CPU Arrays only (no CUDA-aware MPI on local machines); GPU is covered by
# examples/MPI_parallel/test_cannon_sofia.jl.
using Test
using MPI
using LinearAlgebra
using Random
using Zygote
using TeneT
using TeneT: cannon_grid, CannonGrid, cannon_scatter, cannon_gather, FLmap, split_ranges, _cannon_stage1, _cannon_stage1_add!, _cannon_fold, _cannon_stage2

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

println("rank $rank: test_cannon.jl done")
