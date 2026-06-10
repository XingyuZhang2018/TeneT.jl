# Cannon-style 2D distributed FLmap tests. Run via: julia --project=. test/run_test_cannon.jl
# CPU Arrays only (no CUDA-aware MPI on local machines); GPU is covered by
# examples/MPI_parallel/test_cannon_sofia.jl.
using Test
using MPI
using LinearAlgebra
using Random
using Zygote
using TeneT
using TeneT: cannon_grid, CannonGrid, FLmap, split_ranges, _cannon_stage1, _cannon_stage2

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
    G = _cannon_stage1(FL, ALd, M1, M2)
    P = _cannon_stage2(G, ALu)
    ref = FLmap(FL, ALu, ALd, M1, M2)
    @test P ≈ ref rtol = 1e-12
end

println("rank $rank: test_cannon.jl done")
