# M3 Cannon distributed maps (Cmap/FR/AC/ACd) — distributed parity vs serial.
# Run via: julia --project=. test/run_test_cannon_m3.jl  (mpiexec -n 4)
# CPU Arrays only; GPU is examples/MPI_parallel/test_cannon_m3_sofia.jl (Batch E).
using Test, MPI, LinearAlgebra, Random, Zygote
using TeneT
using TeneT: cannon_grid, CannonGrid, cannon_scatter, cannon_gather,
             split_ranges, Cmap, Cmap_cannon
# Batch B appends: FRmap, FRmap_cannon_dist
# Batch C appends: ACmap, ACmap_cannon_dist
# Batch D appends: ACdmap, ACdmap_cannon_dist

MPI.Init()
const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
@assert MPI.Comm_size(comm) == 4 "test_cannon_m3.jl expects exactly 4 ranks"

function make_leg5(χ, D; d=2, seed=42)
    Random.seed!(seed)
    A   = rand(ComplexF64, χ, D, D, χ)   # AC / ACd / FR role tensor
    Bu  = rand(ComplexF64, χ, D, D, χ)   # ARu / FL role
    Bd  = rand(ComplexF64, χ, D, D, χ)   # ARd / FR role
    M1  = rand(ComplexF64, D, D, D, D, d)
    M2  = rand(ComplexF64, D, D, D, D, d)
    W   = rand(ComplexF64, χ, D, D, χ)
    return A, Bu, Bd, M1, M2, W
end

@testset "Cmap_cannon forward parity" begin
    for (N1, N2) in ((2, 2), (1, 4), (4, 1)), χ in (16, 18)
        D = 3
        Random.seed!(2400 + χ + 10N1)
        FL = rand(ComplexF64, χ, D, D, χ)
        FR = rand(ComplexF64, χ, D, D, χ)
        C  = rand(ComplexF64, χ, χ)
        g = cannon_grid(N1, N2)
        ref = Cmap(C, FL, FR)                 # leg4 Cmap: result[e,f] := FL[a,c,d,e] C[a,b] FR[b,c,d,f]
        out = Cmap_cannon(C, cannon_scatter(FL, g), cannon_scatter(FR, g), g)
        @test size(out) == size(ref)          # FULL χ×χ, replicated (NOT a block)
        @test out ≈ ref rtol = 1e-12
    end
end

println("rank $rank: test_cannon_m3.jl batch A done")
