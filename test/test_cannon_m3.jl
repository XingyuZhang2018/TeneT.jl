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

@testset "Cmap_cannon gradient parity" begin
    for (N1, N2) in ((2, 2), (1, 4))
        χ, D = 16, 3
        Random.seed!(2500 + 10N1)
        FL = rand(ComplexF64, χ, D, D, χ); FR = rand(ComplexF64, χ, D, D, χ); C = rand(ComplexF64, χ, χ)
        W  = rand(ComplexF64, χ, χ)
        g = cannon_grid(N1, N2)
        FLb = cannon_scatter(FL, g); FRb = cannon_scatter(FR, g)
        loss_ref(C,FL,FR)   = real(sum(W .* Cmap(C, FL, FR)))
        loss_dist(C,FLb,FRb) = real(sum(W .* Cmap_cannon(C, FLb, FRb, g)))
        g_ref  = Zygote.pullback(loss_ref,  C, FL, FR)[2](1.0)
        g_dist = Zygote.pullback(loss_dist, C, FLb, FRb)[2](1.0)
        a_rs = split_ranges(χ, N1); e_rs = split_ranges(χ, N2)
        blkof(x) = x[a_rs[g.r1+1], :, :, e_rs[g.r2+1]]
        @test g_dist[1] ≈ g_ref[1] rtol = 1e-10              # dC full (replicated)
        @test g_dist[2] ≈ blkof(g_ref[2]) rtol = 1e-10       # dFL block (take-my-block)
        @test g_dist[3] ≈ blkof(g_ref[3]) rtol = 1e-10       # dFR block (take-my-block)
    end
end

@testset "Cmap_cannon leg3 parity + replicated-output invariant" begin
    for (N1, N2) in ((2, 2), (1, 4), (4, 1)), χ in (16, 18)
        D = 3
        Random.seed!(2600 + χ + 10N1)
        FL = rand(ComplexF64, χ, D, χ)        # [a, c, d]  (a=r1, d=r2)
        FR = rand(ComplexF64, χ, D, χ)        # [b, c, e]  (b=r1, e=r2)
        C  = rand(ComplexF64, χ, χ)
        g = cannon_grid(N1, N2)
        ref = Cmap(C, FL, FR)                 # leg3 Cmap: result[d,e] := FL[a,c,d] C[a,b] FR[b,c,e]
        out = Cmap_cannon(C, cannon_scatter(FL, g), cannon_scatter(FR, g), g)
        @test size(out) == size(ref)          # FULL χ×χ
        @test out ≈ ref rtol = 1e-12
        # replicated-output invariant: every rank produced the IDENTICAL full out.
        # rank 0 broadcasts its out; the cross-rank MAX residual must be 0.
        out_bcast = MPI.bcast(out, 0, comm)
        @test MPI.Allreduce(maximum(abs.(out .- out_bcast)), max, comm) ≈ 0 atol = 1e-12
    end
end

@testset "Cmap_cannon leg3 gradient parity" begin
    for (N1, N2) in ((2, 2), (1, 4))
        χ, D = 16, 3
        Random.seed!(2700 + 10N1)
        FL = rand(ComplexF64, χ, D, χ); FR = rand(ComplexF64, χ, D, χ); C = rand(ComplexF64, χ, χ)
        W  = rand(ComplexF64, χ, χ)
        g = cannon_grid(N1, N2)
        FLb = cannon_scatter(FL, g); FRb = cannon_scatter(FR, g)
        loss_ref(C,FL,FR)    = real(sum(W .* Cmap(C, FL, FR)))
        loss_dist(C,FLb,FRb) = real(sum(W .* Cmap_cannon(C, FLb, FRb, g)))
        g_ref  = Zygote.pullback(loss_ref,  C, FL, FR)[2](1.0)
        g_dist = Zygote.pullback(loss_dist, C, FLb, FRb)[2](1.0)
        a_rs = split_ranges(χ, N1); e_rs = split_ranges(χ, N2)
        blkof3(x) = x[a_rs[g.r1+1], :, e_rs[g.r2+1]]   # 3-leg take-my-block
        @test g_dist[1] ≈ g_ref[1] rtol = 1e-10              # dC full (replicated)
        @test g_dist[2] ≈ blkof3(g_ref[2]) rtol = 1e-10      # dFL block (take-my-block)
        @test g_dist[3] ≈ blkof3(g_ref[3]) rtol = 1e-10      # dFR block (take-my-block)
    end
end

println("rank $rank: test_cannon_m3.jl batch A done")
