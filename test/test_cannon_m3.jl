# M3 Cannon distributed maps (Cmap/FR/AC/ACd) — distributed parity vs serial.
# Run via: julia --project=. test/run_test_cannon_m3.jl  (mpiexec -n 4)
# CPU Arrays only; GPU is examples/MPI_parallel/test_cannon_m3_sofia.jl (Batch E).
using Test, MPI, LinearAlgebra, Random, Zygote
using TeneT
using TeneT: cannon_grid, CannonGrid, cannon_scatter, cannon_gather,
             split_ranges, Cmap, Cmap_cannon,
             FRmap, FRmap_cannon_dist
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

# ─── Batch B: FRmap_cannon_dist (gather class, square grid) ──────────────────

@testset "FRmap_cannon_dist forward parity (square grid)" begin
    N1 = N2 = 2
    for χ in (16, 18), n in (1, 3)
        D = 3
        FR, ARu, ARd, M1, M2, _ = make_leg5(χ, D; seed=2100 + χ + n)
        g = cannon_grid(N1, N2)
        ref = FRmap(FR, ARu, ARd, (M1, M2))   # result[a,e,f,i] := ARd[i,j,k,l] FR[d,g,h,l] M1 M2 ARu[a,b,c,d]
        FRb = cannon_scatter(FR, g); ARub = cannon_scatter(ARu, g); ARdb = cannon_scatter(ARd, g)
        out = cannon_gather(FRmap_cannon_dist(FRb, ARub, ARdb, (M1, M2), g; forloop_iter=n), g)
        @test out ≈ ref rtol = 1e-12
        if χ == 18   # off-diagonal (a,i) plane — the trap (a-blk≠i-blk)
            @test out[1:9, :, :, 10:18] ≈ ref[1:9, :, :, 10:18] rtol = 1e-12   # a-blk 0, i-blk 1
            @test out[10:18, :, :, 1:9] ≈ ref[10:18, :, :, 1:9] rtol = 1e-12   # a-blk 1, i-blk 0
        end
        # single-M entry (M2 = conj(M1) internally)
        out1 = cannon_gather(FRmap_cannon_dist(FRb, ARub, ARdb, M1, g), g)
        @test out1 ≈ FRmap(FR, ARu, ARd, M1) rtol = 1e-12
    end
    @test_skip "FRmap_cannon_dist rectangular grid (N1≠N2) deferred to M3 v2"
end

@testset "FRmap_cannon_dist gradient parity (square grid)" begin
    N1 = N2 = 2
    for χ in (16, 18), n in (1, 3)
        D = 3
        FR, ARu, ARd, M1, M2, W = make_leg5(χ, D; seed=2200 + χ + n)
        g = cannon_grid(N1, N2)
        FRb = cannon_scatter(FR, g); ARub = cannon_scatter(ARu, g); ARdb = cannon_scatter(ARd, g)
        Wb = cannon_scatter(W, g)
        loss_ref(FR,ARu,ARd,M1,M2)  = real(sum(W  .* FRmap(FR,ARu,ARd,(M1,M2))))
        loss_dist(FRb,ARub,ARdb,M1,M2) = real(sum(Wb .* FRmap_cannon_dist(FRb,ARub,ARdb,(M1,M2),g; forloop_iter=n)))
        g_ref  = Zygote.pullback(loss_ref,  FR,ARu,ARd,M1,M2)[2](1.0)
        g_dist = Zygote.pullback(loss_dist, FRb,ARub,ARdb,M1,M2)[2](1.0)
        p_rs = split_ranges(χ, N1)
        blkof(x) = x[p_rs[g.r1+1], :, :, p_rs[g.r2+1]]
        @test g_dist[1] ≈ blkof(g_ref[1]) rtol = 1e-10   # dFR block
        @test g_dist[2] ≈ blkof(g_ref[2]) rtol = 1e-10   # dARu block
        @test g_dist[3] ≈ blkof(g_ref[3]) rtol = 1e-10   # dARd block
        @test g_dist[4] ≈ g_ref[4] rtol = 1e-10          # dM1 replicated
        @test g_dist[5] ≈ g_ref[5] rtol = 1e-10          # dM2
    end
    # single-M dM = dM1 + conj(dM2) composition
    FR, ARu, ARd, M1, M2, W = make_leg5(16, 3; seed=2250)
    g = cannon_grid(2, 2)
    FRb=cannon_scatter(FR,g); ARub=cannon_scatter(ARu,g); ARdb=cannon_scatter(ARd,g); Wb=cannon_scatter(W,g)
    lr(FR,M) = real(sum(W  .* FRmap(FR,ARu,ARd,M)))
    ld(FRb,M)= real(sum(Wb .* FRmap_cannon_dist(FRb,ARub,ARdb,M,g)))
    gr = Zygote.pullback(lr, FR, M1)[2](1.0); gd = Zygote.pullback(ld, FRb, M1)[2](1.0)
    p_rs = split_ranges(16, 2)
    @test gd[1] ≈ gr[1][p_rs[g.r1+1], :, :, p_rs[g.r2+1]] rtol = 1e-10
    @test gd[2] ≈ gr[2] rtol = 1e-10
    # bare-sum loss (FillArrays densify guard)
    back = Zygote.pullback(x -> real(sum(FRmap_cannon_dist(x, ARub, ARdb, (M1,M2), g))), FRb)[2]
    dblk = back(1.0)[1]
    dFR_ref = Zygote.pullback(x -> real(sum(FRmap(x, ARu, ARd, M1, M2))), FR)[2](1.0)[1]
    @test dblk ≈ dFR_ref[p_rs[g.r1+1], :, :, p_rs[g.r2+1]] rtol = 1e-10
    # inner_etype Float32 boundary cast (fwd 1e-4, grad 1e-3)
    out32 = cannon_gather(FRmap_cannon_dist(FRb, ARub, ARdb, (M1,M2), g; inner_etype=Float32), g)
    @test eltype(out32) == ComplexF64
    @test out32 ≈ FRmap(FR, ARu, ARd, (M1,M2)) rtol = 1e-4
end

@testset "FRmap_cannon_dist forloop_iter clamp (square grid)" begin
    N1 = N2 = 2
    χ, D = 18, 3
    FR, ARu, ARd, M1, M2, _ = make_leg5(χ, D; seed=2300)
    g = cannon_grid(N1, N2)
    ref = FRmap(FR, ARu, ARd, (M1, M2))
    FRb=cannon_scatter(FR,g); ARub=cannon_scatter(ARu,g); ARdb=cannon_scatter(ARd,g)
    out = cannon_gather(FRmap_cannon_dist(FRb, ARub, ARdb, (M1,M2), g; forloop_iter=99), g)
    @test out ≈ ref rtol = 1e-12
end

println("rank $rank: test_cannon_m3.jl batch B done")
