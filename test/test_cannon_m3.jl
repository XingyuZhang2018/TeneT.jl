# M3 Cannon distributed maps (Cmap/FR/AC/ACd) — distributed parity vs serial.
# Run via: julia --project=. test/run_test_cannon_m3.jl  (mpiexec -n 4)
# CPU Arrays only; GPU is examples/MPI_parallel/test_cannon_m3_sofia.jl (Batch E).
using Test, MPI, LinearAlgebra, Random, Zygote
using TeneT
using TeneT: cannon_grid, CannonGrid, cannon_scatter, cannon_gather,
             split_ranges, Cmap, Cmap_cannon,
             FRmap, FRmap_cannon_dist,
             ACmap, ACmap_cannon_dist,
             ACdmap, ACdmap_cannon_dist

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
    # inner_etype Float32 boundary cast — forward (1e-4) AND gradient (1e-3),
    # both halves like the FLmap template (test_cannon.jl:158-175): the gradient
    # half exercises the rrule do_cast branch (_boundary_cast on the cotangent +
    # T_orig upcast of dFR/dARu/dARd/dM), else that path ships untested.
    out32 = cannon_gather(FRmap_cannon_dist(FRb, ARub, ARdb, (M1,M2), g; inner_etype=Float32), g)
    @test eltype(out32) == ComplexF64
    @test out32 ≈ FRmap(FR, ARu, ARd, (M1,M2)) rtol = 1e-4
    loss32(FRb)  = real(sum(Wb .* FRmap_cannon_dist(FRb, ARub, ARdb, (M1,M2), g; inner_etype=Float32)))
    lref(FR)     = real(sum(W  .* FRmap(FR, ARu, ARd, (M1,M2))))
    dFR32 = Zygote.pullback(loss32, FRb)[2](1.0)[1]
    dFRr  = Zygote.pullback(lref,  FR)[2](1.0)[1]
    @test eltype(dFR32) == ComplexF64                                            # upcast at exit
    @test dFR32 ≈ dFRr[p_rs[g.r1+1], :, :, p_rs[g.r2+1]] rtol = 1e-3            # F32 grad accuracy
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

# ─── Batch C: ACmap_cannon_dist (gather class, SINGLE l-chunk, square grid) ───

@testset "ACmap_cannon_dist forward parity (square grid)" begin
    N1 = N2 = 2
    for χ in (16, 18), n in (1, 3)
        D = 3
        AC, FL, FR, M1, M2, _ = make_leg5(χ, D; seed=2600 + χ + n)
        g = cannon_grid(N1, N2)
        ref = ACmap(AC, FL, FR, (M1, M2))      # result[i,j,k,l] := AC[a,b,c,d] FR[d,g,h,l] M1 M2 FL[a,e,f,i]
        ACb = cannon_scatter(AC, g); FLb = cannon_scatter(FL, g); FRb = cannon_scatter(FR, g)
        out = cannon_gather(ACmap_cannon_dist(ACb, FLb, FRb, (M1,M2), g; forloop_iter=n), g)
        @test out ≈ ref rtol = 1e-12
        # CRITICAL off-diagonal (the diagonal trap): with χ=18, N=2 →
        # p_rs = [1:9, 10:18]; check off-diagonal (i,l) blocks A≠B explicitly.
        if χ == 18
            @test out[1:9, :, :, 10:18] ≈ ref[1:9, :, :, 10:18] rtol = 1e-12   # i-blk 0, l-blk 1
            @test out[10:18, :, :, 1:9] ≈ ref[10:18, :, :, 1:9] rtol = 1e-12   # i-blk 1, l-blk 0
        end
        # single-M entry
        out1 = cannon_gather(ACmap_cannon_dist(ACb, FLb, FRb, M1, g), g)
        @test out1 ≈ ACmap(AC, FL, FR, M1) rtol = 1e-12
    end
    # rectangular grids are M3 v2 — explicitly skipped:
    @test_skip "ACmap_cannon_dist rectangular grid (N1≠N2) deferred to M3 v2"
end

@testset "ACmap_cannon_dist gradient parity (square grid)" begin
    N1 = N2 = 2
    for χ in (16, 18), n in (1, 3)
        D = 3
        AC, FL, FR, M1, M2, W = make_leg5(χ, D; seed=2700 + χ + n)
        g = cannon_grid(N1, N2)
        ACb=cannon_scatter(AC,g); FLb=cannon_scatter(FL,g); FRb=cannon_scatter(FR,g); Wb=cannon_scatter(W,g)
        loss_ref(AC,FL,FR,M1,M2)  = real(sum(W  .* ACmap(AC,FL,FR,(M1,M2))))
        loss_dist(ACb,FLb,FRb,M1,M2) = real(sum(Wb .* ACmap_cannon_dist(ACb,FLb,FRb,(M1,M2),g; forloop_iter=n)))
        g_ref  = Zygote.pullback(loss_ref,  AC,FL,FR,M1,M2)[2](1.0)
        g_dist = Zygote.pullback(loss_dist, ACb,FLb,FRb,M1,M2)[2](1.0)
        p_rs = split_ranges(χ, N1)
        blkof(x) = x[p_rs[g.r1+1], :, :, p_rs[g.r2+1]]
        @test g_dist[1] ≈ blkof(g_ref[1]) rtol = 1e-10   # dAC block
        @test g_dist[2] ≈ blkof(g_ref[2]) rtol = 1e-10   # dFL block
        @test g_dist[3] ≈ blkof(g_ref[3]) rtol = 1e-10   # dFR block
        @test g_dist[4] ≈ g_ref[4] rtol = 1e-10          # dM1 replicated
        @test g_dist[5] ≈ g_ref[5] rtol = 1e-10          # dM2
    end
    # single-M dM = dM1 + conj(dM2) composition
    AC, FL, FR, M1, M2, W = make_leg5(16, 3; seed=2750)
    g = cannon_grid(2, 2)
    ACb=cannon_scatter(AC,g); FLb=cannon_scatter(FL,g); FRb=cannon_scatter(FR,g); Wb=cannon_scatter(W,g)
    lr(AC,M) = real(sum(W  .* ACmap(AC,FL,FR,M)))
    ld(ACb,M)= real(sum(Wb .* ACmap_cannon_dist(ACb,FLb,FRb,M,g)))
    gr = Zygote.pullback(lr, AC, M1)[2](1.0); gd = Zygote.pullback(ld, ACb, M1)[2](1.0)
    p_rs = split_ranges(16, 2)
    @test gd[1] ≈ gr[1][p_rs[g.r1+1], :, :, p_rs[g.r2+1]] rtol = 1e-10
    @test gd[2] ≈ gr[2] rtol = 1e-10
    # bare-sum loss (FillArrays densify guard)
    back = Zygote.pullback(x -> real(sum(ACmap_cannon_dist(x, FLb, FRb, (M1,M2), g))), ACb)[2]
    dblk = back(1.0)[1]
    dAC_ref = Zygote.pullback(x -> real(sum(ACmap(x, FL, FR, M1, M2))), AC)[2](1.0)[1]
    @test dblk ≈ dAC_ref[p_rs[g.r1+1], :, :, p_rs[g.r2+1]] rtol = 1e-10
    # inner_etype Float32 boundary cast — forward (1e-4) AND gradient (1e-3),
    # both halves like the FLmap/FRmap template: the gradient half exercises the
    # rrule do_cast branch (_boundary_cast on the cotangent + T_orig upcast of
    # dAC/dFL/dFR/dM), else that path ships untested.
    out32 = cannon_gather(ACmap_cannon_dist(ACb, FLb, FRb, (M1,M2), g; inner_etype=Float32), g)
    @test eltype(out32) == ComplexF64
    @test out32 ≈ ACmap(AC, FL, FR, (M1,M2)) rtol = 1e-4
    loss32(ACb)  = real(sum(Wb .* ACmap_cannon_dist(ACb, FLb, FRb, (M1,M2), g; inner_etype=Float32)))
    lref(AC)     = real(sum(W  .* ACmap(AC, FL, FR, (M1,M2))))
    dAC32 = Zygote.pullback(loss32, ACb)[2](1.0)[1]
    dACr  = Zygote.pullback(lref,  AC)[2](1.0)[1]
    @test eltype(dAC32) == ComplexF64                                            # upcast at exit
    @test dAC32 ≈ dACr[p_rs[g.r1+1], :, :, p_rs[g.r2+1]] rtol = 1e-3            # F32 grad accuracy
end

@testset "ACmap_cannon_dist iterability" begin
    N1 = N2 = 2; χ, D = 16, 3
    AC, FL, FR, M1, M2, _ = make_leg5(χ, D; seed=2800)
    g = cannon_grid(N1, N2)
    ACb=cannon_scatter(AC,g); FLb=cannon_scatter(FL,g); FRb=cannon_scatter(FR,g)
    out_blk = ACmap_cannon_dist(ACb, FLb, FRb, (M1,M2), g)
    out2 = cannon_gather(ACmap_cannon_dist(out_blk, FLb, FRb, (M1,M2), g), g)
    ref = ACmap(AC, FL, FR, (M1,M2))
    @test out2 ≈ ACmap(ref, FL, FR, (M1,M2)) rtol = 1e-11
end

# ─────────────────────────────────────────────────────────────────────────────
# Batch D — ACdmap_cannon_dist (cross-axis gather class, 2-level i/d chunk).
# result[a,b,c,d] := ACd[i,j,k,l] FR[d,g,h,l] M1[e,j,g,b,p] M2[f,k,h,c,p] FL[a,e,f,i]
# Cross-axis the OTHER way vs ACmap: contracted i (ACd.1=r1, FL.4=r2) + output d
# (FR.1=r1, result.4=r2). Square grid only; off-diagonal (a,d) trap guard.
# ─────────────────────────────────────────────────────────────────────────────

@testset "ACdmap_cannon_dist forward parity (square grid)" begin
    N1 = N2 = 2
    for χ in (16, 18), n in (1, 3)
        D = 3
        ACd, FL, FR, M1, M2, _ = make_leg5(χ, D; seed=2900 + χ + n)
        g = cannon_grid(N1, N2)
        ref = ACdmap(ACd, FL, FR, (M1, M2))   # result[a,b,c,d] := ACd[i,j,k,l] FR[d,g,h,l] M1 M2 FL[a,e,f,i]
        ACdb = cannon_scatter(ACd, g); FLb = cannon_scatter(FL, g); FRb = cannon_scatter(FR, g)
        out = cannon_gather(ACdmap_cannon_dist(ACdb, FLb, FRb, (M1,M2), g; forloop_iter=n), g)
        @test out ≈ ref rtol = 1e-12
        if χ == 18   # off-diagonal (a,d) plane — the trap, transposed (§5.1)
            @test out[1:9, :, :, 10:18] ≈ ref[1:9, :, :, 10:18] rtol = 1e-12
            @test out[10:18, :, :, 1:9] ≈ ref[10:18, :, :, 1:9] rtol = 1e-12
        end
        out1 = cannon_gather(ACdmap_cannon_dist(ACdb, FLb, FRb, M1, g), g)
        @test out1 ≈ ACdmap(ACd, FL, FR, M1) rtol = 1e-12
    end
    @test_skip "ACdmap_cannon_dist rectangular grid (N1≠N2) deferred to M3 v2"
end

# ACdmap is NOT self-iterating (output {a,d} top, input {i,l} bottom). Its iterate
# test feeds its output block into a matching ACmap_cannon_dist (whose AC input is
# {a,d}) and compares the composed serial maps.
@testset "ACdmap_cannon_dist composes into ACmap" begin
    N1 = N2 = 2; χ, D = 16, 3
    ACd, FL, FR, M1, M2, _ = make_leg5(χ, D; seed=3100)
    g = cannon_grid(N1, N2)
    ACdb=cannon_scatter(ACd,g); FLb=cannon_scatter(FL,g); FRb=cannon_scatter(FR,g)
    mid_blk = ACdmap_cannon_dist(ACdb, FLb, FRb, (M1,M2), g)   # [a,b,c,d] block
    # the {a,d} output block convention matches ACmap's AC input {a,d} → feed in
    out = cannon_gather(ACmap_cannon_dist(mid_blk, FLb, FRb, (M1,M2), g), g)
    ref_mid = ACdmap(ACd, FL, FR, (M1,M2))
    @test out ≈ ACmap(ref_mid, FL, FR, (M1,M2)) rtol = 1e-11
end

println("rank $rank: test_cannon_m3.jl batch D done")
