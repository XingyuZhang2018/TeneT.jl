# M3 Slice2D distributed maps (Cmap/FR/AC/ACd) — distributed parity vs serial.
# Run via: julia --project=. test/run_test_slice2d_m3.jl  (mpiexec -n 4)
# CPU Arrays only; GPU is examples/MPI_parallel/validation/test_slice2d_m3_sofia.jl (Batch E).
using Test, MPI, LinearAlgebra, Random, Zygote
using TeneT
using TeneT: slice2d_grid, Slice2DGrid, slice2d_scatter, slice2d_gather,
             split_ranges, Cmap, Cmap_slice2d,
             FRmap, FRmap_slice2d_dist,
             ACmap, ACmap_slice2d_dist,
             ACdmap, ACdmap_slice2d_dist

MPI.Init()
const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
@assert MPI.Comm_size(comm) == 4 "test_slice2d_m3.jl expects exactly 4 ranks"

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

@testset "Cmap_slice2d forward parity" begin
    for (N1, N2) in ((2, 2), (1, 4), (4, 1)), χ in (16, 18)
        D = 3
        Random.seed!(2400 + χ + 10N1)
        FL = rand(ComplexF64, χ, D, D, χ)
        FR = rand(ComplexF64, χ, D, D, χ)
        C  = rand(ComplexF64, χ, χ)
        g = slice2d_grid(N1, N2)
        ref = Cmap(C, FL, FR)                 # leg4 Cmap: result[e,f] := FL[a,c,d,e] C[a,b] FR[b,c,d,f]
        out = Cmap_slice2d(C, slice2d_scatter(FL, g), slice2d_scatter(FR, g), g)
        @test size(out) == size(ref)          # FULL χ×χ, replicated (NOT a block)
        @test out ≈ ref rtol = 1e-12
    end
end

@testset "Cmap_slice2d gradient parity" begin
    for (N1, N2) in ((2, 2), (1, 4))
        χ, D = 16, 3
        Random.seed!(2500 + 10N1)
        FL = rand(ComplexF64, χ, D, D, χ); FR = rand(ComplexF64, χ, D, D, χ); C = rand(ComplexF64, χ, χ)
        W  = rand(ComplexF64, χ, χ)
        g = slice2d_grid(N1, N2)
        FLb = slice2d_scatter(FL, g); FRb = slice2d_scatter(FR, g)
        loss_ref(C,FL,FR)   = real(sum(W .* Cmap(C, FL, FR)))
        loss_dist(C,FLb,FRb) = real(sum(W .* Cmap_slice2d(C, FLb, FRb, g)))
        g_ref  = Zygote.pullback(loss_ref,  C, FL, FR)[2](1.0)
        g_dist = Zygote.pullback(loss_dist, C, FLb, FRb)[2](1.0)
        a_rs = split_ranges(χ, N1); e_rs = split_ranges(χ, N2)
        blkof(x) = x[a_rs[g.r1+1], :, :, e_rs[g.r2+1]]
        @test g_dist[1] ≈ g_ref[1] rtol = 1e-10              # dC full (replicated)
        @test g_dist[2] ≈ blkof(g_ref[2]) rtol = 1e-10       # dFL block (take-my-block)
        @test g_dist[3] ≈ blkof(g_ref[3]) rtol = 1e-10       # dFR block (take-my-block)
    end
end

@testset "Cmap_slice2d leg3 parity + replicated-output invariant" begin
    for (N1, N2) in ((2, 2), (1, 4), (4, 1)), χ in (16, 18)
        D = 3
        Random.seed!(2600 + χ + 10N1)
        FL = rand(ComplexF64, χ, D, χ)        # [a, c, d]  (a=r1, d=r2)
        FR = rand(ComplexF64, χ, D, χ)        # [b, c, e]  (b=r1, e=r2)
        C  = rand(ComplexF64, χ, χ)
        g = slice2d_grid(N1, N2)
        ref = Cmap(C, FL, FR)                 # leg3 Cmap: result[d,e] := FL[a,c,d] C[a,b] FR[b,c,e]
        out = Cmap_slice2d(C, slice2d_scatter(FL, g), slice2d_scatter(FR, g), g)
        @test size(out) == size(ref)          # FULL χ×χ
        @test out ≈ ref rtol = 1e-12
        # replicated-output invariant: every rank produced the IDENTICAL full out.
        # rank 0 broadcasts its out; the cross-rank MAX residual must be 0.
        out_bcast = MPI.bcast(out, 0, comm)
        @test MPI.Allreduce(maximum(abs.(out .- out_bcast)), max, comm) ≈ 0 atol = 1e-12
    end
end

@testset "Cmap_slice2d leg3 gradient parity" begin
    for (N1, N2) in ((2, 2), (1, 4))
        χ, D = 16, 3
        Random.seed!(2700 + 10N1)
        FL = rand(ComplexF64, χ, D, χ); FR = rand(ComplexF64, χ, D, χ); C = rand(ComplexF64, χ, χ)
        W  = rand(ComplexF64, χ, χ)
        g = slice2d_grid(N1, N2)
        FLb = slice2d_scatter(FL, g); FRb = slice2d_scatter(FR, g)
        loss_ref(C,FL,FR)    = real(sum(W .* Cmap(C, FL, FR)))
        loss_dist(C,FLb,FRb) = real(sum(W .* Cmap_slice2d(C, FLb, FRb, g)))
        g_ref  = Zygote.pullback(loss_ref,  C, FL, FR)[2](1.0)
        g_dist = Zygote.pullback(loss_dist, C, FLb, FRb)[2](1.0)
        a_rs = split_ranges(χ, N1); e_rs = split_ranges(χ, N2)
        blkof3(x) = x[a_rs[g.r1+1], :, e_rs[g.r2+1]]   # 3-leg take-my-block
        @test g_dist[1] ≈ g_ref[1] rtol = 1e-10              # dC full (replicated)
        @test g_dist[2] ≈ blkof3(g_ref[2]) rtol = 1e-10      # dFL block (take-my-block)
        @test g_dist[3] ≈ blkof3(g_ref[3]) rtol = 1e-10      # dFR block (take-my-block)
    end
end

println("rank $rank: test_slice2d_m3.jl batch A done")

# ─── Batch B: FRmap_slice2d_dist (gather class, square grid) ──────────────────

@testset "FRmap_slice2d_dist forward parity (square grid)" begin
    N1 = N2 = 2
    for χ in (16, 18), n in (1, 3)
        D = 3
        FR, ARu, ARd, M1, M2, _ = make_leg5(χ, D; seed=2100 + χ + n)
        g = slice2d_grid(N1, N2)
        ref = FRmap(FR, ARu, ARd, (M1, M2))   # result[a,e,f,i] := ARd[i,j,k,l] FR[d,g,h,l] M1 M2 ARu[a,b,c,d]
        FRb = slice2d_scatter(FR, g); ARub = slice2d_scatter(ARu, g); ARdb = slice2d_scatter(ARd, g)
        out = slice2d_gather(FRmap_slice2d_dist(FRb, ARub, ARdb, (M1, M2), g; forloop_iter=n), g)
        @test out ≈ ref rtol = 1e-12
        if χ == 18   # off-diagonal (a,i) plane — the trap (a-blk≠i-blk)
            @test out[1:9, :, :, 10:18] ≈ ref[1:9, :, :, 10:18] rtol = 1e-12   # a-blk 0, i-blk 1
            @test out[10:18, :, :, 1:9] ≈ ref[10:18, :, :, 1:9] rtol = 1e-12   # a-blk 1, i-blk 0
        end
        # single-M entry (M2 = conj(M1) internally)
        out1 = slice2d_gather(FRmap_slice2d_dist(FRb, ARub, ARdb, M1, g), g)
        @test out1 ≈ FRmap(FR, ARu, ARd, M1) rtol = 1e-12
    end
    # Rectangular coverage lives in test_slice2d_m3_rect.jl.
end

@testset "FRmap_slice2d_dist gradient parity (square grid)" begin
    N1 = N2 = 2
    for χ in (16, 18), n in (1, 3)
        D = 3
        FR, ARu, ARd, M1, M2, W = make_leg5(χ, D; seed=2200 + χ + n)
        g = slice2d_grid(N1, N2)
        FRb = slice2d_scatter(FR, g); ARub = slice2d_scatter(ARu, g); ARdb = slice2d_scatter(ARd, g)
        Wb = slice2d_scatter(W, g)
        loss_ref(FR,ARu,ARd,M1,M2)  = real(sum(W  .* FRmap(FR,ARu,ARd,(M1,M2))))
        loss_dist(FRb,ARub,ARdb,M1,M2) = real(sum(Wb .* FRmap_slice2d_dist(FRb,ARub,ARdb,(M1,M2),g; forloop_iter=n)))
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
    g = slice2d_grid(2, 2)
    FRb=slice2d_scatter(FR,g); ARub=slice2d_scatter(ARu,g); ARdb=slice2d_scatter(ARd,g); Wb=slice2d_scatter(W,g)
    lr(FR,M) = real(sum(W  .* FRmap(FR,ARu,ARd,M)))
    ld(FRb,M)= real(sum(Wb .* FRmap_slice2d_dist(FRb,ARub,ARdb,M,g)))
    gr = Zygote.pullback(lr, FR, M1)[2](1.0); gd = Zygote.pullback(ld, FRb, M1)[2](1.0)
    p_rs = split_ranges(16, 2)
    @test gd[1] ≈ gr[1][p_rs[g.r1+1], :, :, p_rs[g.r2+1]] rtol = 1e-10
    @test gd[2] ≈ gr[2] rtol = 1e-10
    # bare-sum loss (FillArrays densify guard)
    back = Zygote.pullback(x -> real(sum(FRmap_slice2d_dist(x, ARub, ARdb, (M1,M2), g))), FRb)[2]
    dblk = back(1.0)[1]
    dFR_ref = Zygote.pullback(x -> real(sum(FRmap(x, ARu, ARd, M1, M2))), FR)[2](1.0)[1]
    @test dblk ≈ dFR_ref[p_rs[g.r1+1], :, :, p_rs[g.r2+1]] rtol = 1e-10
    # inner_etype Float32 boundary cast — forward (1e-4) AND gradient (1e-3),
    # both halves like the FLmap template (test_slice2d.jl:158-175): the gradient
    # half exercises the rrule do_cast branch (_boundary_cast on the cotangent +
    # T_orig upcast of dFR/dARu/dARd/dM), else that path ships untested.
    out32 = slice2d_gather(FRmap_slice2d_dist(FRb, ARub, ARdb, (M1,M2), g; inner_etype=Float32), g)
    @test eltype(out32) == ComplexF64
    @test out32 ≈ FRmap(FR, ARu, ARd, (M1,M2)) rtol = 1e-4
    loss32(FRb)  = real(sum(Wb .* FRmap_slice2d_dist(FRb, ARub, ARdb, (M1,M2), g; inner_etype=Float32)))
    lref(FR)     = real(sum(W  .* FRmap(FR, ARu, ARd, (M1,M2))))
    dFR32 = Zygote.pullback(loss32, FRb)[2](1.0)[1]
    dFRr  = Zygote.pullback(lref,  FR)[2](1.0)[1]
    @test eltype(dFR32) == ComplexF64                                            # upcast at exit
    @test dFR32 ≈ dFRr[p_rs[g.r1+1], :, :, p_rs[g.r2+1]] rtol = 1e-3            # F32 grad accuracy
end

@testset "FRmap_slice2d_dist forloop_iter clamp (square grid)" begin
    N1 = N2 = 2
    χ, D = 18, 3
    FR, ARu, ARd, M1, M2, _ = make_leg5(χ, D; seed=2300)
    g = slice2d_grid(N1, N2)
    ref = FRmap(FR, ARu, ARd, (M1, M2))
    FRb=slice2d_scatter(FR,g); ARub=slice2d_scatter(ARu,g); ARdb=slice2d_scatter(ARd,g)
    out = slice2d_gather(FRmap_slice2d_dist(FRb, ARub, ARdb, (M1,M2), g; forloop_iter=99), g)
    @test out ≈ ref rtol = 1e-12
end

println("rank $rank: test_slice2d_m3.jl batch B done")

# ─── Batch C: ACmap_slice2d_dist (gather class, SINGLE l-chunk, square grid) ───

@testset "ACmap_slice2d_dist forward parity (square grid)" begin
    N1 = N2 = 2
    for χ in (16, 18), n in (1, 3)
        D = 3
        AC, FL, FR, M1, M2, _ = make_leg5(χ, D; seed=2600 + χ + n)
        g = slice2d_grid(N1, N2)
        ref = ACmap(AC, FL, FR, (M1, M2))      # result[i,j,k,l] := AC[a,b,c,d] FR[d,g,h,l] M1 M2 FL[a,e,f,i]
        ACb = slice2d_scatter(AC, g); FLb = slice2d_scatter(FL, g); FRb = slice2d_scatter(FR, g)
        out = slice2d_gather(ACmap_slice2d_dist(ACb, FLb, FRb, (M1,M2), g; forloop_iter=n), g)
        @test out ≈ ref rtol = 1e-12
        # CRITICAL off-diagonal (the diagonal trap): with χ=18, N=2 →
        # p_rs = [1:9, 10:18]; check off-diagonal (i,l) blocks A≠B explicitly.
        if χ == 18
            @test out[1:9, :, :, 10:18] ≈ ref[1:9, :, :, 10:18] rtol = 1e-12   # i-blk 0, l-blk 1
            @test out[10:18, :, :, 1:9] ≈ ref[10:18, :, :, 1:9] rtol = 1e-12   # i-blk 1, l-blk 0
        end
        # single-M entry
        out1 = slice2d_gather(ACmap_slice2d_dist(ACb, FLb, FRb, M1, g), g)
        @test out1 ≈ ACmap(AC, FL, FR, M1) rtol = 1e-12
    end
    # Rectangular coverage lives in test_slice2d_m3_rect.jl.
end

@testset "ACmap_slice2d_dist gradient parity (square grid)" begin
    N1 = N2 = 2
    for χ in (16, 18), n in (1, 3)
        D = 3
        AC, FL, FR, M1, M2, W = make_leg5(χ, D; seed=2700 + χ + n)
        g = slice2d_grid(N1, N2)
        ACb=slice2d_scatter(AC,g); FLb=slice2d_scatter(FL,g); FRb=slice2d_scatter(FR,g); Wb=slice2d_scatter(W,g)
        loss_ref(AC,FL,FR,M1,M2)  = real(sum(W  .* ACmap(AC,FL,FR,(M1,M2))))
        loss_dist(ACb,FLb,FRb,M1,M2) = real(sum(Wb .* ACmap_slice2d_dist(ACb,FLb,FRb,(M1,M2),g; forloop_iter=n)))
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
    g = slice2d_grid(2, 2)
    ACb=slice2d_scatter(AC,g); FLb=slice2d_scatter(FL,g); FRb=slice2d_scatter(FR,g); Wb=slice2d_scatter(W,g)
    lr(AC,M) = real(sum(W  .* ACmap(AC,FL,FR,M)))
    ld(ACb,M)= real(sum(Wb .* ACmap_slice2d_dist(ACb,FLb,FRb,M,g)))
    gr = Zygote.pullback(lr, AC, M1)[2](1.0); gd = Zygote.pullback(ld, ACb, M1)[2](1.0)
    p_rs = split_ranges(16, 2)
    @test gd[1] ≈ gr[1][p_rs[g.r1+1], :, :, p_rs[g.r2+1]] rtol = 1e-10
    @test gd[2] ≈ gr[2] rtol = 1e-10
    # bare-sum loss (FillArrays densify guard)
    back = Zygote.pullback(x -> real(sum(ACmap_slice2d_dist(x, FLb, FRb, (M1,M2), g))), ACb)[2]
    dblk = back(1.0)[1]
    dAC_ref = Zygote.pullback(x -> real(sum(ACmap(x, FL, FR, M1, M2))), AC)[2](1.0)[1]
    @test dblk ≈ dAC_ref[p_rs[g.r1+1], :, :, p_rs[g.r2+1]] rtol = 1e-10
    # inner_etype Float32 boundary cast — forward (1e-4) AND gradient (1e-3),
    # both halves like the FLmap/FRmap template: the gradient half exercises the
    # rrule do_cast branch (_boundary_cast on the cotangent + T_orig upcast of
    # dAC/dFL/dFR/dM), else that path ships untested.
    out32 = slice2d_gather(ACmap_slice2d_dist(ACb, FLb, FRb, (M1,M2), g; inner_etype=Float32), g)
    @test eltype(out32) == ComplexF64
    @test out32 ≈ ACmap(AC, FL, FR, (M1,M2)) rtol = 1e-4
    loss32(ACb)  = real(sum(Wb .* ACmap_slice2d_dist(ACb, FLb, FRb, (M1,M2), g; inner_etype=Float32)))
    lref(AC)     = real(sum(W  .* ACmap(AC, FL, FR, (M1,M2))))
    dAC32 = Zygote.pullback(loss32, ACb)[2](1.0)[1]
    dACr  = Zygote.pullback(lref,  AC)[2](1.0)[1]
    @test eltype(dAC32) == ComplexF64                                            # upcast at exit
    @test dAC32 ≈ dACr[p_rs[g.r1+1], :, :, p_rs[g.r2+1]] rtol = 1e-3            # F32 grad accuracy
end

@testset "ACmap_slice2d_dist iterability" begin
    N1 = N2 = 2; χ, D = 16, 3
    AC, FL, FR, M1, M2, _ = make_leg5(χ, D; seed=2800)
    g = slice2d_grid(N1, N2)
    ACb=slice2d_scatter(AC,g); FLb=slice2d_scatter(FL,g); FRb=slice2d_scatter(FR,g)
    out_blk = ACmap_slice2d_dist(ACb, FLb, FRb, (M1,M2), g)
    out2 = slice2d_gather(ACmap_slice2d_dist(out_blk, FLb, FRb, (M1,M2), g), g)
    ref = ACmap(AC, FL, FR, (M1,M2))
    @test out2 ≈ ACmap(ref, FL, FR, (M1,M2)) rtol = 1e-11
end

# ─────────────────────────────────────────────────────────────────────────────
# Batch D — ACdmap_slice2d_dist (cross-axis gather class, 2-level i/d chunk).
# result[a,b,c,d] := ACd[i,j,k,l] FR[d,g,h,l] M1[e,j,g,b,p] M2[f,k,h,c,p] FL[a,e,f,i]
# Cross-axis the OTHER way vs ACmap: contracted i (ACd.1=r1, FL.4=r2) + output d
# (FR.1=r1, result.4=r2). The 2x2 case below guards the off-diagonal (a,d) trap;
# rectangular coverage lives in test_slice2d_m3_rect.jl.
# ─────────────────────────────────────────────────────────────────────────────

@testset "ACdmap_slice2d_dist forward parity (square grid)" begin
    N1 = N2 = 2
    for χ in (16, 18), n in (1, 3)
        D = 3
        ACd, FL, FR, M1, M2, _ = make_leg5(χ, D; seed=2900 + χ + n)
        g = slice2d_grid(N1, N2)
        ref = ACdmap(ACd, FL, FR, (M1, M2))   # result[a,b,c,d] := ACd[i,j,k,l] FR[d,g,h,l] M1 M2 FL[a,e,f,i]
        ACdb = slice2d_scatter(ACd, g); FLb = slice2d_scatter(FL, g); FRb = slice2d_scatter(FR, g)
        out = slice2d_gather(ACdmap_slice2d_dist(ACdb, FLb, FRb, (M1,M2), g; forloop_iter=n), g)
        @test out ≈ ref rtol = 1e-12
        if χ == 18   # off-diagonal (a,d) plane — the trap, transposed (§5.1)
            @test out[1:9, :, :, 10:18] ≈ ref[1:9, :, :, 10:18] rtol = 1e-12
            @test out[10:18, :, :, 1:9] ≈ ref[10:18, :, :, 1:9] rtol = 1e-12
        end
        out1 = slice2d_gather(ACdmap_slice2d_dist(ACdb, FLb, FRb, M1, g), g)
        @test out1 ≈ ACdmap(ACd, FL, FR, M1) rtol = 1e-12
    end
    # Rectangular coverage lives in test_slice2d_m3_rect.jl.
end

@testset "ACdmap_slice2d_dist gradient parity (square grid)" begin
    N1 = N2 = 2
    for χ in (16, 18), n in (1, 3)
        D = 3
        ACd, FL, FR, M1, M2, W = make_leg5(χ, D; seed=3000 + χ + n)
        g = slice2d_grid(N1, N2)
        ACdb=slice2d_scatter(ACd,g); FLb=slice2d_scatter(FL,g); FRb=slice2d_scatter(FR,g); Wb=slice2d_scatter(W,g)
        loss_ref(ACd,FL,FR,M1,M2)  = real(sum(W  .* ACdmap(ACd,FL,FR,(M1,M2))))
        loss_dist(ACdb,FLb,FRb,M1,M2) = real(sum(Wb .* ACdmap_slice2d_dist(ACdb,FLb,FRb,(M1,M2),g; forloop_iter=n)))
        g_ref  = Zygote.pullback(loss_ref,  ACd,FL,FR,M1,M2)[2](1.0)
        g_dist = Zygote.pullback(loss_dist, ACdb,FLb,FRb,M1,M2)[2](1.0)
        p_rs = split_ranges(χ, N1)
        blkof(x) = x[p_rs[g.r1+1], :, :, p_rs[g.r2+1]]
        @test g_dist[1] ≈ blkof(g_ref[1]) rtol = 1e-10   # dACd block
        @test g_dist[2] ≈ blkof(g_ref[2]) rtol = 1e-10   # dFL block
        @test g_dist[3] ≈ blkof(g_ref[3]) rtol = 1e-10   # dFR block
        @test g_dist[4] ≈ g_ref[4] rtol = 1e-10          # dM1 replicated
        @test g_dist[5] ≈ g_ref[5] rtol = 1e-10          # dM2
    end
end

@testset "ACdmap_slice2d_dist single-M / densify / inner_etype" begin
    N1 = N2 = 2
    ACd, FL, FR, M1, M2, W = make_leg5(16, 3; seed=3040)
    g = slice2d_grid(N1, N2)
    ACdb=slice2d_scatter(ACd,g); FLb=slice2d_scatter(FL,g); FRb=slice2d_scatter(FR,g); Wb=slice2d_scatter(W,g)
    # single-M dM = dM1 + conj(dM2) composition
    lr(ACd,M) = real(sum(W  .* ACdmap(ACd,FL,FR,M)))
    ld(ACdb,M)= real(sum(Wb .* ACdmap_slice2d_dist(ACdb,FLb,FRb,M,g)))
    gr = Zygote.pullback(lr, ACd, M1)[2](1.0); gd = Zygote.pullback(ld, ACdb, M1)[2](1.0)
    p_rs = split_ranges(16, 2)
    @test gd[1] ≈ gr[1][p_rs[g.r1+1], :, :, p_rs[g.r2+1]] rtol = 1e-10
    @test gd[2] ≈ gr[2] rtol = 1e-10
    # bare-sum loss (FillArrays densify guard)
    back = Zygote.pullback(x -> real(sum(ACdmap_slice2d_dist(x, FLb, FRb, (M1,M2), g))), ACdb)[2]
    dblk = back(1.0)[1]
    dACd_ref = Zygote.pullback(x -> real(sum(ACdmap(x, FL, FR, M1, M2))), ACd)[2](1.0)[1]
    @test dblk ≈ dACd_ref[p_rs[g.r1+1], :, :, p_rs[g.r2+1]] rtol = 1e-10
    # inner_etype Float32 boundary cast — forward (1e-4) AND gradient (1e-3), both
    # halves: the gradient half exercises the rrule do_cast branch (_boundary_cast
    # on the cotangent + T_orig upcast of dACd/dFL/dFR/dM).
    out32 = slice2d_gather(ACdmap_slice2d_dist(ACdb, FLb, FRb, (M1,M2), g; inner_etype=Float32), g)
    @test eltype(out32) == ComplexF64
    @test out32 ≈ ACdmap(ACd, FL, FR, (M1,M2)) rtol = 1e-4
    loss32(ACdb) = real(sum(Wb .* ACdmap_slice2d_dist(ACdb, FLb, FRb, (M1,M2), g; inner_etype=Float32)))
    lref(ACd)    = real(sum(W  .* ACdmap(ACd, FL, FR, (M1,M2))))
    dACd32 = Zygote.pullback(loss32, ACdb)[2](1.0)[1]
    dACdr  = Zygote.pullback(lref,  ACd)[2](1.0)[1]
    @test eltype(dACd32) == ComplexF64                                           # upcast at exit
    @test dACd32 ≈ dACdr[p_rs[g.r1+1], :, :, p_rs[g.r2+1]] rtol = 1e-3          # F32 grad accuracy
end

# Non-uniform-bond test (the dim guard the plan review mandated): the output
# middle legs b = size(M1,4), c = size(M2,4) are masked by a uniform D=3 test.
# Build M1 with 4th-dim Db and M2 with 4th-dim Dc, Db≠Dc, so a wrong index
# (size(M1,2)/size(M2,2) = j/k, the ACmap output legs) cannot hide. Forward AND
# gradient parity.
@testset "ACdmap_slice2d_dist non-uniform bond (Db≠Dc) parity" begin
    N1 = N2 = 2; χ = 16
    # distinct M-slot bonds: M1=(e,j,g,b,p), M2=(f,k,h,c,p); b=5 (Db), c=7 (Dc), b≠c.
    De, Df, Dj, Dk, Dg, Dh, Db, Dc, Dp = 2, 2, 3, 3, 4, 4, 5, 7, 6
    Random.seed!(3050)
    M1 = rand(ComplexF64, De, Dj, Dg, Db, Dp)
    M2 = rand(ComplexF64, Df, Dk, Dh, Dc, Dp)
    ACd = rand(ComplexF64, χ, Dj, Dk, χ)         # (i,j,k,l)
    FL  = rand(ComplexF64, χ, De, Df, χ)         # (a,e,f,i)
    FR  = rand(ComplexF64, χ, Dg, Dh, χ)         # (d,g,h,l)
    W   = rand(ComplexF64, χ, Db, Dc, χ)         # out (a,b,c,d)
    g = slice2d_grid(N1, N2)
    ACdb=slice2d_scatter(ACd,g); FLb=slice2d_scatter(FL,g); FRb=slice2d_scatter(FR,g); Wb=slice2d_scatter(W,g)
    ref = ACdmap(ACd, FL, FR, (M1, M2))
    @test size(ref) == (χ, Db, Dc, χ)            # b=size(M1,4)=5, c=size(M2,4)=7 — the dim fix
    out = slice2d_gather(ACdmap_slice2d_dist(ACdb, FLb, FRb, (M1,M2), g), g)
    @test out ≈ ref rtol = 1e-12
    loss_ref(ACd,FL,FR,M1,M2)  = real(sum(W  .* ACdmap(ACd,FL,FR,(M1,M2))))
    loss_dist(ACdb,FLb,FRb,M1,M2) = real(sum(Wb .* ACdmap_slice2d_dist(ACdb,FLb,FRb,(M1,M2),g)))
    g_ref  = Zygote.pullback(loss_ref,  ACd,FL,FR,M1,M2)[2](1.0)
    g_dist = Zygote.pullback(loss_dist, ACdb,FLb,FRb,M1,M2)[2](1.0)
    p_rs = split_ranges(χ, N1); blkof(x) = x[p_rs[g.r1+1], :, :, p_rs[g.r2+1]]
    @test g_dist[1] ≈ blkof(g_ref[1]) rtol = 1e-10
    @test g_dist[4] ≈ g_ref[4] rtol = 1e-10
    @test g_dist[5] ≈ g_ref[5] rtol = 1e-10
end

# ACdmap is NOT self-iterating (output {a,d} top, input {i,l} bottom). Its iterate
# test feeds its output block into a matching ACmap_slice2d_dist (whose AC input is
# {a,d}) and compares the composed serial maps.
@testset "ACdmap_slice2d_dist composes into ACmap" begin
    N1 = N2 = 2; χ, D = 16, 3
    ACd, FL, FR, M1, M2, _ = make_leg5(χ, D; seed=3100)
    g = slice2d_grid(N1, N2)
    ACdb=slice2d_scatter(ACd,g); FLb=slice2d_scatter(FL,g); FRb=slice2d_scatter(FR,g)
    mid_blk = ACdmap_slice2d_dist(ACdb, FLb, FRb, (M1,M2), g)   # [a,b,c,d] block
    # the {a,d} output block convention matches ACmap's AC input {a,d} → feed in
    out = slice2d_gather(ACmap_slice2d_dist(mid_blk, FLb, FRb, (M1,M2), g), g)
    ref_mid = ACdmap(ACd, FL, FR, (M1,M2))
    @test out ≈ ACmap(ref_mid, FL, FR, (M1,M2)) rtol = 1e-11
end

println("rank $rank: test_slice2d_m3.jl batch D done")
