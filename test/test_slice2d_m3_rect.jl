# Focused rectangular-grid coverage for M3 Slice2D distributed maps.
# Run via: julia --project=. test/run_test_slice2d_m3_rect.jl
using Test, MPI, Random, Zygote
using TeneT
using TeneT: slice2d_grid, slice2d_scatter, slice2d_gather, split_ranges,
             slice2d_gather_row, slice2d_gather_col,
             FRmap, FRmap_slice2d_dist, FRmap_slice2d_sliced,
             ACmap, ACmap_slice2d_dist, ACmap_slice2d_sliced,
             ACdmap, ACdmap_slice2d_dist

MPI.Init()
const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
const N1 = parse(Int, get(ENV, "TENET_SLICE2D_N1", "2"))
const N2 = parse(Int, get(ENV, "TENET_SLICE2D_N2", "3"))
@assert N1 != N2 "test_slice2d_m3_rect.jl is specifically for rectangular grids"
@assert MPI.Comm_size(comm) == N1 * N2 "expected $(N1 * N2) ranks for $(N1)x$(N2)"

function make_leg5(χ, D; d=2, seed=42)
    Random.seed!(seed)
    A  = rand(ComplexF64, χ, D, D, χ)
    B  = rand(ComplexF64, χ, D, D, χ)
    C  = rand(ComplexF64, χ, D, D, χ)
    M1 = rand(ComplexF64, D, D, D, D, d)
    M2 = rand(ComplexF64, D, D, D, D, d)
    W  = rand(ComplexF64, χ, D, D, χ)
    return A, B, C, M1, M2, W
end

function block_of(x, g, χ)
    a_rs = split_ranges(χ, g.N1)
    l_rs = split_ranges(χ, g.N2)
    return x[a_rs[g.r1 + 1], :, :, l_rs[g.r2 + 1]]
end

@testset "FRmap_slice2d_dist rectangular forward and gradient parity" begin
    χ, D = 7, 2
    FR, ARu, ARd, M1, M2, W = make_leg5(χ, D; seed=4101)
    g = slice2d_grid(N1, N2)
    FRb = slice2d_scatter(FR, g)
    ARub = slice2d_scatter(ARu, g)
    ARdb = slice2d_scatter(ARd, g)
    Wb = slice2d_scatter(W, g)

    out = slice2d_gather(FRmap_slice2d_dist(FRb, ARub, ARdb, (M1, M2), g; forloop_iter=2), g)
    @test out ≈ FRmap(FR, ARu, ARd, (M1, M2)) rtol=1e-12

    loss_ref(FR, ARu, ARd, M1, M2) = real(sum(W .* FRmap(FR, ARu, ARd, (M1, M2))))
    loss_dist(FRb, ARub, ARdb, M1, M2) =
        real(sum(Wb .* FRmap_slice2d_dist(FRb, ARub, ARdb, (M1, M2), g; forloop_iter=2)))
    g_ref = Zygote.pullback(loss_ref, FR, ARu, ARd, M1, M2)[2](1.0)
    g_dist = Zygote.pullback(loss_dist, FRb, ARub, ARdb, M1, M2)[2](1.0)
    @test g_dist[1] ≈ block_of(g_ref[1], g, χ) rtol=1e-10
    @test g_dist[2] ≈ block_of(g_ref[2], g, χ) rtol=1e-10
    @test g_dist[3] ≈ block_of(g_ref[3], g, χ) rtol=1e-10
    @test g_dist[4] ≈ g_ref[4] rtol=1e-10
    @test g_dist[5] ≈ g_ref[5] rtol=1e-10
end

@testset "FRmap_slice2d_sliced rectangular equivalence" begin
    χ, D = 7, 2
    FR, ARu, ARd, M1, M2, W = make_leg5(χ, D; seed=4151)
    g = slice2d_grid(N1, N2)
    a_rs = split_ranges(χ, g.N1)
    l_rs = split_ranges(χ, g.N2)
    FRb = slice2d_scatter(FR, g)
    ARub = slice2d_scatter(ARu, g)
    ARdb = slice2d_scatter(ARd, g)
    Wb = slice2d_scatter(W, g)
    ARu_g = slice2d_gather_row(ARub, g, l_rs)
    ARd_g = slice2d_gather_col(ARdb, g, a_rs)

    out_s = FRmap_slice2d_sliced(FRb, ARu_g, ARd_g, (M1, M2), g; forloop_iter=2)
    out_d = FRmap_slice2d_dist(FRb, ARub, ARdb, (M1, M2), g; forloop_iter=2)
    @test out_s ≈ out_d rtol=1e-13

    loss_s(FRb, ARub, ARdb) =
        real(sum(Wb .* FRmap_slice2d_sliced(FRb,
            slice2d_gather_row(ARub, g, l_rs),
            slice2d_gather_col(ARdb, g, a_rs),
            (M1, M2), g; forloop_iter=2)))
    loss_d(FRb, ARub, ARdb) =
        real(sum(Wb .* FRmap_slice2d_dist(FRb, ARub, ARdb, (M1, M2), g; forloop_iter=2)))
    g_s = Zygote.gradient(loss_s, FRb, ARub, ARdb)
    g_d = Zygote.gradient(loss_d, FRb, ARub, ARdb)
    for k in 1:3
        @test g_s[k] ≈ g_d[k] rtol=1e-10
    end
end

@testset "ACmap_slice2d_dist rectangular forward and gradient parity" begin
    χ, D = 7, 2
    AC, FL, FR, M1, M2, W = make_leg5(χ, D; seed=4201)
    g = slice2d_grid(N1, N2)
    ACb = slice2d_scatter(AC, g)
    FLb = slice2d_scatter(FL, g)
    FRb = slice2d_scatter(FR, g)
    Wb = slice2d_scatter(W, g)

    out = slice2d_gather(ACmap_slice2d_dist(ACb, FLb, FRb, (M1, M2), g; forloop_iter=2), g)
    @test out ≈ ACmap(AC, FL, FR, (M1, M2)) rtol=1e-12

    loss_ref(AC, FL, FR, M1, M2) = real(sum(W .* ACmap(AC, FL, FR, (M1, M2))))
    loss_dist(ACb, FLb, FRb, M1, M2) =
        real(sum(Wb .* ACmap_slice2d_dist(ACb, FLb, FRb, (M1, M2), g; forloop_iter=2)))
    g_ref = Zygote.pullback(loss_ref, AC, FL, FR, M1, M2)[2](1.0)
    g_dist = Zygote.pullback(loss_dist, ACb, FLb, FRb, M1, M2)[2](1.0)
    @test g_dist[1] ≈ block_of(g_ref[1], g, χ) rtol=1e-10
    @test g_dist[2] ≈ block_of(g_ref[2], g, χ) rtol=1e-10
    @test g_dist[3] ≈ block_of(g_ref[3], g, χ) rtol=1e-10
    @test g_dist[4] ≈ g_ref[4] rtol=1e-10
    @test g_dist[5] ≈ g_ref[5] rtol=1e-10
end

@testset "ACmap_slice2d_sliced rectangular equivalence" begin
    χ, D = 7, 2
    AC, FL, FR, M1, M2, W = make_leg5(χ, D; seed=4251)
    g = slice2d_grid(N1, N2)
    a_rs = split_ranges(χ, g.N1)
    l_rs = split_ranges(χ, g.N2)
    ACb = slice2d_scatter(AC, g)
    FLb = slice2d_scatter(FL, g)
    FRb = slice2d_scatter(FR, g)
    Wb = slice2d_scatter(W, g)
    FL_g = slice2d_gather_row(FLb, g, l_rs)
    FR_g = slice2d_gather_col(FRb, g, a_rs)

    out_s = ACmap_slice2d_sliced(ACb, FL_g, FR_g, (M1, M2), g; forloop_iter=2)
    out_d = ACmap_slice2d_dist(ACb, FLb, FRb, (M1, M2), g; forloop_iter=2)
    @test out_s ≈ out_d rtol=1e-13

    loss_s(ACb, FLb, FRb) =
        real(sum(Wb .* ACmap_slice2d_sliced(ACb,
            slice2d_gather_row(FLb, g, l_rs),
            slice2d_gather_col(FRb, g, a_rs),
            (M1, M2), g; forloop_iter=2)))
    loss_d(ACb, FLb, FRb) =
        real(sum(Wb .* ACmap_slice2d_dist(ACb, FLb, FRb, (M1, M2), g; forloop_iter=2)))
    g_s = Zygote.gradient(loss_s, ACb, FLb, FRb)
    g_d = Zygote.gradient(loss_d, ACb, FLb, FRb)
    for k in 1:3
        @test g_s[k] ≈ g_d[k] rtol=1e-10
    end
end

@testset "ACdmap_slice2d_dist rectangular forward and gradient parity" begin
    χ, D = 7, 2
    ACd, FL, FR, M1, M2, W = make_leg5(χ, D; seed=4301)
    g = slice2d_grid(N1, N2)
    ACdb = slice2d_scatter(ACd, g)
    FLb = slice2d_scatter(FL, g)
    FRb = slice2d_scatter(FR, g)
    Wb = slice2d_scatter(W, g)

    out = slice2d_gather(ACdmap_slice2d_dist(ACdb, FLb, FRb, (M1, M2), g; forloop_iter=2), g)
    @test out ≈ ACdmap(ACd, FL, FR, (M1, M2)) rtol=1e-12

    loss_ref(ACd, FL, FR, M1, M2) = real(sum(W .* ACdmap(ACd, FL, FR, (M1, M2))))
    loss_dist(ACdb, FLb, FRb, M1, M2) =
        real(sum(Wb .* ACdmap_slice2d_dist(ACdb, FLb, FRb, (M1, M2), g; forloop_iter=2)))
    g_ref = Zygote.pullback(loss_ref, ACd, FL, FR, M1, M2)[2](1.0)
    g_dist = Zygote.pullback(loss_dist, ACdb, FLb, FRb, M1, M2)[2](1.0)
    @test g_dist[1] ≈ block_of(g_ref[1], g, χ) rtol=1e-10
    @test g_dist[2] ≈ block_of(g_ref[2], g, χ) rtol=1e-10
    @test g_dist[3] ≈ block_of(g_ref[3], g, χ) rtol=1e-10
    @test g_dist[4] ≈ g_ref[4] rtol=1e-10
    @test g_dist[5] ≈ g_ref[5] rtol=1e-10
end

println("rank $rank: test_slice2d_m3_rect.jl done")
