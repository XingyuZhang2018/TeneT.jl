using Test
using TeneT
using MPI
using Zygote

using TeneT: ParallelMethod, Slice1DMethod, Slice2DMethod,
             SerialMethod, slice1D, slice2D, slice2d_grid,
             FLmap, FLmap_parallel, FLmap_slice2d_dist,
             Mumap, Mumap_slice2d_dist,
             parallel_map, slice2d_scatter

@testset "parallel method public API" begin
    probe = Module(:ParallelAPIExportProbe)
    Core.eval(probe, :(using TeneT))
    for name in (:ParallelMethod, :SerialMethod, :Slice1DMethod, :Slice2DMethod, :slice1D, :slice2D)
        @test isdefined(probe, name)
    end

    m1 = slice1D(; forloop_iter=3)
    @test m1 isa ParallelMethod
    @test m1 isa Slice1DMethod
    @test m1.forloop_iter == 3
    @test m1.inner_etype === nothing

    if !MPI.Initialized()
        MPI.Init()
    end
    if MPI.Comm_size(MPI.COMM_WORLD) == 1
        m2 = slice2D(1, 1; forloop_iter=2)
        @test m2 isa ParallelMethod
        @test m2 isa Slice2DMethod
        @test m2.forloop_iter == 2
        @test m2.grid === slice2d_grid(1, 1)
    end
end

@testset "slice2D routing matches legacy distributed wrapper" begin
    if !MPI.Initialized()
        MPI.Init()
    end
    if MPI.Comm_size(MPI.COMM_WORLD) == 1
        method = slice2D(1, 1; forloop_iter=1)
        FL = rand(ComplexF64, 2, 2, 2, 2)
        ALu = rand(ComplexF64, 2, 2, 2, 2)
        ALd = rand(ComplexF64, 2, 2, 2, 2)
        M = rand(ComplexF64, 2, 2, 2, 2, 2)
        FLb = slice2d_scatter(FL, method.grid)
        ALub = slice2d_scatter(ALu, method.grid)
        ALdb = slice2d_scatter(ALd, method.grid)

        legacy = FLmap_slice2d_dist(FLb, ALub, ALdb, M, method.grid; forloop_iter=1)
        routed = parallel_map(FLmap, method, FLb, ALub, ALdb, M)
        @test routed ≈ legacy

        AC = rand(ComplexF64, 2, 2, 2, 2)
        ACd = rand(ComplexF64, 2, 2, 2, 2)
        FR = rand(ComplexF64, 2, 2, 2, 2)
        Mu = rand(ComplexF64, 2, 2, 2, 2, 2)
        ACb = slice2d_scatter(AC, method.grid)
        ACdb = slice2d_scatter(ACd, method.grid)
        FRb = slice2d_scatter(FR, method.grid)

        legacy_mu = Mumap_slice2d_dist(ACb, ACdb, FLb, FRb, Mu, method.grid; forloop_iter=1)
        routed_mu = parallel_map(Mumap, method, ACb, ACdb, FLb, FRb, Mu)
        @test routed_mu ≈ legacy_mu
    end
end

@testset "slice1D routing matches legacy local wrapper" begin
    FL = rand(ComplexF64, 2, 2, 2, 2)
    ALu = rand(ComplexF64, 2, 2, 2, 2)
    ALd = rand(ComplexF64, 2, 2, 2, 2)
    M = rand(ComplexF64, 2, 2, 2, 2, 2)

    legacy = FLmap_parallel(FL, ALu, ALd, M; ifparallel=false, forloop_iter=2)
    routed = parallel_map(FLmap, SerialMethod(2, nothing), FL, ALu, ALd, M)
    @test routed ≈ legacy
end

@testset "slice1D forwards explicit communicator" begin
    if !MPI.Initialized()
        MPI.Init()
    end
    comm = MPI.Comm_split(MPI.COMM_WORLD, 0, MPI.Comm_rank(MPI.COMM_WORLD))
    method = slice1D(; comm)
    @test method.comm == comm

    FL = rand(ComplexF64, 2, 2, 2, 2)
    ALu = rand(ComplexF64, 2, 2, 2, 2)
    ALd = rand(ComplexF64, 2, 2, 2, 2)
    M = rand(ComplexF64, 2, 2, 2, 2, 2)

    legacy = FLmap_parallel(FL, ALu, ALd, M; ifparallel=true, forloop_iter=1, comm)
    routed = parallel_map(FLmap, method, FL, ALu, ALd, M)
    @test routed ≈ legacy
end

@testset "slice1D explicit communicator supports AD" begin
    if !MPI.Initialized()
        MPI.Init()
    end
    comm = MPI.Comm_split(MPI.COMM_WORLD, 0, MPI.Comm_rank(MPI.COMM_WORLD))
    method = slice1D(; comm)

    FL = rand(ComplexF64, 2, 2, 2, 2)
    ALu = rand(ComplexF64, 2, 2, 2, 2)
    ALd = rand(ComplexF64, 2, 2, 2, 2)
    M = rand(ComplexF64, 2, 2, 2, 2, 2)

    serial_loss(fl, alu, ald, m) =
        real(sum(abs2, FLmap_parallel(fl, alu, ald, m; ifparallel=false, forloop_iter=1)))
    slice1d_loss(fl, alu, ald, m) =
        real(sum(abs2, parallel_map(FLmap, method, fl, alu, ald, m)))

    g_serial = Zygote.gradient(serial_loss, FL, ALu, ALd, M)
    g_slice1d = Zygote.gradient(slice1d_loss, FL, ALu, ALd, M)

    for (gs, gp) in zip(g_serial, g_slice1d)
        @test gp ≈ gs
    end
end

@testset "VUMPS accepts parallel_method" begin
    alg = VUMPS{General}(parallel_method=slice1D(; forloop_iter=2))
    @test alg.parallel_method isa Slice1DMethod
    @test alg.parallel_method.forloop_iter == 2

    TeneT._apply_parallel_method!(alg)
    @test alg.ifparallel
    @test alg.forloop_iter == 2

    if !MPI.Initialized()
        MPI.Init()
    end
    if MPI.Comm_size(MPI.COMM_WORLD) == 1
        alg2 = VUMPS{General}(parallel_method=slice2D(1, 1; forloop_iter=3))
        TeneT._apply_parallel_method!(alg2)
        @test !alg2.ifparallel
        @test alg2.forloop_iter == 3
        @test TeneT._effective_grid(alg2) === alg2.parallel_method.grid
        @test alg2.grid === alg2.parallel_method.grid

        alg3 = VUMPS{General}(grid=alg2.grid, parallel_method=SerialMethod(1, nothing))
        TeneT._apply_parallel_method!(alg3)
        @test alg3.grid === nothing
    end
end

@testset "parallel_method helpers ignore algorithms without routing fields" begin
    alg = QRCTMRG{C3v}()
    @test TeneT._apply_parallel_method!(alg) === alg
    @test TeneT._effective_grid(alg) === nothing
end
