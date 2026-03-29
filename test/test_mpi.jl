using MPI

@testset "MPI (optional)" begin
    if !MPI.Initialized()
        MPI.Init()
    end
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    @testset "split_count / split_ranges with MPI" begin
        counts = split_count(10, nprocs)
        @test sum(counts) == 10
        ranges = split_ranges(10, nprocs)
        @test length(ranges) == nprocs
    end

    @testset "parallel matches forloop" begin
        chi, D = 4, 2
        FL = rand(ComplexF64, chi, D, chi)
        ALu = rand(ComplexF64, chi, D, chi)
        ALd = rand(ComplexF64, chi, D, chi)
        M = rand(ComplexF64, D, D, D, D)

        r_serial = FLmap_parallel(FL, ALu, ALd, M; ifparallel=false, forloop_iter=1)
        r_parallel = FLmap_parallel(FL, ALu, ALd, M; ifparallel=true, forloop_iter=1)
        @test r_serial ≈ r_parallel atol=1e-10
    end
end
