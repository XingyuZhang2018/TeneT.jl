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

    @testset "allgatherv_p2p! correctness" begin
        for atype in ATYPES, T in (Float64, ComplexF64), N in (1023, 8192, 262144)
            counts = Cint.(split_count(N, nprocs))
            displs = cumsum([0; counts[1:end-1]])
            my_count = counts[rank + 1]

            # Each rank fills its own chunk with (rank+1); after allgatherv,
            # every rank should see [1…1, 2…2, 3…3, …, nprocs…nprocs] pattern.
            buf = atype(zeros(T, N))
            buf[displs[rank+1]+1 : displs[rank+1]+my_count] .= T(rank + 1)
            TeneT.allgatherv_p2p!(buf, counts, comm)
            CUDA.functional() && atype == CuArray && CUDA.synchronize()
            MPI.Barrier(comm)

            expected = reduce(vcat, [fill(T(r+1), counts[r+1]) for r in 0:nprocs-1])
            @test Array(buf) == expected
        end
    end

    @testset "ring neighbors + node-range helpers" begin
        prev, next = TeneT._ring_neighbors(rank, nprocs)
        @test prev == mod(rank - 1, nprocs)
        @test next == mod(rank + 1, nprocs)

        # Single-node invocation: local_size == nprocs, so one node range
        # spanning the whole buf.
        N_test = 1000
        counts_test = Cint.(split_count(N_test, nprocs))
        ranges1 = TeneT._node_ranges(counts_test, nprocs, 1)
        @test length(ranges1) == 1
        @test ranges1[1] == (1, N_test)
    end

    @testset "allreduce_p2p! correctness" begin
        for atype in ATYPES, T in (Float64, ComplexF64), N in (1023, 8192, 262144)
            # Pattern 1: all-ones, reduce to nprocs
            buf = atype(ones(T, N))
            TeneT.allreduce_p2p!(buf, +, comm)
            CUDA.functional() && atype == CuArray && CUDA.synchronize()
            MPI.Barrier(comm)
            @test all(Array(buf) .== T(nprocs))

            # Pattern 2: rank-specific constant, reduce to sum(1:nprocs)
            buf = atype(fill(T(rank + 1), N))
            TeneT.allreduce_p2p!(buf, +, comm)
            CUDA.functional() && atype == CuArray && CUDA.synchronize()
            MPI.Barrier(comm)
            @test all(Array(buf) .≈ T(sum(1:nprocs)))
        end
    end

    @testset "packed allreduce correctness" begin
        for atype in ATYPES, T in (Float64, ComplexF64)
            a = atype(fill(T(rank + 1), 17))
            b = atype(fill(T(10 * (rank + 1)), 3, 5))
            c = atype(fill(T(100 * (rank + 1)), 2, 4, 3))

            TeneT._allreduce_many_p2p!((a, b, c), comm)
            CUDA.functional() && atype == CuArray && CUDA.synchronize()
            MPI.Barrier(comm)

            s = sum(1:nprocs)
            @test all(Array(a) .≈ T(s))
            @test all(Array(b) .≈ T(10s))
            @test all(Array(c) .≈ T(100s))
        end
    end
end
