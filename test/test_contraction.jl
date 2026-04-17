@testset "Contraction" begin
    χ = 4
    D = 2

    @testset "_downcast_eltype helper" begin
        using TeneT: _downcast_eltype

        # Real: Float64 → Float32
        A64 = randn(Float64, 3, 4)
        A32 = _downcast_eltype(Float32, A64)
        @test eltype(A32) == Float32
        @test size(A32) == size(A64)
        @test maximum(abs, Float64.(A32) .- A64) < 1e-5

        # Complex: ComplexF64 → ComplexF32
        Z64 = randn(ComplexF64, 3, 4)
        Z32 = _downcast_eltype(Float32, Z64)
        @test eltype(Z32) == ComplexF32
        @test size(Z32) == size(Z64)
        @test maximum(abs, ComplexF64.(Z32) .- Z64) < 1e-5

        # Identity: target equals current eltype — return input unchanged (no copy)
        B64 = randn(Float64, 2, 2)
        @test _downcast_eltype(Float64, B64) === B64

        # Nothing: pass-through
        C64 = randn(ComplexF64, 2, 2)
        @test _downcast_eltype(nothing, C64) === C64
    end

    # =========================================================================
    # basic.jl tests
    # =========================================================================
    @testset "basic.jl — atype=$atype" for atype in ATYPES
        T = ComplexF64

        @testset "ALCtoAC_map" begin
            # leg3: AL is (χ, D, χ), C is (χ, χ)
            AL3 = atype(randn(T, χ, D, χ))
            C   = atype(randn(T, χ, χ))
            AC3 = ALCtoAC_map(AL3, C)
            @test size(AC3) == (χ, D, χ)

            # leg4: AL is (χ, D, D, χ), C is (χ, χ)
            AL4 = atype(randn(T, χ, D, D, χ))
            AC4 = ALCtoAC_map(AL4, C)
            @test size(AC4) == (χ, D, D, χ)
        end

        @testset "CTtoT" begin
            Cm = atype(randn(T, χ, χ))
            T3 = atype(randn(T, χ, D, χ))
            @test size(CTtoT(Cm, T3)) == (χ, D, χ)

            T4 = atype(randn(T, χ, D, D, χ))
            @test size(CTtoT(Cm, T4)) == (χ, D, D, χ)
        end

        @testset "CTCtoT" begin
            Cm = atype(randn(T, χ, χ))
            T3 = atype(randn(T, χ, D, χ))
            @test size(CTCtoT(Cm, T3)) == (χ, D, χ)

            T4 = atype(randn(T, χ, D, D, χ))
            @test size(CTCtoT(Cm, T4)) == (χ, D, D, χ)
        end

        @testset "FLmap leg4" begin
            FL  = atype(randn(T, χ, D, χ))
            ALu = atype(randn(T, χ, D, χ))
            ALd = atype(randn(T, χ, D, χ))
            M   = atype(randn(T, D, D, D, D))
            result = FLmap(FL, ALu, ALd, M)
            @test size(result) == (χ, D, χ)
        end

        @testset "FLmap leg5 (bilayer)" begin
            FL  = atype(randn(T, χ, D, D, χ))
            ALu = atype(randn(T, χ, D, D, χ))
            ALd = atype(randn(T, χ, D, D, χ))
            M   = atype(randn(T, D, D, D, D, D))
            result = FLmap(FL, ALu, ALd, M)
            @test size(result) == (χ, D, D, χ)
        end

        @testset "FRmap leg4" begin
            FR  = atype(randn(T, χ, D, χ))
            ARu = atype(randn(T, χ, D, χ))
            ARd = atype(randn(T, χ, D, χ))
            M   = atype(randn(T, D, D, D, D))
            result = FRmap(FR, ARu, ARd, M)
            @test size(result) == (χ, D, χ)
        end

        @testset "FRmap leg5 (bilayer)" begin
            FR  = atype(randn(T, χ, D, D, χ))
            ARu = atype(randn(T, χ, D, D, χ))
            ARd = atype(randn(T, χ, D, D, χ))
            M   = atype(randn(T, D, D, D, D, D))
            result = FRmap(FR, ARu, ARd, M)
            @test size(result) == (χ, D, D, χ)
        end

        @testset "Lmap and Rmap" begin
            L   = atype(randn(T, χ, χ))
            ALu = atype(randn(T, χ, D, χ))
            ALd = atype(randn(T, χ, D, χ))
            result_L = Lmap(L, ALu, ALd)
            @test size(result_L) == (χ, χ)

            R   = atype(randn(T, χ, χ))
            ARu = atype(randn(T, χ, D, χ))
            ARd = atype(randn(T, χ, D, χ))
            result_R = Rmap(R, ARu, ARd)
            @test size(result_R) == (χ, χ)
        end

        @testset "Lmap and Rmap leg4" begin
            L   = atype(randn(T, χ, χ))
            ALu = atype(randn(T, χ, D, D, χ))
            ALd = atype(randn(T, χ, D, D, χ))
            result_L = Lmap(L, ALu, ALd)
            @test size(result_L) == (χ, χ)

            R   = atype(randn(T, χ, χ))
            ARu = atype(randn(T, χ, D, D, χ))
            ARd = atype(randn(T, χ, D, D, χ))
            result_R = Rmap(R, ARu, ARd)
            @test size(result_R) == (χ, χ)
        end

        @testset "ACmap and Cmap" begin
            # ACmap leg4
            AC = atype(randn(T, χ, D, χ))
            FL = atype(randn(T, χ, D, χ))
            FR = atype(randn(T, χ, D, χ))
            M  = atype(randn(T, D, D, D, D))
            result_AC = ACmap(AC, FL, FR, M)
            @test size(result_AC) == (χ, D, χ)

            # Cmap leg3
            Cm  = atype(randn(T, χ, χ))
            FL3 = atype(randn(T, χ, D, χ))
            FR3 = atype(randn(T, χ, D, χ))
            result_C = Cmap(Cm, FL3, FR3)
            @test size(result_C) == (χ, χ)

            # Cmap leg4
            FL4 = atype(randn(T, χ, D, D, χ))
            FR4 = atype(randn(T, χ, D, D, χ))
            result_C4 = Cmap(Cm, FL4, FR4)
            @test size(result_C4) == (χ, χ)
        end

        @testset "ACdmap leg4" begin
            ACd = atype(randn(T, χ, D, χ))
            FL  = atype(randn(T, χ, D, χ))
            FR  = atype(randn(T, χ, D, χ))
            M   = atype(randn(T, D, D, D, D))
            result = ACdmap(ACd, FL, FR, M)
            @test size(result) == (χ, D, χ)
        end

        @testset "ACdmap leg5 (bilayer)" begin
            ACd = atype(randn(T, χ, D, D, χ))
            FL  = atype(randn(T, χ, D, D, χ))
            FR  = atype(randn(T, χ, D, D, χ))
            M   = atype(randn(T, D, D, D, D, D))
            result = ACdmap(ACd, FL, FR, M)
            @test size(result) == (χ, D, D, χ)
        end

        @testset "Mmap" begin
            AC  = atype(randn(T, χ, D, χ))
            ACd = atype(randn(T, χ, D, χ))
            FL  = atype(randn(T, χ, D, χ))
            FR  = atype(randn(T, χ, D, χ))
            result = Mmap(AC, ACd, FL, FR)
            @test size(result) == (D, D, D, D)
        end
    end

    # =========================================================================
    # forloop_parallel_MPI.jl tests
    # =========================================================================
    @testset "forloop_parallel_MPI.jl" begin

        @testset "split_count" begin
            counts = split_count(10, 3)
            @test sum(counts) == 10
            @test length(counts) == 3
            @test maximum(counts) - minimum(counts) <= 1
        end

        @testset "split_ranges" begin
            ranges = split_ranges(10, 3)
            @test length(ranges) == 3
            all_indices = vcat([collect(r) for r in ranges]...)
            @test all_indices == collect(1:10)
        end

        @testset "parallel forloop — atype=$atype" for atype in ATYPES
            T = ComplexF64

            @testset "FLmap_parallel forloop_iter=1 matches FLmap" begin
                FL  = atype(randn(T, χ, D, χ))
                ALu = atype(randn(T, χ, D, χ))
                ALd = atype(randn(T, χ, D, χ))
                M   = atype(randn(T, D, D, D, D))

                direct   = FLmap(FL, ALu, ALd, M)
                par_res  = FLmap_parallel(FL, ALu, ALd, M; ifparallel=false, forloop_iter=1)
                @test Array(direct) ≈ Array(par_res)
            end

            @testset "FLmap_parallel forloop_iter=2 matches FLmap" begin
                FL  = atype(randn(T, χ, D, χ))
                ALu = atype(randn(T, χ, D, χ))
                ALd = atype(randn(T, χ, D, χ))
                M   = atype(randn(T, D, D, D, D))

                direct   = FLmap(FL, ALu, ALd, M)
                par_res  = FLmap_parallel(FL, ALu, ALd, M; ifparallel=false, forloop_iter=2)
                @test Array(direct) ≈ Array(par_res)
            end

            @testset "FRmap_parallel forloop_iter=1 matches FRmap" begin
                FR  = atype(randn(T, χ, D, χ))
                ARu = atype(randn(T, χ, D, χ))
                ARd = atype(randn(T, χ, D, χ))
                M   = atype(randn(T, D, D, D, D))

                direct   = FRmap(FR, ARu, ARd, M)
                par_res  = FRmap_parallel(FR, ARu, ARd, M; ifparallel=false, forloop_iter=1)
                @test Array(direct) ≈ Array(par_res)
            end

            @testset "ACmap_parallel forloop_iter=1 matches ACmap" begin
                AC = atype(randn(T, χ, D, χ))
                FL = atype(randn(T, χ, D, χ))
                FR = atype(randn(T, χ, D, χ))
                M  = atype(randn(T, D, D, D, D))

                direct   = ACmap(AC, FL, FR, M)
                par_res  = ACmap_parallel(AC, FL, FR, M; ifparallel=false, forloop_iter=1)
                @test Array(direct) ≈ Array(par_res)
            end

            @testset "ACdmap_parallel forloop_iter=1 matches ACdmap" begin
                ACd = atype(randn(T, χ, D, χ))
                FL  = atype(randn(T, χ, D, χ))
                FR  = atype(randn(T, χ, D, χ))
                M   = atype(randn(T, D, D, D, D))

                direct   = ACdmap(ACd, FL, FR, M)
                par_res  = ACdmap_parallel(ACd, FL, FR, M; ifparallel=false, forloop_iter=1)
                @test Array(direct) ≈ Array(par_res)
            end

            @testset "Mmap_parallel forloop_iter=1 matches Mmap" begin
                AC  = atype(randn(T, χ, D, χ))
                ACd = atype(randn(T, χ, D, χ))
                FL  = atype(randn(T, χ, D, χ))
                FR  = atype(randn(T, χ, D, χ))

                direct   = Mmap(AC, ACd, FL, FR)
                par_res  = Mmap_parallel(AC, ACd, FL, FR; ifparallel=false, forloop_iter=1)
                @test Array(direct) ≈ Array(par_res)
            end
        end
    end
end
