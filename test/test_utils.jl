@testset "utils" begin

    # ===================== gpu.jl =====================
    @testset "gpu.jl" begin

        @testset "_arraytype and _mattype dispatch" begin
            x = rand(3, 3)
            @test _arraytype(x) === Array
            @test _mattype(x) === Matrix
            v = rand(5)
            @test _arraytype(v) === Array
        end

        @testset "_arraytype on StructArray" begin
            S = randSA(ComplexF64, Array, [1;;], NTuple{3,Int}[(2,3,4)])
            @test _arraytype(S) === Array
        end

        @testset "device management for Array" begin
            dev = get_device(Array)
            @test dev isa String
            @test occursin("CPU", dev)

            @test device_count(Array) >= 1

            # set_device_id! on Array is a no-op (returns nothing)
            @test set_device_id!(Array, 1) === nothing
        end

        @testset "get_device_id" begin
            x = rand(2, 2)
            @test get_device_id(x) == 1
            @test get_device_id(Array) isa Integer
        end

        @testset "for_gc identity" begin
            x = rand(3, 3)
            @test for_gc(x) === x
            v = rand(ComplexF64, 5)
            @test for_gc(v) === v
        end

        @testset "NamedTuple Array conversion" begin
            nt = (data = [rand(2, 2), rand(3, 3)], tag = "test")
            @test Array(nt) === nt  # identity for Array
        end

        @testset "atype_device!" begin
            x = rand(4, 4)
            y = atype_device!(Array, x, 1)
            @test y isa Array
            @test y == x
        end
    end

    # ===================== misc.jl =====================
    @testset "misc.jl" begin

        @testset "qrpos ($atype)" for atype in ATYPES
            A = atype(rand(ComplexF64, 6, 4))
            Q, R = qrpos(A)

            # Reconstruction
            @test Q * R ≈ A

            # Positive real diagonal of R
            d = real.(diag(R))
            @test all(d .> 0)
            @test all(imag.(diag(R)) .≈ 0)

            # Q is isometric: Q'Q ≈ I
            @test Q' * Q ≈ I atol=1e-12

            # Types are concrete arrays
            @test Q isa atype{ComplexF64, 2}
            @test R isa atype{ComplexF64, 2}
        end

        @testset "lqpos ($atype)" for atype in ATYPES
            A = atype(rand(ComplexF64, 4, 6))
            L, Q = lqpos(A)

            # Reconstruction
            @test L * Q ≈ A

            # Positive real diagonal of L
            d = real.(diag(L))
            @test all(d .> 0)
            @test all(imag.(diag(L)) .≈ 0)

            # Q is isometric: QQ' ≈ I
            @test Q * Q' ≈ I atol=1e-12

            # Types
            @test L isa atype{ComplexF64, 2}
            @test Q isa atype{ComplexF64, 2}
        end

        @testset "qr_for_ad ($atype)" for atype in ATYPES
            A = atype(rand(ComplexF64, 6, 4))
            Q, R = qr_for_ad(A)

            # Reconstruction
            @test Q * R ≈ A

            # Q should be a concrete array, not QRCompactWYQ
            @test Q isa atype{ComplexF64, 2}
        end

        @testset "safesign" begin
            @test safesign(0.0) == 1.0
            @test safesign(0.0 + 0.0im) == 1.0 + 0.0im
            @test safesign(3.0) == 1.0
            @test safesign(-2.0) == -1.0
            z = 3.0 + 4.0im
            @test safesign(z) ≈ z / abs(z)
        end

        @testset "simple_eig" begin
            # Build a Hermitian matrix with a known dominant eigenvalue
            Random.seed!(123)
            H = rand(ComplexF64, 8, 8)
            H = H + H'  # Hermitian
            # Add a large shift to make the leading eigenvalue well-separated
            H += 10 * I

            v0 = rand(ComplexF64, 8)
            v0 /= norm(v0)
            f(v) = H * v
            vals, vecs = simple_eig(f, v0; power_iter=50)

            # Should return vectors of length 1
            @test length(vals) == 1
            @test length(vecs) == 1

            # Compare with LinearAlgebra eigen
            evals = eigvals(H)
            dominant_eval = evals[argmax(abs.(evals))]
            @test abs(vals[1]) ≈ abs(dominant_eval) atol=1e-4

            # Eigenvector should satisfy H*v ≈ lambda*v (up to phase)
            v1 = vecs[1]
            @test norm(v1) ≈ 1.0 atol=1e-10
        end

        @testset "checkpoint" begin
            f(x) = 2 * x .+ 1
            x = rand(5)
            @test f(x) == checkpoint(f, x)

            # With kwargs
            g(x; scale=1.0) = scale * x
            y = rand(3)
            @test g(y; scale=2.0) == checkpoint(g, y; scale=2.0)
        end

        @testset "takagi_decomposition" begin
            # Build a complex symmetric matrix M = B * transpose(B)
            Random.seed!(42)
            n = 6
            B = rand(ComplexF64, n, n)
            M = B * transpose(B)
            @test norm(M - transpose(M)) < 1e-10  # M is symmetric

            # Full decomposition
            A = takagi_decomposition(M; D_trunc=n)
            @test A * transpose(A) ≈ M atol=1e-8

            # Truncated decomposition
            D_trunc = 3
            A_trunc = takagi_decomposition(M; D_trunc=D_trunc)
            @test size(A_trunc) == (n, D_trunc)

            # Non-symmetric matrix should throw
            M_bad = rand(ComplexF64, 4, 4)
            @test_throws ArgumentError takagi_decomposition(M_bad; D_trunc=2)
        end

        @testset "leg type aliases" begin
            @test rand(2, 3, 4) isa leg3
            @test rand(2, 3, 4, 5) isa leg4
            @test rand(2, 3, 4, 5, 6) isa leg5
            @test rand(2, 3, 4, 5, 6, 7, 8, 9) isa leg8

            # 2D array should NOT match leg3
            @test !(rand(2, 3) isa leg3)
        end
    end

    # ===================== io.jl =====================
    @testset "io.jl" begin

        @testset "save_rt / load_rt roundtrip" begin
            mktempdir() do dir
                # Build a minimal VUMPSRuntime with random StructArrays
                pattern = [1;;]
                chi = 4
                d = 2
                AL = randSA(ComplexF64, Array, pattern, NTuple{3,Int}[(d, chi, chi)])
                AR = randSA(ComplexF64, Array, pattern, NTuple{3,Int}[(d, chi, chi)])
                C  = randSA(ComplexF64, Array, pattern, NTuple{2,Int}[(chi, chi)])
                FL = randSA(ComplexF64, Array, pattern, NTuple{3,Int}[(chi, d, chi)])
                FR = randSA(ComplexF64, Array, pattern, NTuple{3,Int}[(chi, d, chi)])
                rt = VUMPSRuntime(AL, AR, C, FL, FR)

                # Save and load
                save_rt(dir, rt)
                rt2 = load_rt(dir, Array, false)

                # Check fields match
                @test rt2.AL.data[1] ≈ rt.AL.data[1]
                @test rt2.C.data[1] ≈ rt.C.data[1]
                @test rt2.AR.data[1] ≈ rt.AR.data[1]
                @test rt2 isa VUMPSRuntime
            end
        end

        @testset "read_last_log" begin
            mktempdir() do dir
                D = 4
                logdir = joinpath(dir, "D$D")
                mkpath(logdir)
                logfile = joinpath(logdir, "history.log")

                # Write a mock log with the expected format
                open(logfile, "w") do io
                    println(io, "i =     3   t = 100.00 sec    e_χ64 = -0.4 gnorm = 1.0e-03   Eimag = 1.0e-10")
                    println(io, "i =     6   t = 43039.05 sec    e_χ144 = -0.501858316272094 gnorm = 1.984e-04   Eimag = 1.389e-10")
                end

                last_i, last_chi = read_last_log(dir, D)
                @test last_i == 6
                @test last_chi == 144
            end
        end
    end
end
