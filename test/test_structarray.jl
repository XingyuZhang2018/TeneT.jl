@testset "StructArray" begin
    @testset "Construction and indexing" for atype in ATYPES
        data = [atype(rand(ComplexF64, 3, 3)), atype(rand(ComplexF64, 3, 3))]
        pattern = [1 2; 2 1]
        S = StructArray(data, pattern)

        @test size(S) == (2, 2)
        @test size(S, 1) == 2
        @test size(S, 2) == 2
        @test length(S) == 2  # number of unique data elements
        @test S[1,1] === data[1]
        @test S[1,2] === data[2]
        @test S[2,1] === data[2]
        @test S[2,2] === data[1]
    end

    @testset "Invalid pattern assertion" begin
        data = [rand(ComplexF64, 2, 2)]
        @test_throws AssertionError StructArray(data, [1 2; 2 1])  # 2 unique indices but only 1 data
    end

    @testset "setindex!" for atype in ATYPES
        data = [atype(rand(ComplexF64, 2, 2)), atype(rand(ComplexF64, 2, 2))]
        S = StructArray(data, [1 2; 2 1])
        new_val = atype(ones(ComplexF64, 2, 2))
        S[1,1] = new_val
        @test S[1,1] === new_val
        @test S[2,2] === new_val  # shares pattern index 1
    end

    @testset "Arithmetic" for atype in ATYPES
        data1 = [atype(rand(ComplexF64, 3, 3)), atype(rand(ComplexF64, 3, 3))]
        data2 = [atype(rand(ComplexF64, 3, 3)), atype(rand(ComplexF64, 3, 3))]
        pattern = [1 2; 2 1]
        S1 = StructArray(data1, pattern)
        S2 = StructArray(data2, pattern)

        # Addition
        S3 = S1 + S2
        @test Array(S3[1,1]) ≈ Array(S1[1,1]) + Array(S2[1,1])

        # Scalar multiplication
        S4 = 2.0 * S1
        @test Array(S4[1,1]) ≈ 2.0 * Array(S1[1,1])

        # Division
        S5 = S1 / 3.0
        @test Array(S5[1,1]) ≈ Array(S1[1,1]) / 3.0

        # Norm
        n = norm(S1)
        @test n ≈ norm(S1.data)
        @test n >= 0

        # Conjugate
        Sc = conj(S1)
        @test Array(Sc[1,1]) ≈ conj(Array(S1[1,1]))
    end

    @testset "rmul! and axpy!" for atype in ATYPES
        data1 = [atype(rand(ComplexF64, 3, 3)), atype(rand(ComplexF64, 3, 3))]
        data2 = [atype(rand(ComplexF64, 3, 3)), atype(rand(ComplexF64, 3, 3))]
        pattern = [1 2; 2 1]
        S1 = StructArray(copy.(data1), pattern)
        S2 = StructArray(copy.(data2), pattern)

        S1_orig = copy(S1)
        rmul!(S1, 0.5)
        @test Array(S1[1,1]) ≈ 0.5 * Array(S1_orig[1,1])

        S2_orig = copy(S2)
        axpy!(2.0, S1, S2)
        @test Array(S2[1,1]) ≈ Array(S2_orig[1,1]) + 2.0 * Array(S1[1,1])
    end

    @testset "similar, zero, copy" for atype in ATYPES
        data = [atype(rand(ComplexF64, 3, 3)), atype(rand(ComplexF64, 3, 3))]
        S = StructArray(data, [1 2; 2 1])

        S_sim = similar(S)
        @test size(S_sim) == size(S)
        @test length(S_sim) == length(S)

        S_z = zero(S)
        @test norm(S_z) ≈ 0 atol=1e-15

        S_c = copy(S)
        @test Array(S_c[1,1]) ≈ Array(S[1,1])
        # copy is independent
        S_c[1,1] = atype(zeros(ComplexF64, 3, 3))
        @test !(Array(S[1,1]) ≈ zeros(ComplexF64, 3, 3))
    end

    @testset "circshift" for atype in ATYPES
        data = [atype(rand(ComplexF64, 2, 2)), atype(rand(ComplexF64, 2, 2)),
                atype(rand(ComplexF64, 2, 2)), atype(rand(ComplexF64, 2, 2))]
        S = StructArray(data, [1 2; 3 4])
        Ss = circshift(S, (0, 1))
        @test Ss.pattern == circshift([1 2; 3 4], (0, 1))
    end

    @testset "isapprox" for atype in ATYPES
        data = [atype(rand(ComplexF64, 3, 3)), atype(rand(ComplexF64, 3, 3))]
        S = StructArray(data, [1 2; 2 1])
        S2 = copy(S)
        @test isapprox(S, S2)
    end

    @testset "iterate and collect" for atype in ATYPES
        data = [atype(rand(ComplexF64, 2, 2)), atype(rand(ComplexF64, 2, 2))]
        S = StructArray(data, [1 2; 2 1])
        collected = collect(S)
        @test length(collected) == 2
    end

    @testset "NamedTuple interop" for atype in ATYPES
        data = [atype(rand(ComplexF64, 3, 3)), atype(rand(ComplexF64, 3, 3))]
        pattern = [1 2; 2 1]
        S = StructArray(data, pattern)
        nt = (data = [atype(rand(ComplexF64, 3, 3)), atype(rand(ComplexF64, 3, 3))],)
        result = S + nt
        @test result isa StructArray
        @test Array(result[1,1]) ≈ Array(S[1,1]) + Array(nt.data[1])
    end

    @testset "Initialization functions" for atype in ATYPES
        pattern = [1 2; 2 1]
        sizes = [(3, 4, 3), (3, 4, 3)]

        # randSA with type
        S = randSA(ComplexF64, atype, pattern, sizes)
        @test size(S) == (2, 2)
        @test size(S[1,1]) == (3, 4, 3)
        @test eltype(S[1,1]) == ComplexF64

        # randSA default type
        S2 = randSA(atype, pattern, sizes)
        @test eltype(S2[1,1]) == ComplexF64

        # randSA from existing StructArray
        S3 = randSA(S)
        @test size(S3[1,1]) == size(S[1,1])

        # cellones
        C = cellones(S)
        @test size(C[1,1]) == (3, 3)  # chi x chi from first dim
        @test Array(C[1,1]) ≈ Matrix{ComplexF64}(I, 3, 3)

        # ISA
        Isa = ISA(ComplexF64, atype, pattern, [(4, 4), (4, 4)])
        @test Array(Isa[1,1]) ≈ Matrix{ComplexF64}(I, 4, 4)
    end

    @testset "GPU roundtrip" for atype in ATYPES
        atype == Array && continue
        data = [rand(ComplexF64, 3, 3), rand(ComplexF64, 3, 3)]
        S = StructArray(data, [1 2; 2 1])
        S_gpu = atype(S)
        @test _arraytype(S_gpu[1,1]) == atype
        S_cpu = Array(S_gpu)
        @test isapprox(S, S_cpu)
    end

    @testset "Zygote.Buffer" for atype in ATYPES
        data = [atype(rand(ComplexF64, 3, 3)), atype(rand(ComplexF64, 3, 3))]
        S = StructArray(data, [1 2; 2 1])
        buf = Zygote.Buffer(S)
        buf[1] = atype(ones(ComplexF64, 3, 3))
        S2 = copy(buf)
        @test S2 isa StructArray
        @test Array(S2[1,1]) ≈ ones(ComplexF64, 3, 3)
    end
end
