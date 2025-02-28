@testset "StructArray Basic" begin
    @testset "Array of Numbers" begin
        data = [1,2]
        pattern = [1 2; 2 1]
        SA = StructArray(data, pattern)
        @test SA[1,1] == SA[2,2]== 1
        @test SA[1,2] == SA[2,1] == 2
        data = [1,2,3]
        pattern = [1 2 3; 3 2 1]
        SA = StructArray(data, pattern)
        @test SA[1,1] == SA[2,3] == 1
        @test SA[1,2] == SA[2,2] == 2
        @test SA[1,3] == SA[2,1] == 3

        SA = similar(SA)
        @test SA[1,1] == SA[2,3]
        @test SA[1,2] == SA[2,2]
        @test SA[1,3] == SA[2,1]
    end

    @testset "Array of Arrays" begin
        A = [1, 2, 3]
        B = [4, 5, 6]
        data = [A, B]
        pattern = [1 2; 2 1]
        SA = StructArray(data, pattern)
        @test SA[1,1] == SA[2,2] == A
        @test SA[1,2] == SA[2,1] == B

        A = rand(ComplexF64, 2, 2)
        B = rand(ComplexF64, 2, 4)
        C = rand(ComplexF64, 3, 5)
        data = [A, B, C]
        pattern = [1 2 3; 3 2 1]
        SA = StructArray(data, pattern)
        @test SA[1,1] == SA[2,3] == A
        @test SA[1,2] == SA[2,2] == B
        @test SA[1,3] == SA[2,1] == C
    end
end

@testset "StructArray Random $atype{$T}" for atype in [Array, CuArray], T in [ComplexF64, Float32]
    @testset "Array of Numbers" begin
        pattern = [1 2; 
                   2 1]
        SA = atype(rand(T, pattern))
        @test CUDA.@allowscalar SA[1,1] == SA[2,2]
        @test CUDA.@allowscalar SA[1,2] == SA[2,1]
        @test size(SA) == (2, 2)
        @test size(SA.data)[1] == length(unique(pattern))
        @test _arraytype(SA.data) == atype

        pattern = [1 2 3; 
                   3 2 1]
        SA = atype(rand(T, pattern))
        @test CUDA.@allowscalar SA[1,1] == SA[2,3]
        @test CUDA.@allowscalar SA[1,2] == SA[2,2]
        @test CUDA.@allowscalar SA[1,3] == SA[2,1]
        @test size(SA) == (2, 3)
        @test size(SA.data)[1] == length(unique(pattern))
        @test _arraytype(SA.data) == atype
    end

    @testset "Array of Arrays" begin
        pattern = [1 2; 2 1]
        sizes = [(2,), (2,)]
        SA = atype(rand(T, sizes, pattern))
        @test SA[1,1] == SA[2,2]
        @test SA[1,2] == SA[2,1]
        @test size(SA) == (2, 2)
        @test size(SA[1,1]) == sizes[1]
        @test size(SA[1,2]) == sizes[2]
        @test size(SA[2,1]) == sizes[2]
        @test size(SA[2,2]) == sizes[1]
        @test _arraytype(SA.data[1]) == atype

        pattern = [1 2 3; 3 2 1]
        sizes = [(2,3), (2,4), (3,5)]
        SA = atype(rand(T, sizes, pattern))
        @test size(SA) == (2, 3)
        @test size(SA[1,1]) == sizes[1]
        @test size(SA[1,2]) == sizes[2]
        @test size(SA[1,3]) == sizes[3]
        @test size(SA[2,1]) == sizes[3]
        @test size(SA[2,2]) == sizes[2]
        @test size(SA[2,3]) == sizes[1] 
        @test _arraytype(SA.data[1]) == atype

        SA = rand!(SA)
        @test size(SA) == (2, 3)
        @test size(SA[1,1]) == sizes[1]
        @test size(SA[1,2]) == sizes[2]
        @test size(SA[1,3]) == sizes[3]
        @test size(SA[2,1]) == sizes[3]
        @test size(SA[2,2]) == sizes[2]
        @test size(SA[2,3]) == sizes[1]
        @test _arraytype(SA.data[1]) == atype
    end

    @testset "Array of TensorMaps" begin
        pattern = [1 2; 2 1]
        spaces = [ℂ^2 ← ℂ^2, ℂ^2 ← ℂ^2]
        SA = atype(rand(T, spaces, pattern))
        @test SA[1,1] == SA[2,2]
        @test SA[1,2] == SA[2,1]
        @test size(SA) == (2, 2)
        @test space(SA[1,1]) == spaces[1]
        @test space(SA[1,2]) == spaces[2]
        @test space(SA[2,1]) == spaces[2]
        @test space(SA[2,2]) == spaces[1]
        @test _arraytype(SA.data[1]) == atype

        pattern = [1 2 3; 3 2 1]
        spaces = [ℂ^2 ← ℂ^3, ℂ^2 ← ℂ^4, ℂ^3 ← ℂ^5]
        SA = atype(rand(T, spaces, pattern))
        @test size(SA) == (2, 3)
        @test space(SA[1,1]) == spaces[1]
        @test space(SA[1,2]) == spaces[2]
        @test space(SA[1,3]) == spaces[3]
        @test space(SA[2,1]) == spaces[3]
        @test space(SA[2,2]) == spaces[2]
        @test space(SA[2,3]) == spaces[1] 
        @test _arraytype(SA.data[1]) == atype

        SA = rand!(SA)
        @test size(SA) == (2, 3)
        @test space(SA[1,1]) == spaces[1]
        @test space(SA[1,2]) == spaces[2]
        @test space(SA[1,3]) == spaces[3]
        @test space(SA[2,1]) == spaces[3]
        @test space(SA[2,2]) == spaces[2]
        @test space(SA[2,3]) == spaces[1]
        @test _arraytype(SA.data[1]) == atype
    end
end


@testset "StructArray Indexing $atype{$T}" for atype in [Array, CuArray], T in [ComplexF64, Float32]
    @testset "Array of Numbers" begin
        pattern = [1 2; 2 1]
        SA = atype(rand(T, pattern))
        CUDA.@allowscalar SA[1,1] = 0.0
        CUDA.@allowscalar @test SA[1,1] == SA[2,2] == 0.0
    end

    @testset "Array of Arrays" begin
        pattern = [1 2; 2 1]
        sizes = [(2,3), (2,3)]
        SA = atype(rand(T, sizes, pattern))
        CUDA.@allowscalar SA[1,1] = zeros(T, 2, 3)
        CUDA.@allowscalar @test SA[1,1] == SA[2,2] == zeros(T, 2, 3)
    end

    @testset "Array of TensorMaps" begin
        pattern = [1 2; 2 1]
        spaces = [ℂ^2 ← ℂ^3, ℂ^2 ← ℂ^3]
        SA = atype(rand(T, spaces, pattern))
        CUDA.@allowscalar SA[1,1] = zeros(T, spaces[1])
        CUDA.@allowscalar @test SA[1,1] == SA[2,2] == zeros(T, spaces[1])
    end
end
