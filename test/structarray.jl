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


@testset "StructArray Random" for atype in [Array, CuArray]
    @testset "Array of Numbers" begin
        pattern = [1 2; 
                   2 1]
        SA = randSA(atype, pattern)
        @test CUDA.@allowscalar SA[1,1] == SA[2,2]
        @test CUDA.@allowscalar SA[1,2] == SA[2,1]
        @test size(SA) == (2, 2)
        @test size(SA.data)[1] == length(unique(pattern))
        @test _arraytype(SA.data) == atype

        pattern = [1 2 3; 
                   3 2 1]
        SA = randSA(atype, pattern)
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
        SA = randSA(atype, pattern, sizes)
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
        SA = randSA(atype, pattern, sizes)
        @test size(SA) == (2, 3)
        @test size(SA[1,1]) == sizes[1]
        @test size(SA[1,2]) == sizes[2]
        @test size(SA[1,3]) == sizes[3]
        @test size(SA[2,1]) == sizes[3]
        @test size(SA[2,2]) == sizes[2]
        @test size(SA[2,3]) == sizes[1] 
        @test _arraytype(SA.data[1]) == atype

        SA = randSA(SA)
        @test size(SA) == (2, 3)
        @test size(SA[1,1]) == sizes[1]
        @test size(SA[1,2]) == sizes[2]
        @test size(SA[1,3]) == sizes[3]
        @test size(SA[2,1]) == sizes[3]
        @test size(SA[2,2]) == sizes[2]
        @test size(SA[2,3]) == sizes[1]
        @test _arraytype(SA.data[1]) == atype
    end
end


@testset "StructArray Indexing" begin
    @testset "Array of Numbers" begin
        pattern = [1 2; 2 1]
        SA = randSA(Array, pattern)
        SA[1,1] = 0.0
        @test SA[1,1] == SA[2,2] == 0.0
    end

    @testset "Array of Arrays" begin
        pattern = [1 2; 2 1]
        sizes = [(2,3), (2,3)]
        SA = randSA(Array, pattern, sizes)
        SA[1,1] = zeros(ComplexF64, 2, 3)
        @test SA[1,1] == SA[2,2] == zeros(ComplexF64, 2, 3)
    end
end

# @testset "StructArray Iteration" begin
#     pattern = [1 2; 
#                2 1]
#     sizes = [(2,3), (2,3)]
#     SA = randSA(Array, pattern, sizes)
    
#     # 测试迭代所有独特元素
#     elements = collect(SA)
#     @test length(elements) == 2  # 只有2个独特的元素
#     @test elements[1] == SA[1,1] == SA[2,2] # 第一个独特元素
#     @test elements[2] == SA[1,2] == SA[2,1]  # 第二个独特元素
    

#     # 测试迭代顺序
#     count = 0
#     expected_elements = [SA[1,1], SA[1,2]]
#     for element in SA
#         count += 1
#         @test element == expected_elements[count]
#     end
#     @test count == 2

#     pattern = [1 2 3; 
#                3 2 1]
#     sizes = [(2,3), (2,3), (2,3)]
#     SA = randSA(Array, pattern, sizes)
    
#     # 测试迭代所有独特元素
#     elements = collect(SA)
#     @test length(elements) == 3  # 只有3个独特的元素
#     @test elements[1] == SA[1,1] == SA[2,3] # 第一个独特元素
#     @test elements[2] == SA[2,1] == SA[1,3] # 第二个独特元素
#     @test elements[3] == SA[1,2] == SA[2,2]  # 第三个独特元素
    
#     # 测试迭代顺序
#     count = 0
#     expected_elements = [SA[1,1], SA[1,2], SA[2,1]]
#     for element in SA
#         count += 1
#         @test element == expected_elements[count]
#     end
#     @test count == 3
# end
