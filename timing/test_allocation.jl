using BenchmarkTools
using CUDA
using Test
using Random
CUDA.allowscalar(false)

@testset "allocation" begin
    Random.seed!(100)
    B = [CuArray(rand(10, 10)) for i in 1:100]
    function foo1(B)
        A = CuArray(zeros(100, 100))
        for i in 1:10, j in 1:10
            A[1 + 10 * (i - 1) : 10 * i, 1 + 10 * (j - 1) : 10 * j] .= B[10 * (i-1) + j]
        end
    end
    function foo2(B)
        A = []
        for i in 1:10
            h = []
            for j in 1:10
                push!(h, B[10 * (i-1) + j])
            end
            hi = hcat(h...)
            push!(A, hi)
        end
        A = vcat(A...)
    end

    foo3(B) = hvcat(ntuple(i->10, 10), B...)
    @show foo1(B) == foo2(B)
    @btime CUDA.@sync $foo1($B)
    @btime CUDA.@sync $foo2($B)
    @btime CUDA.@sync $foo3($B)
end

@testset "allocation" begin
    Random.seed!(100)
    B = [CuArray(rand(2, 2)) for i in 1:4]
    A = CuArray(zeros(4, 4))
    @show A
    A[1:2, 1:2] .= B[1]
    @show A
    B[1] = CuArray(zeros(2, 2))
    @show B A 
end
