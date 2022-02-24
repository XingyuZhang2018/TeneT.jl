using BenchmarkTools
using CUDA
using Test
using Random
CUDA.allowscalar(false)

@testset "allocation" begin
    Random.seed!(100)
    B = [CUDA.rand(Float64, 10, 10) for i in 1:100]
    function foo1(B)
        # A = CuArray(zeros(Float32, 100, 100))
        A = CUDA.zeros(Float64, 100, 100)
        for i in 1:10, j in 1:10
            A[1 + 10 * (i - 1) : 10 * i, 1 + 10 * (j - 1) : 10 * j] .= B[10 * (i-1) + j]
            # copyto!(A[1 + 10 * (i - 1) : 10 * i, 1 + 10 * (j - 1) : 10 * j], B[10 * (i-1) + j])
        end
        A
    end
    foo2(B) = hvcat(ntuple(i->10, 10), B...)
    @show foo1(B) == foo2(B)
    @btime CUDA.@sync $foo1($B)
    # @btime CUDA.@sync $foo2($B)
end