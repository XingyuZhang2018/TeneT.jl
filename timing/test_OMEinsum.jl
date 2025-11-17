using OMEinsum
using Test
using TensorOperations
using CUDA
using cuTENSOR
using BenchmarkTools

@testset "E" for atype in [CuArray]
    d, D, χ = 2, 4, 50
    T = atype(randn(ComplexF64, χ, D, D, χ))
    U = atype(randn(ComplexF64, χ, D, D, χ))
    M = [atype(randn(ComplexF64, D, D, D, D, d)), atype(randn(ComplexF64, D, D, D, D, d))]

    function foo1(T, U) 
        ein"(((aefi,ijkl),ejgbp),fkhcp),abcd -> dghl"(T, U, M[1], M[2], U)
    end

    function foo2(T, U)
        @tensor T[d,g,h,l] := T[a,e,f,i] * U[i,j,k,l] * M[1][e,j,g,b,p] * M[2][f,k,h,c,p] * U[a,b,c,d]
    end

    T1 = CUDA.@time foo1(T, U)
    T2 = CUDA.@time foo2(T, U)
    @test T1 ≈ T2
    @btime CUDA.@sync $foo1($T, $U);
    @btime CUDA.@sync $foo2($T, $U);
end