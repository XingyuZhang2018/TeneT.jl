using VUMPS
using KrylovKit
using CUDA
using Test
using OMEinsum
using Random
CUDA.allowscalar(false)

@testset "OMEinsum with $atype{$dtype} " for atype in [Array, CuArray], dtype in [Float64, ComplexF64]
    Random.seed!(100)
    d = 9
    D = 20
    AL = atype(rand(dtype, D, d, D))
    M = atype(rand(dtype, d, d, d, d))
    FL = atype(rand(dtype, D, d, D))
    @time ein"((γcη,ηpβ),csap),γsα -> αaβ"(FL,AL,M,conj(AL))
end

@testset "KrylovKit with $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64, ComplexF64]
    Random.seed!(100)
    d = 4
    D = 10
    FL = atype(rand(dtype, D, d, D))
    M = atype(rand(dtype, d, d, d, d))
    AL = atype(rand(dtype, D, d, D))
    @time λs, FLs, info = eigsolve(FL -> ein"((γcη,ηpβ),csap),γsα -> αaβ"(FL,AL,M,conj(AL)), FL, 1, :LM; ishermitian = false)

    λl,FL = real(λs[1]),real(FLs[1])
    dFL = atype(rand(dtype, D, d, D))
    @time ξl, info = linsolve(FR -> ein"((ηpβ,βaα),csap),γsα -> ηcγ"(AL, FR, M, conj(AL)), permutedims(dFL, (3, 2, 1)), -λl, 1)
end

@testset "eigsolve with $atype{$dtype}" for N in [3], atype in [CuArray], dtype in [ComplexF64]
    Random.seed!(100)
    D = 1024
    M = atype(rand(dtype, D, D, N))
    x = atype(rand(dtype, D, N))
    @inbounds @views function Mmap(x)
        x = copy(x)
        # mul!(x[:,1], M[:,:,1], x[:,3])
        # mul!(x[:,2], M[:,:,2], x[:,1])
        # mul!(x[:,3], M[:,:,3], x[:,2])
        x[:,1] = M[:,:,1] * x[:,3]
        x[:,2] = M[:,:,2] * x[:,1]
        x[:,3] = M[:,:,3] * x[:,2]
        return x
    end
    # @show x  Mmap(x)
    @time λ,ρs,info = eigsolve(x->Mmap(x), x, 1, :LM; ishermitian = false, maxiter = 1)
    @test λ[1] * ρs[1] ≈ Mmap(ρs[1])
end