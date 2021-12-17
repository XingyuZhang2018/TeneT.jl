using BenchmarkTools
using VUMPS
using KrylovKit
using CUDA
using CUDA.CUSPARSE
using SparseArrays
using Test
using OMEinsum
using SymEngine
using ProfileView
using Random
CUDA.allowscalar(false)

@testset "OMEinsum with $atype{$dtype} " for atype in [CuArray], dtype in [Float64]
    Random.seed!(100)
    d = 25
    D = 50
    AL = atype(rand(dtype, D, d, D))
    M = atype(rand(dtype, d, d, d, d))
    FL = atype(rand(dtype, D, d, D))
    @time CUDA.@sync ein"((γcη,ηpβ),csap),γsα -> αaβ"(FL,AL,M,conj(AL))
    # @btime CUDA.@sync ein"((γcη,ηpβ),csap),γsα -> αaβ"($FL,$AL,$M,conj($AL))
end

@testset "OMEinsum with $atype{$dtype} " for atype in [CuArray], dtype in [Float64]
    Random.seed!(100)
    d = 25
    D = 50
    AL = randZ2(atype, dtype, D, d, D)
    M = randZ2(atype, dtype, d, d, d, d)
    FL = randZ2(atype, dtype, D, d, D)
    @time CUDA.@sync ein"((γcη,ηpβ),csap),γsα -> αaβ"(FL,AL,M,conj(AL))
    # @btime CUDA.@sync ein"((γcη,ηpβ),csap),γsα -> αaβ"($FL,$AL,$M,conj($AL))
end

@testset "KrylovKit with $atype{$dtype}" for atype in [CuArray], dtype in [ComplexF64]
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
 
@testset "KrylovKit with $atype{$dtype}" for atype in [CuArray], dtype in [ComplexF64]
    Random.seed!(100)
    d = 4
    D = 10
    AL = randZ2(atype, dtype, D, d, D)
    M = randZ2(atype, dtype, d, d, d, d)
    FL = randZ2(atype, dtype, D, d, D)
    @time λs, FLs, info = eigsolve(FL -> ein"((γcη,ηpβ),csap),γsα -> αaβ"(FL,AL,M,conj(AL)), FL, 1, :LM; ishermitian = false)

    # λl,FL = real(λs[1]),real(FLs[1])
    # dFL = atype(rand(dtype, D, d, D))
    # @time ξl, info = linsolve(FR -> ein"((ηpβ,βaα),csap),γsα -> ηcγ"(AL, FR, M, conj(AL)), permutedims(dFL, (3, 2, 1)), -λl, 1)
end