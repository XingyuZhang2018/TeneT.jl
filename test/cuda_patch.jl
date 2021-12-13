using BenchmarkTools
using VUMPS
using KrylovKit
using CUDA
using CUDA.CUSPARSE
using SparseArrays
using Test
using OMEinsum
using ProfileView
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
    function profile_test()
        ein"((γcη,ηpβ),csap),γsα -> αaβ"(FL,AL,M,conj(AL))
    end
    ProfileView.@profview profile_test()
end

@testset "OMEinsum with $atype{$dtype} " for atype in [Array], dtype in [Float64]
    Random.seed!(100)
    d = 16
    D = 20
    AL = randZ2(atype, dtype, D, d, D)
    M = randZ2(atype, dtype, d, d, d, d)
    FL = randZ2(atype, dtype, D, d, D)
    tAL, tM, tFL = map(Z2Matrix2tensor,[AL, M, FL])
    @time ein"((γcη,ηpβ),csap),γsα -> αaβ"(FL,AL,M,conj(AL))
    @time ein"((γcη,ηpβ),csap),γsα -> αaβ"(tFL,tAL,tM,conj(tAL))
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
 