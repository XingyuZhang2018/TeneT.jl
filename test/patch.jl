@testset "OMEinsum with $atype{$dtype} " for atype in [Array], dtype in [Float64, ComplexF64]
    Random.seed!(100)
    d = 9
    D = 20
    AL = atype(rand(dtype, D, d, D))
    M = atype(rand(dtype, d, d, d, d))
    FL = atype(rand(dtype, D, d, D))
    @time ein"((γcη,ηpβ),csap),γsα -> αaβ"(FL,AL,M,conj(AL))
end

@testset "KrylovKit with $atype{$dtype}" for atype in [Array], dtype in [Float64, ComplexF64]
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