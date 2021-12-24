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

@testset "OMEinsum with $symmetry $atype{$dtype} " for atype in [CuArray], dtype in [ComplexF64], symmetry in [:none, :Z2]
    Random.seed!(100)
    d = 2
    χ = 200
    FL = randinitial(Val(symmetry), atype, dtype, χ, d, χ)
    M = randinitial(Val(symmetry), atype, dtype, d, d, d, d)
    AL = randinitial(Val(symmetry), atype, dtype, χ, d, χ)
    # @time CUDA.@sync ein"((adf,abc),dgeb),fgh -> ceh"(FL,AL,M,conj(AL))
    @btime CUDA.@sync ein"((adf,abc),dgeb),fgh -> ceh"($FL,$AL,$M,conj($AL))
end

@testset "KrylovKit with $symmetry $atype{$dtype}" for atype in [CuArray], dtype in [ComplexF64], symmetry in [:none, :Z2]
    Random.seed!(100)
    d = 2
    χ = 20
    FL = randinitial(Val(symmetry), atype, dtype, χ, d^2, χ)
    M = randinitial(Val(symmetry), atype, dtype, d^2, d^2, d^2, d^2)
    AL = randinitial(Val(symmetry), atype, dtype, χ, d^2, χ)
    @time λs, FLs, info = eigsolve(FL -> ein"((adf,abc),dgeb),fgh -> ceh"(FL,AL,M,conj(AL)), FL, 1, :LM; ishermitian = false)

    λl,FL = λs[1],FLs[1]
    dFL = randinitial(Val(symmetry), atype, dtype, χ, d^2, χ)
    @time ξl, info = linsolve(FR -> ein"((ceh,abc),dgeb),fgh -> adf"(AL, FR, M, conj(AL)), dFL, -λl, 1)
end