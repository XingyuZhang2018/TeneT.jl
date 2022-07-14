using BenchmarkTools
using KrylovKit
using CUDA
using Test
using OMEinsum
using Random
using LinearAlgebra
CUDA.allowscalar(false)

@testset "OMEinsum with $atype{$dtype} " for atype in [CuArray], dtype in [ComplexF64]
    Random.seed!(100)
    D = 2
    χ = 16

    println("D = $(D) χ = $(χ)")
    AL = atype(reshape(rand(dtype, χ,D,D,χ), χ, D^2, χ))
    M = atype(reshape(rand(dtype, D,D,D,D,D,D,D,D), D^2, D^2, D^2, D^2))
    FL = AL = atype(reshape(rand(dtype, χ,D,D,χ), χ, D^2, χ))

    @btime CUDA.@sync ein"((adf,abc),dgeb),fgh -> ceh"($FL,$AL,$M,conj($AL))
end

@testset "KrylovKit with $atype{$dtype}" for atype in [Array], dtype in [ComplexF64]
    Random.seed!(100)
    D = 2
    χ = 16

    println("D = $(D) χ = $(χ)")
    AL = atype(reshape(rand(dtype, χ,D,D,χ), χ, D^2, χ))
    M = atype(reshape(rand(dtype, D,D,D,D,D,D,D,D), D^2, D^2, D^2, D^2))
    FL = atype(reshape(rand(dtype, χ,D,D,χ), χ, D^2, χ))

    @btime CUDA.@sync λs, FLs, info = eigsolve(FL -> ein"((adf,abc),dgeb),fgh -> ceh"(FL,$AL,$M,conj($AL)), $FL, 1, :LM; ishermitian = false)
end
