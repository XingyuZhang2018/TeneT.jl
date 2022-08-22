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
    FL = atype(reshape(rand(dtype, χ,D,D,χ), χ, D^2, χ))

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

@testset "OMEinsum with $atype{$dtype} " for atype in [CuArray], dtype in [ComplexF64]
    Random.seed!(100)
    D = 4
    χ = 32
    Ni = 2
    Nj = 2
     M = atype(rand(dtype, D^2, D^2, D^2, D^2, Ni, Nj))
    AL = atype(rand(dtype, χ,   D^2, χ,        Ni, Nj))
     C = atype(rand(dtype, χ,        χ,        Ni, Nj))
    AC = atype(rand(dtype, χ,   D^2, χ,        Ni, Nj))
    FL = atype(rand(dtype, χ,   D^2, χ,        Ni, Nj))
    FR = atype(rand(dtype, χ,   D^2, χ,        Ni, Nj))

    function ACmap1(AC, FL, FR, M, j)
            Ni = size(M, 5)
            ACm = copy(AC)
            @inbounds @views for i in 1:Ni
                ir = i + 1 - Ni * (i==Ni)
                ACm[:,:,:,ir] .= ein"((adf,abc),dgeb),ceh -> fgh"(FL[:,:,:,i,j],AC[:,:,:,i],M[:,:,:,:,i,j],FR[:,:,:,i,j])
            end
        return ACm
    end
    
    function ACmap2(ACj, FLj, FRj, Mj)
        ACi = ein"((adfj,abcj),dgebj),cehj -> fghj"(FLj,ACj,Mj,FRj)
        circshift(ACi, (0,0,0,1))
    end
    @test ACmap1(AC[:,:,:,:,1], FL, FR, M, 1) ≈ ACmap2(AC[:,:,:,:,1], FL[:,:,:,:,1], FR[:,:,:,:,1], M[:,:,:,:,:,1])

    @btime CUDA.@sync $ACmap1($(AC[:,:,:,:,1]), $FL, $FR, $M, 1)
    # @btime CUDA.@sync $ACmap2($(AC[:,:,:,:,1]), $(FL[:,:,:,:,1]), $(FR[:,:,:,:,1]), $(M[:,:,:,:,:,1]))
end