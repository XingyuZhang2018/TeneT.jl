using BenchmarkTools
using KrylovKit
using CUDA
using Test
using OMEinsum
using Random
using LinearAlgebra
CUDA.allowscalar(false)

# i9-14900KF RTX 4090
# d = 4 D = 6 χ = 32
#   53.926 ms (635 allocations: 244.87 MiB)
#   50.533 ms (431 allocations: 82.70 MiB)
# Test Summary:                    | Pass  Total   Time
# OMEinsum with Array{ComplexF64}  |    1      1  26.8s
# d = 4 D = 6 χ = 32
#   4.727 ms (1123 allocations: 40.47 KiB)
#   13.570 ms (762 allocations: 25.95 KiB)
# Test Summary:                      | Pass  Total   Time
# OMEinsum with CuArray{ComplexF64}  |    1      1  23.1s
@testset "OMEinsum with $atype{$dtype} " for atype in [Array, CuArray], dtype in [ComplexF64]
    Random.seed!(100)
    d, D, χ = 4, 6, 32

    println("d = $(d) D = $(D) χ = $(χ)")
    AL = atype(rand(dtype, χ,D,D,χ))
    ipeps = atype(rand(dtype, D,D,D,D,d))
    FL = atype(rand(dtype, χ,D,D,χ))

    FL1 = ein"(((aefg,ghil),ehjbq),fikcq),abcd -> djkl"(FL,conj(AL),ipeps,conj(ipeps),AL)
    @btime CUDA.@sync ein"(((aefg,ghil),ehjbq),fikcq),abcd -> djkl"($FL,conj($AL),$ipeps,conj($ipeps),$AL)
    
    AL = atype(reshape(AL, χ, D^2, χ))
    M = atype(reshape(ein"abcde,fghie->afbgchdi"(ipeps, conj(ipeps)), D^2, D^2, D^2, D^2))
    FL = atype(reshape(FL, χ, D^2, χ))

    FL2 = ein"((adf,fgh),dgeb),abc -> ceh"(FL,conj(AL),M,AL)
    @btime CUDA.@sync ein"((adf,fgh),dgeb),abc -> ceh"($FL,conj($AL),$M,$AL)

    @test reshape(FL1, χ, D^2, χ) ≈ FL2
end

# i9-14900KF RTX 4090
# d = 4 D = 6 χ = 32
#   171.311 ms (2062 allocations: 739.72 MiB)
#   158.650 ms (1450 allocations: 253.22 MiB)
# Test Summary:                    | Pass  Total   Time
# KrylovKit with Array{ComplexF64} |    1      1  31.9s
# d = 4 D = 6 χ = 32
#   16.203 ms (4579 allocations: 184.39 KiB)
#   43.923 ms (3496 allocations: 140.72 KiB)
# Test Summary:                      | Pass  Total   Time
# KrylovKit with CuArray{ComplexF64} |    1      1  24.2s
@testset "KrylovKit with $atype{$dtype}" for atype in [Array, CuArray], dtype in [ComplexF64]
    Random.seed!(100)
    d, D, χ = 4, 6, 32

    println("d = $(d) D = $(D) χ = $(χ)")
    AL = atype(rand(dtype, χ,D,D,χ))
    ipeps = atype(rand(dtype, D,D,D,D,d))
    FL = atype(rand(dtype, χ,D,D,χ))

    λs1, = eigsolve(FL -> ein"(((aefg,ghil),ehjbq),fikcq),abcd -> djkl"(FL,conj(AL),ipeps,conj(ipeps),AL), FL, 1, :LM; ishermitian = false)
    @btime CUDA.@sync λs, FLs, info = eigsolve(FL -> ein"(((aefg,ghil),ehjbq),fikcq),abcd -> djkl"($FL,conj($AL),$ipeps,conj($ipeps),$AL), $FL, 1, :LM; ishermitian = false)

    AL = atype(reshape(AL, χ, D^2, χ))
    M = atype(reshape(ein"abcde,fghie->afbgchdi"(ipeps, conj(ipeps)), D^2, D^2, D^2, D^2))
    FL = atype(reshape(FL, χ, D^2, χ))

    λs2, = eigsolve(FL -> ein"((adf,fgh),dgeb),abc -> ceh"(FL,conj(AL),M,AL), FL, 1, :LM; ishermitian = false)
    @btime CUDA.@sync λs, FLs, info = eigsolve(FL -> ein"((adf,fgh),dgeb),abc -> ceh"($FL,conj($AL),$M,$AL), $FL, 1, :LM; ishermitian = false)

    @test λs1 ≈ λs2
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

@testset "Array of CuArray" begin
    Random.seed!(100)
    χ = 1024
    A = [CuArray(rand(ComplexF64, χ, χ)) for _ in 1:10]
    @show dot.(A, A)
    @btime CUDA.@sync dot.($A, $A)
end
