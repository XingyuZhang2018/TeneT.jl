using BenchmarkTools
using KrylovKit
using CUDA
using Test
using OMEinsum
using Random
using LinearAlgebra
using TeneT: left_canonical, right_canonical, leftenv, rightenv
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
    χ = 100
    A = [CuArray(rand(ComplexF64, χ, χ)) for _ in 1:4]
    @btime CUDA.@sync dot.($A, $A)
    # A = CuArray(rand(ComplexF64, χ*4, χ))
    # @btime CUDA.@sync dot($A, $A)

    # A = [CuArray(rand(ComplexF64, χ, χ)) for _ in 1:4]
    function foo(A)
        # B = CuArray(zeros(ComplexF64, 4*χ*χ))
        # B[1:χ^2] .= vec(A[1])
        # B[χ^2+1:2*χ^2] .= vec(A[2])
        # B[2χ^2+1:3*χ^2] .= vec(A[3])
        # B[3*χ^2+1:4*χ^2] .= vec(A[4])
        # return dot(B, B)
        for i in 1:4
            dot(Array(A[i]), Array(A[i]))
        end
    end
    @btime CUDA.@sync $foo($A)
end

# i9-14900KF RTX 4090
# χ = 30 D = 16 Ni = 2 Nj = 2
#   1.005 s (50126 allocations: 1.75 GiB)
#   1.013 s (56488 allocations: 1.87 GiB)
# Test Summary:                                     | Total   Time
# leftenv and rightenv with Array{ComplexF64} 2 x 2 |     0  30.4s
# χ = 30 D = 16 Ni = 2 Nj = 2
#   196.972 ms (141841 allocations: 4.98 MiB)
#   194.184 ms (152401 allocations: 5.35 MiB)
# Test Summary:                                       | Total   Time
# leftenv and rightenv with CuArray{ComplexF64} 2 x 2 |     0  33.0s
@testset "leftenv and rightenv with $atype{$dtype} $Ni x $Nj" for atype in [Array, CuArray], dtype in [ComplexF64], ifobs in [false], Ni in 2:2, Nj in 2:2
    Random.seed!(100)
    χ, D = 30, 16
    println("χ = $(χ) D = $(D) Ni = $(Ni) Nj = $(Nj)")
    A = [atype(rand(dtype, χ, D, χ)) for i in 1:Ni, j in 1:Nj]
    M = [atype(rand(dtype, D, D, D, D)) for i in 1:Ni, j in 1:Nj]

    AL,    =  left_canonical(A)
    @btime λL,FL  =  leftenv($AL, conj($AL), $M; ifobs = $ifobs)
    _, AR, = right_canonical(A)
    @btime λR,FR  = rightenv($AR, conj($AR), $M; ifobs = $ifobs)
end

# i9-14900KF RTX 4090
# χ = 30 D = 16 Ni = 2 Nj = 2
#   1.147 s (56377 allocations: 1.86 GiB)
#   29.043 ms (41332 allocations: 111.91 MiB)
# Test Summary:                               | Total   Time
# ACenv and Cenv with Array{ComplexF64} 2 x 2 |     0  30.7s
# χ = 30 D = 16 Ni = 2 Nj = 2
#   198.691 ms (147859 allocations: 4.39 MiB)
#   2.288 s (126598 allocations: 3.51 MiB)
# Test Summary:                                 | Total   Time
# ACenv and Cenv with CuArray{ComplexF64} 2 x 2 |     0  37.2s
@testset "ACenv and Cenv with $atype{$dtype} $Ni x $Nj" for atype in [CuArray], dtype in [ComplexF64], ifobs in [false], Ni in 1:1, Nj in 1:1
    Random.seed!(100)
    χ, D = 30, 16
    println("χ = $(χ) D = $(D) Ni = $(Ni) Nj = $(Nj)")
    A = [atype(rand(dtype, χ, D, χ)) for i in 1:Ni, j in 1:Nj]
    M = [atype(rand(dtype, D, D, D, D)) for i in 1:Ni, j in 1:Nj]

    AL,  L, _ =  left_canonical(A)
     R, AR, _ = right_canonical(A)
    λL, FL    =  leftenv(AL, conj(AL), M)
    λR, FR    = rightenv(AR, conj(AR), M)

     C =  LRtoC(  L, R)
    AC = ALCtoAC(AL, C)
    @btime λAC, AC = ACenv($AC, $FL, $M, $FR)
    @btime  λC,  C =  Cenv( $C, $FL,     $FR)
    # ProfileView.@profview ACenv(AC, FL, M, FR)
    # ProfileView.@profview λC,  C =  Cenv( C, FL,     FR)
end
