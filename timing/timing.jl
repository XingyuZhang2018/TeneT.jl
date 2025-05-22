using BenchmarkTools
using KrylovKit
using CUDA, AMDGPU
using Test
using OMEinsum
using Random
using LinearAlgebra
using TeneT
using TeneT: left_canonical, right_canonical, leftenv, rightenv, LRtoC, ALCtoAC, ACenv, Cenv, _arraytype, FLmap, set_device_id!, to_N_device, FLmap_parallel
using TeneT: simple_eig
using Zygote
# using ProfileView
# CUDA.allowscalar(false)

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
@testset "OMEinsum with $atype{$dtype} " for atype in [CuArray], dtype in [ComplexF64]
    Random.seed!(100)
    d, D, χ = 4, 6, 32

    println("d = $(d) D = $(D) χ = $(χ)")
    AL = atype(rand(dtype, χ,D,D,χ))
    ipeps = atype(rand(dtype, D,D,D,D,d))
    FL = atype(rand(dtype, χ,D,D,χ))

    # FL1 = FLmap(FL, AL, conj(AL), ipeps)
    # _, FLm1 = simple_eig(FL -> FLmap(FL, AL, conj(AL), ipeps), FL)
    @btime CUDA.@sync FLmap($FL, $AL, conj($AL), $ipeps)
    # @btime CUDA.@sync norm($FL)
    @btime simple_eig(FL -> FLmap(FL, $AL, conj($AL), $ipeps), $FL)

    AL = reshape(AL, χ, D^2, χ)
    M  = reshape(ein"abcde,fghie->afbgchdi"(ipeps, conj(ipeps)), D^2, D^2, D^2, D^2)
    FL = reshape(FL, χ, D^2, χ)

    # FL2 = FLmap(FL, AL, conj(AL), M)
    # _, FLm2 = simple_eig(FL -> FLmap(FL, AL, conj(AL), M), FL)
    @btime CUDA.@sync FLmap($FL, $AL, conj($AL), $M)
    # @btime CUDA.@sync norm($FL)
    @btime simple_eig(FL -> FLmap(FL, $AL, conj($AL), $M), $FL)

    # @test reshape(FL1, χ, D^2, χ) ≈ FL2
    # @test reshape(FLm1, χ, D^2, χ) ≈ FLm2
    # @btime simple_eig(FL -> FLmap(FL, $AL, conj($AL), $ipeps), $FL)

end

@testset "copy with $atype{$dtype} " for atype in [ROCArray], dtype in [ComplexF64]
    d, D, χ = 4, 12, 256
    set_device_id!(atype, 1)
    A = atype(rand(dtype, χ,D,D,χ))
    function multi_copy(A)
        local B,C
        @sync begin
            @async begin
                set_device_id!(atype, 2)
                B = atype(A)
            end
            @async begin
                set_device_id!(atype, 3)
                C = atype(A)
            end
        end

        return B, C
    end
    @btime AMDGPU.@sync $multi_copy($A)
    # @btime AMDGPU.@sync copyto!($B, $A)
    # set_device_id!(atype, 2)
    # B = atype(rand(dtype, χ,D,D,χ))
    # copyto!(B, A)
    # @btime AMDGPU.@sync copyto!($B, $A)
    # @btime AMDGPU.@sync B = $atype($A)
end

@testset "OMEinsum with $atype{$dtype} " for atype in [ROCArray], dtype in [ComplexF64]
    Random.seed!(100)
    d, D, χ = 4, 8, 256

    println("d = $(d) D = $(D) χ = $(χ)")
    set_device_id!(atype, 1)
    AL = atype(rand(dtype, χ,D,D,χ))
    ipeps = atype(rand(dtype, D,D,D,D,d))
    FL = atype(rand(dtype, χ,D,D,χ))

    FL1 = FLmap(FL, AL, AL, ipeps)

    AL = reshape(AL, χ, D^2, χ)
    M  = reshape(ein"abcde,fghie->afbgchdi"(ipeps, conj(ipeps)), D^2, D^2, D^2, D^2)
    FL = reshape(FL, χ, D^2, χ)

    XX = AMDGPU.@time AMDGPU.@sync FLmap(FL, AL, AL, M)
    # @btime AMDGPU.@sync FLmap($FL, $AL, $AL, $M)

    FL22 = AMDGPU.@time FLmap_parallel(FL, AL, AL, M)
    # # _, FLm2 = simple_eig(FL -> FLmap(FL, AL, conj(AL), M), FL)
    
    # # @btime simple_eig(FL -> FLmap(FL, $AL, conj($AL), $M), $FL)
    # set_device_id!(atype, 1)
    @test reshape(FL1, χ, D^2, χ) ≈ FL22
    @btime AMDGPU.@sync FLmap_parallel($FL, $AL, $AL, $M)
    # @test reshape(FLm1, χ, D^2, χ) ≈ FLm2
    # @btime simple_eig(FL -> FLmap(FL, $AL, conj($AL), $ipeps), $FL)

end

@testset "contraction orders" for atype in [CuArray], dtype in [ComplexF64]
    Random.seed!(100)
    d, D, χ = 4, 8, 80

    println("d = $(d) D = $(D) χ = $(χ)")
    AL = atype(rand(dtype, χ,D,D,χ))
    ipeps = atype(rand(dtype, D,D,D,D,d))
    FL = atype(rand(dtype, χ,D,D,χ))

    # FRmap(FR, ARu, ARd, M::leg5) = ein"(((abcd,dghl),ejgbp),fkhcp),ijkl -> aefi"(ARu, FR, M, conj(M), ARd)
    code = ein"aefi,ijkl,ejgbp,fkhcp,abcd -> dghl"
    size_dict = OMEinsum.get_size_dict(getixsv(code), (FL, AL, ipeps, conj(ipeps), AL))
    code = optimize_code(code, size_dict, TreeSA(nslices=0))
    @show code contraction_complexity(code, size_dict)
    # @btime CUDA.@sync $code($FL, $AL, $ipeps, conj($ipeps), $AL)

    code = ein"(((aefi,ijkl),ejgb),fkhc),abcd -> dghl"
    @btime CUDA.@sync sum([$code($FL, $AL, $ipeps[:,:,:,:,i], conj($ipeps[:,:,:,:,i]), $AL) for i in 1:$d])
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

    # λs1, = eigsolve(FL -> ein"(((aefg,ghil),ehjbq),fikcq),abcd -> djkl"(FL,conj(AL),ipeps,conj(ipeps),AL), FL, 1, :LM; ishermitian = false)
    # @btime CUDA.@sync λs, FLs, info = eigsolve(FL -> ein"(((aefg,ghil),ehjbq),fikcq),abcd -> djkl"($FL,conj($AL),$ipeps,conj($ipeps),$AL), $FL, 1, :LM; ishermitian = false)

    AL = atype(reshape(AL, χ, D^2, χ))
    M = atype(reshape(ein"abcde,fghie->afbgchdi"(ipeps, conj(ipeps)), D^2, D^2, D^2, D^2))
    FL = atype(reshape(FL, χ, D^2, χ))

    ProfileView.@profview λs2, = eigsolve(FL -> ein"((adf,fgh),dgeb),abc -> ceh"(FL,conj(AL),M,AL), FL, 1, :LM; ishermitian = false)
    # @btime CUDA.@sync λs, FLs, info = eigsolve(FL -> ein"((adf,fgh),dgeb),abc -> ceh"($FL,conj($AL),$M,$AL), $FL, 1, :LM; ishermitian = false)

    # @test λs1 ≈ λs2
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
# χ = 30 D = 4 d = 2 Ni = 2 Nj = 2
#   1.184 s (76934 allocations: 3.43 GiB)
#   1.130 s (82684 allocations: 3.43 GiB)
# Test Summary:                                     | Total   Time
# leftenv and rightenv with Array{ComplexF64} 2 x 2 |     0  33.2s
# χ = 30 D = 4 d = 2 Ni = 2 Nj = 2
#   183.178 ms (185263 allocations: 6.70 MiB)
#   298.098 ms (195699 allocations: 7.08 MiB)
# Test Summary:                                       | Total   Time
# leftenv and rightenv with CuArray{ComplexF64} 2 x 2 |     0  28.6s
@testset "leftenv and rightenv with $atype{$dtype} $Ni x $Nj" for atype in [CuArray], dtype in [ComplexF64], ifobs in [false], Ni in 2:2, Nj in 2:2
    Random.seed!(100)
    χ, D = 30, 16
    println("χ = $(χ) D = $(D) Ni = $(Ni) Nj = $(Nj)")
    A = [atype(rand(dtype, χ, D, χ)) for i in 1:Ni, j in 1:Nj]
    M = [atype(rand(dtype, D, D, D, D)) for i in 1:Ni, j in 1:Nj]
    # χ, D, d = 30, 4, 2
    # println("χ = $(χ) D = $(D) d = $(d) Ni = $(Ni) Nj = $(Nj)")
    # A = [atype(rand(dtype, χ, D, D, χ)) for i in 1:Ni, j in 1:Nj]
    # M = [atype(rand(dtype, D, D, D, D, d)) for i in 1:Ni, j in 1:Nj]

    alg = VUMPS(ifsimple_eig=true)
    AL,    =  left_canonical(A)
    @time CUDA.@sync λL,FL  =  leftenv(AL, conj(AL), M; ifobs = ifobs, alg=alg)
    # @btime λL,FL  =  leftenv($AL, conj($AL), $M; ifobs = $ifobs, alg=$alg)
    # _, AR, = right_canonical(A)
    # @btime λR,FR  = rightenv($AR, conj($AR), $M; ifobs = $ifobs, alg=$alg)
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
@testset "ACenv and Cenv with $atype{$dtype} $Ni x $Nj" for atype in [Array, CuArray], dtype in [ComplexF64], ifobs in [false], Ni in 2:2, Nj in 2:2
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

begin "test utils"
    function num_grad(f, K; δ::Real=1e-5)
        if eltype(K) == ComplexF64
            (f(K + δ / 2) - f(K - δ / 2)) / δ + 
                (f(K + δ / 2 * 1.0im) - f(K - δ / 2 * 1.0im)) / δ * 1.0im
        else
            (f(K + δ / 2) - f(K - δ / 2)) / δ
        end
    end
    
    function num_grad(f, a::AbstractArray; δ::Real=1e-5)
        b = Array(copy(a))
        df = map(CartesianIndices(b)) do i
            foo = x -> (ac = copy(b); ac[i] = x; f(_arraytype(a)(ac)))
            num_grad(foo, b[i], δ=δ)
        end
        return _arraytype(a)(df)
    end
end

# i9-14900KF RTX 4090
# D = 16 d = 30
# ┌ Warning: `eigsolve` cotangent for eigenvector 1 is sensitive to gauge choice: (|gauge| = 1.0764722446765518e-12)
# └ @ KrylovKitChainRulesCoreExt C:\Users\xingzhan\.julia\packages\KrylovKit\jOhQS\ext\KrylovKitChainRulesCoreExt\eigsolve.jl:141
#   7.896 s (1488022 allocations: 15.63 GiB)
# Test Summary:                       | Total   Time
# 2x2 leftenv and rightenv with Array |     0  27.8s
# D = 16 d = 30
#   757.880 ms (533808 allocations: 24.18 MiB)
# Test Summary:                         | Total   Time
# 2x2 leftenv and rightenv with CuArray |     0  20.5s
@testset "$(Ni)x$(Nj) leftenv and rightenv with $atype" for atype in [Array], ifobs in [false], Ni = [2], Nj = [2]
    Random.seed!(100)
    D, d = 3, 2
    println("D = $(D) d = $(d)")
    A = [atype(rand(ComplexF64, D, d, D         )) for _ in 1:Ni, _ in 1:Nj]
    S = [atype(rand(ComplexF64, D, d, D, D, d, D)) for _ in 1:Ni, _ in 1:Nj]
    M = [atype(rand(ComplexF64, d, d, d, d      )) for _ in 1:Ni, _ in 1:Nj]

       ALu, =  left_canonical(A) 
    _, ARu, = right_canonical(A)

    function foo1(M)
        _,FL = leftenv(ALu, conj(ALu), M; ifobs)
        s = 0.0
        for j in 1:Nj, i in 1:Ni
            A  = Array(ein"(abc,abcdef),def -> "(FL[i,j], S[i,j], FL[i,j]))[]
            B  = Array(ein"abc,abc -> "(FL[i,j], FL[i,j]))[]
            s += norm(A/B)
        end
        return s
    end
    # @btime CUDA.@sync Zygote.gradient($foo1, $M)[1]
    @show norm(Zygote.gradient(foo1, M)[1]  - num_grad(foo1, M))
end