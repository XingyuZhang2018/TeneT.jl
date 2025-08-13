using OMEinsum
using Test
using Random
using AMDGPU
using TeneT
using TeneT: FLmap, FRmap, ACmap
using TeneT: set_device_id!, get_device_id, to_N_device, FLmap_forloop

@testset "FL with $atype{$dtype} " for atype in [ROCArray], dtype in [ComplexF64]
    Random.seed!(100)
    # d, D, χ = 2, 20, 512

    # set_device_id!(atype, 2)
    # AL = atype(rand(dtype, χ,D,D,χ))
    # M = atype(rand(dtype, D,D,D,D,d))
    # FL = atype(rand(dtype, χ,D,D,χ))

    # AL = reshape(AL, χ, D^2, χ)
    # M  = reshape(ein"abcde,fghie->afbgchdi"(M, conj(M)), D^2, D^2, D^2, D^2)
    # FL = reshape(FL, χ, D^2, χ)

    # FL1 = FLmap(FL, AL, view(AL, :,:,:,1:div(χ,4)), M)
    # FL2 = FLmap_parallel(FL, AL, AL, M)
    
    # @test reshape(FL1, χ, D^2, χ) ≈ FL2
    # @show TeneT.get_device_id(FL2s[1])
    # @show TeneT.get_device_id(FL2s[2])
    # @show TeneT.get_device_id(FL2s[3])
    # @show TeneT.get_device_id(FL2s[4])
    # @show typeof(FL1) typeof(FL2)
    # @btime AMDGPU.@sync FLmap($FL, $AL, $AL, $M)
    # @btime AMDGPU.@sync FLmap_parallel($FL, $AL, $AL, $M)
    # TeneT.gc(CuArray)
    # A = ein"aefi,ijkl->aefjkl"(FL,view(AL, :,:,:,1:div(χ,256)))
    # B = ein"aefjkl,ejgbp->agfbkl"(A,M)
    # CUDA.@time begin
    #     N = 512
    #     for _ in 1:N
    #         FLmap(FL, AL, view(AL, :,:,:, 1:div(χ, N)), M)
    #         # A .= ein"aefi,ijkl->aefjkl"(FL,view(AL, :,:,:,1:div(χ,256)))
    #         # B .= ein"aefjkl,ejgbp->agfbkl"(A,M)
    #     end
    # end
    d = 2
    N = 256
    ts = []
    TeneT.set_device_id!(atype, 1)
    for χ in [512], D in [10]
        AL = atype(rand(dtype, χ,D,D,χ))
        M = atype(rand(dtype, D,D,D,D,d))
        FL = atype(rand(dtype, χ,D,D,χ))

        TeneT.FLmap(FL, AL, conj(AL[:,:,:,1:div(χ, N)]), M)
        t = @elapsed AMDGPU.@sync begin
            for _ in 1:N
                TeneT.FLmap(FL, AL, conj(AL[:,:,:,1:div(χ, N)]), M)
            end
        end
        @show D,t
        push!(ts, t)
    end
    @show ts
    # TeneT.gc(CuArray)
    # CUDA.@time begin
    #     for _ in 1:5
    #         FLmap_parallel(FL, AL, AL, M)
    #     end
    # end
end

# @testset "FR with $atype{$dtype} " for atype in [ROCArray], dtype in [ComplexF64]
#     Random.seed!(100)
#     d, D, χ = 2, 2, 16

#     set_device_id!(atype, 1)
#     AR = atype(rand(dtype, χ,D,D,χ))
#     ipeps = atype(rand(dtype, D,D,D,D,d))
#     FR = atype(rand(dtype, χ,D,D,χ))

#     AR = reshape(AR, χ, D^2, χ)
#     M  = reshape(ein"abcde,fghie->afbgchdi"(ipeps, conj(ipeps)), D^2, D^2, D^2, D^2)
#     FR = reshape(FR, χ, D^2, χ)

#     FR1 = FRmap(FR, AR, AR, M)
#     FR2 = FRmap_parallel(FR, AR, AR, M)
    
#     @test reshape(FR1, χ, D^2, χ) ≈ FR2
#     # @btime AMDGPU.@sync FRmap_parallel($FR, $AR, $AR, $M)
# end

# @testset "AC with $atype{$dtype} " for atype in [ROCArray], dtype in [ComplexF64]
#     Random.seed!(100)
#     d, D, χ = 2, 2, 16

#     set_device_id!(atype, 1)
#     AC = atype(rand(dtype, χ,D,D,χ))
#     FL = atype(rand(dtype, χ,D,D,χ))
#     ipeps = atype(rand(dtype, D,D,D,D,d))
#     FR = atype(rand(dtype, χ,D,D,χ))

#     AC = reshape(AC, χ, D^2, χ)
#     FL = reshape(FL, χ, D^2, χ)
#     M  = reshape(ein"abcde,fghie->afbgchdi"(ipeps, conj(ipeps)), D^2, D^2, D^2, D^2)
#     FR = reshape(FR, χ, D^2, χ)

#     AC1 = ACmap(AC, FL, FR, M)
#     AC2 = ACmap_parallel(AC, FL, FR, M)
    
#     @test reshape(AC1, χ, D^2, χ) ≈ AC2
#     # @btime AMDGPU.@sync ACmap_parallel($AC, $FL, $FR, $M)
# end