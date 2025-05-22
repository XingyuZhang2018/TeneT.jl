@testset "FL with $atype{$dtype} " for atype in [ROCArray], dtype in [ComplexF64]
    Random.seed!(100)
    d, D, χ = 2, 2, 16

    set_device_id!(atype, 1)
    AL = atype(rand(dtype, χ,D,D,χ))
    ipeps = atype(rand(dtype, D,D,D,D,d))
    FL = atype(rand(dtype, χ,D,D,χ))

    AL = reshape(AL, χ, D^2, χ)
    M  = reshape(ein"abcde,fghie->afbgchdi"(ipeps, conj(ipeps)), D^2, D^2, D^2, D^2)
    FL = reshape(FL, χ, D^2, χ)

    FL1 = FLmap(FL, AL, AL, M)
    FL2 = FLmap_parallel(FL, AL, AL, M)
    
    @test reshape(FL1, χ, D^2, χ) ≈ FL2
    # @btime AMDGPU.@sync FLmap($FL, $AL, $AL, $M)
    # @btime AMDGPU.@sync FLmap_parallel($FL, $AL, $AL, $M)
end

@testset "FR with $atype{$dtype} " for atype in [ROCArray], dtype in [ComplexF64]
    Random.seed!(100)
    d, D, χ = 2, 2, 16

    set_device_id!(atype, 1)
    AR = atype(rand(dtype, χ,D,D,χ))
    ipeps = atype(rand(dtype, D,D,D,D,d))
    FR = atype(rand(dtype, χ,D,D,χ))

    AR = reshape(AR, χ, D^2, χ)
    M  = reshape(ein"abcde,fghie->afbgchdi"(ipeps, conj(ipeps)), D^2, D^2, D^2, D^2)
    FR = reshape(FR, χ, D^2, χ)

    FR1 = FRmap(FR, AR, AR, M)
    FR2 = FRmap_parallel(FR, AR, AR, M)
    
    @test reshape(FR1, χ, D^2, χ) ≈ FR2
    # @btime AMDGPU.@sync FRmap_parallel($FR, $AR, $AR, $M)
end

@testset "AC with $atype{$dtype} " for atype in [ROCArray], dtype in [ComplexF64]
    Random.seed!(100)
    d, D, χ = 2, 2, 16

    set_device_id!(atype, 1)
    AC = atype(rand(dtype, χ,D,D,χ))
    FL = atype(rand(dtype, χ,D,D,χ))
    ipeps = atype(rand(dtype, D,D,D,D,d))
    FR = atype(rand(dtype, χ,D,D,χ))

    AC = reshape(AC, χ, D^2, χ)
    FL = reshape(FL, χ, D^2, χ)
    M  = reshape(ein"abcde,fghie->afbgchdi"(ipeps, conj(ipeps)), D^2, D^2, D^2, D^2)
    FR = reshape(FR, χ, D^2, χ)

    AC1 = ACmap(AC, FL, FR, M)
    AC2 = ACmap_parallel(AC, FL, FR, M)
    
    @test reshape(AC1, χ, D^2, χ) ≈ AC2
    # @btime AMDGPU.@sync ACmap_parallel($AC, $FL, $FR, $M)
end