using TeneT
using TeneT: FLint, FRint
using TeneT: qrpos,lqpos,left_canonical,right_canonical,leftenv,FLmap,rightenv,FRmap,ACenv,ACmap,Cenv,Cmap,LRtoC,ALCtoAC,ACCtoALAR,error, env_norm
using TeneT: _to_front, _to_tail, permute_fronttail
using CUDA
using LinearAlgebra
using Random
using Test
using OMEinsum
CUDA.allowscalar(false)

test_type = [Array]
χ, D, d = 4, 3, 2
pattern = [1 2 3; 3 2 1]
test_As = [rand(ComplexF64, χ, D, χ)];
test_Ms = [rand(ComplexF64, D, D, D, D)];
alg = VUMPS(pattern=pattern)

@testset "TeneT.jl" begin
    # @testset "patch" begin
    #     println("patch tests running...")
    #     include("patch.jl")
    # end

    @testset "environment" begin
        println("environment tests running...")
        include("environment.jl")
    end

    @testset "fixedpoint" begin
        println("fixedpoint tests running...")
        include("fixedpoint.jl")
    end

    @testset "vumpsruntime.jl" begin
        println("vumpsruntime tests running...")
        include("vumpsruntime.jl")
    end

    @testset "autodiff.jl" begin
        println("autodiff tests running...")
        include("autodiff.jl")
    end
end
