using TeneT
using TeneT: _arraytype
using TeneT: StructArray, ISA
using TeneT: qrpos,lqpos,left_canonical,right_canonical,leftenv,FLmap,rightenv,FRmap,ACenv,ACmap,Cenv,Cmap,LRtoC,ALCtoAC,ACCtoALAR,error, env_norm
using TeneT: _to_front, _to_tail, permute_fronttail
using TeneT: project_AL!, project_AR!, retract!, project_AL, project_AR, permute_fronttail
using Test
using LinearAlgebra
using AMDGPU
using CUDA
using Random
using Test
using OMEinsum
using KrylovKit
using Zygote
CUDA.allowscalar(false)

test_type = [Array, ROCArray];
χ, D, d = 4, 3, 2;
test_As = [randSA(atype, [1 2; 2 1], [(χ, D, χ), (χ, D, χ)]) for atype in test_type];
test_Ms = [randSA(atype, [1 2; 2 1], [(D, D, D, D), (D, D, D, D)]) for atype in test_type];
test_S1s = [randSA(atype, [1 2; 2 1], [(χ, D, χ, χ, D, χ), (χ, D, χ, χ, D, χ)]) for atype in test_type];
test_S2s = [randSA(atype, [1 2; 2 1], [(χ, χ, χ, χ), (χ, χ, χ, χ)]) for atype in test_type];

@testset "TeneT.jl" begin
    @testset "patch" begin
        println("patch tests running...")
        include("patch.jl")
    end

    @testset "structarray" begin
        println("structarray tests running...")
        include("structarray.jl")
    end

    @testset "environment" begin
        println("environment tests running...")
        include("environment.jl")
    end

    @testset "vumpsruntime.jl" begin
        println("vumpsruntime tests running...")
        include("vumpsruntime.jl")
    end

    @testset "autodiff.jl" begin
        println("autodiff tests running...")
        include("autodiff.jl")
    end
end;