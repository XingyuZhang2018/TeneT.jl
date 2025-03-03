using TeneT
using TeneT: _arraytype
using TeneT: leg3, leg4, ipeps, bulk, initial_A, initial_C
using TeneT: ρmap, getL!, getAL, getLsped
using TeneT: left_canonical, right_canonical, _to_front, _to_tail
using TeneT: initial_FL, initial_FR, FLmap, FRmap, leftenv, rightenv
using TeneT: ACmap, Cmap, ACenv, Cenv, LRtoC, ALCtoAC, ACCtoALAR
using TeneT: Lmap, leftCenv
using Test
using LinearAlgebra
using CUDA
using Random
using Test
using TensorKit
using KrylovKit
using Zygote
CUDA.allowscalar(false)

include("../example/exampletensors.jl")
include("../example/exampleobs.jl")

test_type = [Array]

ds = [ℂ^2]
Ds = [ℂ^3]
χs = [ℂ^4]
Ms = [[rand(ComplexF64, [D*D*D'*D' ← d, D*D*D'*D' ← d],  [1 2; 2 1]) for (d, D, χ) in zip(ds, Ds, χs)]...,
      [rand(ComplexF64, [D*D ← D*D, D*D ← D*D],          [1 2; 2 1]) for (d, D, χ) in zip(ds, Ds, χs)]...]
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