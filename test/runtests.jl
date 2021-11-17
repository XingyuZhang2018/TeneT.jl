using VUMPS
using Test

@testset "VUMPS.jl" begin
    @testset "cuda_patch" begin
        println("cuda_patch tests running...")
        include("cuda_patch.jl")
    end

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
end
