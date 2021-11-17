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
end
