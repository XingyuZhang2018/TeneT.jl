using TeneT
using Test

@testset "TeneT.jl" begin
    @testset "patch" begin
        println("patch tests running...")
        include("patch.jl")
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

    @testset "autodiff.jl" begin
        println("autodiff tests running...")
        include("autodiff.jl")
    end
end
