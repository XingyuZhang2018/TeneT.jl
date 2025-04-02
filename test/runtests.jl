println("tests setup...")
include("test_setup.jl")

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