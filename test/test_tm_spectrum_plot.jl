using Test
using TeneT
using CairoMakie

@testset "TM spectrum visualization" begin
    @testset "in-memory spectrum" begin
        mktempdir() do folder
            results = Dict(
                1.0 => [0.3, 0.6],
                -1.0 => [0.2, 0.5],
                0.0 => [0.1, 0.4],
            )
            output = joinpath(folder, "spectrum.png")
            fig = plot_TM_spectrum(
                results;
                save_path=output,
                title="Test spectrum",
                xlimits=(-1, 1),
            )

            @test fig isa Figure
            @test isfile(output)
            @test filesize(output) > 0
        end
    end

    @testset "spectrum log directory" begin
        mktempdir() do folder
            write(joinpath(folder, "k-1.0.log"), "0.2\n0.5\n")
            write(joinpath(folder, "k0.0.log"), "0.1\n")
            write(joinpath(folder, "k1_over_2.log"), "0.3\n0.6\n")
            write(joinpath(folder, "notes.txt"), "ignored\n")

            results = TeneT._read_TM_spectrum(folder; nlevels=2)
            @test sort(collect(keys(results))) == [-1.0, 0.5]
            @test results[-1.0] == [0.2, 0.5]
            @test results[0.5] == [0.3, 0.6]

            output = joinpath(folder, "from_logs.png")
            fig = plot_TM_spectrum(
                folder;
                nlevels=2,
                xscale=0.5,
                save_path=output,
            )
            @test fig isa Figure
            @test isfile(output)
            @test filesize(output) > 0
        end
    end

    @testset "invalid input" begin
        @test_throws ArgumentError plot_TM_spectrum(Dict{Float64, Vector{Float64}}())

        mktempdir() do folder
            write(joinpath(folder, "k0.0.log"), "0.1\n")
            @test_throws ArgumentError plot_TM_spectrum(folder; nlevels=2)
            @test_throws ArgumentError plot_TM_spectrum(folder; xscale=0)
        end
    end
end
