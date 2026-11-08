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

@testset "Honeycomb merge observable visualization" begin
    @testset "J2 sublattice offsets" begin
        @test TeneT._bond_offsets_honeycomb_merge("bond_J2H_energy") ==
              (1, 1, (0, 0), (0, 1))
        @test TeneT._bond_offsets_honeycomb_merge("bond_J2V_energy") ==
              (1, 1, (0, 0), (1, 0))
        @test TeneT._bond_offsets_honeycomb_merge("bond_J2/_energy") ==
              (1, 1, (0, 1), (1, 0))

        @test TeneT._bond_offsets_honeycomb_merge("bond_J2H2_energy") ==
              (2, 2, (0, 0), (0, 1))
        @test TeneT._bond_offsets_honeycomb_merge("bond_J2V2_energy") ==
              (2, 2, (0, 0), (1, 0))
        @test TeneT._bond_offsets_honeycomb_merge("bond_J2/2_energy") ==
              (2, 2, (0, 1), (1, 0))
    end

    mktempdir() do folder
        chi = Char(0x03c7)
        open(joinpath(folder, string(chi, "8.log")), "w") do io
            write(io, "energy_per_site:\n-0.100000000000000\n")
            for bond_type in ("bond_J1_onsite_energy", "bond_J1H_energy",
                              "bond_J1V_energy", "bond_J2H_energy",
                              "bond_J2V_energy", "bond_J2/_energy",
                              "bond_J2H2_energy", "bond_J2V2_energy",
                              "bond_J2/2_energy")
                write(io, "$bond_type: i j energy\n")
                write(io, "1,1 -0.010000000000000\t\n")
            end
            write(io, "magnetization_norm_per_site:\n0.200000000000000\n")
            write(io, "magnetization: i j |M| Mx My Mz\n")
            write(io, "1,1,1 0.200000000000000 0.100000000000000 0.000000000000000 0.170000000000000\n")
            write(io, "1,1,2 0.200000000000000 -0.100000000000000 0.000000000000000 -0.170000000000000\n")
            write(io, "correlation_length:\n0.300000000000000\n")
        end

        TeneT.plot_observables(folder, Honeycomb(:merge), [1;;]; save_format="png", S=0.5)
        @test isfile(joinpath(folder, "convergence.png"))
        @test filesize(joinpath(folder, "convergence.png")) > 0
        @test isfile(joinpath(folder, string("lattice_", chi, "8.png")))
        @test filesize(joinpath(folder, string("lattice_", chi, "8.png"))) > 0
    end
end
