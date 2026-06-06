@testset "TM_spectrum observable" begin
    using TeneT: Heisenberg

    function _tm_test_params(; model, alg, pattern=[1;;])
        return GradientOptimize(model=model,
                                pattern=pattern,
                                boundary_alg=alg,
                                verbosity=0,
                                folder=mktempdir(),
                                ifload_env=false,
                                ifsave_env=false,
                                ifplot=false,
                                ifsave_lbfgs=false,
                                ifload_lbfgs=false)
    end

    @testset "export and input guards" begin
        @test :TM_spectrum in names(TeneT)

        A = rand(Float64, 2, 1, 2, 2, 2, 1)
        χ = 2
        brickwall = Heisenberg(lattice=Honeycomb{:brickwall_h}(),
                               S=0.5, Jx=1.0, Jy=1.0, Jz=1.0,
                               ifrotate=false)

        c4v_params = _tm_test_params(model=brickwall,
                                     alg=VUMPS{C4v}(; verbosity=0))
        c4v_err = try
            TeneT.TM_spectrum(1, 0.0, A, χ, c4v_params)
            nothing
        catch e
            e
        end
        @test c4v_err isa ArgumentError
        @test occursin("TM_spectrum currently supports only VUMPS{General}",
                       sprint(showerror, c4v_err))

        square_model = Heisenberg(lattice=Square(),
                                  S=0.5, Jx=1.0, Jy=1.0, Jz=1.0,
                                  ifrotate=false)
        square_params = _tm_test_params(model=square_model,
                                        alg=VUMPS{General}(; verbosity=0,
                                                           ifupdown=false))
        square_err = try
            TeneT.TM_spectrum(1, 0.0, A, χ, square_params)
            nothing
        catch e
            e
        end
        @test square_err isa ArgumentError
        @test occursin(
            "TM_spectrum currently supports only Honeycomb{:brickwall_h} and Honeycomb{:brickwall_v}",
            sprint(showerror, square_err),
        )
    end

    @testset "spectrum writer paths" begin
        params = _tm_test_params(
            model=Heisenberg(lattice=Honeycomb{:brickwall_h}(),
                             S=0.5, Jx=1.0, Jy=1.0, Jz=1.0,
                             ifrotate=false),
            alg=VUMPS{General}(; verbosity=0, ifupdown=false),
        )

        TeneT._write_tm_spectrum([0.125, 0.25], 0.0, 2, 4, params;
                                 ifdomainwall=false)
        trivial = joinpath(params.folder, "D2_χ4", "TM_spectrum",
                           "trivial", "k0.0.log")
        @test isfile(trivial)
        @test readlines(trivial) == ["0.125000000000000",
                                     "0.250000000000000"]

        TeneT._write_tm_spectrum([0.5], 0.0, 2, 4, params;
                                 ifdomainwall=true)
        nontrivial = joinpath(params.folder, "D2_χ4", "TM_spectrum",
                              "non-trivial", "k0.0.log")
        @test isfile(nontrivial)
        @test readlines(nontrivial) == ["0.500000000000000"]
    end
end
