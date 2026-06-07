@testset "TM_spectrum observable" begin
    using TeneT: Heisenberg, J1J2p

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

    @testset "forloop iteration source" begin
        params = _tm_test_params(
            model=Heisenberg(lattice=Honeycomb{:brickwall_h}(),
                             S=0.5, Jx=1.0, Jy=1.0, Jz=1.0,
                             ifrotate=false),
            alg=VUMPS{General}(; verbosity=0, ifupdown=false,
                               forloop_iter=7),
        )
        params.forloop_iter = 3

        @test TeneT._tm_forloop_iter(params) == 7

        compatibility_params = (boundary_alg=(;), forloop_iter=3)
        @test TeneT._tm_forloop_iter(compatibility_params) == 3
    end

    @testset "Oneside row pairing" begin
        model = J1J2p(lattice=Honeycomb{:brickwall_v}(),
                       S=0.5, J1=1.0, J2p=0.5,
                       ifrotate=false, couplingtype=:uniform)
        pattern = [1 2; 2 1]
        general_params = _tm_test_params(
            model=model,
            alg=VUMPS{General}(; verbosity=0, ifupdown=true),
            pattern=pattern,
        )
        oneside_params = _tm_test_params(
            model=model,
            alg=VUMPS{General}(; verbosity=0, ifupdown=false),
            pattern=pattern,
        )

        @test !TeneT._tm_is_oneside(general_params)
        @test TeneT._tm_is_oneside(oneside_params)
        @test TeneT._tm_partner_row(general_params, 1, 2) == 2
        @test TeneT._tm_partner_row(oneside_params, 1, 2) == 1
        @test TeneT._tm_partner_row(oneside_params, 2, 2) == 2
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
        invalid_inputs = (
            (0, 0.0, χ, "TM_spectrum requires n > 0; got 0."),
            (1, 0.0, 0,
             "TM_spectrum requires χ to be a positive Int; got 0 of type Int64."),
            (1, 0.0, 2.5,
             "TM_spectrum requires χ to be a positive Int; got 2.5 of type Float64."),
            (1, 0.0, 2 // 1,
             "TM_spectrum requires χ to be a positive Int; got 2//1 of type Rational{Int64}."),
            (1, 0.0, true,
             "TM_spectrum requires χ to be a positive Int; got true of type Bool."),
            (1, Inf, χ, "TM_spectrum requires finite k; got Inf."),
            (1, NaN, χ, "TM_spectrum requires finite k; got NaN."),
        )
        for (n_invalid, k_invalid, χ_invalid, message) in invalid_inputs
            err = try
                TeneT.TM_spectrum(n_invalid, k_invalid, A, χ_invalid,
                                  c4v_params)
                nothing
            catch e
                e
            end
            @test err isa ArgumentError
            @test sprint(showerror, err) == "ArgumentError: $message"
        end

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

        rational_folder = joinpath(params.folder, "D3_χ5", "TM_spectrum",
                                   "trivial")
        rational = joinpath(rational_folder, "k1_over_3.log")
        rational_result = try
            TeneT._write_tm_spectrum([0.375], 1 // 3, 3, 5, params;
                                     ifdomainwall=false)
        catch e
            e
        end
        @test rational_result == rational
        @test readdir(rational_folder) == ["k1_over_3.log"]
        @test isfile(rational)
        if isfile(rational)
            @test readlines(rational) == ["0.375000000000000"]
        end

        @test TeneT._tm_k_filename(0.0) == "0.0"
        @test TeneT._tm_k_filename(1 // 3) == "1_over_3"

        large_k = setprecision(BigFloat, 4096) do
            BigFloat(1) / BigFloat(3)
        end
        large_name = TeneT._tm_k_filename(large_k)
        @test large_name == TeneT._tm_k_filename(large_k)
        @test ncodeunits("k$large_name.log") <= 100

        large_folder = joinpath(params.folder, "D6_χ7", "TM_spectrum",
                                "trivial")
        large_path = joinpath(large_folder, "k$large_name.log")
        large_result = try
            TeneT._write_tm_spectrum([0.625], large_k, 6, 7, params;
                                     ifdomainwall=false)
        catch e
            e
        end
        @test large_result == large_path
        @test isfile(large_path)
        if isfile(large_path)
            @test readlines(large_path) == ["0.625000000000000"]
        end

        TeneT._write_tm_spectrum([0.5], 0.0, 2, 4, params;
                                 ifdomainwall=true)
        nontrivial = joinpath(params.folder, "D2_χ4", "TM_spectrum",
                              "non-trivial", "k0.0.log")
        @test isfile(nontrivial)
        @test readlines(nontrivial) == ["0.500000000000000"]
    end

    @testset "small VUMPS General brickwall spectra" begin
        for (seed, lattice) in ((17, Honeycomb{:brickwall_h}()),
                                (23, Honeycomb{:brickwall_v}()))
            Random.seed!(seed)
            D, d, χ = 2, 2, 4
            pattern = [1 2; 2 1]
            N = length(unique(pattern))
            model = Heisenberg(lattice=lattice,
                               S=0.5, Jx=1.0, Jy=1.0, Jz=1.0,
                               ifrotate=false)
            alg = VUMPS{General}(; verbosity=0,
                                  maxiter=2, miniter=0,
                                  maxiter_ad=0, miniter_ad=0,
                                  power_iter=2, show_every=1000,
                                  tol=1e-3, ifupdown=true,
                                  ifsimple_eig=true)
            params = _tm_test_params(; model, alg, pattern)
            params.ifsave_env = true
            A = TeneT._init_random_ipeps(lattice, Float64, D, d, N,
                                         size(pattern)...)
            A ./= norm(A)

            Δ = TM_spectrum(1, 0.0, A, χ, params; ifdomainwall=false)
            @test length(Δ) == 1
            @test all(isfinite, Δ)
            @test isfile(joinpath(params.folder, "D2_χ4", "TM_spectrum",
                                  "trivial", "k0.0.log"))
            @test isfile(joinpath(params.folder, "D2", "environment",
                                  "χ4.jld2"))

            Δdw = TM_spectrum(1, 0.0, A, χ, params; ifdomainwall=true)
            @test length(Δdw) == 1
            @test all(isfinite, Δdw)
            @test isfile(joinpath(params.folder, "D2_χ4", "TM_spectrum",
                                  "non-trivial", "k0.0.log"))
            for file in ("χ4_1.jld2", "χ4_2.jld2")
                @test isfile(joinpath(params.folder, "D2", "environment",
                                      file))
            end

            params.ifload_env = true
            loaded = TeneT._tm_initialize_named_env(
                A, D, χ, params, "χ4_1.jld2";
                restriction_ipeps=TeneT._restriction_ipeps,
            )
            @test loaded isa Tuple
            @test all(rt -> rt isa TeneT.VUMPSRuntime, loaded)
        end
    end

    @testset "small VUMPS Oneside brickwall spectra" begin
        function oneside_restriction(A)
            B = similar(A)
            for site in axes(A, 6)
                tensor = A[:, :, :, :, :, site]
                B[:, :, :, :, :, site] =
                    tensor + permutedims(tensor, (1, 4, 3, 2, 5))
            end
            return B / norm(B)
        end

        cases = (
            (31,
             Heisenberg(lattice=Honeycomb{:brickwall_h}(),
                        S=0.5, Jx=1.0, Jy=1.0, Jz=1.0,
                        ifrotate=false),
             TeneT._restriction_ipeps),
            (37,
             J1J2p(lattice=Honeycomb{:brickwall_v}(),
                    S=0.5, J1=1.0, J2p=0.5,
                    ifrotate=false, couplingtype=:uniform),
             oneside_restriction),
        )

        for (seed, model, restriction_ipeps) in cases
            Random.seed!(seed)
            lattice = model.lattice
            D, d, χ = 2, 2, 4
            pattern = [1 2; 2 1]
            N = length(unique(pattern))
            alg = VUMPS{General}(; verbosity=0,
                                  maxiter=2, miniter=0,
                                  maxiter_ad=0, miniter_ad=0,
                                  power_iter=2, show_every=1000,
                                  tol=1e-3, ifupdown=false,
                                  ifsimple_eig=true)
            params = _tm_test_params(; model, alg, pattern)
            params.ifsave_env = true
            A = TeneT._init_random_ipeps(lattice, Float64, D, d, N,
                                         size(pattern)...)
            A ./= norm(A)

            Δ = TM_spectrum(1, 0.0, A, χ, params;
                            restriction_ipeps, ifdomainwall=false)
            @test length(Δ) == 1
            @test all(isfinite, Δ)
            @test isfile(joinpath(params.folder, "D2_χ4", "TM_spectrum",
                                  "trivial", "k0.0.log"))
            @test isfile(joinpath(params.folder, "D2", "environment",
                                  "χ4.jld2"))

            Δdw = TM_spectrum(1, 0.0, A, χ, params;
                              restriction_ipeps, ifdomainwall=true)
            @test length(Δdw) == 1
            @test all(isfinite, Δdw)
            @test isfile(joinpath(params.folder, "D2_χ4", "TM_spectrum",
                                  "non-trivial", "k0.0.log"))
            for file in ("χ4_1.jld2", "χ4_2.jld2")
                path = joinpath(params.folder, "D2", "environment", file)
                @test isfile(path)
                params.ifload_env = true
                loaded = TeneT._tm_initialize_named_env(
                    A, D, χ, params, file; restriction_ipeps,
                )
                @test loaded isa TeneT.VUMPSRuntime
            end
        end
    end
end
