@testset "Kagome :onehole end-to-end" begin
    using OptimKit: LBFGS, HagerZhangLineSearch
    using TeneT: ObsEnv, energy_value, magnetization_value, build_A,
                 leading_boundary, initialize_env

    # Small, fast settings — just exercise every code path on the :onehole model
    seed = 42
    Random.seed!(seed)
    D, χ, χshift = 2, 4, 2
    pattern = [1 3;
               2 4]
    model = Heisenberg(lattice=Kagome(:onehole),
                       S=0.5, Jx=1.0, Jy=1.0, Jz=1.0,
                       ifrotate=false, couplingtype=:uniform, bondratio=1.0)
    folder = mktempdir()
    boundary_alg = VUMPS{TeneT.General}(ifupdown=true, ifsimple_eig=true,
                                        maxiter=4, miniter=0, maxiter_ad=2, miniter_ad=2,
                                        tol=1e-4, verbosity=0, show_every=1000)
    params = GradientOptimize(model=model, pattern=pattern, boundary_alg=boundary_alg,
                              optimizer=LBFGS(10; maxiter=1, gradtol=1e-3, verbosity=0),
                              maxiter_restart=1, verbosity=0, folder=folder,
                              ifSU=false, SUτ=0.0, ifprecondition=false,
                              reuse_env=true, ifsave_env=false, ifload_env=false,
                              ifsave_lbfgs=false, ifload_lbfgs=false)

    # ---- show string distinguishes :onehole from :merge -----------------------
    @testset "show / folder name" begin
        @test occursin("Kagome_onehole", string(model))
        @test !occursin("Kagome_onehole", string(Heisenberg(lattice=Kagome(:merge), S=0.5)))
    end

    # ---- init & shape ---------------------------------------------------------
    A_raw = init_ipeps(; atype=Array, etype=Float64, No=0, D=D, χ=χ, params=params)
    @test size(A_raw) == (D, D, D, D, 2, 4)
    A = build_A(A_raw, params)

    # ---- VUMPS converges + energy_value returns finite value -----------------
    rt = initialize_env(A_raw, D, χ, params)
    rt, _ = leading_boundary(rt, A, params.boundary_alg)
    env = ObsEnv(rt, A, params.boundary_alg)

    @testset "energy_value: 6 bonds, finite, intensive" begin
        e, e_dict = energy_value(model, A, env, params)
        @test isfinite(e)
        @test length(e_dict) == 6
        for key in ("bond_AC_H_energy", "bond_AB_V_energy", "bond_BC_diag_energy",
                    "bond_CA_H_cross_energy", "bond_BA_V_cross_energy", "bond_BC_diag_cross_energy")
            @test haskey(e_dict, key)
            @test !isempty(e_dict[key])
            for (_, v) in e_dict[key]
                @test isfinite(v)
            end
        end
    end

    # ---- magnetization_value skips empty site --------------------------------
    @testset "magnetization_value: 3 physical sites only" begin
        M_mean, m_dict = magnetization_value(model, A, env, params)
        @test isfinite(M_mean)
        # 3 physical entries (A,B,C) + 1 empty trivial entry = 4 keys
        @test length(m_dict) == 4
        @test haskey(m_dict, "1,1") && haskey(m_dict, "2,1") && haskey(m_dict, "1,2")
        @test haskey(m_dict, "2,2")  # empty entry is still present (for plot)
        @test abs(m_dict["2,2"]["|M|"]) < 1e-10  # but it has |M|=0
    end

    # ---- end-to-end optimise_ipeps + observable + plot -----------------------
    @testset "optimise_ipeps + observable + plot" begin
        function restriction_ipeps(A)
            A = local_min_norm(A, params)
            return A
        end
        # 1 LBFGS iter + chi-shift triggers observable() twice (writes log + plot)
        result = optimise_ipeps(A_raw, χ, χshift, params; restriction_ipeps)
        final_e = result[2]
        @test isfinite(final_e)

        # Both lattice plots should exist
        D_str = "D$(D)"
        for χ_val in (χ, χ + χshift)
            @test isfile(joinpath(folder, D_str, "observable", "lattice_χ$(χ_val).png"))
            @test isfile(joinpath(folder, D_str, "observable", "χ$(χ_val).log"))
        end
    end
end
