using Test
using TeneT
using LinearAlgebra
using Random
using OptimKit
using Zygote

using TeneT: StructArray, VUMPSRuntime, init_env, build_A, GradientOptimize

@testset "Cubic dimer model" begin
    @testset "local PEPO constraint" begin
        model = CubicDimer()
        T = dimer_pepo(model; atype=Array)

        @test size(T) == (2, 2, 2, 2, 2, 2)
        @test sum(T) == 6

        for I in CartesianIndices(T)
            bits = Tuple(I) .- 1
            @test T[I] == (sum(bits) == 1 ? 1.0 : 0.0)
        end
    end

    @testset "PEPO transfer application" begin
        model = CubicDimer()
        A = ones(Float64, 2, 2, 2, 2, 2)
        B = apply_transfer(model, A)

        @test size(B) == (4, 4, 4, 4, 2)
        @test B[1, 1, 1, 1, 1] == 1
        @test B[1, 1, 1, 1, 2] == 1
        @test B[3, 1, 1, 1, 1] == 1
        @test B[3, 1, 1, 1, 2] == 0
        @test B[3, 3, 1, 1, 1] == 0

        A_sa = StructArray([A], [1;;])
        M = mixed_transfer_tensor(model, A_sa)
        @test M isa StructArray
        @test M.pattern == [1;;]
        @test M[1] isa Tuple
        @test size(M[1][1]) == (4, 4, 4, 4, 2)
        @test M[1][2] == conj(A)
    end

    @testset "transfer PEPO framework API" begin
        model = CubicDimer()
        W = TeneT.transfer_pepo(model; atype=Array)
        @test W == dimer_pepo(model; atype=Array)

        A = ones(Float64, 2, 2, 2, 2, 2)
        A_sa = StructArray([A], [1;;])
        M_generic = TeneT.transfer_layer(model, A_sa)
        M_legacy = mixed_transfer_tensor(model, A_sa)
        @test M_generic isa StructArray
        @test M_generic.pattern == M_legacy.pattern
        @test M_generic[1][1] == M_legacy[1][1]
        @test M_generic[1][2] == M_legacy[1][2]
        @test hasmethod(TeneT.optimise_transfer_pepo, Tuple{Any, Vector{Int}, GradientOptimize})
    end

    @testset "mixed VUMPS runtime initializes" begin
        Random.seed!(1234)
        model = CubicDimer()
        A = StructArray([ones(Float64, 2, 2, 2, 2, 2)], [1;;])
        M = mixed_transfer_tensor(model, A)
        alg = VUMPS{General}(;
            ifupdown=false,
            forloop_iter=4,
            maxiter=1,
            maxiter_ad=1,
            power_iter=2,
            power_iter_ad=1,
            miniter=1,
            miniter_ad=1,
            verbosity=0,
        )

        rt = init_env(M, 2, alg)
        @test rt isa VUMPSRuntime
        @test size(rt.AL[1]) == (2, 4, 2, 2)
        @test size(rt.FL[1]) == (2, 4, 2, 2)
        @test size(rt.FR[1]) == (2, 4, 2, 2)
    end

    @testset "entropy observable smoke" begin
        Random.seed!(4321)
        tmp = mktempdir()
        model = CubicDimer()
        alg = VUMPS{General}(;
            ifupdown=false,
            maxiter=1,
            maxiter_ad=1,
            power_iter=2,
            power_iter_ad=1,
            miniter=1,
            miniter_ad=1,
            verbosity=0,
        )
        params = GradientOptimize(;
            model,
            pattern=[1;;],
            boundary_alg=alg,
            folder=tmp,
            verbosity=0,
            ifload_env=false,
            ifsave_env=false,
            ifsave_lbfgs=false,
            ifload_lbfgs=false,
            ifplot=false,
            forloop_iter=4,
            optimizer=LBFGS(1; maxiter=0, verbosity=0),
        )
        baseline_forloop_iter = alg.forloop_iter
        Araw = ones(Float64, 1, 1, 1, 1, 2, 1)
        A = build_A(Araw, params)
        alg_rt = deepcopy(alg)
        alg_rt.forloop_iter = 1
        rt_norm = init_env(A, 1, alg_rt)
        rt_mix = init_env(mixed_transfer_tensor(model, A), 1, alg_rt)

        obs = residual_entropy(Araw, rt_norm, rt_mix, params)
        @test isfinite(obs.entropy)
        @test obs.objective == -obs.entropy
        @test isfinite(obs.log_norm)
        @test isfinite(obs.log_transfer)
        @test isfinite(obs.err_norm)
        @test isfinite(obs.err_transfer)

        xi = cubic_dimer_correlation_length(obs.rt_norm)
        @test isreal(xi)

        written = cubic_dimer_observable(Araw, 1, params)
        @test isfinite(written.entropy)
        @test isfile(joinpath(tmp, "D1", "observable", "χ1.log"))
        @test hasmethod(optimise_cubic_dimer, Tuple{Any, Vector{Int}, GradientOptimize})

        Aopt, entropy, grad, fgnum, history = optimise_cubic_dimer(Araw, [1], params)
        @test size(Aopt) == size(Araw)
        @test isfinite(entropy)
        @test fgnum >= 1
        @test size(history, 2) == 2
        @test alg.forloop_iter == baseline_forloop_iter
    end

    @testset "transfer PEPO environment refresh is fixed-env AD" begin
        Random.seed!(5678)
        model = CubicDimer()
        alg = VUMPS{General}(;
            ifupdown=false,
            maxiter=1,
            maxiter_ad=1,
            power_iter=2,
            power_iter_ad=1,
            miniter=1,
            miniter_ad=1,
            verbosity=0,
            forloop_iter=1,
        )
        params = GradientOptimize(;
            model,
            pattern=[1;;],
            boundary_alg=alg,
            folder=mktempdir(),
            verbosity=0,
            ifload_env=false,
            ifsave_env=false,
            ifsave_lbfgs=false,
            ifload_lbfgs=false,
            ifplot=false,
            optimizer=LBFGS(1; maxiter=0, verbosity=0),
        )
        Araw = ones(Float64, 1, 1, 1, 1, 2, 1)
        A = build_A(Araw, params)
        rt_norm = init_env(A, 1, alg)
        rt_transfer = init_env(TeneT.transfer_layer(model, A), 1, alg)

        only_refresh(x) = begin
            A_built = TeneT._build_transfer_pepo_A(x, params; restriction_ipeps=TeneT._restriction_ipeps)
            _, _, err_norm, err_transfer =
                TeneT._transfer_pepo_refresh_envs(rt_norm, rt_transfer, A_built, params)
            real(err_norm + err_transfer)
        end

        grad = Zygote.gradient(only_refresh, Araw)[1]
        @test grad === nothing || norm(grad) == 0
    end

    @testset "transfer PEPO C4v density smoke" begin
        Random.seed!(2468)
        model = CubicDimer()
        alg = VUMPS{C4v}(;
            maxiter=1,
            maxiter_ad=1,
            power_iter=2,
            power_iter_ad=1,
            miniter=1,
            miniter_ad=1,
            verbosity=0,
            forloop_iter=1,
        )
        params = GradientOptimize(;
            model,
            pattern=[1;;],
            boundary_alg=alg,
            folder=mktempdir(),
            verbosity=0,
            ifload_env=false,
            ifsave_env=false,
            ifsave_lbfgs=false,
            ifload_lbfgs=false,
            ifplot=false,
            optimizer=LBFGS(1; maxiter=0, verbosity=0),
        )
        Araw = ones(Float64, 1, 1, 1, 1, 2, 1)
        A = TeneT._build_transfer_pepo_A(Araw, params; restriction_ipeps=TeneT._restriction_ipeps)
        rt_norm = init_env(A, 1, alg)
        rt_transfer = init_env(TeneT.transfer_layer(model, A), 1, alg)

        obs = TeneT.transfer_pepo_density(Araw, rt_norm, rt_transfer, params)
        @test obs.rt_norm isa TeneT.C4vVUMPSEnv
        @test obs.rt_transfer isa TeneT.C4vVUMPSEnv
        @test isfinite(obs.log_density)
        @test obs.objective == -obs.log_density

        written = cubic_dimer_observable(Araw, 1, params)
        @test written.rt_norm isa TeneT.C4vVUMPSEnv
        @test !isnan(written.xi)
        obs_log = joinpath(params.folder, "D1", "observable", "χ1.log")
        @test isfile(obs_log)
        @test !occursin("correlation_length:\nNaN", read(obs_log, String))
    end

    @testset "optimizer stops on objective stall" begin
        model = CubicDimer()
        alg = VUMPS{General}(; ifupdown=false, verbosity=0)
        params = GradientOptimize(;
            model,
            pattern=[1;;],
            boundary_alg=alg,
            folder=mktempdir(),
            verbosity=0,
            ifplot=false,
        )
        fδ = [NaN, Inf]

        g1 = [1.0, -2.0]
        δ1 = TeneT._cubic_dimer_update_stop!(g1, -0.4, fδ, params, 4)
        @test isinf(δ1)
        @test g1 == [1.0, -2.0]
        @test params.last_stop_reason == :running
        @test params.last_stop_χ == 4

        g2 = [3.0, -4.0]
        δ2 = TeneT._cubic_dimer_update_stop!(g2, -0.4, fδ, params, 4)
        @test δ2 == 0.0
        @test g2 == [0.0, 0.0]
        @test params.last_stop_reason == :objective_stall
        @test params.last_stop_χ == 4
    end
end
