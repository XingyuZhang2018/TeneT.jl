using OptimKit: LBFGS, LBFGSInverseHessian

@testset "OptimKit_patch.jl" begin

    # ---- 1. LBFGSState construction and field access ----
    @testset "LBFGSState construction" begin
        x = rand(5)
        f = 1.0
        g = rand(5)
        H = LBFGSInverseHessian(5, Vector{Float64}[], Vector{Float64}[], Float64[])
        numfg = 1
        numiter = 0
        fhist = [f]
        nghist = [norm(g)]
        t0 = time()

        state = LBFGSState(x, f, g, H, numfg, numiter, fhist, nghist, t0)

        @test state.x == x
        @test state.f == f
        @test state.g == g
        @test state.H isa LBFGSInverseHessian
        @test state.numfg == 1
        @test state.numiter == 0
        @test state.fhistory == [f]
        @test state.normgradhistory == [norm(g)]
        @test state.t₀ == t0
    end

    # ---- 2. LBFGSInverseHessian GPU roundtrip ----
    @testset "LBFGSInverseHessian GPU roundtrip — $atype" for atype in ATYPES
        atype === Array && continue  # skip CPU-only

        S_orig = [rand(5), rand(5)]
        Y_orig = [rand(5), rand(5)]
        rho_orig = [0.5, 0.3]
        H = LBFGSInverseHessian(5, copy(S_orig), copy(Y_orig), copy(rho_orig))

        # Convert to GPU then back to CPU
        H_gpu = atype(H)
        H_cpu = Array(H_gpu)

        for i in 1:2
            @test H_cpu.S[i] ≈ S_orig[i]
            @test H_cpu.Y[i] ≈ Y_orig[i]
        end
    end

    # ---- 3. save/load LBFGS state roundtrip ----
    @testset "save/load LBFGS state roundtrip" begin
        mktempdir() do dir
            x = rand(5)
            f = 2.5
            g = rand(5)
            H = LBFGSInverseHessian(5, Vector{Float64}[], Vector{Float64}[], Float64[])
            state = LBFGSState(x, f, g, H, 3, 7, [2.5], [norm(g)], time())

            filepath = joinpath(dir, "test_state.jld2")
            alg = LBFGS(5; maxiter=10, verbosity=0, gradtol=1e-8)

            ok = save_lbfgs_state(alg, state, filepath)
            @test ok == true

            loaded = load_lbfgs_state(alg, filepath)
            @test loaded !== nothing
            @test loaded.x ≈ x
            @test loaded.f == f
            @test loaded.numfg == 3
            @test loaded.numiter == 7
        end
    end

    # ---- 4. optimize_reload — quadratic function ----
    @testset "optimize_reload — quadratic" begin
        fg(x) = (sum(abs2, x), 2x)
        x0 = [1.0, 2.0, 3.0]
        alg = LBFGS(5; maxiter=20, verbosity=0, gradtol=1e-8)

        x_opt, f_opt, g_opt, numfg, history = optimize_reload(fg, x0, alg)

        @test f_opt < 1e-10
        @test norm(x_opt) < 1e-5
    end

    # ---- 5. optimize_reload — resume from state ----
    @testset "optimize_reload — resume from state" begin
        fg(x) = (sum(abs2, x), 2x)
        x0 = [1.0, 2.0, 3.0]

        mktempdir() do dir
            filepath = joinpath(dir, "resume_state.jld2")
            alg1 = LBFGS(5; maxiter=5, verbosity=0, gradtol=1e-14)

            # Run 5 iterations and save state
            x1, f1, _, _, _ = optimize_reload(fg, x0, alg1;
                                              save_state_to=filepath, save_every=1)

            # Resume from saved state with more iterations
            alg2 = LBFGS(5; maxiter=20, verbosity=0, gradtol=1e-14)
            x2, f2, _, _, _ = optimize_reload(fg, x0, alg2;
                                              resume_from=filepath)

            @test f2 <= f1
        end
    end
end
