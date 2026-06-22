@testset "checkpoint unified API" begin

    # ---- Type hierarchy ----
    @testset "CheckpointMethod type hierarchy" begin
        @test TeneT.Plain()            isa TeneT.CheckpointMethod
        @test TeneT.Recompute()        isa TeneT.CheckpointMethod
        @test TeneT.OffloadRecompute() isa TeneT.CheckpointMethod
        @test TeneT.Offload()          isa TeneT.CheckpointMethod
        # Singleton uniqueness
        @test TeneT.Plain()            === TeneT.Plain()
        @test TeneT.Recompute()        === TeneT.Recompute()
        @test TeneT.OffloadRecompute() === TeneT.OffloadRecompute()
        @test TeneT.Offload()          === TeneT.Offload()
    end

    # ---- Symbol normalization ----
    @testset "_ckpt_method Symbol normalization" begin
        @test TeneT._ckpt_method(:plain)             === TeneT.Plain()
        @test TeneT._ckpt_method(:none)              === TeneT.Plain()
        @test TeneT._ckpt_method(:recompute)         === TeneT.Recompute()
        @test TeneT._ckpt_method(:offload_recompute) === TeneT.OffloadRecompute()
        @test TeneT._ckpt_method(:offload)           === TeneT.Offload()
        # Identity on instances
        @test TeneT._ckpt_method(TeneT.Plain())            === TeneT.Plain()
        @test TeneT._ckpt_method(TeneT.Recompute())        === TeneT.Recompute()
        @test TeneT._ckpt_method(TeneT.OffloadRecompute()) === TeneT.OffloadRecompute()
        @test TeneT._ckpt_method(TeneT.Offload())          === TeneT.Offload()
        # Unknown symbol throws
        @test_throws ArgumentError TeneT._ckpt_method(:foobar)
    end

    # ---- Base.convert integration ----
    @testset "Base.convert(CheckpointMethod, Symbol)" begin
        @test convert(TeneT.CheckpointMethod, :plain)             === TeneT.Plain()
        @test convert(TeneT.CheckpointMethod, :recompute)         === TeneT.Recompute()
        @test convert(TeneT.CheckpointMethod, :offload_recompute) === TeneT.OffloadRecompute()
        @test convert(TeneT.CheckpointMethod, :offload)           === TeneT.Offload()
    end

    # ---- checkpoint(Plain(), f, args...) — identity, no adjoint override ----
    @testset "checkpoint(Plain(), f, args...)" begin
        f(x) = 2 * x .+ 1
        x = rand(5)
        @test checkpoint(TeneT.Plain(), f, x) == f(x)

        # kwargs passthrough
        g(x; scale=1.0) = scale * x
        y = rand(3)
        @test checkpoint(TeneT.Plain(), g, y; scale=2.0) == g(y; scale=2.0)
    end

    # ---- checkpoint(Recompute(), f, args...) — identity + gradient ----
    @testset "checkpoint(Recompute(), f, args...)" begin
        f(x) = sum(x .^ 2)
        x = rand(4)
        @test checkpoint(TeneT.Recompute(), f, x) == f(x)

        # Gradient must match direct Zygote gradient of f(x)
        g_direct = Zygote.gradient(f, x)[1]
        g_ckpt   = Zygote.gradient(x -> checkpoint(TeneT.Recompute(), f, x), x)[1]
        @test g_direct ≈ g_ckpt
    end

    # ---- checkpoint(OffloadRecompute(), f, args...) — identity + gradient ----
    @testset "checkpoint(OffloadRecompute(), f, args...)" begin
        f(x) = sum(x .^ 2)
        x = rand(4)
        @test checkpoint(TeneT.OffloadRecompute(), f, x) == f(x)

        # Gradient must match direct Zygote gradient (CPU: OffloadRecompute
        # is Recompute + host-copy roundtrip, should be numerically identical)
        g_direct = Zygote.gradient(f, x)[1]
        g_ckpt   = Zygote.gradient(x -> checkpoint(TeneT.OffloadRecompute(), f, x), x)[1]
        @test g_direct ≈ g_ckpt
    end

    # ---- checkpoint(Offload(), f, args...) — identity + gradient ----
    @testset "checkpoint(Offload(), f, args...)" begin
        f(x) = sum(x .^ 2)
        x = rand(4)
        @test checkpoint(TeneT.Offload(), f, x) == f(x)

        # Gradient must match direct Zygote gradient (CPU: Offload walker
        # short-circuits since no GPU leaves are found; pb runs unchanged.)
        g_direct = Zygote.gradient(f, x)[1]
        g_ckpt   = Zygote.gradient(x -> checkpoint(TeneT.Offload(), f, x), x)[1]
        @test g_direct ≈ g_ckpt
    end

    # ---- Symbol entry dispatches to singleton ----
    @testset "checkpoint(::Symbol, f, args...)" begin
        f(x) = 2 * x .+ 1
        x = rand(5)
        @test checkpoint(:plain,             f, x) == f(x)
        @test checkpoint(:recompute,         f, x) == f(x)
        @test checkpoint(:offload_recompute, f, x) == f(x)
        @test checkpoint(:offload,           f, x) == f(x)

        # Gradient via Symbol entry
        g(x) = sum(x .^ 2)
        y = rand(3)
        g_direct = Zygote.gradient(g, y)[1]
        @test Zygote.gradient(y -> checkpoint(:recompute,         g, y), y)[1] ≈ g_direct
        @test Zygote.gradient(y -> checkpoint(:offload_recompute, g, y), y)[1] ≈ g_direct
        @test Zygote.gradient(y -> checkpoint(:offload,           g, y), y)[1] ≈ g_direct
    end

    # ---- _assert_inner_method: Offload/OffloadRecompute rejected, others OK ----
    @testset "_assert_inner_method" begin
        @test TeneT._assert_inner_method(TeneT.Plain())     === nothing
        @test TeneT._assert_inner_method(TeneT.Recompute()) === nothing
        @test_throws ArgumentError TeneT._assert_inner_method(TeneT.Offload())
        @test_throws ArgumentError TeneT._assert_inner_method(TeneT.OffloadRecompute())
    end

    # ---- Integration: VUMPS struct new fields ----
    @testset "VUMPS struct new checkpoint fields" begin
        v = VUMPS{General}()
        @test v.segment_checkpoint === TeneT.Plain()
        @test v.inner_checkpoint   === TeneT.Plain()
        @test v.eig_checkpoint     === TeneT.Plain()
        @test v.step_checkpoint    === TeneT.Plain()

        # Symbol coercion through @kwdef constructor
        v2 = VUMPS{General}(; step_checkpoint = :offload)
        @test v2.step_checkpoint === TeneT.Offload()

        v3 = VUMPS{General}(; segment_checkpoint = :plain,
                               inner_checkpoint   = :recompute,
                               eig_checkpoint     = :offload,
                               step_checkpoint    = :offload)
        @test v3.segment_checkpoint === TeneT.Plain()
        @test v3.inner_checkpoint   === TeneT.Recompute()
        @test v3.eig_checkpoint     === TeneT.Offload()
        @test v3.step_checkpoint    === TeneT.Offload()

        # Direct singleton also works
        v4 = VUMPS{General}(; step_checkpoint = TeneT.Recompute())
        @test v4.step_checkpoint === TeneT.Recompute()
    end

    # ---- ifcheckpoint master switch (R2 production preset) ----
    @testset "ifcheckpoint=true applies R2 winner preset" begin
        # default: ifcheckpoint=false → all Plain
        v0 = VUMPS{General}()
        @test v0.ifcheckpoint == false
        @test v0.step_checkpoint    === TeneT.Plain()
        @test v0.subop_checkpoint   === TeneT.Plain()
        @test v0.segment_checkpoint === TeneT.Plain()
        @test v0.eig_checkpoint     === TeneT.Plain()
        @test v0.inner_checkpoint   === TeneT.Plain()

        # ifcheckpoint=true → R2 winner preset
        v1 = VUMPS{General}(; ifcheckpoint = true)
        @test v1.ifcheckpoint == true
        @test v1.step_checkpoint    === TeneT.OffloadRecompute()
        @test v1.subop_checkpoint   === TeneT.OffloadRecompute()
        @test v1.segment_checkpoint === TeneT.OffloadRecompute()
        @test v1.eig_checkpoint     === TeneT.Recompute()
        @test v1.inner_checkpoint   === TeneT.Plain()  # always Plain

        # Explicit override wins over preset
        v2 = VUMPS{General}(; ifcheckpoint = true,
                              step_checkpoint = TeneT.Plain(),
                              eig_checkpoint  = TeneT.Offload())
        @test v2.step_checkpoint    === TeneT.Plain()             # overridden
        @test v2.subop_checkpoint   === TeneT.OffloadRecompute()  # from preset
        @test v2.eig_checkpoint     === TeneT.Offload()            # overridden
    end

    @testset "initialize_env loads QRCTMRG env without ifparallelupdown" begin
        D, d, chi = 2, 2, 4
        folder = mktempdir()
        model = Heisenberg(lattice=Honeycomb(:c3v), S=0.5, Jx=1.0, Jy=1.0, Jz=1.0,
                           ifrotate=true, couplingtype=:uniform, bondratio=1.0)
        boundary_alg = QRCTMRG{C3v}(; verbosity=0, maxiter=1, maxiter_ad=0,
                                    miniter=0, miniter_ad=0)
        params = GradientOptimize(; model, pattern=[1;;], boundary_alg,
                                   optimizer=nothing, folder, verbosity=0,
                                   ifload_env=true, ifsave_env=false,
                                   ifsave_lbfgs=false, ifload_lbfgs=false)
        A = rand(Float64, D, D, D, d, 1)
        envdir = joinpath(folder, "D$(D)", "environment")
        mkpath(envdir)
        saved = CTMEnv(fill(2.0, chi, chi), fill(3.0, chi, D, D, chi))
        save_rt(envdir, saved; file="χ$(chi).jld2")

        loaded = initialize_env(A, D, chi, params)

        @test loaded.C == saved.C
        @test loaded.T == saved.T
    end

    # ---- [1;;] iPEPS gradient regression ----
    # Earlier, leftenv/rightenv/ACenv had `eig_checkpoint isa Plain` branches that
    # built `f(x) = checkpoint(inner_checkpoint, FLmap, ..., ALu[i, :], ...)` with
    # the slice INSIDE the closure. For pattern=[1;;] (1-element data vector),
    # iterating this closure inside simple_eig produced a wrong gradient (~12%
    # rel err that exploded with more power_iter). Routing all eig calls through
    # `_simple_eig_*` wrappers — which slice once and capture as args — fixes it.
    # This regression catches a relapse via a single directional finite-diff.
    @testset "Heisenberg [1;;] energy gradient — 1-element pattern AD" begin
        Random.seed!(42)
        D, chi = 2, 4
        model = Heisenberg(lattice=Square(), S=0.5, Jx=1.0, Jy=1.0, Jz=1.0,
                           ifrotate=true, couplingtype=:uniform, bondratio=1.0)
        boundary_alg = VUMPS{General}(; ifupdown=true, ifdownfromup=false,
                                       ifsimple_eig=true, ifparallel=false,
                                       forloop_iter=1, maxiter=20, miniter=0,
                                       maxiter_ad=4, miniter_ad=4,
                                       power_iter=1, power_iter_ad=5,
                                       tol=1e-10, verbosity=0)
        params = GradientOptimize(; model, pattern=[1;;], boundary_alg,
                                   optimizer=nothing, forloop_iter=1,
                                   verbosity=0,
                                   folder=mktempdir(),
                                   ifSU=false, SUτ=0,
                                   ifprecondition=false, iter_precond=0,
                                   reuse_env=true, ifsave_env=false,
                                   ifload_env=false, ifsave_lbfgs=false,
                                   ifload_lbfgs=false)
        A = init_ipeps(; atype=Array, etype=ComplexF64, No=0, D, χ=chi, params)

        restriction_ipeps(A) = C4v_restriction(A) / norm(C4v_restriction(A))
        rt = TeneT.initialize_env(A, D, chi, params; restriction_ipeps)
        rt′ = deepcopy(rt)
        fδEierr = [1.0, 1.0, 0.0, 0.0]
        fenergy(A) = (TeneT._G_cache[] = nothing;
                      real(TeneT.energy(restriction_ipeps(A), rt, rt′, fδEierr, params)))

        g_zyg, = Zygote.gradient(fenergy, A)

        # Directional finite difference along a fixed random direction
        Random.seed!(7)
        v = randn(ComplexF64, size(A))
        v ./= norm(v)
        δ = 1e-4
        df = (fenergy(A .+ δ .* v) - fenergy(A .- δ .* v)) / (2δ)
        # For real(energy) with complex A, Zygote convention: dE/dδ = real(dot(g_zyg, v))
        df_ad = real(dot(g_zyg, v))
        @test isapprox(df, df_ad; rtol=1e-3)
    end

    # ---- simple_eig segment_checkpoint kwarg propagates ----
    @testset "simple_eig segment_checkpoint kwarg" begin
        # Hermitian matrix with well-separated dominant eigenvalue
        Random.seed!(321)
        H = rand(ComplexF64, 8, 8); H = H + H' + 10I
        v0 = rand(ComplexF64, 8); v0 /= norm(v0)
        f(v) = H * v

        # Default (Plain) — no per-segment checkpoint
        vals_ref, vecs_ref = TeneT.simple_eig(f, v0; power_iter=20)

        # Explicit Plain: must equal default
        vals_p, vecs_p = TeneT.simple_eig(f, v0; power_iter=20,
                                          segment_checkpoint=TeneT.Plain())
        @test vals_p[1] == vals_ref[1]

        # Explicit Recompute: same eigenvalue up to floating-point ordering
        vals_r, vecs_r = TeneT.simple_eig(f, v0; power_iter=20,
                                          segment_checkpoint=TeneT.Recompute())
        @test abs(vals_r[1]) ≈ abs(vals_ref[1]) atol=1e-10
    end

end
