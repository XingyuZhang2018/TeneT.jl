@testset "checkpoint unified API" begin

    # ---- Type hierarchy ----
    @testset "CheckpointMethod type hierarchy" begin
        @test TeneT.Plain() isa TeneT.CheckpointMethod
        @test TeneT.Recompute() isa TeneT.CheckpointMethod
        @test TeneT.Offload() isa TeneT.CheckpointMethod
        # Singleton uniqueness
        @test TeneT.Plain() === TeneT.Plain()
        @test TeneT.Recompute() === TeneT.Recompute()
        @test TeneT.Offload() === TeneT.Offload()
    end

    # ---- Symbol normalization ----
    @testset "_ckpt_method Symbol normalization" begin
        @test TeneT._ckpt_method(:plain)     === TeneT.Plain()
        @test TeneT._ckpt_method(:none)      === TeneT.Plain()
        @test TeneT._ckpt_method(:recompute) === TeneT.Recompute()
        @test TeneT._ckpt_method(:offload)   === TeneT.Offload()
        # Identity on instances
        @test TeneT._ckpt_method(TeneT.Plain())     === TeneT.Plain()
        @test TeneT._ckpt_method(TeneT.Recompute()) === TeneT.Recompute()
        @test TeneT._ckpt_method(TeneT.Offload())   === TeneT.Offload()
        # Unknown symbol throws
        @test_throws ArgumentError TeneT._ckpt_method(:foobar)
    end

    # ---- Base.convert integration ----
    @testset "Base.convert(CheckpointMethod, Symbol)" begin
        @test convert(TeneT.CheckpointMethod, :plain)     === TeneT.Plain()
        @test convert(TeneT.CheckpointMethod, :recompute) === TeneT.Recompute()
        @test convert(TeneT.CheckpointMethod, :offload)   === TeneT.Offload()
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

    # ---- checkpoint(Offload(), f, args...) — identity + gradient ----
    @testset "checkpoint(Offload(), f, args...)" begin
        f(x) = sum(x .^ 2)
        x = rand(4)
        @test checkpoint(TeneT.Offload(), f, x) == f(x)

        # Gradient must match direct Zygote gradient (CPU: Offload is
        # Recompute + CPU copy roundtrip, should be numerically identical)
        g_direct = Zygote.gradient(f, x)[1]
        g_ckpt   = Zygote.gradient(x -> checkpoint(TeneT.Offload(), f, x), x)[1]
        @test g_direct ≈ g_ckpt
    end

    # ---- Symbol entry dispatches to singleton ----
    @testset "checkpoint(::Symbol, f, args...)" begin
        f(x) = 2 * x .+ 1
        x = rand(5)
        @test checkpoint(:plain,     f, x) == f(x)
        @test checkpoint(:recompute, f, x) == f(x)
        @test checkpoint(:offload,   f, x) == f(x)

        # Gradient via Symbol entry
        g(x) = sum(x .^ 2)
        y = rand(3)
        g_direct = Zygote.gradient(g, y)[1]
        @test Zygote.gradient(y -> checkpoint(:recompute, g, y), y)[1] ≈ g_direct
        @test Zygote.gradient(y -> checkpoint(:offload,   g, y), y)[1] ≈ g_direct
    end

    # ---- _assert_inner_method: Offload rejected, others OK ----
    @testset "_assert_inner_method" begin
        @test TeneT._assert_inner_method(TeneT.Plain())     === nothing
        @test TeneT._assert_inner_method(TeneT.Recompute()) === nothing
        @test_throws ArgumentError TeneT._assert_inner_method(TeneT.Offload())
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
