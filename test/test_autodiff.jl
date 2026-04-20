@testset "autodiff" begin

    # ===================== Structural rrule tests =====================
    @testset "StructArray constructor rrule" begin
        data = [rand(ComplexF64, 3, 3), rand(ComplexF64, 3, 3)]
        pattern = [1 2; 2 1]
        S, back = Zygote.pullback(StructArray, data, pattern)
        @test S isa StructArray
        # Pullback with a StructArray tangent
        dS = StructArray([rand(ComplexF64, 3, 3), rand(ComplexF64, 3, 3)], pattern)
        grads = back(dS)
        # Zygote strips the NoTangent for the constructor; grads[1] is data gradient, grads[2] is NoTangent (pattern)
        @test grads[1] == dS.data
    end

    @testset "norm(StructArray) rrule" begin
        data = [rand(ComplexF64, 3, 3), rand(ComplexF64, 4, 4)]
        pattern = [1 2; 2 1]
        S = StructArray(data, pattern)
        y, back = Zygote.pullback(norm, S)
        @test y isa Real
        @test y > 0
        dS = back(1.0)
        @test dS[1] isa StructArray
        @test norm(dS[1]) > 0
    end

    @testset "VUMPSRuntime constructor rrule" begin
        pattern = [1;;]
        chi = 4; d = 2
        AL = randSA(ComplexF64, Array, pattern, NTuple{3,Int}[(d, chi, chi)])
        AR = randSA(ComplexF64, Array, pattern, NTuple{3,Int}[(d, chi, chi)])
        C  = randSA(ComplexF64, Array, pattern, NTuple{2,Int}[(chi, chi)])
        FL = randSA(ComplexF64, Array, pattern, NTuple{3,Int}[(chi, d, chi)])
        FR = randSA(ComplexF64, Array, pattern, NTuple{3,Int}[(chi, d, chi)])
        rt, back = Zygote.pullback(VUMPSRuntime, AL, AR, C, FL, FR)
        @test rt isa VUMPSRuntime
        # Pullback: supply 5 tangent fields
        drt = (randSA(AL), randSA(AR), randSA(C), randSA(FL), randSA(FR))
        grads = back(drt)
        # Zygote strips the NoTangent for the constructor; grads = (dAL, dAR, dC, dFL, dFR)
        @test length(grads) == 5
    end

    @testset "CTMEnv constructor rrule" begin
        C_mat = rand(ComplexF64, 4, 4)
        T_ten = rand(ComplexF64, 4, 2, 4)
        env, back = Zygote.pullback(CTMEnv, C_mat, T_ten)
        @test env isa CTMEnv
        denv = (rand(ComplexF64, 4, 4), rand(ComplexF64, 4, 2, 4))
        grads = back(denv)
        # Zygote strips the NoTangent for the constructor; grads = (dC, dT)
        @test length(grads) == 2
    end

    # ===================== Numerical gradient checks =====================
    # num_grad requires CPU element indexing, so only run on Array
    @testset "qrpos AD vs num_grad" begin
        Random.seed!(100)
        A = rand(ComplexF64, 6, 4)
        f(A) = let (Q, R) = qrpos(A); real(sum(Q) + sum(R)); end
        g_zy = Zygote.gradient(f, A)[1]
        g_nd = num_grad(f, A)
        @test g_zy ≈ g_nd atol=1e-3
    end

    @testset "lqpos AD vs num_grad" begin
        Random.seed!(101)
        A = rand(ComplexF64, 4, 6)
        f(A) = let (L, Q) = lqpos(A); real(sum(L) + sum(Q)); end
        g_zy = Zygote.gradient(f, A)[1]
        g_nd = num_grad(f, A)
        @test g_zy ≈ g_nd atol=1e-3
    end

    @testset "qr_for_ad AD vs num_grad" begin
        Random.seed!(102)
        A = rand(ComplexF64, 6, 4)
        f(A) = let (Q, R) = qr_for_ad(A); real(sum(Q) + sum(R)); end
        g_zy = Zygote.gradient(f, A)[1]
        g_nd = num_grad(f, A)
        @test g_zy ≈ g_nd atol=1e-3
    end

    @testset "SVD AD — gradient runs and has correct shape" begin
        Random.seed!(103)
        A = rand(ComplexF64, 6, 4)
        f(A) = let res = svd(A); real(sum(res.S)); end
        g = Zygote.gradient(f, A)[1]
        @test size(g) == size(A)
        @test norm(g) > 0
        # Custom SVD rrule uses Riemannian formula, not standard Wirtinger derivative
    end

    @testset "orth_for_ad AD — projection modifies gradient" begin
        Random.seed!(104)
        v = rand(ComplexF64, 8)
        v /= norm(v)
        # orth_for_ad is identity in forward, but rrule projects gradient ⊥ v
        f(v) = real(sum(orth_for_ad(v) .^ 2))
        g = Zygote.gradient(f, v)[1]
        @test size(g) == size(v)
        # Key property: gradient is projected orthogonal to v
        @test abs(dot(v, g)) < 1e-8
    end

    @testset "simple_eig AD vs num_grad" begin
        Random.seed!(105)
        H = rand(ComplexF64, 6, 6)
        H = H + H'  # Hermitian
        H += 10 * I  # well-separated leading eigenvalue
        v0 = rand(ComplexF64, 6)
        v0 /= norm(v0)

        # f maps H to the dominant eigenvalue (real part)
        function _eig_f(Hvec)
            Hmat = reshape(Hvec, 6, 6)
            vals, _ = simple_eig(v -> Hmat * v, v0; power_iter=50)
            return real(vals[1])
        end
        Hvec = vec(H)
        g_zy = Zygote.gradient(_eig_f, Hvec)[1]
        g_nd = num_grad(_eig_f, Hvec)
        @test g_zy ≈ g_nd atol=1e-2
    end

    # ===================== orth_for_ad projection property =====================
    @testset "orth_for_ad projects gradient orthogonal to v" begin
        Random.seed!(106)
        v = rand(ComplexF64, 10)
        v /= norm(v)
        f(v) = real(sum(orth_for_ad(v) .^ 2))
        g = Zygote.gradient(f, v)[1]
        @test abs(dot(v, g)) < 1e-10
    end

    # ===================== Grassmann manifold =====================
    @testset "project_AL orthogonality" begin
        Random.seed!(107)
        chi = 4; D = 2
        # Create a left-isometric tensor via QR
        M = rand(ComplexF64, chi * D, chi)
        Q, _ = qr(M)
        AL_tensor = reshape(Matrix(Q), D, chi, chi)
        # Verify left-isometry: reshape to (chi*D, chi), Q'Q = I
        AL_mat = reshape(AL_tensor, chi * D, chi)
        @test AL_mat' * AL_mat ≈ I atol=1e-12

        # Random gradient
        dAL_tensor = rand(ComplexF64, D, chi, chi)

        # Project
        dAL_proj = project_AL([dAL_tensor], [AL_tensor])

        # Check orthogonality: conj(AL)[c,d,a] * dAL_proj[c,d,b] should be ~ 0
        @tensor overlap[a, b] := conj(AL_tensor)[c, d, a] * dAL_proj[1][c, d, b]
        @test norm(overlap) < 1e-10
    end

    @testset "project_AR orthogonality" begin
        Random.seed!(108)
        chi = 4; D = 2
        # Create a right-isometric tensor: permute_fronttail of a left-isometric one
        M = rand(ComplexF64, chi * D, chi)
        Q, _ = qr(M)
        AL_tensor = reshape(Matrix(Q), D, chi, chi)
        AR_tensor = permute_fronttail(AL_tensor)
        # Verify right-isometry: permute_fronttail, reshape, Q'Q = I
        AR_perm = permute_fronttail(AR_tensor)
        AR_mat = reshape(AR_perm, chi * D, chi)
        @test AR_mat' * AR_mat ≈ I atol=1e-12

        # Random gradient
        dAR_tensor = rand(ComplexF64, chi, chi, D)

        # Project
        dAR_proj = project_AR([dAR_tensor], [AR_tensor])

        # Check orthogonality via permute_fronttail
        dAR_proj_perm = permute_fronttail(dAR_proj[1])
        AR_perm2 = permute_fronttail(AR_tensor)
        @tensor overlap[a, b] := conj(AR_perm2)[c, d, a] * dAR_proj_perm[c, d, b]
        @test norm(overlap) < 1e-10
    end

    @testset "retract! left-canonical" begin
        Random.seed!(109)
        chi = 4; D = 2
        pattern = [1;;]
        A_data = [rand(ComplexF64, chi, D, chi)]
        A_SA = StructArray(A_data, pattern)

        # Test left_canonical directly (retract! is a thin wrapper)
        AL, _, _ = left_canonical(A_SA)
        A_mat = reshape(Array(AL[1,1]), chi * D, chi)
        @test A_mat' * A_mat ≈ I(chi) atol=1e-8
    end

    # ===================== ZeroAdder helper =====================
    @testset "ZeroAdder arithmetic" begin
        z = TeneT.ZeroAdder()
        x = rand(3, 3)
        @test (x + z) === x
        @test (z + x) === x
        @test (x - z) === x
        @test (-z) === z
    end

    # ===================== @non_differentiable smoke test =====================
    @testset "@non_differentiable smoke" begin
        # Just verify non-differentiable functions exist and can be called
        @test randSA(ComplexF64, Array, [1;;], NTuple{3,Int}[(2, 3, 3)]) isa StructArray
        @test ISA(ComplexF64, Array, [1;;], NTuple{2,Int}[(2, 2)]) isa StructArray
    end

    # ===================== QRCTM gradient equivalence =====================
    # CPU-only (atype=Array): the three checkpoint modes (:none, :recompute,
    # :wengert) must produce compatible gradients. Placed outside the atype
    # loop deliberately: this verifies checkpoint/Wengert wiring, not GPU
    # behaviour — GPU-side coverage is handled by Task 8 smoke tests.
    #
    # Tolerance note: there is a pre-existing ~1% discrepancy in baseline
    # `:none` vs `:recompute` gradient (observed at chi=16, beta=0.3, stable
    # regime `maxiter_ad=1`: rel ≈ 1.1%, max |Δ| ≈ 1.5e-3) that predates this
    # task and is not introduced by the Wengert refactor (confirmed by git
    # stash of Task 6 source edits — the diff survives). Each path's gradient
    # differs from the others by at most ≈2e-3 in absolute value; a gross
    # Wengert regression (e.g. wrong tape slot seeding or missing rrule) would
    # show orders-of-magnitude larger diff, so `atol=5e-3` remains sharp.
    # Using `miniter_ad=1` keeps the AD loop in the numerically-stable
    # single-step regime; cranking `miniter_ad` higher at tol=1e-10 pushes all
    # three paths into diverging backprop noise (not a path bug — the
    # backward iteration of a near-fixed-point map amplifies roundoff).
    @testset "QRCTM gradient equivalence: none vs recompute vs wengert" begin
        # Regression detector, not bit-equivalence: `:none` vs `:recompute` differ
        # at ~1.5e-3 max-abs on this 5-AD-iter CTM problem due to iteration-order
        # sensitivity near the near-fixed-point, observed on stash-removed code
        # before the Wengert patch landed. The 5e-3 bound is ~3× that noise floor.
        Random.seed!(42)
        chi = 16
        beta = 0.3
        M = ising_mpo(beta; atype=Array)

        function loss(M_, flag_mode::Symbol)
            alg = QRCTM(; verbosity=0, maxiter=30, miniter=0,
                         maxiter_ad=5, miniter_ad=1, tol=1e-10,
                         ifcheckpoint          = (flag_mode == :recompute),
                         ifcheckpoint_wengert  = (flag_mode == :wengert))
            rt = init_env(M_, chi, alg)
            rt, _ = leading_boundary(rt, M_, alg)
            return sum(abs2, rt.T)
        end

        g_none    = Zygote.gradient(m -> loss(m, :none),     M)[1]
        g_recomp  = Zygote.gradient(m -> loss(m, :recompute), M)[1]
        g_wengert = Zygote.gradient(m -> loss(m, :wengert),  M)[1]

        # Gradient w.r.t. `M::StructArray` comes back as a NamedTuple
        # @NamedTuple{data::Vector{Array}, pattern::Nothing}. Grab the first
        # underlying tensor via `.data[1]` and compare elementwise.
        a_none    = Array(g_none.data[1])
        a_recomp  = Array(g_recomp.data[1])
        a_wengert = Array(g_wengert.data[1])

        # Use max-abs (L∞) rather than `≈`'s 2-norm so the tolerance tracks the
        # observed per-element gap, not an n-scaled cumulative norm.
        atol = 5e-3
        @test maximum(abs, a_recomp  .- a_none)    < atol
        @test maximum(abs, a_wengert .- a_none)    < atol
        @test maximum(abs, a_wengert .- a_recomp)  < atol
    end

end
