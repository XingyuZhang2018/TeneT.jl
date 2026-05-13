@testset "iPEPS optimize modules" begin

    # ================================================================
    # 1. _lattice_map -- Square identity
    # ================================================================
    @testset "_lattice_map Square identity" begin
        pattern = [1 2; 2 1]
        D, d = 2, 2
        tensors = [randn(D, D, D, D, d) for _ in 1:2]
        A = StructArray(tensors, pattern)
        A_out = _lattice_map(A, Square(), pattern)
        for i in 1:length(tensors)
            @test A_out[i] ≈ A[i]
        end
    end

    # ================================================================
    # 2. _lattice_map -- Honeycomb brickwall
    # ================================================================
    @testset "_lattice_map Honeycomb brickwall" begin
        pattern = [1 2; 3 4]
        D, d = 2, 2
        tensors = [randn(D, 1, D, D, d) for _ in 1:4]
        A = StructArray(tensors, pattern)
        A_out = _lattice_map(A, Honeycomb{:brickwall_h}(), pattern)
        # _lattice_map iterates over data indices 1:length(unique(pattern))
        # and uses CartesianIndices(pattern) with linear indexing to determine parity.
        # For pattern [1 2; 3 4] (column-major):
        #   cartindex[1] = (1,1), sum=2 (even) → data[1] unchanged
        #   cartindex[2] = (2,1), sum=3 (odd)  → data[2] permuted
        #   cartindex[3] = (1,2), sum=3 (odd)  → data[3] permuted
        #   cartindex[4] = (2,2), sum=4 (even) → data[4] unchanged
        # A[i] maps through pattern: A[2]=data[pattern[2]]=data[3], A[3]=data[pattern[3]]=data[2]
        @test Array(A_out.data[1]) ≈ Array(A.data[1])
        @test Array(A_out.data[2]) ≈ permutedims(Array(A.data[3]), (3, 4, 1, 2, 5))
        @test Array(A_out.data[3]) ≈ permutedims(Array(A.data[2]), (3, 4, 1, 2, 5))
        @test Array(A_out.data[4]) ≈ Array(A.data[4])
    end

    # ================================================================
    # 3. C4v_restriction -- 5-leg
    # ================================================================
    @testset "C4v_restriction 5-leg" begin
        D, d = 3, 2
        A = randn(D, D, D, D, d)
        A_sym = C4v_restriction(A)
        # applying twice should give 16x (4 operations each doubling)
        A_sym2 = C4v_restriction(A_sym)
        @test A_sym2 ≈ 16 * A_sym
        # symmetry check: up-down reflection with conj
        @test A_sym ≈ permutedims(conj(A_sym), (1, 4, 3, 2, 5))
    end

    # ================================================================
    # 4. C4v_restriction -- 6-leg
    # ================================================================
    @testset "C4v_restriction 6-leg" begin
        D, d, N = 3, 2, 1
        A = randn(D, D, D, D, d, N)
        A_sym = C4v_restriction(A)
        A_sym2 = C4v_restriction(A_sym)
        @test A_sym2 ≈ 16 * A_sym
        # symmetry check
        @test A_sym ≈ permutedims(conj(A_sym), (1, 4, 3, 2, 5, 6))
    end

    # ================================================================
    # 5. _restriction_ipeps -- identity
    # ================================================================
    @testset "_restriction_ipeps identity" begin
        D, d = 3, 2
        A = randn(D, D, D, D, d)
        @test _restriction_ipeps(A) === A
    end

    # ================================================================
    # 6. pepsgeneral -- 5-leg reconstruct via ARstoA
    # ================================================================
    @testset "pepsgeneral 5-leg roundtrip" begin
        D, d = 3, 2
        A = randn(D, D, D, D, d)
        Ac, Rs = pepsgeneral(A)
        @test size(Ac) == size(A)
        @test length(Rs) == 4
        @test all(size(R) == (D, D) for R in Rs)
        # Verify Rs are upper triangular
        for R in Rs
            @test norm(tril(R, -1)) < 1e-6
        end
        # Verify reconstruction has same shape
        A_recon = ARstoA(Ac, Rs)
        @test size(A_recon) == size(A)
    end

    # ================================================================
    # 7. pepsgeneral -- 6-leg reconstruct via ARstoA1
    # ================================================================
    @testset "pepsgeneral 6-leg roundtrip" begin
        D, d, N = 3, 2, 1
        A = randn(D, D, D, D, d, N)
        Ac, Rs = pepsgeneral(A)
        @test size(Ac) == size(A)
        @test length(Rs) == 4
        @test all(size(R) == (D, D) for R in Rs)
        # Verify Rs are upper triangular
        for R in Rs
            @test norm(tril(R, -1)) < 1e-6
        end
        # Verify reconstruction has same shape
        A_recon = ARstoA1(Ac, Rs)
        @test size(A_recon) == size(A)
    end

    # ================================================================
    # 8. central_canonical1 and central_canonical2 shape preservation
    # ================================================================
    @testset "central_canonical shape" begin
        D, d = 3, 2
        # central_canonical1/2 use ARstoA1/ARstoA2 which require 6-leg tensors
        N = 1
        A6 = randn(D, D, D, D, d, N)
        A6_cc1 = central_canonical1(A6)
        @test size(A6_cc1) == size(A6)

        A6_cc2 = central_canonical2(A6)
        @test size(A6_cc2) == size(A6)
    end

    # ================================================================
    # 9. to_mcf_ipeps shape
    # ================================================================
    @testset "to_mcf_ipeps shape" begin
        D, d = 3, 2
        A = randn(D, D, D, D, d)
        A_mcf = to_mcf_ipeps(A)
        @test size(A_mcf) == (D, D, D, D, d)
    end

    # ================================================================
    # 10. local_gauge_contraction shape
    # ================================================================
    @testset "local_gauge_contraction shape" begin
        D, d = 3, 2
        A = randn(D, D, D, D, d)
        G = [randn(D, D) for _ in 1:4]
        A_out = local_gauge_contraction(A, G)
        @test size(A_out) == (D, D, D, D, d)
    end

    # ================================================================
    # 11. _init_random_ipeps shapes
    # ================================================================
    @testset "_init_random_ipeps shapes" begin
        D, d, N = 2, 2, 2
        Ni, Nj = 2, 2

        # Square: (D,D,D,D,d,N)
        A_sq = _init_random_ipeps(Square(), Float64, D, d, N, Ni, Nj)
        @test size(A_sq) == (D, D, D, D, d, N)

        # Kagome: (D,D,D,D,d^3,N)
        A_kg = _init_random_ipeps(Kagome(), Float64, D, d, N, Ni, Nj)
        @test size(A_kg) == (D, D, D, D, d^3, N)

        # Honeycomb merge: (D,D,D,D,d^2,N)
        A_hm = _init_random_ipeps(Honeycomb{:merge}(), Float64, D, d, N, Ni, Nj)
        @test size(A_hm) == (D, D, D, D, d^2, N)

        # Honeycomb brickwall: (D,1,D,D,d,N)
        A_hb = _init_random_ipeps(Honeycomb{:brickwall_h}(), Float64, D, d, N, Ni, Nj)
        @test size(A_hb) == (D, 1, D, D, d, N)

        # Kagome :onehole — (D,D,D,D,d,N), requires Ni,Nj even
        A_kh = _init_random_ipeps(Kagome(:onehole), Float64, D, d, N, Ni, Nj)
        @test size(A_kh) == (D, D, D, D, d, N)

        # Odd dimensions should error
        @test_throws ArgumentError _init_random_ipeps(Kagome(:onehole), Float64, D, d, N, 3, 2)
        @test_throws ArgumentError _init_random_ipeps(Kagome(:onehole), Float64, D, d, N, 2, 3)

        # Kagome :onehole_real — same shape as :onehole, requires Ni,Nj even
        A_khr = _init_random_ipeps(Kagome(:onehole_real), Float64, D, d, N, Ni, Nj)
        @test size(A_khr) == (D, D, D, D, d, N)
        @test_throws ArgumentError _init_random_ipeps(Kagome(:onehole_real), Float64, D, d, N, 3, 2)
    end

    # ================================================================
    # _lattice_map(::Kagome{:onehole_real}, ...) injects δ at site 4
    # ================================================================
    @testset "Kagome :onehole_real δ injection" begin
        D = 3
        pattern = [1 3; 2 4]
        # Build a fake input array of the expected shape
        A = randn(D, D, D, D, 2, 4)
        # Wrap as StructArray then apply _lattice_map
        Ar = TeneT.StructArray([A[:,:,:,:,:,i] for i in 1:4], pattern)
        Ar2 = TeneT._lattice_map(Ar, Kagome(:onehole_real), pattern)

        # Sites 1, 2, 3 unchanged (same data references)
        @test Ar2[1,1] === Ar[1,1]
        @test Ar2[2,1] === Ar[2,1]
        @test Ar2[1,2] === Ar[1,2]
        # Site 4 (empty) replaced with d=1 δ tensor
        @test size(Ar2[2,2]) == (D, D, D, D, 1)
        # δ_{u,l} * δ_{r,d}: nonzero only when u==l AND d==r
        # iPEPS index convention is (l, d, r, u, p)
        for l in 1:D, d in 1:D, r in 1:D, u in 1:D
            expected = (u == l && d == r) ? 1.0 : 0.0
            @test Ar2[2,2][l, d, r, u, 1] == expected
        end
    end

    # ================================================================
    # 12. Environment struct fieldnames
    # ================================================================
    @testset "Environment struct fieldnames" begin
        @test fieldnames(VUMPSRuntime) == (:AL, :AR, :C, :FL, :FR)
        @test fieldnames(PlaquetteVUMPSRuntime) == (:AL, :C, :FL)
        @test fieldnames(VUMPSEnv) == (:ACu, :ARu, :ACd, :ARd, :FLu, :FRu, :FLo, :FRo)
        @test fieldnames(PlaquetteVUMPSEnv) == (:AL, :C, :FLu, :FLo)
        @test fieldnames(CTMEnv) == (:C, :T)
    end

    # ================================================================
    # 13. GradientOptimize/SUOptimize/FUOptimize subtypes
    # ================================================================
    @testset "iPEPSOptimize subtypes" begin
        @test GradientOptimize <: TeneT.iPEPSOptimize
        @test SUOptimize <: TeneT.iPEPSOptimize
        @test FUOptimize <: TeneT.iPEPSOptimize
    end

    # ================================================================
    # 14. _inner product
    # ================================================================
    @testset "_inner product" begin
        x = randn(4, 4)
        dx1 = randn(4, 4) + im * randn(4, 4)
        dx2 = randn(4, 4) + im * randn(4, 4)
        result = _inner(x, dx1, dx2)
        @test result isa Real
        @test result ≈ real(dot(dx1, dx2))
    end

    @testset "_init_random_ipeps :brickwall_v" begin
        D, d, N, Ni, Nj = 3, 2, 6, 6, 2
        A = TeneT._init_random_ipeps(Honeycomb{:brickwall_v}(), Float64, D, d, N, Ni, Nj)
        @test size(A) == (1, D, D, D, d, N)   # dim-1 on l leg
        @test eltype(A) == Float64

        # Constraint: Ni, Nj both even
        @test_throws ArgumentError TeneT._init_random_ipeps(Honeycomb{:brickwall_v}(), Float64, D, d, N, 3, 2)
        @test_throws ArgumentError TeneT._init_random_ipeps(Honeycomb{:brickwall_v}(), Float64, D, d, N, 2, 3)
    end

    @testset "_lattice_map :brickwall_v" begin
        D, d = 3, 2
        Ni, Nj = 4, 2
        N = Ni * Nj
        pattern = reshape(1:N, Ni, Nj)
        A_raw = rand(Float64, 1, D, D, D, d, N)  # initial shape for :brickwall_v
        A = TeneT.StructArray([A_raw[:,:,:,:,:,i] for i in 1:N], pattern)
        Ar = TeneT._lattice_map(A, Honeycomb{:brickwall_v}(), pattern)

        # Even-parity site: unchanged → shape (1,D,D,D,d)
        # Odd-parity site: permutedims (3,4,1,2,5) → (D,D,1,D,d)
        for i in 1:N
            pos = findfirst(==(i), pattern)
            if sum(Tuple(pos)) % 2 == 0
                @test size(Ar[i]) == (1, D, D, D, d)
            else
                @test size(Ar[i]) == (D, D, 1, D, d)
            end
        end
    end

    @testset "gauge_transfer dispatches on lattice for :brickwall_v" begin
        using LinearAlgebra: I
        D, d = 2, 2
        Ni, Nj = 2, 2
        N = Ni * Nj
        pattern = reshape(1:N, Ni, Nj)
        # :brickwall_v shape (1, D, D, D, d, N) — dim-1 on l-leg
        A = rand(Float64, 1, D, D, D, d, N)

        # Identity gauges for :brickwall_v.  Gh[q] lives on the r-leg (post-permutation),
        # whose size depends on parity:
        #   even-parity site (no permutation): r-leg size = D3 = D  → Gh = I(D)
        #   odd-parity site  (after (3,4,1,2,5) permutation): new r-leg = old l-leg = D1 = 1
        #                                                    → Gh = I(1)
        # Gv[q] lives on the d-leg, always size D2 = D in :brickwall_v.
        Gh = [begin
                  pos = findfirst(==(q), pattern)
                  sum(Tuple(pos)) % 2 == 0 ? Matrix{Float64}(I, D, D) : Matrix{Float64}(I, 1, 1)
              end for q in 1:N]
        Gv = [Matrix{Float64}(I, D, D) for _ in 1:N]

        mock_params = (pattern=pattern, model=(lattice=Honeycomb{:brickwall_v}(),))
        A2 = TeneT.gauge_transfer(A, [Gh, Gv], mock_params)

        # Identity gauges should leave A unchanged (and the routing through the lattice
        # type ensures the gauges land on the right legs for both parities).
        @test size(A2) == size(A)
        @test A2 ≈ A

        # Sanity: gauge_transfer with `:brickwall_h` (existing logic) still works as before.
        # For :brickwall_h, Gh is uniform I(D) (r-leg always size D), while Gv is parity-mixed
        # (d-leg size 1 at even-parity, D at odd-parity).
        Ah = rand(Float64, D, 1, D, D, d, N)
        Gh_h = [Matrix{Float64}(I, D, D) for _ in 1:N]
        Gv_h = [begin
                    pos = findfirst(==(q), pattern)
                    sum(Tuple(pos)) % 2 == 0 ? Matrix{Float64}(I, 1, 1) : Matrix{Float64}(I, D, D)
                end for q in 1:N]
        mock_params_h = (pattern=pattern, model=(lattice=Honeycomb{:brickwall_h}(),))
        Ah2 = TeneT.gauge_transfer(Ah, [Gh_h, Gv_h], mock_params_h)
        @test size(Ah2) == size(Ah)
        @test Ah2 ≈ Ah
    end

    @testset "enlarge_coupling J1J2p :brickwall_v" begin
        # Uniform: J1h = J1v = J1 for all (i,j)
        m_unif = J1J2p(lattice=Honeycomb{:brickwall_v}(), J1=1.5, J2p=0.3, couplingtype=:uniform)
        @test TeneT.enlarge_coupling(m_unif, 1, 1) == (1.5, 1.5)
        @test TeneT.enlarge_coupling(m_unif, 3, 2) == (1.5, 1.5)
        @test TeneT.enlarge_coupling(m_unif, 6, 2) == (1.5, 1.5)

        # Plaquette mode is intentionally deferred — calling it should error clearly
        m_plaq = J1J2p(lattice=Honeycomb{:brickwall_v}(), J1=1.0, J2p=0.3,
                       couplingtype=:plaquette, bondratio=0.5)
        @test_throws ArgumentError TeneT.enlarge_coupling(m_plaq, 1, 1)
    end

    # ================================================================
    # energy_value(::J1J2p{Honeycomb{:brickwall_v}}, ...) smoke test
    # Exercises: dispatch + bond enumeration (J1V/J1H/J2//J2\\/J2V keys).
    # Indices and absolute values are validated by the Stage-1 :h↔:v
    # benchmark in examples/; this test only catches structural breakage
    # (MethodErrors, wrong arity, missing bond keys).
    # ================================================================
    @testset "energy_value J1J2p :brickwall_v smoke" begin
        using OptimKit: LBFGS
        using TeneT: ObsEnv, energy_value, build_A,
                     leading_boundary, initialize_env, J1J2p

        Random.seed!(7)
        D, χ = 2, 4
        pattern = [1 4; 2 5; 3 6; 4 1; 5 2; 6 3]
        model = J1J2p(lattice=Honeycomb{:brickwall_v}(),
                      S=0.5, J1=1.0, J2p=0.3,
                      ifrotate=false,
                      couplingtype=:uniform, bondratio=1.0)
        folder = mktempdir()
        boundary_alg = VUMPS{TeneT.General}(ifupdown=true, ifsimple_eig=true,
                                            maxiter=3, miniter=0,
                                            maxiter_ad=1, miniter_ad=1,
                                            tol=1e-3, verbosity=0, show_every=1000)
        params = GradientOptimize(model=model, pattern=pattern,
                                  boundary_alg=boundary_alg,
                                  optimizer=LBFGS(10; maxiter=1, gradtol=1e-3, verbosity=0),
                                  maxiter_restart=1, verbosity=0, folder=folder,
                                  ifSU=false, SUτ=0.0, ifprecondition=false,
                                  reuse_env=true, ifsave_env=false, ifload_env=false,
                                  ifsave_lbfgs=false, ifload_lbfgs=false)

        # Raw shape for :brickwall_v is (1, D, D, D, d, N) — dim-1 on l-leg.
        # `pattern` has 6 unique site labels, so N = 6.
        d, N = 2, 6
        A_raw = (rand(Float64, 1, D, D, D, d, N) .- 0.5)
        A_raw /= norm(A_raw)
        A = build_A(A_raw, params)

        rt = initialize_env(A_raw, D, χ, params)
        rt, _ = leading_boundary(rt, A, params.boundary_alg)
        env = ObsEnv(rt, A, params.boundary_alg)

        e, e_dict = energy_value(model, A, env, params)
        @test isfinite(e)
        # All 5 bond categories should be present in the dict
        @test haskey(e_dict, "bond_J1V_energy")
        @test haskey(e_dict, "bond_J1H_energy")
        @test haskey(e_dict, "bond_J2/_energy")
        @test haskey(e_dict, "bond_J2\\_energy")
        @test haskey(e_dict, "bond_J2V_energy")
        # J1V is always-on → every (i,j) contributes
        @test length(e_dict["bond_J1V_energy"]) == length(unique(pattern))
        # Parity-conditional bonds populate half the sites
        @test !isempty(e_dict["bond_J1H_energy"])
        @test !isempty(e_dict["bond_J2/_energy"])
        @test !isempty(e_dict["bond_J2\\_energy"])
        @test !isempty(e_dict["bond_J2V_energy"])
        # Each per-bond entry must be finite
        for key in keys(e_dict)
            for (_, v) in e_dict[key]
                @test isfinite(v)
            end
        end
    end

end
