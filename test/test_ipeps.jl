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
        A_out = _lattice_map(A, Honeycomb{:brickwall}(), pattern)
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
    # 2b. _lattice_map -- Honeycomb brickwall_pi6
    # ================================================================
    @testset "_lattice_map Honeycomb brickwall_pi6" begin
        pattern = [1 2; 3 4]
        D, d = 2, 2
        # In brickwall_pi6 the trivial leg is L (position 1), so init shape is (1,D,D,D,d).
        tensors = [randn(1, D, D, D, d) for _ in 1:4]
        A = StructArray(tensors, pattern)
        A_out = _lattice_map(A, Honeycomb{:brickwall_pi6}(), pattern)
        # Permutation for odd-parity sites is (3, 2, 1, 4, 5): swap L ↔ R, keep D/U/phys.
        @test Array(A_out.data[1]) ≈ Array(A.data[1])
        @test Array(A_out.data[2]) ≈ permutedims(Array(A.data[3]), (3, 2, 1, 4, 5))
        @test Array(A_out.data[3]) ≈ permutedims(Array(A.data[2]), (3, 2, 1, 4, 5))
        @test Array(A_out.data[4]) ≈ Array(A.data[4])
        # After mapping, every site has non-trivial U (leg 4) and D (leg 2);
        # the trivial leg is horizontal (exactly one of L/R is dim 1 per site).
        for k in 1:4
            @test size(A_out.data[k], 2) == D
            @test size(A_out.data[k], 4) == D
            @test (size(A_out.data[k], 1) == 1) ⊻ (size(A_out.data[k], 3) == 1)
        end
    end

    # ================================================================
    # 2c. _lattice_map -- Honeycomb brickwall_pi6 rejects mixed-parity patterns
    # ================================================================
    @testset "_lattice_map Honeycomb brickwall_pi6 mixed-parity rejection" begin
        # Tensor 1 at (1,1) [even] and (3,2) [odd] — must be rejected.
        pattern = [1 2; 3 4; 2 1; 4 3]
        D, d = 2, 2
        tensors = [randn(1, D, D, D, d) for _ in 1:4]
        A = StructArray(tensors, pattern)
        @test_throws ArgumentError _lattice_map(A, Honeycomb{:brickwall_pi6}(), pattern)
    end

    # ================================================================
    # 2d. dumu_symmetrize is an idempotent projector onto D↔U symmetric subspace
    # ================================================================
    @testset "dumu_symmetrize idempotent + D↔U fixed point" begin
        D, d, N = 3, 2, 2
        A = randn(1, D, D, D, d, N)
        A1 = dumu_symmetrize(A)
        A2 = dumu_symmetrize(A1)
        # idempotent
        @test A1 ≈ A2
        # exact D↔U symmetry
        @test A1 ≈ permutedims(A1, (1, 4, 3, 2, 5, 6))
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
        A_hb = _init_random_ipeps(Honeycomb{:brickwall}(), Float64, D, d, N, Ni, Nj)
        @test size(A_hb) == (D, 1, D, D, d, N)

        # Honeycomb brickwall_pi6: (1,D,D,D,d,N) — trivial leg is L (position 1)
        A_hb6 = _init_random_ipeps(Honeycomb{:brickwall_pi6}(), Float64, D, d, N, Ni, Nj)
        @test size(A_hb6) == (1, D, D, D, d, N)

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

end
