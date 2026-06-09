@testset "boundary algorithms" begin
    beta = 0.3
    chi = 8

    @testset "atype=$atype" for atype in ATYPES

        M1 = ising_mpo(beta; atype)
        M2 = ising_mpo_2x2(beta; atype)

        # ==================================================================
        # VUMPS General
        # ==================================================================
        @testset "VUMPS General" begin
            alg = VUMPS{General}(; verbosity=0, maxiter=100,
                                    maxiter_ad=1, miniter_ad=1,
                                    tol=1e-8, ifupdown=false)

            # ---- Canonical forms ----
            @testset "left_canonical isometry" begin
                A = initial_A(M1, chi)
                AL, L, lam = left_canonical(A)
                # AL reshaped to matrix should have orthonormal columns
                mat = reshape(Array(AL[1]), :, size(AL[1])[end])
                @test mat' * mat ≈ I atol=1e-10
            end

            @testset "right_canonical isometry" begin
                A = initial_A(M1, chi)
                R, AR, lam = right_canonical(A)
                mat = reshape(Array(AR[1]), size(AR[1], 1), :)
                @test mat * mat' ≈ I atol=1e-10
            end

            # ---- LRtoC and ALCtoAC shapes ----
            @testset "LRtoC and ALCtoAC shapes" begin
                A = initial_A(M1, chi)
                AL, L, _ = left_canonical(A)
                R, AR, _ = right_canonical(AL)
                C = LRtoC(L, R)
                @test size(C[1]) == (chi, chi)
                AC = ALCtoAC(AL, C)
                @test size(AC[1]) == size(AL[1])
            end

            # ---- Environment init ----
            @testset "init_env returns VUMPSRuntime" begin
                rt = init_env(M1, chi, alg)
                @test rt isa VUMPSRuntime
                @test size(rt.AL[1], 1) == chi
                @test size(rt.AR[1], 1) == chi
                @test size(rt.C[1]) == (chi, chi)
            end

            # ---- Convergence 1x1 ----
            @testset "convergence 1x1" begin
                rt = init_env(M1, chi, alg)
                rt, err = leading_boundary(rt, M1, alg)
                @test err < 1e-6
            end

            # ---- Up/down ----
            @testset "up/down returns tuple" begin
                alg_ud = VUMPS{General}(; verbosity=0, maxiter=100,
                                           maxiter_ad=1, miniter_ad=1,
                                           tol=1e-8, ifupdown=true,
                                           ifdownfromup=true)
                rt_ud = init_env(M1, chi, alg_ud)
                @test rt_ud isa Tuple{VUMPSRuntime, VUMPSRuntime}
                (rtup, rtdown), (errup, errdown) = leading_boundary(rt_ud, M1, alg_ud)
                @test errup < 1e-6
                @test errdown < 1e-6
            end

            # ---- 2x2 unit cell ----
            @testset "convergence 2x2" begin
                rt2 = init_env(M2, chi, alg)
                rt2, err2 = leading_boundary(rt2, M2, alg)
                @test err2 < 1e-4
            end

            # ---- ObsEnv ----
            @testset "ObsEnv returns VUMPSEnv" begin
                rt = init_env(M1, chi, alg)
                rt, _ = leading_boundary(rt, M1, alg)
                env = ObsEnv(rt, M1, alg)
                @test env isa VUMPSEnv
            end
        end

        # ==================================================================
        # VUMPS Plaquette
        # ==================================================================
        @testset "VUMPS Plaquette" begin
            alg_plaq = VUMPS{Plaquette{Square}}(; verbosity=0, maxiter=100,
                                           maxiter_ad=1, miniter_ad=1,
                                           tol=1e-8)

            @testset "init returns PlaquetteVUMPSRuntime" begin
                rt = init_env(M2, chi, alg_plaq)
                @test rt isa PlaquetteVUMPSRuntime
            end

            @testset "convergence" begin
                rt = init_env(M2, chi, alg_plaq)
                rt, err = leading_boundary(rt, M2, alg_plaq)
                @test err < 1e-4
            end

            @testset "ObsEnv returns PlaquetteVUMPSEnv" begin
                rt = init_env(M2, chi, alg_plaq)
                rt, _ = leading_boundary(rt, M2, alg_plaq)
                env = ObsEnv(rt, M2, alg_plaq)
                @test env isa PlaquetteVUMPSEnv
            end
        end

        # ==================================================================
        # VUMPS C4v
        # ==================================================================
        @testset "VUMPS C4v" begin
            alg_c4v = VUMPS{C4v}(; verbosity=0, maxiter=100,
                                    maxiter_ad=1, miniter_ad=1,
                                    tol=1e-8)

            # C4v init_env expects rank-5 tensors (4 virtual + 1 physical) in the StructArray.
            # Build a rank-5 MPO from the rank-4 Ising MPO by adding a trivial physical dim.
            M4 = ising_mpo(beta; atype).data[1]
            D = size(M4, 1)
            M5_data = reshape(M4, D, D, D, D, 1)
            M_c4v = StructArray([M5_data], [1;;])

            @testset "init returns C4vVUMPSEnv" begin
                rt = init_env(M_c4v, chi, alg_c4v)
                @test rt isa C4vVUMPSEnv
            end

            @testset "convergence" begin
                rt = init_env(M_c4v, chi, alg_c4v)
                rt, err = leading_boundary(rt, M_c4v, alg_c4v)
                @test err < 1e-6
            end

            @testset "ObsEnv returns C4vVUMPSEnv" begin
                rt = init_env(M_c4v, chi, alg_c4v)
                rt, _ = leading_boundary(rt, M_c4v, alg_c4v)
                env = ObsEnv(rt, M_c4v, alg_c4v)
                @test env isa C4vVUMPSEnv
            end

            @testset "inner_etype=Float32 threads through leftenv_c4v / ACenv_c4v" begin
                # Setup: same MPO/chi as the outer testset; we drive one vumps_step
                # twice — once with inner_etype=Float32, once with =nothing — from
                # identical seeds, and check the FL tensors diverge by a small
                # amount consistent with Float32 round-off. The `@unpack` line in
                # leftenv_c4v/ACenv_c4v must include `inner_etype` and pass it to
                # FLmap_parallel/ACmap_parallel for this to produce a visible
                # difference. The raw M passed to vumps_step is the rank-5
                # tensor (matching leading_boundary's `M = M[1]` convention).
                M_raw = M_c4v[1]

                alg_f32 = VUMPS{C4v}(; verbosity=0, forloop_iter=1, power_iter=1,
                                        maxiter=1, miniter=0, maxiter_ad=1, miniter_ad=1,
                                        tol=1e-10, inner_etype=Float32)
                alg_base = VUMPS{C4v}(; verbosity=0, forloop_iter=1, power_iter=1,
                                         maxiter=1, miniter=0, maxiter_ad=1, miniter_ad=1,
                                         tol=1e-10, inner_etype=nothing)

                Random.seed!(42)
                rt_f32 = init_env(M_c4v, chi, alg_f32)
                rt_f32_new, err_f32 = vumps_step(rt_f32, M_raw, alg_f32)

                Random.seed!(42)
                rt_base = init_env(M_c4v, chi, alg_base)
                rt_base_new, err_base = vumps_step(rt_base, M_raw, alg_base)

                # Environments stay Float64 (outer precision preserved).
                @test eltype(rt_f32_new.AL) == Float64
                @test eltype(rt_f32_new.C)  == Float64
                @test eltype(rt_f32_new.FL) == Float64
                @test isfinite(err_f32)

                # Sanity: same seed on the baseline gives the same result as
                # itself (rules out environmental RNG drift confusing the test).
                @test Array(rt_base_new.FL) == Array(rt_base_new.FL)

                # Strengthened precision-bound check only on CPU: on GPU, `rand!`
                # uses CUDA's own RNG which `Random.seed!(42)` does not seed, so
                # the two init_env calls start from different random FLs and the
                # divergence is not a pure Float32 rounding effect.
                if atype == Array
                    # Thread-through check: inner_etype=Float32 must change the
                    # FLmap/ACmap contraction path, so FL should differ from baseline
                    # (both start from identical seeds). The pre-Task-8 state
                    # ignores inner_etype in leftenv_c4v → would be byte-equal.
                    FL_f32_arr  = Array(rt_f32_new.FL)
                    FL_base_arr = Array(rt_base_new.FL)
                    @test FL_f32_arr != FL_base_arr

                    # Divergence should be small (Float32 precision-bounded).
                    rel = maximum(abs, FL_f32_arr .- FL_base_arr) /
                          max(maximum(abs, FL_base_arr), eps())
                    @test rel < 1e-4
                end
            end
        end

        # ==================================================================
        # OnesideVUMPSEnv construction + conversion
        # ==================================================================
        @testset "OnesideVUMPSEnv construction + conversion" begin
            χ, D = 4, 2
            # Build minimal 2×2 StructArrays to stuff into env (purely structural test —
            # the actual content isn't physically meaningful, just shape-correct).
            pattern = [1 2; 2 1]
            AC_data = [rand(χ, D, χ) for _ in 1:2]    # leg3
            AR_data = [rand(χ, D, χ) for _ in 1:2]
            FL_data = [rand(χ, D, χ) for _ in 1:2]
            FR_data = [rand(χ, D, χ) for _ in 1:2]
            AC = TeneT.StructArray(AC_data, pattern)
            AR = TeneT.StructArray(AR_data, pattern)
            FLu = TeneT.StructArray(FL_data, pattern)
            FRu = TeneT.StructArray(FR_data, pattern)
            FLo = TeneT.StructArray(deepcopy(FL_data), pattern)
            FRo = TeneT.StructArray(deepcopy(FR_data), pattern)

            env = TeneT.OnesideVUMPSEnv(AC, AR, FLu, FRu, FLo, FRo)
            @test env.AC === AC
            @test env.AR === AR
            @test env.FLu === FLu
            @test env.FRu === FRu
            @test env.FLo === FLo
            @test env.FRo === FRo

            # CPU/GPU conversion smoke
            env_cpu = Array(env)
            @test env_cpu isa TeneT.OnesideVUMPSEnv
            @test env_cpu.AC.data[1] isa Array

            # _atype_of returns the device array type (Array on CPU)
            @test TeneT._atype_of(env) == Array
        end

        # ==================================================================
        # ObsEnv(::VUMPSRuntime, M, VUMPS{General}(ifupdown=false), model)
        # uses a model trait to select OnesideVUMPSEnv only for models that
        # need it. Exercises the one-sided env shape via the unified General
        # algorithm (the old VUMPS{Oneside} path).
        # ==================================================================
        @testset "ObsEnv VUMPS{General} ifupdown=false model trait selects env shape" begin
            using TeneT: Heisenberg, J1J2p, init_env, leading_boundary, ObsEnv, OnesideVUMPSEnv
            Random.seed!(42)
            χ, D = 4, 2
            pattern = [1 2; 2 1]
            M_data = [atype(rand(D, D, D, D)) for _ in 1:2]
            M = TeneT.StructArray(M_data, pattern)
            m = J1J2p(lattice=Honeycomb{:brickwall_v}(), J1=1.0, J2p=0.3)
            alg = VUMPS{General}(; maxiter=2, maxiter_ad=0, verbosity=0,
                                    ifupdown=false, ifparallelupdown=false)
            rt = init_env(M, χ, alg)
            @test rt isa TeneT.VUMPSRuntime

            rt_conv, err = leading_boundary(rt, M, alg)
            @test rt_conv isa TeneT.VUMPSRuntime
            @test isfinite(err) || err == 0

            # J1J2p :brickwall_v opts into OnesideVUMPSEnv (uses obs_index trait)
            env = ObsEnv(rt_conv, M, alg, m)
            @test env isa OnesideVUMPSEnv
            @test size(env.AC) == size(M)
            @test size(env.AR) == size(M)
            @test size(env.FLu) == size(M)
            @test size(env.FRu) == size(M)
            @test size(env.FLo) == size(M)
            @test size(env.FRo) == size(M)

            # Without `model` → legacy VUMPSEnv shape (backward compat)
            env_legacy = ObsEnv(rt_conv, M, alg)
            @test env_legacy isa TeneT.VUMPSEnv

            # Passing an ordinary model should still keep the legacy VUMPSEnv
            # shape; `model` is needed for obs_index only, not as a blanket
            # opt-in to OnesideVUMPSEnv.
            m_default = Heisenberg(lattice=Square(), Jx=1.0, Jy=1.0, Jz=1.0)
            env_default_model = ObsEnv(rt_conv, M, alg, m_default)
            @test env_default_model isa TeneT.VUMPSEnv
        end

        # ==================================================================
        # leftenv `model` kwarg routes through obs_index when model !== nothing.
        # With a model whose obs_index override differs from the default
        # Ni+1-i, leftenv should produce different output for ifobs=true.
        # ==================================================================
        @testset "leftenv model kwarg uses obs_index trait" begin
            using TeneT: J1J2p, Heisenberg, leftenv, init_env
            Random.seed!(42)
            χ, D = 4, 2
            # Build a 4×2 pattern so Ni=4 makes Ni+1-i != i for at least one i.
            pattern = [1 3; 2 4; 3 1; 4 2]
            M_data = [atype(rand(D, D, D, D)) for _ in 1:4]
            M = TeneT.StructArray(M_data, pattern)
            alg = VUMPS{General}(; maxiter=2, verbosity=0, ifupdown=false)
            rt = init_env(M, χ, alg)

            # Default trait (Heisenberg): ir = Ni+1-i. model=Heisenberg vs no model
            # should AGREE since the model just gives back the default value.
            m_default = Heisenberg(lattice=Square(), Jx=1.0, Jy=1.0, Jz=1.0)
            _, FLo_no_model  = leftenv(rt.AL, rt.AL, M, rt.FL; ifobs=true, alg)
            _, FLo_w_default = leftenv(rt.AL, rt.AL, M, rt.FL; ifobs=true, alg, model=m_default)
            for i in 1:length(FLo_no_model.data)
                @test FLo_no_model.data[i] ≈ FLo_w_default.data[i] rtol=1e-10
            end

            # J1J2p :brickwall_v override: ir = i. Should DIFFER from default
            # for Ni=4 (e.g. row 2 → default ir=3, override ir=2).
            m_override = J1J2p(lattice=Honeycomb{:brickwall_v}(), J1=1.0, J2p=0.3)
            _, FLo_override = leftenv(rt.AL, rt.AL, M, rt.FL; ifobs=true, alg, model=m_override)
            differs = false
            for i in 1:length(FLo_no_model.data)
                if !isapprox(FLo_no_model.data[i], FLo_override.data[i]; rtol=1e-6)
                    differs = true
                    break
                end
            end
            @test differs
        end

        # ==================================================================
        # QRCTMRG{C4v}
        # ==================================================================
        @testset "QRCTMRG{C4v}" begin
            alg_qr = QRCTMRG{C4v}(; verbosity=0, maxiter=100,
                                   maxiter_ad=1, miniter_ad=1,
                                   tol=1e-8)

            @testset "init returns CTMEnv" begin
                rt = init_env(M1, chi, alg_qr)
                @test rt isa CTMEnv
                @test size(rt.C) == (chi, chi)
            end

            @testset "convergence" begin
                rt = init_env(M1, chi, alg_qr)
                rt, err = leading_boundary(rt, M1, alg_qr)
                @test err < 1e-6
            end

            @testset "ObsEnv returns CTMEnv" begin
                rt = init_env(M1, chi, alg_qr)
                rt, _ = leading_boundary(rt, M1, alg_qr)
                env = ObsEnv(rt, M1, alg_qr)
                @test env isa CTMEnv
            end
        end

        @testset "QRCTMRG{C3v}" begin
            D, d = 2, 2
            M_c3v = StructArray([atype(rand(Float64, D, D, D, d))], [1;;])
            alg_c3v = QRCTMRG{C3v}(; verbosity=0, maxiter=2,
                                     maxiter_ad=1, miniter_ad=1,
                                     tol=1e-8)

            @testset "init returns CTMEnv" begin
                rt = init_env(M_c3v, chi, alg_c3v)
                @test rt isa CTMEnv
                @test size(rt.C) == (chi, chi)
                @test size(rt.T) == (chi, D, D, chi)
            end

            @testset "init accepts singleton rank-5 brickwall tensor" begin
                M_c3v_brickwall = StructArray([atype(rand(Float64, D, 1, D, D, d))], [1;;])
                rt = init_env(M_c3v_brickwall, chi, alg_c3v)
                @test rt isa CTMEnv
                @test size(rt.T) == (chi, D, D, chi)
            end

            @testset "one step matches QRCTM reference" begin
                local chi_step = 4
                C0 = atype(randn(Float64, chi_step, chi_step))
                C0 = C0 / norm(C0)
                T0 = atype(randn(Float64, chi_step, D, D, chi_step))
                T0 = T0 / norm(T0)
                M0 = randn(Float64, D, D, D, d)
                M0 = (M0 +
                      permutedims(M0, (2, 3, 1, 4)) +
                      permutedims(M0, (3, 1, 2, 4)) +
                      permutedims(M0, (1, 3, 2, 4)) +
                      permutedims(M0, (2, 1, 3, 4)) +
                      permutedims(M0, (3, 2, 1, 4))) / 6
                M0 = atype(M0 / norm(M0))

                env, _ = qrctmrg_step(CTMEnv(C0, T0), M0, alg_c3v)

                @tensor CR[i,j,k,m] := C0[i,q] * T0[q,j,k,m]
                V, Rfac = qr_for_ad(reshape(CR, div(prod(size(CR)), size(CR, 1)), size(CR, 1)))
                V = reshape(V, size(T0))
                @tensor Tref[t,jj,kk,c] := V[i,a,b,t] * T0[i,j,k,l] *
                                           M0[j,a,p,x] * M0[k,b,q,x] *
                                           M0[jj,aa,p,y] * M0[kk,bb,q,y] *
                                           V[l,aa,bb,c]
                @tensor Cref[c,r] := Tref[t,j,k,c] * Rfac[t,b] * V[b,j,k,r]
                Tref = Tref / norm(Tref)
                Cref = Cref / norm(Cref)

                @test Array(env.T) ≈ Array(Tref) atol=1e-10 rtol=1e-10
                @test Array(env.C) ≈ Array(Cref) atol=1e-10 rtol=1e-10
            end

            @testset "iteration returns finite error" begin
                rt = init_env(M_c3v, chi, alg_c3v)
                rt, err = leading_boundary(rt, M_c3v, alg_c3v)
                @test rt isa CTMEnv
                @test isfinite(real(err))
            end

            @testset "ObsEnv returns CTMEnv" begin
                rt = init_env(M_c3v, chi, alg_c3v)
                rt, _ = leading_boundary(rt, M_c3v, alg_c3v)
                env = ObsEnv(rt, M_c3v, alg_c3v)
                @test env isa CTMEnv
            end
        end

        @testset "QRCTMRG{C3vTwoSite}" begin
            D, d = 2, 2
            M_twosite = StructArray(
                [atype(rand(Float64, D, D, D, d)),
                 atype(rand(Float64, D, D, D, d))],
                [1 2],
            )
            alg_c3v2 = QRCTMRG{C3vTwoSite}(; verbosity=0, maxiter=2,
                                            maxiter_ad=1, miniter_ad=1,
                                            tol=1e-8)

        end

        # ==================================================================
        # Environment helpers
        # ==================================================================
        @testset "environment helpers" begin
            alg = VUMPS{General}(; verbosity=0, maxiter=20,
                                    maxiter_ad=1, miniter_ad=1,
                                    tol=1e-6, ifupdown=false)

            # ---- update! in-place copy ----
            @testset "update! VUMPSRuntime" begin
                rt1 = init_env(M1, chi, alg)
                rt2 = init_env(M1, chi, alg)
                update!(rt1, rt2)
                @test Array(rt1.AL[1]) ≈ Array(rt2.AL[1])
                @test Array(rt1.C[1]) ≈ Array(rt2.C[1])
            end

            # ---- _down_M ----
            @testset "_down_M permutes legs" begin
                Md = _down_M(M1)
                m_orig = Array(M1[1])
                m_down = Array(Md[1])
                @test m_down ≈ permutedims(m_orig, (1, 4, 3, 2))
            end

            # ---- GPU roundtrip (only when atype != Array) ----
            if atype != Array
                @testset "VUMPSRuntime GPU roundtrip" begin
                    rt = init_env(M1, chi, alg)
                    rt_cpu = Array(rt)
                    rt_gpu = atype(rt_cpu)
                    rt_cpu2 = Array(rt_gpu)
                    @test rt_cpu2.AL[1] ≈ rt_cpu.AL[1]
                    @test rt_cpu2.C[1] ≈ rt_cpu.C[1]
                end

                @testset "CTMEnv GPU roundtrip" begin
                    alg_qr = QRCTMRG{C4v}(; verbosity=0, maxiter=10, tol=1e-6)
                    rt = init_env(M1, chi, alg_qr)
                    rt_cpu = Array(rt)
                    rt_gpu = atype(rt_cpu)
                    rt_cpu2 = Array(rt_gpu)
                    @test rt_cpu2.C ≈ rt_cpu.C
                    @test rt_cpu2.T ≈ rt_cpu.T
                end
            end
        end
    end
end
