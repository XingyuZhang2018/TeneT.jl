# Tests for src/ipeps_optimize/optimize_fixedpoint.jl
# Built up incrementally per docs/plans/2026-04-27-ipeps-fixedpoint-mcf-plan.md.

using TeneT
using TeneT: iPEPSFixedPointConfig, leading_boundary, init_VUMPSRuntime, build_A
using TensorOperations: @tensor
using LinearAlgebra: norm
using Random
Random.seed!(42)

@testset "iPEPSFixedPointConfig" begin
    cfg = iPEPSFixedPointConfig()
    @test cfg.env_mode == :A
    @test cfg.H_eff_mode == :a
    @test cfg.decompose_method == :X
    @test cfg.mcf_ifignore_gauge == false
    @test cfg.outer_maxiter == 200

    cfg2 = iPEPSFixedPointConfig(env_mode=:C, mcf_ifignore_gauge=true)
    @test cfg2.env_mode == :C
    @test cfg2.mcf_ifignore_gauge == true
end

@testset "build_phi horizontal" begin
    D, d = 2, 2
    A = randn(D, D, D, D, d)  # legs (l, d, r, u, p)
    φ = TeneT.build_phi(A, A, Val(:H))
    # φ legs: (l, d_l, u_l, p_l, d_r, u_r, r, p_r)
    @test ndims(φ) == 8
    @test size(φ) == (D, D, D, d, D, D, D, d)

    # Sanity: contraction value should equal explicit @tensor
    φ_ref = similar(φ)
    @tensor φ_ref[l, dl, ul, pl, dr, ur, r, pr] := A[l, dl, c, ul, pl] * A[c, dr, r, ur, pr]
    @test φ ≈ φ_ref
end

@testset "build_phi vertical" begin
    D, d = 2, 2
    A = randn(D, D, D, D, d)  # legs (l, d, r, u, p)
    φ = TeneT.build_phi(A, A, Val(:V))
    # Vertical: top.down (leg 2) joins bottom.up (leg 4)
    # Output legs: (l_t, u_t, r_t, p_t, l_b, d_b, r_b, p_b)
    @test ndims(φ) == 8
    @test size(φ) == (D, D, D, d, D, D, D, d)

    φ_ref = similar(φ)
    @tensor φ_ref[lt, ut, rt, pt, lb, db, rb, pb] :=
        A[lt, c, rt, ut, pt] * A[lb, db, rb, c, pb]
    @test φ ≈ φ_ref
end

@testset "make_N_op horizontal — sanity" begin
    D, d, χ = 2, 2, 8
    A_raw = randn(D, D, D, D, d, 1)
    A_raw /= norm(A_raw)
    params = TeneT.make_default_params(; D=D, χ=χ)
    A = build_A(A_raw, params)
    rt = init_VUMPSRuntime(A, χ, params.boundary_alg)
    rt, _ = leading_boundary(rt, A, params.boundary_alg)

    φ = TeneT.build_phi(A[1,1], A[1,1], Val(:H))
    N_op = TeneT.make_N_op(rt, A, Val(:H), params)

    Nφ = N_op(φ)
    @test size(Nφ) == size(φ)
    val = sum(conj(φ) .* Nφ)
    @test isfinite(val)

    # Strong consistency check: scalar <φ|N|φ> via N_op should agree
    # with contract_n_12 (the codebase's verified 2-site norm) when
    # φ = build_phi(A, A, Val(:H)).
    env = TeneT.ObsEnv(rt, A, params.boundary_alg)
    n_via_contract = TeneT.contract_n_12(env.FLo[1,1], env.ACu[1,1], A[1,1],
                                         env.ACd[1,1], env.FRo[1,1],
                                         env.ARu[1,1], A[1,1], env.ARd[1,1];
                                         forloop_iter=1, ifparallel=false)
    @test val ≈ n_via_contract  rtol=1e-10
end

@testset "decompose_phi :Z horizontal" begin
    D, d = 2, 2
    A = randn(D, D, D, D, d)
    φ = TeneT.build_phi(A, A, Val(:H))
    A_new, trunc_err = TeneT.decompose_phi(φ, Val(:H); method=:Z, D_max=D)
    @test size(A_new) == size(A)
    @test trunc_err >= 0
    @test all(isfinite, A_new)
    # Input φ has bond rank ≤ D (it's A·A); SVD truncation to D is tight
    @test trunc_err < 1e-10
end

@testset "decompose_phi :Y horizontal" begin
    D, d = 2, 2
    A = randn(D, D, D, D, d)
    φ = TeneT.build_phi(A, A, Val(:H))
    A_new, trunc_err = TeneT.decompose_phi(φ, Val(:H); method=:Y, D_max=D)
    @test size(A_new) == size(A)
    @test trunc_err >= 0
    @test all(isfinite, A_new)
    @test trunc_err < 1e-10
end

@testset "decompose_phi :X horizontal" begin
    D, d = 2, 2
    A = randn(D, D, D, D, d)
    φ = TeneT.build_phi(A, A, Val(:H))
    A_new, trunc_err = TeneT.decompose_phi(φ, Val(:H); method=:X, D_max=D)
    @test size(A_new) == size(A)
    @test trunc_err >= 0
    @test all(isfinite, A_new)
    # :X uses symmetrized φ; rank can grow to 2D=4 when input asymmetric,
    # so truncation to D may be lossy. Just bound trunc_err by total energy.
    φ_refl = permutedims(φ, (7, 5, 6, 8, 2, 3, 1, 4))
    φ_sym  = (φ .+ φ_refl) ./ 2
    @test trunc_err <= norm(φ_sym)^2 + 1e-8
end

@testset "decompose_phi vertical :Z/:Y/:X" begin
    D, d = 2, 2
    A = randn(D, D, D, D, d)
    φ = TeneT.build_phi(A, A, Val(:V))
    for method in (:Z, :Y, :X)
        A_new, trunc_err = TeneT.decompose_phi(φ, Val(:V); method=method, D_max=D)
        @test size(A_new) == size(A)
        @test trunc_err >= 0
        @test all(isfinite, A_new)
    end
end

@testset "make_H_op :a horizontal — sanity" begin
    D, d, χ = 2, 2, 8
    A_raw = randn(D, D, D, D, d, 1)
    A_raw /= norm(A_raw)
    params = TeneT.make_default_params(; D=D, χ=χ)
    A = build_A(A_raw, params)
    rt = init_VUMPSRuntime(A, χ, params.boundary_alg)
    rt, _ = leading_boundary(rt, A, params.boundary_alg)

    φ = TeneT.build_phi(A[1,1], A[1,1], Val(:H))
    H_op = TeneT.make_H_op(rt, A, Val(:H), params; mode=:a)
    N_op = TeneT.make_N_op(rt, A, Val(:H), params)

    Hφ = H_op(φ)
    Nφ = N_op(φ)
    @test size(Hφ) == size(φ)
    val_H = real(sum(conj(φ) .* Hφ))
    val_N = real(sum(conj(φ) .* Nφ))
    e_bond = val_H / val_N
    @test isfinite(e_bond)
    @test abs(e_bond) < 10.0  # bond Heisenberg ⟨h⟩ bounded by ~|J|·S^2

    # Strong sanity: e_bond from H_op should match the bond_H_energy
    # computed by the codebase's energy_value (which uses contract_o_12).
    env = TeneT.ObsEnv(rt, A, params.boundary_alg)
    E_total, e_dict = TeneT.energy_value(params.model, A, env, params)
    e_bond_ref = e_dict["bond_H_energy"]["1,1"]
    @test e_bond ≈ e_bond_ref  rtol=1e-8
end

@testset "make_N_op vertical — sanity" begin
    D, d, χ = 2, 2, 8
    A_raw = randn(D, D, D, D, d, 1)
    A_raw /= norm(A_raw)
    params = TeneT.make_default_params(; D=D, χ=χ)
    A = build_A(A_raw, params)
    rt = init_VUMPSRuntime(A, χ, params.boundary_alg)
    rt, _ = leading_boundary(rt, A, params.boundary_alg)

    φ = TeneT.build_phi(A[1,1], A[1,1], Val(:V))
    N_op = TeneT.make_N_op(rt, A, Val(:V), params)

    Nφ = N_op(φ)
    @test size(Nφ) == size(φ)
    val = sum(conj(φ) .* Nφ)
    @test isfinite(val)

    # Strong consistency: <φ|N|φ> via vertical N_op ≈ contract_n_21.
    env = TeneT.ObsEnv(rt, A, params.boundary_alg)
    n_via_contract = TeneT.contract_n_21(env.ACu[1,1], env.FLu[1,1], A[1,1],
                                         env.FRu[1,1], env.FLo[1,1], A[1,1],
                                         env.FRo[1,1], env.ACd[1,1];
                                         forloop_iter=1, ifparallel=false)
    @test val ≈ n_via_contract  rtol=1e-10
end

@testset "make_H_op :a vertical — sanity" begin
    D, d, χ = 2, 2, 8
    A_raw = randn(D, D, D, D, d, 1)
    A_raw /= norm(A_raw)
    params = TeneT.make_default_params(; D=D, χ=χ)
    A = build_A(A_raw, params)
    rt = init_VUMPSRuntime(A, χ, params.boundary_alg)
    rt, _ = leading_boundary(rt, A, params.boundary_alg)

    φ = TeneT.build_phi(A[1,1], A[1,1], Val(:V))
    H_op = TeneT.make_H_op(rt, A, Val(:V), params; mode=:a)
    N_op = TeneT.make_N_op(rt, A, Val(:V), params)

    Hφ = H_op(φ)
    Nφ = N_op(φ)
    @test size(Hφ) == size(φ)
    val_H = real(sum(conj(φ) .* Hφ))
    val_N = real(sum(conj(φ) .* Nφ))
    e_bond = val_H / val_N
    @test isfinite(e_bond)
    @test abs(e_bond) < 10.0

    # Strong sanity: matches bond_V_energy from energy_value
    env = TeneT.ObsEnv(rt, A, params.boundary_alg)
    _, e_dict = TeneT.energy_value(params.model, A, env, params)
    e_bond_ref = e_dict["bond_V_energy"]["1,1"]
    @test e_bond ≈ e_bond_ref  rtol=1e-8
end
