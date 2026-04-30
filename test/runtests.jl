using Test
using TeneT
using LinearAlgebra
using Random
using CUDA
using Zygote
using TensorOperations
using KrylovKit
using JLD2

using TeneT: StructArray, randSA, cellones, ISA,
             _arraytype, _mattype, set_device_id!, get_device, get_device_id, device_count,
             atype_device!, for_gc, synchronize, reclaim, gc,
             qrpos, lqpos, qr_for_ad, simple_eig, checkpoint, takagi_decomposition, safesign,
             leg3, leg4, leg5, leg8,
             ALCtoAC_map, CTtoT, CTCtoT, _downcast_eltype, FLmap, FRmap, Lmap, Rmap, ACmap, Cmap, ACdmap, Mmap,
             FLmap_parallel, FRmap_parallel, ACmap_parallel, ACdmap_parallel,
             Mmap_parallel, Mumap_parallel, Mdmap_parallel,
             split_count, split_ranges, forloop, forloop_sum,
             contract_n_11, contract_o_11, contract_n_12, contract_n_21,
             contract_o_12, contract_o_21,
             VUMPSRuntime, VUMPSEnv, PlaquetteVUMPSRuntime, PlaquetteVUMPSEnv,
             C4vVUMPSEnv, CTMEnv,
             left_canonical, right_canonical, LRtoC, ALCtoAC,
             FLint, FRint, leftenv, rightenv, ACenv, Cenv,
             ACCtoAL, ACCtoAR, ACCtoALAR,
             leftCenv, rightCenv,
             initial_A, init_VUMPSRuntime, init_env, leading_boundary, ObsEnv,
             vumps_step, qrctm_step,
             ACenv_plaq, Cenv_plaq,
             leftenv_c4v, ACenv_c4v, Cenv_c4v,
             _down_M,
             orth_for_ad, svd_back,
             project_AL, project_AR, project_AL!, project_AR!, retract!,
             permute_fronttail,
             build_A, _lattice_map,
             C4v_restriction, _restriction_ipeps,
             pepsgeneral, pepsgeneral_Ac, ARstoA, ARstoA1, ARstoA2,
             central_canonical1, central_canonical2,
             to_mcf_ipeps,
             local_gauge_contraction, gauge_transfer,
             find_local_min_norm_G, local_min_norm,
             find_local_hermite_G, local_hermite,
             local_min_norm_iter, solve_balancing_gauge,
             SU_parameterization,
             precondition_invese_single_envir,
             init_ipeps, _init_random_ipeps, initialize_env, _create_new_env,
             save_rt, load_rt, read_last_log,
             energy, _inner,
             update!,
             Defaults,
             LBFGSState, save_lbfgs_state, load_lbfgs_state, optimize_reload
using TeneT: GradientOptimize, SUOptimize, FUOptimize

# ---------- Array type setup ----------
const ATYPES = if CUDA.functional()
    [Array, CuArray]
else
    [Array]
end

# ---------- RNG seed ----------
Random.seed!(42)

# ---------- Numerical gradient helper ----------
"""
    num_grad(f, x; delta=1e-5)

Compute the numerical gradient of scalar function `f` with respect to array `x`
using central finite differences. Supports both real and complex arrays.
"""
function num_grad(f, x; delta=1e-5)
    grad = zero(x)
    for i in eachindex(x)
        if eltype(x) <: Complex
            # Real part
            xp = copy(x); xp[i] += delta
            xm = copy(x); xm[i] -= delta
            gr = (f(xp) - f(xm)) / (2 * delta)
            # Imaginary part
            xp = copy(x); xp[i] += delta * im
            xm = copy(x); xm[i] -= delta * im
            gi = (f(xp) - f(xm)) / (2 * delta)
            grad[i] = gr + im * gi
        else
            xp = copy(x); xp[i] += delta
            xm = copy(x); xm[i] -= delta
            grad[i] = (f(xp) - f(xm)) / (2 * delta)
        end
    end
    return grad
end

# ---------- Ising model helpers ----------
"""
    ising_mpo(beta; atype=Array)

Build the 2D classical Ising transfer matrix as a rank-4 tensor
M[s1,s2,s3,s4] = sum_sigma W(s1,sigma)*W(s2,sigma)*W(s3,sigma)*W(s4,sigma)
where W = sqrt(Boltzmann weight matrix), wrapped in a 1x1 StructArray.
"""
function ising_mpo(beta; atype=Array)
    # Boltzmann weight matrix B[s,s'] = exp(beta * s * s') with s in {+1,-1}
    B = [exp(beta) exp(-beta);
         exp(-beta) exp(beta)]
    # W = sqrt of Boltzmann weight matrix
    W = sqrt(B)
    # Build M[s1,s2,s3,s4] = sum_sigma W[s1,sigma]*W[s2,sigma]*W[s3,sigma]*W[s4,sigma]
    # Use stepwise contraction (TensorOperations doesn't allow >2 occurrences of an index)
    @tensor T12[s1, s2, a, b] := W[s1, a] * W[s2, b]
    @tensor T34[s3, s4, a, b] := W[s3, a] * W[s4, b]
    d = size(W, 2)
    T12r = reshape(T12, size(T12,1), size(T12,2), d*d)
    T34r = reshape(T34, size(T34,1), size(T34,2), d*d)
    @tensor M[s1, s2, s3, s4] := T12r[s1, s2, σ] * T34r[s3, s4, σ]
    M = atype(M)
    return StructArray([M], [1;;])
end

"""
    ising_mpo_2x2(beta; atype=Array)

Same as `ising_mpo` but with a 2x2 pattern `[1 1; 1 1]`.
"""
function ising_mpo_2x2(beta; atype=Array)
    B = [exp(beta) exp(-beta);
         exp(-beta) exp(beta)]
    W = sqrt(B)
    @tensor T12[s1, s2, a, b] := W[s1, a] * W[s2, b]
    @tensor T34[s3, s4, a, b] := W[s3, a] * W[s4, b]
    d = size(W, 2)
    T12r = reshape(T12, size(T12,1), size(T12,2), d*d)
    T34r = reshape(T34, size(T34,1), size(T34,2), d*d)
    @tensor M[s1, s2, s3, s4] := T12r[s1, s2, σ] * T34r[s3, s4, σ]
    M = atype(M)
    return StructArray([M], [1 1; 1 1])
end

"""
    ising_exact_free_energy(beta)

Compute the exact free energy per site of the 2D classical Ising model
on the square lattice using Onsager's formula with numerical quadrature.
"""
function ising_exact_free_energy(beta)
    # Onsager's exact solution via numerical integration
    # f = -kT * [ ln(2) + (1/2pi^2) * int_0^pi int_0^pi ln(cosh(2beta)^2
    #             - sinh(2beta)*(cos(t1) + cos(t2))) dt1 dt2 ]
    npts = 10000
    s = 0.0
    dt = pi / npts
    for i in 1:npts
        t1 = (i - 0.5) * dt
        for j in 1:npts
            t2 = (j - 0.5) * dt
            s += log(cosh(2 * beta)^2 - sinh(2 * beta) * (cos(t1) + cos(t2)))
        end
    end
    s *= dt^2 / (2 * pi^2)
    f = -(log(2) + s) / beta
    return f
end

# ---------- Test files ----------
@testset "TeneT.jl" begin
    include("test_types.jl")
    include("test_structarray.jl")
    include("test_utils.jl")
    include("test_checkpoint.jl")
    include("test_contraction.jl")
    include("test_boundary.jl")
    include("test_autodiff.jl")
    include("test_ipeps.jl")
    include("test_patch.jl")
    include("test_kagome_onehole.jl")

    if get(ENV, "TENET_TEST_MPI", "false") == "true"
        include("test_mpi.jl")
    end
end
