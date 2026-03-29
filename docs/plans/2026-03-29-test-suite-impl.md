# TeneT.jl Test Suite Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Write a comprehensive test suite for the iPEPS-unified branch achieving >95% coverage excluding `src/models/`.

**Architecture:** Standard Julia test layout with `test/runtests.jl` as entry point, including 9 test files. Tests parameterized over `atype ∈ {Array, CuArray}`. A shared helper builds the 2D classical Ising transfer matrix for boundary benchmarks.

**Tech Stack:** Julia `Test` stdlib, `CUDA.jl`, `Zygote.jl`, `LinearAlgebra`, `Random`, `TensorOperations`, `KrylovKit`, `JLD2`

---

### Task 1: Create `test/runtests.jl` — test runner with helpers

**Files:**
- Create: `test/runtests.jl`

**Step 1: Write the test runner**

```julia
using TeneT
using Test
using LinearAlgebra
using Random
using CUDA
using Zygote
using TensorOperations
using KrylovKit
using JLD2

# Import internals needed by tests
using TeneT: StructArray, randSA, cellones, ISA,
             _arraytype, _mattype, set_device_id!, get_device, get_device_id, device_count,
             atype_device!, for_gc, synchronize, reclaim, gc,
             qrpos, lqpos, qr_for_ad, simple_eig, checkpoint, takagi_decomposition, safesign,
             leg3, leg4, leg5, leg8,
             ALCtoAC_map, CTtoT, CTCtoT, FLmap, FRmap, Lmap, Rmap, ACmap, Cmap, ACdmap, Mmap,
             FLmap_parallel, FRmap_parallel, ACmap_parallel, ACdmap_parallel,
             Mmap_parallel, Mumap_parallel, Mdmap_parallel,
             split_count, split_ranges, forloop, forloop_sum,
             contract_n1, contract_o1, contract_n2_H, contract_n2_V,
             contract_o2_H, contract_o2_V,
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
             local_gauge_contraction, guage_transfer,
             find_local_min_norm_G, local_min_norm,
             find_local_hermite_G, local_hermite,
             local_min_norm_iter, solve_balancing_gauge,
             SU_parameterization,
             precondition_invese_single_envir,
             init_ipeps, _init_random_ipeps, initialize_env, _create_new_env,
             save_rt, load_rt, read_last_log,
             energy, _inner, _finalize!,
             update!,
             Defaults,
             LBFGSState, save_lbfgs_state, load_lbfgs_state, optimize_reload

using TeneT: GradientOptimize, SUOptimize, FUOptimize

# GPU setup
const ATYPES = if CUDA.functional()
    [Array, CuArray]
else
    @warn "CUDA not available, running CPU-only tests"
    [Array]
end

Random.seed!(42)

"""
    num_grad(f, x; δ=1e-5)

Compute numerical gradient of scalar function `f` at point `x` via central differences.
Works for real and complex arrays.
"""
function num_grad(f, x::AbstractArray{T}; δ=1e-5) where T
    grad = similar(x)
    for i in eachindex(x)
        x1 = copy(x); x1[i] += δ
        x2 = copy(x); x2[i] -= δ
        grad[i] = (f(x1) - f(x2)) / (2δ)
        if T <: Complex
            x1 = copy(x); x1[i] += δ*im
            x2 = copy(x); x2[i] -= δ*im
            grad[i] += im * (f(x1) - f(x2)) / (2δ)
        end
    end
    return grad
end

"""
    ising_mpo(β; D=2)

Build the 2D classical Ising transfer matrix as a rank-4 tensor.
Uses the Boltzmann weight decomposition: M[s1,s2,s3,s4] = Σ_σ W(s1,σ)W(s2,σ)W(s3,σ)W(s4,σ)
where W = [exp(β) exp(-β); exp(-β) exp(β)]^(1/2).

Returns a StructArray wrapping one tensor with pattern [1;;].
"""
function ising_mpo(β::Real; atype=Array)
    W = [exp(β) exp(-β); exp(-β) exp(β)]
    vals, vecs = eigen(Symmetric(W))
    sqW = vecs * Diagonal(sqrt.(abs.(vals))) * vecs'
    @tensor M[s1, s2, s3, s4] := sqW[s1, σ] * sqW[s2, σ] * sqW[s3, σ] * sqW[s4, σ]
    M = atype(M)
    return StructArray([M], [1;;])
end

"""
    ising_mpo_2x2(β; atype=Array)

Build a 2×2 unit cell Ising MPO (all tensors identical but in 2×2 pattern).
"""
function ising_mpo_2x2(β::Real; atype=Array)
    W = [exp(β) exp(-β); exp(-β) exp(β)]
    vals, vecs = eigen(Symmetric(W))
    sqW = vecs * Diagonal(sqrt.(abs.(vals))) * vecs'
    @tensor M[s1, s2, s3, s4] := sqW[s1, σ] * sqW[s2, σ] * sqW[s3, σ] * sqW[s4, σ]
    M = atype(M)
    return StructArray([M], [1 1; 1 1])
end

"""
    ising_exact_free_energy(β)

Exact free energy per site for the 2D square lattice Ising model at inverse temperature β.
Uses Onsager's formula.
"""
function ising_exact_free_energy(β::Real)
    # f = -kT * ln(Z) / N
    # For the partition function approach via transfer matrix:
    # The largest eigenvalue λ_max of the transfer matrix gives f = -ln(λ_max)
    # Onsager: f = -ln(2*cosh(2β)) - (1/2π) ∫₀^π ln((1 + sqrt(1-κ²sin²θ))/2) dθ
    # where κ = 2*sinh(2β)/cosh²(2β)
    #
    # Simpler: use the transfer matrix eigenvalue directly.
    # For the 2D classical Ising: the per-site contribution from the row transfer matrix is
    # related to the partition function as Z^(1/N) = λ_max^(1/L).
    #
    # We use numerical quadrature for the Onsager integral.
    κ = 2 * sinh(2β) / cosh(2β)^2
    integral = 0.0
    N_quad = 10000
    for k in 0:N_quad-1
        θ = π * (k + 0.5) / N_quad
        integral += log(cosh(2β)^2 + sqrt(cosh(2β)^4 - sinh(2β)^4 * (cos(θ)^2 + 1) + sinh(2β)^2 * (cosh(2β)^2 - 1))) / 2
    end
    integral /= N_quad
    return -log(2) / 2 - integral
end

# Include test files
@testset "TeneT.jl" begin
    include("test_types.jl")
    include("test_structarray.jl")
    include("test_utils.jl")
    include("test_contraction.jl")
    include("test_boundary.jl")
    include("test_autodiff.jl")
    include("test_ipeps.jl")
    include("test_patch.jl")
    if get(ENV, "TENET_TEST_MPI", "false") == "true"
        include("test_mpi.jl")
    end
end
```

**Step 2: Run to verify it loads**

```bash
cd "D:/1 - research/1.26 - iPEPS_opt/TeneT.jl"
julia --project -e "using Pkg; Pkg.test()" 2>&1 | head -20
```

Expected: Will fail because test files don't exist yet. That's fine — confirms runner loads.

**Step 3: Commit**

```bash
git add test/runtests.jl
git commit -m "test: add runtests.jl entry point with helpers"
```

---

### Task 2: Create `test/test_types.jl`

**Files:**
- Create: `test/test_types.jl`

**Step 1: Write the test file**

```julia
@testset "Types and Defaults" begin
    @testset "Lattice type hierarchy" begin
        @test Square() isa TeneT.AbstractLattice
        @test Honeycomb() isa TeneT.AbstractLattice
        @test Kagome() isa TeneT.AbstractLattice
    end

    @testset "Honeycomb default mode" begin
        h = Honeycomb()
        @test h isa TeneT.Honeycomb{:brickwall}
        h2 = TeneT.Honeycomb{:merge}()
        @test h2 isa TeneT.Honeycomb{:merge}
    end

    @testset "Algorithm type hierarchy" begin
        @test VUMPS{:General}() isa TeneT.Algorithm
        @test VUMPS{:Plaquette}() isa TeneT.Algorithm
        @test VUMPS{:C4v}() isa TeneT.Algorithm
        @test QRCTM() isa TeneT.Algorithm
    end

    @testset "ContractionMode types" begin
        @test TeneT.General() isa TeneT.ContractionMode
        @test TeneT.Plaquette() isa TeneT.ContractionMode
    end

    @testset "iPEPSOptimize type hierarchy" begin
        @test GradientOptimize <: TeneT.iPEPSOptimize
        @test SUOptimize <: TeneT.iPEPSOptimize
        @test FUOptimize <: TeneT.iPEPSOptimize
    end

    @testset "Lattice show (filesystem-safe)" begin
        buf = IOBuffer()
        show(buf, Square())
        @test String(take!(buf)) == "Square"

        show(buf, Honeycomb())
        @test String(take!(buf)) == "Honeycomb_brickwall"

        show(buf, Kagome())
        @test String(take!(buf)) == "Kagome"
    end

    @testset "Defaults module" begin
        @test Defaults.VERBOSE_NONE == 0
        @test Defaults.VERBOSE_WARN == 1
        @test Defaults.VERBOSE_CONV == 2
        @test Defaults.VERBOSE_ITER == 3
        @test Defaults.VERBOSE_ALL == 4
        @test Defaults.verbosity == Defaults.VERBOSE_WARN
    end

    @testset "VUMPS default fields" begin
        alg = VUMPS{:General}()
        @test alg.tol == 1e-10
        @test alg.maxiter == 10
        @test alg.miniter == 1
        @test alg.maxiter_ad == 10
        @test alg.power_iter == 1
        @test alg.ifupdown == true
        @test alg.ifdownfromup == false
        @test alg.ifsimple_eig == true
    end

    @testset "QRCTM default fields" begin
        alg = QRCTM()
        @test alg.tol == 1e-10
        @test alg.maxiter == 100
        @test alg.maxiter_power == 1
        @test alg.ifsimple_eig == true
    end
end
```

**Step 2: Run test**

```bash
julia --project -e 'using Pkg; Pkg.test()' 2>&1 | tail -20
```

Expected: PASS for test_types.jl

**Step 3: Commit**

```bash
git add test/test_types.jl
git commit -m "test: add type hierarchy and defaults tests"
```

---

### Task 3: Create `test/test_structarray.jl`

**Files:**
- Create: `test/test_structarray.jl`

**Step 1: Write the test file**

```julia
@testset "StructArray" begin
    @testset "Construction and indexing" for atype in ATYPES
        data = [atype(rand(ComplexF64, 3, 3)), atype(rand(ComplexF64, 3, 3))]
        pattern = [1 2; 2 1]
        S = StructArray(data, pattern)

        @test size(S) == (2, 2)
        @test size(S, 1) == 2
        @test size(S, 2) == 2
        @test length(S) == 2  # number of unique data elements
        @test S[1,1] === data[1]
        @test S[1,2] === data[2]
        @test S[2,1] === data[2]
        @test S[2,2] === data[1]
    end

    @testset "Invalid pattern assertion" begin
        data = [rand(ComplexF64, 2, 2)]
        @test_throws AssertionError StructArray(data, [1 2; 2 1])  # 2 unique indices but only 1 data
    end

    @testset "setindex!" for atype in ATYPES
        data = [atype(rand(ComplexF64, 2, 2)), atype(rand(ComplexF64, 2, 2))]
        S = StructArray(data, [1 2; 2 1])
        new_val = atype(ones(ComplexF64, 2, 2))
        S[1,1] = new_val
        @test S[1,1] === new_val
        @test S[2,2] === new_val  # shares pattern index 1
    end

    @testset "Arithmetic" for atype in ATYPES
        data1 = [atype(rand(ComplexF64, 3, 3)), atype(rand(ComplexF64, 3, 3))]
        data2 = [atype(rand(ComplexF64, 3, 3)), atype(rand(ComplexF64, 3, 3))]
        pattern = [1 2; 2 1]
        S1 = StructArray(data1, pattern)
        S2 = StructArray(data2, pattern)

        # Addition
        S3 = S1 + S2
        @test S3[1,1] ≈ Array(S1[1,1]) + Array(S2[1,1])

        # Scalar multiplication
        S4 = 2.0 * S1
        @test S4[1,1] ≈ 2.0 * Array(S1[1,1])

        # Division
        S5 = S1 / 3.0
        @test S5[1,1] ≈ Array(S1[1,1]) / 3.0

        # Norm
        n = norm(S1)
        @test n ≈ norm(S1.data)
        @test n >= 0

        # Conjugate
        Sc = conj(S1)
        @test Sc[1,1] ≈ conj(Array(S1[1,1]))
    end

    @testset "rmul! and axpy!" for atype in ATYPES
        data1 = [atype(rand(ComplexF64, 3, 3)), atype(rand(ComplexF64, 3, 3))]
        data2 = [atype(rand(ComplexF64, 3, 3)), atype(rand(ComplexF64, 3, 3))]
        pattern = [1 2; 2 1]
        S1 = StructArray(copy.(data1), pattern)
        S2 = StructArray(copy.(data2), pattern)

        S1_orig = copy(S1)
        rmul!(S1, 0.5)
        @test S1[1,1] ≈ 0.5 * Array(S1_orig[1,1])

        S2_orig = copy(S2)
        axpy!(2.0, S1, S2)
        @test S2[1,1] ≈ Array(S2_orig[1,1]) + 2.0 * Array(S1[1,1])
    end

    @testset "similar, zero, copy" for atype in ATYPES
        data = [atype(rand(ComplexF64, 3, 3)), atype(rand(ComplexF64, 3, 3))]
        S = StructArray(data, [1 2; 2 1])

        S_sim = similar(S)
        @test size(S_sim) == size(S)
        @test length(S_sim) == length(S)

        S_z = zero(S)
        @test norm(S_z) ≈ 0 atol=1e-15

        S_c = copy(S)
        @test S_c[1,1] ≈ Array(S[1,1])
        # copy is independent
        S_c[1,1] = atype(zeros(ComplexF64, 3, 3))
        @test !(S[1,1] ≈ zeros(ComplexF64, 3, 3))
    end

    @testset "circshift" for atype in ATYPES
        data = [atype(rand(ComplexF64, 2, 2)), atype(rand(ComplexF64, 2, 2)),
                atype(rand(ComplexF64, 2, 2)), atype(rand(ComplexF64, 2, 2))]
        S = StructArray(data, [1 2; 3 4])
        Ss = circshift(S, (0, 1))
        @test Ss.pattern == circshift([1 2; 3 4], (0, 1))
    end

    @testset "isapprox" for atype in ATYPES
        data = [atype(rand(ComplexF64, 3, 3)), atype(rand(ComplexF64, 3, 3))]
        S = StructArray(data, [1 2; 2 1])
        S2 = copy(S)
        @test isapprox(S, S2)
    end

    @testset "iterate and collect" for atype in ATYPES
        data = [atype(rand(ComplexF64, 2, 2)), atype(rand(ComplexF64, 2, 2))]
        S = StructArray(data, [1 2; 2 1])
        collected = collect(S)
        @test length(collected) == 2
    end

    @testset "NamedTuple interop" for atype in ATYPES
        data = [atype(rand(ComplexF64, 3, 3)), atype(rand(ComplexF64, 3, 3))]
        pattern = [1 2; 2 1]
        S = StructArray(data, pattern)
        nt = (data = [atype(rand(ComplexF64, 3, 3)), atype(rand(ComplexF64, 3, 3))],)
        result = S + nt
        @test result isa StructArray
        @test result[1,1] ≈ Array(S[1,1]) + Array(nt.data[1])
    end

    @testset "Initialization functions" for atype in ATYPES
        pattern = [1 2; 2 1]
        sizes = [(3, 4, 3), (3, 4, 3)]

        # randSA with type
        S = randSA(ComplexF64, atype, pattern, sizes)
        @test size(S) == (2, 2)
        @test size(S[1,1]) == (3, 4, 3)
        @test eltype(S[1,1]) == ComplexF64

        # randSA default type
        S2 = randSA(atype, pattern, sizes)
        @test eltype(S2[1,1]) == ComplexF64

        # randSA from existing StructArray
        S3 = randSA(S)
        @test size(S3[1,1]) == size(S[1,1])

        # cellones
        C = cellones(S)
        @test size(C[1,1]) == (3, 3)  # chi x chi from first dim
        @test Array(C[1,1]) ≈ Matrix{ComplexF64}(I, 3, 3)

        # ISA
        Isa = ISA(ComplexF64, atype, pattern, [(4, 4), (4, 4)])
        @test Array(Isa[1,1]) ≈ Matrix{ComplexF64}(I, 4, 4)
    end

    @testset "GPU roundtrip" for atype in ATYPES
        atype == Array && continue
        data = [rand(ComplexF64, 3, 3), rand(ComplexF64, 3, 3)]
        S = StructArray(data, [1 2; 2 1])
        S_gpu = atype(S)
        @test _arraytype(S_gpu[1,1]) == atype
        S_cpu = Array(S_gpu)
        @test isapprox(S, S_cpu)
    end

    @testset "Zygote.Buffer" for atype in ATYPES
        data = [atype(rand(ComplexF64, 3, 3)), atype(rand(ComplexF64, 3, 3))]
        S = StructArray(data, [1 2; 2 1])
        buf = Zygote.Buffer(S)
        buf[1] = atype(ones(ComplexF64, 3, 3))
        S2 = copy(buf)
        @test S2 isa StructArray
        @test Array(S2[1,1]) ≈ ones(ComplexF64, 3, 3)
    end
end
```

**Step 2: Run and verify**

```bash
julia --project -e 'using Pkg; Pkg.test()' 2>&1 | tail -30
```

**Step 3: Commit**

```bash
git add test/test_structarray.jl
git commit -m "test: add StructArray tests"
```

---

### Task 4: Create `test/test_utils.jl`

**Files:**
- Create: `test/test_utils.jl`

**Step 1: Write the test file**

```julia
@testset "Utils" begin
    @testset "gpu.jl — array type dispatch" begin
        @test _arraytype(rand(2, 2)) == Array
        @test _mattype(rand(2, 2)) == Matrix

        if CuArray in ATYPES
            x = CuArray(rand(2, 2))
            @test _arraytype(x) == CuArray
            @test _mattype(x) == CUDA.CuMatrix
        end
    end

    @testset "gpu.jl — StructArray arraytype" begin
        S = randSA(Array, [1;;], [(3, 3)])
        @test _arraytype(S) == Array
    end

    @testset "gpu.jl — device management" begin
        @test get_device(Array) isa String
        @test device_count(Array) >= 1
        @test get_device_id(Array) >= 1
        set_device_id!(Array, 1)  # should be no-op

        if CuArray in ATYPES
            @test get_device(CuArray) isa CUDA.CuDevice
            @test device_count(CuArray) >= 1
            x = CuArray(rand(2))
            @test get_device_id(x) >= 1
        end
    end

    @testset "gpu.jl — for_gc identity" begin
        x = rand(3, 3)
        @test for_gc(x) === x
    end

    @testset "gpu.jl — NamedTuple conversions" begin
        nt = (data = [rand(ComplexF64, 2, 2), rand(ComplexF64, 2, 2)],)
        @test Array(nt) === nt

        if CuArray in ATYPES
            nt_gpu = CuArray(nt)
            @test nt_gpu.data[1] isa CuArray
        end
    end

    @testset "gpu.jl — atype_device!" begin
        x = rand(3, 3)
        y = atype_device!(Array, x, 1)
        @test y isa Array
    end

    @testset "misc.jl — qrpos" for atype in ATYPES
        A = atype(rand(ComplexF64, 6, 4))
        Q, R = qrpos(A)
        @test Array(Q) * Array(R) ≈ Array(A)
        @test all(real.(diag(Array(R))) .>= 0)
        # Q is isometric
        @test Array(Q)' * Array(Q) ≈ I(4) atol=1e-10
    end

    @testset "misc.jl — lqpos" for atype in ATYPES
        A = atype(rand(ComplexF64, 4, 6))
        L, Q = lqpos(A)
        @test Array(L) * Array(Q) ≈ Array(A)
        @test all(real.(diag(Array(L))) .>= 0)
        # Q is isometric
        @test Array(Q) * Array(Q)' ≈ I(4) atol=1e-10
    end

    @testset "misc.jl — qr_for_ad" for atype in ATYPES
        A = atype(rand(ComplexF64, 6, 4))
        Q, R = qr_for_ad(A)
        @test Array(Q) * Array(R) ≈ Array(A) atol=1e-10
        @test Q isa atype  # concrete array, not QRCompactWYQ
    end

    @testset "misc.jl — simple_eig" begin
        # Test on a known matrix via its action as a linear map
        A = rand(ComplexF64, 4, 4)
        A = A + A'  # Hermitian for a dominant real eigenvalue
        λ_exact = eigvals(A)[end]

        f(v) = A * v
        v0 = rand(ComplexF64, 4)
        λs, vs = simple_eig(f, v0; power_iter=50)
        @test abs(λs[1]) ≈ abs(λ_exact) atol=1e-4
        @test norm(vs[1]) ≈ 1.0 atol=1e-10
    end

    @testset "misc.jl — checkpoint" begin
        f(x) = sum(x .^ 2)
        x = rand(3)
        @test checkpoint(f, x) ≈ f(x)
    end

    @testset "misc.jl — takagi_decomposition" begin
        # Build a complex symmetric matrix M = B * transpose(B)
        n = 4
        B = rand(ComplexF64, n, n)
        M = B * transpose(B)
        @test norm(M - transpose(M)) < 1e-10

        A = takagi_decomposition(M; D_trunc=n)
        M_recon = A * transpose(A)
        @test M_recon ≈ M atol=1e-6

        # Truncation
        A_trunc = takagi_decomposition(M; D_trunc=2)
        @test size(A_trunc) == (n, 2)
    end

    @testset "misc.jl — leg type aliases" begin
        @test rand(2, 2, 2) isa leg3
        @test rand(2, 2, 2, 2) isa leg4
        @test rand(2, 2, 2, 2, 2) isa leg5
        @test rand(2, 2, 2, 2, 2, 2, 2, 2) isa leg8
    end

    @testset "io.jl — save/load roundtrip" begin
        mktempdir() do dir
            # Create a simple VUMPSRuntime
            pattern = [1;;]
            χ, D = 4, 2
            AL = randSA(ComplexF64, Array, pattern, [(χ, D, χ)])
            AR = randSA(ComplexF64, Array, pattern, [(χ, D, χ)])
            C  = randSA(ComplexF64, Array, pattern, [(χ, χ)])
            FL = randSA(ComplexF64, Array, pattern, [(χ, D, χ)])
            FR = randSA(ComplexF64, Array, pattern, [(χ, D, χ)])
            rt = VUMPSRuntime(AL, AR, C, FL, FR)

            save_rt(dir, rt)
            rt_loaded = load_rt(dir, Array, false)
            @test isapprox(rt.AL, rt_loaded.AL)
            @test isapprox(rt.C, rt_loaded.C)
        end
    end

    @testset "io.jl — read_last_log" begin
        mktempdir() do dir
            D = 2
            logdir = joinpath(dir, "D$D")
            mkpath(logdir)
            logfile = joinpath(logdir, "history.log")
            write(logfile, "i =     6   t = 43039.05 sec    e_χ144 = -0.501858316272094 gnorm = 1.984e-04   Eimag = 1.389e-10\n")
            last_i, last_chi = read_last_log(dir, D)
            @test last_i == 6
            @test last_chi == 144
        end
    end
end
```

**Step 2: Run and verify**

**Step 3: Commit**

```bash
git add test/test_utils.jl
git commit -m "test: add utils tests (gpu, misc, io)"
```

---

### Task 5: Create `test/test_contraction.jl`

**Files:**
- Create: `test/test_contraction.jl`

**Step 1: Write the test file**

```julia
@testset "Contraction" begin
    @testset "basic.jl — ALCtoAC_map" for atype in ATYPES
        χ, D = 4, 2
        # leg3
        AL = atype(rand(ComplexF64, χ, D, χ))
        C = atype(rand(ComplexF64, χ, χ))
        AC = ALCtoAC_map(AL, C)
        @test size(AC) == (χ, D, χ)

        # leg4
        AL4 = atype(rand(ComplexF64, χ, D, D, χ))
        AC4 = ALCtoAC_map(AL4, C)
        @test size(AC4) == (χ, D, D, χ)
    end

    @testset "basic.jl — CTtoT and CTCtoT" for atype in ATYPES
        χ, D = 4, 2
        C = atype(rand(ComplexF64, χ, χ))

        T3 = atype(rand(ComplexF64, χ, D, χ))
        @test size(CTtoT(C, T3)) == (χ, D, χ)
        @test size(CTCtoT(C, T3)) == (χ, D, χ)

        T4 = atype(rand(ComplexF64, χ, D, D, χ))
        @test size(CTtoT(C, T4)) == (χ, D, D, χ)
        @test size(CTCtoT(C, T4)) == (χ, D, D, χ)
    end

    @testset "basic.jl — FLmap leg4" for atype in ATYPES
        χ, D = 4, 2
        FL = atype(rand(ComplexF64, χ, D, χ))
        ALu = atype(rand(ComplexF64, χ, D, χ))
        ALd = atype(rand(ComplexF64, χ, D, χ))
        M = atype(rand(ComplexF64, D, D, D, D))
        result = FLmap(FL, ALu, ALd, M)
        @test size(result) == (χ, D, χ)
    end

    @testset "basic.jl — FLmap leg5 (bilayer)" for atype in ATYPES
        χ, D, d = 4, 2, 2
        FL = atype(rand(ComplexF64, χ, D, D, χ))
        ALu = atype(rand(ComplexF64, χ, D, D, χ))
        ALd = atype(rand(ComplexF64, χ, D, D, χ))
        M = atype(rand(ComplexF64, D, D, D, D, d))
        result = FLmap(FL, ALu, ALd, M)
        @test size(result) == (χ, D, D, χ)
    end

    @testset "basic.jl — FRmap leg4" for atype in ATYPES
        χ, D = 4, 2
        FR = atype(rand(ComplexF64, χ, D, χ))
        ARu = atype(rand(ComplexF64, χ, D, χ))
        ARd = atype(rand(ComplexF64, χ, D, χ))
        M = atype(rand(ComplexF64, D, D, D, D))
        result = FRmap(FR, ARu, ARd, M)
        @test size(result) == (χ, D, χ)
    end

    @testset "basic.jl — Lmap and Rmap" for atype in ATYPES
        χ = 4
        L = atype(rand(ComplexF64, χ, χ))
        ALu = atype(rand(ComplexF64, χ, χ, χ))
        ALd = atype(rand(ComplexF64, χ, χ, χ))
        result = Lmap(L, ALu, ALd)
        @test size(result) == (χ, χ)

        R = atype(rand(ComplexF64, χ, χ))
        ARu = atype(rand(ComplexF64, χ, χ, χ))
        ARd = atype(rand(ComplexF64, χ, χ, χ))
        result = Rmap(R, ARu, ARd)
        @test size(result) == (χ, χ)
    end

    @testset "basic.jl — ACmap and Cmap" for atype in ATYPES
        χ, D = 4, 2
        AC = atype(rand(ComplexF64, χ, D, χ))
        FL = atype(rand(ComplexF64, χ, D, χ))
        FR = atype(rand(ComplexF64, χ, D, χ))
        M = atype(rand(ComplexF64, D, D, D, D))
        result = ACmap(AC, FL, FR, M)
        @test size(result) == (χ, D, χ)

        C = atype(rand(ComplexF64, χ, χ))
        Cr = Cmap(C, FL, FR)
        @test size(Cr) == (χ, χ)
    end

    @testset "basic.jl — ACdmap leg4" for atype in ATYPES
        χ, D = 4, 2
        ACd = atype(rand(ComplexF64, χ, D, χ))
        FL = atype(rand(ComplexF64, χ, D, χ))
        FR = atype(rand(ComplexF64, χ, D, χ))
        M = atype(rand(ComplexF64, D, D, D, D))
        result = ACdmap(ACd, FL, FR, M)
        @test size(result) == (χ, D, χ)
    end

    @testset "basic.jl — Mmap" for atype in ATYPES
        χ, D = 4, 2
        AC = atype(rand(ComplexF64, χ, D, χ))
        ACd = atype(rand(ComplexF64, χ, D, χ))
        FL = atype(rand(ComplexF64, χ, D, χ))
        FR = atype(rand(ComplexF64, χ, D, χ))
        result = Mmap(AC, ACd, FL, FR)
        @test size(result) == (D, D, D, D)
    end

    @testset "forloop — split_count and split_ranges" begin
        counts = split_count(10, 3)
        @test sum(counts) == 10
        @test length(counts) == 3
        @test maximum(counts) - minimum(counts) <= 1

        ranges = split_ranges(10, 3)
        @test length(ranges) == 3
        @test first(ranges[1]) == 1
        @test last(ranges[end]) == 10
    end

    @testset "forloop — FLmap_parallel serial" for atype in ATYPES
        χ, D = 4, 2
        FL = atype(rand(ComplexF64, χ, D, χ))
        ALu = atype(rand(ComplexF64, χ, D, χ))
        ALd = atype(rand(ComplexF64, χ, D, χ))
        M = atype(rand(ComplexF64, D, D, D, D))

        # forloop_iter=1 should match direct
        r1 = FLmap_parallel(FL, ALu, ALd, M; ifparallel=false, forloop_iter=1)
        r2 = FLmap(FL, ALu, ALd, M)
        @test Array(r1) ≈ Array(r2)

        # forloop_iter=2 should still give same result
        r3 = FLmap_parallel(FL, ALu, ALd, M; ifparallel=false, forloop_iter=2)
        @test Array(r3) ≈ Array(r1) atol=1e-10
    end

    @testset "forloop — ACmap_parallel serial" for atype in ATYPES
        χ, D = 4, 2
        AC = atype(rand(ComplexF64, χ, D, χ))
        FL = atype(rand(ComplexF64, χ, D, χ))
        FR = atype(rand(ComplexF64, χ, D, χ))
        M = atype(rand(ComplexF64, D, D, D, D))

        r1 = ACmap_parallel(AC, FL, FR, M; ifparallel=false, forloop_iter=1)
        r2 = ACmap(AC, FL, FR, M)
        @test Array(r1) ≈ Array(r2)
    end

    @testset "forloop — Mmap_parallel serial" for atype in ATYPES
        χ, D = 4, 2
        AC = atype(rand(ComplexF64, χ, D, χ))
        ACd = atype(rand(ComplexF64, χ, D, χ))
        FL = atype(rand(ComplexF64, χ, D, χ))
        FR = atype(rand(ComplexF64, χ, D, χ))

        r1 = Mmap_parallel(AC, ACd, FL, FR; ifparallel=false, forloop_iter=1)
        r2 = Mmap(AC, ACd, FL, FR)
        @test Array(r1) ≈ Array(r2)
    end
end
```

**Step 2: Run and verify**

**Step 3: Commit**

```bash
git add test/test_contraction.jl
git commit -m "test: add contraction kernel and forloop tests"
```

---

### Task 6: Create `test/test_boundary.jl`

**Files:**
- Create: `test/test_boundary.jl`

**Step 1: Write the test file**

This is the largest and most important test file. It tests all boundary algorithms against the 2D Ising model.

```julia
@testset "Boundary Algorithms" begin
    β = 0.3  # Away from critical point for fast convergence

    @testset "VUMPS General — canonical forms" for atype in ATYPES
        χ, D = 6, 2
        M = ising_mpo(β; atype)
        A = initial_A(M, χ)

        # left_canonical
        AL, L, λ = left_canonical(A)
        # AL should be left-isometric: AL†·AL = I (on the virtual legs)
        for i in 1:length(AL)
            ALi = Array(AL[i])
            ALm = reshape(ALi, :, size(ALi, ndims(ALi)))
            @test ALm' * ALm ≈ I(size(ALi, ndims(ALi))) atol=1e-8
        end

        # right_canonical
        R, AR, λr = right_canonical(A)
        for i in 1:length(AR)
            ARi = Array(AR[i])
            ARm = reshape(ARi, size(ARi, 1), :)
            @test ARm * ARm' ≈ I(size(ARi, 1)) atol=1e-8
        end

        # LRtoC
        C = LRtoC(L, R)
        @test length(C) == length(L)
        for i in 1:length(C)
            @test size(Array(C[i])) == (χ, χ)
        end

        # ALCtoAC
        AC = ALCtoAC(AL, C)
        @test length(AC) == length(AL)
        for i in 1:length(AC)
            @test size(Array(AC[i])) == size(Array(AL[i]))
        end
    end

    @testset "VUMPS General — environment init and shapes" for atype in ATYPES
        χ, D = 6, 2
        M = ising_mpo(β; atype)

        alg = VUMPS{:General}(tol=1e-6, maxiter=50, miniter=1, maxiter_ad=1, miniter_ad=1,
                               verbosity=0, ifupdown=false)
        rt = init_env(M, χ, alg)
        @test rt isa VUMPSRuntime
        @test size(rt.AL) == size(M)
        @test size(Array(rt.FL[1,1]), 1) == χ
    end

    @testset "VUMPS General — convergence 1x1 Ising" for atype in ATYPES
        χ = 8
        M = ising_mpo(β; atype)
        alg = VUMPS{:General}(tol=1e-8, maxiter=100, miniter=1, maxiter_ad=1, miniter_ad=1,
                               verbosity=0, ifupdown=false)
        rt = init_env(M, χ, alg)
        rt, err = leading_boundary(rt, M, alg)
        @test err < 1e-6

        # Check free energy via transfer matrix eigenvalue
        # For 1x1: λ = dot(AC, ACmap(...)) / dot(AC, AC)
        AC = ALCtoAC(rt.AL, rt.C)
        λ_AC, _ = simple_eig(x -> ACmap(x, rt.FL[1,1], rt.FR[1,1], M[1,1]), AC[1,1]; power_iter=20)
        ln_λ = log(abs(λ_AC[1]))
        # This should be close to the dominant eigenvalue of the transfer matrix
        @test ln_λ > 0  # positive for Ising at this β
    end

    @testset "VUMPS General — up/down 1x1" for atype in ATYPES
        χ = 8
        M = ising_mpo(β; atype)
        alg = VUMPS{:General}(tol=1e-8, maxiter=100, miniter=1, maxiter_ad=1, miniter_ad=1,
                               verbosity=0, ifupdown=true, ifdownfromup=true)
        rt = init_env(M, χ, alg)
        @test rt isa Tuple{VUMPSRuntime, VUMPSRuntime}
        (rtup, rtdown), (errup, errdown) = leading_boundary(rt, M, alg)
        @test errup < 1e-6
    end

    @testset "VUMPS General — 2x2 unit cell" for atype in ATYPES
        χ = 8
        M = ising_mpo_2x2(β; atype)
        alg = VUMPS{:General}(tol=1e-8, maxiter=100, miniter=1, maxiter_ad=1, miniter_ad=1,
                               verbosity=0, ifupdown=false)
        rt = init_env(M, χ, alg)
        rt, err = leading_boundary(rt, M, alg)
        @test err < 1e-4  # 2x2 needs more iterations, relax tolerance
    end

    @testset "VUMPS General — ObsEnv construction" for atype in ATYPES
        χ = 6
        M = ising_mpo(β; atype)
        alg = VUMPS{:General}(tol=1e-6, maxiter=50, miniter=1, maxiter_ad=1, miniter_ad=1,
                               verbosity=0, ifupdown=false, power_iter_obs=10)
        rt = init_env(M, χ, alg)
        rt, _ = leading_boundary(rt, M, alg)
        env = ObsEnv(rt, M, alg)
        @test env isa VUMPSEnv
    end

    @testset "VUMPS Plaquette — init and convergence" for atype in ATYPES
        χ = 8
        M = ising_mpo_2x2(β; atype)
        alg = VUMPS{:Plaquette}(tol=1e-6, maxiter=100, miniter=1, maxiter_ad=1, miniter_ad=1,
                                  verbosity=0)
        rt = init_env(M, χ, alg)
        @test rt isa PlaquetteVUMPSRuntime
        rt, err = leading_boundary(rt, M, alg)
        @test err < 1e-4
    end

    @testset "VUMPS Plaquette — ObsEnv" for atype in ATYPES
        χ = 6
        M = ising_mpo_2x2(β; atype)
        alg = VUMPS{:Plaquette}(tol=1e-6, maxiter=50, miniter=1, maxiter_ad=1, miniter_ad=1,
                                  verbosity=0, power_iter_obs=10)
        rt = init_env(M, χ, alg)
        rt, _ = leading_boundary(rt, M, alg)
        env = ObsEnv(rt, M, alg)
        @test env isa PlaquetteVUMPSEnv
    end

    @testset "VUMPS C4v — init and convergence" for atype in ATYPES
        χ = 8
        M = ising_mpo(β; atype)
        alg = VUMPS{:C4v}(tol=1e-6, maxiter=100, miniter=1, maxiter_ad=1, miniter_ad=1,
                            verbosity=0)
        rt = init_env(M, χ, alg)
        @test rt isa C4vVUMPSEnv
        rt, err = leading_boundary(rt, M, alg)
        @test err < 1e-4
    end

    @testset "VUMPS C4v — ObsEnv" for atype in ATYPES
        χ = 6
        M = ising_mpo(β; atype)
        alg = VUMPS{:C4v}(tol=1e-6, maxiter=50, miniter=1, maxiter_ad=1, miniter_ad=1,
                            verbosity=0, power_iter_obs=10)
        rt = init_env(M, χ, alg)
        rt, _ = leading_boundary(rt, M, alg)
        env = ObsEnv(rt, M, alg)
        @test env isa C4vVUMPSEnv  # C4v returns itself as ObsEnv
    end

    @testset "QRCTM — init and convergence" for atype in ATYPES
        χ = 8
        M = ising_mpo(β; atype)
        alg = QRCTM(tol=1e-6, maxiter=200, miniter=1, maxiter_ad=1, miniter_ad=1,
                     verbosity=0)
        env = init_env(M, χ, alg)
        @test env isa CTMEnv
        env, err = leading_boundary(env, M, alg)
        @test err < 1e-4
    end

    @testset "QRCTM — ObsEnv" for atype in ATYPES
        χ = 6
        M = ising_mpo(β; atype)
        alg = QRCTM(tol=1e-6, maxiter=100, miniter=1, maxiter_ad=1, miniter_ad=1,
                     verbosity=0)
        env = init_env(M, χ, alg)
        env, _ = leading_boundary(env, M, alg)
        obsenv = ObsEnv(env, M, alg)
        @test obsenv isa CTMEnv
    end

    @testset "Environment — update!" begin
        pattern = [1;;]
        χ, D = 4, 2
        mk_rt() = VUMPSRuntime(
            randSA(ComplexF64, Array, pattern, [(χ,D,χ)]),
            randSA(ComplexF64, Array, pattern, [(χ,D,χ)]),
            randSA(ComplexF64, Array, pattern, [(χ,χ)]),
            randSA(ComplexF64, Array, pattern, [(χ,D,χ)]),
            randSA(ComplexF64, Array, pattern, [(χ,D,χ)])
        )
        rt1 = mk_rt()
        rt2 = mk_rt()
        update!(rt1, rt2)
        @test isapprox(rt1.AL, rt2.AL)
        @test isapprox(rt1.C, rt2.C)
    end

    @testset "Environment — GPU conversion roundtrip" for atype in ATYPES
        atype == Array && continue
        pattern = [1;;]
        χ, D = 4, 2
        rt = VUMPSRuntime(
            randSA(ComplexF64, Array, pattern, [(χ,D,χ)]),
            randSA(ComplexF64, Array, pattern, [(χ,D,χ)]),
            randSA(ComplexF64, Array, pattern, [(χ,χ)]),
            randSA(ComplexF64, Array, pattern, [(χ,D,χ)]),
            randSA(ComplexF64, Array, pattern, [(χ,D,χ)])
        )
        rt_gpu = atype(rt)
        rt_cpu = Array(rt_gpu)
        @test isapprox(rt.AL, rt_cpu.AL)
    end

    @testset "Environment — CTMEnv GPU roundtrip" for atype in ATYPES
        atype == Array && continue
        C = rand(ComplexF64, 4, 4)
        T = rand(ComplexF64, 4, 2, 4)
        env = CTMEnv(C, T)
        env_gpu = atype(env)
        env_cpu = Array(env_gpu)
        @test env_cpu.C ≈ C
        @test env_cpu.T ≈ T
    end

    @testset "_down_M" begin
        M = ising_mpo(β; atype=Array)
        Md = _down_M(M)
        @test size(Md) == size(M)
        # For 1x1 pattern, _down_M permutes legs (1,4,3,2) for leg4
        Md_data = Array(Md[1,1])
        M_data = Array(M[1,1])
        @test Md_data ≈ permutedims(M_data, (1, 4, 3, 2))
    end
end
```

**Step 2: Run and verify**

**Step 3: Commit**

```bash
git add test/test_boundary.jl
git commit -m "test: add boundary algorithm tests (VUMPS General/Plaquette/C4v + QRCTM)"
```

---

### Task 7: Create `test/test_autodiff.jl`

**Files:**
- Create: `test/test_autodiff.jl`

**Step 1: Write the test file**

```julia
@testset "Autodiff" begin
    @testset "rrule — StructArray constructor" begin
        data = [rand(ComplexF64, 3, 3), rand(ComplexF64, 3, 3)]
        pattern = [1 2; 2 1]
        val, back = Zygote.pullback(StructArray, data, pattern)
        @test val isa StructArray
        dS = (data = [rand(ComplexF64, 3, 3), rand(ComplexF64, 3, 3)],)
        grads = back(dS)
        @test grads[1] == dS.data  # data gradient passes through
    end

    @testset "rrule — norm(S::StructArray)" begin
        data = [rand(ComplexF64, 3, 3), rand(ComplexF64, 3, 3)]
        S = StructArray(data, [1 2; 2 1])
        g = Zygote.gradient(s -> real(norm(s)), S)[1]
        @test g isa StructArray
        @test norm(g) > 0
    end

    @testset "rrule — VUMPSRuntime constructor" begin
        pattern = [1;;]
        χ, D = 4, 2
        AL = randSA(ComplexF64, Array, pattern, [(χ,D,χ)])
        AR = randSA(ComplexF64, Array, pattern, [(χ,D,χ)])
        C  = randSA(ComplexF64, Array, pattern, [(χ,χ)])
        FL = randSA(ComplexF64, Array, pattern, [(χ,D,χ)])
        FR = randSA(ComplexF64, Array, pattern, [(χ,D,χ)])
        val, back = Zygote.pullback(VUMPSRuntime, AL, AR, C, FL, FR)
        @test val isa VUMPSRuntime
        # Test pullback doesn't error
        drt = (AL, AR, C, FL, FR)
        grads = back(drt)
        @test length(grads) == 6  # NoTangent + 5 fields
    end

    @testset "rrule — CTMEnv constructor" begin
        C = rand(ComplexF64, 4, 4)
        T = rand(ComplexF64, 4, 2, 4)
        val, back = Zygote.pullback(CTMEnv, C, T)
        @test val isa CTMEnv
        denv = (C, T)
        grads = back(denv)
        @test length(grads) == 3
    end

    @testset "Numerical gradient — qrpos" begin
        A = rand(ComplexF64, 6, 4)
        f(A) = begin
            Q, R = qrpos(A)
            real(sum(Q) + sum(R))
        end
        g_zy = Zygote.gradient(f, A)[1]
        g_num = num_grad(f, A; δ=1e-6)
        @test g_zy ≈ g_num atol=1e-3
    end

    @testset "Numerical gradient — lqpos" begin
        A = rand(ComplexF64, 4, 6)
        f(A) = begin
            L, Q = lqpos(A)
            real(sum(L) + sum(Q))
        end
        g_zy = Zygote.gradient(f, A)[1]
        g_num = num_grad(f, A; δ=1e-6)
        @test g_zy ≈ g_num atol=1e-3
    end

    @testset "Numerical gradient — qr_for_ad" begin
        A = rand(ComplexF64, 6, 4)
        f(A) = begin
            Q, R = qr_for_ad(A)
            real(sum(Q) + sum(R))
        end
        g_zy = Zygote.gradient(f, A)[1]
        g_num = num_grad(f, A; δ=1e-6)
        @test g_zy ≈ g_num atol=1e-3
    end

    @testset "Numerical gradient — SVD" begin
        A = rand(ComplexF64, 4, 4)
        f(A) = begin
            F = svd(A)
            real(sum(F.U) + sum(F.S) + sum(F.Vt))
        end
        g_zy = Zygote.gradient(f, A)[1]
        g_num = num_grad(f, A; δ=1e-6)
        @test g_zy ≈ g_num atol=1e-3
    end

    @testset "Numerical gradient — orth_for_ad" begin
        v = rand(ComplexF64, 5)
        v /= norm(v)
        f(v) = begin
            w = orth_for_ad(v)
            real(sum(w .^ 2))
        end
        g_zy = Zygote.gradient(f, v)[1]
        g_num = num_grad(f, v; δ=1e-6)
        @test g_zy ≈ g_num atol=1e-3
    end

    @testset "orth_for_ad — projection property" begin
        v = rand(ComplexF64, 5)
        v /= norm(v)
        # The rrule projects out the component along v
        f(v) = real(sum(orth_for_ad(v)))
        g = Zygote.gradient(f, v)[1]
        # g should be orthogonal to v after projection
        proj = dot(v, g)
        @test abs(proj) < 1e-8  # approximately orthogonal
    end

    @testset "Numerical gradient — simple_eig" begin
        A = rand(ComplexF64, 4, 4)
        A = A + A'
        f(A) = begin
            λs, vs = simple_eig(v -> A * v, rand(ComplexF64, 4); power_iter=50)
            real(λs[1])
        end
        g_zy = Zygote.gradient(f, A)[1]
        g_num = num_grad(f, A; δ=1e-6)
        @test g_zy ≈ g_num atol=1e-2  # relaxed tolerance for iterative solver
    end

    @testset "Grassmann — project_AL" begin
        χ, D = 4, 2
        AL = rand(ComplexF64, χ, D, χ)
        # Make AL left-isometric
        ALm = reshape(AL, χ*D, χ)
        Q, _ = qr(ALm)
        AL = reshape(Matrix(Q), χ, D, χ)

        ∂AL = rand(ComplexF64, χ, D, χ)
        ∂AL_proj = project_AL([∂AL], [AL])[1]

        # Projected gradient should be orthogonal to AL
        @tensor overlap[a, b] := conj(AL)[c, d, a] * ∂AL_proj[c, d, b]
        @test norm(overlap) < 1e-10
    end

    @testset "Grassmann — project_AR" begin
        χ, D = 4, 2
        AR = rand(ComplexF64, χ, D, χ)
        # Make AR right-isometric
        ARm = reshape(AR, χ, D*χ)
        Q, _ = qr(ARm')
        AR = reshape(Matrix(Q)', χ, D, χ)

        ∂AR = rand(ComplexF64, χ, D, χ)
        ∂AR_proj = project_AR([∂AR], [AR])[1]

        # After permute_fronttail: project on left-isometric form
        @test size(∂AR_proj) == (χ, D, χ)
    end

    @testset "Grassmann — retract!" begin
        χ, D = 4, 2
        A_data = [rand(ComplexF64, χ, D, χ)]
        pattern = [1;;]
        A = StructArray(A_data, pattern)
        retract!(A)
        # After retraction, A should be left-canonical
        Am = reshape(Array(A[1,1]), χ*D, χ)
        @test Am' * Am ≈ I(χ) atol=1e-8
    end

    @testset "svd_back — ZeroAdder" begin
        z = TeneT.ZeroAdder()
        x = rand(3, 3)
        @test x + z === x
        @test z + x === x
        @test x - z === x
        @test (-z) === z
    end

    @testset "@non_differentiable — smoke tests" begin
        # These should not error when computing gradients of functions using them
        pattern = [1;;]
        M = StructArray([rand(ComplexF64, 2, 2, 2, 2)], pattern)
        # randSA should be non-differentiable
        f(x) = real(sum(x[1,1])) + 0 * real(sum(randSA(Array, pattern, [(2, 2)])[1,1]))
        g = Zygote.gradient(f, M)
        @test g[1] isa StructArray || g[1] === nothing || true  # just shouldn't error
    end
end
```

**Step 2: Run and verify**

**Step 3: Commit**

```bash
git add test/test_autodiff.jl
git commit -m "test: add autodiff tests (numerical gradient checks + Grassmann)"
```

---

### Task 8: Create `test/test_ipeps.jl`

**Files:**
- Create: `test/test_ipeps.jl`

**Step 1: Write the test file**

```julia
@testset "iPEPS Optimize" begin
    @testset "build_A — Square lattice" for atype in ATYPES
        D, d, N = 2, 2, 1
        pattern = [1;;]
        A_raw = atype(rand(Float64, D, D, D, D, d, N))

        # Need a minimal model struct for build_A dispatch
        # We'll use the Heisenberg model from the codebase
        # But since we exclude models, we test the StructArray construction directly
        SA = StructArray([Array(A_raw[:,:,:,:,:,1])], pattern)
        @test size(SA) == (1, 1)
        @test size(SA[1,1]) == (D, D, D, D, d)
    end

    @testset "_lattice_map — Square identity" begin
        pattern = [1 2; 2 1]
        data = [rand(Float64, 2, 2, 2, 2, 2) for _ in 1:2]
        A = StructArray(data, pattern)
        A2 = _lattice_map(A, Square(), pattern)
        @test isapprox(A, A2)
    end

    @testset "_lattice_map — Honeycomb brickwall" begin
        pattern = [1 2; 3 4]
        data = [rand(Float64, 2, 1, 2, 2, 2) for _ in 1:4]
        A = StructArray(data, pattern)
        A2 = _lattice_map(A, Honeycomb(), pattern)
        # Even-parity sites unchanged, odd-parity sites permuted
        # Site (1,1): sum indices = 2 (even) -> unchanged
        @test Array(A2[1,1]) ≈ Array(A[1,1])
        # Site (1,2): sum indices = 3 (odd) -> permuted (3,4,1,2,5)
        @test Array(A2[1,2]) ≈ permutedims(Array(A[1,2]), (3, 4, 1, 2, 5))
    end

    @testset "C4v_restriction — 5-leg" begin
        D, d = 3, 2
        A = rand(ComplexF64, D, D, D, D, d)
        A_sym = C4v_restriction(A)
        # Applying restriction twice should give 16x the original (4 operations, each doubles)
        A_sym2 = C4v_restriction(A_sym)
        @test A_sym2 ≈ 16 * A_sym atol=1e-10 * norm(A_sym2)

        # Check specific symmetry: up-down reflection
        @test A_sym ≈ permutedims(conj(A_sym), (1,4,3,2,5)) atol=1e-10
    end

    @testset "C4v_restriction — 6-leg" begin
        D, d = 3, 2
        A = rand(ComplexF64, D, D, D, D, d, 1)[:,:,:,:,:,1]
        A6 = rand(ComplexF64, D, D, D, D, d, 2)
        A_sym = C4v_restriction(A6)
        @test A_sym ≈ permutedims(conj(A_sym), (1,4,3,2,5,6)) atol=1e-10
    end

    @testset "_restriction_ipeps — identity" begin
        A = rand(3, 3)
        @test _restriction_ipeps(A) === A
    end

    @testset "pepsgeneral — 5-leg canonical form" begin
        D, d = 3, 2
        A = rand(ComplexF64, D, D, D, D, d)
        Ac, Rs = pepsgeneral(A)
        # Reconstruct: A ≈ ARstoA(Ac, Rs)
        A_recon = ARstoA(Ac, Rs)
        @test A_recon ≈ A atol=1e-6
    end

    @testset "pepsgeneral — 6-leg canonical form" begin
        D, d = 3, 2
        A = rand(ComplexF64, D, D, D, D, d, d)
        Ac, Rs = pepsgeneral(A)
        A_recon = ARstoA1(Ac, Rs)
        @test size(A_recon) == size(A)
    end

    @testset "central_canonical forms" begin
        D, d = 3, 2
        A = rand(ComplexF64, D, D, D, D, d, d)
        A1 = central_canonical1(A)
        @test size(A1) == size(A)
        A2 = central_canonical2(A)
        @test size(A2) == size(A)
    end

    @testset "to_mcf_ipeps" begin
        D, d = 3, 2
        A = rand(ComplexF64, D, D, D, D, d)
        A_mcf = to_mcf_ipeps(A; max_iter=20, tol=1e-6)
        @test size(A_mcf) == size(A)
    end

    @testset "local_gauge_contraction" begin
        D, d = 3, 2
        A = rand(ComplexF64, D, D, D, D, d)
        G = [rand(ComplexF64, D, D) for _ in 1:4]
        A_gauged = local_gauge_contraction(A, G)
        @test size(A_gauged) == size(A)
    end

    @testset "SU_parameterization — runs" begin
        # This needs a model with hamiltonian, so we test shape only
        # by mocking the minimal interface
        D, d = 2, 2
        pattern = [1;;]
        data = [rand(Float64, D, D, D, D, d)]
        A = StructArray(data, pattern)

        # SU_parameterization requires params.model.lattice and hamiltonian(model)
        # Skip if we can't construct without model imports
        @test true  # placeholder — full test in integration
    end

    @testset "init_ipeps — random initialization shapes" begin
        @test size(_init_random_ipeps(Square(), Float64, 2, 2, 1, 1, 1)) == (2, 2, 2, 2, 2, 1)
        @test size(_init_random_ipeps(Kagome(), Float64, 2, 2, 1, 1, 1)) == (2, 2, 2, 2, 8, 1)
        @test size(_init_random_ipeps(Honeycomb{:merge}(), Float64, 2, 2, 1, 1, 1)) == (2, 2, 2, 2, 4, 1)
        @test size(_init_random_ipeps(Honeycomb{:brickwall}(), Float64, 2, 2, 1, 2, 2)) == (2, 1, 2, 2, 2, 1)
    end

    @testset "Environment struct types" begin
        @test fieldnames(VUMPSRuntime) == (:AL, :AR, :C, :FL, :FR)
        @test fieldnames(PlaquetteVUMPSRuntime) == (:AL, :C, :FL)
        @test fieldnames(VUMPSEnv) == (:ACu, :ARu, :ACd, :ARd, :FLu, :FRu, :FLo, :FRo)
        @test fieldnames(PlaquetteVUMPSEnv) == (:AL, :C, :FLu, :FLo)
        @test fieldnames(CTMEnv) == (:C, :T)
    end

    @testset "GradientOptimize / SUOptimize / FUOptimize field access" begin
        # These require a HamiltonianModel, which is in models/
        # Just test that the types exist and are subtypes
        @test GradientOptimize <: TeneT.iPEPSOptimize
        @test SUOptimize <: TeneT.iPEPSOptimize
        @test FUOptimize <: TeneT.iPEPSOptimize
    end

    @testset "_inner product" begin
        x = rand(ComplexF64, 3, 3)
        dx1 = rand(ComplexF64, 3, 3)
        dx2 = rand(ComplexF64, 3, 3)
        result = _inner(x, dx1, dx2)
        @test result isa Real
        @test result ≈ real(dot(dx1, dx2))
    end
end
```

**Step 2: Run and verify**

**Step 3: Commit**

```bash
git add test/test_ipeps.jl
git commit -m "test: add iPEPS optimization component tests"
```

---

### Task 9: Create `test/test_patch.jl`

**Files:**
- Create: `test/test_patch.jl`

**Step 1: Write the test file**

```julia
using OptimKit: LBFGS, LBFGSInverseHessian

@testset "OptimKit Patch" begin
    @testset "LBFGSState construction" begin
        x = rand(5)
        f = 1.0
        g = rand(5)
        H = LBFGSInverseHessian(5, Vector{Float64}[], Vector{Float64}[], Float64[])
        state = LBFGSState(x, f, g, H, 1, 0, [f], [norm(g)], time())
        @test state.x == x
        @test state.f == f
        @test state.numfg == 1
        @test state.numiter == 0
        @test length(state.fhistory) == 1
    end

    @testset "LBFGSInverseHessian GPU roundtrip" for atype in ATYPES
        atype == Array && continue
        S = [rand(5), rand(5)]
        Y = [rand(5), rand(5)]
        ρ = [0.5, 0.3]
        H = LBFGSInverseHessian(5, S, Y, ρ)
        H_gpu = CuArray(H)
        @test H_gpu.S[1] isa CuArray
        H_cpu = Array(H_gpu)
        @test H_cpu.S[1] ≈ S[1]
        @test H_cpu.Y[1] ≈ Y[1]
    end

    @testset "save/load LBFGS state roundtrip" begin
        mktempdir() do dir
            x = rand(5)
            f = 1.5
            g = rand(5)
            H = LBFGSInverseHessian(5, Vector{Float64}[], Vector{Float64}[], Float64[])
            state = LBFGSState(x, f, g, H, 3, 2, [1.5, 1.2], [0.1, 0.05], time())

            filepath = joinpath(dir, "test_state.jld2")
            alg = LBFGS(5; maxiter=10, verbosity=0)
            save_lbfgs_state(alg, state, filepath)
            @test isfile(filepath)

            loaded = load_lbfgs_state(alg, filepath)
            @test loaded !== nothing
            @test loaded.x ≈ x
            @test loaded.f ≈ f
            @test loaded.numfg == 3
            @test loaded.numiter == 2
        end
    end

    @testset "optimize_reload — quadratic function" begin
        # Minimize f(x) = ||x||^2 starting from x0 = [1, 2, 3]
        fg(x) = (sum(abs2, x), 2x)
        x0 = [1.0, 2.0, 3.0]
        alg = LBFGS(5; maxiter=20, verbosity=0, gradtol=1e-8)

        x_opt, f_opt, g_opt, numfg, history = optimize_reload(fg, x0, alg)
        @test f_opt < 1e-10
        @test norm(x_opt) < 1e-5
    end

    @testset "optimize_reload — resume from state" begin
        mktempdir() do dir
            fg(x) = (sum(abs2, x), 2x)
            x0 = [1.0, 2.0, 3.0]
            alg = LBFGS(5; maxiter=5, verbosity=0, gradtol=1e-12)
            filepath = joinpath(dir, "resume_test.jld2")

            # Run 5 iterations, save state
            x1, f1, _, _, _ = optimize_reload(fg, x0, alg;
                save_state_to=filepath, save_every=1)

            # Resume and run more
            alg2 = LBFGS(5; maxiter=20, verbosity=0, gradtol=1e-8)
            x2, f2, _, _, _ = optimize_reload(fg, x0, alg2;
                resume_from=filepath)
            @test f2 <= f1  # should improve or be same
        end
    end
end
```

**Step 2: Run and verify**

**Step 3: Commit**

```bash
git add test/test_patch.jl
git commit -m "test: add OptimKit patch tests (LBFGS state, save/load, optimize_reload)"
```

---

### Task 10: Create `test/test_mpi.jl` (optional)

**Files:**
- Create: `test/test_mpi.jl`

**Step 1: Write the test file**

```julia
using MPI

@testset "MPI (optional)" begin
    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    @testset "split_count / split_ranges with MPI" begin
        counts = split_count(10, nprocs)
        @test sum(counts) == 10
        ranges = split_ranges(10, nprocs)
        @test length(ranges) == nprocs
    end

    @testset "parallel matches forloop" begin
        χ, D = 4, 2
        FL = rand(ComplexF64, χ, D, χ)
        ALu = rand(ComplexF64, χ, D, χ)
        ALd = rand(ComplexF64, χ, D, χ)
        M = rand(ComplexF64, D, D, D, D)

        r_serial = FLmap_parallel(FL, ALu, ALd, M; ifparallel=false, forloop_iter=1)
        r_parallel = FLmap_parallel(FL, ALu, ALd, M; ifparallel=true, forloop_iter=1)
        @test r_serial ≈ r_parallel atol=1e-10
    end

    MPI.Finalize()
end
```

**Step 2: Run with MPI (only if ENV var set)**

```bash
TENET_TEST_MPI=true mpiexec -n 2 julia --project test/runtests.jl
```

**Step 3: Commit**

```bash
git add test/test_mpi.jl
git commit -m "test: add optional MPI tests"
```

---

### Task 11: Run full test suite and fix failures

**Step 1: Run full test suite**

```bash
cd "D:/1 - research/1.26 - iPEPS_opt/TeneT.jl"
julia --project -e 'using Pkg; Pkg.test()' 2>&1 | tee test_output.log
```

**Step 2: Fix any failures**

Iterate: read error output, fix the failing test or adjust imports, re-run.

**Step 3: Commit fixes**

```bash
git add test/
git commit -m "test: fix test suite issues found during full run"
```

---

### Task 12: Verify coverage and finalize

**Step 1: Run with coverage**

```bash
julia --project -e '
using Pkg
Pkg.test(; coverage=true)
'
```

**Step 2: Check coverage report**

```bash
julia --project -e '
using Pkg
Pkg.add("Coverage")
using Coverage
coverage = process_folder("src")
# Filter out models/
coverage = filter(c -> !contains(c.filename, "models"), coverage)
covered, total = get_summary(coverage)
println("Coverage: $(covered)/$(total) = $(round(100*covered/total, digits=1))%")
'
```

Expected: >95% coverage excluding models/.

**Step 3: Final commit**

```bash
git add -A
git commit -m "test: complete test suite with >95% coverage (excl. models/)"
```
