# TFIsing Real C4v SU AD Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a fixed-D real C4v TFIsing Simple-Update layer inside the complete TeneT.jl AD objective and validate it at `D = 2`, `χ = 8`, and `τ = 0.01`.

**Architecture:** Keep generic multi-cell SU untouched and route an explicit `SUmode = :c4v_real` to a new pure single-site implementation. The new layer builds TFIsing-specific real gates, performs four rotated one-factor Takagi updates, restores full C4v symmetry, and is used consistently by environment, energy, observables, and physical checkpoint materialization.

**Tech Stack:** Julia 1.x, TeneT.jl, TensorOperations, LinearAlgebra SVD, Zygote, ChainRulesCore, VUMPS{C4v}, OptimKit, JLD2, and Test.

## Global Constraints

- Accept only real `Float32`/`Float64` tensors in `:c4v_real` mode.
- Require square-lattice `TFIsing`, `J >= 0`, `τ >= 0`, `S == 1/2`, pattern `[1;;]`, and equal virtual dimensions.
- Keep D fixed inside the differentiable SU layer.
- Apply full C4v restriction before SU and again after the four-direction sweep.
- Make `τ == 0` return the normalized direct-C4v tensor without entering SVD.
- Do not mutate the raw tensor or any tensor on the Zygote path.
- Differentiate through gates-as-constants, contractions, retained singular data, Takagi factor, C4v projection, boundary, and energy.
- Preserve `ifSU == false` and `SUmode == :generic` behavior.
- Preserve raw checkpoint key `bcipeps`; add unambiguous `physical_bcipeps` only for the opted-in C4v workflow.
- Initial validation is CPU serial with `J = 1`, `h = 3.04438`, `D = 2`, `χ = 8`, and `τ = 0.01`.
- GPU, MPI, complex tensors, multi-site cells, antiferromagnetic J, and production extrapolation claims are outside this plan.

## File Structure

- Create `src/ipeps_optimize/c4v_su_parameterization.jl`: TFIsing gates, real Takagi factor, directional update, validation, and full C4v SU materializer.
- Create `test/test_c4v_su_ad.jl`: self-contained unit, checkpoint, directional-gradient, and χ=8 full-graph tests.
- Create `examples/TFIsing/TFIsing_Square_VUMPS_C4v_SU_smoke.jl`: paired direct/SU smoke from one raw seed.
- Modify `src/ipeps_optimize/interface.jl`: add `GradientOptimize.SUmode`.
- Modify `src/ipeps_optimize/build_A.jl`: central SU-mode routing.
- Modify `src/TeneT.jl`: include the focused C4v SU source.
- Modify `src/ipeps_optimize/optimize.jl`: physical checkpoint materialization and callback wiring.
- Modify `src/utils/io.jl`: raw/physical checkpoint reader.
- Modify `src/observable/correlation_length.jl`: expose the leading C4v transfer eigenvalues used to compute ξ.
- Modify `test/runtests.jl`: include the focused suite after it passes standalone.

---

### Task 1: Add Explicit SU Mode Configuration and Routing

**Files:**
- Modify: `src/ipeps_optimize/interface.jl:28-31`
- Modify: `src/ipeps_optimize/build_A.jl:15-22`
- Create: `test/test_c4v_su_ad.jl`

**Interfaces:**
- Consumes: existing `GradientOptimize`, `build_A`, and `SU_parameterization`.
- Produces: `GradientOptimize.SUmode::Symbol`, `_su_mode(params)::Symbol`, and `_apply_su_parameterization(Ar, params)`.

- [ ] **Step 1: Write the failing configuration and routing tests**

Create `test/test_c4v_su_ad.jl` with this self-contained preamble and first testset:

```julia
using Test
using Random
using LinearAlgebra
using JLD2
using TensorOperations
using Zygote
using OptimKit: LBFGS
using TeneT

function c4v_test_params(; ifSU=false, SUmode=:generic, SUτ=0.0,
                          folder=mktempdir(), optimizer_maxiter=2,
                          boundary_iters=6)
    model = TFIsing(lattice=Square(), J=1.0, h=3.04438)
    boundary_alg = VUMPS{C4v}(
        ifsimple_eig=true, ifparallel=false, ifcheckpoint=false,
        maxiter=boundary_iters, miniter=boundary_iters,
        maxiter_ad=boundary_iters, miniter_ad=boundary_iters,
        power_iter=5, power_iter_ad=5, power_iter_obs=24,
        show_every=1000, tol=1e-10, verbosity=0,
    )
    return GradientOptimize(
        model=model, pattern=[1;;], boundary_alg=boundary_alg,
        optimizer=LBFGS(5; maxiter=optimizer_maxiter, verbosity=0,
                        gradtol=1e-8),
        folder=folder, verbosity=0, ifSU=ifSU, SUmode=SUmode, SUτ=SUτ,
        ifprecondition=false, reuse_env=true,
        ifsave_env=false, ifload_env=false,
        ifsave_lbfgs=false, ifload_lbfgs=false,
        save_every=0, ifplot=false,
    )
end

function normalized_c4v(A)
    B = C4v_restriction(A)
    return B / norm(B)
end

@testset "SU mode routing" begin
    A = randn(Float64, 2, 2, 2, 2, 2, 1)
    @test c4v_test_params().SUmode === :generic

    ignored = c4v_test_params(ifSU=false, SUmode=:not_a_mode)
    @test TeneT.build_A(A, ignored)[1] ≈ A[:, :, :, :, :, 1]

    invalid = c4v_test_params(ifSU=true, SUmode=:not_a_mode)
    err = try
        TeneT.build_A(A, invalid)
        nothing
    catch caught
        caught
    end
    @test err isa ArgumentError
    @test occursin("unsupported SUmode", sprint(showerror, err))
end
```

- [ ] **Step 2: Run the standalone test and verify the new keyword is missing**

Run:

```powershell
julia --project=. test/test_c4v_su_ad.jl
```

Expected: FAIL while constructing `GradientOptimize`, reporting that `SUmode` is not a recognized keyword.

- [ ] **Step 3: Add the parameter and central routing helper**

Add beside `ifSU` in `GradientOptimize`:

```julia
SUτ::Real = 0.0
ifSU::Bool = false
SUmode::Symbol = :generic
```

Add to `build_A.jl` and route the existing six-dimensional method through it:

```julia
_su_mode(params) = hasproperty(params, :SUmode) ? params.SUmode : :generic

function _apply_su_parameterization(Ar, params)
    mode = _su_mode(params)
    if mode === :generic
        return SU_parameterization(Ar, params; D_new=size(Ar[1], 1))
    elseif mode === :c4v_real
        return _c4v_real_su_parameterization(Ar, params)
    end
    throw(ArgumentError("unsupported SUmode=$mode; use :generic or :c4v_real"))
end
```

Replace the current `ifSU` branch with:

```julia
if hasproperty(params, :ifSU) && params.ifSU
    return _apply_su_parameterization(Ar, params)
end
return Ar
```

The `_c4v_real_su_parameterization` name may be unresolved until called; Julia resolves it at runtime after all module includes complete.

- [ ] **Step 4: Run the standalone routing test**

Run `julia --project=. test/test_c4v_su_ad.jl`.

Expected: PASS for `SU mode routing`.

- [ ] **Step 5: Commit the configuration boundary**

```powershell
git add src/ipeps_optimize/interface.jl src/ipeps_optimize/build_A.jl test/test_c4v_su_ad.jl
git commit -m "Add explicit iPEPS SU mode routing"
```

### Task 2: Add Real TFIsing Strang Gates

**Files:**
- Create: `src/ipeps_optimize/c4v_su_parameterization.jl`
- Modify: `src/TeneT.jl:124-130`
- Modify: `test/test_c4v_su_ad.jl`

**Interfaces:**
- Consumes: `TFIsing{Square}`, `_tfising_pauli_operators`, `_su_twosite_gate`, and `_arraytype`.
- Produces: `_tfising_c4v_gates(model::TFIsing{Square}, τ::Real, ::Type{T}, atype) -> (Gx_half, Gzz)`.

- [ ] **Step 1: Add failing analytic gate tests**

Append:

```julia
@testset "real TFIsing C4v gates" begin
    model = TFIsing(lattice=Square(), J=1.0, h=3.04438)
    τ = 0.01
    Gx, Gzz = TeneT._tfising_c4v_gates(model, τ, Float64, Array)
    σx = [0.0 1.0; 1.0 0.0]
    σz = [1.0 0.0; 0.0 -1.0]

    @test Gx ≈ exp((τ * model.h / 2) * σx) atol=1e-14 rtol=1e-14
    @test reshape(Gzz, 4, 4) ≈ exp(τ * model.J * kron(σz, σz)) atol=1e-14 rtol=1e-14
    @test eltype(Gx) === Float64
    @test eltype(Gzz) === Float64
    Gx32, Gzz32 = TeneT._tfising_c4v_gates(model, τ, Float32, Array)
    @test eltype(Gx32) === Float32
    @test eltype(Gzz32) === Float32

    @test_throws ArgumentError TeneT._tfising_c4v_gates(
        TFIsing(lattice=Square(), J=-1.0, h=3.04438), τ, Float64, Array)
    @test_throws ArgumentError TeneT._tfising_c4v_gates(
        model, -τ, Float64, Array)
end
```

- [ ] **Step 2: Run the test and verify the gate helper is undefined**

Run `julia --project=. test/test_c4v_su_ad.jl`.

Expected: FAIL with `UndefVarError: _tfising_c4v_gates not defined`.

- [ ] **Step 3: Include the focused source and implement the exact gates**

Add after `include("ipeps_optimize/su_parameterization.jl")` in `src/TeneT.jl`:

```julia
include("ipeps_optimize/c4v_su_parameterization.jl")
```

Create the source file with:

```julia
# Fixed-D real C4v Simple-Update parameterization for TFIsing{Square}.

function _tfising_c4v_gates(model::TFIsing{Square}, τ::Real,
                             ::Type{T}, atype) where {T<:AbstractFloat}
    model.S == 1 / 2 ||
        throw(ArgumentError("C4v real SU requires TFIsing S=1/2; got $(model.S)"))
    model.J >= 0 ||
        throw(ArgumentError("C4v real SU requires ferromagnetic J>=0; got $(model.J)"))
    τ >= 0 ||
        throw(ArgumentError("C4v real SU requires τ>=0; got $τ"))

    σx0, σz0 = _tfising_pauli_operators(model, Array)
    σx = T.(σx0)
    σz = T.(σz0)
    Gx_half = exp(T(τ * model.h / 2) .* σx)

    hzz = zeros(T, 2, 2, 2, 2)
    @tensor hzz[i, j, k, l] := -T(model.J) * σz[i, j] * σz[k, l]
    Gzz = _su_twosite_gate(hzz, T(τ), 2, Array)
    return atype(Gx_half), atype(Gzz)
end
```

- [ ] **Step 4: Run the standalone gate tests**

Run `julia --project=. test/test_c4v_su_ad.jl`.

Expected: both testsets PASS.

- [ ] **Step 5: Commit the model-specific gate layer**

```powershell
git add src/TeneT.jl src/ipeps_optimize/c4v_su_parameterization.jl test/test_c4v_su_ad.jl
git commit -m "Add real TFIsing C4v SU gates"
```

### Task 3: Implement the Real One-Factor Takagi Kernel

**Files:**
- Modify: `src/ipeps_optimize/c4v_su_parameterization.jl`
- Modify: `test/test_c4v_su_ad.jl`

**Interfaces:**
- Consumes: real symmetric pair matrix `M`, input unfolding `B0`, retained dimension D, and the existing regularized `svd` rrule.
- Produces: `_c4v_pair_matrix`, `_c4v_right_unfolding`, and `_real_c4v_takagi_factor(M, B0, D; tol_psd, context, diagnostics)`.

- [ ] **Step 1: Add failing pair-matrix and Takagi tests**

Append:

```julia
@testset "real C4v Takagi factor" begin
    Random.seed!(3103)
    A = normalized_c4v(randn(Float64, 2, 2, 2, 2, 2))
    model = TFIsing(lattice=Square(), J=1.0, h=3.04438)
    _, Gzz = TeneT._tfising_c4v_gates(model, 0.01, Float64, Array)
    M = TeneT._c4v_pair_matrix(A, Gzz)
    @test M ≈ transpose(M) atol=1e-12 rtol=1e-12
    @test minimum(eigvals(Symmetric(M))) >= -1e-12

    direct = zeros(size(M))
    linear = LinearIndices((2, 2, 2, 2))
    for f in 1:2, a in 1:2, b in 1:2, j in 1:2,
        c in 1:2, d in 1:2, e in 1:2, k in 1:2
        row = linear[f, a, b, j]
        col = linear[c, d, e, k]
        direct[row, col] = sum(A[a, b, g, f, h] * A[d, e, g, c, i] *
                               Gzz[h, i, j, k]
                               for g in 1:2, h in 1:2, i in 1:2)
    end
    @test M ≈ direct atol=1e-12 rtol=1e-12

    B = randn(Float64, 12, 3)
    Mpsd = B * transpose(B)
    B0 = randn(Float64, 12, 3)
    X = TeneT._real_c4v_takagi_factor(Mpsd, B0, 3)
    @test X * transpose(X) ≈ Mpsd atol=1e-10 rtol=1e-10
    @test size(X) == (12, 3)

    Bwide = randn(Float64, 12, 4)
    Mwide = Bwide * transpose(Bwide)
    X2 = TeneT._real_c4v_takagi_factor(Mwide, randn(12, 2), 2)
    eigenvalues = sort(eigvals(Symmetric(Mwide)); rev=true)
    optimal_error = sqrt(sum(abs2, eigenvalues[3:end]))
    @test norm(Mwide - X2 * transpose(X2)) ≈ optimal_error atol=1e-9 rtol=1e-9

    Mindef = Diagonal([-1.0, 2.0]) |> Matrix
    @test_throws ArgumentError TeneT._real_c4v_takagi_factor(
        Mindef, Matrix{Float64}(I, 2, 2), 2)

    M0, B00 = copy(Mpsd), copy(B0)
    Xdiagnostic = TeneT._real_c4v_takagi_factor(Mpsd, B0, 3; diagnostics=true)
    @test Mpsd == M0
    @test B0 == B00
    @test Xdiagnostic ≈ X
end
```

- [ ] **Step 2: Run and verify the Takagi helpers are undefined**

Run `julia --project=. test/test_c4v_su_ad.jl`.

Expected: FAIL at `_c4v_pair_matrix`.

- [ ] **Step 3: Implement pair construction and the reference unfolding**

Add:

```julia
function _c4v_pair_matrix(A::AbstractArray{T,5}, gate) where {T<:Real}
    @tensor pair[f, a, b, j, c, d, e, k] :=
        A[a, b, g, f, h] * A[d, e, g, c, i] * gate[h, i, j, k]
    M = reshape(pair, prod(size(pair)[1:4]), prod(size(pair)[5:8]))
    ignore_derivatives() do
        residual = norm(M - transpose(M)) / max(norm(M), eps(T))
        residual <= 1e-10 ||
            throw(ArgumentError("C4v self-bond pair matrix is not symmetric; residual=$residual"))
    end
    return (M + transpose(M)) / 2
end

function _c4v_right_unfolding(A::AbstractArray{<:Real,5})
    ordered = permutedims(A, (4, 1, 2, 5, 3))
    return reshape(ordered, prod(size(ordered)[1:4]), size(ordered, 5))
end
```

- [ ] **Step 4: Implement PSD checking, one-factor truncation, and Procrustes alignment**

Add:

```julia
function _check_real_takagi_values(λ, scale, tol_psd, context)
    λmin = minimum(Array(λ))
    λmin >= -tol_psd * scale && return nothing
    throw(ArgumentError(
        "real C4v Takagi requires a PSD retained block; " *
        "context=$context min_lambda=$λmin scale=$scale spectrum=$(Array(λ))"))
end

function _real_c4v_takagi_factor(M::AbstractMatrix{T}, B0::AbstractMatrix{T},
                                  D::Int; tol_psd::Real=100eps(T),
                                  context=(turn=0, τ=0.0),
                                  diagnostics::Bool=false) where {T<:AbstractFloat}
    size(M, 1) == size(M, 2) ||
        throw(ArgumentError("real C4v Takagi requires a square matrix"))
    size(B0) == (size(M, 1), D) ||
        throw(ArgumentError("Takagi reference has size $(size(B0)); expected $((size(M, 1), D))"))

    Msym = (M + transpose(M)) / 2
    fac = svd(Msym)
    V = fac.V
    retained = @view V[:, 1:D]
    λ = vec(sum(retained .* (Msym * retained); dims=1))
    scale = max(first(fac.S), one(T))
    ignore_derivatives() do
        _check_real_takagi_values(λ, scale, tol_psd, context)
    end

    λpositive = max.(λ, zero(T))
    X = retained * Diagonal(sqrt.(λpositive))
    alignment = svd(transpose(X) * B0)
    O = alignment.U * alignment.Vt
    Xaligned = X * O

    ignore_derivatives() do
        retained_matrix = retained * Diagonal(λpositive) * transpose(retained)
        factor_residual = norm(retained_matrix - Xaligned * transpose(Xaligned)) /
                          max(norm(retained_matrix), eps(T))
        factor_residual <= 1000eps(T) ||
            throw(ArgumentError("real C4v Takagi reconstruction failed; context=$context residual=$factor_residual"))
    end

    if diagnostics
        ignore_derivatives() do
            @info "real C4v Takagi" context λ=Array(λ) symmetry_residual=(norm(M - transpose(M)) / max(norm(M), eps(T))) reconstruction_residual=(norm(Msym - Xaligned * transpose(Xaligned)) / max(norm(Msym), eps(T)))
        end
    end
    return Xaligned
end
```

- [ ] **Step 5: Run the Takagi tests and commit**

Run `julia --project=. test/test_c4v_su_ad.jl` and expect all current testsets to PASS, then:

```powershell
git add src/ipeps_optimize/c4v_su_parameterization.jl test/test_c4v_su_ad.jl
git commit -m "Add differentiable real C4v Takagi factor"
```

### Task 4: Build the Pure Four-Direction C4v SU Layer

**Files:**
- Modify: `src/ipeps_optimize/c4v_su_parameterization.jl`
- Modify: `src/ipeps_optimize/build_A.jl`
- Modify: `test/test_c4v_su_ad.jl`

**Interfaces:**
- Consumes: Tasks 1-3 helpers and a one-site `StructArray` already restricted by the optimizer's `restriction_ipeps` closure.
- Produces: `_c4v_symmetry_residual`, `_c4v_apply_physical_gate`, `_c4v_takagi_bond_update`, `_c4v_four_turn_sweep`, and `_c4v_real_su_parameterization(Ar, params)`.

- [ ] **Step 1: Add failing state-contract tests**

Append:

```julia
@testset "fixed-D real C4v SU state contract" begin
    Random.seed!(4104)
    Araw = randn(Float64, 2, 2, 2, 2, 2, 1)
    A0 = normalized_c4v(Araw)

    p0 = c4v_test_params(ifSU=true, SUmode=:c4v_real, SUτ=0.0)
    input_before = copy(A0)
    out0 = TeneT.build_A(A0, p0)[1]
    @test A0 == input_before
    @test out0 ≈ A0[:, :, :, :, :, 1] atol=1e-13 rtol=1e-13
    @test norm(out0) ≈ 1.0 atol=1e-13

    p = c4v_test_params(ifSU=true, SUmode=:c4v_real, SUτ=0.01)
    out = TeneT.build_A(A0, p)[1]
    @test A0 == input_before
    @test size(out) == (2, 2, 2, 2, 2)
    @test eltype(out) === Float64
    @test all(isfinite, out)
    @test norm(out) ≈ 1.0 atol=1e-12
    @test TeneT._c4v_symmetry_residual(out) < 1e-12
    @test !(out ≈ out0 atol=1e-10 rtol=1e-10)

    weight = randn(size(out))
    direction = randn(size(Araw)); direction /= norm(direction)
    loss(X) = sum(weight .* TeneT.build_A(normalized_c4v(X), p)[1])
    gradient = Zygote.gradient(loss, Araw)[1]
    ad_directional = dot(gradient, direction)
    errors = [abs((loss(Araw + ε * direction) - loss(Araw - ε * direction)) /
                  (2ε) - ad_directional) / max(abs(ad_directional), 1e-10)
              for ε in (2e-4, 1e-4, 5e-5)]
    @test all(isfinite, gradient)
    @test minimum(errors) < 5e-4

    complex_cell = TeneT.StructArray([complex.(out0)], [1;;])
    @test_throws ArgumentError TeneT._c4v_real_su_parameterization(complex_cell, p)
    @test_throws ArgumentError TeneT._c4v_real_su_parameterization(
        TeneT.StructArray([out0, out0], [1 2]), p)
    @test_throws ArgumentError TeneT._c4v_real_su_parameterization(
        TeneT.StructArray([randn(2, 2, 2, 2, 2)], [1;;]), p)
    @test_throws ArgumentError TeneT._c4v_real_su_parameterization(
        TeneT.StructArray([randn(2, 2, 3, 2, 2)], [1;;]), p)
end
```

- [ ] **Step 2: Run and verify the C4v materializer is undefined**

Run `julia --project=. test/test_c4v_su_ad.jl`.

Expected: FAIL when `build_A` routes to `_c4v_real_su_parameterization`.

- [ ] **Step 3: Implement symmetry checking, onsite application, and one directional update**

Add:

```julia
function _c4v_symmetry_residual(A::AbstractArray{<:Real,5})
    n = max(norm(A), eps(eltype(A)))
    transforms = (
        permutedims(A, (1, 4, 3, 2, 5)),
        permutedims(A, (3, 2, 1, 4, 5)),
        permutedims(A, (2, 1, 4, 3, 5)),
        permutedims(A, (4, 3, 2, 1, 5)),
    )
    return maximum(norm(A - transformed) / n for transformed in transforms)
end

function _c4v_apply_physical_gate(A::AbstractArray{<:Real,5}, G)
    @tensor B[l, d, r, u, q] := A[l, d, r, u, p] * G[q, p]
    return B
end

function _c4v_takagi_bond_update(A::AbstractArray{T,5}, Gzz, D::Int;
                                  turn::Int, τ::Real,
                                  diagnostics::Bool=false) where {T<:AbstractFloat}
    M = _c4v_pair_matrix(A, Gzz)
    B0 = _c4v_right_unfolding(A)
    X = _real_c4v_takagi_factor(
        M, B0, D; context=(turn=turn, τ=τ), diagnostics=diagnostics)
    shaped = reshape(X, size(A, 4), size(A, 1), size(A, 2), size(A, 5), D)
    return permutedims(shaped, (2, 3, 5, 1, 4))
end

function _c4v_four_turn_sweep(A, Gzz, D, τ; diagnostics::Bool=false,
                               update=_c4v_takagi_bond_update)
    work = A
    for turn in 1:4
        work = update(
            work, Gzz, D; turn=turn, τ=τ, diagnostics=diagnostics)
        work = permutedims(work, (2, 3, 4, 1, 5))
    end
    return work
end
```

- [ ] **Step 4: Implement validation and the complete Strang layer**

Add:

```julia
function _validate_c4v_real_su(Ar, params)
    Ar.pattern == [1;;] ||
        throw(ArgumentError("C4v real SU requires the one-site pattern [1;;]; got $(Ar.pattern)"))
    params.model isa TFIsing{Square} ||
        throw(ArgumentError("C4v real SU requires TFIsing{Square}; got $(typeof(params.model))"))
    params.boundary_alg isa VUMPS{C4v} ||
        throw(ArgumentError("C4v real SU requires VUMPS{C4v}; got $(typeof(params.boundary_alg))"))
    A = Ar[1]
    eltype(A) in (Float32, Float64) ||
        throw(ArgumentError("C4v real SU requires real Float32/Float64 tensors; got $(eltype(A))"))
    D = size(A, 1)
    all(==(D), size(A)[1:4]) ||
        throw(ArgumentError("C4v real SU requires equal virtual dimensions; got $(size(A)[1:4])"))
    iszero(norm(A)) && throw(ArgumentError("C4v real SU received a zero tensor"))
    residual = _c4v_symmetry_residual(A)
    residual <= 1e-10 ||
        throw(ArgumentError("C4v real SU input is not C4v; residual=$residual"))
    return D
end

function _c4v_real_su_parameterization(Ar, params; diagnostics::Bool=false)
    A = Ar[1]
    D = ignore_derivatives() do
        _validate_c4v_real_su(Ar, params)
    end
    A0 = A / norm(A)
    τ = params.SUτ
    iszero(τ) && return StructArray([copy(A0)], Ar.pattern)

    Gx_half, Gzz = ignore_derivatives() do
        _tfising_c4v_gates(params.model, τ, eltype(A0), _arraytype(A0))
    end
    work = _c4v_apply_physical_gate(A0, Gx_half)
    work = _c4v_four_turn_sweep(work, Gzz, D, τ; diagnostics)
    work = _c4v_apply_physical_gate(work, Gx_half)
    physical = C4v_restriction(work)
    physical /= norm(physical)

    ignore_derivatives() do
        residual = _c4v_symmetry_residual(physical)
        all(isfinite, physical) ||
            throw(ArgumentError("C4v real SU produced nonfinite entries at τ=$τ"))
        residual <= 1e-12 ||
            throw(ArgumentError("C4v real SU output residual=$residual at τ=$τ"))
    end
    return StructArray([physical], Ar.pattern)
end
```

- [ ] **Step 5: Add a rotation bookkeeping assertion**

Append to the state-contract testset:

```julia
function marker_update(X, _, _; turn, τ, diagnostics=false)
    marker = reshape([0.0, 1.0], 1, 1, 2, 1, 1)
    return X .+ turn .* marker
end
marked = TeneT._c4v_four_turn_sweep(
    zeros(2, 2, 2, 2, 1), nothing, 2, 0.01; update=marker_update)
for axis in 1:4
    first_slice = ntuple(i -> i == axis ? 1 : Colon(), 5)
    second_slice = ntuple(i -> i == axis ? 2 : Colon(), 5)
    @test marked[first_slice...] != marked[second_slice...]
end
```

Run the standalone test and expect all testsets to PASS.

- [ ] **Step 6: Commit the complete materializer**

```powershell
git add src/ipeps_optimize/c4v_su_parameterization.jl src/ipeps_optimize/build_A.jl test/test_c4v_su_ad.jl
git commit -m "Integrate fixed-D real C4v SU materialization"
```

### Task 5: Save and Load Raw Versus Physical Checkpoints

**Files:**
- Modify: `src/ipeps_optimize/optimize.jl:148-203,250-304`
- Modify: `src/utils/io.jl:1-30`
- Modify: `test/test_c4v_su_ad.jl`

**Interfaces:**
- Consumes: `restriction_ipeps`, `build_A`, `_su_mode`, and one-site `StructArray` output.
- Produces: `_materialize_c4v_checkpoint`, `_save_ipeps_checkpoint`, and `load_ipeps_checkpoint(path; state)`.

- [ ] **Step 1: Add failing checkpoint-semantics tests**

Append:

```julia
@testset "raw and physical C4v checkpoints" begin
    Random.seed!(5105)
    folder = mktempdir()
    path = joinpath(folder, "state.jld2")
    raw = randn(Float64, 2, 2, 2, 2, 2, 1)
    params = c4v_test_params(ifSU=true, SUmode=:c4v_real, SUτ=0.01,
                             folder=folder)
    physical = TeneT._materialize_c4v_checkpoint(raw, params, normalized_c4v)
    TeneT._save_ipeps_checkpoint(path, raw, params, 7, 8; physical)

    @test load(path, "bcipeps") == raw
    saved_physical = load(path, "physical_bcipeps")
    @test saved_physical ≈ physical
    @test load(path, "state_semantics") == "raw_plus_c4v_su_v1"
    @test load(path, "SUmode") == "c4v_real"
    @test load(path, "iteration") == 7
    @test load(path, "chi") == 8
    @test TeneT.load_ipeps_checkpoint(path; state=:raw) == raw
    @test TeneT.load_ipeps_checkpoint(path; state=:physical) ≈ physical

    legacy = joinpath(folder, "legacy.jld2")
    save(legacy, "bcipeps", raw)
    @test TeneT.load_ipeps_checkpoint(legacy; state=:physical) == raw
    @test_throws ArgumentError TeneT.load_ipeps_checkpoint(path; state=:unknown)

    measure_params = deepcopy(params)
    measure_params.ifSU = false
    @test TeneT.build_A(saved_physical, measure_params)[1] ≈ physical[:, :, :, :, :, 1]
end
```

- [ ] **Step 2: Run and verify the checkpoint helpers are undefined**

Run `julia --project=. test/test_c4v_su_ad.jl`.

Expected: FAIL at `_materialize_c4v_checkpoint`.

- [ ] **Step 3: Implement focused checkpoint materialization and writing**

Add to `optimize.jl` before `_finalize!`:

```julia
function _materialize_c4v_checkpoint(x, params, restriction_ipeps)
    physical = build_A(restriction_ipeps(x), params)
    physical.pattern == [1;;] ||
        throw(ArgumentError("physical C4v checkpoint requires pattern [1;;]"))
    site = Array(physical[1])
    return reshape(site, size(site)..., 1)
end

function _save_ipeps_checkpoint(path, raw, params, iter, χ; physical=nothing)
    if physical === nothing
        save(path, "bcipeps", Array(raw); iotype=IOStream)
        return path
    end
    save(path,
         "bcipeps", Array(raw),
         "physical_bcipeps", Array(physical),
         "state_semantics", "raw_plus_c4v_su_v1",
         "SUmode", string(_su_mode(params)),
         "SUτ", Float64(params.SUτ),
         "model_J", Float64(params.model.J),
         "model_h", Float64(params.model.h),
         "iteration", Int(iter),
         "chi", Int(χ); iotype=IOStream)
    return path
end
```

Give `_finalize!` a keyword `materialize_checkpoint=nothing`, replace its direct `save` call with:

```julia
physical = materialize_checkpoint === nothing ? nothing : materialize_checkpoint(x)
_save_ipeps_checkpoint(
    joinpath(ipeps_dir, "No.$(iter).jld2"), x, params, iter, χ; physical)
```

In `optimise_ipeps`, create the closure once:

```julia
materialize_checkpoint = _su_mode(params) === :c4v_real ?
    x -> _materialize_c4v_checkpoint(x, params, restriction_ipeps) : nothing
```

Pass it from the OptimKit adapter:

```julia
finalize! = (x, f, g, iter) -> _finalize!(
    x, f, g, iter, rt, rt′, D, χ, params, t0, fδEierr;
    materialize_checkpoint)
```

- [ ] **Step 4: Implement the state-selecting reader**

Append to `src/utils/io.jl`:

```julia
function load_ipeps_checkpoint(path::AbstractString; state::Symbol=:raw)
    state in (:raw, :physical) ||
        throw(ArgumentError("checkpoint state must be :raw or :physical; got $state"))
    return jldopen(path, "r") do file
        if state === :raw
            return file["bcipeps"]
        elseif haskey(file, "physical_bcipeps")
            return file["physical_bcipeps"]
        end
        @warn "checkpoint has no physical_bcipeps; using legacy bcipeps semantics" path
        return file["bcipeps"]
    end
end
```

- [ ] **Step 5: Run checkpoint tests and commit**

Run `julia --project=. test/test_c4v_su_ad.jl` and expect all current testsets to PASS, then:

```powershell
git add src/ipeps_optimize/optimize.jl src/utils/io.jl test/test_c4v_su_ad.jl
git commit -m "Save raw and physical C4v SU checkpoints"
```

### Task 6: Expose the Leading C4v Correlation Spectrum

**Files:**
- Modify: `src/observable/correlation_length.jl:182-213`
- Modify: `test/test_c4v_su_ad.jl`

**Interfaces:**
- Consumes: the existing C4v MPS/channel transfer maps and `eigsolve` call.
- Produces: `cor_len_spectrum_value(env::C4vVUMPSEnv, params, M; method, nev)` and `_correlation_length_from_spectrum(λcs)`.

- [ ] **Step 1: Add a failing spectrum-to-ξ unit test**

Append:

```julia
@testset "C4v correlation spectrum reduction" begin
    λ = ComplexF64[1.0, 0.5, 0.2]
    @test TeneT._correlation_length_from_spectrum(λ) ≈ 1 / log(2)
    @test isinf(TeneT._correlation_length_from_spectrum(ComplexF64[1.0]))
end
```

- [ ] **Step 2: Run and verify the reduction helper is undefined**

Run `julia --project=. test/test_c4v_su_ad.jl`.

Expected: FAIL with `_correlation_length_from_spectrum not defined`.

- [ ] **Step 3: Extract the spectrum without changing current ξ semantics**

Add:

```julia
export cor_len_spectrum_value

function cor_len_spectrum_value(env::C4vVUMPSEnv, params, M;
                                method::Symbol=:mps, nev::Int=5)
    nev > 0 || throw(ArgumentError("cor_len_spectrum_value requires nev>0; got $nev"))
    @unpack AL, C, FL = env
    if method === :channel
        @unpack forloop_iter = params
        @unpack ifparallel = params.boundary_alg
        M_tensor = M isa StructArray ? M[1, 1] : M
        f = FLi -> FLmap_parallel(
            FLi, AL, conj(AL), M_tensor; ifparallel, forloop_iter)
        v_init = FL
    elseif method === :mps
        f = c -> Lmap(c, AL, conj(AL))
        v_init = C
    else
        throw(ArgumentError("cor_len_spectrum_value: use :mps or :channel; got $method"))
    end
    requested = min(nev, length(v_init))
    λcs, _, info = eigsolve(
        f, v_init, requested, :LM; maxiter=100, ishermitian=false)
    info.converged == 0 && @warn "C4v correlation spectrum did not converge" method
    return λcs
end

function _correlation_length_from_spectrum(λcs)
    length(λcs) < 2 && return Inf
    for i in 2:length(λcs)
        if !(norm(λcs[i]) ≈ norm(λcs[1]))
            iszero(λcs[i]) && return Inf
            return -1 / log(abs(λcs[i] / λcs[1]))
        end
    end
    return Inf
end
```

Replace the body of the existing C4v `cor_len_value` method with:

```julia
λcs = cor_len_spectrum_value(env, params, M; method, nev=5)
ξ = _correlation_length_from_spectrum(λcs)
params.verbosity >= 4 && println("ξ ($method) = $ξ")
return ξ
```

- [ ] **Step 4: Run correlation-length regressions**

Run:

```powershell
julia --project=. test/test_c4v_su_ad.jl
```

Expected: focused C4v tests PASS. The complete suite in Task 9 covers existing observable regressions.

- [ ] **Step 5: Commit the reusable spectrum interface**

```powershell
git add src/observable/correlation_length.jl test/test_c4v_su_ad.jl
git commit -m "Expose C4v correlation transfer spectrum"
```

### Task 7: Gate the Complete χ=8 AD Objective

**Files:**
- Modify: `test/test_c4v_su_ad.jl`
- Modify: `test/runtests.jl:180-188`

**Interfaces:**
- Consumes: `normalized_c4v`, `c4v_test_params`, `initialize_env`, `energy`, the complete SU materializer, and VUMPS{C4v} reverse rules.
- Produces: a deterministic regression gate for `raw -> C4v -> SU -> VUMPS -> energy` at `D=2`, `χ=8`, `τ=0.01`.

- [ ] **Step 1: Add a runtime-seeded objective factory**

Append above the integration testset:

```julia
function c4v_energy_objective(rt_seed, params)
    rt = deepcopy(rt_seed)
    rt_next = deepcopy(rt_seed)
    fδEierr = [1.0, 1.0, 0.0, 0.0]
    return X -> real(TeneT.energy(
        normalized_c4v(X), rt, rt_next, fδEierr, params))
end
```

The factory performs `deepcopy` before Zygote records the returned closure, so environment copying is not part of the derivative.

- [ ] **Step 2: Add the complete directional-gradient and optimizer smoke**

Append:

```julia
@testset "TFIsing C4v SU full AD graph at chi=8" begin
    Random.seed!(7107)
    D, χ = 2, 8
    Araw = randn(Float64, D, D, D, D, 2, 1)
    params = c4v_test_params(
        ifSU=true, SUmode=:c4v_real, SUτ=0.01,
        optimizer_maxiter=2, boundary_iters=6)

    rt_seed = TeneT.initialize_env(
        Araw, D, χ, params; restriction_ipeps=normalized_c4v)
    objective = c4v_energy_objective(rt_seed, params)
    energy0, pullback = Zygote.pullback(objective, Araw)
    gradient = pullback(1.0)[1]
    @test isfinite(energy0)
    @test all(isfinite, gradient)
    @test norm(gradient) > 0

    direction = randn(size(Araw)); direction /= norm(direction)
    ad_directional = dot(gradient, direction)
    errors = Float64[]
    for ε in (2e-4, 1e-4, 5e-5)
        plus = c4v_energy_objective(rt_seed, params)(Araw + ε * direction)
        minus = c4v_energy_objective(rt_seed, params)(Araw - ε * direction)
        finite_difference = (plus - minus) / (2ε)
        push!(errors, abs(finite_difference - ad_directional) /
                      max(abs(finite_difference), abs(ad_directional), 1e-9))
    end
    @test minimum(errors) < 5e-2
    @test errors[end] <= 5 * minimum(errors)

    _, optimized_energy, optimized_gradient, _, _ = TeneT.optimise_ipeps(
        copy(Araw), [χ], params; restriction_ipeps=normalized_c4v)
    @test isfinite(optimized_energy)
    @test all(isfinite, optimized_gradient)
    @test optimized_energy <= energy0 + 1e-8
end
```

- [ ] **Step 3: Run the standalone χ=8 integration gate**

Run:

```powershell
julia --project=. test/test_c4v_su_ad.jl
```

Expected: PASS with finite energy/gradient, directional error below `5e-2`, and non-increasing best energy after two LBFGS iterations. If this gate fails, invoke `superpowers:systematic-debugging` before changing tolerances or AD rules.

- [ ] **Step 4: Add the focused suite to the package test entrypoint**

Add immediately after `include("test_ipeps.jl")` in `test/runtests.jl`:

```julia
include("test_c4v_su_ad.jl")
```

Run the standalone file once more after adding the include; expected result remains PASS.

- [ ] **Step 5: Commit the full-graph regression gate**

```powershell
git add test/test_c4v_su_ad.jl test/runtests.jl
git commit -m "Validate real C4v SU AD at chi 8"
```

### Task 8: Add the Paired Direct-versus-SU Research Smoke

**Files:**
- Create: `examples/TFIsing/TFIsing_Square_VUMPS_C4v_SU_smoke.jl`
- Modify: `test/test_c4v_su_ad.jl`

**Interfaces:**
- Consumes: public optimization API, `cor_len_spectrum_value`, and C4v raw/physical checkpoint semantics.
- Produces: a CPU paired smoke that writes `smoke_summary.jld2` with matched direct/SU metrics.

- [ ] **Step 1: Add a failing example-presence test**

Append:

```julia
@testset "paired C4v SU smoke example is installed" begin
    script = joinpath(pkgdir(TeneT), "examples", "TFIsing",
                      "TFIsing_Square_VUMPS_C4v_SU_smoke.jl")
    @test isfile(script)
    source = read(script, String)
    @test occursin("h=3.04438", replace(source, " " => ""))
    @test occursin("SUmode=:c4v_real", replace(source, " " => ""))
    @test occursin("χ=8", replace(source, " " => ""))
end
```

- [ ] **Step 2: Run and verify the example file is absent**

Run `julia --project=. test/test_c4v_su_ad.jl`.

Expected: FAIL at `isfile(script)`.

- [ ] **Step 3: Create the complete paired smoke script**

Create:

```julia
using TeneT
using Random
using LinearAlgebra
using OptimKit
using JLD2

const D = 2
const χ = 8
const SEED = 42
const TAU = 0.01

output_root = isempty(ARGS) ?
    joinpath(pkgdir(TeneT), "data", "TFIsing_C4v_SU_smoke") : abspath(ARGS[1])
mkpath(output_root)

Random.seed!(SEED)
Araw0 = randn(Float64, D, D, D, D, 2, 1)
model = TFIsing(lattice=Square(), J=1.0, h=3.04438)

function restriction_ipeps(A)
    B = C4v_restriction(A)
    return B / norm(B)
end

function smoke_params(label; ifSU, SUτ)
    boundary_alg = VUMPS{C4v}(
        ifsimple_eig=true, ifparallel=false, ifcheckpoint=false,
        maxiter=12, miniter=12, maxiter_ad=6, miniter_ad=6,
        power_iter=6, power_iter_ad=6, power_iter_obs=32,
        show_every=1000, tol=1e-10, verbosity=0,
    )
    return GradientOptimize(
        model=model, pattern=[1;;], boundary_alg=boundary_alg,
        optimizer=LBFGS(8; maxiter=4, verbosity=1, gradtol=1e-8),
        folder=joinpath(output_root, label), verbosity=2,
        ifSU=ifSU, SUmode=:c4v_real, SUτ=SUτ,
        ifprecondition=false, reuse_env=true,
        ifsave_env=false, ifload_env=false,
        ifsave_lbfgs=false, ifload_lbfgs=false,
        save_every=1, ifplot=false,
    )
end

function measure_state(Araw, params)
    M = TeneT.build_A(restriction_ipeps(Araw), params)
    rt = TeneT.initialize_env(
        Araw, D, χ, params; restriction_ipeps=restriction_ipeps)
    rt, _ = TeneT.leading_boundary(rt, M, params.boundary_alg)
    env = TeneT.ObsEnv(rt, M, params.boundary_alg, params.model)
    energy = real(TeneT.energy_value(params.model, M, env, params)[1])
    spectrum = cor_len_spectrum_value(env, params, M; method=:channel, nev=5)
    normalized_spectrum = spectrum ./ spectrum[1]
    xi = TeneT._correlation_length_from_spectrum(spectrum)
    return (; energy, xi, spectrum=normalized_spectrum)
end

results = Dict{String, Any}()
for (label, use_su, τ) in (("direct", false, 0.0),
                           ("su_tau_0p01", true, TAU))
    params = smoke_params(label; ifSU=use_su, SUτ=τ)
    Aopt, _, gradient, _, _ = optimise_ipeps(
        copy(Araw0), [χ], params; restriction_ipeps=restriction_ipeps)
    measured = measure_state(Aopt, params)
    results[label] = merge(measured, (; gradient_norm=norm(gradient)))
    println(label, ": ", results[label])
end

save(joinpath(output_root, "smoke_summary.jld2"),
     "seed", SEED,
     "D", D,
     "chi", χ,
     "tau", TAU,
     "J", model.J,
     "h", model.h,
     "raw_initial", Araw0,
     "direct", results["direct"],
     "su_tau_0p01", results["su_tau_0p01"];
     iotype=IOStream)
```

- [ ] **Step 4: Run the static test and the paired CPU smoke**

Run:

```powershell
julia --project=. test/test_c4v_su_ad.jl
$smokeRoot = Join-Path $env:TEMP 'tenet_tfising_c4v_su_smoke'
julia --project=. examples/TFIsing/TFIsing_Square_VUMPS_C4v_SU_smoke.jl $smokeRoot
```

Expected: the focused suite passes; both labels print finite energy, ξ, gradient norm, and normalized leading eigenvalues; `$smokeRoot\smoke_summary.jld2` exists. No ordering between direct and SU energy is required at χ=8.

- [ ] **Step 5: Commit the paired research entrypoint**

```powershell
git add examples/TFIsing/TFIsing_Square_VUMPS_C4v_SU_smoke.jl test/test_c4v_su_ad.jl
git commit -m "Add paired TFIsing C4v SU smoke"
```

### Task 9: Final Regression and Evidence Review

**Files:**
- No new files.

**Interfaces:**
- Consumes: all implementation tasks and the clean repository state.
- Produces: verification evidence sufficient to decide whether D=3/larger-χ experiments are authorized.

- [ ] **Step 1: Run formatting and focused checks**

```powershell
git diff --check
julia --project=. test/test_c4v_su_ad.jl
```

Expected: no whitespace errors and all focused tests pass.

- [ ] **Step 2: Run the complete package suite**

```powershell
julia --project=. -e "using Pkg; Pkg.test()"
```

Expected: the full TeneT.jl suite passes with `test_c4v_su_ad.jl` included.

- [ ] **Step 3: Re-run the paired smoke from a fresh output directory**

```powershell
$finalSmokeRoot = Join-Path $env:TEMP 'tenet_tfising_c4v_su_smoke_final'
julia --project=. examples/TFIsing/TFIsing_Square_VUMPS_C4v_SU_smoke.jl $finalSmokeRoot
```

Expected: direct and SU runs complete from the same raw seed and write finite metrics plus raw/physical iteration checkpoints.

- [ ] **Step 4: Verify commit and worktree scope**

```powershell
git status --short
git log --oneline 9829fcc..HEAD
git diff 9829fcc..HEAD -- src test examples/TFIsing
```

Expected: no unrelated user files appear; implementation changes since the reviewed specification are limited to the paths listed in this plan.

- [ ] **Step 5: Apply completion skills before reporting success**

Invoke `superpowers:verification-before-completion` with the outputs from Steps 1-4, then `superpowers:requesting-code-review`. Do not claim that extrapolation improved; report only correctness, χ=8 stability, paired metrics, and any remaining numerical uncertainty.
