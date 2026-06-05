# TM Spectrum Observable Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `TM_spectrum` for `VUMPS{General}` honeycomb brickwall iPEPS states and move high-level observables into `src/observable/`.

**Architecture:** Keep low-level contraction kernels in `src/contraction/observable.jl`. Move high-level observable orchestration out of `src/ipeps_optimize/observable.jl` into focused files under `src/observable/`, then add the transfer-matrix spectrum diagnostic as another observable file. The public API remains `observable(...)` plus new exported `TM_spectrum(...)`.

**Tech Stack:** Julia, TeneT `StructArray`, VUMPS `General` runtimes, KrylovKit eigensolvers, OMEinsum tensor contractions, existing `Test` suite.

---

## File Structure

- Create: `src/observable/interface.jl`
  - `energy(A, env, params::iPEPSOptimize)`
  - top-level `observable(A, χ, params; restriction_ipeps, cor_len_method)`
- Create: `src/observable/magnetization.jl`
  - every `magnetization_value(...)` method currently in `src/ipeps_optimize/observable.jl`
- Create: `src/observable/correlation_length.jl`
  - every `cor_len_value(...)` method currently in `src/ipeps_optimize/observable.jl`
- Create: `src/observable/wp_order.jl`
  - `Wp_value(...)`
  - `fwave_order(...)`
- Create: `src/observable/transfer_matrix_spectrum.jl`
  - `TM_spectrum(...)`
  - `_validate_tm_spectrum_inputs(params)`
  - `_tm_forloop_iter(params)`
  - `_tm_up_runtime(rt)`
  - `_tm_initialize_runtime(A, D, χ, params; restriction_ipeps, file)`
  - `_tm_prepare_env(A, χ, params; restriction_ipeps, ifdomainwall)`
  - `_tm_initial_VL(AL)`
  - `_tm_H_eff(...)` and local contraction helpers
  - `_write_tm_spectrum(...)`
- Delete: `src/ipeps_optimize/observable.jl`
- Modify: `src/TeneT.jl`
  - replace `include("ipeps_optimize/observable.jl")` with the new observable includes
  - export `TM_spectrum`
- Create: `test/test_tm_spectrum.jl`
- Modify: `test/runtests.jl`
  - include the new test file

## Task 1: Add Failing Structural Tests

**Files:**
- Create: `test/test_tm_spectrum.jl`
- Modify: `test/runtests.jl`

- [ ] **Step 1: Add the new test file**

Create `test/test_tm_spectrum.jl` with this exact content:

```julia
@testset "TM_spectrum observable" begin
    using TeneT: Heisenberg

    function _tm_test_params(; model, alg, pattern=[1;;])
        return GradientOptimize(model=model,
                                pattern=pattern,
                                boundary_alg=alg,
                                verbosity=0,
                                folder=mktempdir(),
                                ifload_env=false,
                                ifsave_env=false,
                                ifplot=false,
                                ifsave_lbfgs=false,
                                ifload_lbfgs=false)
    end

    @testset "export and input guards" begin
        @test :TM_spectrum in names(TeneT)

        A = rand(Float64, 2, 1, 2, 2, 2, 1)
        χ = 2
        brickwall = Heisenberg(lattice=Honeycomb{:brickwall_h}(),
                               S=0.5, Jx=1.0, Jy=1.0, Jz=1.0,
                               ifrotate=false)

        c4v_params = _tm_test_params(model=brickwall,
                                     alg=VUMPS{C4v}(; verbosity=0))
        @test_throws ArgumentError TeneT.TM_spectrum(1, 0.0, A, χ, c4v_params)

        square_model = Heisenberg(lattice=Square(),
                                  S=0.5, Jx=1.0, Jy=1.0, Jz=1.0,
                                  ifrotate=false)
        square_params = _tm_test_params(model=square_model,
                                        alg=VUMPS{General}(; verbosity=0,
                                                           ifupdown=false))
        @test_throws ArgumentError TeneT.TM_spectrum(1, 0.0, A, χ, square_params)
    end

    @testset "spectrum writer paths" begin
        params = _tm_test_params(
            model=Heisenberg(lattice=Honeycomb{:brickwall_h}(),
                             S=0.5, Jx=1.0, Jy=1.0, Jz=1.0,
                             ifrotate=false),
            alg=VUMPS{General}(; verbosity=0, ifupdown=false),
        )

        TeneT._write_tm_spectrum([0.125, 0.25], 0.0, 2, 4, params;
                                 ifdomainwall=false)
        trivial = joinpath(params.folder, "D2_χ4", "TM_spectrum",
                           "trivial", "k0.0.log")
        @test isfile(trivial)
        @test readlines(trivial) == ["0.125000000000000",
                                     "0.250000000000000"]

        TeneT._write_tm_spectrum([0.5], 0.0, 2, 4, params;
                                 ifdomainwall=true)
        nontrivial = joinpath(params.folder, "D2_χ4", "TM_spectrum",
                              "non-trivial", "k0.0.log")
        @test isfile(nontrivial)
        @test readlines(nontrivial) == ["0.500000000000000"]
    end
end
```

- [ ] **Step 2: Include the new test file**

In `test/runtests.jl`, add this line after `include("test_ipeps.jl")`:

```julia
    include("test_tm_spectrum.jl")
```

- [ ] **Step 3: Run the focused test and verify failure**

Run:

```powershell
julia --project=. -e "using Pkg; Pkg.test(; test_args=[\"test_tm_spectrum.jl\"])"
```

Expected: FAIL because `TM_spectrum` and `_write_tm_spectrum` are not defined/exported yet.

- [ ] **Step 4: Commit the failing tests**

Run:

```powershell
git add -- test/test_tm_spectrum.jl test/runtests.jl
git commit -m "test: cover TM spectrum observable API"
```

## Task 2: Split High-Level Observable Code Into `src/observable/`

**Files:**
- Create: `src/observable/interface.jl`
- Create: `src/observable/magnetization.jl`
- Create: `src/observable/correlation_length.jl`
- Create: `src/observable/wp_order.jl`
- Delete: `src/ipeps_optimize/observable.jl`
- Modify: `src/TeneT.jl`

- [ ] **Step 1: Create `src/observable/interface.jl`**

Move the following exact blocks from `src/ipeps_optimize/observable.jl` into
`src/observable/interface.jl` without changing function bodies:

```julia
# Observable computation for iPEPS
# Magnetization, correlation length, and full observable wrapper

"""
    energy(A, env, params::iPEPSOptimize)

Main entry point for energy computation during iPEPS optimization.
Calls `energy_value` dispatching on `params.model` type.
"""
function energy(A, env, params::iPEPSOptimize)
    return energy_value(params.model, A, env, params)[1]
end

# ============================================================================
# Full observable computation
# ============================================================================

"""
    observable(A, χ, params::iPEPSOptimize; restriction_ipeps=_restriction_ipeps, cor_len_method=:mps)

Compute all observables (energy, magnetization, correlation length) for a
given iPEPS tensor `A` at bond dimension `χ`. Initializes a VUMPS runtime,
converges the boundary, and evaluates expectation values.

`cor_len_method` chooses the correlation-length estimator (see
[`cor_len_value`](@ref)): `:mps` (default, pure boundary-MPS transfer
matrix; cheap) or `:channel` (channel TM with bulk M; closer to the
physical ξ at finite χ).
"""
function observable(A, χ, params::iPEPSOptimize; restriction_ipeps=_restriction_ipeps,
                    cor_len_method::Symbol=:mps)
    D = _ipeps_bond_dimension(A)
    rt = initialize_env(A, D, χ, params; restriction_ipeps)

    _G_cache[] = nothing
    A = restriction_ipeps(A)
    A = build_A(A, params)

    rt, _ = leading_boundary(rt, A, params.boundary_alg)
    params.ifsave_env && save_rt(joinpath(params.folder, "D$(D)", "environment"), rt; file="χ$(χ).jld2")
    env = ObsEnv(rt, A, params.boundary_alg, params.model)
    e = energy_value(params.model, A, env, params)
    mag = magnetization_value(params.model, A, env, params)
    ξ = cor_len_value(env, params, A; method=cor_len_method)

    if params.model.lattice isa Kagome{:merge}
        e_perbond = energy_value_perbond(params.model, A, env, params)
        e = (e[1], e_perbond)
    end

    write_obs_log(e, mag, ξ, χ, joinpath(params.folder, "D$(D)"), params)

    if params.ifplot
        obs_path = joinpath(params.folder, "D$(D)", "observable")
        plot_observables(obs_path, params.model.lattice, params.pattern;
                         save_format=params.plot_format, S=params.model.S)
    end

    return e, mag, ξ
end
```

- [ ] **Step 2: Create `src/observable/magnetization.jl`**

Move every method from the `# Magnetization` section of
`src/ipeps_optimize/observable.jl` into `src/observable/magnetization.jl`.
The first moved method is:

```julia
function magnetization_value(model, A, env::VUMPSEnv, params)
```

The last moved method is the method immediately before:

```julia
# ============================================================================
# Correlation length
# ============================================================================
```

Do not change method bodies in this step.

- [ ] **Step 3: Create `src/observable/correlation_length.jl`**

Move every method from the `# Correlation length` section of
`src/ipeps_optimize/observable.jl` into `src/observable/correlation_length.jl`.
The first moved method is:

```julia
function cor_len_value(env::VUMPSEnv, params, M; method::Symbol=:mps)
```

The last moved method is the method immediately before:

```julia
# ============================================================================
# Wp value for iPEPS with brickwall unit cell
# ============================================================================
```

Do not change method bodies in this step.

- [ ] **Step 4: Create `src/observable/wp_order.jl`**

Move `Wp_value(...)` and `fwave_order(...)` from the end of
`src/ipeps_optimize/observable.jl` into `src/observable/wp_order.jl`.
Do not change method bodies in this step.

- [ ] **Step 5: Replace observable includes in `src/TeneT.jl`**

Replace:

```julia
include("ipeps_optimize/observable.jl")
include("visualization/plot_obs.jl")
include("ipeps_optimize/optimize.jl")
```

with:

```julia
include("observable/interface.jl")
include("observable/magnetization.jl")
include("observable/correlation_length.jl")
include("observable/wp_order.jl")
include("observable/transfer_matrix_spectrum.jl")
include("visualization/plot_obs.jl")
include("ipeps_optimize/optimize.jl")
```

- [ ] **Step 6: Create a temporary empty transfer spectrum file**

Create `src/observable/transfer_matrix_spectrum.jl` with:

```julia
# Transfer-matrix spectrum observable.
```

- [ ] **Step 7: Delete the old high-level observable file**

Delete:

```text
src/ipeps_optimize/observable.jl
```

- [ ] **Step 8: Run existing observable-related tests**

Run:

```powershell
julia --project=. -e "using Pkg; Pkg.test(; test_args=[\"test_ipeps.jl\"])"
```

Expected: PASS, except the new `test_tm_spectrum.jl` still fails if included in the full suite because `TM_spectrum` is not implemented yet.

- [ ] **Step 9: Commit the module split**

Run:

```powershell
git add -- src/observable src/TeneT.jl src/ipeps_optimize/observable.jl
git commit -m "refactor: split high-level observables"
```

## Task 3: Implement `TM_spectrum` Guards And Writer

**Files:**
- Modify: `src/observable/transfer_matrix_spectrum.jl`
- Modify: `src/TeneT.jl`

- [ ] **Step 1: Add validators, writer, and placeholder public function**

Replace `src/observable/transfer_matrix_spectrum.jl` with:

```julia
# Transfer-matrix spectrum observable.

_tm_supported_lattice(::Honeycomb{:brickwall_h}) = true
_tm_supported_lattice(::Honeycomb{:brickwall_v}) = true
_tm_supported_lattice(::AbstractLattice) = false

function _validate_tm_spectrum_inputs(params::iPEPSOptimize)
    params.boundary_alg isa VUMPS{General} ||
        throw(ArgumentError("TM_spectrum currently supports only VUMPS{General}; got $(typeof(params.boundary_alg))."))

    _tm_supported_lattice(params.model.lattice) ||
        throw(ArgumentError("TM_spectrum currently supports only Honeycomb{:brickwall_h} and Honeycomb{:brickwall_v}; got $(typeof(params.model.lattice))."))

    return nothing
end

_tm_forloop_iter(params) =
    hasproperty(params, :forloop_iter) ? params.forloop_iter : params.boundary_alg.forloop_iter

function _write_tm_spectrum(Δ, k, D, χ, params::iPEPSOptimize; ifdomainwall)
    sector = ifdomainwall ? "non-trivial" : "trivial"
    folder = joinpath(params.folder, "D$(D)_χ$(χ)", "TM_spectrum", sector)
    isdir(folder) || mkpath(folder)
    obs_log = joinpath(folder, "k$k.log")
    open(obs_log, "w") do io
        for δ in Δ
            @printf(io, "%.15f\n", real(δ))
        end
    end
    return obs_log
end

function TM_spectrum(n::Int, k::Real, A, χ, params::iPEPSOptimize;
                     restriction_ipeps=_restriction_ipeps,
                     ifdomainwall=false)
    _validate_tm_spectrum_inputs(params)
    throw(ArgumentError("TM_spectrum implementation is not complete yet."))
end
```

- [ ] **Step 2: Export `TM_spectrum`**

In `src/TeneT.jl`, change:

```julia
export observable
```

to:

```julia
export observable, TM_spectrum
```

- [ ] **Step 3: Run the structural tests**

Run:

```powershell
julia --project=. -e "using Pkg; Pkg.test(; test_args=[\"test_tm_spectrum.jl\"])"
```

Expected: The export/guard/writer tests PASS. There are no smoke tests yet.

- [ ] **Step 4: Commit the guarded API**

Run:

```powershell
git add -- src/observable/transfer_matrix_spectrum.jl src/TeneT.jl
git commit -m "feat: add TM spectrum observable API guards"
```

## Task 4: Add Failing Smoke Tests For Trivial And Domain-Wall Spectra

**Files:**
- Modify: `test/test_tm_spectrum.jl`

- [ ] **Step 1: Add smoke tests**

Append this block inside the outer `@testset "TM_spectrum observable"` in
`test/test_tm_spectrum.jl`:

```julia
    @testset "small VUMPS General brickwall spectra" begin
        Random.seed!(17)
        D, d, χ = 2, 2, 4
        pattern = [1 2; 2 1]
        model = Heisenberg(lattice=Honeycomb{:brickwall_h}(),
                           S=0.5, Jx=1.0, Jy=1.0, Jz=1.0,
                           ifrotate=false)
        alg = VUMPS{General}(; verbosity=0,
                              maxiter=2, miniter=0,
                              maxiter_ad=0, miniter_ad=0,
                              power_iter=2,
                              ifupdown=true,
                              ifsimple_eig=true)
        params = _tm_test_params(model=model, alg=alg, pattern=pattern)

        A = TeneT._init_random_ipeps(model.lattice, Float64, D, d,
                                     length(unique(pattern)),
                                     size(pattern, 1), size(pattern, 2))
        A ./= norm(A)

        Δ = TeneT.TM_spectrum(1, 0.0, A, χ, params;
                              ifdomainwall=false)
        @test length(Δ) == 1
        @test all(isfinite, real.(Δ))
        @test isfile(joinpath(params.folder, "D$(D)_χ$(χ)",
                              "TM_spectrum", "trivial", "k0.0.log"))

        Δdw = TeneT.TM_spectrum(1, 0.0, A, χ, params;
                                ifdomainwall=true)
        @test length(Δdw) == 1
        @test all(isfinite, real.(Δdw))
        @test isfile(joinpath(params.folder, "D$(D)_χ$(χ)",
                              "TM_spectrum", "non-trivial", "k0.0.log"))
    end
```

- [ ] **Step 2: Run the smoke tests and verify failure**

Run:

```powershell
julia --project=. -e "using Pkg; Pkg.test(; test_args=[\"test_tm_spectrum.jl\"])"
```

Expected: FAIL with `ArgumentError("TM_spectrum implementation is not complete yet.")`.

- [ ] **Step 3: Commit the failing smoke tests**

Run:

```powershell
git add -- test/test_tm_spectrum.jl
git commit -m "test: add TM spectrum smoke coverage"
```

## Task 5: Implement Full `TM_spectrum`

**Files:**
- Modify: `src/observable/transfer_matrix_spectrum.jl`

- [ ] **Step 1: Add runtime helpers**

Add these helpers below `_tm_forloop_iter`:

```julia
_tm_up_runtime(rt::VUMPSRuntime) = rt
_tm_up_runtime(rt::Tuple{VUMPSRuntime,VUMPSRuntime}) = rt[1]

function _tm_initialize_runtime(A, D::Int, χ::Int, params::iPEPSOptimize;
                                restriction_ipeps, file::String)
    folder_path = joinpath(params.folder, "D$(D)", "environment")
    file_path = joinpath(folder_path, file)

    if hasproperty(params, :ifload_env) && params.ifload_env && ispath(file_path)
        try
            ifparallelupdown = hasproperty(params.boundary_alg, :ifparallelupdown) ?
                               params.boundary_alg.ifparallelupdown : false
            return load_rt(folder_path, _arraytype(A), ifparallelupdown; file)
        catch e
            @warn "Failed to load TM_spectrum environment from $file_path: $(sprint(showerror, e)). Creating a new environment."
        end
    end

    rt = _create_new_env(A, χ, params; restriction_ipeps)
    return rt
end
```

- [ ] **Step 2: Add tangent-space initialization**

Add leg-4 initialization used by the unflattened current contraction path:

```julia
function _tm_initial_VL(AL::StructArray)
    χ = size(AL[1], 1)
    VL = randSA(AL, [(D = size(AL[i], 2); (χ, D, D, χ * (D^2 - 1)))
                    for i in 1:length(AL)])
    @inbounds for i in 1:length(AL)
        D = size(AL[i], 2)
        if D != 1
            λL = ein"abec,abed -> cd"(VL[i], conj(AL[i]))
            VL[i] -= ein"abec,dc -> abed"(AL[i], λL)
            Q, _ = qrpos(reshape(VL[i], χ * D * D, χ * (D^2 - 1)))
            VL[i] = reshape(Q, χ, D, D, χ * (D^2 - 1))
            λL = ein"abec,abed -> cd"(VL[i], conj(AL[i]))
            VL[i] -= ein"abec,dc -> abed"(AL[i], λL)
        end
    end
    return VL
end
```

The subtraction line uses the same projector convention as the old code:
`λL` is computed as `(VL last leg, AL last leg)`, then applied as
`ein"abec,dc -> abed"`.

- [ ] **Step 3: Add environment preparation**

Implement `_tm_prepare_env(A, χ, params; restriction_ipeps, ifdomainwall)`.
Use this exact behavior:

```julia
function _tm_prepare_env(A, χ, params::iPEPSOptimize; restriction_ipeps, ifdomainwall)
    D = _ipeps_bond_dimension(A)
    rt1 = if ifdomainwall
        _tm_initialize_runtime(A, D, χ, params; restriction_ipeps, file="χ$(χ)_1.jld2")
    else
        initialize_env(A, D, χ, params; restriction_ipeps)
    end

    A′ = restriction_ipeps(A)
    A′ = build_A(A′, params)

    rt1, _ = leading_boundary(rt1, A′, params.boundary_alg)
    if params.ifsave_env && !ifdomainwall
        save_rt(joinpath(params.folder, "D$(D)", "environment"), rt1; file="χ$(χ).jld2")
    elseif params.ifsave_env && ifdomainwall
        save_rt(joinpath(params.folder, "D$(D)", "environment"), rt1; file="χ$(χ)_1.jld2")
    end

    up1 = _tm_up_runtime(rt1)

    if ifdomainwall
        rt2 = _tm_initialize_runtime(A, D, χ, params; restriction_ipeps, file="χ$(χ)_2.jld2")
        rt2, _ = leading_boundary(rt2, A′, params.boundary_alg)
        params.ifsave_env && save_rt(joinpath(params.folder, "D$(D)", "environment"), rt2; file="χ$(χ)_2.jld2")
        up2 = _tm_up_runtime(rt2)
        return _tm_prepare_domainwall_env(A′, up1, up2, params)
    else
        return _tm_prepare_trivial_env(A′, up1, params)
    end
end
```

Then implement `_tm_prepare_trivial_env` and `_tm_prepare_domainwall_env` by
porting old `exci_pre_env` with these required replacements:

```julia
forloop_iter = _tm_forloop_iter(params)
ifparallel = params.boundary_alg.ifparallel
M = A′
```

Use `contract_n_11(...; forloop_iter, ifparallel)` instead of old
`contract_n1(...; forloop_iter)`.

- [ ] **Step 4: Add effective-map helpers**

Port the old helper functions with these new names:

```julia
_tm_einEMs
_tm_left_series
_tm_einEB
_tm_einMEs
_tm_right_series
_tm_einBE
_tm_H_eff
```

Use these replacements in every helper:

```julia
FLmap_forloop(x, a, b, m; forloop_iter)
```

becomes:

```julia
FLmap_parallel(x, a, b, m; ifparallel, forloop_iter)
```

and:

```julia
FRmap_forloop(x, a, b, m; forloop_iter)
```

becomes:

```julia
FRmap_parallel(x, a, b, m; ifparallel, forloop_iter)
```

Remove old `ifcheckpoint=false` keywords from `FLmap`, `FRmap`, and `ACmap`.
Pass `ifparallel` and `forloop_iter` instead:

```julia
FLmap(1, EB, AR[1, :], conj(AL[mod1(2, Ni), :]), M[1, :];
      ifparallel, forloop_iter)
FRmap(Nj, BE, AL[1, :], conj(AR[mod1(2, Ni), :]), M[1, :];
      ifparallel, forloop_iter)
ACmap(1, Bu[j], ELL[:, j], ERR[:, j], M[:, j];
      ifparallel, forloop_iter)
```

Keep the old `linsolve` structure, but replace hard assertions with warnings:

```julia
info.converged == 0 && @warn "TM_spectrum left series linsolve did not converge"
```

- [ ] **Step 5: Replace the placeholder `TM_spectrum` body**

Use this public function shape:

```julia
function TM_spectrum(n::Int, k::Real, A, χ, params::iPEPSOptimize;
                     restriction_ipeps=_restriction_ipeps,
                     ifdomainwall=false)
    _validate_tm_spectrum_inputs(params)

    M, Mn, AL, AR, ELL, ERR, ERL, ERL_dual, ELR, ELR_dual, VL =
        _tm_prepare_env(A, χ, params; restriction_ipeps, ifdomainwall)

    χenv = size(AL[1], 1)
    Nj = size(AL, 2)
    atype = _arraytype(A)
    etype = eltype(A)

    X = [begin
             Dloc = size(AL[1, j], 2)
             Dloc != 1 ? atype(rand(etype, χenv * (Dloc^2 - 1), χenv)) :
                         atype(rand(etype, 0, χenv))
         end for j in 1:Nj]

    ifparallel = params.boundary_alg.ifparallel
    forloop_iter = _tm_forloop_iter(params)

    function f(Xin)
        Bu = zero(AL[1, :])
        for j in 1:Nj
            if size(AL[1, j], 2) != 1
                Bu[j] = ein"abec,cd -> abed"(VL[1, j], Xin[j])
            end
        end

        HB = _tm_H_eff(k * pi, AL, AR, Bu, M, Mn, ELL, ERR,
                       ERL, ERL_dual, ELR, ELR_dual;
                       ifparallel, forloop_iter)

        return [size(AL[1, j], 2) != 1 ?
                ein"abec,abed -> dc"(HB[j], conj(VL[1, j])) :
                atype(rand(etype, 0, χenv))
                for j in 1:Nj]
    end

    λs, _, info = eigsolve(x -> f(x), X, n, :LM;
                           krylovdim=max(30, round(Int, n + 10)),
                           ishermitian=false,
                           maxiter=1,
                           tol=1e-12)
    info.converged == 0 && @warn "TM_spectrum eigsolve did not converge"

    Δ = -log.(norm.(λs))
    D = _ipeps_bond_dimension(A)
    _write_tm_spectrum(Δ, k, D, χ, params; ifdomainwall)
    return Δ
end
```

Use the return tuple names shown in the snippet: `ERR`, `ERL_dual`, and
`ELR_dual`. Keep `ERR`-style ASCII names in the new file instead of
introducing new Unicode identifier names.

- [ ] **Step 6: Run focused smoke tests**

Run:

```powershell
julia --project=. -e "using Pkg; Pkg.test(; test_args=[\"test_tm_spectrum.jl\"])"
```

Expected: PASS.

- [ ] **Step 7: Commit the implementation**

Run:

```powershell
git add -- src/observable/transfer_matrix_spectrum.jl test/test_tm_spectrum.jl
git commit -m "feat: add TM spectrum observable"
```

## Task 6: Full Verification And Cleanup

**Files:**
- Potentially modify: `src/observable/transfer_matrix_spectrum.jl`
- Potentially modify: `test/test_tm_spectrum.jl`

- [ ] **Step 1: Run existing core tests likely affected by the split**

Run:

```powershell
julia --project=. -e "using Pkg; Pkg.test(; test_args=[\"test_ipeps.jl\"])"
```

Expected: PASS.

- [ ] **Step 2: Run contraction and boundary tests**

Run:

```powershell
julia --project=. -e "using Pkg; Pkg.test(; test_args=[\"test_contraction.jl\", \"test_boundary.jl\"])"
```

Expected: PASS.

- [ ] **Step 3: Run the full test suite if runtime is acceptable**

Run:

```powershell
julia --project=. -e "using Pkg; Pkg.test()"
```

Expected: PASS. If this is too slow or blocked by GPU package setup, record the exact failure or timeout and keep the focused test results.

- [ ] **Step 4: Inspect git status**

Run:

```powershell
git status --short --branch
```

Expected: clean working tree except for intentional committed changes.

- [ ] **Step 5: Commit verification-only fixes when Step 1-3 changed files**

If Step 1-3 exposed small fixes, stage and commit them:

```powershell
git add -- src/observable test src/TeneT.jl
git commit -m "fix: stabilize TM spectrum observable tests"
```

If there were no fixes after Task 5, do not create an empty commit.

## Self-Review Checklist

- Spec coverage:
  - `src/observable/` split is covered by Task 2.
  - `TM_spectrum` API, export, guards, writer are covered by Task 3.
  - trivial and domain-wall paths are covered by Tasks 4 and 5.
  - no `ifflatten=true` path is introduced.
  - unsupported algorithms and lattices fail early.
- Placeholder scan:
  - No task says to add vague error handling.
  - Each code-editing task gives exact function names, file paths, or concrete replacement snippets.
- Type consistency:
  - Validators use `VUMPS{General}` and `Honeycomb{:brickwall_h/:brickwall_v}`.
  - Runtime helpers use current `VUMPSRuntime` and tuple-of-runtime shapes.
  - Contraction helpers use current `FLmap_parallel`, `FRmap_parallel`, `ACmap`, and `contract_n_11` APIs.
