# TeneT.jl Unified iPEPS Package — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rewrite TeneT.jl to be a unified iPEPS optimization package supporting single and large unit cells, multiple boundary algorithms, multiple lattice types, and all Hamiltonian models from TeneT_demo and ADC4PEPS.

**Architecture:** Bottom-up rewrite in 9 phases — branch setup → types → StructArray → contraction → boundary → models → iPEPS optimization → autodiff → integration tests. Each phase builds on the previous. The package uses Julia's multiple dispatch to unify single/multi-cell and different lattice types behind a single API.

**Tech Stack:** Julia 1.x, Zygote (AD), TensorOperations/OMEinsum (contraction), OptimKit (LBFGS), KrylovKit (eigensolver), CUDA/AMDGPU (GPU), MPI (parallelization), JLD2 (I/O), Parameters (@kwdef)

**Reference code locations:**
- TeneT.jl (current): `D:/1 - research/1.26 - iPEPS_opt/TeneT.jl/src/`
- TeneT_demo: `D:/1 - research/1.26 - iPEPS_opt/TeneT_demo/src/`
- ADC4PEPS: `D:/1 - research/1.26 - iPEPS_opt/ADC4PEPS/src/`

---

## Phase 1: Branch Setup and Directory Scaffolding

### Task 1.1: Create unified branch from parallel-matmul

**Files:**
- Modify: `Project.toml`

**Step 1: Create branch**

```bash
cd "D:/1 - research/1.26 - iPEPS_opt/TeneT.jl"
git checkout origin/parallel-matmul -b iPEPS-unified
```

**Step 2: Cherry-pick Plaquette-specific contraction changes**

The Plaquette branch simplified VUMPS to left-canonical only and added `ifparallel` to forloop. Since parallel-matmul already has `forloop_parallel_MPI.jl`, we mainly need the Plaquette contraction simplifications. Review and selectively merge:

```bash
git diff origin/Array-of-Array..origin/Plaquette -- src/contraction/basic.jl > /tmp/plaquette_basic.patch
# Review the patch, then apply relevant parts manually
```

Key changes to bring from Plaquette:
- Simplified one-liner `@tensor` definitions in `basic.jl`
- ASCII diagram docstrings
- The `vumps_step()` unified function (replacing separate `vumps_step_power` and `vumps_step_Hermitian`)

**Step 3: Restructure src/ directory**

```bash
cd src/
mkdir -p boundary models ipeps autodiff utils
```

**Step 4: Update Project.toml**

Add missing dependencies from ADC4PEPS and TeneT_demo:
- `OptimKit` (LBFGS optimizer)
- `LineSearches` (for LBFGS)
- `ForwardDiff` + `ForwardDiffChainRules` (for preconditioning)
- `MPI` (already in parallel-matmul)

Reference: ADC4PEPS `Project.toml` for exact version pins.

**Step 5: Commit**

```bash
git add -A
git commit -m "scaffold: create iPEPS-unified branch with new directory structure"
```

---

## Phase 2: Core Types

### Task 2.1: Create types.jl with all abstract types and aliases

**Files:**
- Create: `src/types.jl`

**Step 1: Write types.jl**

Combine type definitions from all three codebases. Reference:
- TeneT.jl `src/utilities.jl` lines defining `leg2`, `leg3`, `leg4`, `leg5`, `leg8`
- ADC4PEPS `src/contraction/interface.jl` for `Algorithm` subtypes
- TeneT_demo `src/TeneT_demo.jl` for `iPEPSOptimize`

```julia
# src/types.jl

# Tensor leg type aliases
const leg2{T} = AbstractArray{T, 2}
const leg3{T} = AbstractArray{T, 3}
const leg4{T} = AbstractArray{T, 4}
const leg5{T} = AbstractArray{T, 5}
const leg8{T} = AbstractArray{T, 8}

# Lattice types
abstract type AbstractLattice end
struct Square <: AbstractLattice end
struct Honeycomb <: AbstractLattice end
struct KagomeLattice <: AbstractLattice end

# Contraction modes for VUMPS specialization
abstract type ContractionMode end
struct General <: ContractionMode end
struct Plaquette <: ContractionMode end

# Boundary algorithm base type
abstract type Algorithm end

# iPEPS optimization base type
abstract type iPEPSOptimize end

# Hamiltonian model base type
abstract type HamiltonianModel end
```

**Step 2: Commit**

```bash
git add src/types.jl
git commit -m "feat: add core type definitions (lattice, contraction mode, algorithm, model)"
```

---

### Task 2.2: Create defaults.jl

**Files:**
- Modify: `src/defaults.jl` (already exists from TeneT.jl)

Reference: both codebases have identical `Defaults` module. Keep as-is.

**Step 1: Verify existing defaults.jl is sufficient**

Current `src/defaults.jl` already has VERBOSE_NONE through VERBOSE_ALL. No changes needed.

---

## Phase 3: StructArray (Reuse from TeneT.jl)

### Task 3.1: Verify StructArray works unchanged

**Files:**
- Keep: `src/structarray/base.jl`, `src/structarray/initial.jl`, `src/structarray/buffer.jl`

StructArray from Array-of-Array branch is mature. Verify tests pass:

```bash
julia --project -e 'using Pkg; Pkg.test()'
```

If tests fail due to missing dependencies, fix Project.toml first.

**Step 1: Run existing StructArray tests**

```bash
julia --project -e '
    using Test
    include("test/test_setup.jl")
    include("test/structarray.jl")
'
```

**Step 2: Commit if any fixes needed**

---

## Phase 4: Contraction Layer

### Task 4.1: Rewrite basic.jl — core tensor contraction maps

**Files:**
- Rewrite: `src/contraction/basic.jl`

Merge the best from all branches:
- Plaquette branch: clean one-liner `@tensor` definitions with ASCII diagrams
- parallel-matmul branch: existing leg4/leg5/leg8 support
- ADC4PEPS `src/contraction/unit_contraction/basic.jl`: ALCtoAC, CTtoT helper functions

Key functions to include:
```julia
# From Plaquette (simplified one-liners):
FLmap(FL, ALu, ALd, M::leg4)   # Left environment update
FRmap(FR, ARu, ARd, M::leg4)   # Right environment update
ACmap(AC, FL, M, FR::leg4)     # Active center update
Cmap(C, FL, FR)                # Center tensor update
Lmap(L, ALu, ALd)              # Left overlap
Rmap(R, ARu, ARd)              # Right overlap

# Support leg4, leg5, leg8 variants (from parallel-matmul)
# Add ASCII diagram docstrings (from Plaquette)

# From ADC4PEPS:
ALCtoACmap(AL, C)              # Convert AL,C to AC
Mumap(M, AL, AR)               # MPO update map
```

Reference files:
- Plaquette `src/contraction/basic.jl` for clean @tensor one-liners
- parallel-matmul `src/contraction/basic.jl` for leg5/leg8 variants
- ADC4PEPS `src/contraction/unit_contraction/basic.jl` for ALCtoAC, Mumap

**Step 1: Write the rewritten basic.jl with all contraction maps**

**Step 2: Write test**

```julia
# test/test_contraction.jl
using Test, TeneT, LinearAlgebra
@testset "basic contractions" begin
    # Test FLmap with random tensors
    D, χ = 2, 4
    FL = randn(ComplexF64, χ, D^2, χ)
    ALu = randn(ComplexF64, χ, D^2, χ)
    ALd = randn(ComplexF64, χ, D^2, χ)
    M = randn(ComplexF64, D^2, D^2, D^2, D^2)
    result = FLmap(FL, ALu, ALd, M)
    @test size(result) == (χ, D^2, χ)
end
```

**Step 3: Run test, verify pass**

**Step 4: Commit**

```bash
git add src/contraction/basic.jl test/test_contraction.jl
git commit -m "feat: rewrite basic contraction maps with unified leg support"
```

---

### Task 4.2: Port forloop_parallel_MPI.jl

**Files:**
- Keep/adapt: `src/contraction/forloop_parallel_MPI.jl` (from parallel-matmul)

This file already exists on the parallel-matmul branch. Verify it works and rename if needed.

Reference: parallel-matmul `src/contraction/forloop_parallel_MPI.jl` (268 lines)

Key functions:
- `split_count()`, `split_ranges()` for MPI load distribution
- `forloop()`, `parallel()` for loop/MPI splitting
- `FLmap_parallel()`, `FRmap_parallel()`, `ACmap_parallel()`, `ACdmap_parallel()`, `Mumap_parallel()`

**Step 1: Verify file exists and adapt imports**

**Step 2: Commit if changes needed**

---

### Task 4.3: Create observable contraction functions

**Files:**
- Create: `src/contraction/observable.jl`

Merge observable contraction functions from both codebases.

Reference:
- TeneT_demo `src/contraction.jl`: `oc_H_leg3/leg4`, `oc_V_leg3/leg4`, `contract_n2_H/V`, `contract_o2_H/V`, `contract_n1`, `contract_o1`, `contract_n_D`, `contract_o_D1/D2`, `contract_n3_H/V`, `contract_o3_H/V`, `oc_Q_4_corner`
- ADC4PEPS `src/ipeps_optimization/contraction.jl`: `oc1_leg4`, `contract_n1`, `contract_o1`, `oc_H_leg4`, `oc_Q_4_corner`, `oc_D_leg4`

For overlapping functions (e.g., both have `oc_H_leg4`), compare implementations and keep the cleaner/more general one. TeneT_demo has more variants (leg3 support, 3-site contractions for J1J2J3).

```julia
# src/contraction/observable.jl

# Two-site horizontal contractions
oc_H_leg3(...)  # from TeneT_demo
oc_H_leg4(...)  # merge both

# Two-site vertical contractions
oc_V_leg3(...)  # from TeneT_demo
oc_V_leg4(...)  # from TeneT_demo

# Normalization contractions
contract_n1(...)     # single-site norm
contract_n2_H(...)   # two-site horizontal norm
contract_n2_V(...)   # two-site vertical norm
contract_n_D(...)    # diagonal norm (for J1J2)

# Observable contractions
contract_o1(...)     # single-site observable
contract_o2_H(...)   # two-site horizontal observable
contract_o2_V(...)   # two-site vertical observable
contract_o_D1(...)   # diagonal observable (for J1J2)
contract_o_D2(...)   # diagonal observable variant

# Three-site contractions (for J1J2J3)
contract_n3_H(...)
contract_o3_H(...)
contract_n3_V(...)
contract_o3_V(...)

# Corner contractions
oc_Q_4_corner(...)   # 2x2 corner
```

**Step 1: Write observable.jl merging both implementations**
**Step 2: Write tests for key contractions**
**Step 3: Commit**

```bash
git add src/contraction/observable.jl test/test_contraction_obs.jl
git commit -m "feat: add observable contraction functions (merged from both codebases)"
```

---

## Phase 5: Boundary Algorithms

### Task 5.1: Create algorithm.jl — algorithm type definitions

**Files:**
- Create: `src/boundary/algorithm.jl`

Reference: ADC4PEPS `src/contraction/interface.jl` for FPCTM, CTMRG, QRCTM, VUMPS, PT structs.

```julia
# src/boundary/algorithm.jl

# VUMPS with ContractionMode type parameter
@kwdef mutable struct VUMPS{M<:ContractionMode} <: Algorithm
    mode::M = General()
    tol::Float64 = 1e-10
    maxiter::Int = 10
    miniter::Int = 1
    maxiter_ad::Int = 10
    miniter_ad::Int = 3
    forloop_iter::Int = 1
    power_iter::Int = 1
    power_iter_ad::Int = 5
    power_iter_obs::Int = 20
    show_every::Int = 10
    verbosity::Int = Defaults.verbosity
    ifupdown::Bool = true
    ifdownfromup::Bool = false
    ifparallel::Bool = false
    ifsimple_eig::Bool = true
    ifcheckpoint::Bool = false
end

# From ADC4PEPS
@kwdef mutable struct CTMRG <: Algorithm
    maxiter::Int = 1000
    maxiter_ad::Int = 100
    tol::Float64 = 1e-6
    verbosity::Int = Defaults.verbosity
end

@kwdef mutable struct QRCTM <: Algorithm
    maxiter::Int = 1000
    maxiter_ad::Int = 100
    tol::Float64 = 1e-6
    verbosity::Int = Defaults.verbosity
end

@kwdef mutable struct FPCTM <: Algorithm
    maxiter::Int = 1000
    maxiter_ad::Int = 100
    tol::Float64 = 1e-6
    verbosity::Int = Defaults.verbosity
end

@kwdef mutable struct PT <: Algorithm
    maxiter::Int = 1000
    maxiter_ad::Int = 100
    tol::Float64 = 1e-6
    verbosity::Int = Defaults.verbosity
end
```

**Step 1: Write algorithm.jl**
**Step 2: Commit**

---

### Task 5.2: Create environment.jl — environment structs

**Files:**
- Create: `src/boundary/environment.jl`

Merge environment structs from both codebases.

Reference:
- TeneT.jl `src/environment.jl`: `VUMPSEnv`, `VUMPSRuntime` (with AL, AR, C, FL, FR)
- ADC4PEPS `src/contraction/environment.jl`: `CTMEnv` (C, T), `VUMPSEnv` (AL, C, T)

```julia
# src/boundary/environment.jl

# VUMPS environment (from TeneT.jl, expanded)
struct VUMPSRuntime{AT, CT, FT}
    AL::AT    # Left-canonical tensor
    AR::AT    # Right-canonical tensor
    C::CT     # Center matrix
    FL::FT    # Left environment
    FR::FT    # Right environment
end

struct VUMPSEnv{AT, FT}
    ACu::AT; ARu::AT   # Up environment
    ACd::AT; ARd::AT   # Down environment
    FLu::FT; FRu::FT   # Up boundaries
    FLo::FT; FRo::FT   # Observable boundaries
end

# CTM environment (from ADC4PEPS)
struct CTMEnv{CT, ET}
    C::CT     # Corner tensors
    T::ET     # Edge tensors
end
```

**Step 1: Write environment.jl**
**Step 2: Commit**

---

### Task 5.3: Rewrite vumps.jl — VUMPS algorithm with ContractionMode

**Files:**
- Rewrite: `src/boundary/vumps.jl`

This is the core boundary algorithm. Merge from:
- TeneT.jl `src/environment.jl` (730 lines): canonical forms, fixed-point environments, VUMPS steps
- TeneT.jl `src/vumpsruntime.jl` (275 lines): VUMPS iteration, leading_boundary
- Plaquette branch: simplified `vumps_step()`

Key functions:
```julia
# Canonical form computation
left_canonical(A; kwargs...)
right_canonical(A; kwargs...)
LRtoC(L, R)

# Fixed-point environments
leftenv(AL, M, FL; alg)
rightenv(AR, M, FR; alg)

# VUMPS steps - dispatch on ContractionMode
vumps_step(rt, M, alg::VUMPS{General})
vumps_step(rt, M, alg::VUMPS{Plaquette})  # 2x2 plaquette specialization

# VUMPS iteration loop
vumps_itr(rt, M, alg)

# Public API
leading_boundary(rt, M, alg::VUMPS)
init_VUMPSRuntime(M, alg::VUMPS; kwargs...)
```

For `VUMPS{Plaquette}`, the `vumps_step` contracts 2×2 blocks before environment updates, following the Plaquette branch logic.

**Step 1: Write vumps.jl**
**Step 2: Write test**

```julia
# test/test_boundary/test_vumps.jl
@testset "VUMPS convergence" begin
    # Test with simple 2D classical Ising transfer matrix
    # Verify leading_boundary converges
    # Check energy against known value
end
```

**Step 3: Run test, verify pass**
**Step 4: Commit**

```bash
git add src/boundary/vumps.jl test/test_boundary/test_vumps.jl
git commit -m "feat: rewrite VUMPS with ContractionMode dispatch (General + Plaquette)"
```

---

### Task 5.4: Port CTMRG from ADC4PEPS

**Files:**
- Create: `src/boundary/ctmrg.jl`

Reference: ADC4PEPS `src/contraction/environment.jl` — functions with `alg::CTMRG` dispatch, plus `src/contraction/runtime.jl` for convergence loop.

Key functions:
```julia
environment(M, alg::CTMRG)        # Main entry point
leftmove(env, M, alg::CTMRG)      # Left move step
# Convergence loop from runtime.jl
```

**Step 1: Port CTMRG implementation**
**Step 2: Write test**
**Step 3: Commit**

---

### Task 5.5: Port QRCTM from ADC4PEPS

**Files:**
- Create: `src/boundary/qrctm.jl`

Reference: ADC4PEPS `src/contraction/environment.jl` — functions with `alg::QRCTM` dispatch.

**Step 1: Port QRCTM implementation**
**Step 2: Commit**

---

### Task 5.6: Port FPCTM from ADC4PEPS

**Files:**
- Create: `src/boundary/fpctm.jl`

Reference: ADC4PEPS `src/contraction/environment.jl` — functions with `alg::FPCTM` dispatch.

**Step 1: Port FPCTM implementation**
**Step 2: Commit**

---

### Task 5.7: Port PT from ADC4PEPS

**Files:**
- Create: `src/boundary/pt.jl`

Reference: ADC4PEPS `src/contraction/environment.jl` — functions with `alg::PT` dispatch.

**Step 1: Port PT implementation**
**Step 2: Commit**

---

## Phase 6: Hamiltonian Models

### Task 6.1: Create lattice.jl and models.jl base

**Files:**
- Create: `src/models/lattice.jl`
- Create: `src/models/models.jl`

```julia
# src/models/lattice.jl
# (Already defined in types.jl, re-export here if needed)

# src/models/models.jl
# Spin operator constructors (from TeneT_demo hamiltonian_models.jl)
function const_Sx(S) ... end
function const_Sy(S) ... end
function const_Sz(S) ... end

# Generic interface
function hamiltonian(model::HamiltonianModel) end
function hamiltonian_trunc(model::HamiltonianModel) end
```

Reference: TeneT_demo `src/hamiltonian_models.jl` for `const_Sx/Sy/Sz` implementations.

**Step 1: Write lattice.jl and models.jl**
**Step 2: Commit**

---

### Task 6.2: Implement Heisenberg model with lattice parameter

**Files:**
- Create: `src/models/heisenberg.jl`

```julia
@kwdef struct Heisenberg{L<:AbstractLattice} <: HamiltonianModel
    lattice::L = Square()
    S::Real = 0.5
    Jx::Real = 1.0
    Jy::Real = 1.0
    Jz::Real = 1.0
end

function hamiltonian(model::Heisenberg)
    # Return two-body Hamiltonian h
    # Reference: TeneT_demo hamiltonian_models.jl lines for Heisenberg
end
```

Reference: TeneT_demo `src/hamiltonian_models.jl` for exact implementation.

**Step 1: Write heisenberg.jl**
**Step 2: Write test**

```julia
# test/test_models/test_heisenberg.jl
@testset "Heisenberg Hamiltonian" begin
    model = Heisenberg(Jx=-1.0, Jy=-1.0, Jz=1.0)
    h = hamiltonian(model)
    @test size(h) == (2, 2, 2, 2)  # spin-1/2
    @test h ≈ conj(permutedims(h, (2,1,4,3)))  # Hermitian
end
```

**Step 3: Run test, verify pass**
**Step 4: Commit**

---

### Task 6.3: Implement J1J2 model

**Files:**
- Create: `src/models/j1j2.jl`

Reference: TeneT_demo `src/hamiltonian_models.jl` for J1J2 struct and hamiltonian function.

**Step 1: Write j1j2.jl**
**Step 2: Write test**
**Step 3: Commit**

---

### Task 6.4: Implement J1J2J3 model

**Files:**
- Create: `src/models/j1j2j3.jl`

Reference: TeneT_demo `src/hamiltonian_models.jl`.

**Step 1: Write j1j2j3.jl**
**Step 2: Commit**

---

### Task 6.5: Implement Shastry-Sutherland model

**Files:**
- Create: `src/models/shastry_sutherland.jl`

Reference: TeneT_demo `src/hamiltonian_models.jl` for `SS` struct.

**Step 1: Write shastry_sutherland.jl**
**Step 2: Commit**

---

### Task 6.6: Implement Kagome model

**Files:**
- Create: `src/models/kagome.jl`

Reference: TeneT_demo `src/hamiltonian_models.jl` for `Kagome` struct. Note: Kagome uses `hamiltonian_onsite`, `hamiltonian_right`, `hamiltonian_down` instead of a single two-body `hamiltonian`.

**Step 1: Write kagome.jl**
**Step 2: Commit**

---

## Phase 7: iPEPS Optimization Layer

### Task 7.1: Create interface.jl — optimization parameter structs

**Files:**
- Create: `src/ipeps/interface.jl`

```julia
# src/ipeps/interface.jl

@kwdef mutable struct GradientOptimize <: iPEPSOptimize
    model::HamiltonianModel
    pattern::Matrix{Int}
    boundary_alg::Algorithm = VUMPS()
    optimizer = LBFGS(200; maxiter=100, gradtol=1e-7,
                       linesearch=HagerZhangLineSearch(linesearchorder=6))
    verbosity::Int = Defaults.VERBOSE_WARN
    maxiter::Int = 10
    tol::Real = 1e-10
    ifSU::Bool = false
    SUτ::Real = -0.05
    ifprecondition::Bool = false
    iter_precond::Int = 10
    reuse_env::Bool = true
    folder::String = "data/"
end

@kwdef mutable struct SUOptimize <: iPEPSOptimize
    model::HamiltonianModel
    pattern::Matrix{Int}
    boundary_alg::Algorithm = VUMPS()
    SUτ::Real = -0.05
    maxiter::Int = 100
    tol::Real = 1e-6
end

@kwdef mutable struct FUOptimize <: iPEPSOptimize
    model::HamiltonianModel
    pattern::Matrix{Int}
    boundary_alg::Algorithm = VUMPS()
end
```

Reference:
- TeneT_demo `src/optimise_ipeps.jl` for GradientOptimize fields
- TeneT_demo `src/SUFU.jl` for SUOptimize, FUOptimize
- ADC4PEPS `src/ipeps_optimization/interface.jl` for additional fields (iffixedpoint, α, β)

**Step 1: Write interface.jl**
**Step 2: Commit**

---

### Task 7.2: Create build.jl — build_A and build_M (lattice mapping)

**Files:**
- Create: `src/ipeps/build.jl`

This is the lattice mapping layer. `build_M` transforms non-square lattice tensors to an effective square lattice.

```julia
# src/ipeps/build.jl

# build_A: optional preprocessing (SU parameterization)
function build_A(A, params::iPEPSOptimize)
    if params.ifSU
        return SU_parameterization(A, params)
    else
        return A
    end
end

# build_M: lattice-dependent double-layer tensor construction
function build_M(A, model::HamiltonianModel)
    build_M(A, model, model.lattice)
end

# Square lattice: M[i] = A[i] ⊗ conj(A[i])
function build_M(A, model, ::Square)
    # Reference: TeneT_demo src/build_A_M.jl build_M function
    # For each unique tensor in StructArray:
    #   M[i] = ein"abcde,fgche -> (af)(bg)(ch)(de)"(A[i], conj(A[i]))
end

# Honeycomb lattice: coarse-grain to effective square lattice
function build_M(A, model, ::Honeycomb)
    # Contract honeycomb tensors into effective square-lattice tensors
end

# Kagome lattice: absorb triangle sites into effective square tensor
function build_M(A, model::Kagome, ::KagomeLattice)
    # Reference: TeneT_demo build_A_M.jl for Kagome-specific logic
    # Three physical sites per unit cell absorbed into one large tensor
end
```

Reference:
- TeneT_demo `src/build_A_M.jl` for `build_A`, `build_M` implementations
- TeneT_demo `src/SU_parameterization.jl` for SU-related `build_A` variants

**Step 1: Write build.jl**
**Step 2: Write test for Square lattice build_M**
**Step 3: Commit**

---

### Task 7.3: Create init.jl — iPEPS initialization

**Files:**
- Create: `src/ipeps/init.jl`

```julia
# src/ipeps/init.jl

function init_ipeps(; atype=Array, etype=ComplexF64, No=1, D, d=2, χ, pattern, params)
    # Load from file if exists, otherwise random initialization
    # Reference: TeneT_demo src/init_ipeps_env.jl
    # Reference: ADC4PEPS src/ipeps_optimization/init_ipeps.jl
end

function initialize_env(A, χ, params; restriction_ipeps=identity)
    # Initialize boundary environment (VUMPS runtime or CTM env)
    # Reference: TeneT_demo src/init_ipeps_env.jl initialize_vumps_runtime
end
```

Reference:
- TeneT_demo `src/init_ipeps_env.jl` (full implementation with file loading, SU init, perturbation)
- ADC4PEPS `src/ipeps_optimization/init_ipeps.jl` (simpler version)

**Step 1: Write init.jl**
**Step 2: Commit**

---

### Task 7.4: Create energy.jl — energy computation

**Files:**
- Create: `src/ipeps/energy.jl`

Unified energy computation dispatching on model type and environment type.

```julia
# src/ipeps/energy.jl

# Main entry point
function energy(A, env, params::GradientOptimize)
    expectation_value(params.model, A, env, params)
end

# Dispatch on model type — merge from both codebases
function expectation_value(model::Heisenberg, A, env::VUMPSEnv, params)
    # Reference: TeneT_demo src/observable.jl (supports multi-cell via pattern)
    # Reference: ADC4PEPS src/ipeps_optimization/energy_contraction.jl (single cell)
end

function expectation_value(model::J1J2, A, env::VUMPSEnv, params)
    # Includes diagonal J2 terms
end

# Also support CTMEnv
function expectation_value(model::Heisenberg, A, env::CTMEnv, params)
    # Reference: ADC4PEPS src/ipeps_optimization/energy_contraction.jl
end

# ... similar for other models
```

Reference:
- TeneT_demo `src/observable.jl` for multi-cell `expectation_value` implementations
- ADC4PEPS `src/ipeps_optimization/energy_contraction.jl` for CTMEnv versions

**Step 1: Write energy.jl**
**Step 2: Write test against known benchmark**
**Step 3: Commit**

---

### Task 7.5: Create observable.jl — magnetization, correlation length

**Files:**
- Create: `src/ipeps/observable.jl`

```julia
# src/ipeps/observable.jl

function observable(A, χ, params; restriction_ipeps=identity)
    # Compute full set of observables: energy, magnetization, correlation length
    # Reference: TeneT_demo src/observable.jl observable() function
end

function magnetization_value(model, A, env, params)
    # Mx, My, Mz per site
end

function cor_len_value(env, params)
    # Correlation length from transfer matrix eigenvalues
end

function write_obs_log(e, mag, ξ, χ, folder, params)
    # Write observables to log file
end
```

Reference:
- TeneT_demo `src/observable.jl` (comprehensive, all models)
- ADC4PEPS `src/ipeps_optimization/observable.jl` (single cell)

**Step 1: Write observable.jl**
**Step 2: Commit**

---

### Task 7.6: Create restriction.jl — symmetry constraints and gauge transforms

**Files:**
- Create: `src/ipeps/restriction.jl`

```julia
# src/ipeps/restriction.jl

# C4v symmetry (from ADC4PEPS, also in TeneT_demo)
function C4v_restriction(A) end

# Identity restriction
function _restriction_ipeps(A) A end

# Gauge transforms (from TeneT_demo, more comprehensive)
function pepsgeneral(A; tol=1e-12) end
function local_gauge_contraction(A, G) end
function find_local_hermite_G(A, params) end
function to_mcf_ipeps(T; max_iter=100, tol=1e-12) end
```

Reference:
- ADC4PEPS `src/ipeps_optimization/restriction.jl` (C4v, pepsgeneral)
- TeneT_demo `src/restriction.jl` (C4v, pepsgeneral, MCF, gauge optimization — more complete)

**Step 1: Write restriction.jl merging both**
**Step 2: Commit**

---

### Task 7.7: Create precondition.jl

**Files:**
- Create: `src/ipeps/precondition.jl`

```julia
# src/ipeps/precondition.jl

function precondition_invese_single_envir(A, grad, rt, params, restriction_ipeps, fδEi, iter_precond)
    # Local environment Hessian approximation
end

function precondition_invese_hessian(A, grad, rt, rt′, params, restriction_ipeps, fδEi, iter_precond)
    # ForwardDiff-based Hessian
end
```

Reference:
- TeneT_demo `src/precondition.jl` (3 variants: single_envir, hessian, BP)
- ADC4PEPS `src/ipeps_optimization/precondition.jl` (2 variants: single_envir, N)

**Step 1: Write precondition.jl merging both**
**Step 2: Commit**

---

### Task 7.8: Create su_parameterization.jl

**Files:**
- Create: `src/ipeps/su_parameterization.jl`

Reference: TeneT_demo `src/SU_parameterization.jl` — port directly.

**Step 1: Port SU_parameterization.jl**
**Step 2: Commit**

---

### Task 7.9: Create optimize.jl — unified optimization entry point

**Files:**
- Create: `src/ipeps/optimize.jl`

This is the main unified entry point. The pattern matrix determines single/multi-cell behavior.

```julia
# src/ipeps/optimize.jl

function optimise_ipeps(A, χ, params::GradientOptimize;
                        χshift=0, restriction_ipeps=_restriction_ipeps)
    # Main optimization loop:
    # 1. Initialize environment
    # 2. For each restart:
    #    a. build_A → build_M → environment → energy_and_gradient
    #    b. Apply restriction and optional preconditioning
    #    c. LBFGS step
    #    d. Check convergence, increment χ

    # Internal dispatch based on size(params.pattern):
    #   (1,1) → single cell path (ADC4PEPS-style)
    #   (N,M) → multi cell path (TeneT_demo-style)
end

# LBFGS wrapper with checkpointing
function optimize_reload(fg, x, alg; kwargs...)
    # Reference: TeneT_demo src/optimise_patch.jl
    # Reference: ADC4PEPS src/ipeps_optimization/optimise_patch.jl
end

# SU optimization variant
function optimise_ipeps(A, χ, params::SUOptimize; kwargs...)
    # Reference: TeneT_demo src/SUFU.jl
end
```

Reference:
- TeneT_demo `src/optimise_ipeps.jl` (multi-cell with pattern support)
- ADC4PEPS `src/ipeps_optimization/optimise_ipeps.jl` (single cell, cleaner)
- Both `optimise_patch.jl` files for LBFGS state management

**Step 1: Write optimize.jl**
**Step 2: Write integration test**

```julia
# test/test_ipeps/test_single_cell.jl
@testset "Single cell Heisenberg optimization" begin
    model = Heisenberg(Jx=-1.0, Jy=-1.0, Jz=1.0)
    params = GradientOptimize(
        model=model, pattern=[1;;],
        boundary_alg=VUMPS(maxiter=20, tol=1e-6),
        maxiter=2, tol=1e-4
    )
    D, χ = 2, 8
    A = init_ipeps(; D=D, d=2, χ=χ, pattern=[1;;], params=params)
    E = optimise_ipeps(A, χ, params)
    # Heisenberg ground state energy ≈ -0.6694
    @test E < -0.6  # Loose bound for D=2
end
```

**Step 3: Run test, verify pass**
**Step 4: Commit**

```bash
git add src/ipeps/ test/test_ipeps/
git commit -m "feat: unified optimise_ipeps with single and multi-cell support"
```

---

## Phase 8: Autodiff Layer

### Task 8.1: Create rules.jl — ChainRules and SVD adjoint

**Files:**
- Create: `src/autodiff/rules.jl`

Merge AD rules from all codebases:
- TeneT.jl `src/autodiff.jl` (510 lines): qrpos adjoint, leading_boundary adjoint, StructArray adjoint
- TeneT_demo `src/untils.jl`: SVD adjoint (`svd_back`)
- ADC4PEPS `src/contraction/autodiff.jl` (457 lines): QR adjoint, polar decomposition, fixed-point differentiation

**Step 1: Write rules.jl merging AD rules**
**Step 2: Commit**

---

### Task 8.2: Port grassmann.jl

**Files:**
- Move: `src/grassmann.jl` → `src/autodiff/grassmann.jl`

Already exists in TeneT.jl. Just move to new location.

**Step 1: Move file**
**Step 2: Commit**

---

### Task 8.3: Port simple_eig_ad.jl

**Files:**
- Move: `src/simple_eig_ad.jl` → `src/autodiff/simple_eig_ad.jl`

Already exists on parallel-matmul branch. Just move to new location.

**Step 1: Move file**
**Step 2: Commit**

---

## Phase 9: Main Module and Integration

### Task 9.1: Rewrite TeneT.jl main module

**Files:**
- Rewrite: `src/TeneT.jl`

```julia
module TeneT

using LinearAlgebra, Random, Printf
using Parameters
using CUDA, AMDGPU, cuTENSOR
using TensorOperations, OMEinsum
using Zygote, ChainRulesCore, ForwardDiff
using KrylovKit, VectorInterface
using OptimKit, LineSearches
using JLD2, FileIO
using MPI

# Disable CUDA scalar operations
CUDA.allowscalar(false)

# Core types
include("types.jl")
include("defaults.jl")

# StructArray
include("structarray/base.jl")
include("structarray/initial.jl")
include("structarray/buffer.jl")

# Utilities
include("utils/gpu.jl")
include("utils/io.jl")
include("utils/misc.jl")

# Contraction layer
include("contraction/basic.jl")
include("contraction/forloop_parallel_MPI.jl")
include("contraction/observable.jl")

# Boundary algorithms
include("boundary/algorithm.jl")
include("boundary/environment.jl")
include("boundary/vumps.jl")
include("boundary/ctmrg.jl")
include("boundary/qrctm.jl")
include("boundary/fpctm.jl")
include("boundary/pt.jl")

# Hamiltonian models
include("models/lattice.jl")
include("models/models.jl")
include("models/heisenberg.jl")
include("models/j1j2.jl")
include("models/j1j2j3.jl")
include("models/shastry_sutherland.jl")
include("models/kagome.jl")

# iPEPS optimization
include("ipeps/interface.jl")
include("ipeps/build.jl")
include("ipeps/init.jl")
include("ipeps/energy.jl")
include("ipeps/observable.jl")
include("ipeps/restriction.jl")
include("ipeps/precondition.jl")
include("ipeps/su_parameterization.jl")
include("ipeps/optimize.jl")

# Autodiff
include("autodiff/rules.jl")
include("autodiff/grassmann.jl")
include("autodiff/simple_eig_ad.jl")

# Exports
export StructArray, randSA

export AbstractLattice, Square, Honeycomb, KagomeLattice
export ContractionMode, General, Plaquette
export Algorithm, VUMPS, CTMRG, QRCTM, FPCTM, PT
export VUMPSRuntime, VUMPSEnv, CTMEnv
export leading_boundary, environment

export HamiltonianModel, Heisenberg, J1J2, J1J2J3, SS, Kagome
export hamiltonian, hamiltonian_trunc

export iPEPSOptimize, GradientOptimize, SUOptimize, FUOptimize
export init_ipeps, optimise_ipeps, observable
export C4v_restriction, _restriction_ipeps

end # module
```

**Step 1: Write the main module file**
**Step 2: Commit**

---

### Task 9.2: Create utils/ files

**Files:**
- Create: `src/utils/gpu.jl`
- Create: `src/utils/io.jl`
- Create: `src/utils/misc.jl`

Extract utility functions from existing `src/utilities.jl`:
- `gpu.jl`: `_mattype`, `_arraytype`, `set_device_id!`, `get_device`, `atype_device!`
- `io.jl`: `save_rt`, `load_rt` (from TeneT.jl), plus `read_last_log` (from TeneT_demo)
- `misc.jl`: `simple_eig`, `qrpos`, `lqpos`, `takagi_decomposition`, `svd_back`

Reference:
- TeneT.jl `src/utilities.jl` (229 lines)
- TeneT_demo `src/untils.jl` (SVD adjoint, takagi, logging)

**Step 1: Split utilities into gpu.jl, io.jl, misc.jl**
**Step 2: Commit**

---

### Task 9.3: Update Project.toml with all dependencies

**Files:**
- Modify: `Project.toml`

Ensure all dependencies from both ADC4PEPS and TeneT_demo are included:

```toml
[deps]
AMDGPU = "21141c5a-..."
CUDA = "052768ef-..."
ChainRulesCore = "d360d2e6-..."
FileIO = "5789e2e9-..."
ForwardDiff = "f6369f11-..."
ForwardDiffChainRules = "..."
JLD2 = "033835bb-..."
KrylovKit = "0b1a1467-..."
LineSearches = "d3d80556-..."
LinearAlgebra = "37e2e46d-..."
MPI = "da04e1cc-..."
OMEinsum = "ebe7aa44-..."
OptimKit = "77e91f04-..."
Parameters = "d96e819e-..."
Printf = "de0858da-..."
Random = "9a3f8284-..."
TensorOperations = "6aa20fa7-..."
VectorInterface = "409d34a3-..."
Zygote = "e88e6eb3-..."
cuTENSOR = "011b41b2-..."
```

Reference: ADC4PEPS `Project.toml` for UUIDs and version pins.

**Step 1: Update Project.toml**
**Step 2: Run `Pkg.instantiate()` to verify**
**Step 3: Commit**

---

### Task 9.4: Remove old source files

**Files:**
- Delete: `src/environment.jl` (moved to `src/boundary/`)
- Delete: `src/vumpsruntime.jl` (moved to `src/boundary/`)
- Delete: `src/utilities.jl` (split into `src/utils/`)
- Delete: `src/initial_env.jl` (moved to `src/ipeps/init.jl`)
- Delete: `src/contraction/forloop.jl` (replaced by `forloop_parallel_MPI.jl`)

**Step 1: Remove old files**

```bash
git rm src/environment.jl src/vumpsruntime.jl src/utilities.jl src/initial_env.jl src/contraction/forloop.jl
```

**Step 2: Commit**

```bash
git commit -m "cleanup: remove old files replaced by new module structure"
```

---

### Task 9.5: Write integration tests

**Files:**
- Create: `test/runtests.jl`
- Create: `test/test_ipeps/test_single_cell.jl`
- Create: `test/test_ipeps/test_multi_cell.jl`

```julia
# test/runtests.jl
using Test

@testset "TeneT.jl" begin
    include("test_structarray.jl")
    include("test_contraction.jl")
    include("test_boundary/test_vumps.jl")
    include("test_models/test_heisenberg.jl")
    include("test_ipeps/test_single_cell.jl")
    include("test_ipeps/test_multi_cell.jl")
    include("test_autodiff.jl")
end
```

**Step 1: Write integration tests**
**Step 2: Run full test suite**

```bash
julia --project -e 'using Pkg; Pkg.test()'
```

**Step 3: Fix any failures**
**Step 4: Commit**

```bash
git add test/
git commit -m "test: add integration tests for unified TeneT.jl"
```

---

### Task 9.6: Final cleanup and validation

**Step 1: Run full test suite one more time**
**Step 2: Verify single-cell optimization reproduces ADC4PEPS results**
**Step 3: Verify multi-cell optimization reproduces TeneT_demo results**
**Step 4: Final commit**

```bash
git commit -m "feat: complete TeneT.jl unified iPEPS package"
```

---

## Phase Summary

| Phase | Tasks | Description |
|-------|-------|-------------|
| 1 | 1.1 | Branch setup, directory scaffolding |
| 2 | 2.1-2.2 | Core types and defaults |
| 3 | 3.1 | StructArray verification |
| 4 | 4.1-4.3 | Contraction layer (basic, forloop, observable) |
| 5 | 5.1-5.7 | Boundary algorithms (VUMPS, CTMRG, QRCTM, FPCTM, PT) |
| 6 | 6.1-6.6 | Hamiltonian models (lattice types, 5 models) |
| 7 | 7.1-7.9 | iPEPS optimization (interface, build, init, energy, observable, restriction, precondition, SU, optimize) |
| 8 | 8.1-8.3 | Autodiff (rules, grassmann, simple_eig_ad) |
| 9 | 9.1-9.6 | Main module, utils, Project.toml, cleanup, tests |

Total: ~30 tasks across 9 phases.
