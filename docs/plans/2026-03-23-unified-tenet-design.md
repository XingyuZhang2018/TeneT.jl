# TeneT.jl Unified iPEPS Package Design

## Goal

Merge TeneT_demo (large unit cell iPEPS) and ADC4PEPS (single unit cell iPEPS) into a single TeneT.jl package, rewriting both layers from scratch while building on the existing TeneT.jl repository (parallel-matmul + Plaquette branches).

## Decisions

- **Single package** with submodule-style directory layout
- **All boundary algorithms** retained: VUMPS, CTMRG, QRCTM, FPCTM, PT
- **All Hamiltonian models** retained: Heisenberg, J1J2, J1J2J3, SS, Kagome
- **Unified `optimise_ipeps` interface** — `pattern` matrix determines single/multi unit cell
- **Full rewrite** of both contraction and optimization layers
- **Base repo**: TeneT.jl, new branch from `parallel-matmul` (inherits MPI support)
- **Plaquette** as a `ContractionMode` inside VUMPS, not a separate algorithm

## Directory Structure

```
TeneT.jl/src/
├── TeneT.jl                    # Main module, exports all public API
│
├── types.jl                    # Core abstract types, leg3/leg4 aliases
│
├── structarray/                # StructArray (from existing TeneT.jl)
│   ├── base.jl
│   ├── buffer.jl
│   └── initial.jl
│
├── contraction/                # Low-level tensor contraction
│   ├── basic.jl                # FLmap, FRmap, ACmap etc.
│   ├── forloop.jl              # Memory-efficient loop contraction
│   ├── forloop_parallel_MPI.jl # MPI parallel contraction
│   └── observable.jl           # oc_H_leg3/4, contract_o2_H/V, contract_n2
│
├── boundary/                   # Boundary algorithms (unified interface)
│   ├── algorithm.jl            # abstract Algorithm; VUMPS, CTMRG, QRCTM, FPCTM, PT
│   ├── environment.jl          # VUMPSEnv, CTMEnv structs
│   ├── vumps.jl                # VUMPS with ContractionMode (General, Plaquette, ...)
│   ├── ctmrg.jl
│   ├── qrctm.jl
│   ├── fpctm.jl
│   └── pt.jl
│
├── models/                     # Hamiltonian models (lattice info inside model)
│   ├── lattice.jl              # AbstractLattice, Square, Honeycomb, Kagome
│   ├── models.jl               # abstract HamiltonianModel
│   ├── heisenberg.jl
│   ├── j1j2.jl
│   ├── j1j2j3.jl
│   ├── shastry_sutherland.jl
│   └── kagome.jl
│
├── ipeps/                      # iPEPS optimization (unified single/multi cell)
│   ├── interface.jl            # GradientOptimize, SUOptimize, FUOptimize
│   ├── init.jl                 # init_ipeps, initialize_env
│   ├── optimize.jl             # optimise_ipeps (unified entry point)
│   ├── build.jl                # build_A, build_M (lattice mapping layer)
│   ├── energy.jl               # Energy calculation
│   ├── observable.jl           # Observable computation
│   ├── restriction.jl          # C4v symmetry, gauge transforms
│   ├── precondition.jl         # Preconditioning
│   └── su_parameterization.jl  # SU parameterization
│
├── autodiff/                   # AD support
│   ├── rules.jl                # ChainRules, SVD adjoint
│   ├── grassmann.jl            # Grassmann manifold projection
│   └── simple_eig_ad.jl        # Linear AD for eigensolvers
│
└── utils/
    ├── gpu.jl                  # CUDA/AMDGPU abstraction
    ├── io.jl                   # File I/O (JLD2)
    └── misc.jl                 # SVD, Takagi decomposition etc.
```

## Core Type System

### Lattice Types

```julia
abstract type AbstractLattice end
struct Square <: AbstractLattice end
struct Honeycomb <: AbstractLattice end
struct Kagome <: AbstractLattice end
```

### Hamiltonian Models (lattice info inside)

```julia
abstract type HamiltonianModel end

@kwdef struct Heisenberg{L<:AbstractLattice} <: HamiltonianModel
    lattice::L = Square()
    S::Real = 0.5
    Jx::Real = 1.0; Jy::Real = 1.0; Jz::Real = 1.0
end

@kwdef struct J1J2{L<:AbstractLattice} <: HamiltonianModel
    lattice::L = Square()
    S::Real = 0.5; J1::Real = 1.0; J2::Real = 0.5
end

# J1J2J3, SS, Kagome similarly...

# hamiltonian(model) -> two-body operator h
function hamiltonian(model::HamiltonianModel) end
```

### Contraction Modes (for VUMPS specialization)

```julia
abstract type ContractionMode end
struct General <: ContractionMode end
struct Plaquette <: ContractionMode end
# Future: Plaquette3x3, etc.
```

### Boundary Algorithms

```julia
abstract type Algorithm end

@kwdef mutable struct VUMPS{M<:ContractionMode} <: Algorithm
    mode::M = General()
    maxiter::Int = 100
    maxiter_ad::Int = 30
    tol::Real = 1e-10
    ifupdown::Bool = true
    ifparallel::Bool = false
    forloop_iter::Int = 1
    ifcheckpoint::Bool = false
    power_iter::Int = 1
end

@kwdef mutable struct CTMRG <: Algorithm ... end
@kwdef mutable struct QRCTM <: Algorithm ... end
@kwdef mutable struct FPCTM <: Algorithm ... end
@kwdef mutable struct PT <: Algorithm ... end

# Dispatch on mode
leading_boundary(M, alg::VUMPS{General})   = ...
leading_boundary(M, alg::VUMPS{Plaquette}) = ...
```

### iPEPS Optimization

```julia
abstract type iPEPSOptimize end

@kwdef mutable struct GradientOptimize <: iPEPSOptimize
    model::HamiltonianModel
    pattern::Matrix{Int}           # [1;;] single cell, [1 2; 2 1] multi cell
    boundary_alg::Algorithm = VUMPS()
    optimizer = LBFGS(200)
    ifSU::Bool = false
    ifprecondition::Bool = false
    reuse_env::Bool = true
    folder::String = "data/"
end
```

## Optimization Flow

```
optimise_ipeps(A, χ, params)
│
├─ A′ = build_A(A, params)           # Optional SU parameterization
│
├─ M = build_M(A′, model)            # Lattice mapping layer:
│   dispatch on model.lattice:        #   Square    → A ⊗ conj(A)
│                                     #   Honeycomb → coarse-grain to square
│                                     #   Kagome    → absorb into square tensor
│   Output: M::StructArray on effective square lattice
│
├─ env = environment(M, boundary_alg) # Dispatch to VUMPS/CTMRG/QRCTM/FPCTM/PT
│                                     # All algorithms only see square-lattice M
│
├─ E, grad = energy_and_gradient(A, env, params)
│   # Zygote AD through the full pipeline
│
├─ A = restriction(A, grad, params)   # Symmetry constraints + optional precondition
│
└─ A = optimizer_step(A, grad, optimizer)  # LBFGS update
│
convergence / χ-annealing check
```

### Single vs Multi Cell

- `pattern = [1;;]` → single tensor, simpler contraction (ADC4PEPS-style)
- `pattern = [1 2; 2 1]` → multiple tensors, row/column contraction (TeneT_demo-style)
- Difference encapsulated inside `environment()` and `energy()`

## Code Source Mapping

| Module | Primary Source | Notes |
|--------|---------------|-------|
| `structarray/` | TeneT.jl (Array-of-Array) | Reuse, mature |
| `contraction/basic.jl` | Rewrite, ref TeneT.jl + ADC4PEPS | Unified leg3/leg4/leg8 |
| `contraction/forloop*.jl` | TeneT.jl (parallel-matmul) | MPI parallel |
| `contraction/observable.jl` | Rewrite, merge both | Unified single/multi cell |
| `boundary/vumps.jl` | Rewrite, merge both + Plaquette | VUMPS + ContractionMode |
| `boundary/ctmrg.jl` | ADC4PEPS | Port and adapt |
| `boundary/qrctm.jl` | ADC4PEPS | Port and adapt |
| `boundary/fpctm.jl` | ADC4PEPS | Port and adapt |
| `boundary/pt.jl` | ADC4PEPS | Port and adapt |
| `models/` | Rewrite, TeneT_demo primary | Unified interface + lattice types |
| `ipeps/optimize.jl` | Rewrite, merge both | Unified entry |
| `ipeps/build.jl` | Rewrite, merge both | Lattice mapping layer |
| `ipeps/energy.jl` | Rewrite, merge both | Single cell + multi cell |
| `ipeps/restriction.jl` | Rewrite, merge both | C4v + gauge transforms |
| `ipeps/precondition.jl` | Rewrite, merge both | Extend to multi cell |
| `autodiff/` | parallel-matmul + TeneT_demo | simple_eig_ad + svd_back |
| `utils/` | Merge both | GPU, IO, math tools |

## Test Strategy

```
test/
├── runtests.jl
├── test_structarray.jl
├── test_contraction.jl
├── test_boundary/
│   ├── test_vumps.jl
│   ├── test_vumps_plaquette.jl
│   ├── test_ctmrg.jl
│   └── ...
├── test_models/
│   ├── test_heisenberg.jl
│   └── ...
├── test_ipeps/
│   ├── test_single_cell.jl
│   ├── test_multi_cell.jl
│   └── test_kagome.jl
└── test_autodiff.jl
```

Acceptance criteria: verify against known benchmark energies (e.g., Heisenberg E ≈ -0.6694).

## AD Strategy

- Zygote reverse-mode AD for gradient computation
- `simple_eig_ad` (from parallel-matmul) for eigenvalue linear AD
- `checkpoint` support (from ADC4PEPS) for GPU memory efficiency
- Custom SVD adjoint (from TeneT_demo `svd_back`)

## GPU Support

- `atype` parameter: `Array` (CPU), `CuArray` (NVIDIA), `ROCArray` (AMD)
- All internal operations transparent to array type
