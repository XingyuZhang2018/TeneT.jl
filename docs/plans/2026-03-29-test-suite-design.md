# TeneT.jl Test Suite Design

**Date**: 2026-03-29
**Branch**: iPEPS-unified
**Goal**: >95% test coverage excluding `src/models/`

## Decisions

- **GPU**: CPU + CUDA required; ROCArray basic smoke tests if available
- **AD verification**: Numerical gradient checks for critical rrules (QR, SVD, eigsolve, leftenv/rightenv), structural checks for pass-through rules
- **Boundary benchmarks**: Convergence against known analytical values (2D classical Ising)
- **Optimization**: Unit test each component + light integration (2 LBFGS steps, D=2, χ=4)
- **MPI**: Optional, gated by `ENV["TENET_TEST_MPI"]`
- **Structure**: Flat files following standard Julia library conventions

## File Structure

```
test/
├── runtests.jl           # entry point, CUDA detection, helpers
├── test_types.jl         # types.jl + defaults.jl
├── test_structarray.jl   # structarray/{base,initial,buffer}.jl
├── test_utils.jl         # utils/{gpu,io,misc}.jl
├── test_contraction.jl   # contraction/{basic,observable,forloop}.jl
├── test_boundary.jl      # boundary_algorithm/ (VUMPS General/Plaquette/C4v + QRCTM)
├── test_autodiff.jl      # autodiff/{rules,grassmann}.jl
├── test_ipeps.jl         # ipeps_optimize/ (components + light integration)
├── test_patch.jl         # patch/OptimKit_patch.jl
└── test_mpi.jl           # optional, gated by ENV
```

## runtests.jl

- `using TeneT, Test, LinearAlgebra, Random, CUDA`
- Define `ATYPES = CUDA.functional() ? [Array, CuArray] : [Array]` (warn if no CUDA)
- Define `num_grad(f, x; δ=1e-5)` helper for AD tests
- Include all test files; gate `test_mpi.jl` on env var

## Test Content

### test_types.jl
- Type hierarchy: `Square <: AbstractLattice`, `Honeycomb <: AbstractLattice`, `Kagome <: AbstractLattice`
- `Honeycomb()` defaults to `Honeycomb{:brickwall}`
- Algorithm hierarchy: `VUMPS <: Algorithm`, `QRCTM <: Algorithm`
- `Base.show` for HamiltonianModel subtypes produces filesystem-safe strings
- `Defaults` verbosity constants

### test_structarray.jl
Parameterized over `atype ∈ ATYPES`:
- Construction: `StructArray(data, pattern)`, correct indexing via pattern
- Assertion on invalid pattern (index out of bounds)
- AbstractArray interface: `size`, `getindex`, `setindex!`, `length`, `similar`, `zero`, `copy`, `circshift`
- Arithmetic: `+`, `*`, `/`, `norm`, `conj`, `rmul!`, `axpy!`
- NamedTuple interop (Zygote tangent compatibility)
- `randSA`, `cellones`, `ISA`: correct shapes and element types
- GPU roundtrip: `Array(CuArray(S)) ≈ S`
- `Zygote.Buffer(S::StructArray)` works

### test_utils.jl
- **gpu.jl**: `_mattype`, `_arraytype` dispatch; device management (CUDA only); NamedTuple GPU conversions; `for_gc` identity
- **misc.jl**: `qrpos`/`lqpos` (positive diagonal, Q*R≈A); `qr_for_ad`; `simple_eig` finds leading eigenpair; `takagi_decomposition` (M = A*transpose(A) reconstruction, truncation); `checkpoint` returns correct forward value
- **io.jl**: `save_rt`/`load_rt` roundtrip (temp dir); `read_last_log` parsing

### test_contraction.jl
Small random tensors, verify shapes and contraction correctness:
- `ALCtoAC_map`: output shape, AL*C structure
- `CTtoT`, `CTCtoT`: dimension correctness
- `FLmap`, `FRmap`: fixed-point property with eigenvectors
- `Lmap`, `Rmap`: transfer matrix maps
- `ACmap`, `Cmap`: tensor network contraction
- Rank dispatch: rank-4, rank-5, rank-8 paths
- Observable contractions: `contract_n1`, `contract_o1`, `contract_n2_H/V`, `contract_o2_H/V` return correct-type scalars
- `split_count`, `split_ranges` correctness
- `forloop` result matches direct computation

### test_boundary.jl
Parameterized over `atype ∈ ATYPES`:

**VUMPS General**:
- Unit: `left_canonical`/`right_canonical` isometry; `LRtoC`/`ALCtoAC` consistency; `FLint`/`FRint` shape
- Integration: `init_env` → `leading_boundary` converges for 2D classical Ising (1×1 and 2×2), verify free energy

**VUMPS Plaquette**:
- Unit: `ACenv_plaq`, `Cenv_plaq` shape/type
- Integration: converges for 2×2 Ising

**VUMPS C4v**:
- Unit: single-tensor operations
- Integration: converges for 1×1 Ising, matches General result

**QRCTM**:
- Unit: `init_env` shape, `qrctm_step` updates env
- Integration: converges for Ising, matches VUMPS within tolerance

**ObsEnv**: all variants return correct env types

### test_autodiff.jl
**Structural** (runs without error, correct types):
- `rrule` for `StructArray`, `VUMPSRuntime`, `CTMEnv`, `C4vVUMPSEnv`
- `@non_differentiable` declarations
- `rrule` for `norm(S::StructArray)`

**Numerical gradient checks** (Zygote vs finite diff):
- `qrpos`, `lqpos`, `qr_for_ad` adjoints
- SVD adjoint (`svd_back`)
- `orth_for_ad` projection
- `simple_eig` gradient
- `leftenv`/`rightenv` gradients (small system)
- `ACenv`/`Cenv` gradients

**Grassmann**: `project_AL`/`project_AR` tangent space (⊥ to AL); `retract!` returns left-canonical form

### test_ipeps.jl
- `build_A`: StructArray from 6D array; `_lattice_map` dispatch (Square, Honeycomb brickwall)
- `C4v_restriction`: idempotent, C4v invariant
- `pepsgeneral` canonical form: Ac * R reconstructs A
- `to_mcf_ipeps`: balanced reduced density matrices
- `SU_parameterization`: runs, correct output shape
- `precondition_invese_single_envir`: runs for each env type
- `init_ipeps`: random init shape; save/load roundtrip
- **Integration**: `optimise_ipeps` for 2 LBFGS steps (D=2, χ=4), energy decreases

### test_patch.jl
- `LBFGSState` construction and fields
- GPU conversion roundtrip
- `save_lbfgs_state`/`load_lbfgs_state` roundtrip (temp file)
- `optimize_reload` runs 2 iterations on quadratic function

### test_mpi.jl (gated)
- `parallel` matches `forloop` result
- `FLmap_parallel` matches serial `FLmap`

## Benchmark Model

For boundary convergence tests, use 2D classical Ising transfer matrix (no model dependency):
- Build 4-leg tensor `M[i,j,k,l]` directly from Boltzmann weights
- Known exact free energy: `f = -log(2*cosh(2β)) - (1/2π) ∫ log(...) dθ`
- This avoids depending on `src/models/`

## Coverage Strategy

Every exported function and every internal function reachable from public API gets at least one test. The only exclusions are:
- `src/models/` (user-defined, tested in examples)
- MPI code paths when `TENET_TEST_MPI` is not set
- ROCArray paths when AMDGPU is unavailable
