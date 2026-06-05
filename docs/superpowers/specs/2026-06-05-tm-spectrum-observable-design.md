# Transfer Matrix Spectrum Observable Design

Date: 2026-06-05
Branch: `iPEPS-unified`

## Goal

Add the `TM_spectrum` functionality from
`D:\1 - research\1.21 - Kitaev_spin1.5\Pure_Kitaev_Honeycomb_beyond_one_half`
to the current TeneT.jl branch.

The feature computes transfer-matrix excitation spectra for iPEPS states and
writes the resulting gaps to disk. It should live with observable diagnostics,
not inside the iPEPS optimizer implementation.

## Scope

Supported:

- `params.boundary_alg isa VUMPS{General}`
- `params.model.lattice isa Honeycomb{:brickwall_h}` or `Honeycomb{:brickwall_v}`
- both ordinary spectra (`ifdomainwall=false`) and domain-wall spectra
  (`ifdomainwall=true`)
- current TeneT contraction convention, with no `ifflatten=true` branch

Unsupported:

- `Plaquette`, `C4v`, `QRCTMRG`, and other boundary algorithms
- non-honeycomb-brickwall lattices
- flattened double-layer contractions from the old project

Unsupported inputs should fail early with clear `ArgumentError` messages.

## Module Layout

Create `src/observable/` and move the existing high-level observable code out of
`src/ipeps_optimize/observable.jl`.

Planned files:

- `src/observable/interface.jl`
  - `energy(A, env, params)`
  - top-level `observable(A, χ, params; ...)`
- `src/observable/magnetization.jl`
  - all `magnetization_value(...)` methods
- `src/observable/correlation_length.jl`
  - all `cor_len_value(...)` methods
- `src/observable/wp_order.jl`
  - `Wp_value(...)`
  - `fwave_order(...)`
- `src/observable/transfer_matrix_spectrum.jl`
  - `TM_spectrum(...)`
  - private helper functions used only by the spectrum calculation

`src/contraction/observable.jl` remains unchanged because it contains low-level
contraction kernels, not high-level observable orchestration.

`src/TeneT.jl` will include the new observable files after iPEPS build,
initialization, restriction, and precondition code are loaded, and before
`ipeps_optimize/optimize.jl` if optimize depends on `energy`.

Exports:

- keep `observable`
- add `TM_spectrum`

## API

Keep the old public entry point shape:

```julia
TM_spectrum(n::Int, k::Real, A, χ, params::iPEPSOptimize;
            restriction_ipeps=_restriction_ipeps,
            ifdomainwall=false)
```

Return:

```julia
Δ = -log.(norm.(eigenvalues))
```

The `k` argument follows the old convention: callers pass momentum in units of
π, and the effective map receives `k * pi`.

## Behavior

`TM_spectrum` should:

1. validate that the current parameter object uses `VUMPS{General}` and a
   supported honeycomb brickwall lattice;
2. initialize or load the VUMPS runtime;
3. build and converge the boundary with the current `build_A` and
   `leading_boundary` APIs;
4. prepare the excitation environments for either the trivial or domain-wall
   sector;
5. solve for the largest-magnitude eigenvalues with `KrylovKit.eigsolve`;
6. convert eigenvalues to gaps with `-log.(norm.(λs))`;
7. write one value per line to the spectrum log file;
8. return the gap vector.

Log paths:

- ordinary sector:
  `params.folder/D$(D)_χ$(χ)/TM_spectrum/trivial/k$k.log`
- domain-wall sector:
  `params.folder/D$(D)_χ$(χ)/TM_spectrum/non-trivial/k$k.log`

Runtime environment loading:

- ordinary spectra use the current TeneT `initialize_env` behavior and default
  environment file `params.folder/D$(D)/environment/χ$(χ).jld2`;
- domain-wall spectra use a private helper in the spectrum module to load or
  create two independent VUMPS runtimes from
  `params.folder/D$(D)/environment/χ$(χ)_1.jld2` and
  `params.folder/D$(D)/environment/χ$(χ)_2.jld2`;
- the public `initialize_env` API is not changed.

## Migration Notes

The old project used helper names that no longer exist in this branch:

- old `FLmap_forloop` / `FRmap_forloop` should be adapted to
  `FLmap_parallel` / `FRmap_parallel` with `ifparallel` and `forloop_iter`;
- old `contract_n1` should be adapted to `contract_n_11`;
- old `build_M` is not migrated as a standalone function. The spectrum code
  should use the current TeneT convention where the built iPEPS `StructArray`
  is the rank-5 tensor input accepted by `FLmap`, `FRmap`, and `ACmap`.

The old `ifflatten` branches are omitted because current contraction paths do
not expose that choice.

## Tests

Add tests that are small enough for normal CPU CI.

Structural tests:

- `TM_spectrum` is exported.
- non-`VUMPS{General}` boundary algorithms fail with `ArgumentError`.
- non-brickwall lattices fail with `ArgumentError`.
- spectrum writing creates the expected `trivial` and `non-trivial` paths.

Smoke tests:

- CPU `Array`
- `D=2`, `χ=4`
- `pattern = [1 2; 2 1]`
- honeycomb brickwall model
- `VUMPS{General}(maxiter=1 or 2, maxiter_ad=0, verbosity=0, ifupdown=true)`
- run `TM_spectrum(1, 0.0, A, χ, params; ifdomainwall=false)`
- run `TM_spectrum(1, 0.0, A, χ, params; ifdomainwall=true)`
- assert length, finite values, and output log existence, without asserting
  specific physics values.

## Out Of Scope

- extending the spectrum routine to plaquette, C4v, or QRCTMRG environments
- changing physics conventions beyond the old `TM_spectrum` behavior
- optimizing the spectrum eigensolver
- changing low-level contraction kernels in `src/contraction/observable.jl`
