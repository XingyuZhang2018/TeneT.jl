# Slice2D Distributed Preconditioner

Date: 2026-07-06

## Goal

Enable `ifprecondition=true` for Slice2D iPEPS optimization without building,
gathering, or storing a full boundary environment on every rank.

The first target is the production Slice2D path already supported by the code:

- Plaquette `J1J2{Square}` on a Slice2D grid.
- General VUMPS models whose observation energy path already supports
  block-distributed Slice2D contractions.

The implementation must preserve the Slice2D storage rule: persistent boundary
tensors stay block-distributed with the first chi leg split by `grid.r1` and the
last chi leg split by `grid.r2`. The replicated center matrix `C` remains
allowed because that is already the Slice2D runtime contract.

## Non-Goals

- Do not add a gathered full-env fallback for preconditioning.
- Do not call `gather_env` from the Slice2D preconditioner.
- Do not materialize full `AL`, `AR`, `AC`, `FL`, or `FR` as a complete
  environment on every rank.
- Do not support Slice2D oneside observation environments in this pass.
- Do not rewrite the LBFGS or OptimKit precondition interface.

## Current State

`optimise_ipeps` calls:

```julia
precondition_invese_single_envir(x, g, rt, params, restriction_ipeps, fdelta, iter_precond)
```

when `params.ifprecondition` is true. The existing preconditioner bodies build
an observation environment, precompute normalization values with
`contract_n_11`, then solve a one-step linear system whose action is:

```julia
delta * x + J_R' * T * J_R * x
```

where `T` is applied through `Mumap_parallel`.

Slice2D already has distributed map primitives for energy and environment work:
`FLmap_slice2d_dist`, `FRmap_slice2d_dist`, `ACmap_slice2d_dist`,
`ACdmap_slice2d_dist`, and `slice2d_dot`. There is no Slice2D version of the
preconditioner-only `Mumap` action yet.

## Architecture

Add a distributed preconditioner path gated by `_effective_grid(params.boundary_alg)`.
When no grid is set, the existing serial and Slice1D code remains unchanged.

When a grid is set, the preconditioner must:

1. Build `A_prime` from the restricted iPEPS, as today.
2. Construct a block observation environment using the existing Slice2D `ObsEnv`
   path for supported models.
3. Reject unsupported model/env combinations before the solve starts.
4. Compute normalization scalars using existing `contract_n_11(...; grid)`.
5. Apply the preconditioner transfer action with a new block-distributed
   `Mumap_slice2d_dist`.
6. Keep `T_x` as a replicated site tensor output so the existing `overlap_vjp`
   can remain conceptually unchanged.

The key invariant is that no persistent full environment is produced. The only
large chi communication inside `Mumap_slice2d_dist` is map-local row/column
allgather of working slices, matching the existing Slice2D map design.

## New Primitive

Add:

```julia
Mumap_slice2d_dist(AC_blk, ACd_blk, FL_blk, FR_blk, Mu, grid::Slice2DGrid;
                   forloop_iter = 1, inner_etype = nothing)
```

Input block layout:

- `AC_blk`, `FL_blk`: local first chi block, local last chi block.
- `ACd_blk`, `FR_blk`: same Slice2D convention.
- `Mu`: replicated site variation tensor, with no boundary chi tiling.

Data flow:

1. Recover global chi from the first block leg.
2. Build `a_rs = split_ranges(chi, grid.N1)` and
   `l_rs = split_ranges(chi, grid.N2)`.
3. Gather only the working slices needed by the local contraction:
   - `AC_row = slice2d_gather_row(AC_blk, grid, l_rs)`.
   - `FL_row = slice2d_gather_row(FL_blk, grid, l_rs)`.
   - `FR_col = slice2d_gather_col(FR_blk, grid, a_rs)`.
   - `ACd_col = slice2d_gather_col(ACd_blk, grid, a_rs)`.
4. Compute the local contribution to the replicated `Mumap` output. The
   contraction is algebraically identical to serial `Mumap`:

   ```julia
   result[f,k,h,c,p] :=
       (AC[a,b,c,d] * FR[d,g,h,l]) *
       ((FL[a,e,f,i] * ACd[i,j,k,l]) * Mu[e,j,g,b,p])
   ```

5. Allreduce the result over `grid.comm`, because the output has no
   distributed chi leg and must be replicated on every rank.

The implementation should prefer the existing chain-engine structure where it
fits, but the first version may use a direct tensor contraction if that is
clearer and tests keep dimensions small. If a direct contraction is used, add a
comment naming the serial formula it mirrors.

## Preconditioner Routing

Add a small routing helper:

```julia
_precondition_grid(params) = _effective_grid(params.boundary_alg)
```

and branch inside the existing `precondition_invese_single_envir` methods.

### Plaquette

For `rt::PlaquetteVUMPSRuntime` with a grid:

- Require `_dist_energy_plaq_grid(params.model, params.boundary_alg) !== nothing`.
- Build `env = ObsEnv(rt, A_prime, params.boundary_alg, params.model)`.
- Require that `env.AL`, `env.FLu`, and `env.FLo` are block-distributed tensors.
- Use `AC = ALCtoAC_slice2d(env.AL, env.C, grid)`.
- Use `contract_n_11(...; grid)` for `n_map`.
- Use `Mumap_slice2d_dist(AC[i,j], AC[ir,j], FLo[i,j], FLo[i,jr], A_prime_x_q, grid; forloop_iter)` for `T_x`.

No Plaquette preconditioner path may call `ObsEnv(..., model=nothing)` under a
grid, because that can choose the wrong gathered/blocked behavior.

### General

For `rt::Union{VUMPSRuntime, Tuple{VUMPSRuntime,VUMPSRuntime}}` with a grid:

- Require `_dist_energy_general_grid(params.model, params.boundary_alg) !== nothing`.
- Build `env = ObsEnv(rt, A_prime, params.boundary_alg, params.model)`.
- Reject `OnesideVUMPSEnv` with a clear error.
- Use the block fields `ACu`, `ACd`, `FLo`, and `FRo` directly.
- Use `contract_n_11(...; grid)` and `Mumap_slice2d_dist(...; grid)`.

The row partner rules stay the same as the current full-env preconditioner:
`Ni + 1 - i` for the standard General env. Oneside rules are left out until a
distributed oneside observation environment exists.

## Error Handling

Unsupported Slice2D combinations should fail early with messages naming the
model and algorithm. Examples:

- Slice2D preconditioner requires a model with distributed energy support.
- Slice2D preconditioner does not support `OnesideVUMPSEnv`.
- Slice2D preconditioner requires a block observation environment; full-env
  fallback is disabled.

These failures are preferable to silently gathering a full env or running serial
code on block tensors.

## Testing

Add a CPU MPI test runner:

```text
test/run_test_slice2d_precondition.jl
test/test_slice2d_precondition.jl
```

The test file should run on exactly four ranks with `slice2d_grid(2, 2)`.

Test coverage:

1. `Mumap_slice2d_dist` forward parity against serial `Mumap`:
   - Generate full random `AC`, `ACd`, `FL`, `FR`, `Mu`.
   - Scatter chi tensors.
   - Compare `Mumap_slice2d_dist` to serial `Mumap` on every rank.
2. Plaquette direct precondition call:
   - Build a tiny `J1J2{Square}` Plaquette runtime with block env.
   - Call `precondition_invese_single_envir` after `iter_precond`.
   - Assert output shape matches the gradient and values are finite.
3. End-to-end optimization smoke:
   - Use small `D=2`, small chi, `maxiter=1` or `2`.
   - Set `ifprecondition=true`, `iter_precond=0`.
   - Assert the run completes and rank 0 sees finite energy/history.

Keep test dimensions small; this is a correctness and routing gate, not a
performance benchmark.

## BSC Validation

After local 2x2 CPU MPI passes, submit a BSC smoke with a 1x2 Slice2D grid on
two H100 GPUs.

Required HPC handling:

- Render all submitted scripts under
  `D:\1 - research\1.26 - iPEPS_opt\hpc\rendered\<stamp>_slice2d_precondition_bsc_smoke`.
- Include the Julia entry, Slurm script, manifest/preferences if used, source
  status, and `SUBMISSION.md`.
- Use direct BSC SSH/SCP details from the HPC cluster registry.
- Do not run any user-wide cancellation command.
- After `sbatch` succeeds, update the shared in-flight ledger with Slurm ID,
  workdir, rendered directory, and purpose.

The BSC smoke should use the same tiny Plaquette `J1J2{Square}` case as the CPU
test, with `ifprecondition=true` and `iter_precond=0`. It should exercise GPU
row/column collectives and NCCL if enabled by the existing environment.

## Acceptance Criteria

- No Slice2D preconditioner path calls `gather_env`.
- No Slice2D preconditioner path constructs a full observation environment.
- `Mumap_slice2d_dist` matches serial `Mumap` on a 2x2 CPU MPI test.
- A direct distributed precondition call returns finite output with correct
  shape.
- A short 2x2 CPU Slice2D optimization with `ifprecondition=true` completes.
- A 1x2 two-GPU BSC smoke is submitted and reaches the preconditioner branch.

## Risks

- The `Mumap` chain has two cross-axis contracted chi legs. A wrong row/column
  gather choice can pass shape checks and still compute diagonal-only blocks.
  The parity test must compare against serial on nontrivial random tensors.
- The replicated `Mumap` output requires an allreduce. Missing it can pass on a
  single rank but fail under MPI.
- Unsupported model routing must be strict; accidental full-env fallback would
  violate the memory requirement even if tests pass at small chi.
- Rectangular 1x2 BSC smoke exercises only two ranks. The local 2x2 CPU test
  remains necessary to catch both axes.
