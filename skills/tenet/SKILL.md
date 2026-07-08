---
name: tenet
description: "Use inside the TeneT.jl repository for iPEPS optimization and run design: fresh TeneT iPEPS runs, D or chi growth, GPU/CPU choice, VUMPS/QRCTMRG contraction mode, environment and LBFGS save/load policy, checkpoint/forloop_iter memory tuning, Slice1D/Slice2D parallel method selection, observable-only runs, precondition smoke tests, and interpreting recent TeneT HPC run artifacts. Trigger on TeneT, TeneT.jl, optimise_ipeps, GradientOptimize, init_ipeps_SU, TeneT D-upgrade, TeneT chi continuation, TeneT Slice2D/Slice1D, or Kagome/Honeycomb/Plaquette/C4v/Oneside TeneT examples."
---

# TeneT iPEPS Run Design

Use this skill before designing, modifying, or reviewing TeneT.jl iPEPS optimization runs.
This skill decides physics and runtime parameters. It does not submit Slurm jobs.
For cluster submission, monitoring, syncing, archiving, or cancellation, follow
AGENTS.md and use the user-global `$hpc` skill by default. Use repo-local
`tenet-hpc` only when the user explicitly invokes or enables that optional skill.

## First Checks

1. Read the closest example under `examples/` before inventing a model, pattern,
   contraction mode, or `restriction_ipeps`.
2. Inspect recent `hpc/rendered/*/entry.jl` and `SUBMISSION.md` files when the
   request resembles a previous run.
3. Classify the run as fresh optimization, chi continuation, D upgrade,
   observable-only, Slice2D/precondition smoke, or failure retry.
4. Preserve durable run artifacts. Continue only from saved iPEPS files,
   `history.log` rows, environment files, and LBFGS checkpoints.
5. Do not continue from a timeout candidate that only printed a forward energy
   and did not write `No.N.jld2` or an LBFGS checkpoint.

## Core Defaults

- Use GPU for production runs with `D >= 5`. CPU is for tiny smoke tests only.
- Keep `ifsimple_eig=true` for real iPEPS optimization. Set it false only when
  the task explicitly studies eigensolver or algorithm behavior.
- Keep VUMPS power defaults unless run history gives a reason to change them:
  `power_iter=5`, `power_iter_ad=5`, `power_iter_obs=40`.
- Use the nearest production example's contraction family: `General`,
  `Plaquette`, `C4v`, `Oneside`, or `QRCTMRG`. Do not swap these casually.
- Treat `restriction_ipeps` as physics-specific. Honeycomb brickwall, Kagome,
  C4v, Plaquette, General, and Oneside examples encode different assumptions.

## Environment And LBFGS Policy

- Set `ifsave_lbfgs=true` for optimization runs.
- Set `ifload_lbfgs=true` when continuing from a saved LBFGS checkpoint at the
  same model, pattern, D, chi, and contraction setup.
- Prefer `ifload_env=true` when an environment exists and is trustworthy:
  model, pattern, D, chi, contraction mode, parallel method, and lattice
  orientation must match, with no evidence that the saved environment was
  non-convergent or steering optimization in a bad direction.
- For high-D Slice2D continuations, explicit `ifload_env=false` is a red flag
  when a matching trustworthy env file exists. Verify the reason before spending
  production allocation.
- Turn `ifload_env=false` only when the environment is missing, stale,
  incompatible, from a suspect trend, or intentionally being rebuilt.
- Set `ifsave_env=true` when the environment file is practically small enough.
  Use `<10GB` as the default threshold. Avoid saving very large env files.

## Memory Ladder

Use the memory remedies in this order:

1. Set `ifcheckpoint=true` on both `GradientOptimize` and the VUMPS boundary
   algorithm.
2. Increase `forloop_iter` gradually for OOM as powers of two: try `2^i` with
   `i = 2, 3, ...` (`4`, `8`, `16`, ...), and stop at the first value that
   fits. Do not start with a very large value; it can make the run much slower.
   For `slice1D`, cap the useful maximum near `χ / nprocs`. For `slice2D`,
   analyze the cap from the chosen `(N1,N2)` split instead of reusing the
   `slice1D` rule.
3. Set VUMPS `inner_checkpoint=Recompute()` if memory still fails.
4. Use multi-GPU parallelism. Prefer `slice1D` when it fits; use `slice2D` when
   peak memory is the blocker.
5. For large Kagome-style high-D/high-chi runs, `slice2D(4,4)` on 16 GPUs with a
   high `forloop_iter` is a proven shape, but still size the grid from available
   GPUs and the model.

## Parallelism

- Prefer the newer `parallel_method` API for new runs.
- Use `slice1D(...; forloop_iter=...)` as the first multi-GPU option when memory
  allows.
- Use `slice2D(N1,N2; forloop_iter=...)` when peak memory remains too high.
- Use `ifparallelupdown` mainly for two-GPU runs where up/down environments can
  overlap efficiently and D/chi are not too large.
- For Slice2D, keep collective control flow rank-uniform and avoid gathered
  full-env shortcuts.

## D And Chi Growth

- For fresh chi ladders, use `default_χlist(D; ...)`.
- For continuation from an already optimized high-chi state, use an explicit
  target chi list such as `[1024]`.
- When increasing chi, the default chi ladder already gives later stages more
  VUMPS effort for the environment.
- When increasing D, that extra environment effort is not automatic. Add larger
  `boundary_alg.maxiter` / `opt_obs_maxiter`, or run an explicit env/observable
  warmup before the main LBFGS continuation.
- Prefer `init_ipeps_SU` for D growth. Use `SUτ=0.1` as the practical default
  for initialization, then reset `params.SUτ=0.0` and run with `ifSU=false`
  before `optimise_ipeps`.
- Expect energy to rise immediately after some large-D SU initializations,
  especially at larger D. Judge the run after gradient optimization has a chance
  to relax.

## Observable-Only Runs

- Treat observable-only as a separate run mode.
- Use the source `No.N.jld2` that is durable, not an unsaved timeout candidate.
- Use `maxiter_ad=0` and no LBFGS continuation.
- Set `ifsave_lbfgs=false` and `ifload_lbfgs=false`, except when reusing an
  existing wrapper that already forces them off.

## Precondition And New Paths

- For new Slice2D preconditioner usage, run a short smoke for the model and grid
  before a long production job.
- Use `ifprecondition=true, iter_precond=0` only after confirming the relevant
  model path supports it.
- If a failure is from post-optimization observable gathering, isolate whether
  the optimization path itself passed before changing optimization parameters.

## Cluster Preflight Expectations

When a high-D run will go to a cluster, ensure the HPC workflow includes a
preflight for:

- `using TeneT`
- CUDA kernel execution
- cuTENSOR availability
- MPI rank to GPU mapping
- MPI/NCCL basics for the chosen parallel method

Do not issue `sbatch`, `squeue`, `sacct`, `scancel`, `rsync`, or remote SSH
commands from this skill. Follow AGENTS.md and hand those operations to
user-global `$hpc` by default; use `tenet-hpc` only when explicitly enabled or
invoked.
