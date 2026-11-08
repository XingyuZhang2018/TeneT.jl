---
name: tenet
description: "Use for TeneT.jl iPEPS optimization and run design: fresh runs, D or chi growth, GPU/CPU choice, VUMPS/QRCTMRG contraction mode, environment and LBFGS save/load policy, checkpoint/forloop_iter memory tuning, Slice1D/Slice2D parallel method selection, observable-only and TM_spectrum runs, default-parameter changes, precondition smoke tests, and interpreting TeneT.jl run artifacts. Trigger on TeneT.jl, using TeneT, optimise_ipeps, GradientOptimize, init_ipeps_SU, TM_spectrum, transfer-matrix spectrum, TeneT.jl default parameters, TeneT.jl D-upgrade, TeneT.jl chi continuation, TeneT.jl Slice2D/Slice1D, or Kagome/Honeycomb/Plaquette/C4v/Oneside Julia examples."
---

# TeneT.jl iPEPS Run Design

Use this skill before designing, modifying, or reviewing TeneT.jl iPEPS optimization runs.
This skill decides physics and runtime parameters. It does not submit Slurm jobs.
Use TeneT.jl directly for production, local validation, and diagnosis. Do not
require a cross-library parity check or switch backends automatically.
For cluster submission, monitoring, syncing, archiving, or cancellation, follow
AGENTS.md and use the user-global `$hpc` skill by default. Use repo-local
`tenet-hpc` when explicitly invoked/enabled or selected by the global `$hpc`
workflow as its TeneT.jl supplement.

## First Checks

1. Read the closest TeneT.jl example under `examples/` before inventing a model,
   pattern, contraction mode, or `restriction_ipeps`.
2. Inspect recent `hpc/rendered/*/entry.*`, command scripts, and
   `SUBMISSION.md` files when the request resembles a previous run.
3. Classify the run as fresh optimization, chi continuation, D upgrade,
   observable-only, Slice2D/precondition smoke, or failure retry.
4. Preserve durable run artifacts. For an exact breakpoint continuation, pair
   the saved iPEPS iterate with its matching LBFGS checkpoint/history; verify
   the accepted iteration and optimization-variable layout before launching.
5. Do not continue from a timeout candidate that only printed a forward energy
   and did not write `No.N.jld2` or an LBFGS checkpoint. Do not silently fall
   back to a tensor-only restart when a matching LBFGS checkpoint should exist.

## Core Defaults

- Use TeneT.jl as the implementation for this skill's local and production
  workflows. Do not require a cross-library parity run.
- Before any TeneT.jl HPC submission, run a local `D=2`, `chi=16` TeneT.jl smoke
  for the same model, lattice/pattern, contraction family, scalar type, and
  relevant observable or optimization path. Require finite output and expected
  persistence behavior; archive the command and result with the job.
- Use GPU for production runs with `D >= 5`. CPU is for tiny smoke tests only.
- Keep `ifsimple_eig=true` for real iPEPS optimization. Set it false only when
  the task explicitly studies eigensolver or algorithm behavior.
- Keep VUMPS power defaults unless run history gives a reason to change them:
  `power_iter=5`, `power_iter_ad=5`, `power_iter_obs=40`.
- Use the nearest production example's contraction family: `General`,
  `Plaquette`, `C4v`, `Oneside`, or `QRCTMRG`. Do not swap these casually.
- Treat `restriction_ipeps` as physics-specific. Honeycomb brickwall, Kagome,
  C4v, Plaquette, General, and Oneside examples encode different assumptions.
- When modifying any TeneT default parameter in source code, constructors,
  examples, or run templates, pause before committing and ask the user to
  confirm the default change. Do not decide and commit a default-parameter
  change based only on subjective judgment.
- If TeneT.jl lacks the requested model, lattice, scalar type, TM sector, or
  checkpoint/parallel capability, state the gap and ask the user to choose a
  supported path. Do not switch libraries automatically.

## VUMPS Iteration Limit (User Requirement, 2026-10-21)

- Set the internal forward/untaped VUMPS limit to **at most 50 boundary
  iterations** (`boundary_alg.maxiter`, or the selected entry's equivalent).
  Use a larger value only when the user explicitly specifies or approves it
  for that run or solve. Never raise it automatically to obtain convergence.
- Apply this limit to optimization FG evaluations, environment preparation,
  observables, diagnostics and refinement. Count forward updates for the same
  physical tensor, target chi and boundary side cumulatively across segments,
  retries and restarts; splitting a solve does not authorize another 50 steps.
  Preserve existing work counts and exact state/environment provenance.
- If the 50-step allowance is exhausted without passing the required gate,
  save the partial environment and report its residual/trend. Stop and obtain
  explicit authorization before extending that solve. Historical 200/500-step
  recipes, templates, larger source defaults and automation prompts are not
  authorization to exceed this limit.
- The forward limit is distinct from LBFGS iterations, line-search `maxfg`,
  taped AD depth (`maxiter_ad`/`miniter_ad`, e.g. AD4), and inner Power counts
  (`power_iter`, `power_iter_ad`, `power_iter_obs`). Do not set those knobs to
  50 or change them to evade this cap. Retain their separate accounting and
  any applicable total-work budget.

## Mandatory Pre-Optimization Environment Gate

The following is the default policy. An explicit user instruction about gate
placement or trial acceptance governs that run. If the user specifies a
final-observable-only boundary gate, do not impose this pre-optimization gate
or reject intermediate line-search trials solely for failing it; keep recording
residuals and enforce the requested gate on final observables. Scope this
exception to the named task, not other projects.

Otherwise apply this gate before every fresh optimization, D/chi-grown
optimization, or resumed optimization segment:

1. Check for a durable environment at the exact optimization state and target
   chi. Require matching model, tensor/checkpoint, lattice/pattern,
   restriction, scalar type, contraction family, orientation, D, chi, and
   parallel layout, plus a recorded final VUMPS error.
2. Treat `err <= 1e-6` as the default pass condition. If the error is only near
   that scale but remains above it, report the exact value and recent trend;
   do not silently release optimization without explicit user acceptance.
3. When no passing environment exists, run a separate environment-only stage
   with at most 50 forward boundary VUMPS iterations, reduced by earlier work
   on the same solve. Disable optimizer iterations and LBFGS, set the selected
   boundary/observable VUMPS `maxiter` explicitly, and save the environment.
   Use the established env-only wrapper for that run family; `maxiter_ad=0`
   alone is not a universal optimizer-disable switch.
4. Preserve a partial environment. Resume an interrupted or shorter segment
   only within the remaining 50-step allowance, with `ifload_env=true` and
   `ifsave_env=true`; never reset its counts. Once that allowance is exhausted,
   stop rather than automatically launching another segment. A larger solve
   requires the user's explicit specification or approval.
5. Start optimization only from the passed environment with `ifload_env=true`.
   Require an explicit environment-load success marker and re-check the loaded
   metadata. A missing or failed load blocks LBFGS; the optimizer's own boundary
   iterations do not replace this precondition.

For D growth, apply the gate after SU has produced the new-D tensor. For chi
growth, converge and save the target-chi environment before optimizing at that
chi.

## Environment And LBFGS Policy

- Set `ifsave_lbfgs=true` for optimization runs.
- Treat a breakpoint, walltime, preemption, requeue, or new job-segment resume
  as an optimizer-state continuation, not a fresh optimization. Set
  `ifload_lbfgs=true` and directly load the LBFGS history matching the saved
  iPEPS iterate whenever the optimization problem and variable layout match.
- Make LBFGS-history loading the mandatory default for large-scale runs.
  Discarding valid quasi-Newton history repeats expensive gradient and line-search
  work; a scheduler restart, wrapper change, resource-count change, or rebuilt
  environment alone is not a reason to reset it.
- Verify that the tensor and LBFGS checkpoint belong to the same accepted
  iteration and compatible model, ansatz/restriction, pattern, symmetry, D,
  scalar type, parameter ordering, objective, and optimizer-state format. Make
  the loader report success; fail closed and diagnose instead of silently
  starting with empty history.
- Reset or set `ifload_lbfgs=false` only when the history is missing, incomplete,
  unreadable, corrupt, incompatible with the resumed optimization variables, or
  reproducibly causes non-finite values, invalid curvature/line-search behavior,
  or an immediate pathological step. Record the evidence and preserve the old
  checkpoint before starting fresh.
- Decide environment and optimizer history independently. A missing, stale, or
  suspect environment can justify `ifload_env=false` during an environment-only
  rebuild without discarding an otherwise compatible LBFGS history; it never
  justifies starting optimization without first passing the environment gate.
- Prefer `ifload_env=true` when an environment exists and is trustworthy:
  model, pattern, D, chi, contraction mode, parallel method, and lattice
  orientation must match, with no evidence that the saved environment was
  non-convergent or steering optimization in a bad direction.
- For high-D Slice2D continuations, explicit `ifload_env=false` is a red flag
  when a matching trustworthy env file exists. Verify the reason before spending
  production allocation.
- Turn `ifload_env=false` only when the environment is missing, stale,
  incompatible, from a suspect trend, or intentionally being rebuilt.
- Set `ifsave_env=true` for the mandatory pre-optimization environment. If its
  estimated file size exceeds 10GB, pause and confirm the storage/persistence
  plan with the user instead of bypassing the loadable-environment gate. For
  optional non-gate environments, use `<10GB` as the default save threshold.

## Live Monitor Layout

When creating or adapting a TeneT monitor, use the shared
[HPC monitor layout](C:/Users/xingzhan/.codex/skills/hpc/reference/dashboard-layout.md).
Put complete parameters in the first collapsed block; keep current work
prominent and put historical jobs and historical curves in separate collapsed
blocks. Show energy, gradient and the configured environment residual over
cumulative chi-continuation steps, with stage boundaries and independent
seed/initialization chains kept distinct. Absolute energies use decimal notation;
outlier handling preserves all raw points. Apply the actual Julia
logs/checkpoints and gate policy rather than copying C++ field names or
campaign-specific numerical settings.

## Required Post-Run Energy Trend

- After every optimization segment reaches a terminal state, including normal
  completion, an iteration cap, walltime, preemption, or a recoverable failure,
  wait for the durable history and checkpoints to be available and then
  automatically create or update `energy_vs_iter.png`. Do this before
  recommending or launching another continuation. Prefer local post-processing
  after result sync instead of spending reserved compute resources on plotting.
- Plot only accepted-iterate energies from `history.log` or an equivalent
  accepted-step table. Do not mix in line-search trial evaluations, forward-only
  prints, observable energies from a different chi/model, or an unsaved timeout
  candidate.
- For a restart chain, preserve the raw logs and stitch all compatible segments
  in accepted-step order. Deduplicate repeated records and mark chi/stage and
  restart boundaries. Save a companion `energy_vs_iter.tsv` with at least the
  iteration, chi, energy, gradient norm when available, segment/source, and
  enough configuration provenance to audit the plot.
- Use the plot to make an explicit `continue`, `stop/hold`, or `diagnose first`
  recommendation. Report the best and final energy, recent-window energy gain,
  gradient-norm trend, accepted-step cost, stop reason, and availability of the
  matched iPEPS/LBFGS checkpoint. Judge whether the recent gain is material
  relative to numerical noise and compute cost.
- Do not treat a flat or attractive energy curve alone as convergence. Cross-check
  the gradient, chi/environment adequacy, physical observables, and terminal
  markers. Recommend diagnosis rather than more iterations when the energy rises
  or oscillates pathologically, gradients are non-finite or stagnant, the chi is
  likely limiting the objective, or the loaded optimizer history is suspect.
- If fewer than two accepted iterations exist, record that the trend is
  insufficient instead of fabricating a curve or continuation verdict. For
  observable-only and TM-spectrum runs with no optimization iterations, mark
  this post-run plot as not applicable.

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
- For `D <= 8` and `χ <= 512`, do not use multi-GPU parallelism by default.
  Use a single GPU unless measured memory pressure or throughput justifies
  parallel execution.
- Use `slice1D(...; forloop_iter=...)` as the first multi-GPU option when memory
  allows.
- Use `slice2D(N1,N2; forloop_iter=...)` when peak memory remains too high.
- Use `ifparallelupdown` mainly for two-GPU runs where up/down environments can
  overlap efficiently and D/chi are not too large.
- For Slice2D, keep collective control flow rank-uniform and avoid gathered
  full-env shortcuts.

## D And Chi Growth

- For fresh chi ladders, use `default_χlist(D; ...)`; its default minimum
  boundary dimension is `2 * D^2`.
- For continuation from an already optimized high-chi state, use an explicit
  target chi list such as `[1024]`.
- When increasing chi, inspect the ladder's effective VUMPS limits and cap
  them at 50 unless the user explicitly approves more. The ladder does not
  replace the mandatory target-chi environment gate before optimization.
- If the optimization gradient norm stalls and refuses to decrease, first
  suspect that chi is too small and the physical variational space is poor.
  Hermitian cases may tolerate this better; in non-Hermitian optimizations this
  is especially often a chi bottleneck. Increase chi and recompute the
  environment before spending time on optimizer hyperparameters.
- When increasing D, prepare a matching target-D environment before LBFGS.
  Keep `boundary_alg.maxiter` / `opt_obs_maxiter` and any env/observable warmup
  within the 50-step allowance; do not automatically increase them beyond it.
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

## TM Spectrum Runs

- Before computing a `TM_spectrum` k-point grid, check whether the matching
  boundary environment already exists. Load it when it is trustworthy; recompute
  only when it is missing, stale, incompatible, or suspect.
- Do not recompute the boundary environment separately for each k point. All k
  points in the same sector should load/reuse the same converged environment.
- For the trivial sector, reuse the single matching `χ<chi>.jld2` environment
  across k points.
- For `ifdomainwall=true`, do not treat the setup as left/right two-boundary
  reuse. It is the same side with two different MPS fixed points/environments:
  prepare or load the two matching named environments once, then reuse that pair
  for every k point.
- If saving environments is affordable, keep `ifsave_env=true` for the first TM
  run so later k points and retries can use `ifload_env=true`.

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

- local `D=2`, `chi=16` TeneT.jl same-path smoke evidence
- `using TeneT` and the selected Julia entry script smoke
- CUDA kernel execution
- cuTENSOR/cuBLAS/cuSOLVER availability as applicable to the backend
- MPI rank to GPU mapping
- Slurm GPU count, MPI ranks, and the Julia entry's `parallel_method` must match
  the planned single-GPU or multi-GPU execution.
- MPI/NCCL basics for the chosen parallel method

Do not issue `sbatch`, `squeue`, `sacct`, `scancel`, `rsync`, or remote SSH
commands from this skill. Follow AGENTS.md and hand those operations to
user-global `$hpc` by default. Use `tenet-hpc` when explicitly enabled/invoked
or selected as a supplement by that global HPC workflow.
