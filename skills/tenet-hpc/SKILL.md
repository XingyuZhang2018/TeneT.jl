---
name: tenet-hpc
description: "Optional repository-local HPC operations supplement for TeneT.jl: Slurm submission, queue/accounting checks, sync, monitoring, archiving, local history, resource matching, local TeneT.jl smoke gates, Julia/CUDA/MPI preflight, and crash retry coordination for TeneT.jl iPEPS jobs on BSC, Sofia, JSC, MareNostrum, and GPU clusters. Use when explicitly invoked or when the user-global $hpc skill detects a TeneT.jl job and reads repo-local supplements."
---

# TeneT.jl HPC Operations

Use this optional skill for TeneT.jl-specific cluster operations when explicitly
enabled or invoked, or when the user-global `$hpc` skill detects a TeneT.jl job and
reads repo-local supplements. If AGENTS.md requires `$hpc`, follow `$hpc` first.

`$hpc` safety, cancellation, history, and ledger rules remain authoritative;
this skill only adds TeneT-specific preflight and provenance details.

This skill handles operations. It does not choose iPEPS physics parameters. Use
`tenet` first for model, D/chi, environment, checkpointing, and parallel-method
decisions.

## Safety

- Never run `scancel -u <user>` or any user-wide cancellation.
- Cancel only explicit Slurm job IDs that `$hpc` or the active run context has
  identified. Let `$hpc` resolve the authoritative shared ledger before any
  cancellation.
- Ask before cancelling more than five jobs.
- Treat `COMPLETED 0:0` as insufficient when stdout or stderr contains fatal
  signatures.
- Never delete remote run or data directories unless the user explicitly asks.
- Keep internal forward/untaped VUMPS solves within the user-required 50-step
  allowance unless explicitly authorized otherwise. Enforce the `tenet` and
  `$hpc` cumulative accounting across segments, retries and refinements;
  templates or automation prompts do not authorize additional work.

## Responsibilities

Use this skill for:

- Preparing a TeneT Slurm submission after `tenet` has fixed the run parameters.
- Checking `squeue`, `sacct`, stdout, stderr, and progress files.
- Pulling selected outputs back to local history directories.
- Recording Slurm IDs, source commit or bundle, remote workdir, and key runtime
  choices in `SUBMISSION.md`.
- Monitoring live jobs and classifying failures before retry.
- Coordinating crash-smoke retries before resubmitting long production jobs.

## Local History

Before submitting, create a local history directory under the project-level HPC
tree, not under the TeneT package root:

```text
D:\1 - research\1.26 - iPEPS_opt\hpc\rendered\<timestamp>_<run-name>
```

Archive at least:

- Slurm script
- exact Julia entry script and command
- `Project.toml`, `Manifest.toml`, LocalPreferences files, Julia version, and
  package/artifact metadata used by the run
- source commit, source bundle, or worktree archive metadata
- `SUBMISSION.md`
- helper preflight scripts

After submission, update `SUBMISSION.md` with the remote workdir and Slurm ID.

## Resource Match

Before submitting, verify the Slurm allocation matches the TeneT entry:

- Do not request multiple GPUs for an entry that runs single-GPU code.
- Match GPU count, MPI rank count, `CUDA_VISIBLE_DEVICES`, and the Julia entry's
  `parallel_method` and Slice1D/Slice2D grid.
- For single-GPU runs, request one GPU and one GPU-driving rank.
- For multi-GPU runs, confirm every allocated GPU is used by Slice1D, Slice2D,
  or another explicit parallel path before submission.
- Record GPU count, rank count, and parallel method/grid in `SUBMISSION.md`.

## Local TeneT.jl Smoke Gate

Before submitting any TeneT.jl HPC job:

- Run a local `D=2`, `chi=16` TeneT.jl smoke for the same model,
  lattice/pattern, contraction family, scalar type, and relevant observable or
  optimization path.
- Require finite output, expected terminal markers, and successful persistence
  or load behavior when the production run depends on it.
- Archive commands, logs, values, tolerances, and pass/fail result in the local
  rendered history directory. Block submission when the smoke is missing or
  failed. Do not require a cross-library comparison.

## Environment-First Submission Gate

Apply the default gate below unless the user's explicit run-specific gate
placement says otherwise; follow `$hpc` and `tenet` for that exception. The
50-step forward limit and cumulative accounting still apply.

1. Check whether a matching saved environment already has recorded
   `err <= 1e-6` evidence. If not, submit an environment-only job with at most
   50 forward boundary VUMPS iterations, optimizer/LBFGS disabled, and
   `ifsave_env=true`.
2. After the job ends, verify scheduler state, logs, the exact final VUMPS
   error, a finite environment, and a durable loadable environment artifact.
   Record cumulative VUMPS steps, path, checksum, and final error. Mark the
   attempt as `ENV_SUCCESS` only when the convergence gate passes; otherwise
   preserve it as `ENV_PARTIAL`.
3. For `ENV_PARTIAL`, continue only within the remaining authorized allowance,
   with `ifload_env=true` and `ifsave_env=true`. Verify the prior environment
   load and include its consumed steps in the cumulative count. Without an
   explicit larger authorization, stop at 50 cumulative forward steps on the
   same physical tensor, target chi and side. Preserve the partial environment
   and obtain authorization before extending; restarting or creating another
   job never resets the count. When a larger cumulative allowance is explicitly
   approved, retain its prior usage and the per-block cap specified by the user.
4. Do not submit the optimizer merely as a scheduler dependency behind an
   unvalidated environment job. Wait for scientific `ENV_SUCCESS`, then submit
   optimization with `ifload_env=true` and require an environment-load success
   marker before LBFGS begins.
5. If the final error is close to but above `1e-6`, report the exact residual
   and trend and wait for explicit user acceptance before releasing
   optimization.

The environment and optimization must use compatible model, tensor checkpoint,
restriction, pattern, D, chi, contraction family, orientation, scalar type, and
parallel layout. A scheduler `COMPLETED` state alone never passes this gate.

## Cluster Preflight

For TeneT GPU jobs, include a short preflight before the expensive entry:

- local `D=2`, `chi=16` TeneT.jl same-path smoke evidence
- `using TeneT` and the selected Julia entry script smoke
- rank to GPU mapping
- CUDA kernel execution
- cuTENSOR/cuBLAS/cuSOLVER contraction and linear algebra availability as
  required by the Julia/CUDA path
- MPI/NCCL setup for the requested parallel method

If a preflight fails, fix the environment before running the physics entry.

## Live Monitor Layout

Use the user-global
[HPC monitor layout](C:/Users/xingzhan/.codex/skills/hpc/reference/dashboard-layout.md)
for TeneT dashboards. It defines the top parameter fold, current-work cards,
separate historical-job and historical-curve folds, cumulative chi plots,
decimal energies and seed/initialization comparisons. Follow the adjacent
`dashboard.md` for local-cache refresh and collection; this display guidance
changes no numerical policy or job authorization.

## Boundary With `tenet`

Use `tenet` for:

- GPU/CPU choice
- D/chi growth policy
- environment and LBFGS save/load policy
- VUMPS and checkpoint settings
- Slice1D/Slice2D choice
- model examples and `restriction_ipeps`

Use this skill for:

- Slurm mechanics
- safe cancellation
- sync and monitoring
- provenance and local history
- enforcing the local TeneT.jl smoke gate before submission
- retry protocol after a classified failure
