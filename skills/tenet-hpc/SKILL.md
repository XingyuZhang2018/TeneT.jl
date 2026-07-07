---
name: tenet-hpc
description: "Optional repository-local HPC operations skill for TeneT.jl: Slurm submission, squeue/sacct checks, sync/pull, monitoring, archiving, local history, cluster preflight, and crash retry coordination for TeneT iPEPS jobs on BSC, Sofia, JSC, MareNostrum, and GPU clusters. Use only when this repo-local skill is explicitly enabled or invoked; otherwise prefer the user-global $hpc skill required by AGENTS.md."
---

# TeneT HPC Operations

Use this optional skill for TeneT-specific cluster operations only when it has
explicitly been enabled or invoked. If the user-global `$hpc` skill is available
and AGENTS.md requires it, follow `$hpc` first.

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
- Julia entry script
- `Project.toml`, `Manifest.toml`, and LocalPreferences files used by the run
- source commit, source bundle, or worktree archive metadata
- `SUBMISSION.md`
- helper preflight scripts

After submission, update `SUBMISSION.md` with the remote workdir and Slurm ID.

## Cluster Preflight

For TeneT GPU jobs, include a short preflight before the expensive entry:

- `using TeneT`
- rank to GPU mapping
- CUDA kernel execution
- cuTENSOR contraction availability
- MPI/NCCL setup for the requested parallel method

If a preflight fails, fix the environment before running the physics entry.

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
- retry protocol after a classified failure
