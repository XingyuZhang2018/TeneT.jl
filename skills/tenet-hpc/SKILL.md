---
name: tenet-hpc
description: "Optional repository-local HPC operations supplement for TeneT.c/TeneT.jl: Slurm submission, squeue/sacct checks, sync/pull, monitoring, archiving, local history, resource matching, local TeneT.c-vs-TeneT.jl parity gates, cluster preflight, and crash retry coordination for TeneT iPEPS jobs on BSC, Sofia, JSC, MareNostrum, and GPU clusters. Use when explicitly invoked or when the user-global $hpc skill detects a TeneT job and reads repo-local supplements."
---

# TeneT HPC Operations

Use this optional skill for TeneT-specific cluster operations when explicitly
enabled or invoked, or when the user-global `$hpc` skill detects a TeneT job and
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
- TeneT.c app/command script and build metadata, or the Julia entry script when
  a TeneT.jl fallback was explicitly approved
- `CMakeLists.txt`, compile options, source commit, and binary path for TeneT.c;
  for TeneT.jl fallback also archive `Project.toml`, `Manifest.toml`, and
  LocalPreferences files used by the run
- source commit, source bundle, or worktree archive metadata
- `SUBMISSION.md`
- helper preflight scripts

After submission, update `SUBMISSION.md` with the remote workdir and Slurm ID.

## Resource Match

Before submitting, verify the Slurm allocation matches the TeneT entry:

- Do not request multiple GPUs for an entry that runs single-GPU code.
- Match GPU count, MPI rank count, `CUDA_VISIBLE_DEVICES`, and the TeneT.c
  app's Slice1D/Slice2D grid (`n1`, `n2`, rank count, or equivalent flags). For
  an approved TeneT.jl fallback, match the Julia `parallel_method`.
- For single-GPU runs, request one GPU and one GPU-driving rank.
- For multi-GPU runs, confirm every allocated GPU is used by Slice1D, Slice2D,
  or another explicit parallel path before submission.
- Record the intended backend (`TeneT.c` by default), GPU count, rank count,
  and parallel method/grid in
  `SUBMISSION.md`.

## Local Parity Gate

Before submitting any TeneT HPC job:

- Run a local D=2, chi=16 comparison between TeneT.c and TeneT.jl for the same
  model, lattice/pattern, contraction family, seed or exported fixture, and
  relevant observable/optimization path.
- Treat TeneT.c as the production backend and TeneT.jl as the reference oracle.
- Archive the commands, logs, compared scalar values, tolerances, and pass/fail
  result in the local rendered history directory.
- Block submission when the parity check is missing, failed, or not applicable.
  If TeneT.c lacks the needed capability, ask the user before using TeneT.jl as
  the production backend.

## Cluster Preflight

For TeneT GPU jobs, include a short preflight before the expensive entry:

- local D=2, chi=16 TeneT.c/TeneT.jl parity evidence
- TeneT.c build and selected app smoke, or `using TeneT` only for an approved
  TeneT.jl fallback
- rank to GPU mapping
- CUDA kernel execution
- cuTENSOR/cuBLAS/cuSOLVER contraction and linear algebra availability as
  applicable to the backend
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
- enforcing the local TeneT.c/TeneT.jl parity gate before submission
- retry protocol after a classified failure
