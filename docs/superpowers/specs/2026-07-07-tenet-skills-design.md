# TeneT Skills Design

Date: 2026-07-07

## Goal

Create repository-local Codex skills for TeneT so agents working in this repo
can reliably choose production iPEPS optimization settings and, optionally,
follow the TeneT-specific HPC workflow.

The primary skill is the TeneT optimization skill. Its display name should be
`TeneT`, while the technical skill folder/name should be `tenet` because Codex
skill names are lowercase hyphen-case. Its job is to guide iPEPS run design:
model example selection, GPU use, environment and LBFGS continuation, memory
controls, VUMPS/power-method choices, parallel methods, D/chi growth, and
observable-only runs.

The optional HPC skill should be separate, with a technical name such as
`tenet-hpc`, to avoid colliding with any user-global `$hpc` skill. Its job is
to capture TeneT-specific supercomputer operating rules and to delegate physics
parameter choices back to the TeneT optimization skill.

## Non-Goals

- Do not merge TeneT optimization guidance and Slurm/HPC operations into one
  skill.
- Do not duplicate the full user-global HPC skill implementation inside the
  TeneT optimization skill.
- Do not create run-generation scripts in this pass unless the later
  implementation plan explicitly calls for them.
- Do not change TeneT.jl runtime code, examples, tests, or physics behavior as
  part of creating the skills.

## Skill Layout

Add source skills under the repository root:

```text
skills/
  tenet/
    SKILL.md
    agents/openai.yaml
  tenet-hpc/
    SKILL.md
    agents/openai.yaml
```

`skills/tenet/SKILL.md` is the canonical repo source for the TeneT skill.
`skills/tenet-hpc/SKILL.md` is optional and intentionally narrower than the
existing user-global HPC skill.

For Codex auto-discovery on the current machine, link or copy the primary TeneT
skill into `$CODEX_HOME/skills` (falling back to `~/.codex/skills`) during
implementation:

```text
$CODEX_HOME/skills/tenet      -> <repo>/skills/tenet
```

Prefer a directory junction/symlink when supported so future repo edits are
immediately visible to Codex. If links are unavailable, copy the directories and
document that they need refreshing after repo edits.

Keep `tenet-hpc` in the repository as an opt-in skill. Do not auto-install it
by default on this machine because the user already has a global `$hpc` skill
and AGENTS.md directs HPC requests there. If the user later asks to enable the
repo-local HPC skill, link or copy:

```text
$CODEX_HOME/skills/tenet-hpc  -> <repo>/skills/tenet-hpc
```

## TeneT Skill Content

The TeneT skill should stay concise and procedural. It should tell Codex to
inspect examples and recent submitted jobs before drafting or changing a
production run.

Core rules:

- Use GPU for `D >= 5` production work. CPU is acceptable for tiny local smoke
  tests, not for serious high-D optimization.
- Keep `ifsimple_eig=true` for real iPEPS optimization. Only set it false when
  the task is explicitly about studying eigensolver behavior.
- Use VUMPS defaults for the power method unless the run history motivates a
  change. The current production baseline is `power_iter=5`,
  `power_iter_ad=5`, and `power_iter_obs=40`.
- Set `ifsave_lbfgs=true` for optimization runs; set `ifload_lbfgs=true` when
  continuing from a saved LBFGS checkpoint.
- Prefer `ifload_env=true` when an environment exists and is trustworthy:
  matching model, pattern, D, chi, contraction mode, parallel method, and no
  evidence of env non-convergence or a bad trend. Turn it off only when the env
  is stale, incompatible, missing, too suspect, or intentionally being rebuilt.
- Set `ifsave_env=true` when the environment file size is acceptable, with
  `<10GB` as the practical threshold. Avoid saving very large env files by
  default.
- Treat saved iPEPS files, history rows, environment files, and LBFGS
  checkpoints as durable artifacts. Do not continue from a forward-energy-only
  candidate that timed out before writing `history.log`, `No.N.jld2`, or an
  LBFGS checkpoint.

Memory rules:

- First enable `ifcheckpoint=true` on both `GradientOptimize` and the VUMPS
  boundary algorithm.
- Increase `forloop_iter` before moving to heavier remedies, but cap or adapt
  it for small chi stages. A requested `forloop_iter` larger than the active chi
  can be harmful.
- If memory still fails, set VUMPS `inner_checkpoint=Recompute()`.
- If that still fails, prefer multi-GPU parallelism; use `slice1D` first when it
  fits, then `slice2D` when peak memory remains too high.
- For large Kagome-style high-D/high-chi runs, the proven production shape is
  `slice2D(4,4)` on 16 GPUs with high `forloop_iter`, but the skill should
  still size this from available GPUs and model scale.

Parallelism rules:

- `ifparallelupdown` is mainly for two-GPU runs where up/down work can overlap
  efficiently and D/chi are not too large.
- Prefer the newer `parallel_method` API over ad hoc legacy flags when writing
  new runs.
- Use `slice1D(...; forloop_iter=...)` as the first multi-GPU option when
  memory allows.
- Use `slice2D(N1,N2; forloop_iter=...)` when the issue is peak memory, not
  just throughput.

D and chi growth rules:

- For fresh chi ladders, use `default_χlist(D; ...)`. For continuation from an
  already optimized high-chi state, use an explicit target list such as
  `[1024]`.
- When increasing chi, rely on the existing staged optimization behavior, which
  recomputes the boundary environment at the new chi with the configured VUMPS
  settings.
- When increasing D, do not assume the default env solve is enough. Use more
  VUMPS effort for the first D-upgrade stage, for example larger
  `boundary_alg.maxiter` / `opt_obs_maxiter`, or an explicit env/observable
  warmup before the main LBFGS continuation.
- Prefer `init_ipeps_SU` for D growth. Use `SUτ=0.1` as the current practical
  default for initialization, then reset `params.SUτ=0.0` and run with
  `ifSU=false` before calling `optimise_ipeps`.
- Record that energy can rise immediately after a large-D SU initialization,
  especially at larger D, and may drop quickly once gradient optimization starts.

Model and example rules:

- Start from the closest existing example under `examples/` before inventing a
  contraction mode, lattice orientation, pattern, or `restriction_ipeps`.
- Treat model-specific restrictions as part of the physics setup, not boilerplate.
  Honeycomb brickwall, Kagome merge, C4v, Plaquette, General, and Oneside paths
  require different choices.
- For different models, inspect the example contraction method and recent
  submitted `entry.jl` files before deciding between General, Plaquette, C4v,
  Oneside, QRCTMRG, Slice1D, or Slice2D.

Operational rules:

- Observable-only runs should be recognized as a separate mode: no LBFGS
  continuation, `maxiter_ad=0`, and no unnecessary LBFGS save/load.
- New Slice2D preconditioner usage should start with a smoke test for the
  relevant model/grid before a long production run.
- High-D cluster runs should include a short preflight for `using TeneT`, CUDA
  kernel execution, cuTENSOR availability, rank-to-GPU mapping, and MPI/NCCL
  basics before the expensive entry.

## TeneT HPC Skill Content

The optional HPC skill should be a TeneT-specific operations wrapper, not a
physics-parameter guide.

It should cover:

- Slurm submission, sync, monitoring, archiving, and retry workflow for TeneT
  runs.
- Safety rules such as never cancelling all jobs by user and only cancelling
  explicit Slurm IDs.
- Local history requirements: archive every submitted script, Julia entry,
  manifest/project/preferences, source commit or bundle, and `SUBMISSION.md`.
- Cluster preflight expectations for BSC/Sofia/JSC when running TeneT.
- The boundary with `tenet`: use `tenet` for iPEPS parameter choices; use
  `tenet-hpc` for submitting, monitoring, syncing, and preserving run history.

It should not duplicate the detailed iPEPS optimization heuristics above. It may
link to or instruct Codex to use the TeneT skill first when the user asks for a
new iPEPS production run.

## Data Flow

For a new production run:

1. Use the TeneT skill to classify the physics task: fresh run, chi continuation,
   D upgrade, observable-only, precondition smoke, or failure retry.
2. Inspect the closest example and recent submitted jobs for the same model or
   contraction path.
3. Select model, pattern, restriction, contraction mode, GPU/parallel method,
   env/LBFGS save-load policy, checkpoint policy, and D/chi growth strategy.
4. If the task requires a cluster, hand off the operational part to the HPC skill
   or the user-global `$hpc` skill.
5. Preserve the generated run artifacts and submit history through the HPC
   workflow.

For a continuation:

1. Verify the source artifact is durable.
2. Prefer loading LBFGS and a trustworthy env.
3. Keep source and target D/chi/model/mode compatibility explicit in the run
   notes.
4. If the previous job timed out, continue from the latest saved `No.N` and
   LBFGS checkpoint, not from an unsaved candidate.

## Error Handling

The skills should tell Codex to stop and diagnose rather than guess when:

- The closest example uses a different lattice orientation or contraction mode.
- The env file exists but its provenance does not match the requested run.
- A run hit CUDA OOM, cuTENSOR errors, MPI preference errors, or rank/device
  initialization failures.
- A Slurm job reports `COMPLETED 0:0` but stdout/stderr contain fatal signatures.
- A D-upgrade run has no durable checkpoint after a timeout.

For OOM retries, use the memory ladder in order: checkpointing, larger/adaptive
`forloop_iter`, `inner_checkpoint=Recompute()`, then Slice2D or more GPUs.

## Validation

Skill implementation should be validated with:

- `quick_validate.py` on both skill folders.
- A content check that the Codex-discovered `tenet` copy or link points at the
  repo source skill.
- A dry-run prompt asking for a TeneT iPEPS D-upgrade plan, verifying the answer
  chooses GPU, SU D growth with `SUτ=0.1`, env/LBFGS policy, VUMPS effort for D
  growth, and delegates Slurm details to the user-global `$hpc` skill or to
  `tenet-hpc` if that optional repo skill has been enabled.
- A dry-run prompt asking for TeneT job sync/monitoring, verifying the answer
  uses an HPC skill and does not duplicate optimization heuristics.
