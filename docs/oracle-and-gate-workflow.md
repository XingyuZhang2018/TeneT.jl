# Oracle And Gate Workflow

This document is the operational companion to `REPOSITORY_RULES.md` and
`oracles/gates.toml`.

## Choose Gates By Blast Radius

- Documentation-only change: run `--gate oracle_runner` if gate metadata or
  workflow docs changed.
- Tensor map or chain-engine change: run `--gate chain_maps`; if distributed
  wrappers are affected, also run the relevant `slice2d_*` gate.
- AD, rrule, or checkpoint change: run `--gate checkpoint` plus the smallest
  map/env gate that exercises the changed rrule.
- Boundary algorithm change: run the matching VUMPS/QRCTMRG local test and the
  relevant Slice2D gate when `grid` or `parallel_method` behavior is touched.
- Physics energy or observable change: run the model-specific test plus any
  analytic or parity oracle listed in `oracles/gates.toml`.
- MPI collective or Slice2D communication change: run local MPI gates first;
  use cluster GPU gates only after the local parity signal is clean.
- GPU performance change: use a benchmark report under `docs/benchmarks/`.
  Run each benchmark arm in a fresh Julia process.

## Gate Runner

List gates:

```powershell
julia --project=. scripts/run_oracle_gates.jl --list
```

Dry-run a group:

```powershell
julia --project=. scripts/run_oracle_gates.jl --group local-mpi --dry-run
```

Run a single local gate:

```powershell
julia --project=. scripts/run_oracle_gates.jl --gate chain_maps
```

Cluster gates are blocked by default:

```powershell
julia --project=. scripts/run_oracle_gates.jl --gate sofia_slice2d_m5_gpu
```

To submit a cluster gate, first follow the HPC protocol in `AGENTS.md` and the
`$hpc` skill, then pass `--allow-cluster` only when the submit script and target
job are intentional.

## Recording Results

- Local test results can be summarized in the PR or worklog.
- Benchmark results belong under `docs/benchmarks/`.
- Design-level acceptance criteria belong in the corresponding dated design
  document.
- If a gate is skipped, record why. "Too slow" is acceptable only with a smaller
  gate run in its place.
