# TeneT.jl Repository Rules

This file is the repository source of truth for agentic and human edits. It
turns the project's current design notes, parity tests, and benchmark practice
into rules that are easy to find before touching code.

## Core Principle

Treat tests, oracle gates, benchmark reports, and design documents as the
authority. Do not trust a change because it looks plausible. Trust it after the
right external checks pass.

## Hot-Path Tensor Code

- Do not replace `TensorOperations`, cuTENSOR-backed contractions, or the
  chain engine with naive elementwise tensor loops on production paths.
- When adding a map variant, express the contraction through an existing
  map abstraction or a chain declaration unless a design doc justifies a
  hand-written kernel.
- Preserve eager-free behavior in chain and Slice2D paths. Do not keep large
  intermediates alive for convenience in AD closures.
- New chain declarations need forward parity and gradient parity against the
  trusted path before becoming default behavior.

## AD And Checkpointing

- Do not route high-memory tensor maps back through generic Zygote pullbacks
  when a hand rrule or chain-engine rrule exists.
- Checkpoint changes must preserve the documented semantics of `Plain`,
  `Recompute`, `OffloadRecompute`, and `Offload`.
- Rrules for communication primitives must use the mathematical adjoint:
  gather pairs with reduce-scatter, allreduce is self-adjoint for sums, and
  scatter/gather seams must not over-count replicated cotangents.
- Never weaken gradient tolerances just to make a new implementation pass.
  If a tolerance changes, record the numerical reason in a design or benchmark
  document first.

## Slice2D, MPI, And GPU Rules

- Slice2D changes must keep rank-uniform collective control flow. A branch that
  only some ranks enter is a deadlock bug until proven otherwise.
- Keep MPI tags and communicators separated by operation class. Do not reuse a
  tag across unrelated collectives on the same communicator.
- Keep cluster submissions behind the HPC protocol. Never submit or cancel jobs
  through a generic local gate command.
- Do not run GPU benchmark A/B arms in the same Julia process. Use fresh
  processes so CUDA allocator state and JIT state do not pollute the comparison.
- If a benchmark depends on Sofia, BSC, or JSC environment details, record those
  details with the result.

## Oracle Gates

The gate registry lives in `oracles/gates.toml`. The runner is:

```powershell
julia --project=. scripts/run_oracle_gates.jl --list
```

Use the smallest gate that covers the changed behavior, then broaden if the
change touches shared AD, MPI, boundary, or physics contracts. Cluster gates
are blocked by default and require explicit `--allow-cluster` plus the HPC
protocol.

## Documentation As Source Of Truth

- Dated design docs under `docs/` explain why an algorithm exists.
- Benchmark reports under `docs/benchmarks/` explain whether a performance
  change is acceptable.
- `oracles/README.md` explains the correctness references.
- `docs/oracle-and-gate-workflow.md` explains which gate to run for each kind
  of change.

If code and docs disagree, do not silently pick one. Fix the stale side or
record the discrepancy before building more work on top of it.

## Rust Or Other Backend Experiments

Rust, C++, CUDA, or other backend experiments are welcome as isolated kernels
or oracle-backed prototypes. They should not replace Julia production paths
until they pass the relevant forward, gradient, distributed, and benchmark
gates against the current implementation.
