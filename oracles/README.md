# TeneT.jl Oracles

This directory records the external checks that decide whether an
implementation is correct enough to trust. An oracle is not "the code looks
reasonable"; it is a reproducible reference, invariant, parity comparison, or
benchmark gate.

## Registry

`gates.toml` is the machine-readable gate registry. List it with:

```powershell
julia --project=. scripts/run_oracle_gates.jl --list
```

Run the default quick gate:

```powershell
julia --project=. scripts/run_oracle_gates.jl
```

Run one gate:

```powershell
julia --project=. scripts/run_oracle_gates.jl --gate chain_maps
```

Run a group:

```powershell
julia --project=. scripts/run_oracle_gates.jl --group local-mpi
```

Cluster gates are intentionally blocked unless `--allow-cluster` is passed.
Even then, follow the project HPC protocol before submitting jobs.

## Oracle Classes

- Analytic physics: exact results such as the 2D classical Ising free energy.
- Serial parity: a new path must match the trusted serial path within the
  documented tolerance.
- Gradient parity: AD and hand rrules must agree with trusted references.
- Distributed parity: Slice2D/MPI paths must match serial or replicated
  reference computations.
- Benchmark gates: performance changes are accepted or rejected against a
  recorded, reproducible baseline.

## Adding A Gate

1. Add or identify the test, benchmark, or launcher.
2. Add a `[gate.<name>]` entry to `gates.toml`.
3. Put the gate in at least one group.
4. Mark `requires_mpi`, `requires_gpu`, and `requires_cluster` accurately.
5. Run `julia --project=. scripts/run_oracle_gates.jl --list`.
6. If the gate is local and cheap, run it directly.

Do not weaken an oracle to make a change pass. If a tolerance needs changing,
record the numerical reason in the design or benchmark report first.
