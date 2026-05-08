# Distributed FR in VUMPS Runtime

## Problem

In `ACmap_parallel`, each GPU stores the full FR tensor but only uses its local
slice (split on the last dimension). Peak memory includes a redundant
`(1 - 1/nprocs)` fraction of FR per GPU — not just during ACenv's eigensolver,
but for the entire VUMPS iteration since `VUMPSRuntime` stores FR.

## Design

Store only the local FR slice in `VUMPSRuntime`. Each `vumps_step` allgathers
FR once at the start (for `rightenv` and `Cenv`), then prescatters the updated
FR before `ACenv` and stores the distributed version back in the runtime.

### Contraction analysis (leg4 case)

```
result[f,g,h] := AC[a,b,c] * FR[c,e,h] * M[d,g,e,b] * FL[a,d,f]
```

| Tensor | Per-GPU need | Reason |
|--------|-------------|--------|
| FR     | **slice only** (h_slice) | h is the parallelized free index |
| FL     | full | f is free, a/d contracted across all values |
| AC     | full | a/b/c all contracted |
| M      | full | d/g/e/b contracted or free |

### Changes

1. **`prescatter_for_parallel` / `allgather_for_parallel`** in
   `forloop_parallel_MPI.jl` — scatter/gather a single tensor along `split_dim`.
   Each has a custom rrule in `rules.jl`.

2. **`parallel()` in forloop_parallel_MPI.jl** — `input_prescattered::Bool` kwarg.
   When true, `args[N_in[1]]` is already the local portion for this rank.

3. **`ACmap_parallel` / `ACmap(I::Int, ...)`** — thread `fr_distributed` kwarg
   through to `parallel(...; input_prescattered=true)`.

4. **`_prescatter_FR` / `_allgather_FR`** in `general.jl` — StructArray-level
   helpers that apply prescatter/allgather to each unique data tensor.

5. **`vumps_step` / `vumps_step_power`** — allgather FR at start, run rightenv
   and Cenv with full FR, prescatter FR, run ACenv with distributed FR, store
   distributed FR in returned runtime.  Cenv is reordered before ACenv.

6. **`init_VUMPSRuntime`** — prescatter FR after `rightenv` when `ifparallel`.

7. **`ACenv()`** — accepts `fr_distributed` kwarg from caller; no longer
   prescatters internally.

8. **`ObsEnv()`** — allgathers FR before rightenv and VUMPSEnv construction.

9. **`_simple_eig_ACmap`** — threads `fr_distributed` to ACmap closures.

### Unchanged

- `simple_eig()` (operates on full, allgathered AC)
- `rightenv()` / `FRmap` (produces full FR as before)
- `Cenv()` / `Cmap()` (uses full FR from allgather)
- `leftenv()` / `FLmap`
- Serial path (`ifparallel=false`)
- Plaquette / C4v variants (use FL on both sides, no FR)
- `VUMPSRuntime` struct (untyped `FR::StructArray` — shape change is transparent)

### Communication cost

One extra `allgatherv` per `vumps_step` (to reconstruct full FR for rightenv/Cenv).
This is a single collective per step — negligible relative to the eigensolver work.

### Memory savings

Per GPU, persistent (entire VUMPS iteration):
`size(FR) * (1 - 1/nprocs)`.

Full FR exists only briefly within each `vumps_step` (between allgather and
prescatter). After prescatter, `FR_full` is unreferenced and eligible for GC.

For D=16, d=2, chi=768, Float64, nprocs=4: ~0.84 GiB / GPU (per site).
For D=16, d=2, chi=768, Float64, nprocs=8: ~0.98 GiB / GPU (per site).

### AD correctness

Forward: `FR_dist → allgather → FR_full → rightenv → FR_full' → prescatter → FR_dist'`

Backward (chain rule):
- prescatter backward: zero-pad + allreduce → full gradient
- rightenv backward: full → full
- allgather backward: extract local slice (no communication)
