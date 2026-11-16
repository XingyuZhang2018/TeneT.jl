# Automatic contraction chunking (`forloop_iter = 0`) as the default

Date: 2026-11-16

## Change

`forloop_iter = 0` now means automatic chunking, and it is the default on
`VUMPS`, `QRCTMRG`, `GradientOptimize`, `slice1D` and `slice2D`.

Every map that chunks a leg resolves its own chunk count from the local length
`L` of the leg it actually slices:

    n = resolve_forloop_iter(forloop_iter, L) = forloop_iter > 0 ? forloop_iter : max(1, cld(L, 8))

- Serial `forloop` / `forloop_sum`: `L` is the sliced χ leg (χ = 192 → 24 chunks).
- `slice1D` (`parallel` / `parallel_sum`): `L = cld(D_split, nprocs)`, the
  per-rank share, so each rank's slices are at most 8 wide.
- `slice2D` maps: `L` is the local `l` block after the grid partition; the
  cuTENSOR 2^31 floor in `_ring_l_chunks` still applies on top.

Positive values keep their old meaning (exact chunk count). Negative values
throw `ArgumentError`. The resolved count depends only on tensor shapes, so a
forward map and its rrule always use the same partition, and all Slice2D ranks
resolve the same count (rank-uniform control flow).

## Why

With the old default `forloop_iter = 1` (no chunking), large-χ runs hit memory
limits before users knew to tune the knob. Observed on an RTX 4090 (24 GB),
measuring a merged-honeycomb (d = 4) D = 4 iPEPS at χ = 192 with VUMPS General
from a fresh environment, each configuration in its own Julia process:

| `forloop_iter` | GPU memory | time per VUMPS step | outcome |
|---|---|---|---|
| 1 | 24 GB (full) | — | did not finish in 45 min |
| 24 (= cld(192, 8)) | 9.8 GB | 31 s | converged to err < 1e-10 in 44 steps |

The converged energy matched an independent reference to 10 significant
digits.

Width 8 is a starting point, not an optimum for every shape: narrower slices
reduce intermediates but launch more kernels; wider slices can be faster when
memory allows. Explicit values remain available for tuning.

## Numerical effect

Chunking reorders floating-point accumulation, so results with the automatic
default can differ from `forloop_iter = 1` at the rounding level (see
`2026-04-17-inner-float32-vumps-design.md`). Small-χ runs (χ ≤ 8) resolve to a
single chunk and are unchanged.

Full-pipeline check (VUMPS AD steps + energy, merged-honeycomb D = 3 state,
χ = 36 → 5 automatic chunks, CPU, one shared converged environment, same seed):

| Comparison | relative gradient difference |
|---|---|
| auto vs 1, model without randomized loops (\|g\| = 0.61) | 2.3e-11 |
| auto vs 1, model with randomized 2×2 loops (\|g\| = 5e-5) | 9.7e-2 |
| forloop 1, seed A vs seed B, same model (\|g\| = 5e-5) | 1.1e-1 |
| forloop 1, same seed, repeated | 0 |

Without randomized loops the chunked gradient matches to rounding. With the
randomized 2×2 loop (`oc_22`) the near-zero gradient of a nearly optimized
state is dominated by the loop's random-start variation; chunking moves it by
the same amount as changing the seed, through the QR pullback of the random
range finder. Forward energies agree to 1e-15 with a shared environment.

## Tests

- `test/test_contraction.jl`: `resolve_forloop_iter` values and errors;
  `FLmap_parallel` with `forloop_iter = 0` matches `FLmap` in the forward pass
  and its Zygote gradient matches `forloop_iter = 1` (χ = 20 → 3 chunks).
- `test/test_types.jl`: default fields are now `0`.
- Full serial suite (`Pkg.test()`, CPU): 2240 / 2240 pass.
- Not run here: local-MPI Slice1D/Slice2D gates and GPU cluster gates
  (no MPI launcher on the development machine).
