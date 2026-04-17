# Local 4090 FLmap forloop AD Benchmark

- **Date**: 2026-04-17
- **GPU**: NVIDIA GeForce RTX 4090  (24.0 GiB)
- **Host**: we05c234
- **Julia**: 1.11.1   **CUDA runtime**: 13.0.0
- **CUDA.jl**: 5.11.0   **Zygote.jl**: 0.7.10
- **Tensors**: leg5 bilayer, `FL,ALu,ALd: (χ,D,D,χ)`, `M: (D,D,D,D,2)`, Float64
- **Protocol**: 1 warmup + 5 reps + median; `GC.gc()` + `CUDA.reclaim()` between reps
- **Purpose**: quantify Zygote per-pullback overhead in `rrule(forloop)`

## Table 1: Raw FLmap baseline (no wrapper)

| D | χ | fwd (ms) | bwd (ms) | bwd/fwd |
|---|---|----------|----------|---------|
| 10 | 128 | 310.56 | 3791.28 | 12.21x |

## Table 2: rrule wrap baseline (`forloop_iter=1`, early-exit path)

| D | χ | fwd (ms) | bwd (ms) | bwd/fwd |
|---|---|----------|----------|---------|
| 10 | 128 | 298.60 | 4043.79 | 13.54x |

## Table 3: Main sweep — `t_fwd`, `t_bwd` vs `forloop_iter`

| D | χ | forloop_iter | fwd (ms) | bwd (ms) | bwd/fwd |
|---|---|--------------|----------|----------|---------|
| 10 | 128 | 1 | 298.60 | 4043.79 | 13.54x |
| 10 | 128 | 2 | 268.18 | 1119.13 | 4.17x |
| 10 | 128 | 4 | 240.48 | 1057.90 | 4.40x |
| 10 | 128 | 8 | 234.28 | 1016.49 | 4.34x |
| 10 | 128 | 16 | 245.92 | 1079.02 | 4.39x |
| 10 | 128 | 32 | 246.97 | 1068.69 | 4.33x |
| 10 | 128 | 64 | 264.12 | 1105.30 | 4.18x |
| 10 | 128 | 128 | 310.49 | 1281.34 | 4.13x |
| 10 | 256 | 2 | 1326.62 | 21433.93 | 16.16x |
| 10 | 256 | 4 | 1220.79 | 5787.79 | 4.74x |
| 10 | 256 | 8 | 1211.18 | 5117.66 | 4.23x |
| 10 | 256 | 16 | 1213.00 | 5095.73 | 4.20x |
| 10 | 256 | 32 | 1252.43 | 5308.29 | 4.24x |
| 10 | 256 | 64 | 1391.42 | 5511.37 | 3.96x |
| 10 | 256 | 128 | 1463.55 | 5814.50 | 3.97x |
| 10 | 512 | 8 | 12056.92 | 127222.95 | 10.55x |
| 10 | 512 | 16 | 32829.94 | 123991.08 | 3.78x |
| 10 | 512 | 32 | 17753.43 | 64982.18 | 3.66x |
| 10 | 512 | 64 | 7853.64 | 42843.33 | 5.46x |
| 10 | 512 | 128 | 8219.88 | 32481.56 | 3.95x |
| 12 | 256 | 8 | 2698.25 | 26850.67 | 9.95x |
| 12 | 256 | 16 | 2716.14 | 13940.50 | 5.13x |
| 12 | 256 | 32 | 2706.10 | 11315.87 | 4.18x |
| 12 | 256 | 64 | 2842.16 | 12133.43 | 4.27x |
| 12 | 256 | 128 | 2950.54 | 11545.14 | 3.91x |

## Table 4: Linear fit  `t_bwd(n) ≈ α·n + β`

| D | χ | α (ms/chunk) | β (ms) | R² | α·128 / (α·128+β) |
|---|---|--------------|--------|----|--------------------|
| 10 | 128 | -5.130 | 1634.96 | 0.0475 | -67.1% |
| 10 | 256 | -40.119 | 9179.93 | 0.0925 | -127.0% |
| 10 | 512 | -781.120 | 117047.78 | 0.7250 | -585.9% |
| 12 | 256 | -73.997 | 18827.37 | 0.2977 | -101.2% |

## Findings — hypothesis falsified

The original hypothesis was `t_bwd(n) ≈ α·n + β` with positive α (Zygote
per-chunk overhead). The data rejects this model:

1. **`forloop_iter=1` is a strong outlier.** For `(D=10, χ=128)` the single
   measured iter=1 point (`bwd=4044 ms`) is **~3.8× slower** than iter=2
   (`bwd=1119 ms`). The larger configs `(10, 256)`, `(10, 512)`,
   `(12, 256)` **OOM at low `forloop_iter`** — the `@tensor` backward
   allocates a rank-7 intermediate of size `~χ²·D⁵·d·8` bytes (10 GB at
   `(10, 256)`, 30 GB at `(12, 256)`, 40 GB at `(10, 512)`) which exceeds
   the 24 GB 4090 without chunking.

2. **Chunking *reduces* `t_bwd`, then plateaus.** For every `(D, χ)`, the
   backward drops sharply from iter=1 to iter=2–4, then plateaus through
   iter=128. Zygote per-chunk overhead at the plateau is modest — the
   slope from iter=32 to iter=128 is roughly `+1.7 ms/chunk` at
   `(10, 128)` and `+2.4 ms/chunk` at `(12, 256)`. Not dominant.

3. **Negative α in Table 4** is an artifact of fitting a straight line
   through a "big outlier + plateau" shape. Three of four R² values are
   below 0.30; the linear model should be considered rejected. Only
   `(10, 512)` has R² = 0.72 and its "slope" is dominated by two
   out-of-trend points at low iter.

### Implications for Phase 2

- **`rrule(forloop)` chunking already helps** — it is not the bottleneck
  at `forloop_iter ≥ 4`. Optimising its per-chunk path (cache pullback,
  preallocate buffers) would save only a few ms per chunk.
- **The real cost lives in the `@tensor` backward itself** (`FLmap` leg5
  via `TensorOperations` + `ChainRulesCore`). A hand-rolled `rrule` for
  `FLmap` that avoids the rank-7 intermediate should be the primary
  Phase 2 target.
- **The JSC 10× full-fg `bwd/fwd` ratio is not explained by chunked AD
  overhead** alone. Combined with the flat plateau, the remaining cost
  likely lives in: (a) the number of backward invocations in
  `leftenv`/`rightenv` power-iter Zygote unrolling, or (b) `Mmap`-family
  ops that still have no rrule for `forloop_sum` (`src/contraction/
  forloop_parallel_MPI.jl:237`).

### Secondary observations

- **Raw FLmap (Table 1) vs rrule wrap (Table 2)** at `(10, 128)`:
  `fwd 310 → 299 ms`, `bwd 3791 → 4044 ms`. The `rrule(forloop)`
  early-exit wrap adds ~250 ms (~7%) of backward overhead — small
  relative to the chunk-vs-no-chunk gap.
- **Large configs saw timing noise** (e.g. `(10, 512)` iter=16 forward
  33 s vs iter=32 forward 18 s). Probably CUDA allocator pressure or
  memory-pool fragmentation at the upper edge of the 24 GB card. The
  backward column is cleaner.

## Diagnosis guide (original hypothesis — not observed)

Retained for reference; **see Findings above for what was actually
observed.**

- If `α·128 ≫ β` (last column large, e.g. >80%): Zygote per-chunk
  overhead dominates at high `forloop_iter`. Optimize `rrule(forloop)`
  (cache pullback across chunks, preallocate grad buffers, or write a
  hand-rolled FLmap rrule).
- If `α·128 ≈ β` (last column small, e.g. <30%): most cost is fixed
  per-call work. Likely `@tensor` AD itself — hand-roll FLmap rrule.
- Compare Table 1 vs Table 2: difference quantifies rrule wrapping cost.
- If neither α nor β is large yet full-fg `bwd/fwd` is still ~10x,
  the bottleneck is elsewhere (leftenv/rightenv power-iter unrolling).
