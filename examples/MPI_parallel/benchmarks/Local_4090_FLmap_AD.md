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

## Diagnosis guide

- If `α·128 ≫ β` (last column large, e.g. >80%): Zygote per-chunk
  overhead dominates at high `forloop_iter`. Optimize `rrule(forloop)`
  (cache pullback across chunks, preallocate grad buffers, or write a
  hand-rolled FLmap rrule).
- If `α·128 ≈ β` (last column small, e.g. <30%): most cost is fixed
  per-call work. Likely `@tensor` AD itself — hand-roll FLmap rrule.
- Compare Table 1 vs Table 2: difference quantifies rrule wrapping cost.
- If neither α nor β is large yet full-fg `bwd/fwd` is still ~10x,
  the bottleneck is elsewhere (leftenv/rightenv power-iter unrolling).
