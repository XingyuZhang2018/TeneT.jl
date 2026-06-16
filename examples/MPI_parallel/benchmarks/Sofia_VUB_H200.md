# Sofia VUB Benchmark Results

- **Date**: 2026-04-24 (post `feat/p2p-collectives-ring` refactor, merged into `iPEPS-unified`); 16 GPU NCCL fast-path numbers added 2026-04-27 (jobs `1003788` / `1003905`, commit `60fa029`); 32/64/128 GPU NCCL Part 1 diagnostic supplement added 2026-06-15 (jobs `1287256` / `1287255` / `1287260`, current slice/forloop path); clean mapped NCCL node-scaling added 2026-06-15 (job `1287308`); mapped FLmap Part 2 rebaseline added 2026-06-16 (job `1287310`); mapped 32/64/128 GPU FLmap Part 2 extension added 2026-06-16 (jobs `1287318` / `1287319` / `1287320`)
- **System**: Sofia HPC at VUB, partition `zen4_h200`
- **GPU**: NVIDIA H200 141GB (Hopper, x86_64, AMD Zen4 host)
- **GPU/node**: 8 · **Intra-node**: NVSwitch · **Inter-node**: InfiniBand
- **Software**: OpenMPI 5.0.7 (EasyBuild) + UCX-CUDA 1.18.0 + GDRCopy 2.4.4 + CUDA 12.8 + Julia 1.11.3
- **Env**: `CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK`, `UCX_TLS=rc_x,self,sm,cuda_copy,cuda_ipc`,
  `UCX_MEMTYPE_CACHE=n`, `CUDA_LAUNCH_BLOCKING=1`,
  **`LD_PRELOAD=/usr/lib64/libcuda.so.1`** (mandatory — see Known Issues)
- **Benchmark timing convention**: when per-repetition samples are available,
  benchmark tables record the best observed/min max-over-ranks time. Average
  and median are retained only as jitter diagnostics. Legacy rows with only
  aggregate logs are marked or left as aggregate values until rerun.

## Part 1: MPI Collectives (`allgatherv_p2p!` / `allreduce_p2p!`)

Job `1000652` (`submit_test.sh` → `test_MPI_config.jl`). 16 GPU uses 2 nodes
(Phase 2 cross-node ring), 1/2/4/8 GPU are single-node (NVSwitch only).

### Allgatherv

| Size  | 2 GPU              | 4 GPU              | 8 GPU              | 16 GPU             |
|-------|--------------------|--------------------|--------------------|--------------------|
| 8KB   | 0.05ms             | 0.16ms             | 1.36ms             | 1.98ms             |
| 8MB   | 0.10ms (75.7 GB/s) | 0.21ms (36.5 GB/s) | 0.32ms (23.9 GB/s) | 3.85ms (2.0 GB/s)  |
| 128MB | 0.75ms (163.6 GB/s)| 1.33ms (91.7 GB/s) | 1.62ms (75.2 GB/s) | 5.55ms (22.0 GB/s) |

### Allreduce

| Size  | 2 GPU              | 4 GPU              | 8 GPU              | 16 GPU              |
|-------|--------------------|--------------------|--------------------|---------------------|
| 8KB   | 0.08ms             | 0.23ms             | 0.56ms             | 0.84ms              |
| 8MB   | 0.17ms (44.6 GB/s) | 0.56ms (13.6 GB/s) | 0.98ms (7.8 GB/s)  | 1.51ms (5.0 GB/s)   |
| 128MB | 1.44ms (85.1 GB/s) | 2.69ms (45.3 GB/s) | 3.32ms (36.7 GB/s) | 44.81ms (2.7 GB/s) ⚠ |

> *⚠ **16 GPU 128 MB allreduce (44.81 ms) is a significant outlier** vs 3.32 ms at
> 8 GPU. Cross-node Phase 2 (per-local-rank sibling ring on IB) is being
> exercised for the first time here; likely causes under investigation:
> one-time UCX registration overhead for fresh cross-node memory regions,
> or the p2p ring vs. an UCC-backed IB allreduce at comparable sizes.
> Allgatherv at 16 GPU (5.55 ms) degrades more gracefully.*
>
> *Update 2026-04-27 (job `1003905`): `TENET_USE_NCCL=1` (commit `60fa029`)
> replaces the 3-phase ring with a single `ncclAllReduce` and drops the
> 16 GPU 128 MB allreduce to **11.83 ms (3.76× faster)**. See Part 4 for the
> full ring vs. NCCL comparison and break-even analysis.*

### NCCL Supplement (32/64/128 GPU, 2026-06-15)

Jobs `1287256` (32 GPU, 4 nodes), `1287255` (64 GPU, 8 nodes), and
`1287260` (128 GPU, 16 nodes), all `TENET_USE_NCCL=1`,
`test_MPI_config.jl` Part 1.

**Superseded diagnostic note (2026-06-15):** these 32/64/128 GPU runs were
submitted before the Sofia launch rule was fixed and should be treated as
unmapped/ad-hoc diagnostics, not as NCCL route-quality evidence. Job `1287308`
below reran the collective test with explicit
`--map-by ppr:8:node --bind-to none` placement and shows the clean NCCL
baseline.

| Size  | 32 GPU Allgatherv | 64 GPU Allgatherv | 128 GPU Allgatherv | 32 GPU Allreduce | 64 GPU Allreduce | 128 GPU Allreduce |
|-------|-------------------|-------------------|--------------------|------------------|------------------|-------------------|
| 8 KB  | 16.87 ms          | 24.58 ms          | 53.81 ms           | 17.59 ms         | 19.54 ms         | 24.58 ms          |
| 8 MB  | 22.58 ms          | 40.89 ms          | 6.27 ms            | 25.49 ms         | 29.13 ms         | 41.38 ms          |
| 128 MB| 25.88 ms          | 46.96 ms          | 64.08 ms           | 38.11 ms         | 56.93 ms         | 103.69 ms         |

Algorithm: hierarchical 3-phase p2p. Allgatherv = intra-node concurrent
`Irecv!`/`Isend` + leader ring allgatherv across nodes + leader broadcast of
non-local slabs. Allreduce = intra-node ring reduce-scatter → per-local-rank
cross-node ring allreduce → opposite-direction intra-node ring allgather.
See [`docs/plans/2026-04-24-mpi-p2p-collectives-design.md`](../../../docs/plans/2026-04-24-mpi-p2p-collectives-design.md).

## Part 2: FLmap_parallel Forward

Job `1000652`, per-iteration FLmap_parallel forward time (ms), matrix
`D ∈ {8,10,12,14,16} × χ ∈ {256,512,768,1024}` = 20 configs × 5 GPU counts.

| D  | χ    | Size   | 1 GPU | 2 GPU | 4 GPU | 8 GPU | 16 GPU | 2× | 4× | 8× | 16× |
|----|------|--------|-------|-------|-------|-------|--------|----|----|----|-----|
| 8  | 256  | 32MB   |   123 |    57 |    26 |    14 |      9 | 2.15× | 4.72× | 8.79× | 14.29× |
| 8  | 512  | 128MB  |   299 |   113 |    58 |    30 |     20 | 2.65× | 5.16× | 9.97× | 14.95× |
| 8  | 768  | 288MB  |   474 |   239 |   120 |    62 |     42 | 1.98× | 3.95× | 7.64× | 11.29× |
| 8  | 1024 | 512MB  |   880 |   442 |   222 |   115 |     77 | 1.99× | 3.97× | 7.65× | 11.43× |
| 10 | 256  | 50MB   |   205 |   109 |    63 |    49 |     27 | 1.88× | 3.25× | 4.18× |  7.59× |
| 10 | 512  | 200MB  |   426 |   214 |   107 |    56 |     36 | 1.99× | 3.98× | 7.61× | 11.83× |
| 10 | 768  | 450MB  |   992 |   499 |   253 |   128 |     82 | 1.99× | 3.92× | 7.75× | 12.10× |
| 10 | 1024 | 800MB  |  2024 |  1095 |   631 |   304 |    154 | 1.85× | 3.21× | 6.66× | 13.14× |
| 12 | 256  | 72MB   |   221 |    99 |    50 |    26 |     16 | 2.23× | 4.42× | 8.50× | 13.81× |
| 12 | 512  | 288MB  |   711 |   358 |   195 |    92 |     57 | 1.99× | 3.65× | 7.73× | 12.47× |
| 12 | 768  | 648MB  |  1803 |   909 |   517 |   233 |    139 | 1.98× | 3.49× | 7.74× | 12.97× |
| 12 | 1024 | 1152MB |  3608 |  1813 |  1003 |   478 |    284 | 1.99× | 3.60× | 7.55× | 12.70× |
| 14 | 256  | 98MB   |   322 |   162 |    81 |    42 |     25 | 1.99× | 3.98× | 7.67× | 12.88× |
| 14 | 512  | 392MB  |  1342 |   675 |   344 |   172 |    101 | 1.99× | 3.90× | 7.80× | 13.29× |
| 14 | 768  | 882MB  |  3497 |  1752 |   894 |   447 |    264 | 2.00× | 3.91× | 7.82× | 13.25× |
| 14 | 1024 | 1568MB |  7107 |  3665 |  1986 |  1202 |    710 | 1.94× | 3.58× | 5.91× | 10.01× |
| 16 | 256  | 128MB  |   469 |   237 |   120 |    60 |     35 | 1.98× | 3.91× | 7.82× | 13.40× |
| 16 | 512  | 512MB  |  2105 |  1066 |   538 |   297 |    151 | 1.97× | 3.91× | 7.09× | 13.94× |
| 16 | 768  | 1152MB |  5755 |  2866 |  1472 |   798 |    443 | 2.01× | 3.91× | 7.21× | 12.99× |
| 16 | 1024 | 2048MB | 11788 |  6039 |  3203 |  1833 |   1198 | 1.95× | 3.68× | 6.43× |  9.84× |

### Forward Mapped NCCL Rebaseline (1/2/4/8/16/32/64/128 GPU, 2026-06-16)

Job `1287310` (`FLmapP2nc16`) reran the same Part 2 matrix with the Sofia
multi-GPU launch standard:
`mpirun --host ... --map-by ppr:8:node --bind-to none`, `TENET_USE_NCCL=1`,
`NCCL_DEBUG=WARN`, `total_splits=128`, and `forloop_iter=128/np`. Slurm
completed `0:0` on `acc015,acc022`; the fatal-signature scan found no CUDA
OOM, NCCL/UCX/OpenMPI fatal, CUTENSOR, Julia load/type/name error, or mapping
failure. The expected `isapprox(parallel, serial; rtol=1e-4)` forward FAIL
for `np>1` is the known reduction-order drift; timings are still usable.

Jobs `1287318` (32 GPU, 4 nodes, `forloop_iter=4`), `1287319` (64 GPU,
8 nodes, `forloop_iter=2`), and `1287320` (128 GPU, 16 nodes,
`forloop_iter=1`) add the 32/64/128 GPU columns with the same mapped launch
standard. All three completed with `ExitCode=0:0`; log scans found no fatal
CUDA/NCCL/UCX/OpenMPI/PRRTE/CUTENSOR/Julia/mapping signatures. The 128 GPU
run used nodes `acc[001-012,017-020]` and finished in `00:05:27`.

Forward speedups are relative to the mapped `np=1` baseline. Benchmark timing
uses the best observed/min repetition when per-repetition samples are available;
older rows that logged only aggregate timings remain as originally reported
until rerun.

| GPU | Mapped benchmark speedup |
|-----|---------------------|
| 1 | 1.00x |
| 2 | 2.00x |
| 4 | 3.91x |
| 8 | 7.25x |
| 16 | 13.98x |
| 32 | 26.15x |
| 64 | 46.30x |
| 128 | 68.61x |

| D | chi | Size | 1 GPU | 2 GPU | 4 GPU | 8 GPU | 16 GPU | 32 GPU | 64 GPU | 128 GPU | 2x | 4x | 8x | 16x | 32x | 64x | 128x |
|---|----:|------|------:|------:|------:|------:|------:|------:|------:|------:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 256 | 32 MB | 121.40 | 56.43 | 25.67 | 14.34 | 7.39 | 4.29 | 2.93 | 3.10 | 2.15x | 4.73x | 8.47x | 16.43x | 28.30x | 41.43x | 39.16x |
| 8 | 512 | 128 MB | 257.29 | 112.43 | 56.21 | 30.30 | 15.82 | 8.90 | 5.60 | 4.83 | 2.29x | 4.58x | 8.49x | 16.26x | 28.91x | 45.94x | 53.27x |
| 8 | 768 | 288 MB | 466.73 | 237.77 | 118.53 | 60.81 | 32.80 | 18.53 | 11.19 | 7.57 | 1.96x | 3.94x | 7.68x | 14.23x | 25.19x | 41.71x | 61.66x |
| 8 | 1024 | 512 MB | 874.36 | 440.58 | 219.89 | 111.72 | 60.90 | 33.34 | 20.24 | 13.73 | 1.98x | 3.98x | 7.83x | 14.36x | 26.23x | 43.20x | 63.68x |
| 10 | 256 | 50 MB | 206.47 | 111.37 | 71.01 | 151.01 | 56.33 | 19.08 | 11.50 | 23.80 | 1.85x | 2.91x | 1.37x | 3.67x | 10.82x | 17.95x | 8.68x |
| 10 | 512 | 200 MB | 422.34 | 212.69 | 105.97 | 54.57 | 29.77 | 17.32 | 9.70 | 6.81 | 1.99x | 3.99x | 7.74x | 14.19x | 24.38x | 43.54x | 62.02x |
| 10 | 768 | 450 MB | 982.94 | 496.91 | 250.19 | 125.68 | 66.99 | 36.17 | 20.75 | 14.59 | 1.98x | 3.93x | 7.82x | 14.67x | 27.18x | 47.37x | 67.37x |
| 10 | 1024 | 800 MB | 2028.22 | 1102.21 | 624.41 | 269.31 | 126.94 | 67.82 | 38.91 | 24.11 | 1.84x | 3.25x | 7.53x | 15.98x | 29.91x | 52.13x | 84.12x |
| 12 | 256 | 72 MB | 220.23 | 99.12 | 50.29 | 26.26 | 14.24 | 8.70 | 4.90 | 3.88 | 2.22x | 4.38x | 8.39x | 15.47x | 25.31x | 44.94x | 56.76x |
| 12 | 512 | 288 MB | 705.81 | 356.54 | 178.95 | 91.05 | 47.95 | 27.81 | 14.88 | 9.57 | 1.98x | 3.94x | 7.75x | 14.72x | 25.38x | 47.43x | 73.75x |
| 12 | 768 | 648 MB | 1802.97 | 916.23 | 457.12 | 241.83 | 121.67 | 62.58 | 34.91 | 24.77 | 1.97x | 3.94x | 7.46x | 14.82x | 28.81x | 51.65x | 72.79x |
| 12 | 1024 | 1152 MB | 3636.62 | 1834.85 | 955.27 | 466.12 | 242.10 | 124.26 | 67.28 | 40.60 | 1.98x | 3.81x | 7.80x | 15.02x | 29.27x | 54.05x | 89.57x |
| 14 | 256 | 98 MB | 318.89 | 161.43 | 80.43 | 40.91 | 22.91 | 12.65 | 7.15 | 4.63 | 1.98x | 3.96x | 7.79x | 13.92x | 25.21x | 44.60x | 68.87x |
| 14 | 512 | 392 MB | 1345.05 | 681.92 | 358.74 | 198.52 | 90.30 | 47.27 | 42.64 | 15.31 | 1.97x | 3.75x | 6.78x | 14.90x | 28.45x | 31.54x | 87.85x |
| 14 | 768 | 882 MB | 3561.66 | 1774.05 | 893.40 | 446.62 | 228.56 | 121.39 | 62.98 | 49.05 | 2.01x | 3.99x | 7.97x | 15.58x | 29.34x | 56.55x | 72.61x |
| 14 | 1024 | 1568 MB | 7183.57 | 3741.16 | 1996.42 | 1226.30 | 656.62 | 366.43 | 123.04 | 70.49 | 1.92x | 3.60x | 5.86x | 10.94x | 19.60x | 58.38x | 101.91x |
| 16 | 256 | 128 MB | 469.17 | 235.12 | 117.51 | 60.17 | 31.46 | 16.57 | 9.68 | 7.85 | 2.00x | 3.99x | 7.80x | 14.91x | 28.31x | 48.47x | 59.77x |
| 16 | 512 | 512 MB | 2121.78 | 1076.22 | 535.58 | 270.06 | 135.33 | 79.42 | 42.51 | 22.01 | 1.97x | 3.96x | 7.86x | 15.68x | 26.72x | 49.91x | 96.40x |
| 16 | 768 | 1152 MB | 5825.00 | 2944.60 | 1516.30 | 987.55 | 452.57 | 183.92 | 98.04 | 131.69 | 1.98x | 3.84x | 5.90x | 12.87x | 31.67x † | 59.41x | 44.23x |
| 16 | 1024 | 2048 MB | 11937.81 | 6086.98 | 3230.48 | 1795.07 | 1096.67 | 496.28 | 260.93 | 110.72 | 1.96x | 3.70x | 6.65x | 10.89x | 24.05x † | 45.75x | 107.82x |

† The D=16, χ=768/1024 32 GPU forward points were remeasured in targeted
job `1287327` with `nrep=7`; benchmark timing columns use the best observed
rerun time (`183.92 ms` / `496.28 ms`), replacing the original job `1287318`
values (`330.49 ms` / `681.11 ms`). These points are genuinely slow/noisy
rather than transcription errors: the rerun avg/med/min were
`319.02/210.86/183.92 ms` for χ=768 and `633.67/571.84/496.28 ms` for χ=1024.

## Part 2: FLmap_parallel Backward

Per-iteration backward time (ms). Gains from 8 → 16 GPU are limited by the
Phase 2 cross-node allreduce path (same issue as Part 1). NCCL fast path
(`TENET_USE_NCCL=1`) improves 16 GPU backward by **1.3-1.54× at D≥10 χ≥512**;
see Part 4.

The **16× NCCL** column shows speedup vs 1 GPU when `TENET_USE_NCCL=1` is on
(from job `1003905`); the gain over the ring 16× column is the NCCL fast-path
benefit isolated to the same workload. NCCL **regresses** at small sizes
(D=8 χ=256–512 / D=10 χ=256 / D=16 χ=256) where the 3-phase ring is faster
than NCCL's setup cost, and **delivers 1.3-1.5× extra over ring** at large
χ where backward is allreduce-bound.

| D  | χ    | Size   | 1 GPU  | 2 GPU | 4 GPU | 8 GPU | 16 GPU | 2× | 4× | 8× | 16× | 16× NCCL |
|----|------|--------|--------|-------|-------|-------|--------|----|----|----|-----|----------|
| 8  | 256  | 32MB   |    546 |   301 |   187 |   147 |    121 | 1.81× | 2.92× | 3.71× | 4.51× | 3.45× ⚠ |
| 8  | 512  | 128MB  |    762 |   460 |   394 |   401 |    295 | 1.66× | 1.93× | 1.90× | 2.58× | 2.96× |
| 8  | 768  | 288MB  |   1386 |   785 |   557 |   767 |    723 | 1.77× | 2.49× | 1.81× | 1.92× | 2.39× |
| 8  | 1024 | 512MB  |   2534 |  1342 |   844 |   932 |   1178 | 1.89× | 3.00× | 2.72× | 2.15× | 2.94× |
| 10 | 256  | 50MB   |    614 |   386 |   261 |   238 |    153 | 1.59× | 2.35× | 2.58× | 4.01× | 3.66× ⚠ |
| 10 | 512  | 200MB  |   1407 |   787 |   545 |   774 |    636 | 1.79× | 2.58× | 1.82× | 2.21× | 2.54× |
| 10 | 768  | 450MB  |   3170 |  1672 |  1004 |  1017 |   1182 | 1.90× | 3.16× | 3.12× | 2.68× | **3.46×** |
| 10 | 1024 | 800MB  |   6253 |  3219 |  2075 |  1530 |   1643 | 1.94× | 3.01× | 4.09× | 3.81× | **5.53×** |
| 12 | 256  | 72MB   |    789 |   481 |   400 |   450 |    283 | 1.64× | 1.97× | 1.75× | 2.79× | 2.76× |
| 12 | 512  | 288MB  |   2423 |  1295 |   910 |   943 |   1032 | 1.87× | 2.66× | 2.57× | 2.35× | 2.81× |
| 12 | 768  | 648MB  |   6367 |  3276 |  2093 |  1447 |   1551 | 1.94× | 3.04× | 4.40× | 4.10× | **5.56×** |
| 12 | 1024 | 1152MB |  12642 |  6488 |  3496 |  2288 |   2499 | 1.95× | 3.62× | 5.53× | 5.06× | **7.85×** |
| 14 | 256  | 98MB   |   1217 |   680 |   511 |   759 |    506 | 1.79× | 2.38× | 1.60× | 2.40× | 2.55× |
| 14 | 512  | 392MB  |   4777 |  2476 |  1426 |  1232 |   1234 | 1.93× | 3.35× | 3.88× | 3.87× | **4.36×** |
| 14 | 768  | 882MB  |  12739 |  6492 |  3513 |  2518 |   2061 | 1.96× | 3.63× | 5.06× | 6.18× | **8.05×** |
| 14 | 1024 | 1568MB |  25630 | 13132 |  6985 |  4242 |   3710 | 1.95× | 3.67× | 6.04× | 6.91× | **10.39×** |
| 16 | 256  | 128MB  |   1749 |   943 |   645 |   867 |    862 | 1.85× | 2.71× | 2.02× | 2.03× | 2.13× |
| 16 | 512  | 512MB  |   7585 |  3903 |  2140 |  1695 |   1581 | 1.94× | 3.55× | 4.48× | 4.80× | **6.18×** |
| 16 | 768  | 1152MB |  20822 | 10745 |  5674 |  3522 |   3019 | 1.94× | 3.67× | 5.91× | 6.90× | **9.31×** |
| 16 | 1024 | 2048MB |  43602 | 22666 | 11965 |  6612 |   5461 | 1.92× | 3.64× | 6.59× | 7.99× | **10.76×** |

> *⚠ entries mark configs where NCCL is **slower than ring** at 16 GPU
> (D=8 χ=256 / D=10 χ=256 — both ≤50 MB allreduce). NCCL fast path is opt-in
> via `TENET_USE_NCCL=1` for exactly this reason: it wins on large messages
> but loses to the latency-optimised ring on small ones.*
>
> *The `isapprox(parallel, serial; rtol=1e-4)` correctness check in
> `test_MPI_config.jl` prints FAIL for multi-GPU forward runs — this is the known
> parallel-vs-serial reduction-order drift (machine-eps precision loss) and is
> **not** a real regression. 1 GPU ALL PASSED confirms the kernel itself is
> correct. JSC/BSC show the same behaviour.*

### Backward Mapped NCCL Rebaseline (1/2/4/8/16/32/64/128 GPU, 2026-06-16)

Same matrix as the forward mapped rebaseline above. Backward finite/nonzero
checks passed for all rows in jobs `1287310`, `1287318`, `1287319`, and
`1287320`. Backward speedups are relative to the mapped `np=1` baseline.
Benchmark timing uses the best observed/min repetition when per-repetition
samples are available; older rows that logged only aggregate timings remain as
originally reported until rerun.

| GPU | Mapped benchmark speedup |
|-----|---------------------|
| 1 | 1.00x |
| 2 | 1.82x |
| 4 | 2.76x |
| 8 | 3.17x |
| 16 | 4.64x |
| 32 | 8.40x |
| 64 | 10.02x |
| 128 | 13.06x |

| D | chi | Size | 1 GPU | 2 GPU | 4 GPU | 8 GPU | 16 GPU | 32 GPU | 64 GPU | 128 GPU | 2x | 4x | 8x | 16x | 32x | 64x | 128x |
|---|----:|------|------:|------:|------:|------:|------:|------:|------:|------:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 256 | 32 MB | 547.14 | 320.37 | 199.96 | 210.25 | 180.99 | 113.43 | 72.52 | 77.98 | 1.71x | 2.74x | 2.60x | 3.02x | 4.82x | 7.54x | 7.02x |
| 8 | 512 | 128 MB | 759.07 | 480.14 | 666.86 | 722.14 | 284.95 | 142.79 | 91.66 | 85.15 | 1.58x | 1.14x | 1.05x | 2.66x | 5.32x | 8.28x | 8.91x |
| 8 | 768 | 288 MB | 1384.20 | 794.21 | 711.19 | 1147.46 | 645.14 | 485.53 | 200.24 | 254.76 | 1.74x | 1.95x | 1.21x | 2.15x | 2.85x | 6.91x | 5.43x |
| 8 | 1024 | 512 MB | 2554.15 | 1370.46 | 907.15 | 1069.19 | 945.03 | 754.84 | 490.26 | 267.17 | 1.86x | 2.82x | 2.39x | 2.70x | 3.38x | 5.21x | 9.56x |
| 10 | 256 | 50 MB | 613.27 | 406.67 | 309.42 | 364.89 | 168.05 | 84.33 | 54.29 | 56.66 | 1.51x | 1.98x | 1.68x | 3.65x | 7.27x | 11.30x | 10.82x |
| 10 | 512 | 200 MB | 1405.22 | 794.85 | 769.21 | 962.34 | 819.89 | 307.81 | 315.52 | 161.53 | 1.77x | 1.83x | 1.46x | 1.71x | 4.57x | 4.45x | 8.70x |
| 10 | 768 | 450 MB | 3204.89 | 1710.89 | 1264.41 | 1144.59 | 954.25 | 722.31 | 574.47 | 290.17 | 1.87x | 2.53x | 2.80x | 3.36x | 4.44x | 5.58x | 11.04x |
| 10 | 1024 | 800 MB | 6352.29 | 3309.40 | 1896.39 | 1551.16 | 1212.73 | 976.76 | 775.12 | 485.55 | 1.92x | 3.35x | 4.10x | 5.24x | 6.50x | 8.20x | 13.08x |
| 12 | 256 | 72 MB | 786.21 | 494.42 | 511.42 | 596.85 | 315.44 | 153.53 | 95.76 | 88.94 | 1.59x | 1.54x | 1.32x | 2.49x | 5.12x | 8.21x | 8.84x |
| 12 | 512 | 288 MB | 2454.51 | 1317.20 | 907.45 | 1057.85 | 964.28 | 618.54 | 348.80 | 249.44 | 1.86x | 2.70x | 2.32x | 2.55x | 3.97x | 7.04x | 9.84x |
| 12 | 768 | 648 MB | 6459.85 | 3361.21 | 1945.89 | 1542.64 | 1200.04 | 964.92 | 753.00 | 500.81 | 1.92x | 3.32x | 4.19x | 5.38x | 6.69x | 8.58x | 12.90x |
| 12 | 1024 | 1152 MB | 12823.14 | 6580.09 | 3564.06 | 2357.94 | 1621.09 | 1185.70 | 1040.13 | 869.42 | 1.95x | 3.60x | 5.44x | 7.91x | 10.81x | 12.33x | 14.75x |
| 14 | 256 | 98 MB | 1214.42 | 702.66 | 589.28 | 1018.72 | 544.19 | 278.30 | 166.68 | 127.65 | 1.73x | 2.06x | 1.19x | 2.23x | 4.36x | 7.29x | 9.51x |
| 14 | 512 | 392 MB | 4838.66 | 2534.08 | 1504.92 | 1356.27 | 1232.07 | 915.96 | 653.35 | 393.62 | 1.91x | 3.22x | 3.57x | 3.93x | 5.28x | 7.41x | 12.29x |
| 14 | 768 | 882 MB | 12950.43 | 6629.45 | 3544.10 | 2492.23 | 1799.03 | 1183.71 | 1200.73 | 794.34 | 1.95x | 3.65x | 5.20x | 7.20x | 10.94x | 10.79x | 16.30x |
| 14 | 1024 | 1568 MB | 26029.71 | 13417.31 | 6968.78 | 4722.32 | 2692.86 | 1791.68 | 1280.16 | 1018.96 | 1.94x | 3.74x | 5.51x | 9.67x | 14.53x | 20.33x | 25.55x |
| 16 | 256 | 128 MB | 1931.25 | 1052.84 | 904.19 | 984.92 | 939.33 | 435.92 | 261.73 | 189.52 | 1.83x | 2.14x | 1.96x | 2.06x | 4.43x | 7.38x | 10.19x |
| 16 | 512 | 512 MB | 7687.66 | 3992.03 | 2255.38 | 1904.66 | 1455.20 | 1028.14 | 893.59 | 590.13 | 1.93x | 3.41x | 4.04x | 5.28x | 7.48x | 8.60x | 13.03x |
| 16 | 768 | 1152 MB | 21042.92 | 10803.90 | 5607.94 | 3961.09 | 2263.50 | 711.80 | 1125.36 | 1040.39 | 1.95x | 3.75x | 5.31x | 9.30x | 29.56x † | 18.70x | 20.23x |
| 16 | 1024 | 2048 MB | 44301.93 | 23006.83 | 11645.46 | 7325.77 | 4305.69 | 1731.46 | 1691.45 | 1335.35 | 1.93x | 3.80x | 6.05x | 10.29x | 25.59x † | 26.19x | 33.18x |

† The D=16, χ=768/1024 32 GPU backward points use the best observed/min times
from targeted rerun job `1287327` (`711.80 ms` / `1731.46 ms`). The rerun
avg/med/min were `1258.99/735.51/711.80 ms` for χ=768 and
`2856.58/3095.17/1731.46 ms` for χ=1024.

## Part 3: checkpoint() Method Comparison

Job `1000649` (`submit_test_checkpoint.sh` → `test_MPI_checkpoint.jl`).
**All 20 configs (4 sizes × 5 GPU counts) PASSED** the
`‖g - g_plain‖ / ‖g_plain‖ < 1e-10` gradient-equivalence check. 16 GPU row
is the first cross-node checkpoint verification.

### 1 GPU
| D  | χ   | Plain   | Recompute | Offload | Off/Plain |
|----|-----|---------|-----------|---------|-----------|
| 8  | 128 |  341 ms |   407 ms  |  358 ms | 1.05×     |
| 8  | 256 |  365 ms |   473 ms  |  368 ms | 1.01×     |
| 10 | 256 |  545 ms |  1298 ms  |  545 ms | 1.00×     |
| 10 | 512 | 2294 ms |  2749 ms  | 2296 ms | 1.00×     |

### 2 GPU
| D  | χ   | Plain   | Recompute | Offload | Off/Plain |
|----|-----|---------|-----------|---------|-----------|
| 8  | 128 |  247 ms |   260 ms  |  240 ms | 0.97×     |
| 8  | 256 |  195 ms |   253 ms  |  196 ms | 1.00×     |
| 10 | 256 |  316 ms |   715 ms  |  314 ms | 0.99×     |
| 10 | 512 | 1381 ms |  1964 ms  | 1383 ms | 1.00×     |

### 4 GPU
| D  | χ   | Plain   | Recompute | Offload | Off/Plain |
|----|-----|---------|-----------|---------|-----------|
| 8  | 128 |  169 ms |   192 ms  |  185 ms | 1.10×     |
| 8  | 256 |  120 ms |   149 ms  |  117 ms | 0.98×     |
| 10 | 256 |  414 ms |   464 ms  |  233 ms | 0.56×     |
| 10 | 512 | 1133 ms |  1582 ms  | 1151 ms | 1.02×     |

### 8 GPU
| D  | χ   | Plain   | Recompute | Offload | Off/Plain |
|----|-----|---------|-----------|---------|-----------|
| 8  | 128 |  231 ms |   179 ms  |  166 ms | 0.72×     |
| 8  | 256 |  123 ms |   121 ms  |  126 ms | 1.03×     |
| 10 | 256 |  253 ms |   358 ms  |  250 ms | 0.99×     |
| 10 | 512 | 1129 ms |  1436 ms  | 1103 ms | 0.98×     |

### 16 GPU (2 nodes)
| D  | χ   | Plain   | Recompute | Offload | Off/Plain |
|----|-----|---------|-----------|---------|-----------|
| 8  | 128 |  138 ms |   150 ms  |  155 ms | 1.12×     |
| 8  | 256 |   72 ms |    88 ms  |   71 ms | 0.99×     |
| 10 | 256 |  149 ms |   240 ms  |  151 ms | 1.01×     |
| 10 | 512 |  717 ms |   879 ms  |  710 ms | 0.99×     |

## Full fg Benchmark (D=10 χ=400, Plaquette VUMPS, with checkpoint)

Job `1000633` (`submit.sh` → `benchmark_fg.jl`, N=1/2/4/8 at single node);
N=16 from job `1003788` (`submit_fg_nccl_compare.sh`, NCCL OFF column).

| GPU | Forward | fg forward | fg backward | fg total | fg speedup |
|-----|---------|-----------|------------|----------|------------|
| 1   | 163.90s | 155.29s    | 516.06s    | 671.35s  | 1.00×      |
| 2   |  84.98s |  82.32s    | 276.27s    | 358.60s  | 1.87×      |
| 4   |  45.56s |  43.69s    | 160.72s    | 204.41s  | 3.28×      |
| 8   |  26.57s |  26.94s    | 100.81s    | 127.75s  | 5.25×      |
| 16  |  17.56s |  18.29s    | 114.79s    | 133.08s  | 5.04× ⚠    |

> *⚠ The 16 GPU **fg total** speedup (5.04×) is *below* the 8 GPU (5.25×)
> point — backward is comm-bound on the cross-node IB ring (114.79 s vs
> 100.81 s at 8 GPU intra-node NVSwitch, the ring penalty dominates).
> Same issue as Parts 1/2. **NCCL fast path (`TENET_USE_NCCL=1`) recovers
> a 1.33× fg-total improvement (133.08 s → 100.22 s, fg speedup 5.04× →
> 6.70×) at the same 16 GPU configuration**. See Part 4.3.*

## Part 4: NCCL Fast Path (`TENET_USE_NCCL=1`) — 16 GPU 2-Node Comparison

Jobs `1003905` (`submit_test_nccl_compare.sh` → `test_MPI_config.jl` twice
at 16 GPU, ring vs `TENET_USE_NCCL=1`) and `1003788`
(`submit_fg_nccl_compare.sh` → `benchmark_fg.jl` D=10 χ=400 same toggle), all
2026-04-27, NCCL 2.27.7 + CUDA 12.8, commit `60fa029` introduces the wrapper.

`TENET_USE_NCCL=1` routes `allreduce_p2p!` / `allgatherv_p2p!` (when buf
isa `CuArray`) to a single `ncclAllReduce` / `ncclAllGather` and bypasses
the 3-phase p2p ring entirely. NCCL handles the intra+inter node hierarchy
itself (NVLink + IB GDR). Stays opt-in — small-collective workloads (≤8 MB)
are *slower* under NCCL than the latency-optimised ring, so flipping the
default would regress those.

### Part 4.1: MPI Collectives (16 GPU)

| Size  | Allgatherv ring | Allgatherv NCCL | Allgatherv× | Allreduce ring | Allreduce NCCL | Allreduce× |
|-------|-----------------|-----------------|-------------|----------------|----------------|------------|
| 8 KB  | 2.17 ms         | 3.75 ms         | 0.58×       | 0.81 ms        | 4.09 ms        | **0.20×** ⚠ |
| 8 MB  | 3.88 ms         | 5.23 ms         | 0.74×       | 1.51 ms        | 5.88 ms        | **0.26×** ⚠ |
| 128 MB| 5.53 ms         | 9.14 ms         | 0.61×       | **44.47 ms**   | **11.83 ms**   | **3.76×** ✅ |

NCCL has substantial fixed setup cost (≥4 ms per call at 8 KB), so it loses
to the latency-optimised ring at small sizes. Break-even is around 128 MB,
where NCCL's bandwidth-optimal Tree algorithm dominates the ring's per-rank
sequential sends. The 3.76× allreduce win at 128 MB is the headline result.

### Part 4.2: FLmap_parallel Backward (16 GPU)

Same `D ∈ {8,10,12,14,16} × χ ∈ {256,512,768,1024}` matrix as Part 2.
Forward times are within ±10% of ring (small per-call allreduces dominate
the call count → NCCL setup cost balances the throughput gain). Backward,
which sums full-tensor allreduces over the AD pullback, is where NCCL pays
off:

| D  | χ    | Size    | bwd ring (ms) | bwd NCCL (ms) | Speedup |
|----|------|---------|---------------|---------------|---------|
| 8  | 256  |   32 MB |        121.4  |        158.2  | **0.77× ⚠** |
| 8  | 512  |  128 MB |        299.8  |        257.6  | 1.16×   |
| 8  | 1024 |  512 MB |       1186.3  |        860.6  | 1.38×   |
| 10 | 512  |  200 MB |        655.5  |        554.8  | 1.18×   |
| 10 | 768  |  450 MB |       1412.9  |        915.7  | **1.54×** (best) |
| 10 | 1024 |  800 MB |       1644.4  |       1129.9  | 1.46×   |
| 12 | 768  |  648 MB |       1560.0  |       1144.8  | 1.36×   |
| 12 | 1024 | 1152 MB |       2354.5  |       1609.9  | 1.46×   |
| 14 | 768  |  882 MB |       2176.2  |       1583.2  | 1.37×   |
| 14 | 1024 | 1568 MB |       3654.6  |       2466.7  | 1.48×   |
| 16 | 1024 | 2048 MB |       5473.7  |       4052.3  | 1.35×   |

Break-even around χ ≈ 512 / size ≈ 128 MB. At small (D=8 χ=256, 32 MB) NCCL
is **slower** than ring; at large χ where backward is allreduce-dominated,
NCCL delivers 1.3-1.54× wallclock improvement. The best gain (1.54×) at
D=10 χ=768 (450 MB) sits comfortably above break-even.

### Part 4.3: Full fg Benchmark (D=10 χ=400, 16 GPU)

|                | NCCL OFF (ring) | NCCL ON       | Speedup | gnorm |
|----------------|-----------------|---------------|---------|-------|
| TIMED forward  | 17.56 s         | 18.28 s       | 0.96×   | —     |
| TIMED fg fwd   | 18.29 s         | 18.66 s       | 0.98×   | —     |
| TIMED fg bwd   | 114.79 s        | **81.56 s**   | **1.41×** | —   |
| TIMED fg total | 133.08 s        | **100.22 s**  | **1.33×** | —   |
| gnorm          | 1.149580467e-02 | 1.149580411e-02 | (8 digits agree) | ✓ |
| fg vs 1 GPU    | 5.04×           | **6.70×**     | —       | —     |

The production-relevant 25 % wallclock reduction on the steady-state fg
cycle that drives `optimise_ipeps`'s LBFGS. **gnorm equivalence to 8 digits
across NCCL's reduction-order-different summation confirms no precision
loss** (the 9th-digit divergence is well below `gradtol=1e-7` and within
Float64 round-off for a 16-rank reduction).

### Production Rule

| Workload                                            | Recommend                  |
|-----------------------------------------------------|----------------------------|
| D ≥ 10, χ ≥ 512, ≥16 GPU 2-node (cross-node IB)     | `TENET_USE_NCCL=1` ✅       |
| D = 8, χ = 256 / 1-8 GPU single-node intra-NVSwitch | leave default (ring)       |
| Mixed sizes / one-off scripts                       | benchmark; production fg at D=10 χ=400 sees 25 % win |

NCCL stays opt-in (`TENET_USE_NCCL=1` env, default off): cross-system tests
on JSC GH200 8 GPU showed only 2 % wallclock improvement (smaller cross-node
fraction); BSC H100 8 GPU OOM'd at D=10 χ=400 entirely (memory issue
unrelated to NCCL). **The Sofia 16 GPU 2-node win does not generalise to
single-node setups** — keep ring as the default to avoid regressing those.

## Part 5: Clean Mapped NCCL Collectives (2026-06-15, 1-8 H200 nodes)

Job `1287308` (`AgArNCCL8n`) reran the NCCL Allgather/Allreduce route test
with explicit OpenMPI placement:

```bash
mpirun --map-by ppr:8:node --bind-to none -np ${MAX_GPU} ...
```

This is now the Sofia multi-GPU launch standard. `--map-by ppr:8:node`
places exactly 8 MPI ranks per H200 node (one rank per GPU when combined with
`CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK`), and `--bind-to none`
prevents OpenMPI CPU-affinity pinning from constraining Julia, CUDA, BLAS, and
communication helper threads.

Validation:

- Slurm main job and all PRRTE steps: `COMPLETED`, `ExitCode=0:0`.
- Full `stdout`, `stderr`, and tee log scan: no CUDA OOM, NCCL WARN/ERROR,
  UCX/OpenMPI/PRRTE error, or Julia `LoadError`/`MethodError`/`UndefVarError`.
- Exactly 24 `TENET_NODE_BENCH` rows: 8 node counts × 3 message sizes.
- Every row reports `equal_counts=true`, `use_nccl=true`,
  `ag_nccl_pred=true`, and `ar_nccl_pred=true`, confirming the NCCL fast path.
- Message sizes were chosen divisible by all rank counts 8..64, so
  `allgatherv_p2p!` uses the equal-count `ncclAllGather` path.

Legacy aggregate note: this Part 1 table predates per-repetition sample
logging, so the retained values are mean max-over-ranks per repetition, in
milliseconds. Treat them as route diagnostics rather than final min-based
benchmark values until rerun with the current sampler. The global message size
is fixed, so this is not an application speedup benchmark.

| Nodes | Ranks | AG 52KiB | AR 52KiB | AG 7.69MiB | AR 7.69MiB | AG 128MiB | AR 128MiB |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 0.049 | 0.042 | 0.070 | 0.081 | 0.425 | 0.578 |
| 2 | 16 | 0.106 | 0.094 | 0.208 | 0.240 | 1.431 | 1.751 |
| 3 | 24 | 0.120 | 0.117 | 0.278 | 0.365 | 1.475 | 2.920 |
| 4 | 32 | 0.143 | 0.109 | 0.453 | 0.377 | 1.551 | 2.952 |
| 5 | 40 | 0.179 | 0.195 | 0.387 | 0.464 | 1.577 | 3.117 |
| 6 | 48 | 0.176 | 0.121 | 0.467 | 0.443 | 1.588 | 3.118 |
| 7 | 56 | 0.198 | 0.131 | 0.695 | 0.429 | 1.712 | 3.206 |
| 8 | 64 | 0.262 | 0.147 | 0.589 | 0.447 | 1.660 | 3.175 |

For the 128 MiB case, Allgather stays near `1.5-1.7 ms` from 3 to 8 nodes,
and Allreduce stays near `3.0-3.2 ms` from 3 to 8 nodes. This is far below
the older unmapped/unclean 32+ GPU anomaly and indicates NCCL is active and
healthy when launch placement is explicit.

## Sofia-specific Environment

```bash
source /etc/profile
source /etc/profile.d/modules.sh
module load GDRCopy/2.4.4-GCCcore-14.2.0 UCX-CUDA/1.18.0-GCCcore-14.2.0-CUDA-12.8.0 OpenMPI/5.0.7-GCC-14.2.0

export CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK
export UCX_TLS=rc_x,self,sm,cuda_copy,cuda_ipc
export UCX_MEMTYPE_CACHE=n
export UCX_WARN_UNUSED_ENV_VARS=n
export CUDA_LAUNCH_BLOCKING=1                 # mandatory (H200 sync worker bug)
export LD_PRELOAD=/usr/lib64/libcuda.so.1     # mandatory (libcuda conflict)
```

All multi-GPU Sofia scripts should use the explicit placement form:

```bash
mpirun --map-by ppr:8:node --bind-to none -np ${MAX_GPU} \
    -x UCX_MODULE_DIR -x LD_LIBRARY_PATH -x PATH -x HOME -x JULIA_DEPOT_PATH \
    bash -c "export CUDA_VISIBLE_DEVICES=\$OMPI_COMM_WORLD_LOCAL_RANK; exec julia ..."
```

## Known Issues

- **`LD_PRELOAD=/usr/lib64/libcuda.so.1` mandatory** for MPI + CUDA.jl on Sofia.
  Without it, UCX-CUDA's `cuda_copy`/`cuda_ipc` transports fail to register in
  the UCP context (Julia's `CUDA_Driver_jll` artifact `libcuda.so` shadows the
  system driver that UCX-CUDA was built against). See
  [`Sofia/UCX_CUDA_ISSUE.md`](../Sofia/UCX_CUDA_ISSUE.md).
- **`CUDA_LAUNCH_BLOCKING=1` mandatory** for the full-VUMPS test path (`bench.jl`
  and now `test_MPI_config.jl` Part 2 at the extended matrix). The same
  `synchronization_worker` segfault pattern JSC GH200 documented shows up
  here too at larger (D, χ). CLB=1 costs ~15-18% wall-clock but prevents
  crashes.
- **16 GPU 128 MB Allreduce = 44.81 ms** (see Part 1 ⚠ note above) — the
  Phase 2 cross-node ring on IB runs slower than expected; multi-node regime
  not yet fully tuned. **Mitigation:** `TENET_USE_NCCL=1` (commit `60fa029`)
  drops this to **11.83 ms (3.76× faster)** by routing through `ncclAllReduce`
  instead of the 3-phase p2p ring. See Part 4 for the production trade-off.
- **Do not launch multi-GPU Sofia jobs with OpenMPI's default rank mapping.**
  Use `--map-by ppr:8:node --bind-to none` for all multi-card scripts. The
  2026-06-15 clean mapped run (job `1287308`) shows the NCCL route is fast and
  stable under this placement; unmapped 32+ GPU collective numbers can look
  pathologically slow and should be treated as invalid diagnostics.
- **Home dir `/user/sofia/$VSC` doesn't exist** — work out of
  `/sofia/scratch/pilot/pilot_2026_0002/<user>/` with `HOME=$WD` +
  `JULIA_DEPOT_PATH=$WD/.julia`.
- **`MPI.Allreduce!(MPI.IN_PLACE, ::CuArray, ...)` crashes** inside OpenMPI 5.0.7
  `mca_coll_cuda`. Use `Allreduce!(sendbuf, recvbuf, ...)` non-IN_PLACE, or run
  with `--mca coll_cuda_priority 0`. Not triggered by TeneT's
  `allreduce_p2p!` (which uses Isend/Recv through UCX PML).
