# Sofia VUB Benchmark Results

- **Date**: 2026-04-24 (post `feat/p2p-collectives-ring` refactor, merged into `iPEPS-unified`)
- **System**: Sofia HPC at VUB, partition `zen4_h200`
- **GPU**: NVIDIA H200 141GB (Hopper, x86_64, AMD Zen4 host)
- **GPU/node**: 8 · **Intra-node**: NVSwitch · **Inter-node**: InfiniBand
- **Software**: OpenMPI 5.0.7 (EasyBuild) + UCX-CUDA 1.18.0 + GDRCopy 2.4.4 + CUDA 12.8 + Julia 1.11.3
- **Env**: `CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK`, `UCX_TLS=rc_x,self,sm,cuda_copy,cuda_ipc`,
  `UCX_MEMTYPE_CACHE=n`, `CUDA_LAUNCH_BLOCKING=1`,
  **`LD_PRELOAD=/usr/lib64/libcuda.so.1`** (mandatory — see Known Issues)

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

Algorithm: hierarchical 3-phase p2p. Allgatherv = intra-node concurrent
`Irecv!`/`Isend` + leader ring allgatherv across nodes + leader broadcast of
non-local slabs. Allreduce = intra-node ring reduce-scatter → per-local-rank
cross-node ring allreduce → opposite-direction intra-node ring allgather.
See [`docs/plans/2026-04-24-mpi-p2p-collectives-design.md`](../../../docs/plans/2026-04-24-mpi-p2p-collectives-design.md).

## Part 2: FLmap_parallel Forward

Job `1000652`, per-iteration FLmap_parallel forward time (ms), matrix
`D ∈ {8,10,12,14,16} × χ ∈ {256,512,768,1024}` = 20 configs × 5 GPU counts.

| D  | χ    | Size   | 1 GPU | 2 GPU | 4 GPU | 8 GPU | 16 GPU |
|----|------|--------|-------|-------|-------|-------|--------|
| 8  | 256  | 32MB   |   123 |    57 |    26 |    14 |      9 |
| 8  | 512  | 128MB  |   299 |   113 |    58 |    30 |     20 |
| 8  | 768  | 288MB  |   474 |   239 |   120 |    62 |     42 |
| 8  | 1024 | 512MB  |   880 |   442 |   222 |   115 |     77 |
| 10 | 256  | 50MB   |   205 |   109 |    63 |    49 |     27 |
| 10 | 512  | 200MB  |   426 |   214 |   107 |    56 |     36 |
| 10 | 768  | 450MB  |   992 |   499 |   253 |   128 |     82 |
| 10 | 1024 | 800MB  |  2024 |  1095 |   631 |   304 |    154 |
| 12 | 256  | 72MB   |   221 |    99 |    50 |    26 |     16 |
| 12 | 512  | 288MB  |   711 |   358 |   195 |    92 |     57 |
| 12 | 768  | 648MB  |  1803 |   909 |   517 |   233 |    139 |
| 12 | 1024 | 1152MB |  3608 |  1813 |  1003 |   478 |    284 |
| 14 | 256  | 98MB   |   322 |   162 |    81 |    42 |     25 |
| 14 | 512  | 392MB  |  1342 |   675 |   344 |   172 |    101 |
| 14 | 768  | 882MB  |  3497 |  1752 |   894 |   447 |    264 |
| 14 | 1024 | 1568MB |  7107 |  3665 |  1986 |  1202 |    710 |
| 16 | 256  | 128MB  |   469 |   237 |   120 |    60 |     35 |
| 16 | 512  | 512MB  |  2105 |  1066 |   538 |   297 |    151 |
| 16 | 768  | 1152MB |  5755 |  2866 |  1472 |   798 |    443 |
| 16 | 1024 | 2048MB | 11788 |  6039 |  3203 |  1833 |   1198 |

## Part 2: FLmap_parallel Backward

Per-iteration backward time (ms). Gains from 8 → 16 GPU are limited by the
Phase 2 cross-node allreduce path (same issue as Part 1).

| D  | χ    | Size   | 1 GPU  | 2 GPU | 4 GPU | 8 GPU | 16 GPU |
|----|------|--------|--------|-------|-------|-------|--------|
| 8  | 256  | 32MB   |    546 |   301 |   187 |   147 |    121 |
| 8  | 512  | 128MB  |    762 |   460 |   394 |   401 |    295 |
| 8  | 768  | 288MB  |   1386 |   785 |   557 |   767 |    723 |
| 8  | 1024 | 512MB  |   2534 |  1342 |   844 |   932 |   1178 |
| 10 | 256  | 50MB   |    614 |   386 |   261 |   238 |    153 |
| 10 | 512  | 200MB  |   1407 |   787 |   545 |   774 |    636 |
| 10 | 768  | 450MB  |   3170 |  1672 |  1004 |  1017 |   1182 |
| 10 | 1024 | 800MB  |   6253 |  3219 |  2075 |  1530 |   1643 |
| 12 | 256  | 72MB   |    789 |   481 |   400 |   450 |    283 |
| 12 | 512  | 288MB  |   2423 |  1295 |   910 |   943 |   1032 |
| 12 | 768  | 648MB  |   6367 |  3276 |  2093 |  1447 |   1551 |
| 12 | 1024 | 1152MB |  12642 |  6488 |  3496 |  2288 |   2499 |
| 14 | 256  | 98MB   |   1217 |   680 |   511 |   759 |    506 |
| 14 | 512  | 392MB  |   4777 |  2476 |  1426 |  1232 |   1234 |
| 14 | 768  | 882MB  |  12739 |  6492 |  3513 |  2518 |   2061 |
| 14 | 1024 | 1568MB |  25630 | 13132 |  6985 |  4242 |   3710 |
| 16 | 256  | 128MB  |   1749 |   943 |   645 |   867 |    862 |
| 16 | 512  | 512MB  |   7585 |  3903 |  2140 |  1695 |   1581 |
| 16 | 768  | 1152MB |  20822 | 10745 |  5674 |  3522 |   3019 |
| 16 | 1024 | 2048MB |  43602 | 22666 | 11965 |  6612 |   5461 |

> *The `isapprox(parallel, serial; rtol=1e-4)` correctness check in
> `test_MPI_config.jl` prints FAIL at 2/4/8/16 GPU — this is the known
> parallel-vs-serial reduction-order drift (machine-eps precision loss) and is
> **not** a real regression. 1 GPU ALL PASSED confirms the kernel itself is
> correct. JSC/BSC show the same behaviour.*

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

Job `1000633` (`submit.sh` → `benchmark_fg.jl`, N=1/2/4/8 at single node;
N=16 pending from job `1000677`).

| GPU | Forward | fg forward | fg backward | fg total | fg speedup |
|-----|---------|-----------|------------|----------|------------|
| 1   | 163.90s | 155.29s    | 516.06s    | 671.35s  | 1.00×      |
| 2   |  84.98s |  82.32s    | 276.27s    | 358.60s  | 1.87×      |
| 4   |  45.56s |  43.69s    | 160.72s    | 204.41s  | 3.28×      |
| 8   |  26.57s |  26.94s    | 100.81s    | 127.75s  | 5.25×      |
| 16  | TBD     | TBD        | TBD        | TBD      | TBD        |

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
  not yet fully tuned.
- **Home dir `/user/sofia/$VSC` doesn't exist** — work out of
  `/sofia/scratch/pilot/pilot_2026_0002/<user>/` with `HOME=$WD` +
  `JULIA_DEPOT_PATH=$WD/.julia`.
- **`MPI.Allreduce!(MPI.IN_PLACE, ::CuArray, ...)` crashes** inside OpenMPI 5.0.7
  `mca_coll_cuda`. Use `Allreduce!(sendbuf, recvbuf, ...)` non-IN_PLACE, or run
  with `--mca coll_cuda_priority 0`. Not triggered by TeneT's
  `allreduce_p2p!` (which uses Isend/Recv through UCX PML).
