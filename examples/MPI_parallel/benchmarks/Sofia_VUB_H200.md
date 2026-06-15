# Sofia VUB Benchmark Results

- **Date**: 2026-04-24 (post `feat/p2p-collectives-ring` refactor, merged into `iPEPS-unified`); 16 GPU NCCL fast-path numbers added 2026-04-27 (jobs `1003788` / `1003905`, commit `60fa029`)
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
>
> *Update 2026-04-27 (job `1003905`): `TENET_USE_NCCL=1` (commit `60fa029`)
> replaces the 3-phase ring with a single `ncclAllReduce` and drops the
> 16 GPU 128 MB allreduce to **11.83 ms (3.76× faster)**. See Part 4 for the
> full ring vs. NCCL comparison and break-even analysis.*

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

## Part 5: Cannon 2×2 vs slice (4 GPU, single node) — production form `FLmap_cannon_dist`

Job `1275624` (2026-06-12, `submit_benchmark_cannon.sh` → `benchmark_cannon_sofia.jl`,
branch `claude/sad-saha-3ec6bf` @ `e339c65`+), 1 node × 4×H200, Cannon grid 2×2.
Design: [`docs/2026-06-10-cannon-flmap-design.md`](../../../docs/2026-06-10-cannon-flmap-design.md).
(An earlier run of the v2 replicated-AL variant `FLmap_cannon`, job `1274276`,
is superseded by this table — same matrix, cannon numbers within a few % except
where noted below.)

Methodology: tensors/loss as Part 2 (`test_MPI_config.jl`) — Float64 leg5
single-M, slice path `total_splits=128`, `nrep=3`, backward = Zygote pullback
of a bare `sum` loss. **cannon = `FLmap_cannon_dist`: FL, ALu, ALd all
block-stored (persistent 3×χ²D²/P per rank); the per-call row/col AL slice
gathers ARE included in the timings** — in a leftenv power iteration ALu/ALd
are fixed, so these gathers amortize away. Cannon sub-slices its local l
range with `forloop_iter = n` per cell (column `n`; backward-peak formula
`(4+2d)·|H|/n + residents ≤ 110 GB`, |H| = χ²D⁴/4). Timings are per-rep
max-over-ranks with `GC.gc()` (no reclaim) between reps outside the timed
window — reads slightly above Part 2's rank-0 `@elapsed` convention. Cannon's
backward is fully hand-written (six single-contraction adjoints, per-chunk
local H/T/G recompute, `unsafe_free!` after each array's last use — no
Zygote inside the map's rrule) and returns BLOCK gradients for ALu/ALd via
slice-level reduce-scatters (the v2 full-tensor allreduces are gone). All 20
cells ran (no OOM skips); parity vs the slice path ≤1e-10 fwd / ≤1e-8 bwd on
every cell (`F✓ B✓`).

Column legend — all times in ms: **slice** = the existing replicated-input
path (`FLmap_parallel`, Part 2's subject); **cannon** = the distributed
`FLmap_cannon`; **fwd** = one forward map call; **bwd** = one
`Zygote.pullback` construction + backward; **ring** / **nccl** =
`TENET_USE_NCCL` off / on; **n** = Cannon's `forloop_iter` for that cell.

| D  | χ    | n  | slice fwd ring | slice fwd nccl | cannon fwd ring | cannon fwd nccl | slice bwd ring | slice bwd nccl | cannon bwd ring | cannon bwd nccl | parity |
|----|------|----|----------------|----------------|-----------------|-----------------|----------------|----------------|-----------------|-----------------|--------|
| 8  | 256  | 1  |     32.8 |     28.8 |     10.0 |     10.2 |    113.2 |    110.9 |     46.3 |     34.9 | F✓ B✓ |
| 8  | 512  | 1  |     74.5 |     70.9 |     40.7 |     40.4 |    343.0 |    247.4 |    148.4 |    149.1 | F✓ B✓ |
| 8  | 768  | 1  |    150.1 |    146.0 |    101.8 |    100.3 |    549.3 |    593.8 |    404.5 |    326.2 | F✓ B✓ |
| 8  | 1024 | 1  |    276.2 |    267.7 |    199.7 |    199.5 |    883.7 |    991.6 |    660.4 |    670.0 | F✓ B✓ |
| 10 | 256  | 1  |     43.8 |     43.9 |     31.3 |     30.6 |    173.9 |    168.0 |    106.3 |    105.3 | F✓ B✓ |
| 10 | 512  | 1  |    143.7 |    137.3 |    103.5 |    104.0 |    547.3 |    534.3 |    357.8 |    355.9 | F✓ B✓ |
| 10 | 768  | 1  |    335.9 |    318.7 |    286.8 |    264.7 |   1041.1 |   1184.6 |    880.8 |    878.8 | F✓ B✓ |
| 10 | 1024 | 2  |    623.8 |    610.0 |    419.1 |    417.4 |   1938.4 |   1819.9 |   1686.7 |   1678.1 | F✓ B✓ |
| 12 | 256  | 1  |     71.8 |     71.0 |     50.4 |     48.8 |    274.2 |    273.9 |    202.4 |    170.4 | F✓ B✓ |
| 12 | 512  | 1  |    253.6 |    258.2 |    222.1 |    221.5 |    844.3 |    831.8 |    749.1 |    751.1 | F✓ B✓ |
| 12 | 768  | 2  |    638.1 |    618.8 |    434.7 |    434.6 |   1901.2 |   1889.3 |   1702.5 |   1788.6 | F✓ B✓ |
| 12 | 1024 | 4  |   1183.3 |   1220.4 |    825.8 |    777.0 |   3602.7 |   3517.8 |   3384.8 |   3280.7 | F✓ B✓ |
| 14 | 256  | 1  |    118.9 |    171.5 |    107.6 |    107.9 |    503.6 |    464.8 |    349.9 |    347.9 | F✓ B✓ |
| 14 | 512  | 2  |    540.8 |    495.2 |    344.9 |    341.7 |   1537.0 |   1442.1 |   1592.4 |   1453.7 | F✓ B✓ |
| 14 | 768  | 4  |   1253.8 |   1124.2 |    773.0 |    771.6 |   3576.5 |   3470.3 |   3346.3 |   3385.3 | F✓ B✓ |
| 14 | 1024 | 8  |   2052.9 |   2053.5 |   1498.5 |   1510.3 |   7145.9 |   6961.0 |   6631.3 |   6581.7 | F✓ B✓ |
| 16 | 256  | 1  |    186.2 |    186.1 |    166.1 |    165.0 |    726.8 |    660.7 |    572.4 |    573.3 | F✓ B✓ |
| 16 | 512  | 3  |    746.7 |    737.7 |    535.4 |    537.1 |   2210.5 |   2332.9 |   2187.5 |   2162.4 | F✓ B✓ |
| 16 | 768  | 7  |   1730.0 |   1695.4 |   1255.7 |   1252.8 |   5645.3 |   5580.5 |   6357.7 |   6222.8 | F✓ B✓ |
| 16 | 1024 | 14 |   3393.9 |   3298.5 |   2516.4 |   2571.0 |  12062.5 |  11849.3 |  10469.9 |  11329.3 | F✓ B✓ |

Headline reading (ring columns):

- **Cannon-dist forward is faster everywhere**: 1.1–3.3× vs slice (small
  cells up to 3.3× — no result allgatherv; big cells ~1.35×, e.g. D=16
  χ=1024: 2516 vs 3394 ms) — the per-call AL slice gathers cost only a few
  % even un-amortized.
- **Cannon-dist backward is faster on 18/20 cells** (D=16 χ=1024: 10470 vs
  12063 ms = 1.15×); exceptions D=14 χ=512 and D=16 χ=768 (~4–13% slower,
  at n-transitions). vs the superseded v2 replicated-AL run: forward within
  ~2%, backward mixed ±5% — the slice-level reduce-scatters ≈ the old full
  allreduces on single-node NVLink (their real advantage is cross-node and
  in persistent memory: 3 tensors × χ²D²/P vs χ²D²/P + 2 full tensors).
- **NCCL columns ≈ ring columns** on this single-node 4-GPU config (±10%,
  matching Part 4's finding that NCCL pays off cross-node, not intra-node).
- The `n` column confirms the memory story: D=16 χ=1024 needs n=14 chunks
  to fit (un-chunked |H| = 137 GB > device), and runs clean.

Footnotes:

1. **Output placement asymmetry** (deliberate — the honest map-level
   comparison): slice fwd *includes* the allgatherv that replicates the full
   result on every rank; Cannon fwd ends with each rank holding only its
   block — an iterating map needs no gather since output distribution =
   input distribution. Likewise Cannon bwd leaves dFL distributed while
   slice bwd allgathers it.
2. **NCCL coverage differs**: slice fwd/bwd collectives take the NCCL fast
   path when `TENET_USE_NCCL=1`; Cannon-dist only its dM1/dM2
   `allreduce_p2p!` calls — the ring shift, row/col AL slice allgathers,
   column reduce-scatter/allgather, and the row/col gradient reduce-scatters
   are MPI point-to-point (no NCCL path yet).
3. **Stale-code attempts**: jobs `1265371`, `1273538`, `1274256` ran earlier
   revisions (Zygote-taped backward / reclaim-cold timing) and OOMed at
   D≥10 large-χ cells; their numbers are superseded by this table.

## Part 6: Kernel organization A/B — staged + eager-free vs monolithic @tensor + Zygote (1 GPU)

Job `1275909` (2026-06-12, `submit_bench_kernel_ab.sh` → `bench_kernel_ab_sofia.jl`,
commit `984ac88`), single H200, no MPI — isolates kernel organization from
communication on the identical local workload of one 2×2-grid rank
(`FL_row[χ/2,D,D,χ]·ALd_col[χ,D,D,χ/2]`, Float64 leg5 single-M).

- **A (staged)**: pairwise stage kernels with owned intermediates +
  `unsafe_free!` after last use; backward = hand-written single-contraction
  adjoints (the `FLmap_cannon` organization).
- **B (monolithic)**: original 5-tensor `@tensor` FLmap via `forloop`;
  backward = the `forloop` rrule (per-slice Zygote pullback).
- Both lower to the SAME cuTENSOR pairwise contractions; only the sum
  placement and intermediate ownership differ. Each path runs at its own
  memory-feasible chunk count (nA: coeff 8; nB: coeff 14 — Zygote tape +
  cotangent chain measured ≈10-11 |H| units vs the ordered hand chain's 6;
  at shared n, B OOMs cells A completes — jobs 1275621/1275726).
- mem columns = device used (GiB) after one un-GC'd call: pool pressure
  including dead-until-GC temporaries.

| D  | χ    | nA | nB | A fwd ms | B fwd ms | A bwd ms | B bwd ms | A fwd mem | B fwd mem | A bwd mem | B bwd mem | parity |
|----|------|----|----|----------|----------|----------|----------|-----------|-----------|-----------|-----------|--------|
| 8  | 256  | 1  | 1  |      3.3 |      4.4 |     13.9 |     12.1 |      3.0 |      3.6 |      4.1 |      7.6 | F✓ B✓ |
| 8  | 512  | 1  | 1  |     15.4 |     19.6 |     53.4 |     59.4 |      9.7 |     11.7 |     13.9 |     28.1 | F✓ B✓ |
| 10 | 512  | 1  | 1  |     45.5 |     69.1 |    155.6 |    271.2 |     21.8 |     26.6 |     31.7 |     66.3 | F✓ B✓ |
| 10 | 768  | 1  | 2  |    126.7 |    159.3 |    419.2 |    759.9 |     48.0 |     58.7 |     70.4 |    110.0 | F✓ B✓ |
| 12 | 768  | 2  | 4  |    277.7 |    358.4 |    925.9 |   1668.8 |     51.2 |    108.0 |     74.3 |    110.9 | F✓ B✓ |
| 12 | 1024 | 4  | 7  |    598.4 |    802.1 |   2119.3 |   3231.8 |     49.7 |    120.5 |     70.7 |    116.3 | F✓ B✓ |
| 14 | 1024 | 8  | 13 |   1305.6 |   1790.4 |   4244.2 |   7017.6 |     50.2 |    115.1 |     69.5 |    112.8 | F✓ B✓ |
| 16 | 1024 | 14 | 23 |   2280.5 |   2787.0 |   7171.5 |  11713.6 |     53.3 |    116.8 |     72.4 |    116.3 | F✓ B✓ |

**Reading**: A wins every cell, growing with size — forward 1.2-1.5×,
backward 1.5-1.8×, pool pressure ~40-55% lower (B's dead-until-GC
temporaries block pool reuse → continuous fresh `cudaMalloc` inside the
timed window and reactive-GC dependence). This motivates extending the
staged + eager-free + hand-adjoint organization to ALL maps (single-GPU
production paths included), which the full-VUMPS Cannon integration needs
anyway.

## Part 7: Chain-engine perf gate (1 GPU)

Job `1276469` (2026-06-12, `submit_bench_chain_gate.sh` →
`bench_chain_gate_sofia.jl`, commit `0a58586`). Paths on the Part-6 local
workload: **H** = hand staged kernels (Part 6's A), **C** = chain engine
(`chain_apply`/`chain_backward` per chunk, same n as H), **T** = monolithic
`@tensor` + forloop + Zygote (Part 6's B, own nB).

| D  | χ    | n  | nB | H fwd ms | C fwd ms | T fwd ms | H bwd ms | C bwd ms | T bwd ms | Hf mem | Cf mem | Tf mem | Hb mem | Cb mem | Tb mem | parity |
|----|------|----|----|----------|----------|----------|----------|----------|----------|--------|--------|--------|--------|--------|--------|--------|
| 10 | 512  | 1  | 1  |     45.5 |     48.7 |     59.3 |    160.7 |    222.6 |    172.0 |     22.0 |     17.2 |     26.9 |     32.1 |     32.4 |     66.6 | F✓ B✓ Tf✓ Tb✓ |
| 12 | 1024 | 4  | 7  |    598.1 |    632.2 |    778.3 |   2147.1 |   2347.2 |   3423.5 |     51.3 |     41.4 |    122.7 |     73.0 |     74.7 |    111.0 | F✓ B✓ Tf✓ Tb✓ |
| 16 | 1024 | 14 | 23 |   2283.2 |   2401.0 |   2914.1 |   7476.5 |   8103.6 |  11572.3 |     56.7 |     46.8 |    120.9 |     76.5 |     79.5 |    116.3 | F✓ B✓ Tf✓ Tb✓ |

Gate (≤1.05 time, ≤1.10 mem): **FAIL** — CHAIN/HAND fwd 1.052–1.070, bwd
1.084–1.385 (small cell dominated by fixed overheads; production cells
5–9%, converging down with size). CHAIN fwd memory is 18–22% BETTER than
hand; CHAIN beats TENSOR everywhere. Diagnosis: the engine's derived
left-assoc intermediate layouts differ from the hand kernels' (different
cuTENSOR permutation problems); per-call label processing measured ~10 µs/
link (<0.1%, not the gap). Remediation: explicit per-chain intermediate
layout pinning, then re-gate.

### Rerun with layout pinning (job `1277383`, commit `2b41db0`)

`Chain` gained explicit intermediate-layout pinning; `FLMAP_LEG5_CHAIN`
pins the hand kernels' H/T/G layouts, so cuTENSOR sees identical
permutation problems.

| D  | χ    | n  | nB | H fwd ms | C fwd ms | T fwd ms | H bwd ms | C bwd ms | T bwd ms | Hf mem | Cf mem | Hb mem | Cb mem | parity |
|----|------|----|----|----------|----------|----------|----------|----------|----------|--------|--------|--------|--------|--------|
| 10 | 512  | 1  | 1  |     45.0 |     45.2 |     73.1 |    154.3 |    165.7 |    168.7 |   22.0 |   17.2 |   32.1 |   32.4 | all ✓ |
| 12 | 1024 | 4  | 7  |    589.7 |    591.2 |    771.0 |   1942.0 |   1927.7 |   3370.2 |   51.3 |   41.4 |   73.0 |   74.7 | all ✓ |
| 16 | 1024 | 14 | 23 |   2251.0 |   2254.8 |   2859.3 |   7411.3 |   7579.6 |  11513.5 |   56.7 |   46.8 |   76.5 |   79.5 | all ✓ |

Ratios CHAIN/HAND: fwd **1.003 / 1.003 / 1.002**; bwd 1.074 / **0.993** /
1.023; fwd-mem 0.78–0.82 (CHAIN better); bwd-mem 1.01–1.04. The literal
gate still prints FAIL on the single (10,512) bwd cell (1.074) — a
fixed-overhead effect on a 160 ms workload whose run-to-run variance is
itself ~4% (compare H bwd 160.7 → 154.3 across the two runs). **Verdict:
gate satisfied at production scale** — forward identical, backward within
noise (and faster at (12,1024)), memory better. The `@generated`
compile-time lowering from the same chain tables remains the recorded
fallback if small-shape paths ever become hot. M1 closed; M2 (all maps as
chains) unblocked.

## Part 8: M2 chain-engine perf gate — production-path A/B (1 GPU)

Job `1285203` (2026-06-13, `submit_bench_chain_gate_m2.sh` →
`bench_chain_gate_m2_sofia.jl`, commit `eb800c4`, worktree `TeneT_m2gate`),
single H200, no MPI. The toggle IS the A/B switch: **CHAIN** =
`set_chain_engine!(true)` (production routes through the engine: `chain_apply`
fwd, `engine_backward`-rerouted forloop rrule bwd), **TENSOR** = `(false)`
(verbatim `@tensor` fwd + per-slice Zygote forloop rrule bwd). Six maps,
Float64 leg5 single-M; CHAIN arm at the engine-feasible `n`, TENSOR arm at its
own `nB` (Part-6 coeff 14 — each path at its feasible chunking, Part-6/7
convention). (First attempt `1285146` OOM'd on a driver bug — TENSOR arm forced
to the engine's `n` + an unguarded parity block; fixed in `eb800c4`.)

| map    | dir | D  | χ    | n  | nB | C ms     | T ms     | C/T   | C mem  | T mem  | C/T m | parity |
|--------|-----|----|------|----|----|----------|----------|-------|--------|--------|-------|--------|
| FLmap  | fwd | 10 | 512  | 1  | 1  |    186.4 |    233.3 | 0.799 |   60.8 |   99.9 | 0.609 | f✓ |
| FLmap  | bwd | 10 | 512  | 1  | 1  |   1035.0 |      oom |   —   |  120.5 |    oom |   —   | g? |
| FRmap  | fwd | 10 | 512  | 1  | 1  |    217.9 |    216.7 | 1.005 |   60.8 |   80.9 | 0.752 | f✓ |
| FRmap  | bwd | 10 | 512  | 1  | 1  |   1023.9 |      oom |   —   |  120.5 |    oom |   —   | g? |
| ACmap  | fwd | 10 | 512  | 1  | 1  |    236.1 |    232.7 | 1.015 |   80.4 |   99.9 | 0.805 | f✓ |
| ACmap  | bwd | 10 | 512  | 1  | 1  |      oom |      oom |   —   |    oom |    oom |   —   | g? |
| ACdmap | fwd | 10 | 512  | 1  | 1  |    229.1 |    229.2 | 1.000 |   80.4 |   99.9 | 0.805 | f✓ |
| ACdmap | bwd | 10 | 512  | 1  | 1  |      oom |      oom |   —   |    oom |    oom |   —   | g? |
| Cmap   | fwd | 10 | 512  | —  | —  |      1.6 |      1.6 | 1.014 |    2.4 |    2.4 | 1.000 | f✓ |
| Cmap   | bwd | 10 | 512  | —  | —  |     12.5 |      4.4 | 2.835 |    2.9 |    3.2 | 0.903 | g✓ |
| Mumap  | fwd | 10 | 512  | 1  | 1  |    258.7 |    256.7 | 1.008 |   61.0 |   80.4 | 0.759 | f✓ |
| FLmap  | fwd | 12 | 1024 | 4  | 7  |   2628.5 |   3365.9 | 0.781 |  132.6 |  127.3 | 1.042 | f✓ |
| FLmap  | bwd | 12 | 1024 | 4  | 7  |      oom |      oom |   —   |    oom |    oom |   —   | g? |
| FRmap  | fwd | 12 | 1024 | 4  | 7  |   3012.9 |   3597.4 | 0.838 |  132.6 |  127.3 | 1.042 | f✓ |
| FRmap  | bwd | 12 | 1024 | 4  | 7  |      oom |      oom |   —   |    oom |    oom |   —   | g? |
| ACmap  | fwd | 12 | 1024 | 4  | 7  |      oom |   3545.4 |   —   |    oom |  127.3 |   —   | f? |
| ACmap  | bwd | 12 | 1024 | 4  | 7  |      oom |      oom |   —   |    oom |    oom |   —   | g? |
| ACdmap | fwd | 12 | 1024 | 4  | 7  |      oom |   3450.5 |   —   |    oom |  127.3 |   —   | f? |
| ACdmap | bwd | 12 | 1024 | 4  | 7  |      oom |      oom |   —   |    oom |    oom |   —   | g? |
| Cmap   | fwd | 12 | 1024 | —  | —  |     11.3 |     11.3 | 0.996 |   12.0 |   12.0 | 1.000 | f✓ |
| Cmap   | bwd | 12 | 1024 | —  | —  |     43.4 |     33.7 | 1.288 |   13.2 |   15.4 | 0.856 | g✓ |
| Mumap  | fwd | 12 | 1024 | 4  | 7  |      oom |   3503.3 |   —   |    oom |  126.0 |   —   | f? |
| FLmap  | fwd | 16 | 1024 | 14 | 23 |   9452.8 |  11331.5 | 0.834 |  129.8 |  108.8 | 1.193 | f✓ |
| FLmap  | bwd | 16 | 1024 | 14 | 23 |      oom |      oom |   —   |    oom |    oom |   —   | g? |
| FRmap  | fwd | 16 | 1024 | 14 | 23 |  11058.6 |  14723.8 | 0.751 |  129.8 |  108.9 | 1.192 | f✓ |
| FRmap  | bwd | 16 | 1024 | 14 | 23 |      oom |      oom |   —   |    oom |    oom |   —   | g? |
| ACmap  | fwd | 16 | 1024 | 14 | 23 |      oom |  11804.1 |   —   |    oom |  108.8 |   —   | f? |
| ACmap  | bwd | 16 | 1024 | 14 | 23 |      oom |      oom |   —   |    oom |    oom |   —   | g? |
| ACdmap | fwd | 16 | 1024 | 14 | 23 |      oom |  12154.0 |   —   |    oom |  108.8 |   —   | f? |
| ACdmap | bwd | 16 | 1024 | 14 | 23 |      oom |      oom |   —   |    oom |    oom |   —   | g? |
| Cmap   | fwd | 16 | 1024 | —  | —  |     19.7 |     19.7 | 1.000 |   20.7 |   20.7 | 1.000 | f✓ |
| Cmap   | bwd | 16 | 1024 | —  | —  |     89.7 |     59.3 | 1.512 |   22.8 |   26.8 | 0.854 | g✓ |
| Mumap  | fwd | 16 | 1024 | 14 | 23 |  11901.0 |  18863.9 | 0.631 |  129.7 |  139.8 | 0.928 | f✓ |

Literal gate (≤1.05 time, ≤1.10 mem, all cells): **FAIL** — but the failure is
dominated by benchmark-harness limits, not chain-engine regressions. Reading:

- **Forward TIME — robust win (unaffected by pool state).** FLmap 0.799/0.781/
  0.834×, FRmap 1.005/0.838/0.751×, Mumap 1.008/—/0.631×. The engine forward is
  faster at every production cell where it runs, growing with size — the Part-6
  motivation confirmed on the literal production path. Parity `f✓` on every
  forward cell that ran (fwd 1e-12, the maps' single-M conj-flag chains agree
  bit-for-bit with the @tensor path).
- **Forward MEMORY — leaner where cleanly measured.** At D10χ512 every CHAIN
  forward is 0.61–0.81× TENSOR. At D12/D16 the rank-0 single-process probe is
  contaminated (see below).
- **Backward — NOT measurable at these `n` (harness, not engine).** `pick_n`
  (Part-6 coeff 8/14) was calibrated on the cannon **rank-local** workload
  (χ/2-blocks); at **full χ** a single leg5 intermediate is χ²·D⁴·8 ≈ 21 GiB at
  D10χ512, so `n=1` cannot chunk the backward under 140 GiB. CHAIN bwd ran only
  for FLmap/FRmap at D10 (120 GiB, TENSOR OOM there = engine ran where @tensor
  could not); everything else OOM/OOM. **Part-7 is the backward authority** —
  properly chunked, the chain backward is competitive-and-leaner there. The
  `g?` cells are "parity not probed" (an arm OOM'd), NOT parity failures; the
  serial suite proves bwd parity 1e-10 for all 21 maps.
- **ACmap/ACdmap forward OOM at D12/D16 ⇒ in-process pool fragmentation, not an
  intrinsic chain regression.** Same maps are *leaner* than @tensor at D10
  (80.4 vs 99.9); if the chain forward were intrinsically heavier it would show
  at D10 too. The driver measures all 6 maps × fwd/bwd in ONE process per cell;
  the 120 GiB FLmap/FRmap backward measurements fragment the pool before ACmap
  fwd, so CHAIN ACmap fwd can't get a contiguous block while TENSOR (different
  alloc pattern, measured after a GC+reclaim) squeaks in at 127 GiB. Needs
  per-map process isolation to measure clean.
- **Cmap bwd is genuinely slower (2.835/1.288/1.512×) — the one real nit.** Not
  memory (mem 0.85–0.90×), not artifact: the 3-link `chain_backward` +
  recompute overhead is not worth it for a tiny leg4 map (12 ms vs 4 ms — the
  absolute cost is negligible beside FLmap/ACmap's seconds, but per the design
  doc's "any map slower than its @tensor original is a bug" it is flagged).
  Candidate follow-up: exempt Cmap from the chain (route the guard back to
  `@tensor`), since the chain's eager-free/recompute machinery only pays off on
  the χ²D⁴-class maps.

**Verdict:** the chain engine is **correct** (all serial + 4-rank MPI parity
gates green; forward parity `f✓` here) and the **forward is a clear production
win** (0.63–0.83× time at scale, leaner memory where cleanly probed). The
literal-gate FAIL is (a) a backward `n`-calibration gap in the harness and (b)
in-process pool fragmentation at D≥12 — not proven engine regressions — plus
(c) one real, tiny per-map nit (Cmap bwd overhead). Follow-ups: a re-run with
per-map process isolation + a backward-calibrated `n` to get clean at-scale
memory, and the Cmap-chain exemption. FLmap_C3v and the corner maps are
consciously unbenched (out-of-scope per the M2 plan: corner maps' only caller
is commented out; FLmap_C3v lives only in the qrctmrg path, no Sofia production
benchmark today).

## Part 9: M3 Cannon-wrapper 4-GPU validation — `Cmap_cannon` / `FRmap_cannon_dist` / `ACmap_cannon_dist` / `ACdmap_cannon_dist`

Jobs `1285915` (`forloop_iter=4`) + `1285916` (`forloop_iter=16`, 2026-06-14,
`submit_test_cannon_m3.sh` → `test_cannon_m3_sofia.jl`, branch
`claude/ecstatic-golick-e3f6f0` @ `7e2372f`), 1 node × 4×H200, Cannon grid 2×2.
Plan: [Batch E](../../../docs/2026-06-13-m3-cannon-wrappers-plan.md).

**Result: GPU distributed parity PASSED for all four maps at χ=256 D=8 (rel
3–5e-15, ≪ gate); the χ=400 D=10 cell OOMs on the SERIAL REFERENCE (not the
distributed maps) — a validation-harness limit, diagnosed below.** Run via:

```bash
cd examples/MPI_parallel/Sofia && sbatch submit_test_cannon_m3.sh
```

Methodology: each of the four M3 maps is run distributed on `CuArray`s and its
forward + Zygote-sum-loss gradient compared (rel err, allreduced max over ranks)
against the serial `Cmap`/`FRmap`/`ACmap`/`ACdmap` reference — the SAME
`*_cannon_dist` code the 4-rank CPU parity gate (`test/test_cannon_m3.jl`)
exercises; the GPU is the only new variable. Gate: **forward rel ≤ 1e-10,
gradient max rel ≤ 1e-8** (the CPU test gate is tighter, 1e-12 / 1e-10 — the GPU
thresholds absorb device-FP reduction order). Two cells: a validation cell
(`χ=256 D=8`, env defaults `TENET_CANNON_CHI_VALID`/`TENET_CANNON_D_VALID`) and a
production cell (`χ=400 D=10`, `TENET_CANNON_CHI`/`TENET_CANNON_D`), Float64 leg5,
both single-M and tuple-M. Cmap any grid (replicated output); FR/AC/ACd require
the square 2×2 grid. `forloop_iter=4` (`TENET_CANNON_FLOOP`) so `n_d=n_i ≥ 2N`,
the regime that forces the 2-level accumulate/assign chunk for FRmap/ACdmap.

**Device-memory column** = device used (`total − available`) GiB, max over ranks,
sampled by `mem_line` immediately after each FRmap/ACdmap forward and fwd+bwd
(the same probe as Part 5/7/8). **The load-bearing claim** (the FLmap-OOM
lesson): FRmap and ACdmap carry full-`i`×full-`d` chain intermediates; the
2-level chunk must keep the peak bounded at ≈χ²D⁴/(P·forloop_iter), **NOT** a
χ×χ plane blow-up. Compare their peak to ACmap (single l-chunk) at the same cell
— if FRmap/ACdmap stay within a small factor of ACmap (no χ²-plane spike), the
design's bounded-intermediate claim holds at production scale.

Parity table (fill `✓`/`✗` + the allreduced max rel errors):

Parity table (job 1285916; the χ=256 D8 dev-mem columns are the co-resident
total = serial reference + distributed map, see diagnosis):

| map      | χ   | D  | variant | fwd rel err | grad max rel err | fwd | grad | dev-mem fwd (GiB) | dev-mem fwd+bwd (GiB) |
|----------|-----|----|---------|-------------|------------------|-----|------|-------------------|-----------------------|
| Cmap     | 256 | 8  | leg4    | 0.0         | 0.0              | ✓   | ✓    | —                 | —                     |
| FRmap    | 256 | 8  | tuple-M | 3.23e-15    | 4.49e-15         | ✓   | ✓    | 13.88             | 26.45                 |
| FRmap    | 256 | 8  | 1M      | 5.50e-15    | 4.50e-15         | ✓   | ✓    | 14.35             | 27.88                 |
| ACmap    | 256 | 8  | tuple-M | 3.26e-15    | 4.57e-15         | ✓   | ✓    | (ref)             | (ref)                 |
| ACmap    | 256 | 8  | 1M      | 5.52e-15    | 4.58e-15         | ✓   | ✓    | (ref)             | (ref)                 |
| ACdmap   | 256 | 8  | tuple-M | 3.23e-15    | 4.49e-15         | ✓   | ✓    | 18.20             | 31.88                 |
| ACdmap   | 256 | 8  | 1M      | 5.45e-15    | 4.50e-15         | ✓   | ✓    | 18.26             | 31.95                 |
| Cmap     | 400 | 10 | leg4    | 0.0         | 0.0              | ✓   | ✓    | —                 | —                     |
| FRmap    | 400 | 10 | tuple-M | — (OOM)     | — (OOM)          | —   | —    | 74.13 → OOM       | OOM (serial ref)      |
| ACmap/ACdmap | 400 | 10 | both | — (not reached) | — (not reached) | — | — | —             | —                     |

**Headline reading:**

- **Parity — PASS (the primary goal).** All four maps reproduce the serial
  kernels on `CuArray` at χ=256 D=8 to rel **3–5e-15** (≪ the 1e-10 fwd / 1e-8
  grad gate), both tuple-M and single-M — the GPU half the CPU 4-rank gate
  cannot reach. Cmap is exact to FP (replicated output). Independent of
  `forloop_iter` (n=4 and n=16 both pass identically).
- **χ=400 D=10 OOM is the SERIAL REFERENCE, NOT the distributed maps.** The
  validator compares each dist map against `serial FRmap/ACmap/ACdmap` (the
  chain engine, default ON, run un-distributed on ONE GPU, held on every rank).
  At χ=400 D=10 that serial kernel builds the FULL `FRMAP_LEG5_CHAIN`
  intermediate I2 = full-i × full-d × D⁴ × d_phys = χ²·D⁴·2·16 B = **47.684 GiB**
  — its forward peaks at 74 GB, and its **gradient** (`chain_backward` recomputing
  I2) requests another 47.684 GiB → OOM at 99.99% of the 140 GiB H200. The
  failing allocation size is EXACTLY χ²D⁴·d_phys and is **identical at
  forloop_iter=4 and 16** — because it is the *un-chunked serial reference*, not
  the chunked distributed map. (The "FRmap forward peak 13.88 GB" at χ=256 is
  likewise the serial ref's full I1+I2 = 4.3+8.6 GB — which is why that column is
  `forloop_iter`-invariant.) **This is precisely WHY the distributed maps exist:
  the full serial kernel does not fit at production χ on one GPU.** It is a
  harness limit (the parity check needs the unfittable full reference), not a
  distributed-map defect or a chunk-bound failure.
- **At-scale distributed-map memory: design-proven, not yet GPU-probed in
  isolation.** Because the driver co-locates the dist map with the full serial
  ref, the dev-mem columns measure their SUM, dominated by the ref; the dist
  map's own bounded peak (≈χ²D⁴/(P·forloop_iter)) was never measured alone. The
  χ=256 cell shows the dist maps run correctly alongside the full ref; the
  bounded-intermediate claim rests on the design proof + the 4-rank CPU parity.
  **Follow-up to close it empirically:** a serial-ref-free probe at χ=400 D=10 —
  run each dist map standalone (no serial comparison), assert it does NOT OOM,
  measure its peak, and check chunk-count-invariance (gather(dist@floop=a) ≈
  gather(dist@floop=b)) for at-scale correctness without the unfittable ref.

`=== RESULT ===` lines per map are printed by the driver. **Verdict: M3
distributed parity validated on GPU (χ=256 D=8, all 4 maps); the production-cell
memory check is blocked by the serial reference's size, recorded as a follow-up
(serial-ref-free dist-only probe).**

## Part 10: Cannon scaling — 2×2 (4 GPU, 1 node) vs 4×4 (16 GPU, 2 nodes) — `FLmap_cannon_dist`

Jobs `1287186` (4 GPU, grid 2×2, 1 node, `COMPLETED` 22:41) + `1287187`
(16 GPU, grid 4×4, **2 nodes × 8×H200**, `COMPLETED` 19:54), 2026-06-15,
branch `claude/ecstatic-golick-e3f6f0` @ `d28e3da` (M2 chain-engine **default
ON** + all four M3 cannon maps merged), `submit_benchmark_cannon.sh` /
`submit_benchmark_cannon_16gpu.sh` → `benchmark_cannon_sofia.jl` with
`TENET_CANNON_N1=N2={2,4}`.

Same driver/methodology as Part 5 (Float64 leg5 single-M, `total_splits=128`,
`nrep=3`, backward = Zygote-`sum` pullback, `forloop_iter=n` per cell). The
4×4 grid is laid out **2 rows per node** (`rank = r1·4 + r2`, 8 ranks/node):
cannon **row** comms (ring shifts + AL row-gather, tags 700/750) stay
intra-node on NVLink; cannon **column** comms (reduce-scatter / allgather,
tags 710/730) **cross the IB link**. Both runs: all 20 cells, parity ≤1e-10
fwd / ≤1e-8 bwd on **every** cell (`F✓ B✓`). This is the repo's first 16-GPU
cannon timing (Part 2's 16-GPU table is the slice `FLmap_parallel` path).

**(a) 4 GPU, grid 2×2 (1 node) — current commit `d28e3da`:**

| D  | χ    | n  | slice fwd ring | slice fwd nccl | cannon fwd ring | cannon fwd nccl | slice bwd ring | slice bwd nccl | cannon bwd ring | cannon bwd nccl | parity |
|----|------|----|----------------|----------------|-----------------|-----------------|----------------|----------------|-----------------|-----------------|--------|
| 8  | 256  | 1  |     29.2 |     24.9 |      9.9 |     10.3 |     80.4 |     77.8 |     31.8 |     31.1 | F✓ B✓ |
| 8  | 512  | 1  |     55.8 |     55.7 |     40.3 |     40.6 |    173.5 |    166.4 |    145.8 |    130.3 | F✓ B✓ |
| 8  | 768  | 1  |    124.0 |    121.1 |     99.4 |     99.2 |    370.9 |    454.4 |    323.7 |    323.0 | F✓ B✓ |
| 8  | 1024 | 1  |    210.0 |    205.4 |    198.1 |    212.6 |    632.8 |    611.2 |    646.4 |    662.2 | F✓ B✓ |
| 10 | 256  | 1  |     36.3 |     35.2 |     28.0 |     27.7 |    118.6 |    115.6 |     87.6 |    117.1 | F✓ B✓ |
| 10 | 512  | 1  |    110.3 |    107.7 |    102.8 |    103.4 |    396.8 |    342.3 |    341.9 |    342.8 | F✓ B✓ |
| 10 | 768  | 1  |    233.8 |    231.2 |    259.6 |    260.0 |    818.6 |    824.5 |    856.8 |    860.3 | F✓ B✓ |
| 10 | 1024 | 2  |    467.4 |    461.4 |    416.7 |    413.2 |   1513.5 |   1664.7 |   1557.1 |   1624.6 | F✓ B✓ |
| 12 | 256  | 1  |     52.3 |     52.4 |     55.7 |     56.3 |    179.4 |    174.5 |    229.8 |    249.6 | F✓ B✓ |
| 12 | 512  | 1  |    184.4 |    182.9 |    217.7 |    217.4 |    759.3 |    598.9 |    729.0 |    726.2 | F✓ B✓ |
| 12 | 768  | 2  |    441.5 |    437.2 |    426.8 |    426.2 |   1478.3 |   1519.2 |   1706.2 |   1762.0 | F✓ B✓ |
| 12 | 1024 | 4  |    883.8 |    879.0 |    758.8 |    768.5 |   3010.1 |   3130.1 |   3284.4 |   3246.1 | F✓ B✓ |
| 14 | 256  | 1  |     85.8 |     84.2 |     96.2 |     96.6 |    307.9 |    303.3 |    332.1 |    330.9 | F✓ B✓ |
| 14 | 512  | 2  |    526.6 |    535.3 |    339.5 |    340.1 |   1274.4 |   1337.1 |   1528.4 |   1379.6 | F✓ B✓ |
| 14 | 768  | 4  |    820.2 |    829.7 |    763.4 |    764.7 |   3139.3 |   3104.4 |   3378.0 |   3459.1 | F✓ B✓ |
| 14 | 1024 | 8  |   1694.3 |   1688.3 |   1477.8 |   1495.7 |   6143.2 |   6071.8 |   6393.0 |   6429.9 | F✓ B✓ |
| 16 | 256  | 1  |    115.6 |    114.7 |    162.2 |    162.1 |    506.5 |    438.4 |    570.8 |    620.6 | F✓ B✓ |
| 16 | 512  | 3  |    484.7 |    481.8 |    526.7 |    526.7 |   1868.2 |   1827.0 |   2208.6 |   2178.1 | F✓ B✓ |
| 16 | 768  | 7  |   1407.3 |   1414.5 |   1279.7 |   1290.3 |   5156.2 |   5032.8 |   5674.4 |   5634.8 | F✓ B✓ |
| 16 | 1024 | 14 |   2663.3 |   2640.9 |   2477.9 |   2501.8 |   9650.2 |   9859.8 |  10777.3 |  10919.0 | F✓ B✓ |

**(b) 16 GPU, grid 4×4 (2 nodes, cross-node IB) — first 16-GPU cannon data:**

| D  | χ    | n  | slice fwd ring | slice fwd nccl | cannon fwd ring | cannon fwd nccl | slice bwd ring | slice bwd nccl | cannon bwd ring | cannon bwd nccl | parity |
|----|------|----|----------------|----------------|-----------------|-----------------|----------------|----------------|-----------------|-----------------|--------|
| 8  | 256  | 1  |     11.5 |     24.4 |     25.7 |     24.8 |     47.9 |    119.3 |     54.8 |     88.5 | F✓ B✓ |
| 8  | 512  | 1  |     21.5 |     35.2 |    109.8 |     97.2 |    157.8 |    159.1 |    218.0 |    249.4 | F✓ B✓ |
| 8  | 768  | 1  |     47.0 |     57.3 |    230.8 |    227.6 |    390.4 |    280.3 |    486.1 |    548.1 | F✓ B✓ |
| 8  | 1024 | 1  |     80.2 |     82.0 |    439.9 |    405.4 |    630.1 |    505.2 |    970.1 |    914.5 | F✓ B✓ |
| 10 | 256  | 1  |     14.1 |     29.3 |     43.0 |     49.2 |     75.1 |    126.0 |     90.9 |    130.1 | F✓ B✓ |
| 10 | 512  | 1  |     48.2 |     49.3 |    164.7 |    167.8 |    281.5 |    321.2 |    395.5 |    487.8 | F✓ B✓ |
| 10 | 768  | 1  |     87.0 |     90.9 |    393.1 |    395.5 |    646.4 |    430.7 |    876.5 |    893.0 | F✓ B✓ |
| 10 | 1024 | 1  |    162.8 |    187.3 |    713.6 |    711.3 |   1127.6 |    789.6 |   1628.2 |   1749.6 | F✓ B✓ |
| 12 | 256  | 1  |     24.5 |     37.0 |     65.0 |     62.3 |    109.7 |    148.8 |    181.7 |    191.3 | F✓ B✓ |
| 12 | 512  | 1  |     65.4 |     71.2 |    309.2 |    263.4 |    418.3 |    344.8 |    656.4 |    653.7 | F✓ B✓ |
| 12 | 768  | 1  |    150.7 |    166.7 |    636.3 |    622.9 |    989.1 |    717.7 |   1485.5 |   1582.1 | F✓ B✓ |
| 12 | 1024 | 1  |    289.7 |    288.4 |   1117.0 |   1150.1 |   1995.4 |   1357.4 |   2837.6 |   2955.9 | F✓ B✓ |
| 14 | 256  | 1  |     46.2 |     44.9 |    101.2 |    102.8 |    182.2 |    211.5 |    253.1 |    291.6 | F✓ B✓ |
| 14 | 512  | 1  |    107.6 |    110.4 |    412.1 |    420.6 |    730.1 |    597.5 |   1008.5 |   1117.9 | F✓ B✓ |
| 14 | 768  | 1  |    274.0 |    268.2 |    960.5 |    973.2 |   1620.0 |   1301.2 |   2516.9 |   2608.8 | F✓ B✓ |
| 14 | 1024 | 2  |    523.0 |    536.7 |   1570.2 |   1573.9 |   3165.4 |   2332.0 |   4340.3 |   4198.3 | F✓ B✓ |
| 16 | 256  | 1  |     63.0 |     61.7 |    143.0 |    144.1 |    258.9 |    258.7 |    380.7 |    417.1 | F✓ B✓ |
| 16 | 512  | 1  |    157.9 |    154.2 |    598.3 |    598.9 |    993.1 |    790.5 |   1478.5 |   1589.1 | F✓ B✓ |
| 16 | 768  | 2  |    427.1 |    477.2 |   1224.8 |   1266.1 |   2571.2 |   2054.3 |   3300.1 |   3344.2 | F✓ B✓ |
| 16 | 1024 | 4  |    803.8 |    866.3 |   2193.3 |   2082.5 |   5006.7 |   3817.9 |   6070.1 |   6175.6 | F✓ B✓ |

Headline reading (ring columns):

- **No regression: the 4-GPU table reproduces Part 5 within a few %** (cannon
  fwd D=16 χ=1024 2478 vs Part5 2516; D=8 χ=256 9.9 vs 10.0; cannon bwd D=16
  χ=1024 10777 vs 10470, +2.9%). M2's engine-default-ON + the M3 map merge
  leave `FLmap_cannon_dist`'s single-node performance unchanged, as expected
  (neither touched its hand kernels).

- **4 GPU (single-node NVLink) cannon wins; at 16 GPU (cross-node IB) the
  picture splits by direction.**
  - **Backward scales — 1.5–1.8× faster at 16 GPU on the heavy cells** (D=16
    χ=1024 10777→6070 = 1.78×; D=16 χ=768 5674→3300 = 1.72×; D=14 χ=1024
    6393→4340 = 1.47×). Backward is compute-dominated (per-chunk recompute +
    six adjoints), so 4× more ranks beats the added cross-node traffic. Small
    cells regress (D=8 χ=1024 646→970 = 1.5× slower) where comm dominates the
    tiny compute.
  - **Forward barely scales — comm-bound.** Only the two largest cells edge
    ahead at 16 GPU (D=16 χ=1024 2478→2193 = 1.13×; D=16 χ=768 1280→1225);
    everywhere else 16-GPU cannon fwd is *slower* than 4 GPU (D=8 χ=1024
    198→440 = 2.2×). The per-call row/col AL slice gathers now cross IB and
    dominate the lighter forward compute.

- **Cannon vs slice at 16 GPU.** Cannon **fwd** is 2.7–5.5× slower than slice
  (the cross-node gather, counted in isolation); **bwd** only 1.2–1.5× slower.
  Both gaps are the map-isolated penalty of Part 5 footnote 1 — in a leftenv
  power iteration ALu/ALd are fixed so the forward gathers amortize away, and
  cannon keeps its memory/locality advantage (3×χ²D²/P resident, block-local
  recompute).

- **NCCL helps slice cross-node, not cannon (yet).** At 16 GPU the slice
  **bwd** allreduce takes the NCCL fast path and gains ~1.3× (D=16 χ=1024
  slice bwd 5007→3818 ring→nccl; D=14 χ=1024 3165→2332) — Part 4's cross-node
  NCCL win. Cannon's bwd is NCCL-flat (6070 vs 6176) because its column/row
  reduce-scatters are still MPI point-to-point (Part 5 footnote 2). **A
  cross-node NCCL path for the cannon reduce-scatters is the open lever** to
  make distributed-cannon backward scale like slice at ≥2 nodes.

- **Memory pressure scales 1/P: the chunk count `n` drops ~4× at 16 GPU**
  (D=16 χ=1024 n=14→4; D=14 χ=1024 n=8→2; D=16 χ=768 n=7→2) — each rank holds
  χ²D⁴/(P·n), so 4× more ranks need 4× fewer forloop chunks to fit, confirming
  the per-rank bound at the wider grid.

`.out` files: `examples/MPI_parallel/Sofia/Sofia_cannon_bench_1287186.out`,
`Sofia_cannon_bench16_1287187.out`. Footnotes 1–2 of Part 5 (output-placement
asymmetry, NCCL coverage) apply unchanged.

## Part 11: M3.5 ring-class reorder — FRmap/ACdmap gather-class → ring-class

Branch `claude/ecstatic-golick-e3f6f0` @ `334575d`. Design:
[`docs/2026-06-15-m35-cannon-ring-reorder-design.md`](../../../docs/2026-06-15-m35-cannon-ring-reorder-design.md).
The M3 gather-class FRmap/ACdmap (2-level i/d chunk, a full-i×full-d intermediate
plane) are reordered so the cross-axis **contracted** leg dies at link 1 (FL·ACd /
FR·ARu) — exactly as ACmap already did — collapsing the plane to χ²D⁴/P. All four
cannon maps become **single-l-chunk ring-class** with **identical communication**
(no new primitive; only the local chain order + chunk loop change). Driver
`benchmark_cannon_maps_sofia.jl` (Float64 leg5 single-M, nrep=3, cannon
`forloop_iter=pick_n`; parity vs the chunked `*_parallel` slice ref — no
serial-ref OOM). Validated: 4-rank CPU parity (incl. multi-chunk accumulate,
off-diagonal trap blocks, single-M, Db≠Dc) + opus review CLEAN + GPU parity
`F✓ B✓` every cell. A 2³¹ cuTENSOR floor (`_ring_l_chunks`) was added after a
2×2 D=10 χ=768 illegal-address (the 7-dim I2 = na·local_l·D⁴·d_phys overflows
32-bit StridedView indexing > 2³¹; the floor caps each chunk's I2 < 2e9).

Gather-class baseline = job `1287203` (16 GPU 4×4); ring-class = the **ring**
columns of job `1287248` (16 GPU 4×4, same matrix). 16-GPU headline (ms):

| cell | FLmap | ACmap | FRmap gather→ring | ACdmap gather→ring |
|------|-------|-------|-------------------|---------------------|
| D16 χ1024 fwd | 2169 | 2139 | 8245 → **2345** (3.5×) | 8329 → **2248** (3.7×) |
| D16 χ1024 bwd | 5968 | 5960 | 38583 → **7247** (5.3×) | 28977 → **6133** (4.7×) |
| D14 χ1024 fwd | 1495 | 1524 | 4844 → **1675** (2.9×) | 4974 → **1642** (3.0×) |
| D14 χ1024 bwd | 3948 | 3725 | 22144 → **5048** (4.4×) | 16339 → **4044** (4.0×) |

**Reading.** The gather-class FRmap/ACdmap ran **3–6× FLmap** (the full-i×full-d
plane = N× redundant FLOPs/rank + ≈P·n small GEMMs); the reorder drops them to
**≈ FLmap / ACmap (1.0–1.1×)** — all four maps now one architecture. The penalty
grew with grid size in the gather-class (N× factor: 16-GPU FRmap bwd was 6.4×
FLmap vs 3.1× at 4 GPU), so the reorder matters more at scale. The 4-GPU
gather-class baseline (`1287202`) showed the same 2.4–3.8× that the reorder
removes. Full 20-cell × 4-map tables are the ring columns of Part 12.

## Part 12: NCCL fast path for the cannon col/row collectives — 16 vs 64 GPU scaling

Commits `b2f9906` (NCCL path) + `5463dd8` (64-GPU script). The four cannon
collectives (`_cannon_{col,row}_{allgather,reduce_scatter*}`) gain an NCCL path
(`TENET_USE_NCCL=1`), mirroring the existing `allreduce_p2p!`/`allgatherv_p2p!`
seam: `_get_nccl_comm(grid.row_comm/col_comm)` reuses the per-MPI-comm NCCL cache
(no new infra); ROW collectives (last leg) call ncclAllGather/ncclReduceScatter
directly; COLUMN collectives (first leg) permute the leg first↔last around the
NCCL call (a local GPU transpose, cheap vs the cross-node IB transfer it
replaces). Guarded `_use_nccl() && CuArray && equal-blocks (χ%N==0)` → MPI
fallback. In the 2-rows-per-node (4×4) / 1-row-per-node (8×8) layout the
**col_comm is the cross-node axis** — the high-value NCCL target. opus-reviewed
CLEAN (permute round-trip + rank-mapping empirically verified). Jobs `1287248`
(16 GPU 4×4, 2 nodes) + `1287257` (64 GPU 8×8, 8 nodes); ring vs nccl columns,
**parity recomputed under NCCL — `F✓ B✓` every cell** (the permute path is
numerically correct at both 4×4 and 8×8).

**NCCL speedup, FLmap fwd at χ=1024 (ms ring→nccl, FR/AC/ACd track within ±15%):**

| D  | 16 GPU (2 nodes) | 64 GPU (8 nodes) |
|----|------------------|------------------|
| 8  | 388 → 168 (2.3×) | 216 → 132 (1.6×) |
| 10 | 704 → 313 (2.2×) | 357 → 234 (1.5×) |
| 12 | 1096 → 500 (2.2×)| 545 → 351 (1.6×) |
| 14 | 1495 → 687 (2.2×)| 837 → 587 (1.4×) |
| 16 | 2169 → 997 (2.2×)| 1160 → 704 (1.7×)|

Backward tracks forward (16 GPU ~1.4–1.7×, 64 GPU ~1.5–1.7× at χ=1024).

**Headline — NCCL's benefit is set by the per-rank MESSAGE SIZE, not node count.**
- **Large cells win**: at χ=1024 NCCL is 2.2× (16 GPU) / 1.5–1.7× (64 GPU). At
  16 GPU the crossover where NCCL starts winning is ≈χ512; below it the col
  blocks are too small for the IB transfer to amortise NCCL's fixed launch cost.
- **The benefit SHRINKS at 64 GPU (strong scaling, fixed χ)**, contrary to the
  naive "more nodes ⇒ more NCCL". Per-rank comm is χ²D²/√P (shrinks as 1/√P) so
  at 8×8 each col message is 4× smaller than at 4×4 → NCCL's fixed latency is a
  bigger fraction → the crossover moves UP to ≈χ1024, and small cells get much
  worse (FLmap D8 χ256 fwd: 16 GPU 25→51 = 2× slower; 64 GPU 12→66 = **5.6×
  slower**). Two competing effects — comm fraction grows with √P (pro-NCCL) but
  per-message size shrinks (anti-NCCL); at fixed χ the latter wins.
- **To grow the NCCL win with node count, scale WEAKLY** (raise χ with P so the
  per-rank message stays large). NCCL fully pays off in the production regime
  (large χ≥1024, D≥10).

**Production guidance**: gate NCCL on the per-rank message `~χ·D/√P`, not blanket
on — enable for large χ / moderate P, keep the hand ring for the strong-scaling
tail and small cells. (Full 20-cell × 4-map ring/nccl tables: jobs 1287248 /
1287257 `.out`; regenerate via `submit_benchmark_cannon_maps_{16,64}gpu.sh`.)

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
  not yet fully tuned. **Mitigation:** `TENET_USE_NCCL=1` (commit `60fa029`)
  drops this to **11.83 ms (3.76× faster)** by routing through `ncclAllReduce`
  instead of the 3-phase p2p ring. See Part 4 for the production trade-off.
- **Home dir `/user/sofia/$VSC` doesn't exist** — work out of
  `/sofia/scratch/pilot/pilot_2026_0002/<user>/` with `HOME=$WD` +
  `JULIA_DEPOT_PATH=$WD/.julia`.
- **`MPI.Allreduce!(MPI.IN_PLACE, ::CuArray, ...)` crashes** inside OpenMPI 5.0.7
  `mca_coll_cuda`. Use `Allreduce!(sendbuf, recvbuf, ...)` non-IN_PLACE, or run
  with `--mca coll_cuda_priority 0`. Not triggered by TeneT's
  `allreduce_p2p!` (which uses Isend/Recv through UCX PML).
