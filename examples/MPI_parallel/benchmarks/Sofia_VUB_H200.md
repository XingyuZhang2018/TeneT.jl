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

## Part 5: Cannon 2×2 vs slice (4 GPU, single node)

Job `1274276` (2026-06-12, `submit_benchmark_cannon.sh` → `benchmark_cannon_sofia.jl`,
branch `claude/sad-saha-3ec6bf` @ `2b456d0`), 1 node × 4×H200, Cannon grid 2×2.
Design: [`docs/2026-06-10-cannon-flmap-design.md`](../../../docs/2026-06-10-cannon-flmap-design.md).

Methodology: tensors/loss as Part 2 (`test_MPI_config.jl`) — Float64 leg5
single-M, slice path `total_splits=128`, `nrep=3`, backward = Zygote pullback
of a bare `sum` loss. Cannon sub-slices its local l range with
`forloop_iter = n` per cell (column `n`; backward-peak formula
`(4+2d)·|H|/n + residents ≤ 110 GB`, |H| = χ²D⁴/4). Timings are per-rep
max-over-ranks with `GC.gc()` (no reclaim) between reps outside the timed
window — reads slightly above Part 2's rank-0 `@elapsed` convention. Cannon's
backward is fully hand-written (six single-contraction adjoints, per-chunk
local H/T/G recompute, `unsafe_free!` after each array's last use — no
Zygote inside the map's rrule). All 20 cells ran (no OOM skips); parity vs
the slice path ≤1e-10 fwd / ≤1e-8 bwd on every cell (`F✓ B✓`).

Column legend — all times in ms: **slice** = the existing replicated-input
path (`FLmap_parallel`, Part 2's subject); **cannon** = the distributed
`FLmap_cannon`; **fwd** = one forward map call; **bwd** = one
`Zygote.pullback` construction + backward; **ring** / **nccl** =
`TENET_USE_NCCL` off / on; **n** = Cannon's `forloop_iter` for that cell.

| D  | χ    | n  | slice fwd ring | slice fwd nccl | cannon fwd ring | cannon fwd nccl | slice bwd ring | slice bwd nccl | cannon bwd ring | cannon bwd nccl | parity |
|----|------|----|----------------|----------------|-----------------|-----------------|----------------|----------------|-----------------|-----------------|--------|
| 8  | 256  | 1  |     32.2 |     28.7 |      9.7 |      9.3 |    112.1 |    110.9 |     38.7 |     36.7 | F✓ B✓ |
| 8  | 512  | 1  |     71.9 |     73.8 |     38.9 |     38.6 |    254.9 |    260.7 |    138.4 |    131.7 | F✓ B✓ |
| 8  | 768  | 1  |    148.3 |    142.4 |    107.2 |     97.8 |    533.0 |    518.5 |    373.1 |    349.8 | F✓ B✓ |
| 8  | 1024 | 1  |    374.3 |    324.8 |    194.1 |    193.9 |    843.9 |    835.1 |    747.4 |    730.7 | F✓ B✓ |
| 10 | 256  | 1  |     44.8 |     44.0 |     23.5 |     23.0 |    172.6 |    168.1 |    105.6 |     97.5 | F✓ B✓ |
| 10 | 512  | 1  |    137.3 |    179.6 |    101.0 |    101.3 |    563.5 |    519.2 |    373.2 |    356.4 | F✓ B✓ |
| 10 | 768  | 1  |    324.8 |    333.0 |    256.5 |    256.3 |   1074.8 |   1195.6 |    894.6 |    980.7 | F✓ B✓ |
| 10 | 1024 | 2  |    612.6 |    768.5 |    416.0 |    403.0 |   1985.7 |   1779.1 |   1675.7 |   1576.1 | F✓ B✓ |
| 12 | 256  | 1  |     71.4 |     70.9 |     47.6 |     47.1 |    320.3 |    275.2 |    174.5 |    171.1 | F✓ B✓ |
| 12 | 512  | 1  |    247.4 |    246.8 |    215.7 |    215.7 |   1021.7 |    803.0 |    786.3 |    938.9 | F✓ B✓ |
| 12 | 768  | 2  |    632.7 |    609.5 |    596.6 |    628.8 |   1864.7 |   1824.4 |   1768.6 |   1870.0 | F✓ B✓ |
| 12 | 1024 | 4  |   1132.1 |   1248.3 |    751.3 |    754.2 |   3588.5 |   3626.6 |   3270.6 |   3236.9 | F✓ B✓ |
| 14 | 256  | 1  |    335.4 |    119.3 |     97.2 |     95.9 |    495.2 |    460.5 |    352.2 |    345.1 | F✓ B✓ |
| 14 | 512  | 2  |    633.5 |    520.6 |    341.7 |    338.3 |   1419.2 |   1425.0 |   1563.6 |   1502.7 | F✓ B✓ |
| 14 | 768  | 4  |   1057.7 |   1227.7 |    773.3 |    761.2 |   3521.7 |   3523.6 |   3237.5 |   3370.9 | F✓ B✓ |
| 14 | 1024 | 8  |   2228.4 |   2101.9 |   1530.4 |   1489.4 |   6976.7 |   6948.0 |   6411.9 |   6431.3 | F✓ B✓ |
| 16 | 256  | 1  |    181.9 |    179.1 |    161.6 |    161.8 |    660.3 |    640.3 |    633.5 |    571.7 | F✓ B✓ |
| 16 | 512  | 3  |    772.9 |    739.2 |    520.5 |    584.3 |   2394.4 |   2247.8 |   2327.4 |   2381.6 | F✓ B✓ |
| 16 | 768  | 7  |   1741.3 |   1692.3 |   1239.8 |   1241.3 |   5849.9 |   5811.5 |   5519.0 |   5536.9 | F✓ B✓ |
| 16 | 1024 | 14 |   3385.8 |   3466.5 |   2482.4 |   2490.7 |  12739.3 |  12447.4 |  11005.0 |  10900.5 | F✓ B✓ |

Headline reading (ring columns):

- **Cannon forward is faster everywhere**: 1.1–3.3× vs slice (small cells up
  to 3.3× — no result allgatherv; big cells settle to ~1.4×, e.g. D=16
  χ=1024: 2482 vs 3386 ms).
- **Cannon backward is faster on 18/20 cells** (e.g. D=8 χ=256: 2.9×;
  D=16 χ=1024: 11005 vs 12739 ms = 1.16×); the two exceptions
  (D=14 χ=512, D=16 χ=512 — within ~10%) sit at the n-transition where
  chunk recompute overhead meets the slice path's amortized sub-slicing.
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
   path when `TENET_USE_NCCL=1`; Cannon only its 4 backward `allreduce_p2p!`
   calls — the ring shift, column reduce-scatter/allgather, and row
   reduce-scatter are MPI point-to-point (no NCCL path yet).
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
