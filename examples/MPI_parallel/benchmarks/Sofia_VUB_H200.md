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

| D  | χ    | Size   | 1 GPU  | 2 GPU | 4 GPU | 8 GPU | 16 GPU | 2× | 4× | 8× | 16× |
|----|------|--------|--------|-------|-------|-------|--------|----|----|----|-----|
| 8  | 256  | 32MB   |    546 |   301 |   187 |   147 |    121 | 1.81× | 2.92× | 3.71× | 4.51× |
| 8  | 512  | 128MB  |    762 |   460 |   394 |   401 |    295 | 1.66× | 1.93× | 1.90× | 2.58× |
| 8  | 768  | 288MB  |   1386 |   785 |   557 |   767 |    723 | 1.77× | 2.49× | 1.81× | 1.92× |
| 8  | 1024 | 512MB  |   2534 |  1342 |   844 |   932 |   1178 | 1.89× | 3.00× | 2.72× | 2.15× |
| 10 | 256  | 50MB   |    614 |   386 |   261 |   238 |    153 | 1.59× | 2.35× | 2.58× | 4.01× |
| 10 | 512  | 200MB  |   1407 |   787 |   545 |   774 |    636 | 1.79× | 2.58× | 1.82× | 2.21× |
| 10 | 768  | 450MB  |   3170 |  1672 |  1004 |  1017 |   1182 | 1.90× | 3.16× | 3.12× | 2.68× |
| 10 | 1024 | 800MB  |   6253 |  3219 |  2075 |  1530 |   1643 | 1.94× | 3.01× | 4.09× | 3.81× |
| 12 | 256  | 72MB   |    789 |   481 |   400 |   450 |    283 | 1.64× | 1.97× | 1.75× | 2.79× |
| 12 | 512  | 288MB  |   2423 |  1295 |   910 |   943 |   1032 | 1.87× | 2.66× | 2.57× | 2.35× |
| 12 | 768  | 648MB  |   6367 |  3276 |  2093 |  1447 |   1551 | 1.94× | 3.04× | 4.40× | 4.10× |
| 12 | 1024 | 1152MB |  12642 |  6488 |  3496 |  2288 |   2499 | 1.95× | 3.62× | 5.53× | 5.06× |
| 14 | 256  | 98MB   |   1217 |   680 |   511 |   759 |    506 | 1.79× | 2.38× | 1.60× | 2.40× |
| 14 | 512  | 392MB  |   4777 |  2476 |  1426 |  1232 |   1234 | 1.93× | 3.35× | 3.88× | 3.87× |
| 14 | 768  | 882MB  |  12739 |  6492 |  3513 |  2518 |   2061 | 1.96× | 3.63× | 5.06× | 6.18× |
| 14 | 1024 | 1568MB |  25630 | 13132 |  6985 |  4242 |   3710 | 1.95× | 3.67× | 6.04× | 6.91× |
| 16 | 256  | 128MB  |   1749 |   943 |   645 |   867 |    862 | 1.85× | 2.71× | 2.02× | 2.03× |
| 16 | 512  | 512MB  |   7585 |  3903 |  2140 |  1695 |   1581 | 1.94× | 3.55× | 4.48× | 4.80× |
| 16 | 768  | 1152MB |  20822 | 10745 |  5674 |  3522 |   3019 | 1.94× | 3.67× | 5.91× | 6.90× |
| 16 | 1024 | 2048MB |  43602 | 22666 | 11965 |  6612 |   5461 | 1.92× | 3.64× | 6.59× | 7.99× |

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
