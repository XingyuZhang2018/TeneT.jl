# JSC Jupiter Benchmark Results

- **Date**: 2026-04-16
- **GPU**: NVIDIA GH200 120GB (Grace Hopper, ARM aarch64)
- **Interconnect**: NVLink (intra-node), InfiniBand (inter-node)
- **Software**: OpenMPI 5.0.8, NVHPC 25.9, CUDA 13, Julia 1.11.1, UCX
- **Config**: CUDA.jl artifacts (strip NVHPC math_libs from LD_LIBRARY_PATH), `CUDA_VISIBLE_DEVICES=$SLURM_LOCALID`, `UCX_MEMTYPE_CACHE=n`, `UCX_TLS=rc_x,self,sm,cuda_copy`, `CUDA_LAUNCH_BLOCKING=1`
- **Important**: `using CUDA` must come before `using MPI` in Julia scripts (artifact libcudart must load before libmpi)

## Full fg Benchmark (D=10 χ=400, Plaquette VUMPS, with checkpoint)

| GPU | Forward | fg (fwd+bwd) | fg Speedup |
|-----|---------|-------------|------------|
| 1   | 59.2s   | 831s        | 1x         |
| 2   | 31.1s   | 457s        | 1.82x      |
| 4   | 19.5s   | 324s        | 2.56x      |
| 8 (2 nodes) | 11.8s | 231s  | 3.60x      |

## Part 1: MPI Collectives (allgatherv_p2p! / allreduce_p2p!)

### Allgatherv

| Size | 2 GPU | 4 GPU | 8 GPU (2 nodes) |
|------|-------|-------|-----------------|
| 8KB  | 0.05ms | 0.11ms | 0.19ms |
| 8MB  | 0.23ms (33.5 GB/s) | 0.52ms (14.6 GB/s) | 0.85ms (8.9 GB/s) |
| 128MB | 2.72ms (44.9 GB/s) | 6.66ms (18.3 GB/s) | 13.19ms (9.3 GB/s) |

### Allreduce

| Size | 2 GPU | 4 GPU | 8 GPU (2 nodes) |
|------|-------|-------|-----------------|
| 8KB  | 0.12ms | 0.31ms | 0.29ms |
| 8MB  | 0.79ms (9.7 GB/s) | 1.73ms (4.4 GB/s) | 2.68ms (2.8 GB/s) |
| 128MB | 10.77ms (11.3 GB/s) | 21.55ms (5.7 GB/s) | 37.25ms (3.3 GB/s) |

## Part 2: FLmap_parallel Forward

| D | χ | Size | 1 GPU | 2 GPU | 4 GPU | 8 GPU | 2x speedup | 4x speedup | 8x speedup |
|---|---|------|-------|-------|-------|-------|-------------|-------------|-------------|
| 8 | 256 | 32MB | 128ms | 59ms | 42ms | 19ms | 2.17x | 3.05x | 6.84x |
| 8 | 512 | 128MB | 243ms | 122ms | 80ms | 50ms | 1.99x | 3.04x | 4.86x |
| 8 | 1024 | 512MB | 1037ms | 534ms | 294ms | 172ms | 1.94x | 3.53x | 6.03x |
| 10 | 256 | 50MB | 201ms | 128ms | 93ms | 76ms | 1.57x | 2.16x | 2.64x |
| 10 | 512 | 200MB | 447ms | 232ms | 132ms | 75ms | 1.93x | 3.39x | 5.96x |
| 10 | 1024 | 800MB | 2552ms | 1380ms | 822ms | 375ms | 1.85x | 3.10x | 6.81x |
| 12 | 256 | 72MB | 217ms | 120ms | 72ms | 35ms | 1.81x | 3.01x | 6.20x |
| 12 | 512 | 288MB | 812ms | 415ms | 228ms | 132ms | 1.96x | 3.56x | 6.15x |
| 12 | 1024 | 1152MB | 4541ms | 2295ms | 1191ms | 692ms | 1.98x | 3.81x | 6.56x |
| 14 | 256 | 98MB | 525ms | 206ms | 109ms | 56ms | 2.55x | 4.82x | 9.38x |
| 14 | 512 | 392MB | 1625ms | 825ms | 428ms | 237ms | 1.97x | 3.80x | 6.86x |
| 14 | 1024 | 1568MB | 8974ms | 4600ms | 2449ms | 1493ms | 1.95x | 3.66x | 6.01x |
| 16 | 256 | 128MB | 560ms | 282ms | 153ms | 80ms | 1.99x | 3.66x | 7.00x |
| 16 | 512 | 512MB | 2559ms | 1290ms | 667ms | 361ms | 1.98x | 3.84x | 7.09x |
| 16 | 1024 | 2048MB | 14790ms | 7532ms | 3961ms | 2259ms | 1.96x | 3.73x | 6.55x |

## Part 2: FLmap_parallel Backward

| D | χ | Size | 1 GPU | 2 GPU | 4 GPU | 8 GPU | 2x speedup | 4x speedup | 8x speedup |
|---|---|------|-------|-------|-------|-------|-------------|-------------|-------------|
| 8 | 256 | 32MB | 561ms | 347ms | 304ms | 242ms | 1.62x | 1.85x | 2.32x |
| 8 | 512 | 128MB | 835ms | 512ms | 535ms | 715ms | 1.63x | 1.56x | 1.17x |
| 8 | 1024 | 512MB | 3139ms | 1724ms | 1202ms | 1330ms | 1.82x | 2.61x | 2.36x |
| 10 | 256 | 50MB | 664ms | 399ms | 418ms | 425ms | 1.66x | 1.59x | 1.56x |
| 10 | 512 | 200MB | 1658ms | 915ms | 756ms | 916ms | 1.81x | 2.19x | 1.81x |
| 10 | 1024 | 800MB | 8347ms | 4390ms | 2623ms | 2263ms | 1.90x | 3.18x | 3.69x |
| 12 | 256 | 72MB | 899ms | 506ms | 491ms | 725ms | 1.78x | 1.83x | 1.24x |
| 12 | 512 | 288MB | 2980ms | 1594ms | 1111ms | 1134ms | 1.87x | 2.68x | 2.63x |
| 12 | 1024 | 1152MB | 16940ms | 8771ms | 4985ms | 3622ms | 1.93x | 3.40x | 4.68x |
| 14 | 256 | 98MB | 1555ms | 859ms | 665ms | 789ms | 1.81x | 2.34x | 1.97x |
| 14 | 512 | 392MB | 6205ms | 3231ms | 1903ms | 1663ms | 1.92x | 3.26x | 3.73x |
| 14 | 1024 | 1568MB | 34643ms | 17883ms | 9741ms | 6253ms | 1.94x | 3.56x | 5.54x |
| 16 | 256 | 128MB | 2237ms | 1175ms | 856ms | 904ms | 1.90x | 2.61x | 2.47x |
| 16 | 512 | 512MB | 9829ms | 5049ms | 2965ms | 2193ms | 1.95x | 3.31x | 4.48x |
| 16 | 1024 | 2048MB | 85680ms | 48034ms | 23759ms | 13923ms | 1.78x | 3.61x | 6.15x |

## Part 3: checkpoint() Method Comparison (FLmap_parallel, 2026-04-21)

Gradient-equivalence + wall-clock cost for the unified `CheckpointMethod`
singletons (`Plain()` / `Recompute()` / `Offload()`) wrapping `FLmap_parallel`
under MPI multi-GPU (job `384613`, `test_MPI_checkpoint.jl`).

**Correctness (all 16 configs = 4 sizes × 4 GPU counts)**:
- All three methods produce **identical gradients**: `‖g_method - g_plain‖ / ‖g_plain‖ < 1e-10`.
- **No MPI deadlock** in any run — Offload's host-copy + device-restore
  brackets around FLmap_parallel's internal allgatherv / allreduce
  collectives execute correctly.
- Offload gradient lands on the correct per-rank `CuArray`
  (not rank-0's device).

Times below are **backward-pass milliseconds** (forward is bit-identical
across methods: only adjoints differ, primals are identical). D≥8 only —
D=4/6 hit a cuTENSOR JIT edge case on JSC (job 384563 CUTENSOR_STATUS_INVALID_VALUE).

### 1 GPU
| D | χ | Plain | Recompute | Offload | Off/Plain |
|---|---|-------|-----------|---------|-----------|
| 8  | 128 |  462 ms |  551 ms |  556 ms | 1.20× |
| 8  | 256 |  616 ms |  769 ms |  775 ms | 1.26× |
| 10 | 256 |  884 ms | 1023 ms | 1035 ms | 1.17× |
| 10 | 512 | 1888 ms | 2308 ms | 2805 ms | 1.49× |

### 2 GPU
| D | χ | Plain | Recompute | Offload | Off/Plain |
|---|---|-------|-----------|---------|-----------|
| 8  | 128 |  296 ms |  343 ms |  345 ms | 1.17× |
| 8  | 256 |  326 ms |  405 ms |  410 ms | 1.26× |
| 10 | 256 |  509 ms |  626 ms |  637 ms | 1.25× |
| 10 | 512 | 1078 ms | 1327 ms | 1386 ms | 1.29× |

### 4 GPU
| D | χ | Plain | Recompute | Offload | Off/Plain |
|---|---|-------|-----------|---------|-----------|
| 8  | 128 |  307 ms |  382 ms |  327 ms | 1.07× |
| 8  | 256 |  332 ms |  396 ms |  412 ms | 1.24× |
| 10 | 256 |  484 ms |  574 ms |  591 ms | 1.22× |
| 10 | 512 | 1167 ms | 1333 ms | 1476 ms | 1.26× |

### 8 GPU (2 nodes)
| D | χ | Plain | Recompute | Offload | Off/Plain |
|---|---|-------|-----------|---------|-----------|
| 8  | 128 |  245 ms |  277 ms |  275 ms | 1.12× |
| 8  | 256 |  262 ms |  359 ms |  380 ms | 1.45× |
| 10 | 256 |  505 ms |  695 ms |  699 ms | 1.38× |
| 10 | 512 | 1492 ms | 1809 ms | 1821 ms | 1.22× |

### Findings

- **Multi-GPU Offload is correct**: the concern that Offload's
  host↔device bracket might race with `FLmap_parallel`'s internal MPI
  collectives turned out unfounded. All 16 configs produce matching
  gradients; no hangs, no device-residency errors.
- **Offload overhead ≈ 1.1–1.5× vs Plain**, tracking Recompute closely.
  Extra cost comes from host-copy + copy-back of args (FL + ALu + ALd
  + M = 4× per-arg size). On GH200 with NVLink the transfer penalty
  stays under ~25% for most sizes.
- **Worst case D=10 χ=512 @ 1 GPU = 1.49×**: host allocator pressure
  (800 MB × 4 = 3.2 GB per backward) is likely the dominant cost.
  Gets better at 2/4/8 GPU because per-rank size shrinks (tiling).
- **Recommendation**: use `step_checkpoint=Offload()` on VRAM-constrained
  runs. Expect ~20-30% extra backward time for the VRAM savings
  described in the "Feature Comparison" section; correctness and
  MPI-safety verified up to 8 GPU / 2 nodes.

## JSC-specific Environment

```bash
module load Stages/2026 NVHPC/25.9-CUDA-13 OpenMPI/5.0.8

# Strip NVHPC CUDA math libs to use CUDA.jl artifacts (faster + no cuBLAS crash)
CLEAN_LD=$(echo $LD_LIBRARY_PATH | tr ":" "\n" | \
  grep -v "math_libs\|compilers/lib\|CUDA/13/targets\|CUDA/13/nvvm\|CUDA/13/extras\|CUDA/13/stubs" | \
  tr "\n" ":" | sed "s/:$//")

export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
export LD_LIBRARY_PATH=$CLEAN_LD
export UCX_MEMTYPE_CACHE=n
export UCX_TLS=rc_x,self,sm,cuda_copy
export UCX_WARN_UNUSED_ENV_VARS=n
export CUDA_LAUNCH_BLOCKING=1
```

## Known Issues

- **`using CUDA` before `using MPI`**: Required when stripping CUDA paths from LD_LIBRARY_PATH. CUDA.jl artifact loads libcudart, which libmpi depends on.
- **CUDA_LAUNCH_BLOCKING=1**: Required on GH200 ARM to avoid CUDA.jl `synchronization_worker` segfault.
- **UCX_TLS excludes cuda_ipc and gdr_copy**: Incompatible with `CUDA_VISIBLE_DEVICES` isolation.
- **NVHPC system cuBLAS crashes**: `JULIA_CUDA_USE_BINARYBUILDER=false` causes `CUBLAS_STATUS_INVALID_VALUE` in VUMPS. Use CUDA.jl artifacts instead.
- **GPFS O_TMPFILE**: First-time `Pkg.instantiate()` fails. Workaround: `JULIA_DEPOT_PATH=/tmp/julia_depot:$HOME/.julia`.

## Mixed-Precision Test (D=10 χ=400, 4 GPU)

| inner_etype | polish | Forward | fg | energy diff |
|-------------|--------|---------|-----|-------------|
| Float64     | —      | 17.1s   | 304s | baseline   |
| Float32     | 2      | 24.0s (+41%) | 330s (+8%) | 3e-14 |

**Finding**: Mixed-precision (Float32 inner_etype) is SLOWER on GH200, not faster.
- GH200 has high FP64 throughput (tensor cores), so Float32 speedup is minimal
- Per-call Float64↔Float32 conversion overhead exceeds computation savings
- Recommendation: use default Float64 on GH200/H100-class GPUs

Float32 mixed-precision may still benefit consumer GPUs or systems with
significantly lower FP64 throughput.

## Feature Comparison (D=10 χ=400, 4 GPU, Plaquette VUMPS, with checkpoint)

Exhaustive sweep of new features: `whole_vumps_etype` (whole-VUMPS Float32),
`ifoffload_eig` (fine-grain host-memory offload of eigen-solver states),
`ifoffload_step` (coarse-grain offload of full VUMPS step checkpoints).

| Config | Forward | fg | gnorm | Status |
|--------|---------|-----|-------|--------|
| baseline (Float64)    | 17.1s | 304s    | 0.01150 | ✓ reference |
| whole_f32             | 20s   | 271-275s | **1e18** | BROKEN (AD) |
| offload_eig           | 17s   | 302-304s | 0.01150 | ✓ zero overhead |
| offload_step          | 17s   | 318-326s | 0.01150 | ✓ 4-7% overhead |
| offload_both          | 17s   | 327-329s | 0.01150 | ✓ 8% overhead |
| whole_f32_offload     | 20s   | 283-286s | **1e18** | BROKEN (AD) |

**Findings**:
- **`ifoffload_eig` is free on GH200** — matches baseline 304s within noise.
  Pure VRAM savings, recommended whenever VRAM is constrained.
- **`ifoffload_step` costs 4-7%** for additional VRAM savings on top of `ifoffload_eig`.
  Worthwhile when pushing χ to the VRAM limit.
- **`ifoffload_both` costs ~8%** (stacks the two offload modes).
- **`whole_vumps_etype=Float32` has broken AD**: forward energy is correct to 1e-10,
  but backward `gnorm` explodes to ~1e18 (should be ~0.01150). Same bug appears in
  `whole_f32_offload`. Forward-only speed is ~19% faster (20s vs 17.1s is misleading;
  actually the f32 fwd is faster per-iter but whole fg is only marginally faster because
  AD backward is unusable). **Upstream bug — do not use until fixed.**
- **Recommendation on GH200**:
  - Default: Float64, no offload.
  - VRAM-constrained: enable `ifoffload_eig=true` (free).
  - Extreme VRAM-constrained: add `ifoffload_step=true` (4-7% slowdown).
  - Do NOT enable `whole_vumps_etype=Float32` until AD is fixed.

## Parallel-level Mixed-Precision (D=10 χ=400, Plaquette VUMPS, with checkpoint)

Branch `feat/parallel-level-mixed-precision` (commit `ee048bc`) moves the
`inner_etype` Float32 cast from every FLmap/ACmap kernel call to the
`parallel()` / `forloop()` boundary. Cast happens once per parallel() call;
MPI allgatherv/allreduce travel in Float32 (2× bandwidth).

| GPU | Prec | Forward | fg | gnorm | fg vs F64 |
|-----|------|---------|-----|-------|-----------|
| 1 | F64 | 59.8s | 849s | 0.011496 | — |
| 1 | F32 | 72.1s (+20%) | 833s | 0.011496 | **-2%** ✓ |
| 2 | F64 | 35.4s | 531s | 0.011496 | — |
| 4 | F64 | 17.0s | 304s | 0.011496 | — |
| 4 | F32 | 21.1s (+24%) | **298s** | 0.011496 | **-2%** ✓ |

**Comparison old kernel-level F32 vs new parallel-level F32 (4 GPU):**

| Approach | Forward | fg | AD |
|----------|---------|-----|-----|
| kernel-level (commit `a15c18f`) | 24.0s (+41%) | 330s (+8%) | ✓ |
| **parallel-level (commit `ee048bc`)** | **21.1s (+24%)** | **298s (-2%)** | ✓ |

**Findings**:
- **AD correctness preserved**: gnorm = 0.011496 matches F64 baseline
  to ~1e-8, unlike `whole_vumps_etype=Float32` which explodes to 1e18.
- **fg turns net-positive**: +8% overhead → -2% speedup by moving cast
  to boundary. Backward chain amortizes cast overhead better (longer
  rrule nesting + F32 MPI bandwidth savings on partial gradients).
- **Forward still slower than F64** (+24%) but significantly better
  than old kernel-level (+41%). GH200 FP64 tensor cores are fast enough
  that F32 compute savings are marginal; cast memory bandwidth is a
  real cost that can't fully hide.
- **Per parallel() call cast count**: old = 2 × forloop_iter (=64 at
  4 GPU, forloop_iter=32); new = 2. 32× reduction in casts.

**Recommendation**:
- For pure forward VUMPS (observable eval, no AD): keep F64.
- For fg optimization workloads: enable `inner_etype=Float32` +
  `inner_etype_final_steps=2` — 2% faster, gradient correct.
- `whole_vumps_etype=Float32` remains broken (upstream AD bug).

## Env-level Mixed-Precision + Deep Investigation (D=10 χ=400)

Commit `dcecb98` moves the cast **one more level up** — from `parallel()`
boundary to the env solvers (`leftenv`, `rightenv`, `ACenv`, `ACenv_plaq`).
Rationale: Plaquette envs drive multiple `parallel()` calls per invocation
(2 simple_eig × FLmap-wrapper Nj-loop + Nj-1 naked inner-loop = ~6 calls
per env at pattern=[1 3; 2 4]), so parallel-level cast casts the same
tensors repeatedly. Env-level cast collapses this to 1 cast per env.

### Probed `vumps_step` breakdowns (single iter, 10 medians)

**C4v** (1 parallel() per env, baseline):
| step | F64 ms | F32 ms | ratio | F32 saving |
|------|--------|--------|-------|-----------|
| leftenv_c4v | 213.36 | 167.54 | 0.785 | -46 ms |
| ACenv_c4v | 216.35 | 163.56 | 0.756 | -53 ms |
| TOTAL | 465.23 | 366.28 | **0.787** | -99 ms ✓ F32 快 21% |

**Plaquette with env-level cast** (dcecb98):
| step | F64 ms | F32 ms | ratio | F32 saving |
|------|--------|--------|-------|-----------|
| leftenv | 500.13 | 507.96 | 1.016 | +8 ms |
| ACenv_plaq | 502.53 | 496.51 | 0.988 | -6 ms |
| TOTAL | 1092.75 | 1094.43 | **1.002** | +2 ms ≈ 持平 |

C4v gets full 21% F32 speedup. Plaquette doesn't budge despite the
same kernel, with either parallel-level OR env-level cast.

### Root cause — cuTENSOR warm-cache fast-path

Single-call ACmap_parallel at production size (Real Float64, D=10 χ=400,
warmed up, 10× median):
| Variant | Time | Ratio |
|---------|------|-------|
| Real F64 (Test 3, pure baseline) | 101.89 ms | 1.000 |
| Real F64 + inner_etype=Float32 (parallel-cast, Test 1) | 84.20 ms | 0.826 |
| Real F32 + inner_etype=nothing (env-cast equiv, Test 2) | 82.81 ms | **0.813** |

**Real F32 IS 19% faster than Real F64 at the isolated-call level.**
The 22% Complex-F32 speedup seen in mbmpi_380627 generalizes to Real.

But in Plaquette env loops, F64 per-call averages to ~83 ms/call
(500ms / 6 calls) — exactly the F32 isolated time. **F64 in Plaquette's
sequential same-shape call pattern is already running at fast-path
speed**, leaving no headroom for F32.

Hypothesis (unverified): cuTENSOR JIT plan cache warms up over
consecutive same-shape calls in a stream. Isolated tests with per-call
`CUDA.synchronize()+MPI.Barrier()` may interrupt this, keeping F64 on
a colder path (~100 ms) while Plaquette's uninterrupted flow hits the
fast-path (~83 ms).

### Dead-end findings (documented for future reference)

- **CUDA_LAUNCH_BLOCKING=1 costs 15-18%** on both F64 and F32
  (removing it dropped F64 vumps_step from 1092 to 914 ms, F32 from 1094
  to 902 ms). Worth a separate follow-up to see if safe to disable.
- **Cast overhead is trivial (<1%)**: env-level cast only marginally
  beats parallel-level cast (1.002 vs 1.009 ratio), confirming the
  cast cost was never the bottleneck.
- **Zygote.checkpoint is innocent**: identical F32 speedup in C4v
  probed test with `ifcheckpoint` on and off.
- **A bug in a preliminary diagnostic**: casting Real Float64 →
  ComplexF32 via `ComplexF32.(…)` inflates flops 2×, producing a
  spurious 86% F32 slowdown. The correct `_downcast_eltype` path
  preserves realness and was always operating correctly.

### Recommendation

- Keep `dcecb98` env-level cast implementation as the preferred F32
  pathway — cleanest code and marginally better than parallel-level.
  Benefits are hardware-dependent; other systems (H100, A100, consumer
  GPUs) may still see speedup when F64 hasn't saturated the fast-path.
- On **GH200 specifically** for Plaquette J1J2 VUMPS: **F32 mixed-precision
  does not accelerate wall-clock time**. F64 already runs fast. Prefer
  default Float64 + offload_eig for VRAM savings.
- **Can we drop CUDA_LAUNCH_BLOCKING=1 for the 15-18% speedup?**
  **No.** Tested at 4-GPU with `test_MPI_config.jl` (job 381094):
  - MPI collectives (allgatherv_p2p!, allreduce_p2p!) **pass correctness**
    and run at similar speed (~0-5% difference, not the 15-18% hoped for).
  - FLmap forward D=14 χ=256 **fails correctness check** (result ≠ serial
    reference), followed by `synchronization_worker` segfault:
    ```
    jlcapi_synchronization_worker_16635 ...
    CUDA/Il00B/lib/cudadrv/synchronization.jl:119
    ```
  - This is the documented GH200 ARM CUDA.jl bug: without CLB=1, kernel
    results race with downstream readers at large tensor sizes.
  - Shorter (~30-60s) microbenches do not trigger the segfault, but FLmap
    at production shapes (D≥14, χ≥256) does.
  - **Conclusion**: CLB=1 is a mandatory ~15-18% tax on GH200 ARM. It
    guarantees FLmap kernel correctness and prevents segfault during long
    runs. Revisit when CUDA.jl fixes the ARM synchronization_worker bug.

