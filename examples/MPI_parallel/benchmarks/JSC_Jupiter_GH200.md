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
