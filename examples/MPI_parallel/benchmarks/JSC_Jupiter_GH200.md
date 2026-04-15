# JSC Jupiter Benchmark Results

- **Date**: 2026-04-15
- **GPU**: NVIDIA GH200 120GB (Grace Hopper, ARM aarch64)
- **Interconnect**: NVLink (intra-node), InfiniBand (inter-node)
- **Software**: OpenMPI 5.0.8, NVHPC 25.9, CUDA 13, Julia 1.11.1, UCX
- **Config**: `CUDA_VISIBLE_DEVICES=$SLURM_LOCALID`, `UCX_MEMTYPE_CACHE=n`, `UCX_TLS=rc_x,self,sm,cuda_copy`, `CUDA_LAUNCH_BLOCKING=1`

## Part 1: MPI Collectives (allgatherv_p2p! / allreduce_p2p!)

### Allgatherv

| Size | 2 GPU | 4 GPU | 8 GPU (2 nodes) |
|------|-------|-------|-----------------|
| 8KB  | 0.05ms | 0.12ms | 0.19ms |
| 8MB  | 0.23ms (33.5 GB/s) | 0.55ms (13.8 GB/s) | 0.85ms (9.0 GB/s) |
| 30MB | 0.72ms (42.1 GB/s) | 1.72ms (17.7 GB/s) | 3.81ms (8.0 GB/s) |
| 122MB | 2.72ms (44.8 GB/s) | 6.67ms (18.3 GB/s) | 12.29ms (9.9 GB/s) |
| 381MB | 8.40ms (45.4 GB/s) | 20.83ms (18.3 GB/s) | 42.51ms (9.0 GB/s) |

### Allreduce

| Size | 2 GPU | 4 GPU | 8 GPU (2 nodes) |
|------|-------|-------|-----------------|
| 8KB  | 0.12ms | 0.30ms | 0.31ms |
| 8MB  | 0.79ms (9.7 GB/s) | 1.71ms (4.5 GB/s) | 3.05ms (2.5 GB/s) |
| 30MB | 2.79ms (10.9 GB/s) | 6.11ms (5.0 GB/s) | 9.98ms (3.1 GB/s) |
| 122MB | 10.76ms (11.3 GB/s) | 21.55ms (5.7 GB/s) | 37.27ms (3.3 GB/s) |
| 381MB | 33.50ms (11.4 GB/s) | 67.59ms (5.6 GB/s) | 115.99ms (3.3 GB/s) |

## Part 2: FLmap_parallel Forward

| D | χ | Size | 1 GPU | 2 GPU | 4 GPU | 8 GPU | 2x speedup | 4x speedup | 8x speedup |
|---|---|------|-------|-------|-------|-------|-------------|-------------|-------------|
| 8 | 256 | 32MB | 127ms | 59ms | 42ms | 18ms | 2.15x | 3.02x | 7.06x |
| 8 | 512 | 128MB | 243ms | 120ms | 75ms | 52ms | 2.03x | 3.24x | 4.67x |
| 8 | 1024 | 512MB | 1064ms | 536ms | 294ms | 174ms | 1.99x | 3.62x | 6.12x |
| 10 | 256 | 50MB | 128ms | 120ms | 95ms | 73ms | 1.07x | 1.35x | 1.75x |
| 10 | 512 | 200MB | 423ms | 238ms | 133ms | 77ms | 1.78x | 3.18x | 5.49x |
| 10 | 1024 | 800MB | 2503ms | 1386ms | 824ms | 390ms | 1.81x | 3.04x | 6.42x |
| 12 | 256 | 72MB | 180ms | 110ms | 70ms | 35ms | 1.64x | 2.57x | 5.14x |
| 12 | 512 | 288MB | 791ms | 422ms | 227ms | 126ms | 1.87x | 3.49x | 6.28x |
| 12 | 1024 | 1152MB | 4529ms | 2356ms | 1205ms | 686ms | 1.92x | 3.76x | 6.60x |

## Part 2: FLmap_parallel Backward

| D | χ | Size | 1 GPU | 2 GPU | 4 GPU | 8 GPU | 2x speedup | 4x speedup | 8x speedup |
|---|---|------|-------|-------|-------|-------|-------------|-------------|-------------|
| 8 | 256 | 32MB | 539ms | 345ms | 300ms | 250ms | 1.56x | 1.80x | 2.16x |
| 8 | 512 | 128MB | 835ms | 504ms | 534ms | 721ms | 1.66x | 1.56x | 1.16x |
| 8 | 1024 | 512MB | 3197ms | 1735ms | 1286ms | 1321ms | 1.84x | 2.49x | 2.42x |
| 10 | 256 | 50MB | 408ms | 396ms | 435ms | 433ms | 1.03x | 0.94x | 0.94x |
| 10 | 512 | 200MB | 1442ms | 932ms | 785ms | 930ms | 1.55x | 1.84x | 1.55x |
| 10 | 1024 | 800MB | 8136ms | 4430ms | 2667ms | 2307ms | 1.84x | 3.05x | 3.53x |
| 12 | 256 | 72MB | 657ms | 504ms | 493ms | 716ms | 1.30x | 1.33x | 0.92x |
| 12 | 512 | 288MB | 2775ms | 1606ms | 1111ms | 1168ms | 1.73x | 2.50x | 2.38x |
| 12 | 1024 | 1152MB | 16724ms | 9004ms | 5140ms | 3650ms | 1.86x | 3.25x | 4.58x |

## Cross-system Comparison (122MB, p2p collectives)

| Metric | BSC H100 2GPU | JSC GH200 2GPU | BSC H100 8GPU | JSC GH200 8GPU |
|--------|---------------|----------------|---------------|----------------|
| Allgatherv | 2.89ms | 2.72ms | 18.1ms | 12.3ms |
| Allreduce | 7.66ms | 10.76ms | 29.6ms | 37.3ms |

## JSC-specific Environment

```bash
module load Stages/2026 NVHPC/25.9-CUDA-13 OpenMPI/5.0.8
export JULIA_CUDA_USE_BINARYBUILDER=false
export UCX_MEMTYPE_CACHE=n
export UCX_TLS=rc_x,self,sm,cuda_copy    # no cuda_ipc, no gdr_copy (incompatible with CVD)
export CUDA_LAUNCH_BLOCKING=1             # avoid CUDA.jl sync worker segfault on ARM
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
```

## Known Issues

- **CUDA_LAUNCH_BLOCKING=1 required**: Without it, CUDA.jl `synchronization_worker` thread segfaults on GH200 ARM after large GPU buffer operations. ~5% performance overhead.
- **UCX_TLS must exclude cuda_ipc and gdr_copy**: Both fail with `CUDA_VISIBLE_DEVICES` isolation (`invalid permissions` / `gdr_pin_buffer failed`).
- **GPFS O_TMPFILE**: Julia Pkg fails with `Unknown system error -122` on GPFS. First-time workaround: `JULIA_DEPOT_PATH=/tmp/julia_depot:$HOME/.julia`.
- **Backward scaling limited for small tensors**: D=10 χ=256 backward shows no speedup on multi-GPU due to MPI communication overhead dominating small tensor computation.
