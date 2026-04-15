# JSC Jupiter Benchmark Results

- **Date**: 2026-04-15
- **GPU**: NVIDIA GH200 120GB (Grace Hopper, ARM aarch64)
- **Interconnect**: NVLink (intra-node), InfiniBand (inter-node)
- **Software**: OpenMPI 5.0.8, NVHPC 25.9, CUDA 13, Julia 1.11.1, UCX
- **Config**: `CUDA_VISIBLE_DEVICES=$SLURM_LOCALID`, `UCX_MEMTYPE_CACHE=n`, `UCX_TLS=rc_x,self,sm,cuda_copy`
- **Note**: `JULIA_CUDA_USE_BINARYBUILDER=false` uses system CUDA libs from NVHPC module

## Part 1: MPI Collectives (allgatherv_p2p! / allreduce_p2p!)

### Allgatherv

| Size | 2 GPU | 4 GPU | 8 GPU (2 nodes) |
|------|-------|-------|-----------------|
| 2MB  | 0.66ms (2.9 GB/s) | 1.14ms (1.7 GB/s) | 0.81ms (2.4 GB/s) |
| 8MB  | 0.22ms (34.8 GB/s) | 0.50ms (15.3 GB/s) | 0.78ms (9.8 GB/s) |
| 30MB | 0.72ms (42.1 GB/s) | 1.72ms (17.7 GB/s) | 3.81ms (8.0 GB/s) |
| 122MB | 2.73ms (44.7 GB/s) | 7.72ms (15.8 GB/s) | 12.89ms (9.5 GB/s) |
| 381MB | 8.40ms (45.4 GB/s) | 20.83ms (18.3 GB/s) | 42.51ms (9.0 GB/s) |

### Allreduce

| Size | 2 GPU | 4 GPU | 8 GPU (2 nodes) |
|------|-------|-------|-----------------|
| 2MB  | 2.73ms (0.7 GB/s) | 3.50ms (0.6 GB/s) | 0.87ms (2.2 GB/s) |
| 8MB  | 0.77ms (9.9 GB/s) | 2.04ms (3.7 GB/s) | 2.57ms (3.0 GB/s) |
| 30MB | 2.79ms (10.9 GB/s) | 6.11ms (5.0 GB/s) | 9.98ms (3.1 GB/s) |
| 122MB | 10.81ms (11.3 GB/s) | 21.98ms (5.6 GB/s) | 37.33ms (3.3 GB/s) |
| 381MB | 33.50ms (11.4 GB/s) | 67.59ms (5.6 GB/s) | 115.99ms (3.3 GB/s) |

## Part 2: FLmap_parallel (1 GPU only, multi-GPU blocked by CUDA.jl ARM bug)

| D | χ | Size | Forward | Backward | bwd/fwd |
|---|---|------|---------|----------|---------|
| 8 | 256 | 32MB | 103ms | 420ms | 4.1x |
| 8 | 512 | 128MB | 193ms | 577ms | 3.0x |
| 10 | 256 | 50MB | 123ms | 407ms | 3.3x |
| 10 | 512 | 200MB | 422ms | 1442ms | 3.4x |
| 10 | 1024 | 800MB | 2503ms | 8136ms | 3.3x |
| 12 | 256 | 72MB | 180ms | 657ms | 3.7x |
| 12 | 512 | 288MB | 791ms | 2775ms | 3.5x |
| 12 | 1024 | 1152MB | 4529ms | 16724ms | 3.7x |

## Cross-system Comparison (122MB, p2p)

| Metric | BSC H100 2GPU | JSC GH200 2GPU | BSC H100 8GPU | JSC GH200 8GPU |
|--------|---------------|----------------|---------------|----------------|
| Allgatherv | 2.89ms | 2.73ms | 18.1ms | 12.9ms |
| Allreduce | 7.66ms | 10.81ms | 29.6ms | 37.3ms |

## Known Issues

- **χ=1024 segfault (1 GPU)**: cuTENSOR crashes on GH200 ARM with large tensors when using NVHPC system CUDA libs (`JULIA_CUDA_USE_BINARYBUILDER=false`). Works with `=true` for some sizes but not all.
- **Multi-GPU CUDA.jl crash**: `synchronization_worker` thread segfaults after large p2p transfers complete on 4+ GPUs. Data is correct before crash. Likely CUDA.jl ARM compatibility issue.
- **UCX_TLS**: Must exclude `cuda_ipc` and `gdr_copy` when using `CUDA_VISIBLE_DEVICES` isolation. Both fail with `invalid permissions` or `gdr_pin_buffer failed`.
- **GPFS O_TMPFILE**: Julia's Pkg on GPFS home directory fails with `Unknown system error -122`. Workaround: `JULIA_DEPOT_PATH=/tmp/julia_depot:$HOME/.julia` for first install only.

## JSC-specific Environment

```bash
module load Stages/2026 NVHPC/25.9-CUDA-13 OpenMPI/5.0.8
export JULIA_CUDA_USE_BINARYBUILDER=false
export UCX_MEMTYPE_CACHE=n
export UCX_TLS=rc_x,self,sm,cuda_copy    # no cuda_ipc, no gdr_copy
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
```
