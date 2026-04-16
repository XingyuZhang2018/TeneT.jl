# Cross-System Performance Comparison

Benchmark results across different supercomputers for MPI multi-GPU iPEPS optimization.

## Systems

| | BSC MareNostrum5 | JSC Jupiter |
|--|------------------|-------------|
| GPU | NVIDIA H100 64GB | NVIDIA GH200 120GB |
| Architecture | x86_64 | ARM aarch64 (Grace Hopper) |
| GPU/node | 4 | 4 |
| Intra-node | NVLink | NVLink |
| Inter-node | InfiniBand | InfiniBand |
| MPI | OpenMPI 4.1.5 | OpenMPI 5.0.8 |
| UCX | 1.16.0 | default |
| Julia | 1.11.3 | 1.11.1 |
| CUDA libs | CUDA.jl artifacts | CUDA.jl artifacts (strip NVHPC math_libs) |

## Full fg Benchmark (D=10 χ=400, Plaquette VUMPS)

### Forward (leading_boundary only)

| GPU | BSC H100 | JSC GH200 | JSC/BSC |
|-----|----------|-----------|---------|
| 1   | 85.2s    | 59.2s     | **1.44x faster** |
| 2   | 44.1s    | 31.1s     | **1.42x faster** |
| 4   | 23.4s    | 19.5s     | **1.20x faster** |
| 8   | 13.6s    | 11.8s     | **1.15x faster** |

### Full fg (forward + backward)

| GPU | BSC H100 | JSC GH200 | JSC/BSC | BSC speedup | JSC speedup |
|-----|----------|-----------|---------|-------------|-------------|
| 1   | 923s     | 831s      | **1.11x** | 1x        | 1x          |
| 2   | 555s     | 457s      | **1.21x** | 1.66x     | **1.82x**   |
| 4   | 408s     | 324s      | **1.26x** | 2.26x     | **2.56x**   |
| 8   | 354s     | 231s      | **1.53x** | 2.61x     | **3.60x**   |

### Key Observations

- GH200 forward is 15-44% faster than H100 (higher memory bandwidth: 4TB/s vs 3.35TB/s)
- GH200 multi-GPU fg scaling is significantly better (3.60x vs 2.61x at 8 GPU)
- The gap widens with more GPUs: 1-GPU 11% faster → 8-GPU 53% faster
- GH200 backward benefits from larger memory (120GB vs 64GB, no checkpoint memory pressure)

## FLmap_parallel Forward Comparison (selected sizes)

| D | χ | BSC H100 1GPU | JSC GH200 1GPU | JSC/BSC | BSC 8GPU | JSC 8GPU | JSC/BSC 8GPU |
|---|---|---------------|----------------|---------|----------|----------|--------------|
| 8 | 512 | 356ms | 243ms | 1.47x | 55ms | 50ms | 1.10x |
| 10 | 512 | 682ms | 447ms | 1.53x | 100ms | 75ms | 1.33x |
| 10 | 1024 | 3558ms | 2552ms | 1.39x | 624ms | 375ms | 1.66x |
| 12 | 512 | 1151ms | 812ms | 1.42x | 168ms | 132ms | 1.27x |
| 12 | 1024 | 5858ms | 4541ms | 1.29x | 850ms | 692ms | 1.23x |

## MPI Collectives Comparison (p2p, 128MB)

| | BSC 2GPU | JSC 2GPU | BSC 4GPU | JSC 4GPU | BSC 8GPU | JSC 8GPU |
|--|---------|---------|---------|---------|---------|---------|
| Allgatherv | 2.89ms | 2.72ms | 5.30ms | 6.66ms | 18.1ms | 13.2ms |
| Allreduce | 7.66ms | 10.77ms | 19.3ms | 21.6ms | 29.6ms | 37.3ms |

- Allgatherv: JSC faster at 8 GPU (better IB bandwidth)
- Allreduce: BSC faster (UCX 1.16 vs JSC UCX default, different tree topology)

## Environment Configuration Summary

| Setting | BSC | JSC |
|---------|-----|-----|
| `CUDA_VISIBLE_DEVICES` | `$OMPI_COMM_WORLD_LOCAL_RANK` | `$SLURM_LOCALID` |
| `UCX_MEMTYPE_CACHE` | `n` (auto by MPI.jl) | `n` (must set explicitly) |
| `UCX_TLS` | default | `rc_x,self,sm,cuda_copy` |
| `CUDA_LAUNCH_BLOCKING` | not needed | `1` (ARM sync worker bug) |
| LD_LIBRARY_PATH | clean (no NVHPC) | strip NVHPC math_libs |
| MPI launcher | `mpirun` | `srun` |
| `using` order | any | **CUDA before MPI** |
