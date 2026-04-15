# MPI Multi-GPU Parallel iPEPS Optimization

Run iPEPS gradient optimization across multiple GPUs using MPI parallelism.

## Quick Start

```bash
# Edit submit.sh: uncomment your system's module load block, set account
sbatch submit.sh          # 4 GPUs on 1 node
sbatch --nodes=2 submit.sh  # 8 GPUs across 2 nodes

# Run config test first to verify setup
sbatch submit_test.sh
```

## Required Environment Variables

| Variable | Value | Why |
|----------|-------|-----|
| `CUDA_VISIBLE_DEVICES` | `$OMPI_COMM_WORLD_LOCAL_RANK` (BSC) or `$SLURM_LOCALID` (JSC) | Each process sees only its own GPU. Without this, CUDA IPC mappings waste ~33GB/GPU |
| `UCX_MEMTYPE_CACHE` | `n` | Disable UCX CUDA memory cache. Prevents cudaMalloc interception crash with Julia |

### System-specific

| Variable | BSC | JSC |
|----------|-----|-----|
| MPI launcher | `mpirun` | `srun` |
| Local rank var | `OMPI_COMM_WORLD_LOCAL_RANK` | `SLURM_LOCALID` |
| `JULIA_CUDA_USE_BINARYBUILDER` | not needed | `false` (NVHPC provides CUDA) |
| Modules | auto-loaded | `Stages/2026 NVHPC/25.9-CUDA-13 OpenMPI/5.0.8` |

## How It Works

The `forloop_iter` parameter splits the chi-dimension contraction into chunks. With MPI, each GPU processes `total_splits / nprocs` chunks in parallel, then results are gathered via custom `allgatherv_p2p!` (forward) and `allreduce_p2p!` (backward).

## Performance (D=10, chi=400, BSC H100 64GB)

| GPUs | Forward | fg (fwd+bwd) | Speedup |
|------|---------|-------------|---------|
| 1    | 85s     | 923s        | 1x      |
| 2    | 44s     | 555s        | 1.67x   |
| 4    | 23s     | 408s        | 2.26x   |
| 8    | 14s     | 354s        | 2.61x   |

## Parameters to Adjust

| Parameter | Description | Guidance |
|-----------|-------------|----------|
| `D` | iPEPS bond dimension | Problem-dependent |
| `chi` | Boundary bond dimension | Larger = more accurate, more memory |
| `total_splits` | Total forloop chunks | 32 is optimal; 128 for small chi |
| `ifcheckpoint` | AD checkpointing | `true` for D >= 8, chi >= 256 |
| `power_iter_obs` | ObsEnv power iterations | 40 default, reduce to save memory |

## Files

| File | Description |
|------|-------------|
| `MPI_parallel.jl` | Main optimization script |
| `submit.sh` | Slurm submit for optimization |
| `test_MPI_config.jl` | Configuration validation test |
| `submit_test.sh` | Slurm submit for test (1/2/4/8 GPU scaling) |
| `benchmarks/` | Benchmark results by system |
