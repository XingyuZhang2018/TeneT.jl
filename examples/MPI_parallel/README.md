# MPI Multi-GPU Parallel iPEPS Optimization

Run iPEPS gradient optimization across multiple GPUs using MPI parallelism.

## Quick Start

```bash
# Single GPU (no MPI needed)
julia --project=../.. MPI_parallel.jl

# 2 GPUs on one node
sbatch --ntasks-per-node=2 --gres=gpu:2 submit.sh

# 4 GPUs on one node
sbatch submit.sh

# 8 GPUs across 2 nodes
sbatch --nodes=2 submit.sh
```

## How It Works

The `forloop_iter` parameter splits the chi-dimension contraction into chunks. With MPI, each GPU processes `total_splits / nprocs` chunks in parallel, then results are gathered via custom `allgatherv_p2p!` (forward) and `allreduce_p2p!` (backward).

## Performance (D=10, chi=400, H100 64GB)

| GPUs | Forward | fg (fwd+bwd) | Speedup |
|------|---------|-------------|---------|
| 1    | 85s     | 923s        | 1x      |
| 2    | 44s     | 555s        | 1.67x   |
| 4    | 23s     | 408s        | 2.26x   |
| 8    | 14s     | 354s        | 2.61x   |

## Key Environment Variable

```bash
export CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK
```

This is **required** for multi-GPU. Without it, CUDA IPC mappings consume ~33GB per GPU, causing out-of-memory errors. The submit script handles this automatically.

## Parameters to Adjust

| Parameter | Description | Guidance |
|-----------|-------------|----------|
| `D` | iPEPS bond dimension | Problem-dependent |
| `chi` | Boundary bond dimension | Larger = more accurate, more memory |
| `total_splits` | Total forloop chunks | 128 works well for chi <= 512 |
| `ifcheckpoint` | AD checkpointing | `true` for D >= 8, chi >= 256 |
| `power_iter_obs` | ObsEnv power iterations | 40 default, reduce to save memory |
