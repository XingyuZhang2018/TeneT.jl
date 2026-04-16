# MPI Multi-GPU Parallel iPEPS Optimization

Run iPEPS gradient optimization across multiple GPUs using MPI parallelism.

## Directory Structure

```
MPI_parallel/
├── MPI_parallel.jl         # Main optimization script (shared)
├── test_MPI_config.jl      # Configuration validation test (shared)
├── BSC/                    # BSC MareNostrum5 (H100 64GB)
│   ├── submit.sh           # Slurm submit for optimization
│   ├── submit_test.sh      # Slurm submit for scaling test
│   └── LocalPreferences.toml
├── JSC/                    # JSC Jupiter (GH200 120GB)
│   ├── submit.sh
│   ├── submit_test.sh
│   └── LocalPreferences.toml
├── benchmarks/             # Benchmark results
│   ├── BSC_MareNostrum5_H100.md
│   ├── JSC_Jupiter_GH200.md
│   └── cross_system_comparison.md
└── README.md
```

## Quick Start

```bash
# 1. Copy LocalPreferences.toml to project root
cp BSC/LocalPreferences.toml ../../LocalPreferences.toml   # or JSC/

# 2. Run scaling test
cd BSC   # or JSC
sbatch submit_test.sh

# 3. Run optimization
sbatch submit.sh
```

## Setup for a New System

1. Create a new directory (e.g., `LUMI/`)
2. Write `submit.sh` and `submit_test.sh` with system-specific:
   - Module loads
   - MPI launcher (`mpirun` vs `srun`)
   - Local rank variable (`OMPI_COMM_WORLD_LOCAL_RANK` vs `SLURM_LOCALID`)
   - Environment variables (see below)
3. Create `LocalPreferences.toml` with MPI binary config
4. Copy to project root: `cp LUMI/LocalPreferences.toml ../../`
5. Run `submit_test.sh` to validate
6. Save results to `benchmarks/LUMI_MI250X.md`

## Required Environment Variables

| Variable | Purpose | BSC | JSC |
|----------|---------|-----|-----|
| `CUDA_VISIBLE_DEVICES` | GPU isolation (critical) | `$OMPI_COMM_WORLD_LOCAL_RANK` | `$SLURM_LOCALID` |
| `UCX_MEMTYPE_CACHE` | Disable CUDA memory cache | `n` (auto) | `n` (explicit) |
| `UCX_TLS` | UCX transport selection | default | `rc_x,self,sm,cuda_copy` |
| `UCX_WARN_UNUSED_ENV_VARS` | Suppress warnings | not needed | `n` |
| `CUDA_LAUNCH_BLOCKING` | Sync GPU ops | not needed | `1` (ARM bug) |
| `LD_LIBRARY_PATH` | CUDA lib path | clean | strip NVHPC math_libs |

## Julia Import Order

**`using CUDA` must come before `using MPI`** on systems where NVHPC CUDA paths are stripped from `LD_LIBRARY_PATH`. CUDA.jl artifact loads `libcudart`, which `libmpi` depends on.

## Performance (D=10 χ=400)

| GPU | BSC H100 fg | JSC GH200 fg |
|-----|-------------|--------------|
| 1   | 923s        | 831s         |
| 2   | 555s        | 457s         |
| 4   | 408s        | 324s         |
| 8   | 354s        | 231s         |

See `benchmarks/cross_system_comparison.md` for detailed comparison.
