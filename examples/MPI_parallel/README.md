# MPI Multi-GPU Parallel iPEPS Optimization

Run iPEPS gradient optimization across multiple GPUs using MPI parallelism.

## Directory Structure

```
MPI_parallel/
├── examples/               # Runnable optimization examples
│   ├── MPI_parallel.jl
│   └── J1J2_Square_VUMPS_Plaquette_slice2d.jl
├── validation/             # MPI validation and smoke-test scripts
│   ├── test_MPI_config.jl
│   ├── test_MPI_checkpoint.jl
│   ├── test_slice2d_sofia.jl
│   ├── test_slice2d_m3_sofia.jl
│   ├── test_slice2d_m4_sofia.jl
│   ├── test_slice2d_m5_sofia.jl
│   ├── run_test_slice2d_m3_sofia_cpu.jl
│   ├── run_test_slice2d_m4_sofia_cpu.jl
│   └── run_test_slice2d_m5_sofia_cpu.jl
├── benchmarks/             # Benchmark drivers and result notes
│   ├── benchmark_fg.jl
│   ├── benchmark_slice2d_maps_sofia.jl
│   ├── benchmark_slice2d_sofia.jl
│   ├── bench_chain_gate_m2_sofia.jl
│   ├── bench_chain_gate_sofia.jl
│   ├── bench_kernel_ab_sofia.jl
│   ├── BSC_MareNostrum5_H100.md
│   ├── JSC_Jupiter_GH200.md
│   ├── Sofia_VUB_H200.md
│   └── cross_system_comparison.md
├── BSC/                    # BSC MareNostrum5 (H100 64GB)
│   ├── submit.sh           # Slurm submit for optimization
│   ├── submit_test.sh      # Slurm submit for scaling test
│   └── LocalPreferences.toml
├── JSC/                    # JSC Jupiter (GH200 120GB)
│   ├── submit.sh
│   ├── submit_test.sh
│   ├── submit_test_checkpoint.sh
│   └── LocalPreferences.toml
├── Sofia/                  # Sofia VUB (H200 141GB, 8 GPU/node)
│   ├── submit.sh
│   ├── submit_test.sh
│   ├── submit_test_checkpoint.sh
│   ├── UCX_CUDA_ISSUE.md   # Debug log for LD_PRELOAD libcuda fix
│   └── LocalPreferences.toml
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

| Variable | Purpose | BSC | JSC | Sofia |
|----------|---------|-----|-----|-------|
| `CUDA_VISIBLE_DEVICES` | GPU isolation (critical) | `$OMPI_COMM_WORLD_LOCAL_RANK` | `$SLURM_LOCALID` | `$OMPI_COMM_WORLD_LOCAL_RANK` |
| `UCX_MEMTYPE_CACHE` | Disable CUDA memory cache | `n` (auto) | `n` (explicit) | `n` (explicit) |
| `UCX_TLS` | UCX transport selection | default | `rc_x,self,sm,cuda_copy` | `rc_x,self,sm,cuda_copy,cuda_ipc` |
| `UCX_WARN_UNUSED_ENV_VARS` | Suppress warnings | not needed | `n` | `n` |
| `CUDA_LAUNCH_BLOCKING` | Sync GPU ops | not needed | `1` (ARM bug) | `1` (H200 sync worker bug) |
| `LD_LIBRARY_PATH` | CUDA lib path | clean | strip NVHPC math_libs | EasyBuild stack |
| `LD_PRELOAD` | Shadow Julia artifact libcuda | not needed | not needed | `/usr/lib64/libcuda.so.1` (MANDATORY) |

## Julia Import Order

**`using CUDA` must come before `using MPI`** on systems where NVHPC CUDA paths are stripped from `LD_LIBRARY_PATH`. CUDA.jl artifact loads `libcudart`, which `libmpi` depends on.

## Performance (D=10 χ=400)

| GPU | BSC H100 fg | JSC GH200 fg | Sofia H200 fg |
|-----|-------------|--------------|---------------|
| 1   | 923s        | 831s         | **670s**      |
| 2   | 555s        | 457s         | **369s**      |
| 4   | 408s        | 324s         | **232s**      |
| 8   | 354s        | 231s         | **203s**      |

See `benchmarks/cross_system_comparison.md` for detailed comparison.
