#!/bin/bash
#SBATCH --job-name=iPEPS_MPI
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=acc
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4
# #SBATCH --account=<your_account>
# #SBATCH --qos=<your_qos>

# ─── Environment ────────────────────────────────────────────────────
# Required environment variables for MPI + CUDA + Julia:
#
# CUDA_VISIBLE_DEVICES: Each MPI process must see only its own GPU.
#   Without this, CUDA IPC mappings consume ~33GB per GPU, causing OOM.
#   BSC uses OMPI_COMM_WORLD_LOCAL_RANK, JSC uses SLURM_LOCALID.
#
# UCX_MEMTYPE_CACHE=n: Disable UCX CUDA memory type cache.
#   Required by MPI.jl to avoid cudaMalloc interception crashes.
#   Set explicitly because some systems (JSC) don't inherit from MPI.jl.
#
# JULIA_CUDA_USE_BINARYBUILDER=false: (Optional, JSC only)
#   Use system CUDA libraries instead of Julia artifacts.
#   Needed when NVHPC module provides CUDA and conflicts with CUDA.jl artifacts.
#
# JULIA_CUDA_MEMORY_POOL: Do NOT set. Default stream-ordered pool is fastest.

# ─── Module load (uncomment ONE block for your system) ──────────────
# BSC MareNostrum5:
#   (modules auto-loaded via .bashrc)

# JSC Jupiter:
#   module load Stages/2026 NVHPC/25.9-CUDA-13 OpenMPI/5.0.8
#   export JULIA_CUDA_USE_BINARYBUILDER=false
#   export UCX_TLS=rc_x,self,sm,gdr_copy,cuda_copy  # remove cuda_ipc (incompatible with CVD)

# ─── MPI launch ─────────────────────────────────────────────────────
NGPU=${SLURM_NTASKS}
# Detect local rank variable (OpenMPI vs Slurm)
CVD_VAR=OMPI_COMM_WORLD_LOCAL_RANK
command -v srun &>/dev/null && [ -z "$(command -v mpirun 2>/dev/null)" ] && CVD_VAR=SLURM_LOCALID

echo "=== iPEPS MPI Parallel ==="
echo "Nodes: ${SLURM_NNODES}  GPUs: ${NGPU}  GPUs/node: ${SLURM_NTASKS_PER_NODE}"
echo "Job ID: ${SLURM_JOB_ID}  Start: $(date)"

# Use mpirun (BSC) or srun (JSC) depending on availability
if command -v mpirun &>/dev/null; then
    mpirun -np ${NGPU} bash -c \
        "export CUDA_VISIBLE_DEVICES=\$${CVD_VAR}; export UCX_MEMTYPE_CACHE=n; exec julia --project=../.. MPI_parallel.jl"
else
    srun -n ${NGPU} --gpus-per-task=1 bash -c \
        "export CUDA_VISIBLE_DEVICES=\$${CVD_VAR}; export UCX_MEMTYPE_CACHE=n; exec julia --project=../.. MPI_parallel.jl"
fi

echo "=== Done: $(date) ==="
