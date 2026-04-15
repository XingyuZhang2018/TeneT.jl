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

# ─── Configuration ──────────────────────────────────────────────────
# Number of GPUs to use (adjust --nodes and --ntasks-per-node above accordingly)
#   1 node  = up to 4 GPUs (intra-node NVLink)
#   2 nodes = up to 8 GPUs (inter-node InfiniBand)
#   N nodes = up to 4N GPUs
NGPU=${SLURM_NTASKS}

# ─── Environment ────────────────────────────────────────────────────
# CRITICAL: Each MPI process must see only its own GPU.
# Without this, CUDA IPC mappings consume ~33GB per GPU, causing OOM.
# OMPI_COMM_WORLD_LOCAL_RANK gives the node-local rank (0,1,2,3).
# With CUDA_VISIBLE_DEVICES set, each process sees 1 GPU as device 0.
#
# NOTE: UCX_MEMTYPE_CACHE is already set to "n" by MPI.jl. Do NOT change it.
# NOTE: Do NOT set JULIA_CUDA_MEMORY_POOL — the default stream-ordered pool is fastest.

echo "=== iPEPS MPI Parallel ==="
echo "Nodes: ${SLURM_NNODES}  GPUs: ${NGPU}  GPUs/node: ${SLURM_NTASKS_PER_NODE}"
echo "Job ID: ${SLURM_JOB_ID}  Start: $(date)"

mpirun -np ${NGPU} bash -c \
    'export CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK; exec julia --project=../.. MPI_parallel.jl'

echo "=== Done: $(date) ==="
