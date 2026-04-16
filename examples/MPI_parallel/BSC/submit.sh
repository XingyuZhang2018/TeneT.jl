#!/bin/bash
#SBATCH --job-name=iPEPS_MPI
#SBATCH --output=%x_%j.out
#SBATCH --partition=acc
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4
#SBATCH --account=ehpc670
#SBATCH --qos=acc_ehpc

# BSC MareNostrum5 — H100 64GB
# Modules auto-loaded via .bashrc (OpenMPI 4.1.5, UCX 1.16.0)

NGPU=${SLURM_NTASKS}
JULIA=julia

echo "=== iPEPS MPI Parallel (BSC) ==="
echo "Nodes: ${SLURM_NNODES}  GPUs: ${NGPU}  Start: $(date)"

mpirun -np ${NGPU} bash -c \
    "export CUDA_VISIBLE_DEVICES=\$OMPI_COMM_WORLD_LOCAL_RANK; \
     exec $JULIA --project=../../.. ../MPI_parallel.jl"

echo "=== Done: $(date) ==="
