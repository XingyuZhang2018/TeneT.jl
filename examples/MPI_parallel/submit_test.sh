#!/bin/bash
#SBATCH --job-name=MPI_test
#SBATCH --output=%x_%j.out
#SBATCH --partition=acc
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4
# #SBATCH --account=<your_account>
# #SBATCH --qos=<your_qos>

# Quick MPI configuration test: collectives + FLmap forward/backward
# Adjust --nodes and --ntasks-per-node for different GPU counts.

NGPU=${SLURM_NTASKS}

echo "=== MPI Configuration Test ==="
echo "Nodes: ${SLURM_NNODES}  GPUs: ${NGPU}  Start: $(date)"

mpirun -np ${NGPU} bash -c \
    'export CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK; exec julia --project=../.. test_MPI_config.jl'

echo "=== Done: $(date) ==="
