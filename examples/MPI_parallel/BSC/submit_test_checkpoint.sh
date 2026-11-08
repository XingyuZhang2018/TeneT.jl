#!/bin/bash
#SBATCH --job-name=MPI_ckpt_test
#SBATCH --output=%x_%j.out
#SBATCH --partition=acc
#SBATCH --time=00:30:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4
#SBATCH --account=ehpc670
#SBATCH --qos=acc_ehpc

# BSC MareNostrum5 — H100 64GB
# Probes checkpoint(Plain/Recompute/Offload, FLmap_parallel) under MPI multi-GPU.

JULIA=julia
MAX_GPU=${SLURM_NTASKS}

echo "=== BSC MareNostrum5 MPI checkpoint() Test ==="
echo "Nodes: ${SLURM_NNODES}  Max GPUs: ${MAX_GPU}  Start: $(date)"

for N in 1 2 4 8; do
    [ $N -gt $MAX_GPU ] && continue
    echo "========== ${N} GPU =========="
    mpirun -np $N bash -c \
        "export CUDA_VISIBLE_DEVICES=\$OMPI_COMM_WORLD_LOCAL_RANK; \
         exec $JULIA --project=../../.. ../validation/test_MPI_checkpoint.jl"
    echo ""
done

echo "=== Done: $(date) ==="
