#!/bin/bash
#SBATCH --job-name=BSC_iPEPS_bench
#SBATCH --output=%x_%j.out
#SBATCH --partition=acc
#SBATCH --time=02:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4
#SBATCH --account=ehpc670
#SBATCH --qos=acc_ehpc

# BSC MareNostrum5 — H100 64GB, 4 GPU/node, 2 nodes for 8 GPU.
# iPEPS fg benchmark at D=10 χ=400 across 1/2/4/8 GPUs.

JULIA=julia
MAX_GPU=${SLURM_NTASKS}

echo "=== BSC iPEPS fg Benchmark (D=10 χ=400) ==="
echo "Nodes: ${SLURM_NNODES}  Max GPUs: ${MAX_GPU}  Start: $(date)"

for N in 1 2 4 8; do
    [ $N -gt $MAX_GPU ] && continue
    echo ""
    echo "========== ${N} GPU =========="
    mpirun -np $N bash -c \
        "export CUDA_VISIBLE_DEVICES=\$OMPI_COMM_WORLD_LOCAL_RANK; \
         exec $JULIA --project=../../.. ../benchmarks/benchmark_fg.jl"
done

echo ""
echo "=== Done: $(date) ==="
