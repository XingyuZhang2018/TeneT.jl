#!/bin/bash
#SBATCH --job-name=MPI_test
#SBATCH --output=%x_%j.out
#SBATCH --partition=acc
#SBATCH --time=00:40:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4
# #SBATCH --account=<your_account>
# #SBATCH --qos=<your_qos>

# MPI configuration test with multi-GPU scaling comparison.
# Tests collectives (allgatherv, allreduce) and FLmap forward/backward
# across 1, 2, 4, 8 GPUs (adjust --nodes for more).
#
# Usage:
#   sbatch submit_test.sh                          # 1/2/4/8 GPU (2 nodes)
#   sbatch --nodes=1 --gres=gpu:2 submit_test.sh   # 1/2 GPU only

CVD='export CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK; exec julia --project=../..'
SCRIPT=test_MPI_config.jl
MAX_GPU=${SLURM_NTASKS}

echo "=== MPI Scaling Test ==="
echo "Nodes: ${SLURM_NNODES}  Max GPUs: ${MAX_GPU}  Start: $(date)"
echo ""

for N in 1 2 4 8; do
    [ $N -gt $MAX_GPU ] && continue
    echo "========== ${N} GPU =========="
    mpirun -np $N bash -c "$CVD $SCRIPT"
    echo ""
done

echo "=== Done: $(date) ==="
