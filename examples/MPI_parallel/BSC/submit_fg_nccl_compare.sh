#!/bin/bash
#SBATCH --job-name=BSC_fg_nccl_compare
#SBATCH --output=%x_%j.out
#SBATCH --partition=acc
#SBATCH --time=00:30:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4
#SBATCH --account=ehpc670
#SBATCH --qos=acc_ehpc

# BSC MN5 H100 8-GPU benchmark_fg: NCCL on vs off.
# Hypothesis: enabling TENET_USE_NCCL=1 dramatically improves backward by
# replacing the 3-phase p2p ring (~30 ms / 128 MB allreduce at 8 GPU per
# cross_system_comparison.md) with a single ncclAllReduce in <2 ms.

module load nccl/2.24.3-1

JULIA=julia

run_case() {
    label="$1"
    extra="$2"
    echo ""
    echo "========== ${label} =========="
    mpirun -np 8 bash -c \
        "export CUDA_VISIBLE_DEVICES=\$OMPI_COMM_WORLD_LOCAL_RANK; \
         export NCCL_DEBUG=WARN; \
         $extra exec $JULIA --project=../../.. ../benchmark_fg.jl" 2>&1 \
        | grep -vE 'NCCL INFO'
}

echo "=== BSC 8-GPU benchmark_fg: NCCL on vs off ==="
echo "Start: $(date)"

run_case "(A) NCCL OFF (3-phase ring, current default)" ""
run_case "(B) NCCL ON  (TENET_USE_NCCL=1)" "export TENET_USE_NCCL=1;"

echo ""
echo "=== Done: $(date) ==="
