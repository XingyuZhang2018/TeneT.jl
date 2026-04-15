#!/bin/bash
#SBATCH --job-name=MPI_test
#SBATCH --output=%x_%j.out
#SBATCH --partition=acc
#SBATCH --time=01:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4
# #SBATCH --account=<your_account>
# #SBATCH --qos=<your_qos>

# MPI configuration test with multi-GPU scaling comparison.
# Tests collectives (allgatherv, allreduce) and FLmap forward/backward
# across 1, 2, 4, 8 GPUs.
#
# Usage:
#   sbatch submit_test.sh                          # 1/2/4/8 GPU (2 nodes)
#   sbatch --nodes=1 --gres=gpu:2 submit_test.sh   # 1/2 GPU only

# ─── Module load (uncomment ONE block for your system) ──────────────
# BSC MareNostrum5:
#   (modules auto-loaded via .bashrc)

# JSC Jupiter:
#   module load Stages/2026 NVHPC/25.9-CUDA-13 OpenMPI/5.0.8
#   export JULIA_CUDA_USE_BINARYBUILDER=false

# ─── Detect launcher and local rank variable ────────────────────────
SCRIPT=test_MPI_config.jl
MAX_GPU=${SLURM_NTASKS}
CVD_VAR=OMPI_COMM_WORLD_LOCAL_RANK
LAUNCHER="mpirun"
if ! command -v mpirun &>/dev/null; then
    LAUNCHER="srun"
    CVD_VAR=SLURM_LOCALID
fi

echo "=== MPI Scaling Test ==="
echo "Nodes: ${SLURM_NNODES}  Max GPUs: ${MAX_GPU}  Launcher: ${LAUNCHER}  Start: $(date)"
echo ""

for N in 1 2 4 8; do
    [ $N -gt $MAX_GPU ] && continue
    echo "========== ${N} GPU =========="
    if [ "$LAUNCHER" = "mpirun" ]; then
        mpirun -np $N bash -c "export CUDA_VISIBLE_DEVICES=\$${CVD_VAR}; export UCX_MEMTYPE_CACHE=n; exec julia --project=../.. $SCRIPT"
    else
        srun -n $N --gpus-per-task=1 bash -c "export CUDA_VISIBLE_DEVICES=\$${CVD_VAR}; export UCX_MEMTYPE_CACHE=n; exec julia --project=../.. $SCRIPT"
    fi
    echo ""
done

echo "=== Done: $(date) ==="
