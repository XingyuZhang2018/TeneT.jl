#!/bin/bash
#SBATCH --job-name=JSC_fg_nccl_compare
#SBATCH --output=%x_%j.out
#SBATCH --partition=booster
#SBATCH --time=00:30:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:4
#SBATCH --account=e-dev-2026d01-011

# JSC GH200 8-GPU benchmark_fg: NCCL on vs off.
# Hypothesis: enabling TENET_USE_NCCL=1 dramatically improves backward by
# replacing the 3-phase p2p ring (37 ms / 128 MB allreduce at 8 GPU per
# JSC_Jupiter_GH200.md Part 1) with a single ncclAllReduce that handles
# intra+inter node hierarchy in <2 ms.

module load Stages/2026 NVHPC/25.9-CUDA-13 OpenMPI/5.0.8 NCCL/default-CUDA-13

JULIA=$HOME/tools/julia-1.11.1/bin/julia

CLEAN_LD=$(echo $LD_LIBRARY_PATH | tr ":" "\n" | \
    grep -v "math_libs\|CUDA/13/targets\|CUDA/13/nvvm\|CUDA/13/extras\|CUDA/13/stubs" | \
    tr "\n" ":" | sed "s/:$//")

BASE_ENVS="export CUDA_VISIBLE_DEVICES=\$SLURM_LOCALID; \
export LD_LIBRARY_PATH=$CLEAN_LD; \
export UCX_MEMTYPE_CACHE=n; \
export UCX_TLS=rc_x,self,sm,cuda_copy; \
export UCX_WARN_UNUSED_ENV_VARS=n; \
export CUDA_LAUNCH_BLOCKING=1; \
export NCCL_DEBUG=WARN"

run_case() {
    label="$1"
    extra="$2"
    echo ""
    echo "========== ${label} =========="
    srun -n 8 --gpus-per-task=1 bash -c "$BASE_ENVS; $extra exec $JULIA --project=../../.. ../benchmark_fg.jl" 2>&1 \
        | grep -vE 'NCCL INFO'
}

echo "=== JSC 8-GPU benchmark_fg: NCCL on vs off ==="
echo "Start: $(date)"

echo "--- Precompile warmup ---"
srun -n 1 --gpus-per-task=1 bash -c "$BASE_ENVS; $JULIA --project=../../.. -e 'using TeneT, CUDA, MPI'" 2>&1 | tail -3

run_case "(A) NCCL OFF (3-phase ring, current default)" ""
run_case "(B) NCCL ON  (TENET_USE_NCCL=1)" "export TENET_USE_NCCL=1;"

echo ""
echo "=== Done: $(date) ==="
