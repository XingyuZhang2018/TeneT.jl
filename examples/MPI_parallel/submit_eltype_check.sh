#!/bin/bash
#SBATCH --job-name=mbet
#SBATCH --output=%x_%j.out
#SBATCH --partition=booster
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:4
#SBATCH --account=e-dev-2026d01-011

# Diagnostic: verify F32 eltype actually flows through env-level cast,
# and compare parallel-level-cast vs env-level-cast single-call latency.

module load Stages/2026 NVHPC/25.9-CUDA-13 OpenMPI/5.0.8
JULIA=$HOME/tools/julia-1.11.1/bin/julia

CLEAN_LD=$(echo $LD_LIBRARY_PATH | tr ":" "\n" | \
    grep -v "math_libs\|compilers/lib\|CUDA/13/targets\|CUDA/13/nvvm\|CUDA/13/extras\|CUDA/13/stubs" | \
    tr "\n" ":" | sed "s/:$//")

BASE_ENVS="export CUDA_VISIBLE_DEVICES=\$SLURM_LOCALID;\
export LD_LIBRARY_PATH=$CLEAN_LD;\
export UCX_MEMTYPE_CACHE=n;\
export UCX_TLS=rc_x,self,sm,cuda_copy;\
export UCX_WARN_UNUSED_ENV_VARS=n;\
export CUDA_LAUNCH_BLOCKING=1"

echo "========== 4 GPU MPI eltype diagnostic =========="
srun -n 4 --gpus-per-task=1 bash -c "$BASE_ENVS; exec $JULIA --project=../.. ./microbench_eltype_check.jl"
