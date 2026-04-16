#!/bin/bash
#SBATCH --job-name=iPEPS_MPI
#SBATCH --output=%x_%j.out
#SBATCH --partition=booster
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:4
#SBATCH --account=e-dev-2026d01-011

# JSC Jupiter — GH200 120GB (ARM aarch64)
module load Stages/2026 NVHPC/25.9-CUDA-13 OpenMPI/5.0.8

NGPU=${SLURM_NTASKS}
JULIA=$HOME/tools/julia-1.11.1/bin/julia

# Strip NVHPC math_libs to use CUDA.jl artifacts (faster + no cuBLAS crash)
CLEAN_LD=$(echo $LD_LIBRARY_PATH | tr ":" "\n" | \
    grep -v "math_libs\|compilers/lib\|CUDA/13/targets\|CUDA/13/nvvm\|CUDA/13/extras\|CUDA/13/stubs" | \
    tr "\n" ":" | sed "s/:$//")

ENVS="export CUDA_VISIBLE_DEVICES=\$SLURM_LOCALID;\
export LD_LIBRARY_PATH=$CLEAN_LD;\
export UCX_MEMTYPE_CACHE=n;\
export UCX_TLS=rc_x,self,sm,cuda_copy;\
export UCX_WARN_UNUSED_ENV_VARS=n;\
export CUDA_LAUNCH_BLOCKING=1"

echo "=== iPEPS MPI Parallel (JSC) ==="
echo "Nodes: ${SLURM_NNODES}  GPUs: ${NGPU}  Start: $(date)"

srun -n ${NGPU} --gpus-per-task=1 bash -c \
    "$ENVS; exec $JULIA --project=../../.. ../MPI_parallel.jl"

echo "=== Done: $(date) ==="
