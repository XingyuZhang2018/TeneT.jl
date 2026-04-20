#!/bin/bash
#SBATCH --job-name=parallelmp
#SBATCH --output=%x_%j.out
#SBATCH --partition=booster
#SBATCH --time=02:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:4
#SBATCH --account=e-dev-2026d01-011

# Benchmark the new parallel-level mixed precision
# (branch: feat/parallel-level-mixed-precision, commit ee048bc).
#
# Compares Float64 baseline vs Float32 (POLISH=2) across 1/2/4 GPU.
# Reference old kernel-level F32 (from prior job 379196, commit a15c18f):
#   4 GPU: fwd=24.0s (+41%), fg=330s (+8%), gnorm=0.01150 (correct)
# Hoped for new parallel-level F32:
#   fwd <= 17.1s (baseline), fg <= 304s, gnorm = 0.01150

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
export CUDA_LAUNCH_BLOCKING=1;\
export D=10; export CHI=400; export POLISH=2"

for NP in 1 2 4; do
    for PREC in Float64 Float32; do
        echo "========== $NP GPU INNER_ETYPE=$PREC =========="
        srun -n $NP --gpus-per-task=1 bash -c "$BASE_ENVS; export INNER_ETYPE=$PREC; exec $JULIA --project=../.. ./profile_fg.jl"
        echo ""
    done
done
