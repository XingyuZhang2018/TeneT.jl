#!/bin/bash
#SBATCH --job-name=microbench
#SBATCH --output=%x_%j.out
#SBATCH --partition=booster
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:1
#SBATCH --account=e-dev-2026d01-011

# FLmap_parallel benchmark: sweep forloop_iter to isolate the F32 win/loss
# at different kernel granularities.

module load Stages/2026 NVHPC/25.9-CUDA-13 OpenMPI/5.0.8
JULIA=$HOME/tools/julia-1.11.1/bin/julia

CLEAN_LD=$(echo $LD_LIBRARY_PATH | tr ":" "\n" | \
    grep -v "math_libs\|compilers/lib\|CUDA/13/targets\|CUDA/13/nvvm\|CUDA/13/extras\|CUDA/13/stubs" | \
    tr "\n" ":" | sed "s/:$//")

export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
export LD_LIBRARY_PATH=$CLEAN_LD
export UCX_MEMTYPE_CACHE=n
export UCX_WARN_UNUSED_ENV_VARS=n
export CUDA_LAUNCH_BLOCKING=1

# D=10 chi=400 is production size. chi=100/200 already run in job 380543.
# Restart just chi=400 with forloop_iter>=4 to avoid 23 GiB intermediate OOM.
for CHI in 400; do
    echo "========== D=10 chi=$CHI =========="
    $JULIA --project=../.. ./microbench_cast.jl 10 $CHI 20
    echo ""
done
