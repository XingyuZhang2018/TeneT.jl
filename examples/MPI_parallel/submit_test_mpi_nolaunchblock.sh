#!/bin/bash
#SBATCH --job-name=testmpi_nolb
#SBATCH --output=%x_%j.out
#SBATCH --partition=booster
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:4
#SBATCH --account=e-dev-2026d01-011

# Verify MPI collectives (allgatherv_p2p! / allreduce_p2p!) correctness
# AND FLmap forward/backward correctness WITHOUT CUDA_LAUNCH_BLOCKING=1.
#
# Rationale: earlier probe showed CUDA_LAUNCH_BLOCKING=1 costs ~15-18%
# on vumps_step, but it was set in JSC config because of a CUDA.jl
# `synchronization_worker` segfault note. Need to confirm whether the
# p2p collectives (which race with kernels on GH200 ARM without
# explicit blocking) still produce correct results when CLB=0.
#
# Uses `test_MPI_config.jl` which contains explicit correctness checks
# (allgatherv twice → same result; allreduce of ones → nprocs).

module load Stages/2026 NVHPC/25.9-CUDA-13 OpenMPI/5.0.8
JULIA=$HOME/tools/julia-1.11.1/bin/julia

CLEAN_LD=$(echo $LD_LIBRARY_PATH | tr ":" "\n" | \
    grep -v "math_libs\|compilers/lib\|CUDA/13/targets\|CUDA/13/nvvm\|CUDA/13/extras\|CUDA/13/stubs" | \
    tr "\n" ":" | sed "s/:$//")

BASE_ENVS="export CUDA_VISIBLE_DEVICES=\$SLURM_LOCALID;\
export LD_LIBRARY_PATH=$CLEAN_LD;\
export UCX_MEMTYPE_CACHE=n;\
export UCX_TLS=rc_x,self,sm,cuda_copy;\
export UCX_WARN_UNUSED_ENV_VARS=n"

# NOTE: NO CUDA_LAUNCH_BLOCKING=1
# If p2p collectives fail (✗ marks) or hang → CLB=1 is required.
# If all ✓ and speedup visible → CLB=0 is safe to drop.

echo "========== 4 GPU MPI config test — NO CUDA_LAUNCH_BLOCKING =========="
srun -n 4 --gpus-per-task=1 bash -c "$BASE_ENVS; exec $JULIA --project=../.. ./test_MPI_config.jl"
