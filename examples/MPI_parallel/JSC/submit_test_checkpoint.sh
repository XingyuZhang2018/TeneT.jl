#!/bin/bash
#SBATCH --job-name=MPI_ckpt_test
#SBATCH --output=%x_%j.out
#SBATCH --partition=booster
#SBATCH --time=00:30:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:4
#SBATCH --account=e-dev-2026d01-011

# JSC Jupiter — GH200 120GB (ARM aarch64)
# Probes checkpoint(Plain/Recompute/Offload, FLmap_parallel) under MPI multi-GPU.
# Primary concern: Offload's args-to-host + args-to-device bracket around MPI
# collectives during backward re-forward.

module load Stages/2026 NVHPC/25.9-CUDA-13 OpenMPI/5.0.8

JULIA=$HOME/tools/julia-1.11.1/bin/julia
MAX_GPU=${SLURM_NTASKS}

# Strip NVHPC math_libs to use CUDA.jl artifacts (required on JSC — see
# submit_test.sh for context)
CLEAN_LD=$(echo $LD_LIBRARY_PATH | tr ":" "\n" | \
    grep -v "math_libs\|compilers/lib\|CUDA/13/targets\|CUDA/13/nvvm\|CUDA/13/extras\|CUDA/13/stubs" | \
    tr "\n" ":" | sed "s/:$//")

ENVS="export CUDA_VISIBLE_DEVICES=\$SLURM_LOCALID;\
export LD_LIBRARY_PATH=$CLEAN_LD;\
export UCX_MEMTYPE_CACHE=n;\
export UCX_TLS=rc_x,self,sm,cuda_copy;\
export UCX_WARN_UNUSED_ENV_VARS=n;\
export CUDA_LAUNCH_BLOCKING=1"

echo "=== JSC Jupiter MPI checkpoint() Test ==="
echo "Nodes: ${SLURM_NNODES}  Max GPUs: ${MAX_GPU}  Start: $(date)"

# Scale from 1 GPU up — issues may only surface at N >= 2
for N in 1 2 4 8; do
    [ $N -gt $MAX_GPU ] && continue
    echo "========== ${N} GPU =========="
    srun -n $N --gpus-per-task=1 bash -c "$ENVS; exec $JULIA --project=../../.. ../test_MPI_checkpoint.jl"
    echo ""
done

echo "=== Done: $(date) ==="
