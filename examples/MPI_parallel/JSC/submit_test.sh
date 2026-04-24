#!/bin/bash
#SBATCH --job-name=MPI_test
#SBATCH --output=%x_%j.out
#SBATCH --partition=booster
#SBATCH --time=01:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:4
#SBATCH --account=e-dev-2026d01-011

# JSC Jupiter — GH200 120GB (ARM aarch64)
module load Stages/2026 NVHPC/25.9-CUDA-13 OpenMPI/5.0.8

JULIA=$HOME/tools/julia-1.11.1/bin/julia
MAX_GPU=${SLURM_NTASKS}

# Strip NVHPC math_libs + CUDA/13 subdirs (cuBLAS/cuTENSOR crash sources)
# but KEEP compilers/lib (libmpi NEEDED-deps live there).
CLEAN_LD=$(echo $LD_LIBRARY_PATH | tr ":" "\n" | \
    grep -v "math_libs\|CUDA/13/targets\|CUDA/13/nvvm\|CUDA/13/extras\|CUDA/13/stubs" | \
    tr "\n" ":" | sed "s/:$//")

ENVS="export CUDA_VISIBLE_DEVICES=\$SLURM_LOCALID;\
export LD_LIBRARY_PATH=$CLEAN_LD;\
export UCX_MEMTYPE_CACHE=n;\
export UCX_TLS=rc_x,self,sm,cuda_copy;\
export UCX_WARN_UNUSED_ENV_VARS=n;\
export CUDA_LAUNCH_BLOCKING=1"

echo "=== JSC Jupiter MPI Scaling Test ==="
echo "Nodes: ${SLURM_NNODES}  Max GPUs: ${MAX_GPU}  Start: $(date)"

# Serial warmup so all ranks find a consistent precompile cache (avoids the
# multi-rank AtomixCUDAExt precompile race seen in job 390782).
echo "--- Precompile warmup ---"
srun -n 1 --gpus-per-task=1 bash -c "$ENVS; $JULIA --project=../../.. -e 'using TeneT, CUDA, MPI'" 2>&1 | tail -3
echo ""

for N in 1 2 4 8; do
    [ $N -gt $MAX_GPU ] && continue
    echo "========== ${N} GPU =========="
    srun -n $N --gpus-per-task=1 bash -c "$ENVS; exec $JULIA --project=../../.. ../test_MPI_config.jl"
    echo ""
done

echo "=== Done: $(date) ==="
