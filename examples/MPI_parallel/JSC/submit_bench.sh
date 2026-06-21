#!/bin/bash
#SBATCH --job-name=JSC_iPEPS_bench
#SBATCH --output=%x_%j.out
#SBATCH --partition=booster
#SBATCH --time=02:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:4
#SBATCH --account=e-dev-2026d01-011

# JSC Jupiter — GH200 120GB (ARM aarch64), 4 GPU/node, 2 nodes for 8 GPU.
# iPEPS fg benchmark at D=10 χ=400 across 1/2/4/8 GPUs.

module load Stages/2026 NVHPC/25.9-CUDA-13 OpenMPI/5.0.8

JULIA=$HOME/tools/julia-1.11.1/bin/julia
MAX_GPU=${SLURM_NTASKS}

CLEAN_LD=$(echo $LD_LIBRARY_PATH | tr ":" "\n" | \
    grep -v "math_libs\|CUDA/13/targets\|CUDA/13/nvvm\|CUDA/13/extras\|CUDA/13/stubs" | \
    tr "\n" ":" | sed "s/:$//")

ENVS="export CUDA_VISIBLE_DEVICES=\$SLURM_LOCALID; \
export LD_LIBRARY_PATH=$CLEAN_LD; \
export UCX_MEMTYPE_CACHE=n; \
export UCX_TLS=rc_x,self,sm,cuda_copy; \
export UCX_WARN_UNUSED_ENV_VARS=n; \
export CUDA_LAUNCH_BLOCKING=1"

echo "=== JSC iPEPS fg Benchmark (D=10 χ=400) ==="
echo "Nodes: ${SLURM_NNODES}  Max GPUs: ${MAX_GPU}  Start: $(date)"

# Serial warmup so all ranks find a consistent precompile cache.
echo "--- Precompile warmup ---"
srun -n 1 --gpus-per-task=1 bash -c "$ENVS; $JULIA --project=../../.. -e 'using TeneT, CUDA, MPI'" 2>&1 | tail -3
echo ""

for N in 1 2 4 8; do
    [ $N -gt $MAX_GPU ] && continue
    echo ""
    echo "========== ${N} GPU =========="
    srun -n $N --gpus-per-task=1 bash -c "$ENVS; exec $JULIA --project=../../.. ../benchmarks/benchmark_fg.jl"
done

echo ""
echo "=== Done: $(date) ==="
