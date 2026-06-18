#!/bin/bash -l
#SBATCH --job-name=Sofia_fg_nccl_compare
#SBATCH --output=%x_%j.out
#SBATCH --partition=zen4_h200
#SBATCH --time=00:30:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:nvidia_h200:8
#SBATCH --account=pilot_2026_0002

# Production profiling pivot: compare benchmark_fg.jl at 16 GPU
# WITH vs WITHOUT TENET_USE_NCCL=1.
#
# All prior Sofia_iPEPS_bench runs left this env var unset → ran the slow
# 3-phase p2p ring (44 ms / 128 MB allreduce). The NCCL fast path is
# documented in nccl_wrapper.jl as "1.5 ms / 128 MB allreduce" — 30× win.
# Backward at 16 GPU is comm-bottlenecked (Sofia_VUB_H200.md Part 2 backward
# scales only 2-8x), so this should produce a real improvement on fg total.

source /etc/profile
source /etc/profile.d/modules.sh
module load GDRCopy/2.4.4-GCCcore-14.2.0 \
            UCX-CUDA/1.18.0-GCCcore-14.2.0-CUDA-12.8.0 \
            OpenMPI/5.0.7-GCC-14.2.0 \
            CUDA/12.8.0 \
            NCCL/2.27.7-GCCcore-14.2.0-CUDA-12.8.0

WD=/sofia/scratch/pilot/pilot_2026_0002/xz
export JULIA_DEPOT_PATH=$WD/.julia
export HOME=$WD
JULIA=$WD/julia-1.11.3/bin/julia

CLEAN_LD=$(echo $LD_LIBRARY_PATH | tr ':' '\n' | grep -v 'CUDA/12.8.0' | tr '\n' ':')

BASE_ENVS="export CUDA_VISIBLE_DEVICES=\$OMPI_COMM_WORLD_LOCAL_RANK; \
export UCX_TLS=rc_x,self,sm,cuda_copy,cuda_ipc; \
export UCX_MEMTYPE_CACHE=n; \
export UCX_WARN_UNUSED_ENV_VARS=n; \
export CUDA_LAUNCH_BLOCKING=1; \
export LD_PRELOAD=/usr/lib64/libcuda.so.1; \
export NCCL_DEBUG=WARN"

run_case() {
    label="$1"
    extra="$2"
    echo ""
    echo "========== ${label} =========="
    echo "extra env: $extra"
    mpirun -np 16 -x UCX_MODULE_DIR -x LD_LIBRARY_PATH=$CLEAN_LD -x PATH -x HOME -x JULIA_DEPOT_PATH \
        bash -c "$BASE_ENVS; $extra exec $JULIA --project=../../.. ../benchmarks/benchmark_fg.jl" 2>&1 \
        | grep -vE 'This may cause|was loaded|If you.re|ensure that|In any other|file an issue|^\s*│|^\s*└|^\s*┌|Precompiling|ms  ✓|already precompiled|Being precompiled|NCCL INFO'
}

echo "=== Sofia 16-GPU benchmark_fg: NCCL on vs off ==="
echo "Start: $(date)"

run_case "(A) NCCL OFF (3-phase ring, current production default)" ""
run_case "(B) NCCL ON  (TENET_USE_NCCL=1)" \
    "export TENET_USE_NCCL=1;"

echo ""
echo "=== Done: $(date) ==="
