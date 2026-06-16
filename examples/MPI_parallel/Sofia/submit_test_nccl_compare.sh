#!/bin/bash -l
#SBATCH --job-name=Sofia_MPI_test_nccl
#SBATCH --output=%x_%j.out
#SBATCH --partition=zen4_h200
#SBATCH --time=01:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:nvidia_h200:8
#SBATCH --account=pilot_2026_0002

# Sofia 16-GPU test_MPI_config.jl with NCCL OFF (3-phase ring) vs NCCL ON
# (TENET_USE_NCCL=1). Direct apples-to-apples comparison of the unit-test
# matrix that produced Sofia_VUB_H200.md Part 1 (MPI collectives) and
# Part 2 (FLmap_parallel fwd/bwd at D∈{8,10,12,14,16} × χ∈{256..1024}).

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
    mpirun --map-by ppr:8:node --bind-to none -np 16 -x UCX_MODULE_DIR -x LD_LIBRARY_PATH=$CLEAN_LD -x PATH -x HOME -x JULIA_DEPOT_PATH \
        bash -c "$BASE_ENVS; $extra exec $JULIA --project=../../.. ../test_MPI_config.jl" 2>&1 \
        | grep -vE 'NCCL INFO'
}

echo "=== Sofia 16-GPU test_MPI_config.jl: NCCL on vs off ==="
echo "Start: $(date)"

run_case "(A) NCCL OFF (3-phase ring, current default)" ""
run_case "(B) NCCL ON  (TENET_USE_NCCL=1)" "export TENET_USE_NCCL=1;"

echo ""
echo "=== Done: $(date) ==="
