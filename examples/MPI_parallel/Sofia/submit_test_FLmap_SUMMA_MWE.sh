#!/bin/bash
#SBATCH --job-name=flmap_summa_mwe
#SBATCH --output=%x_%j.out
#SBATCH --partition=zen4_h200
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:nvidia_h200:4
#SBATCH --account=pilot_2026_0002

# SUMMA-FLmap MWE — validates 2D distributed FLmap against 1D FLmap_parallel.
# Square grid required for v1 → run on 4 GPUs (2×2 grid).
# See docs/2026-05-12-summa-flmap-design.md for the algorithm.

source /etc/profile
source /etc/profile.d/modules.sh
module load GDRCopy/2.4.4-GCCcore-14.2.0 UCX-CUDA/1.18.0-GCCcore-14.2.0-CUDA-12.8.0 OpenMPI/5.0.7-GCC-14.2.0 NCCL/2.27.7-GCCcore-14.2.0-CUDA-12.8.0

WD=/sofia/scratch/pilot/pilot_2026_0002/xz
export JULIA_DEPOT_PATH=$WD/.julia
export HOME=$WD
JULIA=$WD/julia-1.11.3/bin/julia
MAX_GPU=${SLURM_NTASKS}

ENVS="export CUDA_VISIBLE_DEVICES=\$OMPI_COMM_WORLD_LOCAL_RANK; \
export UCX_TLS=rc_x,self,sm,cuda_copy,cuda_ipc; \
export UCX_MEMTYPE_CACHE=n; \
export UCX_WARN_UNUSED_ENV_VARS=n; \
export CUDA_LAUNCH_BLOCKING=1; \
export LD_PRELOAD=/usr/lib64/libcuda.so.1; \
export TENET_USE_NCCL=1; \
export TENET_NCCL_REGISTER=1"

echo "=== test_FLmap_SUMMA_MWE.jl, 4 GPU (2x2 grid), $(date) ==="

bash -c "$ENVS; export CUDA_VISIBLE_DEVICES=0; \
    $JULIA --project=$WD/TeneT-2d-validation -e 'using TeneT, MPI, CUDA, TensorOperations; @info \"warm-up loaded\"'"

mpirun -np ${MAX_GPU} \
    -x UCX_MODULE_DIR -x LD_LIBRARY_PATH -x PATH -x HOME -x JULIA_DEPOT_PATH \
    -x TENET_USE_NCCL -x TENET_NCCL_REGISTER \
    bash -c "$ENVS; exec $JULIA --project=$WD/TeneT-2d-validation $WD/TeneT-2d-validation/examples/MPI_parallel/test_FLmap_SUMMA_MWE.jl"

echo "=== Done $(date) ==="
