#!/bin/bash -l
#SBATCH --job-name=Sofia_diag_16gpu
#SBATCH --output=%x_%j.out
#SBATCH --partition=zen4_h200
#SBATCH --time=00:10:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:nvidia_h200:8
#SBATCH --account=pilot_2026_0002

source /etc/profile
source /etc/profile.d/modules.sh
module load GDRCopy/2.4.4-GCCcore-14.2.0 UCX-CUDA/1.18.0-GCCcore-14.2.0-CUDA-12.8.0 OpenMPI/5.0.7-GCC-14.2.0

WD=/sofia/scratch/pilot/pilot_2026_0002/xz
export JULIA_DEPOT_PATH=$WD/.julia
export HOME=$WD
JULIA=$WD/julia-1.11.3/bin/julia

ENVS="export CUDA_VISIBLE_DEVICES=\$OMPI_COMM_WORLD_LOCAL_RANK; \
export UCX_TLS=rc_x,self,sm,cuda_copy,cuda_ipc; \
export UCX_MEMTYPE_CACHE=n; \
export UCX_WARN_UNUSED_ENV_VARS=n; \
export CUDA_LAUNCH_BLOCKING=1; \
export LD_PRELOAD=/usr/lib64/libcuda.so.1"

echo "=== Sofia 16 GPU cross-node Allreduce diag ==="
echo "Start: $(date)"
mpirun --map-by ppr:8:node --bind-to none -np 16 -x UCX_MODULE_DIR -x LD_LIBRARY_PATH -x PATH -x HOME -x JULIA_DEPOT_PATH \
    bash -c "$ENVS; exec $JULIA --project=../../.. diag_16gpu_allreduce.jl"
echo "=== Done: $(date) ==="
