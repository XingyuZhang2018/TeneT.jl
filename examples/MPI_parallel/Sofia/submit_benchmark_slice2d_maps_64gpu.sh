#!/bin/bash -l
#SBATCH --job-name=Sofia_slice2d_maps64
#SBATCH --output=%x_%j.out
#SBATCH --partition=zen4_h200
#SBATCH --time=01:30:00
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:nvidia_h200:8
#SBATCH --account=pilot_2026_0002

# 64-GPU (8×8, 8 nodes) slice2d MAP timing: FLmap / FRmap / ACmap / ACdmap fwd+bwd,
# ring vs NCCL, over the Part-2 D×χ matrix. 8 ranks/node ⇒ row_comm is intra-node
# (NVLink), col_comm spans ALL 8 nodes over IB — the maximal cross-node stress for
# the NCCL col/reduce-scatter path. Driver: ../benchmark_slice2d_maps_sofia.jl with
# TENET_SLICE2D_N1=N2=8. Results → Sofia_VUB_H200.md (NCCL scaling).

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
export TENET_SLICE2D_N1=8; \
export TENET_SLICE2D_N2=8; \
export UCX_TLS=rc_x,self,sm,cuda_copy,cuda_ipc; \
export UCX_MEMTYPE_CACHE=n; \
export UCX_WARN_UNUSED_ENV_VARS=n; \
export CUDA_LAUNCH_BLOCKING=1; \
export LD_PRELOAD=/usr/lib64/libcuda.so.1; \
export NCCL_DEBUG=WARN"

echo "=== Sofia 64-GPU (8×8, 8 nodes) slice2d MAP timing (FL/FR/AC/ACd) — RING vs NCCL ==="
echo "Start: $(date)"
mpirun -np 64 -x UCX_MODULE_DIR -x LD_LIBRARY_PATH=$CLEAN_LD -x PATH -x HOME -x JULIA_DEPOT_PATH \
    bash -c "$BASE_ENVS; exec $JULIA --project=../../.. ../benchmark_slice2d_maps_sofia.jl"
echo "=== Done: $(date) ==="
