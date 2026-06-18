#!/bin/bash -l
#SBATCH --job-name=Sofia_slice2d_bench16
#SBATCH --output=%x_%j.out
#SBATCH --partition=zen4_h200
#SBATCH --time=01:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:nvidia_h200:8
#SBATCH --account=pilot_2026_0002

# Slice2D 4×4 (16 GPU, 2 nodes) vs slice FLmap benchmark over the Part-2 D×χ
# matrix. Rank layout: rank = r1*4 + r2 with 8 ranks/node ⇒ row communicators
# (ring shifts) stay intra-node on NVLink; column communicators (reduce-
# scatter/allgather) cross the IB link. Driver: ../benchmark_slice2d_sofia.jl
# with TENET_SLICE2D_N1=4 TENET_SLICE2D_N2=4.

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
export NCCL_DEBUG=WARN; \
export TENET_SLICE2D_N1=4; \
export TENET_SLICE2D_N2=4"

echo "=== Sofia 16-GPU (2-node) Slice2D 4x4 vs slice benchmark ==="
echo "Start: $(date)"
mpirun -np 16 -x UCX_MODULE_DIR -x LD_LIBRARY_PATH=$CLEAN_LD -x PATH -x HOME -x JULIA_DEPOT_PATH \
    bash -c "$BASE_ENVS; exec $JULIA --project=../../.. ../benchmark_slice2d_sofia.jl"
echo "=== Done: $(date) ==="
