#!/bin/bash -l
#SBATCH --job-name=Sofia_slice2d_m4
#SBATCH --output=%x_%j.out
#SBATCH --partition=zen4_h200
#SBATCH --time=00:45:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:nvidia_h200:8
#SBATCH --account=pilot_2026_0002

# Sofia 16-GPU (2-node, 4×4 Slice2D grid) M4 Batch-A validation: distributed
# `leftenv_slice2d` (env-level Slice2D + gather hoisting) forward eigenpair + AD
# gradient parity vs the SERIAL `leftenv`, on GPU, at χ=256 D=8. Rank layout:
# rank = r1*4 + r2, 8 ranks/node ⇒ row comms (ring shifts) stay intra-node on
# NVLink; COLUMN comms (the hoisted gather reduce-scatter/allgather adjoints)
# cross the IB link — the high-value cross-node axis this run exercises on-device
# (incl. the GPU-densify fix in the slice2d_gather_row/col rrules). NCCL OFF this
# run (hand ring) to isolate env-layer correctness; a follow-up flips TENET_USE_NCCL=1.
# Driver: ../test_slice2d_m4_sofia.jl (grid auto-derived from nprocs: 16 → 4×4).
# Design: docs/2026-06-15-m4-env-slice2d-integration-design.md (Batch A / Gate 5).

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
export TENET_SLICE2D_CHI=256; \
export TENET_SLICE2D_D=8; \
export TENET_SLICE2D_POWER=40"

echo "=== Sofia 16-GPU (2-node) M4 Batch-A leftenv_slice2d validation (4×4) ==="
echo "Start: $(date)"

# Precompile warmup (single rank): the M4 src edits stale the depot's TeneT
# cache; warming it once avoids 16 ranks racing on Pkg precompilation.
echo "--- precompile warmup ---"
mpirun -np 1 -x UCX_MODULE_DIR -x LD_LIBRARY_PATH=$CLEAN_LD -x PATH -x HOME -x JULIA_DEPOT_PATH \
    bash -c "$BASE_ENVS; exec $JULIA --project=../../.. -e 'using TeneT; println(\"TeneT precompiled\")'"

echo "--- 16-rank validation ---"
mpirun -np 16 -x UCX_MODULE_DIR -x LD_LIBRARY_PATH=$CLEAN_LD -x PATH -x HOME -x JULIA_DEPOT_PATH \
    bash -c "$BASE_ENVS; exec $JULIA --project=../../.. ../test_slice2d_m4_sofia.jl"
echo ""
echo "=== Done: $(date) ==="
