#!/bin/bash -l
#SBATCH --job-name=Sofia_slice2d_m5
#SBATCH --output=%x_%j.out
#SBATCH --partition=zen4_h200
#SBATCH --time=00:45:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:nvidia_h200:8
#SBATCH --account=pilot_2026_0002

# Sofia 16-GPU (2-node, 4×4 Slice2D grid) M5 validation: the full distributed
# `vumps_step_slice2d` (four M4 env solvers + the two QR gather seams + orchestration)
# forward + AD-gradient parity vs the SERIAL `vumps_step`, on GPU, at χ=256 D=8. The
# NEW on-device coverage over M4: the QR-seam gather(AC)→QR→scatter(AL/AR) and the
# entry-seam ALCtoAC gather/scatter, cross-node over IB (column axis). Parity uses a
# RANDOM rt with power_iter=1 (one map application — serial & slice2d run the SAME short
# deterministic function; convergence/canonicalization irrelevant to parity, and a high
# power_iter on a random non-canonical rt amplifies the block-vs-full FP-order diff into
# O(1) via the ill-conditioned AC eigen-map — a fixture artifact, not a defect: the seam
# is bit-exact, see test/test_slice2d_m5.jl Gate M5-0). NCCL OFF (hand ring) to isolate
# correctness. Driver: ../validation/test_slice2d_m5_sofia.jl (grid auto-derived: 16 → 4×4).
# Design: docs/2026-06-15-m5-vumps-step-slice2d-assembly-design.md (§6).

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
export TENET_SLICE2D_POWER=1"

echo "=== Sofia 16-GPU (2-node) M5 vumps_step_slice2d validation (4×4) ==="
echo "Start: $(date)"

# Precompile warmup (single rank): the M5 src edits stale the depot's TeneT cache;
# warm it once so 16 ranks don't race on Pkg precompilation.
echo "--- precompile warmup ---"
mpirun -np 1 -x UCX_MODULE_DIR -x LD_LIBRARY_PATH=$CLEAN_LD -x PATH -x HOME -x JULIA_DEPOT_PATH \
    bash -c "$BASE_ENVS; exec $JULIA --project=../../.. -e 'using TeneT; println(\"TeneT precompiled\")'"

echo "--- 16-rank validation ---"
mpirun -np 16 -x UCX_MODULE_DIR -x LD_LIBRARY_PATH=$CLEAN_LD -x PATH -x HOME -x JULIA_DEPOT_PATH \
    bash -c "$BASE_ENVS; exec $JULIA --project=../../.. ../validation/test_slice2d_m5_sofia.jl"
echo ""
echo "=== Done: $(date) ==="
