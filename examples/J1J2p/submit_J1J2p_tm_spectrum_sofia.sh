#!/bin/bash
#SBATCH --job-name=J1J2p_TM_D8
#SBATCH --output=%x_%A_%a.out
#SBATCH --partition=zen4_h200
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --account=pilot_2026_0002
#SBATCH --array=0-1

source /etc/profile
source /etc/profile.d/modules.sh
module load GDRCopy/2.4.4-GCCcore-14.2.0 UCX-CUDA/1.18.0-GCCcore-14.2.0-CUDA-12.8.0 OpenMPI/5.0.7-GCC-14.2.0
set -euo pipefail

WD=/sofia/scratch/pilot/pilot_2026_0002/xz
export HOME=$WD
export JULIA_DEPOT_PATH=$WD/.julia
export JULIA_NUM_THREADS=1
export LD_PRELOAD=/usr/lib64/libcuda.so.1
export CUDA_LAUNCH_BLOCKING=1

JULIA=$WD/julia-1.11.3/bin/julia
: "${REPO:=$WD/TeneT_tm_spectrum}"
: "${CHECKPOINT:?CHECKPOINT is required}"
: "${RUN_ROOT:?RUN_ROOT is required}"
: "${CHI:=336}"
: "${NLEVELS:=10}"
: "${SEED:=89012807}"

if [[ "$SLURM_ARRAY_TASK_ID" == "0" ]]; then
    SECTOR=trivial
else
    SECTOR=non-trivial
fi

echo "Host=$(hostname) Start=$(date)"
echo "Sector=$SECTOR chi=$CHI nlevels=$NLEVELS"
echo "Checkpoint=$CHECKPOINT"
echo "Run root=$RUN_ROOT"

"$JULIA" --project="$REPO" \
    "$REPO/examples/J1J2p/J1J2p_tm_spectrum_sofia.jl" \
    "$CHECKPOINT" "$RUN_ROOT" "$CHI" "$NLEVELS" "$SECTOR" "$SEED"

echo "Completed sector=$SECTOR at $(date)"
