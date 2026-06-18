#!/bin/bash
#SBATCH --job-name=PlqDQR1024N10
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=zen4_h200
#SBATCH --time=24:00:00
#SBATCH --nodes=8
#SBATCH --ntasks=64
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:nvidia_h200:8
#SBATCH --account=pilot_2026_0002

set -eo pipefail

source /etc/profile
source /etc/profile.d/modules.sh
module load GDRCopy/2.4.4-GCCcore-14.2.0 UCX-CUDA/1.18.0-GCCcore-14.2.0-CUDA-12.8.0 OpenMPI/5.0.7-GCC-14.2.0

WD=/sofia/scratch/pilot/pilot_2026_0002/xz
PROJECT=$WD/TeneT_codex_2a4a_20260616_225907_tsqr
DATA_ROOT=$WD/TeneT.jl/data
JULIA=$WD/julia-1.11.3/bin/julia
DRIVER=$SLURM_SUBMIT_DIR/entry.jl
MAX_GPU=${MPI_NPROCS:-${SLURM_NTASKS:-64}}

export JULIA_DEPOT_PATH=$WD/.julia
export HOME=$WD

# Keep Sofia on the manifest/depot combination that already ran the TSQR jobs.
cp "$WD/TeneT_m3gate/Manifest.toml" "$PROJECT/Manifest.toml"
cp "$PROJECT/examples/MPI_parallel/Sofia/LocalPreferences.toml" "$PROJECT/LocalPreferences.toml"
CLEAN_LD=$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v 'CUDA/12.8.0' | tr '\n' ':' || true)

ENVS="export CUDA_VISIBLE_DEVICES=\$OMPI_COMM_WORLD_LOCAL_RANK; \
export UCX_TLS=rc_x,self,sm,cuda_copy,cuda_ipc; \
export UCX_MEMTYPE_CACHE=n; \
export UCX_WARN_UNUSED_ENV_VARS=n; \
export CUDA_LAUNCH_BLOCKING=1; \
export LD_PRELOAD=/usr/lib64/libcuda.so.1; \
export JULIA_NUM_THREADS=1; \
export DATA_ROOT=$DATA_ROOT; \
export D=${D:-16}; export SEED=${SEED:-42}; \
export J2=${J2:-0.5}; \
export CHI_LOAD=${CHI_LOAD:-768}; export NO_LOAD=${NO_LOAD:-10}; \
export CHI_OPT=${CHI_OPT:-1024}; export OPT_MAXITER=${OPT_MAXITER:-20}; \
export VUMPS_MAXITER=${VUMPS_MAXITER:-30}; export SHOW_EVERY=${SHOW_EVERY:-1}; \
export MAXITER_AD=${MAXITER_AD:-4}; export MINITER_AD=${MINITER_AD:-4}; \
export POWER_ITER=${POWER_ITER:-5}; export POWER_ITER_AD=${POWER_ITER_AD:-5}; \
export POWER_ITER_OBS=${POWER_ITER_OBS:-20}; \
export ENV_TOL=${ENV_TOL:-1e-8}; export FORLOOP_ITER=${FORLOOP_ITER:-32}; \
export SAVE_EVERY=${SAVE_EVERY:-1}; export SAVE_ENV=${SAVE_ENV:-false}; \
export DISTRIBUTED_QR=${DISTRIBUTED_QR:-true}; \
export STEP_CKPT=${STEP_CKPT:-recompute}; export BOND_CKPT=${BOND_CKPT:-recompute}"

echo "=== Slice2D Plaquette distributed_qr AD optimize D=${D:-16} J2=${J2:-0.5} chi${CHI_LOAD:-768} No.${NO_LOAD:-10} -> chi${CHI_OPT:-1024} ==="
echo "Project=$PROJECT"
echo "Data root=$DATA_ROOT"
echo "Driver=$DRIVER"
echo "Dependency=none; source No.10 already exists"
echo "Nodes=${SLURM_NNODES} AllocTasks=${SLURM_NTASKS} LaunchRanks=${MAX_GPU} Host=$(hostname) Start=$(date)"
echo "OPT_MAXITER=${OPT_MAXITER:-20} VUMPS_MAXITER=${VUMPS_MAXITER:-30} MAXITER_AD=${MAXITER_AD:-4} MINITER_AD=${MINITER_AD:-4}"
echo "POWER_ITER=${POWER_ITER:-5} POWER_ITER_AD=${POWER_ITER_AD:-5} POWER_ITER_OBS=${POWER_ITER_OBS:-20}"
echo "ENV_TOL=${ENV_TOL:-1e-8} FORLOOP_ITER=${FORLOOP_ITER:-32} SAVE_ENV=${SAVE_ENV:-false} DISTRIBUTED_QR=${DISTRIBUTED_QR:-true}"
echo "STEP_CKPT=${STEP_CKPT:-recompute} BOND_CKPT=${BOND_CKPT:-recompute}"

echo "=== precompile warmup (single rank) ==="
bash -c "$ENVS; export CUDA_VISIBLE_DEVICES=0; $JULIA --project=$PROJECT -e 'using CUDA, MPI, TeneT; @info \"warm-up loaded\"'"
echo "=== warmup done; launching $MAX_GPU ranks (ppr:8:node bind-none) ==="

mpirun --map-by ppr:8:node --bind-to none -np "${MAX_GPU}" \
    -x UCX_MODULE_DIR -x LD_LIBRARY_PATH="$CLEAN_LD" -x PATH -x HOME -x JULIA_DEPOT_PATH \
    bash -c "$ENVS; exec $JULIA --project=$PROJECT $DRIVER"

echo "=== done $(date) ==="
