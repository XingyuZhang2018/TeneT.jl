#!/bin/bash
#SBATCH --job-name=PlqCanD16x768
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=zen4_h200
#SBATCH --time=06:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:nvidia_h200:8
#SBATCH --account=pilot_2026_0002

# Slice2D (block-distributed χ) Plaquette J1J2 Square — D16 χ768 16-GPU (2 nodes × 8 H200, 4×4 grid).
# Loads the converged J2=0.5 D16 χ512 No.50 from the production TeneT.jl/data tree, optimizes at
# χ768 from the TeneT_m3gate clone (has M3–M6: slice2d boundary + distributed energy_value).
#
# SMOKE (default): OPT_MAXITER=0 → env build + one forward + one production backward at χ768,
# no LBFGS step. Validates memory fit + M6 distributed-energy GPU correctness.
# Full run: resubmit with OPT_MAXITER=20 POWER_ITER=5 VUMPS_MAXITER=30 ENV_TOL=1e-8 (+ raise --time).
set -eo pipefail

source /etc/profile
source /etc/profile.d/modules.sh
module load GDRCopy/2.4.4-GCCcore-14.2.0 UCX-CUDA/1.18.0-GCCcore-14.2.0-CUDA-12.8.0 OpenMPI/5.0.7-GCC-14.2.0

WD=/sofia/scratch/pilot/pilot_2026_0002/xz
GATE=$WD/TeneT_m3gate
export JULIA_DEPOT_PATH=$WD/.julia
export HOME=$WD
JULIA=$WD/julia-1.11.3/bin/julia
MAX_GPU=${SLURM_NTASKS}
DRIVER=$GATE/examples/MPI_parallel/examples/J1J2_Square_VUMPS_Plaquette_slice2d.jl

CLEAN_LD=$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v 'CUDA/12.8.0' | tr '\n' ':' || true)

# Physics / slice2d params — overridable from the sbatch environment (--export) for the full run.
ENVS="export CUDA_VISIBLE_DEVICES=\$OMPI_COMM_WORLD_LOCAL_RANK; \
export UCX_TLS=rc_x,self,sm,cuda_copy,cuda_ipc; \
export UCX_MEMTYPE_CACHE=n; \
export UCX_WARN_UNUSED_ENV_VARS=n; \
export CUDA_LAUNCH_BLOCKING=1; \
export LD_PRELOAD=/usr/lib64/libcuda.so.1; \
export DATA_ROOT=$WD/TeneT.jl/data; \
export D=${D:-16}; export J2=${J2:-0.5}; export CHI_LOAD=${CHI_LOAD:-512}; export NO_LOAD=${NO_LOAD:-50}; \
export CHI_OPT=${CHI_OPT:-768}; export OPT_MAXITER=${OPT_MAXITER:-0}; \
export POWER_ITER=${POWER_ITER:-2}; export VUMPS_MAXITER=${VUMPS_MAXITER:-10}; \
export POWER_ITER_OBS=${POWER_ITER_OBS:-20}; export ENV_TOL=${ENV_TOL:-1e-6}; \
export FORLOOP_ITER=${FORLOOP_ITER:-16}"

echo "=== Slice2D Plaquette D=${D:-16} J2=${J2:-0.5} χ${CHI_LOAD:-512}→χ${CHI_OPT:-768} OPT_MAXITER=${OPT_MAXITER:-0} forloop=${FORLOOP_ITER:-16} ==="
echo "Project=$GATE  DATA_ROOT=$WD/TeneT.jl/data  Driver=$DRIVER"
echo "Nodes=${SLURM_NNODES} Tasks=${SLURM_NTASKS} GPUs=${MAX_GPU} Host=$(hostname) Start=$(date)"

echo "=== precompile warmup (single rank; M6 src was just synced → cache is stale) ==="
bash -c "$ENVS; export CUDA_VISIBLE_DEVICES=0; $JULIA --project=$GATE -e 'using TeneT, MPI, CUDA; @info \"warm-up loaded\"'"
echo "=== warmup done; launching $MAX_GPU ranks (ppr:8:node bind-none) ==="

mpirun --map-by ppr:8:node --bind-to none -np "${MAX_GPU}" \
    -x UCX_MODULE_DIR -x LD_LIBRARY_PATH="$CLEAN_LD" -x PATH -x HOME -x JULIA_DEPOT_PATH \
    bash -c "$ENVS; exec $JULIA --project=$GATE $DRIVER"

echo "=== done $(date) ==="
