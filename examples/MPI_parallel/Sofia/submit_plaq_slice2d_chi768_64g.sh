#!/bin/bash
#SBATCH --job-name=PlqCanD16x768_64g
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=zen4_h200
#SBATCH --time=02:00:00
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:nvidia_h200:8
#SBATCH --account=pilot_2026_0002

# Slice2D (block-distributed χ) Plaquette J1J2 Square — D16 χ768 64-GPU (8 nodes × 8 H200, 8×8 grid).
# Same SMOKE as the 16-GPU run (submit_plaq_slice2d_chi768.sh, job 1287345) but on 4× the ranks:
# each GPU now holds χ/8 × χ/8 = 96×96 blocks (vs 192×192 on 4×4) → much more memory headroom, and
# the driver's new t_fwd / t_bwd timers give the forward(record) + backward(grad) wall-clock split.
#
# SMOKE (submit with OPT_MAXITER=0 VUMPS_MAXITER=0): distributed init + one forward + one production
# backward at χ768, no LBFGS step, no observable. Answers: does χ768 fit (and how roomy) on 64 GPU,
# and what is the per-pass timing on an 8×8 grid (for χ768 production walltime estimation).
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
# Defaults reproduce the 16-GPU smoke (1287345): OPT_MAXITER=0 VUMPS_MAXITER=0 χ512→χ768 D16 J2=0.5.
ENVS="export CUDA_VISIBLE_DEVICES=\$OMPI_COMM_WORLD_LOCAL_RANK; \
export UCX_TLS=rc_x,self,sm,cuda_copy,cuda_ipc; \
export UCX_MEMTYPE_CACHE=n; \
export UCX_WARN_UNUSED_ENV_VARS=n; \
export CUDA_LAUNCH_BLOCKING=1; \
export LD_PRELOAD=/usr/lib64/libcuda.so.1; \
export DATA_ROOT=$WD/TeneT.jl/data; \
export D=${D:-16}; export J2=${J2:-0.5}; export CHI_LOAD=${CHI_LOAD:-512}; export NO_LOAD=${NO_LOAD:-50}; \
export CHI_OPT=${CHI_OPT:-768}; export OPT_MAXITER=${OPT_MAXITER:-0}; \
export POWER_ITER=${POWER_ITER:-2}; export VUMPS_MAXITER=${VUMPS_MAXITER:-0}; \
export POWER_ITER_OBS=${POWER_ITER_OBS:-20}; export ENV_TOL=${ENV_TOL:-1e-6}; \
export FORLOOP_ITER=${FORLOOP_ITER:-16}"

echo "=== Slice2D Plaquette D=${D:-16} J2=${J2:-0.5} χ${CHI_LOAD:-512}→χ${CHI_OPT:-768} OPT_MAXITER=${OPT_MAXITER:-0} VUMPS_MAXITER=${VUMPS_MAXITER:-0} forloop=${FORLOOP_ITER:-16} | 64 GPU 8x8 ==="
echo "Project=$GATE  DATA_ROOT=$WD/TeneT.jl/data  Driver=$DRIVER"
echo "Nodes=${SLURM_NNODES} Tasks=${SLURM_NTASKS} GPUs=${MAX_GPU} Host=$(hostname) Start=$(date)"

echo "=== precompile warmup (single rank; M6 src was just synced → cache is stale) ==="
bash -c "$ENVS; export CUDA_VISIBLE_DEVICES=0; $JULIA --project=$GATE -e 'using TeneT, MPI, CUDA; @info \"warm-up loaded\"'"
echo "=== warmup done; launching $MAX_GPU ranks (ppr:8:node bind-none) ==="

mpirun --map-by ppr:8:node --bind-to none -np "${MAX_GPU}" \
    -x UCX_MODULE_DIR -x LD_LIBRARY_PATH="$CLEAN_LD" -x PATH -x HOME -x JULIA_DEPOT_PATH \
    bash -c "$ENVS; exec $JULIA --project=$GATE $DRIVER"

echo "=== done $(date) ==="
