#!/bin/bash
#SBATCH --job-name=flmap2d_gpu
#SBATCH --output=%x_%j.out
#SBATCH --partition=zen4_h200
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:nvidia_h200:4
#SBATCH --account=pilot_2026_0002
#SBATCH --chdir=/sofia/scratch/pilot/pilot_2026_0002/xz/TeneT-2d-validation

# Mirrored from examples/MPI_parallel/Sofia/submit_D16_chi768_2node_sofia.sh.
# Production-grade env setup is required for NCCL + CUDA-aware MPI to work
# correctly on Sofia H200 nodes.

source /etc/profile
source /etc/profile.d/modules.sh
module load GDRCopy/2.4.4-GCCcore-14.2.0 UCX-CUDA/1.18.0-GCCcore-14.2.0-CUDA-12.8.0 OpenMPI/5.0.7-GCC-14.2.0 NCCL/2.27.7-GCCcore-14.2.0-CUDA-12.8.0

WD=/sofia/scratch/pilot/pilot_2026_0002/xz
export JULIA_DEPOT_PATH=$WD/.julia
export HOME=$WD
JULIA=$WD/julia-1.11.3/bin/julia
MAX_GPU=${SLURM_NTASKS}

# Production env vars for Sofia GPU + NCCL fast path.
ENVS="export CUDA_VISIBLE_DEVICES=\$OMPI_COMM_WORLD_LOCAL_RANK; \
export UCX_TLS=rc_x,self,sm,cuda_copy,cuda_ipc; \
export UCX_MEMTYPE_CACHE=n; \
export UCX_WARN_UNUSED_ENV_VARS=n; \
export CUDA_LAUNCH_BLOCKING=1; \
export LD_PRELOAD=/usr/lib64/libcuda.so.1; \
export TENET_USE_NCCL=1; \
export TENET_NCCL_REGISTER=1; \
export BENCH_D=${BENCH_D:-10}; \
export BENCH_CHI=${BENCH_CHI:-512}; \
export BENCH_N1=${BENCH_N1:-2}; \
export BENCH_USE_GPU=1; \
export BENCH_REPEATS=${BENCH_REPEATS:-3}; \
export BENCH_WARMUP=${BENCH_WARMUP:-1}"

echo "=== FLmap_parallel_2D vs FLmap_parallel GPU bench (Sofia H200) ==="
echo "Nodes: ${SLURM_NNODES}  GPUs: ${MAX_GPU}  Start: $(date)"
echo "D=${BENCH_D:-10} CHI=${BENCH_CHI:-512} N1=${BENCH_N1:-2}"

# Warm-up: trigger precompile outside mpirun.
bash -c "$ENVS; export CUDA_VISIBLE_DEVICES=0; \
    $JULIA --project=$WD/TeneT-2d-validation -e 'using TeneT, MPI, CUDA; @info \"warm-up loaded\"'"

echo "=== Warm-up done; launching MPI ranks ==="

mpirun -np ${MAX_GPU} \
    -x UCX_MODULE_DIR -x LD_LIBRARY_PATH -x PATH -x HOME -x JULIA_DEPOT_PATH \
    -x TENET_USE_NCCL -x TENET_NCCL_REGISTER \
    -x BENCH_D -x BENCH_CHI -x BENCH_N1 -x BENCH_USE_GPU \
    -x BENCH_REPEATS -x BENCH_WARMUP \
    bash -c "$ENVS; exec $JULIA --project=$WD/TeneT-2d-validation $WD/TeneT-2d-validation/timing/bench_flmap_2d_vs_1d.jl"

echo "=== Done: $(date) ==="
