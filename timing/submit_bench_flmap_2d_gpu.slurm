#!/bin/bash -l
#SBATCH --job-name=flmap2d_gpu
#SBATCH --account=pilot_2026_0002
#SBATCH --partition=zen4_h200
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --time=00:45:00
#SBATCH --chdir=/sofia/scratch/pilot/pilot_2026_0002/xz/TeneT-2d-validation
#SBATCH --output=flmap2d_gpu_%j.out

# GPU benchmark of FLmap_parallel_2D vs FLmap_parallel (1D) at production scale.
# 4 H200 GPUs, 2x2 grid, D=10 chi=512 — comparable to Kagome 1003757 OOM regime.

source /etc/profile
module load OpenMPI/5.0.7-GCC-14.2.0

# LD_LIBRARY_PATH: strip CUDA per memory note reference_sofia_ld_path
export LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | tr ':' '\n' | grep -v cuda | tr '\n' ':' | sed 's/:$//')

# UCX/MPI fix per memory note project_sofia_hpc_ucx_fix
[ -f /usr/lib64/libcuda.so.1 ] && export LD_PRELOAD=/usr/lib64/libcuda.so.1
export UCX_WARN_UNUSED_ENV_VARS=n

export PATH=/sofia/scratch/pilot/pilot_2026_0002/xz/julia-1.11.3/bin:$PATH
export JULIA_DEPOT_PATH=/sofia/scratch/pilot/pilot_2026_0002/xz/.julia

# Production-scale config: D=10 chi=512 (close to Kagome 1003757 regime)
export BENCH_D=${BENCH_D:-10}
export BENCH_CHI=${BENCH_CHI:-512}
export BENCH_N1=${BENCH_N1:-2}
export BENCH_USE_GPU=1
export BENCH_REPEATS=${BENCH_REPEATS:-3}
export BENCH_WARMUP=${BENCH_WARMUP:-1}

echo "[$(date)] starting flmap_2d_vs_1d GPU bench"
echo "  D=${BENCH_D}, CHI=${BENCH_CHI}, N1=${BENCH_N1}, USE_GPU=${BENCH_USE_GPU}"
echo "  Per-rank tensor size (each of FL/ALu/ALd) = chi^2 * D^2 / N = $(( BENCH_CHI * BENCH_CHI * BENCH_D * BENCH_D * 16 / 4 )) bytes"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "(nvidia-smi not on login node — will run on compute node)"

mpirun -n 4 julia --project=. timing/bench_flmap_2d_vs_1d.jl
RC=$?
echo "[$(date)] bench exit=${RC}"
exit $RC
