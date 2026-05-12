#!/bin/bash -l
#SBATCH --job-name=flmap2d_bench
#SBATCH --account=pilot_2026_0002
#SBATCH --partition=zen5_vis
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --chdir=/sofia/scratch/pilot/pilot_2026_0002/xz/TeneT-2d-validation
#SBATCH --output=flmap2d_bench_%j.out

# Bench FLmap_parallel_2D (new 2D-distributed) vs FLmap_parallel (existing
# 1D). Run on Sofia zen5_vis (CPU) with N=4 MPI ranks arranged as a 2x2
# grid. CPU partition is sufficient for FLmap correctness validation; for
# GPU-resident timing run on zen4_h200 with the same script + BENCH_USE_GPU=1.

source /etc/profile
module load OpenMPI/5.0.7-GCC-14.2.0
export PATH=/sofia/scratch/pilot/pilot_2026_0002/xz/julia-1.11.3/bin:$PATH
export JULIA_DEPOT_PATH=/sofia/scratch/pilot/pilot_2026_0002/xz/.julia
export UCX_WARN_UNUSED_ENV_VARS=n

# Default bench knobs (override via export BENCH_FOO=... before sbatch).
# Modest D/χ for first Sofia validation — correctness is the primary goal.
export BENCH_D=${BENCH_D:-4}
export BENCH_CHI=${BENCH_CHI:-64}
export BENCH_N1=${BENCH_N1:-2}
export BENCH_USE_GPU=${BENCH_USE_GPU:-0}
export BENCH_REPEATS=${BENCH_REPEATS:-5}
export BENCH_WARMUP=${BENCH_WARMUP:-2}

echo "[$(date)] starting flmap_2d_vs_1d bench"
echo "  D=${BENCH_D}, CHI=${BENCH_CHI}, N1=${BENCH_N1}, USE_GPU=${BENCH_USE_GPU}"
mpirun -n 4 julia --project=. timing/bench_flmap_2d_vs_1d.jl
RC=$?
echo "[$(date)] bench exit=${RC}"
exit $RC
