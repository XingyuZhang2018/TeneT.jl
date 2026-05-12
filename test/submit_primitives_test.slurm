#!/bin/bash -l
#SBATCH --job-name=t1_3_6_prim
#SBATCH --account=pilot_2026_0002
#SBATCH --partition=zen5_vis
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --chdir=/sofia/scratch/pilot/pilot_2026_0002/xz/TeneT-2d-validation
#SBATCH --output=test_t1_3_6_%j.out

source /etc/profile
module load OpenMPI/5.0.7-GCC-14.2.0
export PATH=/sofia/scratch/pilot/pilot_2026_0002/xz/julia-1.11.3/bin:$PATH
export JULIA_DEPOT_PATH=/sofia/scratch/pilot/pilot_2026_0002/xz/.julia
export UCX_WARN_UNUSED_ENV_VARS=n

echo '[Tasks 1.3-1.6] Running distributed_primitives_test.jl with N=4...'
mpirun -n 4 julia --project=. test/distributed_primitives_test.jl
echo '[Tasks 1.3-1.6] exit:' $?
