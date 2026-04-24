#!/bin/bash -l
#SBATCH --job-name=mpi_regression
#SBATCH --output=%x_%j.out
#SBATCH --partition=zen4_h200
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:nvidia_h200:4
#SBATCH --account=pilot_2026_0002

# Targeted regression driver for test/test_mpi.jl (MPI p2p collectives).
# Uses sofia_mpi_test_driver.jl to avoid the full runtests.jl suite and
# its unrelated pre-existing failures (save_rt etc.).

source /etc/profile
source /etc/profile.d/modules.sh
module load GDRCopy/2.4.4-GCCcore-14.2.0 UCX-CUDA/1.18.0-GCCcore-14.2.0-CUDA-12.8.0 OpenMPI/5.0.7-GCC-14.2.0

WD=/sofia/scratch/pilot/pilot_2026_0002/xz
export JULIA_DEPOT_PATH=$WD/.julia
export HOME=$WD
JULIA=$WD/julia-1.11.3/bin/julia

ENVS="export CUDA_VISIBLE_DEVICES=\$OMPI_COMM_WORLD_LOCAL_RANK; \
export UCX_TLS=rc_x,self,sm,cuda_copy,cuda_ipc; \
export UCX_MEMTYPE_CACHE=n; \
export UCX_WARN_UNUSED_ENV_VARS=n; \
export LD_PRELOAD=/usr/lib64/libcuda.so.1"

echo "=== Sofia MPI regression (test_mpi.jl only) ==="
echo "Start: $(date)"

cd $WD/TeneT.jl

for N in 2 4; do
    echo ""
    echo "========== mpiexec -n ${N} =========="
    mpirun -np $N -x UCX_MODULE_DIR -x LD_LIBRARY_PATH -x PATH -x HOME -x JULIA_DEPOT_PATH \
        bash -c "$ENVS; exec $JULIA --project=. test/sofia_mpi_test_driver.jl"
    echo "---------- mpiexec -n ${N} exit=$? ----------"
done

echo ""
echo "=== Done: $(date) ==="
