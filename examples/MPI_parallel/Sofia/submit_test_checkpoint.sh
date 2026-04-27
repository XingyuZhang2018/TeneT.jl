#!/bin/bash -l
#SBATCH --job-name=Sofia_MPI_ckpt_test
#SBATCH --output=%x_%j.out
#SBATCH --partition=zen4_h200
#SBATCH --time=00:30:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:nvidia_h200:8
#SBATCH --account=pilot_2026_0002

# Sofia (VUB) — checkpoint() correctness + timing test (Plain/Recompute/Offload).
# See submit_test.sh for the LD_PRELOAD rationale.

source /etc/profile
source /etc/profile.d/modules.sh
module load GDRCopy/2.4.4-GCCcore-14.2.0 UCX-CUDA/1.18.0-GCCcore-14.2.0-CUDA-12.8.0 OpenMPI/5.0.7-GCC-14.2.0

WD=/sofia/scratch/pilot/pilot_2026_0002/xz
export JULIA_DEPOT_PATH=$WD/.julia
export HOME=$WD
JULIA=$WD/julia-1.11.3/bin/julia
MAX_GPU=${SLURM_NTASKS}

CLEAN_LD=$(echo $LD_LIBRARY_PATH | tr ':' '\n' | grep -v 'CUDA/12.8.0' | tr '\n' ':')

ENVS="export CUDA_VISIBLE_DEVICES=\$OMPI_COMM_WORLD_LOCAL_RANK; \
export UCX_TLS=rc_x,self,sm,cuda_copy,cuda_ipc; \
export UCX_MEMTYPE_CACHE=n; \
export UCX_WARN_UNUSED_ENV_VARS=n; \
export CUDA_LAUNCH_BLOCKING=1; \
export LD_PRELOAD=/usr/lib64/libcuda.so.1"

echo "=== Sofia MPI checkpoint() Test ==="
echo "Nodes: ${SLURM_NNODES}  Max GPUs: ${MAX_GPU}  Start: $(date)"

for N in 1 2 4 8 16; do
    [ $N -gt $MAX_GPU ] && continue
    echo ""
    echo "========== ${N} GPU =========="
    mpirun -np $N -x UCX_MODULE_DIR -x LD_LIBRARY_PATH=$CLEAN_LD -x PATH -x HOME -x JULIA_DEPOT_PATH \
        bash -c "$ENVS; exec $JULIA --project=../../.. ../test_MPI_checkpoint.jl"
done

echo ""
echo "=== Done: $(date) ==="
