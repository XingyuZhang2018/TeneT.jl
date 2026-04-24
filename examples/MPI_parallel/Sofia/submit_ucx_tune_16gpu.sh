#!/bin/bash -l
#SBATCH --job-name=Sofia_ucx_tune
#SBATCH --output=%x_%j.out
#SBATCH --partition=zen4_h200
#SBATCH --time=00:30:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:nvidia_h200:8
#SBATCH --account=pilot_2026_0002

# Scan UCX tuning knobs against 16-GPU cross-node Allreduce on CuArray.
# Each run measures TeneT.allreduce_p2p! at 128 MB across 20 reps.

source /etc/profile
source /etc/profile.d/modules.sh
module load GDRCopy/2.4.4-GCCcore-14.2.0 UCX-CUDA/1.18.0-GCCcore-14.2.0-CUDA-12.8.0 OpenMPI/5.0.7-GCC-14.2.0

WD=/sofia/scratch/pilot/pilot_2026_0002/xz
export JULIA_DEPOT_PATH=$WD/.julia
export HOME=$WD
JULIA=$WD/julia-1.11.3/bin/julia

# Print UCX defaults as seen on compute node (with CUDA plugin loaded)
echo "=== UCX defaults on compute node ==="
srun -n 1 --gpus-per-task=1 bash -c "
    source /etc/profile.d/modules.sh
    module load GDRCopy UCX-CUDA OpenMPI 2>/dev/null
    ucx_info -c 2>&1 | grep -iE 'CUDA|GDR|RNDV|GPU_DIRECT|ZCOPY|BCOPY|IPC' | head -60
"

BASE_ENVS="export CUDA_VISIBLE_DEVICES=\$OMPI_COMM_WORLD_LOCAL_RANK; \
export UCX_MEMTYPE_CACHE=n; \
export UCX_WARN_UNUSED_ENV_VARS=n; \
export CUDA_LAUNCH_BLOCKING=1; \
export LD_PRELOAD=/usr/lib64/libcuda.so.1"

run_case() {
    label="$1"
    extra="$2"
    echo ""
    echo "=== $label ==="
    echo "  extra env: $extra"
    mpirun -np 16 -x UCX_MODULE_DIR -x LD_LIBRARY_PATH -x PATH -x HOME -x JULIA_DEPOT_PATH \
        bash -c "$BASE_ENVS; $extra; exec $JULIA --project=../../.. diag_16gpu_allreduce.jl" 2>&1 \
        | grep -vE 'This may cause|was loaded|If you.re|ensure that|In any other|file an issue|^\s*│|^\s*└' \
        | grep -E 'TeneT|Phase|sib_size|(host|MPI)|MB \(|UCX.*WARN'
}

# Baseline
run_case "baseline" "export UCX_TLS=rc_x,self,sm,cuda_copy,cuda_ipc"

# Force GDR on (default is try)
run_case "GDR=yes" "export UCX_TLS=rc_x,self,sm,cuda_copy,cuda_ipc; export UCX_IB_GPU_DIRECT_RDMA=yes"

# Bigger cuda frag (default cuda:4M → cuda:16M)
run_case "FRAG=cuda:16M" "export UCX_TLS=rc_x,self,sm,cuda_copy,cuda_ipc; export UCX_RNDV_FRAG_SIZE=host:512K,cuda:16M"

# Disable cuda frag (force zcopy)
run_case "FRAG_MEM_TYPES=host" "export UCX_TLS=rc_x,self,sm,cuda_copy,cuda_ipc; export UCX_RNDV_FRAG_MEM_TYPES=host"

# Force put_zcopy rendezvous
run_case "RNDV=put_zcopy" "export UCX_TLS=rc_x,self,sm,cuda_copy,cuda_ipc; export UCX_RNDV_SCHEME=put_zcopy"

# Force get_zcopy rendezvous
run_case "RNDV=get_zcopy" "export UCX_TLS=rc_x,self,sm,cuda_copy,cuda_ipc; export UCX_RNDV_SCHEME=get_zcopy"

# Only 1 RNDV rail (default 2)
run_case "RAILS=1" "export UCX_TLS=rc_x,self,sm,cuda_copy,cuda_ipc; export UCX_MAX_RNDV_RAILS=1"

# Increase RNDV rails
run_case "RAILS=4" "export UCX_TLS=rc_x,self,sm,cuda_copy,cuda_ipc; export UCX_MAX_RNDV_RAILS=4"

# Drop zcopy threshold (always zcopy, no bcopy staging)
run_case "ZCOPY=0" "export UCX_TLS=rc_x,self,sm,cuda_copy,cuda_ipc; export UCX_ZCOPY_THRESH=0"

# UCC-like: large rndv threshold (forces eager)
run_case "RNDV_THRESH=inf" "export UCX_TLS=rc_x,self,sm,cuda_copy,cuda_ipc; export UCX_RNDV_THRESH=inf"

echo ""
echo "=== Done: $(date) ==="
