#!/bin/bash -l
#SBATCH --job-name=bsc_nccl_smoke
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=acc
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:2
#SBATCH --account=ehpc670
#SBATCH --qos=acc_ehpc

set -euo pipefail

module load nccl/2.24.3-1

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PROJECT_DIR="${SCRIPT_DIR}/nccl_smoke_project"
SMOKE="${SCRIPT_DIR}/nccl_smoke.jl"
export TENET_NCCL_WRAPPER="${REPO_ROOT}/src/contraction/parallel/nccl_wrapper.jl"

JULIA="${JULIA:-/home/ugen/ugen563458/tools/julia-1.11.3/bin/julia}"

echo "=== BSC 2-GPU NCCL smoke ==="
echo "script_dir=${SCRIPT_DIR}"
echo "repo_root=${REPO_ROOT}"
echo "start=$(date)"
echo "nodes=${SLURM_NNODES} tasks=${SLURM_NTASKS} gpus=${SLURM_GPUS:-unset}"
echo "julia=${JULIA}"
echo "mpirun=$(command -v mpirun)"

export JULIA_DEPOT_PATH="${HOME}/.julia:"
export JULIA_NUM_THREADS=1

"${JULIA}" --project="${PROJECT_DIR}" --startup-file=no -e 'using MPI, CUDA; include(ENV["TENET_NCCL_WRAPPER"]); println("PROJECT_LOAD_OK")'

mpirun -np 2 bash -c "export CUDA_VISIBLE_DEVICES=\${OMPI_COMM_WORLD_LOCAL_RANK}; export TENET_USE_NCCL=1; export NCCL_DEBUG=WARN; export TENET_NCCL_WRAPPER=${TENET_NCCL_WRAPPER}; exec ${JULIA} --project=${PROJECT_DIR} --startup-file=no ${SMOKE}"

echo "done=$(date)"
