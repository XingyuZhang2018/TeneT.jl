#!/bin/bash -l
#SBATCH --job-name=Sofia_kernel_ab
#SBATCH --output=%x_%j.out
#SBATCH --partition=zen4_h200
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --account=pilot_2026_0002

# Single-GPU kernel A/B: staged slice2d kernels vs monolithic FLmap+forloop
# on the identical local workload (no MPI). Driver: ../benchmarks/bench_kernel_ab_sofia.jl

source /etc/profile
source /etc/profile.d/modules.sh
module load GDRCopy/2.4.4-GCCcore-14.2.0 \
            UCX-CUDA/1.18.0-GCCcore-14.2.0-CUDA-12.8.0 \
            OpenMPI/5.0.7-GCC-14.2.0 \
            CUDA/12.8.0 \
            NCCL/2.27.7-GCCcore-14.2.0-CUDA-12.8.0

WD=/sofia/scratch/pilot/pilot_2026_0002/xz
export JULIA_DEPOT_PATH=$WD/.julia
export HOME=$WD
JULIA=$WD/julia-1.11.3/bin/julia
CLEAN_LD=$(echo $LD_LIBRARY_PATH | tr ':' '\n' | grep -v 'CUDA/12.8.0' | tr '\n' ':')

export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1
export LD_PRELOAD=/usr/lib64/libcuda.so.1

echo "=== Sofia 1-GPU kernel A/B ==="
echo "Start: $(date)"
LD_LIBRARY_PATH=$CLEAN_LD $JULIA --project=../../.. ../benchmarks/bench_kernel_ab_sofia.jl
echo "=== Done: $(date) ==="
