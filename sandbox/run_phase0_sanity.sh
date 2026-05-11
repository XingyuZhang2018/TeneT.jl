#!/bin/bash
# Phase 0 sandbox driver for Sofia (or any 4-MPI-rank Linux host).
#
# Runs all 4 Phase 0 sandbox scripts and prints clear PASS/FAIL summary.
# Does NOT need GPUs — pure MPI cart grid + comms on CPU.
#
# Usage on Sofia:
#   salloc -p zen4_h200 -N 1 --ntasks=4 --time=00:30:00
#   cd /sofia/scratch/pilot/pilot_2026_0002/xz/TeneT.jl    # or wherever
#   git fetch origin claude/2d-distributed-vumps
#   git worktree add /tmp/2d-validation claude/2d-distributed-vumps
#   cp Manifest.toml /tmp/2d-validation/  # Manifest is gitignored
#   cd /tmp/2d-validation
#   bash sandbox/run_phase0_sanity.sh
#
# Or non-interactive sbatch:
#   sbatch sandbox/submit_phase0_sanity.slurm    # if you want batch
#
# Exit code 0 = all 4 pass; non-zero = something failed.

set -uo pipefail
cd "$(dirname "$0")/.."

# Strip CUDA from LD_LIBRARY_PATH for Julia (per memory note reference_sofia_ld_path)
export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v cuda | tr '\n' ':' | sed 's/:$//')

# Sofia UCX/MPI fix (per memory note project_sofia_hpc_ucx_fix) — needed if any
# CUDA-aware MPI startup probing happens. Harmless if /usr/lib64/libcuda.so.1
# doesn't exist.
[ -f /usr/lib64/libcuda.so.1 ] && export LD_PRELOAD=/usr/lib64/libcuda.so.1

# UCX warns about UCX_MEMTYPE_CACHE / UCX_ERROR_SIGNALS env vars that the
# system module sets but UCX 1.18 doesn't use; suppress to keep output clean.
export UCX_WARN_UNUSED_ENV_VARS=n

# Discover Julia (env-specific)
JULIA=${JULIA:-julia}
command -v "$JULIA" >/dev/null 2>&1 || { echo "ERROR: julia not on PATH"; exit 1; }
echo "Julia: $($JULIA --version)"

# Discover MPI launcher
if command -v mpirun >/dev/null 2>&1; then
    MPIRUN="mpirun"
elif command -v srun >/dev/null 2>&1; then
    MPIRUN="srun"
else
    echo "ERROR: neither mpirun nor srun found"; exit 1
fi
echo "MPI launcher: $MPIRUN"

# Instantiate once before launching parallel ranks (avoid race on shared Manifest)
echo "[setup] Instantiating Julia env (first run only; can take a few minutes)..."
$JULIA --project=. -e 'using Pkg; Pkg.instantiate()' 2>&1 | tail -5

PASS=0
FAIL=0
FAIL_LIST=()

run_test() {
    local name=$1
    local nranks=$2
    local script=$3
    echo ""
    echo "========================================================================"
    echo "[Phase 0 sandbox] $name (N=$nranks ranks)"
    echo "========================================================================"

    if [ "$nranks" == "1" ]; then
        $JULIA --project=. "$script"
    else
        $MPIRUN -n "$nranks" $JULIA --project=. "$script"
    fi
    local rc=$?

    if [ $rc -eq 0 ]; then
        echo "  -> PASS"
        PASS=$((PASS+1))
    else
        echo "  -> FAIL (exit $rc)"
        FAIL=$((FAIL+1))
        FAIL_LIST+=("$name")
    fi
}

run_test "Task 0.1 grid sanity"           4 "sandbox/2d_grid_sanity.jl"
run_test "Task 0.2 allgather sanity"      4 "sandbox/2d_allgather_sanity.jl"
run_test "Task 0.3 alltoall (N1=N2=2)"    4 "sandbox/2d_alltoall_sanity.jl"
run_test "Task 0.4 Zygote rrule mock"     1 "sandbox/2d_grad_sanity.jl"

echo ""
echo "========================================================================"
echo "Phase 0 sandbox summary: $PASS pass, $FAIL fail"
echo "========================================================================"
if [ $FAIL -gt 0 ]; then
    echo "Failed: ${FAIL_LIST[*]}"
    exit 1
fi
echo "All Phase 0 sandbox checks PASS — ready to proceed to Phase 1"
