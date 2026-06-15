# Launcher for the M4 Batches A–D distributed-env tests (4 ranks, CPU).
# Usage: julia --project=. test/run_test_cannon_m4.jl
using MPI
const julia_exe = first(Base.julia_cmd().exec)
const proj = dirname(Base.active_project())
run(`$(MPI.mpiexec()) -n 4 $julia_exe --project=$proj $(joinpath(@__DIR__, "test_cannon_m4.jl"))`)
