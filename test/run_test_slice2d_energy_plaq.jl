# Launcher for the distributed energy_value parity gate (4 ranks, CPU).
# Usage: julia --project=. test/run_test_slice2d_energy_plaq.jl
using MPI
const julia_exe = first(Base.julia_cmd().exec)
const proj = dirname(Base.active_project())
run(`$(MPI.mpiexec()) -n 4 $julia_exe --project=$proj $(joinpath(@__DIR__, "test_slice2d_energy_plaq.jl"))`)
