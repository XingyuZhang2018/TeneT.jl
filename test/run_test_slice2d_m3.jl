# Launch test_slice2d_m3.jl under 4 MPI ranks. Usage: julia --project=. test/run_test_slice2d_m3.jl
using MPI
const julia_exe = first(Base.julia_cmd().exec)
const proj = dirname(Base.active_project())
run(`$(MPI.mpiexec()) -n 4 $julia_exe --project=$proj $(joinpath(@__DIR__, "test_slice2d_m3.jl"))`)
