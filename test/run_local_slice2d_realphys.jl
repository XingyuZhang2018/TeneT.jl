# Launcher: julia --project=. test/run_local_slice2d_realphys.jl  (4 ranks, CPU)
using MPI
const julia_exe = first(Base.julia_cmd().exec)
const proj = dirname(Base.active_project())
run(`$(MPI.mpiexec()) -n 4 $julia_exe --project=$proj $(joinpath(@__DIR__, "local_slice2d_realphys.jl"))`)
