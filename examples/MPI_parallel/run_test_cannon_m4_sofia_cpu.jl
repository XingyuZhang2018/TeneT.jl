# Local CPU smoke for the Sofia M4 Batch-A validation driver: launches
# test_cannon_m4_sofia.jl under 4 MPI ranks on plain Arrays (TENET_BENCH_CPU=1,
# tiny dims, NO GPU/CUDA needed). Confirms the driver parses + runs end-to-end —
# the SAME leftenv_cannon code the cluster run exercises — before it reaches Sofia.
# (The cluster run uses Sofia/submit_test_cannon_m4.sh on CuArrays, 4×4 = 16 GPU.)
#
# Usage: julia --project=. examples/MPI_parallel/run_test_cannon_m4_sofia_cpu.jl
using MPI
const julia_exe = first(Base.julia_cmd().exec)
const proj = dirname(Base.active_project())
const driver = joinpath(@__DIR__, "test_cannon_m4_sofia.jl")
withenv("TENET_BENCH_CPU" => "1") do
    run(`$(MPI.mpiexec()) -n 4 $julia_exe --project=$proj $driver`)
end
