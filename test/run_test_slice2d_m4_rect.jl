using MPI

const julia_exe = first(Base.julia_cmd().exec)
const proj = dirname(Base.active_project())
const n1 = parse(Int, get(ENV, "TENET_SLICE2D_N1", "2"))
const n2 = parse(Int, get(ENV, "TENET_SLICE2D_N2", "3"))

run(`$(MPI.mpiexec()) -n $(n1 * n2) $julia_exe --project=$proj $(joinpath(@__DIR__, "test_slice2d_m4_rect.jl"))`)
