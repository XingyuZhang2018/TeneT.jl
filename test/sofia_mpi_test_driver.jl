# Standalone driver for test/test_mpi.jl — used by Sofia sbatch to run
# only the MPI regression tests, avoiding the unrelated pre-existing
# failures in test_utils.jl (save_rt signature drift) etc.
using Test, Random, CUDA, TeneT, MPI
using TeneT: split_count, split_ranges, FLmap_parallel

const ATYPES = CUDA.functional() ? [Array, CuArray] : [Array]

Random.seed!(42)

@testset "TeneT MPI-only" begin
    include("test_mpi.jl")
end
