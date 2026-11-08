using Test
using Random
using MPI
using CUDA
using TeneT
using TeneT: slice2d_grid, _acc_to_ar_tslq_one

if !MPI.Initialized()
    MPI.Init()
end

@testset "General Slice2D AR seam runs on local 1x1 CuArray" begin
    if !CUDA.functional()
        @info "Skipping local Slice2D CuArray smoke: CUDA is not functional"
        @test true
    else
        CUDA.allowscalar(false)
        CUDA.device!(0)
        Random.seed!(20260620)

        grid = slice2d_grid(1, 1)
        chi, D = 16, 2
        AC = CuArray(rand(Float64, chi, D, D, chi))
        C = CuArray(rand(Float64, chi, chi))

        AR, err = _acc_to_ar_tslq_one(AC, C, grid)
        CUDA.synchronize()

        @test AR isa CuArray
        @test size(AR) == size(AC)
        @test isfinite(err)
    end
end
