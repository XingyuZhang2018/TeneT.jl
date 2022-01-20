using CUDA
using Random
using Test
using VUMPS
using VUMPS: vumps, vumps_env

@testset "$(Ni)x$(Nj) VUMPSRuntime with $(symmetry) $atype{$dtype}" for Ni = [1,2,3], Nj = [1,2,3], atype = [Array, CuArray], dtype = [ComplexF64], symmetry in [:none, :Z2]
    @test SquareLattice <: AbstractLattice

    d = 4
    M = [randinitial(Val(symmetry), atype, dtype, d, d, d, d) for i in 1:Ni, j in 1:Nj]
    rt = SquareVUMPSRuntime(M, Val(:random), 2)
end

@testset "$(Ni)x$(Nj) vumps with $(symmetry) $atype{$dtype}" for Ni = [2], Nj = [2], atype = [Array, CuArray], dtype = [Float64, ComplexF64], symmetry in [:none, :Z2]
    d = 2
    M = [randinitial(Val(symmetry), atype, dtype, d, d, d, d) for i in 1:Ni, j in 1:Nj]
    rt = SquareVUMPSRuntime(M, Val(:random), 2)
    env = vumps(rt)
    @test env !== nothing
end

@testset "$(Ni)x$(Nj) vumps_env with $(symmetry) $atype{$dtype}" for Ni = [2], Nj = [2], atype = [Array, CuArray], dtype = [Float64, ComplexF64], symmetry in [:none, :Z2]
    d = 2
    M = [randinitial(Val(symmetry), atype, dtype, d, d, d, d) for i in 1:Ni, j in 1:Nj]
    env = vumps_env(M; χ = 10)
    @test env !== nothing
end

@testset "$(Ni)x$(Nj) obs_env with $(symmetry) $atype{$dtype}" for Ni = [2], Nj = [2], atype = [CuArray], dtype = [ComplexF64], symmetry in [:Z2]
    d = 4
    M = [randinitial(Val(symmetry), atype, dtype, d, d, d, d) for i in 1:Ni, j in 1:Nj]
    env = obs_env(M; χ = 10, verbose = true, savefile = true, infolder = "./data2", outfolder = "./data2", show_every = 3, maxiter = 10)
    @test env !== nothing
end