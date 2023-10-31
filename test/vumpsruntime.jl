using CUDA
using Random
using Test
using TeneT
using TeneT: vumps, vumps_env

@testset "$(Ni)x$(Nj) VUMPSRuntime with $(symmetry) $atype{$dtype}" for Ni = [2], Nj = [2], atype = [Array], dtype = [ComplexF64], symmetry in [:none, :Z2, :U1]
    @test SquareLattice <: AbstractLattice

    d = 4
    M = [randinitial(Val(symmetry), atype, dtype, d, d, d, d; dir = [-1,1,1,-1]) for i in 1:Ni, j in 1:Nj]
    rt = SquareVUMPSRuntime(M, Val(:random), 4; dir = [-1,1,1])
end

@testset "$(Ni)x$(Nj) vumps with $(symmetry) $atype{$dtype}" for Ni = [2], Nj = [2], atype = [Array], dtype = [ComplexF64], symmetry in [:none, :Z2, :U1]
    d = 2
    M = [randinitial(Val(symmetry), atype, dtype, d, d, d, d; dir = [-1,1,1,-1]) for i in 1:Ni, j in 1:Nj]
    rt = SquareVUMPSRuntime(M, Val(:random), 4; dir = [-1,1,1])
    env = vumps(rt)
    @test env !== nothing
end

@testset "$(Ni)x$(Nj) vumps_env with $(symmetry) $atype{$dtype}" for Ni = [2], Nj = [2], atype = [Array], dtype = [ComplexF64], symmetry in [:none, :Z2, :U1]
    d = 2
    M = [randinitial(Val(symmetry), atype, dtype, d, d, d, d; dir = [-1,1,1,-1]) for i in 1:Ni, j in 1:Nj]
    env = vumps_env(M; χ = 10)
    @test env !== nothing
end

@testset "$(Ni)x$(Nj) obs_env with $(symmetry) $atype{$dtype}" for Ni = [2], Nj = [2], atype = [Array], dtype = [ComplexF64], symmetry in [:none, :Z2, :U1]
    d = 4
    M = [randinitial(Val(symmetry), atype, dtype, d, d, d, d; dir = [-1,1,1,-1]) for i in 1:Ni, j in 1:Nj]
    env = obs_env(M; χ = 10, verbose = true, savefile = true, infolder = "./test/data/$(symmetry)/", outfolder = "./test/data/$(symmetry)/", show_every = 3, maxiter = 10)
    @test env !== nothing
end