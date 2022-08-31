using CUDA
using Random
using Test
using VUMPS
using VUMPS: vumps, vumps_env, env_norm

@testset "$(Ni)x$(Nj) VUMPSRuntime with $atype{$dtype}" for Ni = [1,2,3], Nj = [1,2,3], atype = [Array], dtype = [Float64, ComplexF64]
    Random.seed!(100)
    @test SquareLattice <: AbstractLattice

    M = atype(rand(dtype,2,2,2,2,Ni,Nj))
    rt = SquareVUMPSRuntime(M, Val(:random), 4)
end

@testset "$(Ni)x$(Nj) vumps with $atype{$dtype}" for Ni = [2], Nj = [2], atype = [Array], dtype = [ComplexF64]
    Random.seed!(100)
    M = atype(rand(dtype,2,2,2,2,Ni,Nj))
    rt = SquareVUMPSRuntime(M, Val(:random), 2)
    env = vumps(rt; verbose = true)
    @test env !== nothing
end

@testset "$(Ni)x$(Nj) vumps_env with $atype{$dtype}" for Ni = [2], Nj = [2], atype = [Array], dtype = [Float64, ComplexF64]
    Random.seed!(100)
    M = atype(rand(dtype,2,2,2,2,Ni,Nj))
    env = vumps_env(M; χ = 10, verbose = true)
    @test env !== nothing
end

@testset "$(Ni)x$(Nj) obs_env" for Ni = [2], Nj = [2], atype = [CuArray], dtype = [Float64, ComplexF64]
    Random.seed!(100)
    M = atype(rand(dtype,2,2,2,2,Ni,Nj))
    env = obs_env(M; χ = 10, verbose = true, savefile = true, infolder = "./test/data1/", outfolder = "./test/data2/", show_every = 3)
    @test env !== nothing
end