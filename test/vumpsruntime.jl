using CUDA
using Random
using Test
using VUMPS
using VUMPS: vumps, vumps_env

@testset "$(Ni)x$(Nj) VUMPSRuntime with $atype{$dtype}" for Ni = [1,2,3], Nj = [1,2,3], atype = [Array, CuArray], dtype = [Float64, ComplexF64]
    @test SquareLattice <: AbstractLattice

    M = Array{atype{dtype,4},2}(undef, Ni, Nj)
    for j = 1:Nj, i = 1:Ni
        M[i,j] = atype(rand(dtype,2,2,2,2))
    end
    rt = SquareVUMPSRuntime(M, Val(:random), 2)
end

@testset "$(Ni)x$(Nj) vumps with $atype{$dtype}" for Ni = [2], Nj = [2], atype = [Array, CuArray], dtype = [Float64, ComplexF64]
    M = Array{atype{dtype,4},2}(undef, Ni, Nj)
    for j = 1:Nj, i = 1:Ni
        M[i,j] = atype(rand(dtype,2,2,2,2))
    end
    rt = SquareVUMPSRuntime(M, Val(:random), 2)
    env = vumps(rt)
    @test env !== nothing
end

@testset "$(Ni)x$(Nj) vumps_env with $atype{$dtype}" for Ni = [2], Nj = [2], atype = [Array, CuArray], dtype = [Float64, ComplexF64]
    M = Array{atype{dtype,4},2}(undef, Ni, Nj)
    for j = 1:Nj, i = 1:Ni
        M[i,j] = atype(rand(dtype,2,2,2,2))
    end
    env = vumps_env(M; χ = 10)
    @test env !== nothing
end

@testset "$(Ni)x$(Nj) obs_env" for Ni = [2], Nj = [2], atype = [Array, CuArray], dtype = [Float64, ComplexF64]
    M = Array{atype{dtype,4},2}(undef, Ni, Nj)
    for j = 1:Nj, i = 1:Ni
        M[i,j] = atype(rand(dtype,2,2,2,2))
    end
    env = obs_env(M; χ = 10, verbose = true, savefile = true, infolder = "./data1", outfolder = "./data2", show_every = 3)
    @test env !== nothing
end