using CUDA
using Random
using Test
using TeneT
using TeneT: vumps, vumps_env

@testset "$(Ni)x$(Nj) VUMPSRuntime with $(symmetry) $atype{$dtype}" for Ni = [2], Nj = [2], stype = [electronZ2()], atype = [Array], dtype = [ComplexF64], symmetry in [:none, :U1]
    @test SquareLattice <: AbstractLattice

    ST = SymmetricType(Val(symmetry), stype, atype, dtype)
    qnD = [0,1]
    qnχ = [0,1]
    dimsD = [1,1]
    dimsχ = [3,3]
    alg = VUMPS(χ=sum(dimsχ), U1info=(qnD, qnχ, dimsD, dimsχ))
    D2 = sum(dimsD)^2
    M = [randinitial(ST, D2,D2,D2,D2; dir = [-1,1,1,-1]) for i in 1:Ni, j in 1:Nj]
    rt = SquareVUMPSRuntime(M, Val(:random), alg.χ; verbose = alg.verbose, U1info = alg.U1info)
end

@testset "$(Ni)x$(Nj) vumps with $(symmetry) $atype{$dtype}" for Ni = [2], Nj = [2], stype = [electronZ2()], atype = [Array], dtype = [ComplexF64], symmetry in [:none, :U1]

    ST = SymmetricType(Val(symmetry), stype, atype, dtype)
    qnD = [0,1]
    qnχ = [0,1]
    dimsD = [1,1]
    dimsχ = [3,3]
    alg = VUMPS(χ=sum(dimsχ), U1info=(qnD, qnχ, dimsD, dimsχ))
    D2 = sum(dimsD)^2
    M = [randinitial(ST, D2,D2,D2,D2; dir = [-1,1,1,-1]) for i in 1:Ni, j in 1:Nj]
    rt = SquareVUMPSRuntime(M, Val(:random), alg.χ; verbose = alg.verbose, U1info = alg.U1info)
    env = vumps(rt, alg)
    @test env !== nothing
end

@testset "$(Ni)x$(Nj) vumps_env with $(symmetry) $atype{$dtype}" for Ni = [2], Nj = [2], stype = [electronZ2()], atype = [Array], dtype = [ComplexF64], symmetry in [:none, :U1]
    
    ST = SymmetricType(Val(symmetry), stype, atype, dtype)
    qnD = [0,1]
    qnχ = [0,1]
    dimsD = [1,1]
    dimsχ = [3,3]
    alg = VUMPS(χ=sum(dimsχ), U1info=(qnD, qnχ, dimsD, dimsχ), savefile=false)
    D2 = sum(dimsD)^2
    M = [randinitial(ST, D2,D2,D2,D2; dir = [-1,1,1,-1]) for i in 1:Ni, j in 1:Nj]
    env = vumps_env(M, alg; direction = "up")
    @test env !== nothing
end

@testset "$(Ni)x$(Nj) obs_env with $(symmetry) $atype{$dtype}" for Ni = [2], Nj = [2], stype = [electronZ2()], atype = [Array], dtype = [ComplexF64], symmetry in [:none, :U1]
    
    ST = SymmetricType(Val(symmetry), stype, atype, dtype)
    qnD = [0,1]
    qnχ = [0,1]
    dimsD = [1,1]
    dimsχ = [3,3]
    alg = VUMPS(χ=sum(dimsχ), U1info=(qnD, qnχ, dimsD, dimsχ), savefile=false)
    D2 = sum(dimsD)^2
    M = [randinitial(ST, D2,D2,D2,D2; dir = [-1,1,1,-1]) for i in 1:Ni, j in 1:Nj]
    env = obs_env(M, alg)
    @test env !== nothing
end