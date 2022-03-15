include("./exampletensors.jl")
include("./exampleobs.jl")

using CUDA
using Random
using Test
using VUMPS: parity_conserving
using Zygote

@testset "$(Ni)x$(Nj) rand forward with $(symmetry) symmetry $atype array" for Ni = [1], Nj = [1], atype = [Array], symmetry in [:U1]
    Random.seed!(10)
    m = randinitial(Val(symmetry), atype, ComplexF64, 4, 4, 4, 4; dir = [-1,1,1,-1])
    # m = asArray(m)
    # m = parity_conserving(m)
    # m = asSymmetryArray(m, symmetry; dir = [-1,1,1,-1])
    β = 0.2
    M = [β * m for i in 1:Ni, j in 1:Nj]
    env = obs_env(M; χ = 10, verbose = true, savefile = true, infolder = "./example/data/$(Ni)x$(Nj)rand/$symmetry/", outfolder = "./example/data/$(Ni)x$(Nj)rand/$symmetry/", maxiter = 10, miniter = 10, updown = false)
    Zsymmetry = Z(env, M)
    m = asArray(m)
    # @show m
    M = [β * m for i in 1:Ni, j in 1:Nj]
    env = obs_env(M; χ = 10, verbose = true, savefile = true, infolder = "./example/data/$(Ni)x$(Nj)rand/$(symmetry)_none/", outfolder = "./example/data/$(Ni)x$(Nj)rand/$(symmetry)_none/", maxiter = 10, miniter = 10, updown = false)
    Znone = Z(env, M)
    @show norm(Zsymmetry-Znone)
    @test Zsymmetry ≈ Znone
end

@testset "$(Ni)x$(Nj) rand backward with $(symmetry) symmetry $atype array" for Ni = [1], Nj = [1], atype = [CuArray], symmetry in [:none, :Z2]
    Random.seed!(100)
    m = randinitial(Val(:none), atype, ComplexF64, 4, 4, 4, 4)
    m = parity_conserving(m)
    symmetry == :Z2 && (m = asZ2Array(m))
    function foo(β)
        M = [β * m for i in 1:Ni, j in 1:Nj]
        env = obs_env(M; χ = 20, verbose = true, savefile = true, infolder = "./example/data/$(Ni)x$(Nj)rand/$symmetry/", outfolder = "./example/data/$(Ni)x$(Nj)rand/$symmetry/", maxiter = 10, miniter = 10, updown = false)
        real(Z(env, M))
    end
    @show Zygote.gradient(foo, 0.2)
end