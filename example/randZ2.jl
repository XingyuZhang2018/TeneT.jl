include("./exampletensors.jl")
include("./exampleobs.jl")

using Random
using Test
using VUMPS: tensor2Z2tensor, parity_conserving
using Zygote

@testset "$(Ni)x$(Nj) rand forward with $(symmetry) symmetry $atype array" for Ni = [1], Nj = [1], atype = [CuArray], symmetry in [:none, :Z2]
    Random.seed!(100)
    m = randinitial(Val(:none), atype, ComplexF64, 2, 2, 2, 2)
    m = parity_conserving(m)
    symmetry == :Z2 && (m = tensor2Z2tensor(m))
    β = 0.2
    M = [β * m for i in 1:Ni, j in 1:Nj]
    env = obs_env(M; χ = 20, verbose = true, savefile = true, infolder = "./example/data/$(Ni)x$(Nj)rand/$symmetry/", outfolder = "./example/data/$(Ni)x$(Nj)rand/$symmetry/", maxiter = 10, miniter = 10, updown = false)
    @show Z(env, M)
end

@testset "$(Ni)x$(Nj) rand backward with $(symmetry) symmetry $atype array" for Ni = [2], Nj = [2], atype = [Array], symmetry in [:none, :Z2]
    Random.seed!(100)
    m = randinitial(Val(:none), atype, ComplexF64, 2, 2, 2, 2)
    m = parity_conserving(m)
    symmetry == :Z2 && (m = tensor2Z2tensor(m))
    function foo(β)
        M = [β * m for i in 1:Ni, j in 1:Nj]
        env = obs_env(M; χ = 20, verbose = true, savefile = true, infolder = "./example/data/$(Ni)x$(Nj)rand/$symmetry/", outfolder = "./example/data/$(Ni)x$(Nj)rand/$symmetry/", maxiter = 10, miniter = 10, updown = false)
        real(Z(env, M))
    end
    @show Zygote.gradient(foo, 0.2)
end