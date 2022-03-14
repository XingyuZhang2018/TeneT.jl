include("./exampletensors.jl")
include("./exampleobs.jl")

using Random
using Test
using VUMPS: asZ2Array, parity_conserving
using CUDA
using Zygote

@testset "$(Ni)x$(Nj) ising forward with $(symmetry) symmetry $atype array" for Ni = [1], Nj = [1], atype = [Array], symmetry in [:none, :Z2]
    β = 0.2
    model = Ising(Ni, Nj, β)
    M = model_tensor(model)
    M = parity_conserving.(M)
    symmetry == :Z2 && (M = asZ2Array.(M))
    env = obs_env(M; χ = 20, verbose = true, savefile = true, infolder = "./example/data/$model/$symmetry/", outfolder = "./example/data/$(Ni)x$(Nj)rand/$symmetry/", maxiter = 10, miniter = 10, updown = false)
    @show Z(env, M) # 2.08450374046259
end

@testset "$(Ni)x$(Nj) ising backward with $(symmetry) symmetry $atype array" for Ni = [1], Nj = [1], atype = [Array], symmetry in [:none, :Z2]
    Random.seed!(100)
    β = 0.2
    model = Ising(Ni, Nj, β)
    M = model_tensor(model)
    M = parity_conserving.(M)
    symmetry == :Z2 && (M = asZ2Array.(M))
    function foo(β)
        M = [β * M[1] for i in 1:Ni, j in 1:Nj]
        env = obs_env(M; χ = 20, verbose = true, savefile = true, infolder = "./example/data/$model/$symmetry/", outfolder = "./example/data/$(Ni)x$(Nj)rand/$symmetry/", maxiter = 10, miniter = 10, updown = false)
        real(Z(env, M))
    end
    @show Zygote.gradient(foo, 0.2)
end