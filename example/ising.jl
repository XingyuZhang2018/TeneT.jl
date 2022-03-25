include("./exampletensors.jl")
include("./exampleobs.jl")

using Random
using Test
using VUMPS: asZ2Array, parity_conserving
using CUDA
using LinearAlgebra: norm
using Zygote

@testset "$(Ni)x$(Nj) ising forward with $(symmetry) symmetry $atype array" for Ni = [2], Nj = [2], atype = [Array], symmetry in [:Z2, :U1]
    β = 0.2
    model = Ising(Ni, Nj, β)
    M = model_tensor(model)
    # M = parity_conserving.(M)
    # symmetry == :Z2 && (M = asZ2Array.(M))
    M = asSymmetryArray.(M, symmetry; dir = [-1,1,1,-1])
    env = obs_env(M; χ = 10, verbose = true, savefile = false, infolder = "./example/data/$model/$symmetry/", outfolder = "./example/data/$(Ni)x$(Nj)rand/$symmetry/", maxiter = 10, miniter = 10, updown = false)
    Zsymmetry = Z(env, M)

    M = asArray.(M)
    env = obs_env(M; χ = 10, verbose = true, savefile = false, infolder = "./example/data/$model/$symmetry/", outfolder = "./example/data/$(Ni)x$(Nj)rand/$symmetry/", maxiter = 10, miniter = 10, updown = false)
    Znone = Z(env, M)
    @show norm(Zsymmetry-Znone)
    @test Zsymmetry ≈ Znone
end

@testset "$(Ni)x$(Nj) ising backward with $(symmetry) symmetry $atype array" for Ni = [1], Nj = [1], atype = [Array], symmetry in [:U1]
    Random.seed!(100)
    β = 0.2
    model = Ising(Ni, Nj, β)
    M = model_tensor(model)
    M = asSymmetryArray.(M, symmetry; dir = [-1,1,1,-1])
    function foo(M, β)
        M = [β * M[1] for i in 1:Ni, j in 1:Nj]
        env = obs_env(M; χ = 10, verbose = true, savefile = false, infolder = "./example/data/$model/$symmetry/", outfolder = "./example/data/$(Ni)x$(Nj)rand/$symmetry/", maxiter = 10, miniter = 10, updown = false)
        real(Z(env, M))
    end
    dZsymmetry = Zygote.gradient(x->foo(M,x), 0.2)[1]
    M = asArray.(M)
    dZnone = Zygote.gradient(x->foo(M,x), 0.2)[1]
    @show norm(dZsymmetry-dZnone) dZsymmetry dZnone
    @test dZsymmetry ≈ dZnone
end