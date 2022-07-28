include("./exampletensors.jl")
include("./exampleobs.jl")

using Random
using Test
using CUDA
using LinearAlgebra: norm
using Zygote

@testset "$(Ni)x$(Nj) ising forward with $atype" for Ni = [2], Nj = [2], atype = [Array, CuArray]
    Random.seed!(100)
    β = 0.5
    model = Ising(Ni, Nj, β)
    M = atype(model_tensor(model, Val(:bulk)))
    env = obs_env(M; χ = 10, maxiter = 10, miniter = 1, 
         infolder = "./example/data/$model/", 
        outfolder = "./example/data/$model/", 
        updown = false, verbose = true, savefile = false
        )
    @test observable(env, model, Val(:Z)     ) ≈ 2.789305993957602
    @test observable(env, model, Val(:mag)   ) ≈ magofβ(model) 
    @test observable(env, model, Val(:energy)) ≈ -1.745564581767667
end

@testset "$(Ni)x$(Nj) ising backward with $atype" for Ni = [2], Nj = [2], atype = [Array, CuArray]
    Random.seed!(100)
    function logZ(β)
        model = Ising(Ni, Nj, β)
        M = atype(model_tensor(model, Val(:bulk)))
        env = obs_env(M; χ = 10, maxiter = 10, miniter = 1, 
             infolder = "./example/data/$model/", 
            outfolder = "./example/data/$model/", 
            updown = false, verbose = true, savefile = false
            )
        -log(real(observable(env, model, Val(:Z))))
    end
    @test Zygote.gradient(β->logZ(β), 0.5)[1] ≈ -1.745564581767667
end