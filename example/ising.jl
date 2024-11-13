include("./exampletensors.jl")
include("./exampleobs.jl")

using Random
using Test
using CUDA
using Zygote

@testset "$(Ni)x$(Nj) ising forward with $atype" for Ni = 1:1, Nj = 1:1, atype = [Array, CuArray]
    Random.seed!(100)
    β = 0.5
    χ = 10
    model = Ising(Ni, Nj, β)
    M = atype.(model_tensor(model, Val(:bulk)))
    alg = VUMPS(maxiter=100, miniter=1, verbosity=3, ifupdown=true)
    
    rt = @time VUMPSRuntime(M, χ, alg)
    rt = @time leading_boundary(rt, M, alg)
    env = VUMPSEnv(rt, M)

    @test observable(env, model, Val(:Z)     ) ≈ 2.789305993957602
    @test observable(env, model, Val(:mag)   ) ≈ magofβ(model) 
    @test observable(env, model, Val(:energy)) ≈ -1.745564581767667
end

@testset "$(Ni)x$(Nj) ising backward with $atype" for Ni in 1:3, Nj in 1:3, atype = [Array]
    Random.seed!(100)

    alg = VUMPS(maxiter=100, miniter=1, verbosity=3, ifupdown=true)
    χ = 10
    function logZ(β)
        model = Ising(Ni, Nj, β)
        M = atype.(model_tensor(model, Val(:bulk)))
        rt = VUMPSRuntime(M, χ, alg)
        rt = leading_boundary(rt, M, alg)
        env = VUMPSEnv(rt, M)
        return log(real(observable(env, model, Val(:Z))))
    end
    @test Zygote.gradient(β->-logZ(β), 0.5)[1] ≈ -1.745564581767667
end