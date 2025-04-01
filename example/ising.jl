include("./exampletensors.jl")
include("./exampleobs.jl")

using Random
using Test
using CUDA
using Zygote

@testset "pattern $pattern ising forward with $atype" for pattern in [[1;;]], atype = [Array]
    Random.seed!(100)
    β = 0.5
    χ = 10
    model = Ising(β)
    l = length(unique(pattern))
    data =[atype(model_tensor(model, Val(:bulk))) for _ in 1:l]
    M = StructArray(data, pattern)
    alg = VUMPS(maxiter=100, miniter=1, verbosity=3, ifupdown=false, ifdownfromup=true, ifgpu_cpu_combo=false)
    
    rt = @time VUMPSRuntime(M, χ, alg)
    rt = @time leading_boundary(rt, M, alg)
    env = VUMPSEnv(rt, M, alg)

    # @test observable(env, model, Val(:Z)     ) ≈ 2.789305993957602
    # @test observable(env, model, Val(:mag)   ) ≈ magofβ(model) 
    @test observable(env, model, pattern, Val(:energy)) ≈ -1.745564581767667
end

@testset "$(Ni)x$(Nj) ising backward with $atype" for Ni in 1:1, Nj in 1:1, atype = [Array]
    Random.seed!(100)

    alg = VUMPS(maxiter=100, miniter=1, maxiter_ad=5, miniter_ad=5, verbosity=3, ifupdown=false)
    χ = 10
    function logZ(β)
        model = Ising(Ni, Nj, β)
        M = atype.(model_tensor(model, Val(:bulk)))
        rt = VUMPSRuntime(M, χ, alg)
        rt = leading_boundary(rt, M, alg)
        env = VUMPSEnv(rt, M)
        return log(real(observable(env, model, Val(:Z))))
    end
    # @show Zygote.gradient(β->-logZ(β), 0.5)
    @test Zygote.gradient(β->-logZ(β), 0.5)[1] ≈ -1.745564581767667
end