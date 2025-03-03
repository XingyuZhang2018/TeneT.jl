include("./exampletensors.jl")
include("./exampleobs.jl")

using Random
using Test
using CUDA
using Zygote

@testset "2D classical ising" for atype in [Array], pattern in [[1;;], [1 1; 1 1], [1 2; 2 1], [1 2; 3 4], [1 1; 2 2], [1 2; 3 4; 5 6],
    [1 2 3 4 5 6; 4 5 6 1 2 3], [1 4; 2 5; 3 6; 4 1;5 2; 6 3]]
    # [1;;], [1 1; 1 1], [1 2; 2 1], [1 2; 3 4], [1 1; 2 2]
    # [1 3 2 2 3 1; 2 3 1 1 3 2]
    β = 0.5
    χ = ℂ^10
    model = Ising(β)
    l = length(unique(pattern))
    data =[TensorMap(atype(model_tensor(model, Val(:bulk))), ℂ^2*ℂ^2 ← ℂ^2*ℂ^2) for _ in 1:l]
    M = StructArray(data, pattern)
    alg = VUMPS(maxiter=100, miniter=1, verbosity=1, ifupdown=false, ifdownfromup = false)
    
    rt = @time VUMPSRuntime(M, χ, alg)
    rt = @time leading_boundary(rt, M, alg)
    env = VUMPSEnv(rt, M, alg)

    @test rt isa VUMPSRuntime

    env = VUMPSEnv(rt, M, alg)
    @test env isa VUMPSEnv

    @test observable(env, model, pattern, Val(:Z); alg) ≈ 2.789305993957602
end 

@testset "ising backward with $atype $ifupdown $pattern" for atype = [Array], ifupdown in [true], pattern in [[1;;], [1 1; 1 1], [1 2; 2 1], [1 2; 3 4], [1 1; 2 2], [1 2; 3 4; 5 6],
    [1 2 3 4 5 6; 4 5 6 1 2 3], [1 4; 2 5; 3 6; 4 1;5 2; 6 3]]
    Random.seed!(100)
    alg = VUMPS(maxiter=10, 
                miniter=1, 
                maxiter_ad=10, 
                miniter_ad=5, 
                verbosity=1, 
                ifupdown=ifupdown, 
                ifdownfromup=true, 
                ifsimple_eig=false)
    χ = ℂ^10

    function logZ(β)
        model = Ising(β)
        l = length(unique(pattern))
        data = [TensorMap(atype(model_tensor(model, Val(:bulk))), ℂ^2*ℂ^2 ← ℂ^2*ℂ^2) for _ in 1:l]
        M = StructArray(data, pattern)
        rt = VUMPSRuntime(M, χ, alg)
        rt′ = leading_boundary(rt, M, alg)
        env = VUMPSEnv(rt′, M, alg)
        return log(real(observable(env, model, pattern, Val(:Z); alg)))
    end

    function energy(β)
        model = Ising(β)
        l = length(unique(pattern))
        data = [TensorMap(atype(model_tensor(model, Val(:bulk))), ℂ^2*ℂ^2 ← ℂ^2*ℂ^2) for _ in 1:l]
        M = StructArray(data, pattern)
        rt = VUMPSRuntime(M, χ, alg)
        rt′ = leading_boundary(rt, M, alg)
        env = VUMPSEnv(rt′, M, alg)
        return real(observable(env, model, pattern, Val(:energy)))
    end

    β = 0.3
    @test Zygote.gradient(β->-logZ(β), β)[1] ≈ num_grad(β->-logZ(β), β) ≈ energy(β)
    @test Zygote.gradient(energy, β)[1] ≈ num_grad(energy, β)
end                            