include("./exampletensors.jl")
include("./exampleobs.jl")

using Random
using Test
using Zygote
using CUDA, AMDGPU

@testset "pattern $pattern ising forward with $atype" for pattern in [[1;;]], atype = [ROCArray]
    Random.seed!(100)
    β = 0.5
    χ = 40
    model = Ising(β)
    l = length(unique(pattern))
    TeneT.set_device_id!(atype, 1)
    # M = zeros(ComplexF64, (2,2,2,2))
    # M[2,1,1,1]=1.0
    # M[1,2,1,1]=1.0
    # M[2,2,1,1]=1.0
    # M[1,2,2,2]=1.0
    # M[2,1,2,2]=1.0
    # M[1,1,2,2]=1.0
    data =[atype(model_tensor(model, Val(:bulk))) for _ in 1:l]
    # data = [atype(M) for _ in 1:l]
    M = StructArray(data, pattern)
    alg = VUMPS(maxiter=1000, miniter=10, verbosity=3, 
                show_every = 1,
                ifsimple_eig=true,
                ifupdown=false, ifdownfromup=true, ifparallel=true)
    
    rt = @time VUMPSRuntime(M, χ, alg)
    rt = @time leading_boundary(rt, M, alg)
    env = VUMPSEnv(rt, M, alg)

    # @show log(observable(env, M, Val(:Z), alg)) - 1.0257928172049902
    # @test observable(env, M, Val(:Z), alg) ≈ 2.789305993957602
    # @test observable(env, model, Val(:mag)   ) ≈ magofβ(model) 
    # @show observable(env, model, pattern, Val(:energy), alg) 
    # @test observable(env, model, pattern, Val(:energy)) ≈ -1.745564581767667
end

@testset "ising backward with $atype $pattern" for atype = [ROCArray], pattern in [[1;;]]
    # [1;;], [1 1; 1 1], [1 2; 2 1], [1 2; 3 4], [1 1; 2 2]
    # [1 3 2 2 3 1; 2 3 1 1 3 2]
    Random.seed!(100)
    alg = VUMPS(maxiter=1000, 
                miniter=1, 
                maxiter_ad=10, 
                miniter_ad=3, 
                verbosity=2, 
                ifupdown=false, 
                ifdownfromup=true, 
                ifparallel=true,
                ifsimple_eig=false)
    χ = 40
    TeneT.set_device_id!(atype, 1)

    function energy(β)
        model = Ising(β)
        l = length(unique(pattern))
        data =[atype(model_tensor(model, Val(:bulk))) for _ in 1:l]
        M = StructArray(data, pattern)
        rt = VUMPSRuntime(M, χ, alg)
        rt′ = leading_boundary(rt, M, alg)
        env = VUMPSEnv(rt′, M, alg)
        return log(real(observable(env, M, Val(:Z), alg)))
    end
    @show energy(0.3)
    @show Zygote.gradient(energy, 0.3)[1]
    # @test Zygote.gradient(energy, 0.3)[1] ≈ num_grad(energy, 0.3) atol=1e-6
end