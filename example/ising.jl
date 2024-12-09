include("./exampletensors.jl")
include("./exampleobs.jl")

using Random
using Test
using CUDA
using Zygote

@testset "$(Ni)x$(Nj) ising forward with $atype" for Ni = 1:1, Nj = 2:2, atype = [Array]
    Random.seed!(100)
    β = 0.5
    χ = 10
    model = Ising(Ni, Nj, β)
    M = atype.(model_tensor(model, Val(:bulk)))
    L = HoneycombLattice
    # L = SquareLattice
    alg = VUMPS{L}(maxiter=100, miniter=1, verbosity=3, ifupdown=false, ifdownfromup = true)
    
    rt = @time VUMPSRuntime(M, χ, alg)
    @show typeof(rt)
    rt = @time leading_boundary(rt, M, alg)
    # env = VUMPSEnv(rt, M)

    @test observable(rt, model, Val(:Z)     ) ≈ 2.789305993957602
    @test observable(rt, model, Val(:mag)   ) ≈ magofβ(model) 
    @test observable(rt, model, Val(:energy)) ≈ -1.745564581767667
end

@testset "$(Ni)x$(Nj) ising backward with $atype" for Ni in 1:1, Nj in 1:1, atype = [Array]
    Random.seed!(100)

    alg = VUMPS{HoneycombLattice}(maxiter=100, miniter=1, verbosity=3, ifupdown=false, ifdownfromup = true)
    χ = 10
    function logZ(β)
        model = Ising(Ni, Nj, β)
        M = atype.(model_tensor(model, Val(:bulk)))
        rt = VUMPSRuntime(M, χ, alg)
        rt = leading_boundary(rt, M, alg)
        # env = VUMPSEnv(rt, M)
        return log(real(observable(rt, model, Val(:Z))))
    end
    # @show Zygote.gradient(β->-logZ(β), 0.5)
    @test Zygote.gradient(β->-logZ(β), 0.5)[1] ≈ -1.745564581767667
end