include("./exampletensors.jl")
include("./exampleobs.jl")

using Random
using Test
using Zygote
using CUDA, AMDGPU
using MPI

# MPI.Init()
# comm = MPI.COMM_WORLD
# rank = MPI.Comm_rank(comm)
# # select device
# comm_l = MPI.Comm_split_type(comm, MPI.COMM_TYPE_SHARED, rank)
# rank_l = MPI.Comm_rank(comm_l)

# println("Hostname: ", gethostname())
# println("in rankl $rank_l")

# @testset "pattern $pattern ising forward with $atype" for pattern in [[1;;]], atype = [ROCArray]
#     Random.seed!(100)
#     β = 0.5
#     χ = 512
#     D = 12
#     d = 2
#     model = Ising(β)
#     l = length(unique(pattern))
#     TeneT.set_device_id!(atype, rank_l+1)
#     # TeneT.set_device_id!(atype, 1)
#     # M = zeros(ComplexF64, (2,2,2,2))
#     A = rand(ComplexF64, D,D,D,D,d)
#     # M[2,1,1,1]=1.0
#     # M[1,2,1,1]=1.0
#     # M[2,2,1,1]=1.0
#     # M[1,2,2,2]=1.0
#     # M[2,1,2,2]=1.0
#     # M[1,1,2,2]=1.0
#     # data =[atype(model_tensor(model, Val(:bulk))) for _ in 1:l]
#     data = [atype(A) for _ in 1:l]
#     # data = [atype(M) for _ in 1:l]
#     M = StructArray(data, pattern)
#     verbosity = rank == 0 ? 3 : 0
#     alg = VUMPS(maxiter = 1000, 
#                 miniter = 10, 
#                 verbosity = verbosity, 
#                 show_every = 1,
#                 forloop_iter = 4,
#                 power_iter = 5,
#                 power_iter_obs = 80,
#                 ifsimple_eig=true,
#                 ifupdown=false, 
#                 ifdownfromup=false, 
#                 ifparallel=true
#     )
    
#     rt = @time VUMPSRuntime(M, χ, alg)
#     rt = @time leading_boundary(rt, M, alg)
#     rt = @time leading_boundary(rt, M, alg)
#     env = VUMPSEnv(rt, M, alg)

#     # @show log(observable(env, M, Val(:Z), alg)) - 1.0257928172049902
#     # @test observable(env, M, Val(:Z), alg) ≈ 2.789305993957602
#     # @test observable(env, model, Val(:mag)   ) ≈ magofβ(model) 
#     # @show observable(env, model, pattern, Val(:energy), alg) 
#     # @test observable(env, model, pattern, Val(:energy)) ≈ -1.745564581767667
# end

@testset "ising backward with $atype $pattern" for atype = [Array], pattern in [[1;;]]
    # [1;;], [1 1; 1 1], [1 2; 2 1], [1 2; 3 4], [1 1; 2 2]
    # [1 3 2 2 3 1; 2 3 1 1 3 2]
    Random.seed!(100)
    alg = VUMPS(maxiter = 1000, 
                miniter = 10, 
                verbosity = 3, 
                show_every = 1,
                forloop_iter = 1,
                power_iter = 5,
                power_iter_obs = 80,
                ifsimple_eig=true,
                ifupdown=false, 
                ifdownfromup=false, 
                ifparallel=false,
                ifcheckpoint=false
    )

    χ = 20
    TeneT.set_device_id!(atype, 1)

    D, d = 2, 1
    A = rand(ComplexF64, D,D,D,D,d)

    function energy(β)
        model = Ising(β)
        l = length(unique(pattern))
        # data =[atype(model_tensor(model, Val(:bulk))) for _ in 1:l]
        data =[atype(A) for _ in 1:l]
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