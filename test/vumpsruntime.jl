@testset "VUMPSRuntime with $atype" for atype = [Array], M in test_Ms
    Random.seed!(100)
    χ = 10
    alg = VUMPS(ifupdown=false)
    rt = CUDA.@time VUMPSRuntime(M, χ, alg)

    @test rt isa VUMPSRuntime

    env = VUMPSEnv(rt, M, alg)
    @test env isa VUMPSEnv
end

@testset "VUMPSRuntime with $atype" for atype = [Array], ifupdown in [true, false]
    Random.seed!(100)

    M = randSA(atype, [1 2; 3 4], [(1,2,3,4), (3,5,1,6), (7,4,8,2), (8,6,7,5)])
    alg = VUMPS(ifupdown=ifupdown)
    rt = VUMPSRuntime(M, 2, alg)
    @test ifupdown ? rt isa Tuple{VUMPSRuntime, VUMPSRuntime} : rt isa VUMPSRuntime

    env = VUMPSEnv(rt, M, alg)
    @test env isa VUMPSEnv
end

@testset "oneside vumps with $atype" for atype = [Array], M in test_Ms
    Random.seed!(100)
    χ = 3
    alg = VUMPS(maxiter=100, verbosity=2, ifupdown=false)
    rt = VUMPSRuntime(M, χ, alg)
    rt = leading_boundary(rt, M, alg)
    @test rt isa VUMPSRuntime

    env = VUMPSEnv(rt, M, alg)
    @test env isa VUMPSEnv
end

@testset "oneside vumps with $atype" for atype = [Array]
    Random.seed!(100)
    χ = 3
    M = randSA(atype, [1 2; 3 4], [(1,2,3,4), (3,5,1,6), (7,4,8,2), (8,6,7,5)])
    alg = VUMPS(maxiter=100, verbosity=3, show_every=10, ifupdown=false)
    rt = VUMPSRuntime(M, χ, alg)
    rt = leading_boundary(rt, M, alg)
    @test rt isa VUMPSRuntime

    env = VUMPSEnv(rt, M, alg)
    @test env isa VUMPSEnv

    d = 2
    M = randSA(atype, [1 2; 3 4], [(1,2,3,4, d), (3,5,1,6, d), (7,4,8,2, d), (8,6,7,5, d)])
    alg = VUMPS(maxiter=100, verbosity=3, show_every=10, ifupdown=false)
    rt = VUMPSRuntime(M, χ, alg)
    rt = leading_boundary(rt, M, alg)
    @test rt isa VUMPSRuntime

    env = VUMPSEnv(rt, M, alg)
    @test env isa VUMPSEnv
end


@testset "twoside vumps with $atype" for atype = [Array], M in test_Ms
    Random.seed!(100)
    χ = 3
    alg = VUMPS(maxiter=100, verbosity=3, ifupdown=true)
    rt = VUMPSRuntime(M, χ, alg)
    rt = leading_boundary(rt, M, alg)
    @test rt isa Tuple{VUMPSRuntime, VUMPSRuntime}

    env = VUMPSEnv(rt, M, alg)
    @test env isa VUMPSEnv
end

@testset "twoside vumps with $atype" for atype = [Array]
    Random.seed!(100)
    χ = 3
    M = randSA(atype, [1 2; 3 4], [(1,2,3,4), (3,5,1,6), (7,4,8,2), (8,6,7,5)])
    alg = VUMPS(maxiter=100, verbosity=2, ifupdown=true)
    rt = VUMPSRuntime(M, χ, alg)
    rt = leading_boundary(rt, M, alg)
    @test rt isa Tuple{VUMPSRuntime, VUMPSRuntime}

    env = VUMPSEnv(rt, M, alg)
    @test env isa VUMPSEnv

    d = 2
    M = randSA(atype, [1 2; 3 4], [(1,2,3,4, d), (3,5,1,6, d), (7,4,8,2, d), (8,6,7,5, d)])
    alg = VUMPS(maxiter=100, verbosity=2, ifupdown=true)
    rt = VUMPSRuntime(M, χ, alg)
    rt = leading_boundary(rt, M, alg)
    @test rt isa Tuple{VUMPSRuntime, VUMPSRuntime}

    env = VUMPSEnv(rt, M, alg)
    @test env isa VUMPSEnv
end

# test_As = [rand(ComplexF64, χ, D, χ), rand(ComplexF64, χ, D, D, χ)];
# @testset "fix_gauge $atype $Ni x $Nj" for atype = [Array], a in [test_As[1]], Ni = [1], Nj = [1]
#     A = [atype(a) for i in 1:Ni, j in 1:Nj]
#     _, AR, _ = TeneT.right_canonical(A)
#     U = TeneT.qrpos!(rand(ComplexF64, χ, χ))[1]
#     AR′ = [ein"ab,bcd,de->ace"(U', AR, U) for AR in AR]

#     _, σ = TeneT.rightCenv(AR, conj.(AR′); ifobs=false) 
#     U′, _ = TeneT.qrpos!(σ[1])
#     λ = U[1] / U′[1]
#     U′ *= λ
#     @test U ≈ U′
#     @test AR ≈ [ein"ab,bcd,de->ace"(U, AR′, U') for AR′ in AR′]
# end

# include("../example/exampletensors.jl")
# include("../example/exampleobs.jl")

# @testset "fix_gauge_vumps_step with $atype $Ni x $Nj" for atype = [Array], Ni = [1], Nj = [1]
#     β = asinh(1) / 2
#     model = Ising(Ni, Nj, β)
#     M = atype.(model_tensor(model, Val(:bulk)))

#     alg = VUMPS(maxiter=200, miniter=100, verbosity=2, tol=1e-12, ifupdown=false)
#     χ = 2
#     rt = VUMPSRuntime(M, χ, alg)
#     rt = leading_boundary(rt, M, alg)
#     @test rt isa VUMPSRuntime

#     rt′, err = fix_gauge_vumps_step(rt, M, alg)
#     @test norm(rt.AR[1] - rt′.AR[1]) < 1e-9
#     @test norm(rt.AL[1] - rt′.AL[1]) < 1e-9
#     @test norm(rt.C[1] - rt′.C[1]) < 1e-9
#     @test norm(rt.FL[1] - rt′.FL[1]) < 1e-9
#     @test norm(rt.FR[1] - rt′.FR[1]) < 1e-9
# end