@testset "VUMPSRuntime isotropy" for M in Ms, (d, D, χ) in zip(ds, Ds, χs)
    Random.seed!(100)
    alg = VUMPS(ifupdown=false)
    rt = CUDA.@time VUMPSRuntime(M, χ, alg)

    @test rt isa VUMPSRuntime

    env = VUMPSEnv(rt, M, alg)
    @test env isa VUMPSEnv
end

@testset "VUMPSRuntime anisotropy with $atype" for atype = [Array], ifupdown in [false, true], (d, D, χ) in zip(ds, Ds, χs)
    Random.seed!(100)

    M = atype(rand(ComplexF64, [ℂ^1*ℂ^2 ← ℂ^3*ℂ^4, 
                                ℂ^3*ℂ^5 ← ℂ^1*ℂ^6, 
                                ℂ^7*ℂ^4 ← ℂ^8*ℂ^2,
                                ℂ^8*ℂ^6 ← ℂ^7*ℂ^5], [1 2; 3 4]))
    alg = VUMPS(ifupdown=ifupdown)
    rt = VUMPSRuntime(M, χ, alg)
    @test ifupdown ? rt isa Tuple{VUMPSRuntime, VUMPSRuntime} : rt isa VUMPSRuntime

    env = VUMPSEnv(rt, M, alg)
    @test env isa VUMPSEnv

    M = atype(rand(ComplexF64, [ℂ^1*ℂ^2*(ℂ^3)'*(ℂ^4)' ← ℂ^2, 
                                ℂ^3*ℂ^5*(ℂ^1)'*(ℂ^6)' ← ℂ^2, 
                                ℂ^7*ℂ^4*(ℂ^8)'*(ℂ^2)' ← ℂ^2,
                                ℂ^8*ℂ^6*(ℂ^7)'*(ℂ^5)' ← ℂ^2], [1 2; 3 4]))
    alg = VUMPS(ifupdown=ifupdown)
    rt = VUMPSRuntime(M, χ, alg)
    @test ifupdown ? rt isa Tuple{VUMPSRuntime, VUMPSRuntime} : rt isa VUMPSRuntime

    env = VUMPSEnv(rt, M, alg)
    @test env isa VUMPSEnv
end

@testset "vumps isotropy" for M in Ms, (d, D, χ) in zip(ds, Ds, χs), ifupdown in [false, true]
    Random.seed!(100)
    alg = VUMPS(maxiter=100, verbosity=2, ifupdown=ifupdown)
    rt = VUMPSRuntime(M, χ, alg)
    rt = leading_boundary(rt, M, alg)
    @test ifupdown ? rt isa Tuple{VUMPSRuntime, VUMPSRuntime} : rt isa VUMPSRuntime

    env = VUMPSEnv(rt, M, alg)
    @test env isa VUMPSEnv
end


@testset "vumps anisotropy with $atype" for atype = [Array], ifupdown in [false, true], (d, D, χ) in zip(ds, Ds, χs)
    Random.seed!(100)

    M = atype(rand(ComplexF64, [ℂ^1*ℂ^2 ← ℂ^3*ℂ^4, 
                                ℂ^3*ℂ^5 ← ℂ^1*ℂ^6, 
                                ℂ^7*ℂ^4 ← ℂ^8*ℂ^2,
                                ℂ^8*ℂ^6 ← ℂ^7*ℂ^5], [1 2; 3 4]))
    alg = VUMPS(ifupdown=ifupdown)
    rt = VUMPSRuntime(M, χ, alg)
    rt = leading_boundary(rt, M, alg)
    @test ifupdown ? rt isa Tuple{VUMPSRuntime, VUMPSRuntime} : rt isa VUMPSRuntime

    env = VUMPSEnv(rt, M, alg)
    @test env isa VUMPSEnv

    M = atype(rand(ComplexF64, [ℂ^1*ℂ^2*(ℂ^3)'*(ℂ^4)' ← ℂ^2, 
                                ℂ^3*ℂ^5*(ℂ^1)'*(ℂ^6)' ← ℂ^2, 
                                ℂ^7*ℂ^4*(ℂ^8)'*(ℂ^2)' ← ℂ^2,
                                ℂ^8*ℂ^6*(ℂ^7)'*(ℂ^5)' ← ℂ^2], [1 2; 3 4]))
    alg = VUMPS(ifupdown=ifupdown)
    rt = VUMPSRuntime(M, χ, alg)
    rt = leading_boundary(rt, M, alg)
    @test ifupdown ? rt isa Tuple{VUMPSRuntime, VUMPSRuntime} : rt isa VUMPSRuntime

    env = VUMPSEnv(rt, M, alg)
    @test env isa VUMPSEnv
end

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
