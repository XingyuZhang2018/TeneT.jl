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

@testset "2D classical ising" begin
    β = 0.5
    ham = Zygote.@ignore ComplexF64[-1. 1;1 -1]
    w = exp.(- β * ham)
    wsq = sqrt(w)
    m = zeros(ComplexF64, 2,2,2,2)
    for i in 1:2, j in 1:2, k in 1:2, l in 1:2, s in 1:2
        m[i,j,l,k] += wsq[i,s] * wsq[j,s] * wsq[l,s] * wsq[k,s]
    end
    M = StructArray([TensorMap(m, ℂ^2*ℂ^2 ← ℂ^2*ℂ^2)], [1;;])
    alg = VUMPS(maxiter=100, verbosity=2, ifupdown=false)
    χ = ℂ^10
    rt = VUMPSRuntime(M, χ, alg)
    rt = leading_boundary(rt, M, alg)
    @test rt isa VUMPSRuntime

    env = VUMPSEnv(rt, M, alg)
    @test env isa VUMPSEnv

    FL = env.FLu[1]
    FR = env.FRu[1]
    ACu = env.ACu[1]
    Cu = rt.C[1]
    @tensoropt n = FL[4 3; 1] * Cu[1; 2] * FR[2 3; 5] * conj(Cu[4; 5])
    @tensoropt N = FLmap(FL, ACu, adjoint(ACu), M[1])[1 2; 3] * FR[3 2; 1] 
    @test N/n ≈ 2.789305993957602
end 
