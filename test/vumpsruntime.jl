using CUDA
using Random
using Test
using TeneT

@testset "VUMPSRuntime with $atype{$dtype} $Ni x $Nj" for atype = [CuArray], dtype = [ComplexF64], Ni = [1,2,3], Nj = [1,2,3]
    Random.seed!(100)
    D, χ = 2, 10
    M = [atype(rand(dtype,D,D,D,D)) for i in 1:Ni, j in 1:Nj]
    rt = CUDA.@time VUMPSRuntime(M, χ)
    @test rt isa VUMPSRuntime

    env = VUMPSEnv(rt, M)
    @test env isa VUMPSEnv
end

@testset "VUMPSRuntime with $atype{$dtype} $Ni x $Nj" for atype = [Array], dtype = [ComplexF64], Ni = [2], Nj = [2]
    Random.seed!(100)
    M = Matrix{atype}(undef, Ni, Nj)
    M[1,1] = atype(rand(dtype,1,2,3,4))
    M[1,2] = atype(rand(dtype,3,5,1,6))
    M[2,1] = atype(rand(dtype,7,4,8,2))
    M[2,2] = atype(rand(dtype,8,6,7,5))
    rt = VUMPSRuntime(M, 2)
    @test rt isa VUMPSRuntime

    env = VUMPSEnv(rt, M)
    @test env isa VUMPSEnv
end

@testset "oneside vumps with $atype{$dtype} $Ni x $Nj" for atype = [Array, CuArray], dtype = [ComplexF64], Ni = [1,2,3], Nj = [1,2,3]
    Random.seed!(100)
    D, χ = 2, 3
    M = [atype(rand(dtype,D,D,D,D)) for i in 1:Ni, j in 1:Nj]
    alg = VUMPS(maxiter=100, verbosity=2, ifupdown=false)
    rt = VUMPSRuntime(M, χ, alg)
    rt = leading_boundary(rt, M, alg)
    @test rt isa VUMPSRuntime

    env = VUMPSEnv(rt, M)
    @test env isa VUMPSEnv
end

@testset "oneside vumps with $atype{$dtype} $Ni x $Nj" for atype = [Array], dtype = [ComplexF64], Ni = [2], Nj = [2]
    Random.seed!(100)
    χ = 10
    M = Matrix{atype}(undef, Ni, Nj)
    M[1,1] = atype(rand(dtype,1,2,3,4))
    M[1,2] = atype(rand(dtype,3,5,1,6))
    M[2,1] = atype(rand(dtype,7,4,8,2))
    M[2,2] = atype(rand(dtype,8,6,7,5))
    alg = VUMPS(maxiter=100, verbosity=3, show_every=1, ifupdown=false)
    rt = VUMPSRuntime(M, χ, alg)
    rt = leading_boundary(rt, M, alg)
    @test rt isa VUMPSRuntime

    env = VUMPSEnv(rt, M)
    @test env isa VUMPSEnv
end

@testset "twoside vumps with $atype{$dtype} $Ni x $Nj" for atype = [Array], dtype = [ComplexF64], Ni = [1,2,3], Nj = [1,2,3]
    Random.seed!(100)
    D, χ = 2, 3
    M = [atype(rand(dtype,D,D,D,D)) for i in 1:Ni, j in 1:Nj]
    alg = VUMPS(maxiter=100, verbosity=3, ifupdown=true)
    rt = VUMPSRuntime(M, χ, alg)
    rt = leading_boundary(rt, M, alg)
    @test rt isa Tuple{VUMPSRuntime, VUMPSRuntime}

    env = VUMPSEnv(rt, M)
    @test env isa VUMPSEnv
end

@testset "oneside vumps with $atype{$dtype} $Ni x $Nj" for atype = [Array], dtype = [ComplexF64], Ni = [2], Nj = [2]
    Random.seed!(100)
    D, χ = 2, 3
    M = Matrix{atype}(undef, Ni, Nj)
    M[1,1] = atype(rand(dtype,1,2,3,4))
    M[1,2] = atype(rand(dtype,3,5,1,6))
    M[2,1] = atype(rand(dtype,7,4,8,2))
    M[2,2] = atype(rand(dtype,8,6,7,5))
    alg = VUMPS(maxiter=100, verbosity=2, ifupdown=true)
    rt = VUMPSRuntime(M, χ, alg)
    rt = leading_boundary(rt, M, alg)
    @test rt isa Tuple{VUMPSRuntime, VUMPSRuntime}

    env = VUMPSEnv(rt, M)
    @test env isa VUMPSEnv
end