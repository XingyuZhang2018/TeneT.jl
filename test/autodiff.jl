using VUMPS
using VUMPS:qrpos,lqpos,leftorth,leftenv,rightorth,rightenv,ACenv,Cenv,LRtoC,ALCtoAC,ACCtoALAR
using ChainRulesCore
using CUDA
using LinearAlgebra
using OMEinsum
using Random
using Test
using Zygote
CUDA.allowscalar(false)

@testset "matrix autodiff with $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64, ComplexF64]
    Random.seed!(100)
    a = atype(randn(dtype, 10, 10))
    @test Zygote.gradient(norm, a)[1] ≈ num_grad(norm, a)

    function foo1(x) 
        norm(atype(dtype[x 2x; 3x x]))
    end
    @test Zygote.gradient(foo1, 1)[1] ≈ num_grad(foo1, 1)
end

@testset "QR factorization with $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64, ComplexF64]
    Random.seed!(100)
    M = atype(rand(dtype, 3, 3))
    function foo(M)
        Q, R = qrpos(M)
        return norm(Q) + norm(R)
    end
    @test Zygote.gradient(foo, M)[1] ≈ num_grad(foo, M) atol = 1e-8
end

@testset "LQ factorization with $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64, ComplexF64]
    Random.seed!(100)
    M = atype(rand(dtype, 3, 3))
    function foo(M)
        L, Q = lqpos(M)
        return  norm(Q) + norm(L)
    end
    @test Zygote.gradient(foo, M)[1] ≈ num_grad(foo, M) atol = 1e-8
end

@testset "loop_einsum mistake with  $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64, ComplexF64]
    Random.seed!(100)
    D = 5
    A = atype(rand(dtype, D,D,D))
    B = atype(rand(dtype, D,D))
    function foo(x)
        C = A * x
        D = B * x
        E = ein"abc,abc -> "(C,C)
        F = ein"ab,ab -> "(D,D)
        return norm(Array(E)[]/Array(F)[])
        # E = ein"abc,abc -> "(C,C)[]
        # F = ein"ab,ab -> "(D,D)[]
        # return norm(E/F) mistake for GPU
    end 
    @test Zygote.gradient(foo, 1)[1] ≈ num_grad(foo, 1) atol = 1e-8
end

@testset "$(Ni)x$(Nj) leftenv and rightenv with $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64, ComplexF64], Ni = [2], Nj = [2]
    Random.seed!(100)
    D, d = 3, 2
    A = Array{atype{dtype,3},2}(undef, Ni, Nj)
    S = Array{atype{ComplexF64,6},2}(undef, Ni, Nj)
    M = Array{atype{ComplexF64,4},2}(undef, Ni, Nj)
    for j in 1:Nj, i in 1:Ni
        A[i,j] = atype(rand(dtype, D, d, D))
        S[i,j] = atype(rand(ComplexF64, D, d, D, D, d, D))
        M[i,j] = atype(rand(ComplexF64, d, d, d, d))
    end

    ALu, = leftorth(A) 
    ALd, = leftorth(A)
    _, ARu, = rightorth(A)
    _, ARd, = rightorth(A)

    function foo1(x)
        _,FL = leftenv(ALu, ALd, M*x)
        s = 0
        for j in 1:Nj, i in 1:Ni
            A = ein"(abc,abcdef),def -> "(FL[i,j], S[i,j], FL[i,j])
            B = ein"abc,abc -> "(FL[i,j], FL[i,j])
            s += norm(Array(A)[]/Array(B)[])
        end
        return s
    end 
    @test Zygote.gradient(foo1, 1)[1] ≈ num_grad(foo1, 1) atol = 1e-7

    function foo2(x)
        _,FR = rightenv(ARu, ARd, M*x)
        s = 0
        for j in 1:Nj, i in 1:Ni
            A = ein"(abc,abcdef),def -> "(FR[i,j], S[i,j], FR[i,j])
            B = ein"abc,abc -> "(FR[i,j], FR[i,j])
            s += norm(Array(A)[]/Array(B)[])
        end
        return s
    end 
    @test Zygote.gradient(foo2, 1)[1] ≈ num_grad(foo2, 1) atol = 1e-7
end

@testset "$(Ni)x$(Nj) ACenv and Cenv with $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64, ComplexF64], Ni = [2], Nj = [2]
    Random.seed!(100)
    D, d = 3, 2
    A = Array{atype{dtype,3},2}(undef, Ni, Nj)
    S1 = Array{atype{ComplexF64,6},2}(undef, Ni, Nj)
    S2 = Array{atype{ComplexF64,4},2}(undef, Ni, Nj)
    M = Array{atype{ComplexF64,4},2}(undef, Ni, Nj)
    for j in 1:Nj, i in 1:Ni
        A[i,j] = atype(rand(dtype, D, d, D))
        S1[i,j] = atype(rand(ComplexF64, D, d, D, D, d, D))
        S2[i,j] = atype(rand(ComplexF64, D, D, D, D))
        M[i,j] = atype(rand(ComplexF64, d, d, d, d))
    end

    AL, L, _ = leftorth(A) 
    R, AR, _ = rightorth(A)
    _, FL = leftenv(AL, AL, M)
    _, FR = rightenv(AR, AR, M)

    C = LRtoC(L, R)
    AC = ALCtoAC(AL, C)
    function foo1(x)
        _, AC = ACenv(AC, FL, M*x, FR)
        s = 0
        for j in 1:Nj, i in 1:Ni
            A = ein"(abc,abcdef),def -> "(AC[i,j], S1[i,j], AC[i,j])
            B = ein"abc,abc -> "(AC[i,j], AC[i,j])
            s += norm(Array(A)[]/Array(B)[])
        end
        return s
    end
    @test Zygote.gradient(foo1, 1)[1] ≈ num_grad(foo1, 1) atol = 1e-8

    function foo2(x)
        _, FL = leftenv(AL, AL, M*x)
        _, FR = rightenv(AR, AR, M*x)
        _, C = Cenv(C, FL, FR)
        s = 0
        for j in 1:Nj, i in 1:Ni
            A = ein"(ab,abcd),cd -> "(C[i,j], S2[i,j], C[i,j])
            B = ein"ab,ab -> "(C[i,j], C[i,j])
            s += norm(Array(A)[]/Array(B)[])
        end
        return s
    end
    @test Zygote.gradient(foo2, 1)[1] ≈ num_grad(foo2, 1) atol = 1e-8
end

@testset "$(Ni)x$(Nj) ACCtoALAR with $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64, ComplexF64], Ni = [2], Nj = [2]
    Random.seed!(100)
    D, d = 3, 2
    A = Array{atype{dtype,3},2}(undef, Ni, Nj)
    S1 = Array{atype{ComplexF64,6},2}(undef, Ni, Nj)
    S2 = Array{atype{ComplexF64,4},2}(undef, Ni, Nj)
    M = Array{atype{ComplexF64,4},2}(undef, Ni, Nj)
    for j in 1:Nj, i in 1:Ni
        A[i,j] = atype(rand(dtype, D, d, D))
        S1[i,j] = atype(rand(ComplexF64, D, d, D, D, d, D))
        S2[i,j] = atype(rand(ComplexF64, D, D, D, D))
        M[i,j] = atype(rand(ComplexF64, d, d, d, d))
    end

    AL, L, _ = leftorth(A) 
    R, AR, _ = rightorth(A)
    _, FL = leftenv(AL, AL, M)
    _, FR = rightenv(AR, AR, M)

    Co = LRtoC(L, R)
    ACo = ALCtoAC(AL, Co)
    _, Co = Cenv(Co, FL, FR)
    function foo1(x)
        M *= x
        _, AC = ACenv(ACo, FL, M, FR)
        AL, AR = ACCtoALAR(AC, Co) 
        s = 0
        for j in 1:Nj, i in 1:Ni
            A = ein"(abc,abcdef),def -> "(AL[i,j], S1[i,j], AL[i,j])
            B = ein"abc,abc -> "(AL[i,j], AL[i,j])
            s += norm(Array(A)[]/Array(B)[])
            A = ein"(abc,abcdef),def -> "(AR[i,j], S1[i,j], AR[i,j])
            B = ein"abc,abc -> "(AR[i,j], AR[i,j])
            s += norm(Array(A)[]/Array(B)[])
            A = ein"(abc,abcdef),def -> "(AC[i,j], S1[i,j], AC[i,j])
            B = ein"abc,abc -> "(AC[i,j], AC[i,j])
            s += norm(Array(A)[]/Array(B)[])
        end
        return s
    end
    @test Zygote.gradient(foo1, 1)[1] ≈ num_grad(foo1, 1) atol = 1e-2
end

@testset "observable leftenv and rightenv with $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64, ComplexF64], Ni = [2], Nj = [2]
    Random.seed!(100)
    D, d = 3, 2
    A = Array{atype{dtype,3},2}(undef, Ni, Nj)
    S = Array{atype{ComplexF64,6},2}(undef, Ni, Nj)
    M = Array{atype{ComplexF64,4},2}(undef, Ni, Nj)
    for j in 1:Nj, i in 1:Ni
        A[i,j] = atype(rand(dtype, D, d, D))
        S[i,j] = atype(rand(ComplexF64, D, d, D, D, d, D))
        M[i,j] = atype(rand(ComplexF64, d, d, d, d))
    end

    ALu, = leftorth(A) 
    ALd, = leftorth(A)
    _, ARu, = rightorth(A)
    _, ARd, = rightorth(A)

    function foo1(x)
        _,FL = obs_FL(ALu, ALd, M*x)
        s = 0
        for j in 1:Nj, i in 1:Ni
            A = ein"(abc,abcdef),def -> "(FL[i,j], S[i,j], FL[i,j])
            B = ein"abc,abc -> "(FL[i,j], FL[i,j])
            s += norm(Array(A)[]/Array(B)[])
        end
        return s
    end 
    @test Zygote.gradient(foo1, 1)[1] ≈ num_grad(foo1, 1) atol = 1e-7

    function foo2(x)
        _,FR = obs_FR(ARu, ARd, M*x)
        s = 0
        for j in 1:Nj, i in 1:Ni
            A = ein"(abc,abcdef),def -> "(FR[i,j], S[i,j], FR[i,j])
            B = ein"abc,abc -> "(FR[i,j], FR[i,j])
            s += norm(Array(A)[]/Array(B)[])
        end
        return s
    end 
    @test Zygote.gradient(foo2, 1)[1] ≈ num_grad(foo2, 1) atol = 1e-7
end

@testset "bigleftenv and bigrightenv with $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64, ComplexF64], Ni = [2], Nj = [2]
    Random.seed!(100)
    D, d = 3, 2
    A = Array{atype{dtype,3},2}(undef, Ni, Nj)
    S = Array{atype{ComplexF64,8},2}(undef, Ni, Nj)
    M = Array{atype{ComplexF64,4},2}(undef, Ni, Nj)
    for j in 1:Nj, i in 1:Ni
        A[i,j] = atype(rand(dtype, D, d, D))
        S[i,j] = atype(rand(ComplexF64, D, d, d, D, D, d, d, D))
        M[i,j] = atype(rand(ComplexF64, d, d, d, d))
    end

    ALu, = leftorth(A) 
    ALd, = leftorth(A) 
    _, ARu, = rightorth(A)
    _, ARd, = rightorth(A)
    function foo1(x)
        _, FL4 = bigleftenv(ALu, ALd, M*x)
        s = 0
        for j in 1:Nj, i in 1:Ni
            A = ein"(abcd,abcdefgh),efgh -> "(FL4[i,j], S[i,j], FL4[i,j])
            B = ein"abcd,abcd -> "(FL4[i,j], FL4[i,j])
            s += norm(Array(A)[]/Array(B)[])
        end
        return s
    end 
    @test Zygote.gradient(foo1, 1)[1] ≈ num_grad(foo1, 1) atol = 1e-7

    function foo2(x)
        _, FR4 = bigrightenv(ARu, ARd, M*x)
        s = 0
        for j in 1:Nj, i in 1:Ni
            A = ein"(abcd,abcdefgh),efgh -> "(FR4[i,j], S[i,j], FR4[i,j])
            B = ein"abcd,abcd -> "(FR4[i,j], FR4[i,j])
            s += norm(Array(A)[]/Array(B)[])
        end
        return s
    end 
    @test Zygote.gradient(foo2, 1)[1] ≈ num_grad(foo2, 1) atol = 1e-7
end