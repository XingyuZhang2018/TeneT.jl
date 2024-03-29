using TeneT
using TeneT:qrpos,lqpos,leftorth,leftenv,rightorth,rightenv,ACenv,Cenv,LRtoC,ALCtoAC,ACCtoALAR,env_norm
using ChainRulesCore
using CUDA
using LinearAlgebra
using OMEinsum
using Random
using Test
using Zygote
CUDA.allowscalar(false)

@testset "matrix autodiff with $atype{$dtype}" for atype in [Array], dtype in [ComplexF64]
    Random.seed!(100)
    a = atype(randn(dtype, 10, 10))
    @test Zygote.gradient(norm, a)[1] ≈ num_grad(norm, a)

    function foo1(x) 
        norm(atype(dtype[x 2x; 3x x]))
    end
    @test Zygote.gradient(foo1, 1)[1] ≈ num_grad(foo1, 1)
end

@testset "zygote mutable arrays with $atype{$dtype}" for atype in [Array], dtype in [ComplexF64]
    Random.seed!(100)
    function foo(F) 
        buf = Zygote.Buffer(F) # https://fluxml.ai/Zygote.jl/latest/utils/#Zygote.Buffer
        @inbounds @views for j in 1:2, i in 1:2 
            buf[:,:,:,i,j] = F[:,:,:,i,j]./norm(F[:,:,:,i,j]) 
        end
        return norm(copy(buf))
    end
    F = atype(rand(dtype, 3,2,3,2,2))
    @test Zygote.gradient(foo, F)[1] ≈ num_grad(foo, F) atol = 1e-8
end

@testset "loop_einsum mistake with $atype{$dtype}" for atype in [Array], dtype in [Float64, ComplexF64]
    Random.seed!(100)
    D = 5
    A = atype(rand(dtype, D,D,D))
    B = atype(rand(dtype, D,D))
    function foo(x)
        C = A * x
        D = B * x
        E = Array(ein"abc,abc -> "(C,C))[]
        F = Array(ein"ab,ab -> "(D,D))[]
        return norm(E/F)
        # E = ein"abc,abc -> "(C,C)[]
        # F = ein"ab,ab -> "(D,D)[]
        # return norm(E/F) mistake for GPU
    end 
    @test Zygote.gradient(foo, 1)[1] ≈ num_grad(foo, 1) atol = 1e-8
end

@testset "QR factorization with $atype{$dtype}" for atype in [Array], dtype in [Float64, ComplexF64]
    Random.seed!(100)
    M = atype(rand(dtype, 3, 3))
    function foo(M)
        Q, R = qrpos(M)
        return norm(Q) + norm(R)
    end
    @test Zygote.gradient(foo, M)[1] ≈ num_grad(foo, M) atol = 1e-8
end

@testset "LQ factorization with $atype{$dtype}" for atype in [Array], dtype in [Float64, ComplexF64]
    Random.seed!(100)
    M = atype(rand(dtype, 3, 3))
    function foo(M)
        L, Q = lqpos(M)
        return  norm(Q) + norm(L)
    end
    @test Zygote.gradient(foo, M)[1] ≈ num_grad(foo, M) atol = 1e-8
end

@testset "$(Ni)x$(Nj) leftenv and rightenv with $atype{$dtype}" for atype in [Array], dtype in [ComplexF64], ifobs in [false, true], Ni = [3], Nj = [3]
    Random.seed!(100)
    D, d = 3, 2
    A = atype(rand(dtype,      D, d, D,          Ni, Nj))
    S = atype(rand(ComplexF64, D, d, D, D, d, D, Ni, Nj))
    M = atype(rand(ComplexF64, d, d, d, d,       Ni, Nj))

       ALu, =  leftorth(A) 
       ALd, =  leftorth(A)
    _, ARu, = rightorth(A)
    _, ARd, = rightorth(A)

    function foo1(M)
        _,FL = leftenv(ALu, conj(ALd), M; ifobs = ifobs)
        s = 0.0
        for j in 1:Nj, i in 1:Ni
            A  = Array(ein"(abc,abcdef),def -> "(FL[:,:,:,i,j], S[:,:,:,:,:,:,i,j], FL[:,:,:,i,j]))[]
            B  = Array(ein"abc,abc -> "(FL[:,:,:,i,j], FL[:,:,:,i,j]))[]
            s += norm(A/B)
        end
        return s
    end 
    @test Zygote.gradient(foo1, M)[1] ≈ num_grad(foo1, M) atol = 1e-7

    function foo2(M)
        _,FR = rightenv(ARu, conj(ARd), M; ifobs = ifobs)
        s = 0
        for j in 1:Nj, i in 1:Ni
            A  = Array(ein"(abc,abcdef),def -> "(FR[:,:,:,i,j], S[:,:,:,:,:,:,i,j], FR[:,:,:,i,j]))[]
            B  = Array(ein"abc,abc -> "(FR[:,:,:,i,j], FR[:,:,:,i,j]))[]
            s += norm(A/B)
        end
        return s
    end 
    @test Zygote.gradient(foo2, M)[1] ≈ num_grad(foo2, M) atol = 1e-7
end

@testset "$(Ni)x$(Nj) ACenv and Cenv with $atype{$dtype}" for atype in [Array], dtype in [ComplexF64], Ni = [2], Nj = [2]
    Random.seed!(100)
    D, d = 3, 2
     A = atype(rand(dtype,      D, d, D,          Ni, Nj))
    S1 = atype(rand(ComplexF64, D, d, D, D, d, D, Ni, Nj))
    S2 = atype(rand(ComplexF64, D, D, D, D,       Ni, Nj))
     M = atype(rand(ComplexF64, d, d, d, d,       Ni, Nj))

    AL, L, _ =  leftorth(A) 
    R, AR, _ = rightorth(A)
    _, FL    =  leftenv(AL, conj(AL), M)
    _, FR    = rightenv(AR, conj(AR), M)

     C =   LRtoC( L, R)
    AC = ALCtoAC(AL, C)
    function foo1(M)
        _, AC = ACenv(AC, FL, M, FR)
        s = 0
        for j in 1:Nj, i in 1:Ni
            A = Array(ein"(abc,abcdef),def -> "(AC[:,:,:,i,j], S1[:,:,:,:,:,:,i,j], AC[:,:,:,i,j]))[]
            B = Array(ein"abc,abc -> "(AC[:,:,:,i,j], AC[:,:,:,i,j]))[]
            s += norm(A/B)
        end
        return s
    end
    @test Zygote.gradient(foo1, M)[1] ≈ num_grad(foo1, M) atol = 1e-7

    function foo2(M)
        _, FL = leftenv(AL, conj(AL), M)
        _, FR = rightenv(AR, conj(AR), M)
        _, C = Cenv(C, FL, FR)
        s = 0
        for j in 1:Nj, i in 1:Ni
            A = Array(ein"(ab,abcd),cd -> "(C[:,:,i,j], S2[:,:,:,:,i,j], C[:,:,i,j]))[]
            B = Array(ein"ab,ab -> "(C[:,:,i,j], C[:,:,i,j]))[]
            s += norm(A/B)
        end
        return s
    end
    @test Zygote.gradient(foo2, M)[1] ≈ num_grad(foo2, M) atol = 1e-7
end

@testset "$(Ni)x$(Nj) ACCtoALAR with $atype{$dtype}" for atype in [Array], dtype in [ComplexF64], Ni = [1], Nj = [1]
    Random.seed!(100)
    D, d = 3, 2
     A = atype(rand(dtype,      D, d, D,          Ni, Nj))
    S1 = atype(rand(ComplexF64, D, d, D, D, d, D, Ni, Nj))
    S2 = atype(rand(ComplexF64, D, D, D, D,       Ni, Nj))
     M = atype(rand(ComplexF64, d, d, d, d,       Ni, Nj))

    AL, L, _ =  leftorth(A) 
    R, AR, _ = rightorth(A)
    _, FL    =  leftenv(AL, conj(AL), M)
    _, FR    = rightenv(AR, conj(AR), M)

     Co =   LRtoC( L, R)
    ACo = ALCtoAC(AL, Co)
    _, Co = Cenv(Co, FL, FR)
    function foo1(M)
        _, AC = ACenv(ACo, FL, M, FR)
        AL, AR = ACCtoALAR(AC, Co) 
        s = 0
        for j in 1:Nj, i in 1:Ni
            A  = Array(ein"(abc,abcdef),def -> "(AL[:,:,:,i,j], S1[:,:,:,:,:,:,i,j], AL[:,:,:,i,j]))[]
            B  = Array(ein"abc,abc -> "(AL[:,:,:,i,j], AL[:,:,:,i,j]))[]
            s += norm(A/B)
            A  = Array(ein"(abc,abcdef),def -> "(AR[:,:,:,i,j], S1[:,:,:,:,:,:,i,j], AR[:,:,:,i,j]))[]
            B  = Array(ein"abc,abc -> "(AR[:,:,:,i,j], AR[:,:,:,i,j]))[]
            s += norm(A/B)
            A  = Array(ein"(abc,abcdef),def -> "(AC[:,:,:,i,j], S1[:,:,:,:,:,:,i,j], AC[:,:,:,i,j]))[]
            B  = Array(ein"abc,abc -> "(AC[:,:,:,i,j], AC[:,:,:,i,j]))[]
            s += norm(A/B)
        end
        return s
    end
    @test Zygote.gradient(foo1, M)[1] ≈ num_grad(foo1, M) atol = 1e-4
end