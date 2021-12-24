using VUMPS
using VUMPS:qrpos,lqpos,leftorth,leftenv,rightorth,rightenv,ACenv,Cenv,LRtoC,ALCtoAC,ACCtoALAR,obs_FL,obs_FR,parity_conserving, Z2tensor2tensor,AbstractZ2Array
using ChainRulesCore
using CUDA
using LinearAlgebra
using OMEinsum
using Random
using Test
using Zygote
CUDA.allowscalar(false)

@testset "matrix autodiff with $(symmetry) $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64], symmetry in [:none, :Z2]
    Random.seed!(100)
    A = randinitial(Val(symmetry), atype, dtype, 4, 4)
    @test Zygote.gradient(norm, A)[1] ≈ num_grad(norm, A)

    function foo1(x) 
        norm(atype(dtype[x 2x; 3x x]))
    end
    @test Zygote.gradient(foo1, 1)[1] ≈ num_grad(foo1, 1)

    # example to solve differential of array of array
    # use `[]` list then reshape
    A = [randinitial(Val(symmetry), atype, dtype, 2, 2) for i in 1:2, j in 1:2]
    function foo2(x)
        # B[i,j] = A[i,j].*x   # mistake
        B = reshape([A[i]*x for i=1:4],2,2)
        return norm(sum(B))
    end
    @test Zygote.gradient(foo2, 1)[1] ≈ num_grad(foo2, 1)
end

@testset "QR factorization with $(symmetry) $atype{$dtype}" for atype in [Array, CuArray], dtype in [ComplexF64], symmetry in [:none, :Z2]
    Random.seed!(100)
    M = randinitial(Val(symmetry), atype, dtype, 3, 2, 3)
    function foo(M)
        M = reshape(M, 6, 3)
        Q, R = qrpos(M)
        return norm(Q) + norm(R)
    end

    @test Zygote.gradient(foo, M)[1] ≈ num_grad(foo, M)  atol = 1e-8
end

@testset "LQ factorization with $(symmetry) $atype{$dtype}" for atype in [Array, CuArray], dtype in [ComplexF64], symmetry in [:none, :Z2]
    Random.seed!(100)
    M = randinitial(Val(symmetry), atype, dtype, 3, 2, 3)
    function foo(M)
        M = reshape(M, 3, 6)
        L, Q = lqpos(M)
        return  norm(Q) + norm(L)
    end
    @test Zygote.gradient(foo, M)[1] ≈ num_grad(foo, M) atol = 1e-8
end

@testset "loop_einsum mistake with  $(symmetry) $atype{$dtype}" for atype in [Array], dtype in [Float64], symmetry in [:Z2]
    Random.seed!(100)
    D = 2
    A = randinitial(Val(symmetry), atype, dtype, D, D, D)
    B = randinitial(Val(symmetry), atype, dtype, D, D)

    function foo(x)
        C = A * x
        D = B * x
        E = ein"abc,abc -> "(C,C)[]
        F = ein"ab,ab -> "(D,D)[]
        return E/F
        # E = ein"abc,abc -> "(C,C)[]
        # F = ein"ab,ab -> "(D,D)[]
        # return norm(E/F) mistake for GPU
    end 
    @test Zygote.gradient(foo, 1)[1] ≈ num_grad(foo, 1) atol = 1e-8
end

@testset "$(symmetry) OMEinsum ad $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64], symmetry in [:Z2]
    Random.seed!(100)
    D,d = 3,2
	FL = randinitial(Val(symmetry), atype, dtype, D, d, D)
	S = randinitial(Val(symmetry), atype, dtype, D, d, D, D, d, D)
    # FLtensor = Z2tensor2tensor(FL)
	# Stensor = Z2tensor2tensor(S)
    foo1(x) = norm(Array(ein"(abc,abcdef),def ->"(FL*x, S*x, FL*x))[])
    foo2(x) = norm(Array(ein"(abc,abcdef),def ->"(FLtensor*x, Stensor*x, FLtensor*x))[])
    @show Zygote.gradient(foo1, 1)[1]
	# @test Zygote.gradient(foo1, 1)[1] ≈ num_grad(foo1, 1) ≈ Zygote.gradient(foo2, 1)[1] ≈ num_grad(foo2, 1)
end

@testset "$(Ni)x$(Nj) leftenv and rightenv with $(symmetry) $atype{$dtype}" for atype in [Array, CuArray], dtype in [ComplexF64], symmetry in [:none, :Z2], Ni = [2], Nj = [2]
    Random.seed!(100)
    D, d = 3, 2
    A = [randinitial(Val(symmetry), atype, dtype, D, d, D) for i in 1:Ni, j in 1:Nj]
    M = [randinitial(Val(symmetry), atype, ComplexF64, d, d, d, d) for i in 1:Ni, j in 1:Nj]
    S = [randinitial(Val(symmetry), atype, ComplexF64, D, d, D, D, d, D) for i in 1:Ni, j in 1:Nj]

    ALu, = leftorth(A) 
    ALd, = leftorth(A)
    _, ARu, = rightorth(A)
    _, ARd, = rightorth(A)

    function foo1(x)
        _, FL = leftenv(ALu, ALd, M*x)
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

@testset "$(Ni)x$(Nj) ACenv and Cenv with $(symmetry) $atype{$dtype}" for atype in [Array, CuArray], dtype in [ComplexF64], symmetry in [:Z2], Ni = [2], Nj = [2]
    Random.seed!(100)
    D, d = 3, 2
    A = [randinitial(Val(symmetry), atype, dtype, D, d, D) for i in 1:Ni, j in 1:Nj]
    M = [randinitial(Val(symmetry), atype, ComplexF64, d, d, d, d) for i in 1:Ni, j in 1:Nj]
    S1 = [randinitial(Val(symmetry), atype, ComplexF64, D, d, D, D, d, D) for i in 1:Ni, j in 1:Nj]
    S2 = [randinitial(Val(symmetry), atype, ComplexF64, D, D, D, D) for i in 1:Ni, j in 1:Nj]

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

@testset "$(Ni)x$(Nj) ACCtoALAR with $(symmetry) $atype{$dtype}" for atype in [Array], dtype in [ComplexF64], symmetry in [:Z2], Ni = [2], Nj = [2]
    Random.seed!(100)
    D, d = 3, 2
    A = [randinitial(Val(symmetry), atype, dtype, D, d, D) for i in 1:Ni, j in 1:Nj]
    S1 = [randinitial(Val(symmetry), atype, ComplexF64, D, d, D, D, d, D) for i in 1:Ni, j in 1:Nj]
    S2 = [randinitial(Val(symmetry), atype, ComplexF64, D, D, D, D) for i in 1:Ni, j in 1:Nj]
    M = [randinitial(Val(symmetry), atype, ComplexF64, d, d, d, d) for i in 1:Ni, j in 1:Nj]

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
    @show Zygote.gradient(foo1, 1)
    # @test Zygote.gradient(foo1, 1)[1] ≈ num_grad(foo1, 1) atol = 1e-2
end

@testset "observable leftenv and rightenv with $(symmetry) $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64, ComplexF64], symmetry in [:none, :Z2], Ni = [2], Nj = [2]
    Random.seed!(100)
    D, d = 3, 2
    A = [randinitial(Val(symmetry), atype, dtype, D, d, D) for i in 1:Ni, j in 1:Nj]
    S = Array{atype{ComplexF64,6},2}(undef, Ni, Nj)
    M = [randinitial(Val(symmetry), atype, ComplexF64, d, d, d, d) for i in 1:Ni, j in 1:Nj]
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

@testset "parity_conserving" for atype in [Array,CuArray], dtype in [ComplexF64], Ni = [2], Nj = [2]
    Random.seed!(100)
    D = 2
    T = atype(rand(dtype,D,D,4,D,D,Ni*Nj))
    function foo(T)
        ipeps = reshape([parity_conserving(T[:,:,:,:,:,i]) for i = 1:4], (2, 2))
        norm(ipeps)
    end
    @test Zygote.gradient(foo, T)[1] ≈ num_grad(foo, T) atol = 1e-8
end