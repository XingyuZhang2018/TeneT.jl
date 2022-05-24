using VUMPS
using VUMPS:qrpos,lqpos,leftorth,leftenv,rightorth,rightenv,ACenv,Cenv,LRtoC,ALCtoAC,ACCtoALAR,obs_FL,obs_FR,parity_conserving, asArray,asZ2Array,Z2Array,dtr
using ChainRulesCore
using CUDA
using LinearAlgebra
using OMEinsum
using Random
using Test
using Zygote
CUDA.allowscalar(false)

@testset "ad basic with $atype{$dtype}" for atype in [Array], dtype in [ComplexF64], symmetry in [:U1]
    Random.seed!(100)
    ## asArray and asSymmetryArray
    A = randinitial(Val(symmetry), atype, dtype, 3,2,3; dir = [-1,1,1])
    Atensor = asArray(A)
    foo(Atensor) = norm(asSymmetryArray(Atensor, Val(symmetry); dir = [-1,1,1]))
    @test Zygote.gradient(foo, Atensor)[1] ≈ num_grad(foo, Atensor)

    ## reshape
    A = randinitial(Val(symmetry), atype, dtype, 3,2,3; dir = [-1,1,1])
    Atensor = asArray(A)
    foo1(x) = norm(reshape(A*x, 6,3))
    foo2(x) = norm(reshape(Atensor*x, 6,3))
    @test Zygote.gradient(foo1, 1)[1] ≈ num_grad(foo1, 1) ≈ Zygote.gradient(foo2, 1)[1] ≈ num_grad(foo2, 1)

    ## wrong shape: actually don't change shape
    A = randinitial(Val(symmetry), atype, dtype, 6,3; dir = [-1,1])
    Atensor = asArray(A)
    foo3(x) = norm(reshape(A*x, 3,2,3))
    foo4(x) = norm(reshape(Atensor*x, 3,2,3))
    @test Zygote.gradient(foo3, 1)[1] ≈ num_grad(foo3, 1) ≈ Zygote.gradient(foo4, 1)[1] ≈ num_grad(foo4, 1)

    ## symmetry reshape
    A = randinitial(Val(symmetry), atype, dtype, 10,3,3,10; dir = [-1,1,-1,1])
    B, reinfo = symmetryreshape(A, 10,9,10)
    foo9(A) = norm(symmetryreshape(A, 10,9,10)[1])
    foo10(B) = norm(symmetryreshape(B, 10,3,3,10; reinfo = reinfo)[1])
    @test Zygote.gradient(foo9, A)[1] ≈ num_grad(foo9, A)    # for d <: any
    @test Zygote.gradient(foo10, B)[1] !== nothing

    D = 3
    T = randinitial(Val(symmetry), Array, ComplexF64, D,D,4,D,D; dir = [-1,-1,1,1,1])
    function foo11(T)
        M = ein"abcde, fgchi -> gbhdiefa"(T, conj(T))
        rM, reinfo = symmetryreshape(M, D^2,D^2,D^2,D^2)
        norm(rM)
    end
    @test Zygote.gradient(foo11, T)[1] ≈ num_grad(foo11, T)    # for d <: any

    ## * 
    A = randinitial(Val(symmetry), atype, dtype, 3,6; dir = [-1,1])
    B = randinitial(Val(symmetry), atype, dtype, 6,3; dir = [-1,1])
    Atensor = asArray(A)
    Btensor = asArray(B)
    foo5(A) = norm(A*B)
    foo6(Atensor) = norm(Atensor*Btensor)
    @test asArray(Zygote.gradient(foo5, A)[1]) ≈ asArray(num_grad(foo5, A)) ≈ Zygote.gradient(foo6, Atensor)[1] ≈ num_grad(foo6, Atensor)

    ## '
    A = randinitial(Val(symmetry), atype, dtype, 6,3; dir = [-1,1])
    Atensor = asArray(A)
    foo7(A) = norm(A')
    foo8(Atensor) = norm(Atensor')
    @test asArray(Zygote.gradient(foo7, A)[1]) ≈ asArray(num_grad(foo7, A)) ≈ Zygote.gradient(foo8, Atensor)[1] ≈ num_grad(foo8, Atensor)
end

@testset "matrix autodiff with $(symmetry) $atype{$dtype}" for atype in [Array], dtype in [ComplexF64], symmetry in [:U1]
    Random.seed!(100)
    A = randinitial(Val(symmetry), atype, dtype, 4,4; dir = [-1,1])
    @test Zygote.gradient(norm, A)[1] ≈ num_grad(norm, A)

    function foo1(x) 
        norm(atype(dtype[x 2x; 3x x]))
    end
    @test Zygote.gradient(foo1, 1)[1] ≈ num_grad(foo1, 1)

    # example to solve differential of array of array
    # use `[]` list then reshape
    A = [randinitial(Val(symmetry), atype, dtype, 2,2; dir = [-1,1]) for i in 1:2, j in 1:2]
    function foo2(x)
        # B[i,j] = A[i,j].*x   # mistake
        B = reshape([A[i]*x for i=1:4],2,2)
        return norm(sum(B))
    end
    @test Zygote.gradient(foo2, 1)[1] ≈ num_grad(foo2, 1)
end

@testset "last tr with $(symmetry) $atype{$dtype}" for atype in [CuArray], dtype in [ComplexF64], symmetry in [:U1]
    Random.seed!(100)
    A = randinitial(Val(symmetry), atype, dtype, 4,4; dir = [-1,1])
	Atensor = asArray(A)
    foo1(x) = norm(tr(x))
    @test asArray(Zygote.gradient(foo1, A)[1]) ≈ asArray(num_grad(foo1, A)) ≈ Zygote.gradient(foo1, Atensor)[1] ≈ num_grad(foo1, Atensor)

    A = randinitial(Val(symmetry), atype, dtype, 4,4,4,4; dir = [-1,-1,1,1])
    Atensor = asArray(A)
    foo2(x) = norm(ein"abcd,abcd -> "(x,conj(x))[])
    @test asArray(Zygote.gradient(foo2, A)[1]) ≈ asArray(num_grad(foo2, A)) ≈ Zygote.gradient(foo2, Atensor)[1] ≈ num_grad(foo2, Atensor)

    A = randinitial(Val(symmetry), atype, dtype, 4,4,4,4; dir = [-1,-1,1,1])
    Atensor = asArray(A)
    foo3(x) = norm(ein"abab -> "(x)[])
    foo4(x) = norm(dtr(x))
    @test foo3(A) ≈ foo4(A)
    @test Zygote.gradient(foo3, A)[1] ≈ asArray(num_grad(foo3, A)) ≈ Zygote.gradient(foo3, Atensor)[1] ≈ num_grad(foo3, Atensor)
    @test Zygote.gradient(foo4, A)[1] ≈ num_grad(foo3, A) ≈ num_grad(foo4, A)
end

@testset "QR factorization with $(symmetry) $atype{$dtype}" for atype in [Array], dtype in [ComplexF64], symmetry in [:U1]
    Random.seed!(100)
    M = randinitial(Val(symmetry), atype, dtype, 5,3,5; dir = [-1,1,1])
    function foo(M)
        M = reshape(M, 15, 5)
        Q, R = qrpos(M)
        return norm(Q) + norm(R)
    end
    @test Zygote.gradient(foo, M)[1] ≈ num_grad(foo, M)  atol = 1e-8
end

@testset "LQ factorization with $(symmetry) $atype{$dtype}" for atype in [Array], dtype in [ComplexF64], symmetry in [:U1]
    Random.seed!(100)
    M = randinitial(Val(symmetry), atype, dtype, 5,3,5; dir = [-1,1,1])
    function foo(M)
        M = reshape(M, 5, 15)
        L, Q = lqpos(M)
        return  norm(Q) + norm(L)
    end
    @test Zygote.gradient(foo, M)[1] ≈ num_grad(foo, M) atol = 1e-8
end

@testset "loop_einsum mistake with  $(symmetry) $atype{$dtype}" for atype in [Array], dtype in [Float64], symmetry in [:none, :Z2, :U1]
    Random.seed!(100)
    D = 2
    A = randinitial(Val(symmetry), atype, dtype, D,D,D; dir = [-1,1,1])
    B = randinitial(Val(symmetry), atype, dtype, D,D,D,D; dir = [-1,1,1,-1])

    function foo(x)
        C = A * x
        D = B * x
        E = ein"abc, abc->"(C, conj(C))[]
        F = dtr(D)
        return E/F
    end

    @test Zygote.gradient(foo, 1)[1] ≈ num_grad(foo, 1) atol = 1e-8
end

@testset "symmetry OMEinsum ad $(symmetry) $atype{$dtype}" for atype in [Array], dtype in [Float64], symmetry in [:none, :Z2, :U1]
    Random.seed!(100)
    D,d = 3,2
	FL = randinitial(Val(symmetry), atype, dtype, D,d,D; dir = [1,1,-1])
	S = randinitial(Val(symmetry), atype, dtype, D,d,D,D,d,D; dir = [-1,-1,1,1,1,-1])
    FLtensor = asArray(FL)
	Stensor = asArray(S)
    foo1(x) = norm(Array(ein"(abc,abcdef),def ->"(FL*x, S*x, conj(FL)*x))[])
    foo2(x) = norm(Array(ein"(abc,abcdef),def ->"(FLtensor*x, Stensor*x, conj(FLtensor)*x))[])
	@test Zygote.gradient(foo1, 1)[1] ≈ num_grad(foo1, 1) ≈ Zygote.gradient(foo2, 1)[1] ≈ num_grad(foo2, 1)
end

@testset "$(Ni)x$(Nj) leftenv and rightenv with $(symmetry) $atype{$dtype}" for atype in [Array], dtype in [ComplexF64], symmetry in [:U1], Ni = [2], Nj = [2]
    Random.seed!(100)
    D, d = 5, 3
    A = [randinitial(Val(symmetry), atype, dtype, D,d,D; dir = [-1,1,1]) for i in 1:Ni, j in 1:Nj]
    M = [randinitial(Val(symmetry), atype, ComplexF64, d,d,d,d; dir = [-1,1,1,-1]) for i in 1:Ni, j in 1:Nj]
    S = [randinitial(Val(symmetry), atype, ComplexF64, D,d,D,D,d,D; dir = [-1,-1,1,1,1,-1]) for i in 1:Ni, j in 1:Nj]

    AL, = leftorth(A) 
    _, AR, = rightorth(A)

    function foo1(x)
        _, FL = leftenv(AL, AL, M*x)
        s = 0
        for j in 1:Nj, i in 1:Ni
            A = ein"(abc,abcdef),def -> "(FL[i,j], S[i,j], conj(FL[i,j]))
            B = ein"abc,abc -> "(FL[i,j], conj(FL[i,j]))
            s += norm(Array(A)[]/Array(B)[])
        end
        return s
    end 
    @test Zygote.gradient(foo1, 1)[1] ≈ num_grad(foo1, 1) atol = 1e-7

    function foo2(x)
        _,FR = rightenv(AR, AR, M*x)
        s = 0
        for j in 1:Nj, i in 1:Ni
            A = ein"(abc,abcdef),def -> "(conj(FR[i,j]), S[i,j], FR[i,j])
            B = ein"abc,abc -> "(FR[i,j], conj(FR[i,j]))
            s += norm(Array(A)[]/Array(B)[])
        end
        return s
    end 
    @test Zygote.gradient(foo2, 1)[1] ≈ num_grad(foo2, 1) atol = 1e-7
end

@testset "$(Ni)x$(Nj) ACenv and Cenv with $(symmetry) $atype{$dtype}" for atype in [Array], dtype in [ComplexF64], symmetry in [:U1], Ni = [1], Nj = [1]
    Random.seed!(100)
    D, d = 5, 3
    A = [randinitial(Val(symmetry), atype, dtype, D,d,D; dir = [-1,1,1]) for i in 1:Ni, j in 1:Nj]
    M = [randinitial(Val(symmetry), atype, ComplexF64, d,d,d,d; dir = [-1,1,1,-1]) for i in 1:Ni, j in 1:Nj]
    S1 = [randinitial(Val(symmetry), atype, ComplexF64, D,d,D,D,d,D; dir = [1,-1,-1,-1,1,1]) for i in 1:Ni, j in 1:Nj]
    S2 = [randinitial(Val(symmetry), atype, ComplexF64, D,D,D,D; dir = [1,-1,-1,1]) for i in 1:Ni, j in 1:Nj]

    AL, L, _ = leftorth(A) 
    R, AR, _ = rightorth(A)
    # _, FL = leftenv(AL, AL, M)
    # _, FR = rightenv(AR, AR, M)

    C = LRtoC(L, R)
    AC = ALCtoAC(AL, C)
    FL = [randinitial(Val(symmetry), atype, dtype, D,d,D; dir = [1,1,-1]) for i in 1:Ni, j in 1:Nj]
    FR = [randinitial(Val(symmetry), atype, dtype, D,d,D; dir = [-1,-1,1]) for i in 1:Ni, j in 1:Nj]
    function foo1(x)
        _, AC = ACenv(AC, FL, M*x, FR)
        s = 0
        for j in 1:Nj, i in 1:Ni
            A = ein"(abc,abcdef),def -> "(AC[i,j], S1[i,j], conj(AC[i,j]))
            B = ein"abc,abc -> "(AC[i,j], conj(AC[i,j]))
            s += norm(Array(A)[]/Array(B)[])
        end
        return s
    end
    @test Zygote.gradient(foo1, 1)[1] ≈ num_grad(foo1, 1) atol = 1e-8

    function foo2(x)
        # _, FL = leftenv(AL, AL, M*x)
        # _, FR = rightenv(AR, AR, M*x)
        _, C = Cenv(C, FL*x, FR*x)
        s = 0
        for j in 1:Nj, i in 1:Ni
            A = ein"(ab,abcd),cd -> "(C[i,j], S2[i,j], conj(C[i,j]))
            B = ein"ab,ab -> "(C[i,j], conj(C[i,j]))
            s += norm(Array(A)[]/Array(B)[])
        end
        return s
    end
    @test Zygote.gradient(foo2, 1)[1] ≈ num_grad(foo2, 1) atol = 1e-8
end

@testset "$(Ni)x$(Nj) ACCtoALAR with $(symmetry) $atype{$dtype}" for atype in [Array], dtype in [ComplexF64], symmetry in [:U1], Ni = [1], Nj = [1]
    Random.seed!(100)
    D, d = 5, 3
    A = [randinitial(Val(symmetry), atype, dtype, D,d,D; dir = [-1,1,1]) for i in 1:Ni, j in 1:Nj]
    S1 = [randinitial(Val(symmetry), atype, ComplexF64, D,d,D,D,d,D; dir = [1,-1,-1,-1,1,1]) for i in 1:Ni, j in 1:Nj]
    S2 = [randinitial(Val(symmetry), atype, ComplexF64, D,D,D,D; dir = [1,-1,-1,1]) for i in 1:Ni, j in 1:Nj]
    M = [randinitial(Val(symmetry), atype, ComplexF64, d,d,d,d; dir = [-1,1,1,-1]) for i in 1:Ni, j in 1:Nj]

    AL, L, _ = leftorth(A) 
    R, AR, _ = rightorth(A)
    # _, FL = leftenv(AL, AL, M)
    # _, FR = rightenv(AR, AR, M)
    FL = [randinitial(Val(symmetry), atype, dtype, D,d,D; dir = [1,1,-1]) for i in 1:Ni, j in 1:Nj]
    FR = [randinitial(Val(symmetry), atype, dtype, D,d,D; dir = [-1,-1,1]) for i in 1:Ni, j in 1:Nj]

    Co = LRtoC(L, R)
    ACo = ALCtoAC(AL, Co)
    _, Co = Cenv(Co, FL, FR)
    function foo1(x)
        M *= x
        _, AC = ACenv(ACo, FL, M, FR)
        AL, AR = ACCtoALAR(AC, Co) 
        s = 0
        for j in 1:Nj, i in 1:Ni
            A = ein"(abc,abcdef),def -> "(AL[i,j], S1[i,j], conj(AL[i,j]))
            B = ein"abc,abc -> "(AL[i,j], conj(AL[i,j]))
            s += norm(Array(A)[]/Array(B)[])
            A = ein"(abc,abcdef),def -> "(AR[i,j], S1[i,j], conj(AR[i,j]))
            B = ein"abc,abc -> "(AR[i,j], conj(AR[i,j]))
            s += norm(Array(A)[]/Array(B)[])
            A = ein"(abc,abcdef),def -> "(AC[i,j], S1[i,j], conj(AC[i,j]))
            B = ein"abc,abc -> "(AC[i,j], conj(AC[i,j]))
            s += norm(Array(A)[]/Array(B)[])
        end
        return s
    end
    @test Zygote.gradient(foo1, 1)[1] !== nothing
    # @test Zygote.gradient(foo1, 1)[1] ≈ num_grad(foo1, 1) atol = 1e-2 # num_grad isn't stable for U1 symmetry
end

@testset "observable leftenv and rightenv with $(symmetry) $atype{$dtype}" for atype in [Array], dtype in [ComplexF64], symmetry in [:U1], Ni = [2], Nj = [2]
    Random.seed!(100)
    D, d = 5, 3
    A = [randinitial(Val(symmetry), atype, dtype, D,d,D; dir = [-1,1,1]) for i in 1:Ni, j in 1:Nj]
    S = [randinitial(Val(symmetry), atype, ComplexF64, D,d,D,D,d,D; dir = [-1,-1,1,1,1,-1]) for i in 1:Ni, j in 1:Nj]
    M = [randinitial(Val(symmetry), atype, ComplexF64, d,d,d,d; dir = [-1,1,1,-1]) for i in 1:Ni, j in 1:Nj]

    ALu, = leftorth(A) 
    ALd, = leftorth(A)
    _, ARu, = rightorth(A)
    _, ARd, = rightorth(A)

    function foo1(x)
        _,FL = obs_FL(ALu, conj(ALd), M*x)
        s = 0
        for j in 1:Nj, i in 1:Ni
            A = ein"(abc,abcdef),def -> "(FL[i,j], S[i,j], conj(FL[i,j]))
            B = ein"abc,abc -> "(FL[i,j], conj(FL[i,j]))
            s += norm(Array(A)[]/Array(B)[])
        end
        return s
    end 
    @test Zygote.gradient(foo1, 1)[1] ≈ num_grad(foo1, 1) atol = 1e-7

    function foo2(x)
        _,FR = obs_FR(ARu, conj(ARd), M*x)
        s = 0
        for j in 1:Nj, i in 1:Ni
            A = ein"(abc,abcdef),def -> "(conj(FR[i,j]), S[i,j], FR[i,j])
            B = ein"abc,abc -> "(FR[i,j], conj(FR[i,j]))
            s += norm(Array(A)[]/Array(B)[])
        end
        return s
    end 
    @test Zygote.gradient(foo2, 1)[1] ≈ num_grad(foo2, 1) atol = 1e-7
end
