using TeneT
using TeneT: qrpos,lqpos,leftorth,rightorth,leftenv,FLmap,rightenv,FRmap,ACenv,ACmap,Cenv,Cmap,LRtoC,ALCtoAC,ACCtoALAR,error
using TeneT: _arraytype, asArray, FLint
using CUDA
using LinearAlgebra
using Random
using Test
using OMEinsum
CUDA.allowscalar(false)

@testset "qr with $(symmetry) $atype{$dtype}" for atype in [Array], dtype in [ComplexF64], symmetry in [:none]
    Random.seed!(100)
    A = randinitial(Val(symmetry), atype, dtype, 10,10; dir = [-1, 1], q = [0])
    Q, R = qrpos(A)
    @test Q*R ≈ A
    @test all(real.(diag(R)) .> 0)
    @test all(imag.(diag(R)) .≈ 0)
end

@testset "lq with $(symmetry) $atype{$dtype}" for atype in [Array], dtype in [ComplexF64], symmetry in [:none]
    Random.seed!(100)
    A = randinitial(Val(symmetry), atype, dtype, 10, 10; dir = [1,-1], q = [0])
    L, Q = lqpos(A)
    @test L*Q ≈ A
    @test all(real.(diag(L)) .> 0)
    @test all(imag.(diag(L)) .≈ 0)
end

@testset "leftorth and rightorth with $(symmetry) $atype{$dtype}" for atype in [Array], dtype in [ComplexF64], symmetry in [:none], Ni = [1], Nj = [1]
    Random.seed!(100)
    χ, D = 4, 4
    # A = [randinitial(Val(symmetry), atype, dtype, χ, D, χ; dir = [-1,1,1]) for i in 1:Ni, j in 1:Nj]
    A = [symmetryreshape(randinitial(Val(symmetry), atype, dtype, χ,Int(sqrt(D)),Int(sqrt(D)),χ; dir = [-1, -1, 1, 1]), χ,D,χ)[1] for i in 1:Ni, j in 1:Nj]
    AL, L, λ = leftorth(A)
    R, AR, λ = rightorth(A)
    for j = 1:Nj,i = 1:Ni
        M = ein"cda,cdb -> ab"(AL[i,j],conj(AL[i,j]))
        M = asArray(M)
        @test (Array(M) ≈ I(χ))

        LA = ein"ab, bcd -> acd"(L[i,j], A[i,j])
        ALL = ein"abc, cd -> abd"(AL[i,j], L[i,j]) * λ[i,j]
        @test ALL ≈ LA

        M = ein"acd,bcd -> ab"(AR[i,j],conj(AR[i,j]))
        M = asArray(M)
        @test (Array(M) ≈ I(χ))

        AxR = ein"abc, cd -> abd"(A[i,j], R[i,j])
        RAR = ein"ab, bcd -> acd"(R[i,j], AR[i,j]) * λ[i,j]
        @test RAR ≈ AxR
    end
end

@testset "leftenv and rightenv with $(symmetry) $atype{$dtype}" for atype in [Array], dtype in [ComplexF64], symmetry in [:none], ifobs in [false, true], Ni = [2], Nj = [2]
    Random.seed!(100)
    χ, D = 2, 4
    A = [symmetryreshape(randinitial(Val(symmetry), atype, dtype, χ,Int(sqrt(D)),Int(sqrt(D)),χ; dir = [-1, -1, 1, 1]), χ,D,χ)[1] for i in 1:Ni, j in 1:Nj]
    T = [randinitial(Val(symmetry), atype, dtype, Int(sqrt(D)),Int(sqrt(D)),4,Int(sqrt(D)),Int(sqrt(D)); dir = [-1,-1,1,1,1], q=[0]) for i in 1:Ni, j in 1:Nj]
    # M = [randinitial(Val(symmetry), atype, dtype, d, d, d, d; dir = [-1,1,1,-1]) for i in 1:Ni, j in 1:Nj]
    TT = [ein"abcde, fgchi -> gbhdiefa"(T[i,j], conj(T[i,j])) for i in 1:Ni, j in 1:Nj]
    M = [symmetryreshape(TT[i,j], D,D,D,D)[1] for i in 1:Ni, j in 1:Nj]

    AL, = leftorth(A)
    FL = FLint(AL, M; info = nothing)
    λL,FL = leftenv(AL, AL, M, FL)
    _, AR, = rightorth(A)
    λR,FR = rightenv(AR, AR, M)

    ALt = asArray(AL)
    ARt = asArray(AR)
    Mt = asArray(M)
    λLt, FLt = leftenv(ALt, ALt, Mt)
    λRt, FRt = rightenv(ARt, ARt, Mt)
    @test λL ≈ λLt 
    @test λR ≈ λRt
    for i in 1:Ni
        ir = ifobs ? Ni+1-i : mod1(i+1, Ni)
        @test λL[i] * FL[i,:] ≈ FLmap(AL[i,:], AL[ir,:], M[i,:], FL[i,:])
        @test λR[i] * FR[i,:] ≈ FRmap(AR[i,:], AR[ir,:], M[i,:], FR[i,:])
    end
end


@testset "ACenv and Cenv with $(symmetry) $atype{$dtype}" for atype in [Array], dtype in [ComplexF64], symmetry in [:none], Ni = [2], Nj = [2]
    Random.seed!(100)
    χ, D = 4, 4
    A = [symmetryreshape(randinitial(Val(symmetry), atype, dtype, χ,Int(sqrt(D)),Int(sqrt(D)),χ; dir = [-1, -1, 1, 1]), χ,D,χ)[1] for i in 1:Ni, j in 1:Nj]
    T = [randinitial(Val(symmetry), atype, dtype, Int(sqrt(D)),Int(sqrt(D)),4,Int(sqrt(D)),Int(sqrt(D)); dir = [-1,-1,1,1,1], q=[0]) for i in 1:Ni, j in 1:Nj]
    # M = [randinitial(Val(symmetry), atype, dtype, d, d, d, d; dir = [-1,1,1,-1]) for i in 1:Ni, j in 1:Nj]
    TT = [ein"abcde, fgchi -> gbhdiefa"(T[i,j], conj(T[i,j])) for i in 1:Ni, j in 1:Nj]
    M = [symmetryreshape(TT[i,j], D,D,D,D)[1] for i in 1:Ni, j in 1:Nj]

    AL, L = leftorth(A)
    λL,FL = leftenv(AL, AL, M)
    R, AR, = rightorth(A)
    λR,FR = rightenv(AR, AR, M)

    C = LRtoC(L, R)
    AC = ALCtoAC(AL, C)
    
    λAC, AC = ACenv(AC, FL, M, FR)
    λC, C = Cenv(C, FL, FR)
    ACt, Ct, Mt, FLt, FRt = map(asArray, [AC, C, M, FL, FR])
    λACt, ACt = ACenv(ACt, FLt, Mt, FRt)
    λCt, Ct = Cenv(Ct, FLt, FRt)
    @test λAC ≈ λACt
    @test λC ≈ λCt
    for j in 1:Ni
        jr = mod1(j+1, Ni)
        @test λAC[j] * AC[:,j] ≈ ACmap(AC[:,j], FL[:,j],  FR[:,j], M[:,j])
        @test  λC[j] *  C[:,j] ≈  Cmap( C[:,j], FL[:,jr], FR[:,j])
    end
end

@testset "bcvumps unit test with $(symmetry) $atype{$dtype}" for atype in [Array], dtype in [ComplexF64], symmetry in [:none], Ni = [2], Nj = [2]
    Random.seed!(100)
    χ, D = 4, 4
    A = [symmetryreshape(randinitial(Val(symmetry), atype, dtype, χ,Int(sqrt(D)),Int(sqrt(D)),χ; dir = [-1, -1, 1, 1]), χ,D,χ)[1] for i in 1:Ni, j in 1:Nj]
    T = [randinitial(Val(symmetry), atype, dtype, Int(sqrt(D)),Int(sqrt(D)),4,Int(sqrt(D)),Int(sqrt(D)); dir = [-1,-1,1,1,1], q=[0]) for i in 1:Ni, j in 1:Nj]
    # M = [randinitial(Val(symmetry), atype, dtype, d, d, d, d; dir = [-1,1,1,-1]) for i in 1:Ni, j in 1:Nj]
    TT = [ein"abcde, fgchi -> gbhdiefa"(T[i,j], conj(T[i,j])) for i in 1:Ni, j in 1:Nj]
    M = [symmetryreshape(TT[i,j], D,D,D,D)[1] for i in 1:Ni, j in 1:Nj]

    AL, L = leftorth(A)
    λL,FL = leftenv(AL, AL, M)
    R, AR, = rightorth(A)
    λR,FR = rightenv(AR, AR, M)

    C = LRtoC(L,R)
    AC = ALCtoAC(AL,C)
    
    λAC, AC = ACenv(AC, FL, M, FR)
    λC, C = Cenv(C, FL, FR)
    AL, AR = ACCtoALAR(AC, C)
    err = error(AL,C,AR,FL,M,FR)
    @test err !== nothing

    for j = 1:Nj,i = 1:Ni
        M = ein"cda,cdb -> ab"(AL[i,j],conj(AL[i,j]))
        M = asArray(M)
        @test (Array(M) ≈ I(χ))

        M = ein"acd,bcd -> ab"(AR[i,j],conj(AR[i,j]))
        M = asArray(M)
        @test (Array(M) ≈ I(χ))
    end
end