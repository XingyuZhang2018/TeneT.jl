using VUMPS
using VUMPS: qrpos,lqpos,leftorth,rightorth,leftenv,FLmap,rightenv,FRmap,ACenv,ACmap,Cenv,Cmap,LRtoC,ALCtoAC,ACCtoALAR,error, obs_FL, obs_FR, norm_FL, norm_FR, norm_FLmap, norm_FRmap
using VUMPS: _arraytype, asArray
using CUDA
using LinearAlgebra
using Random
using Test
using OMEinsum
CUDA.allowscalar(false)

@testset "qr with $(symmetry) $atype{$dtype}" for atype in [Array], dtype in [ComplexF64], symmetry in [:none, :Z2, :U1]
    Random.seed!(100)
    A = randinitial(Val(symmetry), atype, dtype, 10,10; dir = [-1, 1])
    Q, R = qrpos(A)
    @test Q*R ≈ A
    @test all(real.(diag(R)) .> 0)
    @test all(imag.(diag(R)) .≈ 0)
end

@testset "lq with $(symmetry) $atype{$dtype}" for atype in [Array], dtype in [ComplexF64], symmetry in [:none, :Z2, :U1]
    Random.seed!(100)
    A = randinitial(Val(symmetry), atype, dtype, 10, 10; dir = [1,-1])
    L, Q = lqpos(A)
    @test L*Q ≈ A
    @test all(real.(diag(L)) .> 0)
    @test all(imag.(diag(L)) .≈ 0)
end

@testset "leftorth and rightorth with $(symmetry) $atype{$dtype}" for atype in [Array], dtype in [ComplexF64], symmetry in [:none, :Z2, :U1], Ni = [2], Nj = [2]
    Random.seed!(9)
    D, d = 10, 4
    A = [randinitial(Val(symmetry), atype, dtype, D, d, D; dir = [-1,1,1]) for i in 1:2, j in 1:2]
    AL, L, λ = leftorth(A)
    R, AR, λ = rightorth(A)
    for j = 1:Nj,i = 1:Ni
        M = ein"cda,cdb -> ab"(AL[i,j],conj(AL[i,j]))
        M = asArray(M)
        @test (Array(M) ≈ I(D))

        LA = ein"ab, bcd -> acd"(L[i,j], A[i,j])
        ALL = ein"abc, cd -> abd"(AL[i,j], L[i,j]) * λ[i,j]
        @test ALL ≈ LA

        M = ein"acd,bcd -> ab"(AR[i,j],conj(AR[i,j]))
        M = asArray(M)
        @test (Array(M) ≈ I(D))

        AxR = ein"abc, cd -> abd"(A[i,j], R[i,j])
        RAR = ein"ab, bcd -> acd"(R[i,j], AR[i,j]) * λ[i,j]
        @test RAR ≈ AxR
    end
end

@testset "leftenv and rightenv with $(symmetry) $atype{$dtype}" for atype in [Array], dtype in [ComplexF64], symmetry in [:none, :Z2, :U1], Ni = [2], Nj = [2]
    Random.seed!(100)
    D, d = 10, 4
    A = [randinitial(Val(symmetry), atype, dtype, D, d, D; dir = [-1,1,1]) for i in 1:2, j in 1:2]
    M = [randinitial(Val(symmetry), atype, dtype, d, d, d, d; dir = [-1,1,1,-1]) for i in 1:2, j in 1:2]

    AL, = leftorth(A)
    λL,FL = leftenv(AL, AL, M)
    _, AR, = rightorth(A)
    λR,FR = rightenv(AR, AR, M)

    for j = 1:Nj, i = 1:Ni
        ir = Ni + 1 - i
        @test λL[i,j] * FL[i,j] ≈ FLmap(AL[i,:], conj(AL[ir,:]), M[i,:], FL[i,j], j)
        @test λR[i,j] * FR[i,j] ≈ FRmap(AR[i,:], conj(AR[ir,:]), M[i,:], FR[i,j], j)
    end
end

@testset "observable leftenv and rightenv with $(symmetry) $atype{$dtype}" for atype in [Array], dtype in [ComplexF64], symmetry in [:none, :Z2, :U1], Ni = [2], Nj = [2]
    Random.seed!(100)
    D, d = 10, 4
    Au = [randinitial(Val(symmetry), atype, dtype, D, d, D; dir = [-1,1,1]) for i in 1:2, j in 1:2]
    Ad = [randinitial(Val(symmetry), atype, dtype, D, d, D; dir = [1,-1,-1]) for i in 1:2, j in 1:2]
    M = [randinitial(Val(symmetry), atype, dtype, d, d, d, d; dir = [-1,1,1,-1]) for i in 1:2, j in 1:2]

    ALu, = leftorth(Au)
    ALd, = leftorth(Ad)
    λL,FL = obs_FL(ALu, ALd, M)
    _, ARu, = rightorth(Au)
    _, ARd, = rightorth(Ad)
    λR,FR = obs_FR(ARu, ARd, M)

    for j = 1:Nj, i = 1:Ni
        ir = Ni + 1 - i
        @test λL[i,j] * FL[i,j] ≈ FLmap(ALu[i,:], ALd[ir,:], M[i,:], FL[i,j], j)
        @test λR[i,j] * FR[i,j] ≈ FRmap(ARu[i,:], ARd[ir,:], M[i,:], FR[i,j], j)
    end
end

@testset "ACenv and Cenv with $(symmetry) $atype{$dtype}" for atype in [Array], dtype in [ComplexF64], symmetry in [:none, :Z2, :U1], Ni = [2], Nj = [2]
    Random.seed!(100)
    D, d = 10, 4
    A = [randinitial(Val(symmetry), atype, dtype, D, d, D; dir = [-1,1,1]) for i in 1:2, j in 1:2]
    M = [randinitial(Val(symmetry), atype, dtype, d, d, d, d; dir = [-1,1,1,-1]) for i in 1:2, j in 1:2]

    AL, L = leftorth(A)
    λL,FL = leftenv(AL, AL, M)
    R, AR, = rightorth(A)
    λR,FR = rightenv(AR, AR, M)

    C = LRtoC(L, R)
    AC = ALCtoAC(AL, C)

    λAC, AC = ACenv(AC, FL, M, FR)
    λC, C = Cenv(C, FL, FR)
    for j = 1:Nj, i = 1:Ni
        jr = j + 1 - Nj * (j==Nj)
        @test λAC[i,j] * AC[i,j] ≈ ACmap(AC[i,j], FL[:,j], FR[:,j], M[:,j], i)
        @test λC[i,j] * C[i,j] ≈ Cmap(C[i,j], FL[:,jr], FR[:,j], i)
    end
end

@testset "bcvumps unit test with $(symmetry) $atype{$dtype}" for atype in [Array], dtype in [ComplexF64], symmetry in [:none, :Z2, :U1], Ni = [2], Nj = [2]
    Random.seed!(100)
    D, d = 10, 4
    A = [randinitial(Val(symmetry), atype, dtype, D, d, D; dir = [-1,1,1]) for i in 1:2, j in 1:2]
    M = [randinitial(Val(symmetry), atype, dtype, d, d, d, d; dir = [-1,1,1,-1]) for i in 1:2, j in 1:2]

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
end

@testset "norm leftenv and rightenv with $(symmetry) $atype{$dtype}" for atype in [Array], dtype in [ComplexF64], symmetry in [:none, :Z2, :U1], Ni = [2], Nj = [2]
    Random.seed!(10)
    D, d = 10, 4
    Au = [randinitial(Val(symmetry), atype, dtype, D, d, D; dir = [-1,1,1]) for i in 1:2, j in 1:2]
    Ad = [randinitial(Val(symmetry), atype, dtype, D, d, D; dir = [-1,-1,1]) for i in 1:2, j in 1:2]
    ALu, = leftorth(Au)
    ALd, = leftorth(Ad)
    λL, FL_norm = norm_FL(ALu, ALd)
    _, ARu, = rightorth(Au)
    _, ARd, = rightorth(Ad)
    λR, FR_norm = norm_FR(ARu, ARd)

    for j = 1:Nj, i = 1:Ni
        @test λL[i,j] * FL_norm[i,j] ≈ norm_FLmap(ALu[i,:], ALd[i,:], FL_norm[i,j], j)
        @test λR[i,j] * FR_norm[i,j] ≈ norm_FRmap(ARu[i,:], ARd[i,:], FR_norm[i,j], j)
    end
end