using VUMPS
using VUMPS:qrpos,lqpos,leftorth,rightorth,leftenv,FLmap,rightenv,FRmap,ACenv,ACmap,Cenv,Cmap,LRtoC,ALCtoAC,ACCtoALAR,error,bigleftenv,BgFLmap,bigrightenv,BgFRmap, obs_FL, obs_FR, norm_FL, norm_FR, norm_FLmap, norm_FRmap
using CUDA
using LinearAlgebra
using Random
using Test
using OMEinsum
CUDA.allowscalar(false)

@testset "qr with $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64, ComplexF64]
    Random.seed!(100)
    A = atype(rand(dtype, 4,4))
    Q, R = qrpos(A)
    @test Array(Q*R) ≈ Array(A)
    @test all(real.(diag(R)) .> 0)
    @test all(imag.(diag(R)) .≈ 0)
end

@testset "lq with $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64, ComplexF64]
    Random.seed!(100)
    A = atype(rand(dtype, 4,4))
    L, Q = lqpos(A)
    @test Array(L*Q) ≈ Array(A)
    @test all(real.(diag(L)) .> 0)
    @test all(imag.(diag(L)) .≈ 0)
end

@testset "leftorth and rightorth with $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64, ComplexF64], Ni = [2], Nj = [2]
    Random.seed!(100)
    D, d = 3, 2
    A = Array{atype{dtype,3},2}(undef, Ni, Nj)
    for j = 1:Nj, i = 1:Ni
        A[i,j] = atype(rand(dtype, D, d, D))
    end
    AL, L, λ = leftorth(A)
    R, AR, λ = rightorth(A)

    for j = 1:Nj,i = 1:Ni
        M = ein"cda,cdb -> ab"(AL[i,j],conj(AL[i,j]))
        @test (Array(M) ≈ I(D))

        LA = reshape(L[i,j] * reshape(A[i,j], D, d*D), d*D, D)
        ALL = reshape(AL[i,j], d*D, D) * L[i,j] * λ[i,j]
        @test (Array(ALL) ≈ Array(LA))

        M = ein"acd,bcd -> ab"(AR[i,j],conj(AR[i,j]))
        @test (Array(M) ≈ I(D))

        AxR = reshape(reshape(A[i,j], d*D, D)*R[i,j], D, d*D)
        RAR = R[i,j] * reshape(AR[i,j], D, d*D) * λ[i,j]
        @test (Array(RAR) ≈ Array(AxR))
    end
end

@testset "leftenv and rightenv with $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64, ComplexF64], Ni = [2], Nj = [2]
    Random.seed!(100)
    D, d = 3, 2
    A = Array{atype{dtype,3},2}(undef, Ni, Nj)
    M = Array{atype{dtype,4},2}(undef, Ni, Nj)
    for j = 1:Nj, i = 1:Ni
        A[i,j] = atype(rand(dtype, D, d, D))
        M[i,j] = atype(rand(dtype, d, d, d, d))
    end

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

@testset "observable leftenv and rightenv with $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64, ComplexF64], Ni = [2], Nj = [2]
    Random.seed!(100)
    D, d = 3, 2
    A = Array{atype{dtype,3},2}(undef, Ni, Nj)
    M = Array{atype{dtype,4},2}(undef, Ni, Nj)
    for j = 1:Nj, i = 1:Ni
        A[i,j] = atype(rand(dtype, D, d, D))
        M[i,j] = atype(rand(dtype, d, d, d, d))
    end

    AL, = leftorth(A)
    λL,FL = obs_FL(AL, AL, M)
    _, AR, = rightorth(A)
    λR,FR = obs_FR(AR, AR, M)

    for j = 1:Nj, i = 1:Ni
        ir = Ni + 1 - i
        @test λL[i,j] * FL[i,j] ≈ FLmap(AL[i,:], AL[ir,:], M[i,:], FL[i,j], j)
        @test λR[i,j] * FR[i,j] ≈ FRmap(AR[i,:], AR[ir,:], M[i,:], FR[i,j], j)
    end
end

@testset "ACenv and Cenv with $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64, ComplexF64], Ni = [2], Nj = [2]
    Random.seed!(100)
    D, d = 3, 2
    A = Array{atype{dtype,3},2}(undef, Ni, Nj)
    M = Array{atype{dtype,4},2}(undef, Ni, Nj)
    for j = 1:Nj, i = 1:Ni
        A[i,j] = atype(rand(dtype, D, d, D))
        M[i,j] = atype(rand(dtype, d, d, d, d))
    end

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

@testset "bcvumps unit test with $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64, ComplexF64], Ni = [2], Nj = [2]
    Random.seed!(100)
    D, d = 3, 2
    A = Array{atype{dtype,3},2}(undef, Ni, Nj)
    M = Array{atype{dtype,4},2}(undef, Ni, Nj)
    for j = 1:Nj, i = 1:Ni
        A[i,j] = atype(rand(dtype, D, d, D))
        M[i,j] = atype(rand(dtype, d, d, d, d))
    end

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

@testset "bigleftenv and bigrightenv with $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64, ComplexF64], Ni = [2], Nj = [2]
    Random.seed!(100)
    D, d = 3, 2
    A = Array{atype{dtype,3},2}(undef, Ni, Nj)
    M = Array{atype{dtype,4},2}(undef, Ni, Nj)
    for j = 1:Nj, i = 1:Ni
        A[i,j] = atype(rand(dtype, D, d, D))
        M[i,j] = atype(rand(dtype, d, d, d, d))
    end

    ALu, = leftorth(A)
    ALd, = leftorth(A)
    λL,BgFL = bigleftenv(ALu, ALd, M)
    _, ARu, = rightorth(A)
    _, ARd, = rightorth(A)
    λR,BgFR = bigrightenv(ARu, ARd, M)

    for j = 1:Nj, i = 1:Ni
        ir = i + 1 - Ni * (i==Ni)
        irr = i + 2 - Ni * (i + 2 > Ni)
        @test λL[i,j] * BgFL[i,j] ≈ BgFLmap(ALu[i,:], ALd[irr,:], M[i,:], M[ir,:], BgFL[i,j], j)
        @test λR[i,j] * BgFR[i,j] ≈ BgFRmap(ARu[i,:], ARd[irr,:], M[i,:], M[ir,:], BgFR[i,j], j)
    end
end

@testset "norm leftenv and rightenv with $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64, ComplexF64], Ni = [2], Nj = [2]
    Random.seed!(100)
    D, d = 3, 2
    A = Array{atype{dtype,3},2}(undef, Ni, Nj)
    for j = 1:Nj, i = 1:Ni
        A[i,j] = atype(rand(dtype, D, d, D))
    end

    ALu, = leftorth(A)
    ALd, = leftorth(A)
    λL, FL_norm = norm_FL(ALu, ALd)
    _, ARu, = rightorth(A)
    _, ARd, = rightorth(A)
    λR, FR_norm = norm_FR(ARu, ARd)

    for j = 1:Nj, i = 1:Ni
        @test λL[i,j] * FL_norm[i,j] ≈ norm_FLmap(ALu[i,:], ALu[i,:], FL_norm[i,j], j)
        @test λR[i,j] * FR_norm[i,j] ≈ norm_FRmap(ARu[i,:], ARd[i,:], FR_norm[i,j], j)
    end
end