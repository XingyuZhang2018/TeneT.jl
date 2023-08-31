using TeneT
using TeneT:qrpos,lqpos,leftorth,rightorth,leftenv,FLmap,rightenv,FRmap,ACenv,AC2env,ACmap,AC2map,Cenv,Cmap,LRtoC,ALCtoAC,ALCARtoAC,ACCtoALAR, AC2toALCAR, error, env_norm
using CUDA
using LinearAlgebra
using Random
using Test
using OMEinsum
CUDA.allowscalar(false)

@testset "qr with $atype{$dtype}" for atype in [Array], dtype in [Float64, ComplexF64]
    Random.seed!(100)
    A = atype(rand(dtype, 4,4))
    Q, R = qrpos(A)
    @test Array(Q*R) ≈ Array(A)
    @test all(real.(diag(R)) .> 0)
    @test all(imag.(diag(R)) .≈ 0)
end

@testset "lq with $atype{$dtype}" for atype in [Array], dtype in [Float64, ComplexF64]
    Random.seed!(100)
    A = atype(rand(dtype, 4,4))
    L, Q = lqpos(A)
    @test Array(L*Q) ≈ Array(A)
    @test all(real.(diag(L)) .> 0)
    @test all(imag.(diag(L)) .≈ 0)
end

@testset "leftorth and rightorth with $atype{$dtype}" for atype in [Array], dtype in [ComplexF64], Ni = [2], Nj = [2]
    Random.seed!(100)
    χ, D = 3, 2
    A = atype(rand(dtype, χ, D, χ, Ni, Nj))
    AL,  L, λ =  leftorth(A)
     R, AR, λ = rightorth(A)
     
    for j = 1:Nj,i = 1:Ni
        M = ein"cda,cdb -> ab"(AL[:,:,:,i,j],conj(AL[:,:,:,i,j]))
        @test (Array(M) ≈ I(χ))

        LA = reshape(L[:,:,i,j] * reshape(A[:,:,:,i,j], χ, D*χ), D*χ, χ)
        ALL = reshape(AL[:,:,:,i,j], χ*D, χ) * L[:,:,i,j] * λ[i,j]
        @test (Array(ALL) ≈ Array(LA))

        M = ein"acd,bcd -> ab"(AR[:,:,:,i,j],conj(AR[:,:,:,i,j]))
        @test (Array(M) ≈ I(χ))

        AxR = reshape(reshape(A[:,:,:,i,j], D*χ, χ)*R[:,:,i,j], χ, D*χ)
        RAR = R[:,:,i,j] * reshape(AR[:,:,:,i,j], χ, D*χ) * λ[i,j]
        @test (Array(RAR) ≈ Array(AxR))
    end
end

@testset "leftenv and rightenv with $atype{$dtype}" for atype in [Array], dtype in [ComplexF64], ifobs in [false, true], Ni = [3], Nj = [3]
    Random.seed!(100)
    χ, D = 3, 2
    A = atype(rand(dtype, χ, D, χ, Ni, Nj))
    M = atype(rand(dtype, D, D, D, D, Ni, Nj))

    AL,    =  leftorth(A)
    λL,FL  =  leftenv(AL, conj(AL), M; ifobs = ifobs)
    _, AR, = rightorth(A)
    λR,FR  = rightenv(AR, conj(AR), M; ifobs = ifobs)

    for i in 1:Ni
        ir = ifobs ? Ni+1-i : i+1-Ni*(i==Ni)
        @test λL[i] * FL[:,:,:,i,:] ≈ FLmap(AL[:,:,:,i,:], conj(AL[:,:,:,ir,:]), M[:,:,:,:,i,:], FL[:,:,:,i,:])
        @test λR[i] * FR[:,:,:,i,:] ≈ FRmap(AR[:,:,:,i,:], conj(AR[:,:,:,ir,:]), M[:,:,:,:,i,:], FR[:,:,:,i,:])
    end
end

@testset "ACenv and Cenv with $atype{$dtype}" for atype in [Array], dtype in [ComplexF64], Ni = [2], Nj = [2]
    Random.seed!(100)
    χ, D = 3, 2
    A = atype(rand(dtype, χ, D, χ, Ni, Nj))
    M = atype(rand(dtype, D, D, D, D, Ni, Nj))

    AL,  L, _ =  leftorth(A)
     R, AR, _ = rightorth(A)
    λL, FL    =  leftenv(AL, conj(AL), M)
    λR, FR    = rightenv(AR, conj(AR), M)

     C =  LRtoC(  L, R)
    AC = ALCtoAC(AL, C)

    λAC, AC = ACenv(AC, FL, M, FR)
     λC,  C =  Cenv( C, FL,    FR)

    for j in 1:Nj
        jr = j + 1 - Ni*(j==Ni)
        @test λAC[j] * AC[:,:,:,:,j] ≈ ACmap(AC[:,:,:,:,j], FL[:,:,:,:,j],  FR[:,:,:,:,j], M[:,:,:,:,:,j])
        @test  λC[j] *  C[:,:,  :,j] ≈  Cmap( C[:,:,  :,j], FL[:,:,:,:,jr], FR[:,:,:,:,j])
    end
end

@testset "AC2env with $atype{$dtype}" for atype in [Array], dtype in [ComplexF64], Ni = [2], Nj = [2]
    Random.seed!(100)
    χ, D = 3, 2
    A = atype(rand(dtype, χ, D, χ, Ni, Nj))
    M = atype(rand(dtype, D, D, D, D, Ni, Nj))

    AL,  L, _ =  leftorth(A)
     R, AR, _ = rightorth(A)
    λL, FL    =  leftenv(AL, conj(AL), M)
    λR, FR    = rightenv(AR, conj(AR), M)

      C = LRtoC(L, R)
    AC2 = ALCARtoAC(AL, C, AR)

    λAC2, AC2 = AC2env(AC2, FL, M, FR)

    for j in 1:Nj
        jr = mod1(j+1, Nj)
        @test λAC2[j] * AC2[:,:,:,:,:,j] ≈ AC2map(AC2[:,:,:,:,:,j], FL[:,:,:,:,j],  FR[:,:,:,:,j], M[:,:,:,:,:,j], M[:,:,:,:,:,jr])
    end
end

@testset "bcvumps unit test with $atype{$dtype}" for atype in [Array], dtype in [ComplexF64], Ni = [2], Nj = [2]
    Random.seed!(100)
    χ, D = 3, 2
    A = atype(rand(dtype, χ, D, χ, Ni, Nj))
    M = atype(rand(dtype, D, D, D, D, Ni, Nj))

    AL,  L, _ =  leftorth(A)
     R, AR, _ = rightorth(A)
    λL, FL    =  leftenv(AL, conj(AL), M)
    λR, FR    = rightenv(AR, conj(AR), M)

    C = LRtoC(L,R)
    AC = ALCtoAC(AL,C)
    
    λAC, AC = ACenv(AC, FL, M, FR)
     λC,  C =  Cenv( C, FL,    FR)
    AL, AR = ACCtoALAR(AC, C)
    err = error(AL,C,AR,FL,M,FR)
    @test err !== nothing
end

@testset "bcvumps unit test with $atype{$dtype}" for atype in [Array], dtype in [ComplexF64], Ni = [2], Nj = [2]
    Random.seed!(100)
    χ, D = 3, 2
    A = atype(rand(dtype, χ, D, χ, Ni, Nj))
    M = atype(rand(dtype, D, D, D, D, Ni, Nj))

    AL,  L, _ =  leftorth(A)
     R, AR, _ = rightorth(A)
    λL, FL    =  leftenv(AL, conj(AL), M)
    λR, FR    = rightenv(AR, conj(AR), M)

    C = LRtoC(L,R)
    AC2 = ALCARtoAC(AL, C, AR)
    
    λAC2, AC2 = AC2env(AC2, FL, M, FR)

    AL, C, AR, err1 = AC2toALCAR(AC2)
    err2 = error(AL,C,AR,FL,M,FR)
    @show err1 err2
end