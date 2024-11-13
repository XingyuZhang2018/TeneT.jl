using TeneT
using TeneT:qrpos,lqpos,left_canonical,right_canonical,leftenv,FLmap,rightenv,FRmap,ACenv,ACmap,Cenv,Cmap,LRtoC,ALCtoAC,ACCtoALAR,error, env_norm
using CUDA
using LinearAlgebra
using Random
using Test
using OMEinsum
CUDA.allowscalar(false)

test_type = [Array, CuArray]

@testset "qr with $atype{$dtype}" for atype in test_type, dtype in [Float64, ComplexF64]
    Random.seed!(100)
    A = atype(rand(dtype, 4,4))
    Q, R = qrpos(A)
    @test Array(Q*R) ≈ Array(A)
    @test all(real.(diag(R)) .> 0)
    @test all(imag.(diag(R)) .≈ 0)
end

@testset "lq with $atype{$dtype}" for atype in test_type, dtype in [Float64, ComplexF64]
    Random.seed!(100)
    A = atype(rand(dtype, 4,4))
    L, Q = lqpos(A)
    @test Array(L*Q) ≈ Array(A)
    @test all(real.(diag(L)) .> 0)
    @test all(imag.(diag(L)) .≈ 0)
end

@testset "left_canonical and right_canonical with $atype{$dtype} $Ni x $Nj" for atype in test_type, dtype in [ComplexF64], Ni in 1:3, Nj in 1:3
    Random.seed!(100)
    χ, D = 3, 2
    A = [atype(rand(dtype, χ, D, χ)) for i in 1:Ni, j in 1:Nj]
    AL,  L, λ =  left_canonical(A)
     R, AR, λ = right_canonical(A)
     
    for j = 1:Nj,i = 1:Ni
        @test Array(ein"cda,cdb -> ab"(AL[i,j],conj(AL[i,j]))) ≈ I(χ)

        LA = reshape(L[i,j] * reshape(A[i,j], χ, D*χ), D*χ, χ)
        ALL = reshape(AL[i,j], χ*D, χ) * L[i,j] * λ[i,j]
        @test (Array(ALL) ≈ Array(LA))

        @test (Array(ein"acd,bcd -> ab"(AR[i,j],conj(AR[i,j]))) ≈ I(χ))

        AxR = reshape(reshape(A[i,j], D*χ, χ)*R[i,j], χ, D*χ)
        RAR = R[i,j] * reshape(AR[i,j], χ, D*χ) * λ[i,j]
        @test (Array(RAR) ≈ Array(AxR))
    end
end

@testset "leftenv and rightenv with $atype{$dtype} $Ni x $Nj" for atype in test_type, dtype in [ComplexF64], ifobs in [false, true], Ni in 1:3, Nj in 1:3
    Random.seed!(100)
    χ, D = 3, 2
    A = [atype(rand(dtype, χ, D, χ)) for i in 1:Ni, j in 1:Nj]
    M = [atype(rand(dtype, D, D, D, D)) for i in 1:Ni, j in 1:Nj]

    AL,    =  left_canonical(A)
    λL,FL  =  leftenv(AL, conj(AL), M; ifobs = ifobs)
    _, AR, = right_canonical(A)
    λR,FR  = rightenv(AR, conj(AR), M; ifobs = ifobs)

    for i in 1:Ni
        ir = ifobs ? Ni+1-i : mod1(i+1, Ni)
        for j in 1:Nj
            @test λL[i] * FL[i,j] ≈ FLmap(j, FL[i,j], AL[i,:], conj.(AL)[ir,:], M[i,:]) rtol = 1e-12
            @test λR[i] * FR[i,j] ≈ FRmap(j, FR[i,j], AR[i,:], conj.(AR)[ir,:], M[i,:]) rtol = 1e-12
        end
    end
end

@testset "ACenv and Cenv with $atype{$dtype} $Ni x $Nj" for atype in test_type, dtype in [ComplexF64], ifobs in [false, true], Ni in 1:3, Nj in 1:3
    Random.seed!(100)
    χ, D = 3, 2
    A = [atype(rand(dtype, χ, D, χ)) for i in 1:Ni, j in 1:Nj]
    M = [atype(rand(dtype, D, D, D, D)) for i in 1:Ni, j in 1:Nj]

    AL,  L, _ =  left_canonical(A)
     R, AR, _ = right_canonical(A)
    λL, FL    =  leftenv(AL, conj(AL), M)
    λR, FR    = rightenv(AR, conj(AR), M)

     C =  LRtoC(  L, R)
    AC = ALCtoAC(AL, C)

    λAC, AC = ACenv(AC, FL, M, FR)
     λC,  C =  Cenv( C, FL,    FR)

    for j in 1:Nj
        jr = mod1(j + 1, Nj)
        for i in 1:Ni
            ir = mod1(i + 1, Ni)
            @test λAC[j] * AC[i,j] ≈ ACmap(i, AC[i,j], FL[:,j],  FR[:,j], M[:,j]) rtol = 1e-12
            @test  λC[j] *  C[i,j] ≈  Cmap(i,  C[i,j], FL[:,jr], FR[:,j]) rtol = 1e-10
        end
    end
end

@testset "bcvumps unit test with $atype{$dtype} $Ni x $Nj" for atype in test_type, dtype in [ComplexF64], ifobs in [false, true], Ni in 1:3, Nj in 1:3
    Random.seed!(100)
    χ, D = 3, 2
    A = [atype(rand(dtype, χ, D, χ)) for i in 1:Ni, j in 1:Nj]
    M = [atype(rand(dtype, D, D, D, D)) for i in 1:Ni, j in 1:Nj]

    AL,  L, _ =  left_canonical(A)
     R, AR, _ = right_canonical(A)
    λL, FL    =  leftenv(AL, conj(AL), M)
    λR, FR    = rightenv(AR, conj(AR), M)

    C = LRtoC(L,R)
    AC = ALCtoAC(AL,C)
    
    λAC, AC = ACenv(AC, FL, M, FR)
     λC,  C =  Cenv( C, FL,    FR)
    AL, AR, errL, errR = ACCtoALAR(AC, C)
    @test errL + errR !== nothing
end