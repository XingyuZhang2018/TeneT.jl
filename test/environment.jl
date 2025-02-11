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

@testset "left_canonical and right_canonical with $atype" for atype in test_type, a in test_As
    Random.seed!(100)

    len = length(unique(pattern))
    A = [atype(a) for i in 1:len]
    AL,  L, λ =  left_canonical(A)
     R, AR, λ = right_canonical(A)
     
    for i in 1:len
        @test Array(_to_tail(AL[i])' * _to_tail(AL[i])) ≈ I(χ)

        LA = _to_tail(L[i] * _to_front(A[i]))
        ALL = _to_tail(AL[i]) * L[i] * λ[i]
        @test (Array(ALL) ≈ Array(LA))

        @test Array(_to_front(AR[i]) * _to_front(AR[i])') ≈ I(χ)

        AxR = _to_front(_to_tail(A[i]) * R[i])
        RAR = R[i] * _to_front(AR[i]) * λ[i]
        @test (Array(RAR) ≈ Array(AxR))
    end
end

@testset "leftenv and rightenv with $atype" for atype in test_type, (a, m) in zip(test_As, test_Ms), ifobs in [false]
    Random.seed!(100)
    len = length(unique(pattern))
    A = [atype(a) for i in 1:len]
    M = [atype(m) for i in 1:len]

    AL, L, _ =  left_canonical(A)
    R, AR, _ = right_canonical(A)
    FL = FLint(AL, M)
    FR = FRint(AR, M)
    C = LRtoC(L, R)
    # λL,FL  =  leftenv(AL, conj(AL), M; ifobs = ifobs)
    # 
    # λR,FR  = rightenv(AR, conj(AR), M; ifobs = ifobs)

    # for i in 1:Ni
    #     ir = ifobs ? Ni+1-i : mod1(i+1, Ni)
    #     for j in 1:Nj
    #         @test λL[i] * FL[i,j] ≈ FLmap(j, FL[i,j], AL[i,:], conj.(AL)[ir,:], M[i,:]) rtol = 1e-12
    #         @test λR[i] * FR[i,j] ≈ FRmap(j, FR[i,j], AR[i,:], conj.(AR)[ir,:], M[i,:]) rtol = 1e-12
    #     end
    # end
end

@testset "ACenv and Cenv with $atype $Ni x $Nj" for atype in test_type, (a, m) in zip(test_As, test_Ms), ifobs in [false, true], Ni in 1:3, Nj in 1:3
    Random.seed!(100)
    A = [atype(a) for i in 1:Ni, j in 1:Nj]
    M = [atype(m) for i in 1:Ni, j in 1:Nj]

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

@testset "bcvumps unit test with $atype $Ni x $Nj" for atype in test_type, (a, m) in zip(test_As, test_Ms), ifobs in [false, true], Ni in 1:3, Nj in 1:3
    Random.seed!(100)
    A = [atype(a) for i in 1:Ni, j in 1:Nj]
    M = [atype(m) for i in 1:Ni, j in 1:Nj]

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