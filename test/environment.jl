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

@testset "left_canonical and right_canonical with $atype" for atype in test_type, A in test_As
    Random.seed!(100)
    AL,  L, λ =  left_canonical(A)
     R, AR, λ = right_canonical(A)
    Ni, Nj = size(A)

    @test AL[1,1] == AL[2,2]
    # @test AL[2,1] == AL[1,2]
    @test AR[1,1] == AR[2,2]
    # @test AR[2,1] == AR[1,2]
    for p in 1:length(AL.data)
        i, j = Tuple(findfirst(==(p), AL.pattern))
        @test Array(_to_tail(AL[i,j])' * _to_tail(AL[i,j])) ≈ I(χ)

        LA = _to_tail(L[i,j] * _to_front(A[i,j]))
        ALL = _to_tail(AL[i,j]) * L[i,j] * λ[i,j]
        @test (Array(ALL) ≈ Array(LA))

        @test Array(_to_front(AR[i,j]) * _to_front(AR[i,j])') ≈ I(χ)

        AxR = _to_front(_to_tail(A[i,j]) * R[i,j])
        RAR = R[i,j] * _to_front(AR[i,j]) * λ[i,j]
        @test (Array(RAR) ≈ Array(AxR))
    end
end

@testset "leftenv and rightenv with $atype" for atype in test_type, (A, M) in zip(test_As, test_Ms), ifobs in [false, true]
    Random.seed!(100)
    χ, D = 3, 2

    AL,    =  left_canonical(A)
    alg = VUMPS(ifsimple_eig = true)
    λL,FL  =  leftenv(AL, conj(AL), M; ifobs=ifobs, ifvalue=true, alg=alg)

    _, AR, = right_canonical(A)
    λR,FR  = rightenv(AR, conj(AR), M; ifobs=ifobs, ifvalue=true, alg=alg)

    Ni, Nj = size(A)
    for i in 1:Ni
        ir = ifobs ? Ni+1-i : mod1(i+1, Ni)
        for j in 1:Nj
            @test λL[i,j] * FL[i,j] ≈ FLmap(j, FL[i,j], AL[i,:], conj(AL[ir,:]), M[i,:]) rtol = 1e-12
            @test λR[i,j] * FR[i,j] ≈ FRmap(j, FR[i,j], AR[i,:], conj(AR[ir,:]), M[i,:]) rtol = 1e-12
        end
    end
end

@testset "ACenv and Cenv with $atype" for atype in test_type, (A, M)  in zip(test_As, test_Ms), ifobs in [false, true]
    Random.seed!(100)
    Ni, Nj = size(A)

    AL,  L, _ =  left_canonical(A)
     R, AR, _ = right_canonical(A)
    alg = VUMPS(ifsimple_eig = true)
    λL, FL    =  leftenv(AL, conj(AL), M; ifobs=ifobs, ifvalue=true, alg=alg)
    λR, FR    = rightenv(AR, conj(AR), M; ifobs=ifobs, ifvalue=true, alg=alg)

     C =  LRtoC(  L, R)
    AC = ALCtoAC(AL, C)


    λAC, AC = ACenv(AC, FL, M, FR; ifvalue=true, alg=alg)
     λC,  C =  Cenv( C, FL,    FR; ifvalue=true, alg=alg)

    for j in 1:Nj
        jr = mod1(j + 1, Nj)
        for i in 1:Ni
            ir = mod1(i + 1, Ni)
            @test λAC[i,j] * AC[i,j] ≈ ACmap(i, AC[i,j], FL[:,j],  FR[:,j], M[:,j]) rtol = 1e-12
            @test  λC[i,j] *  C[i,j] ≈  Cmap(i,  C[i,j], FL[:,jr], FR[:,j]) rtol = 1e-10
        end
    end
end

@testset "bcvumps unit test with $atype" for atype in test_type, (A, M) in zip(test_As, test_Ms), ifobs in [false]
    Random.seed!(100)
    Ni, Nj = size(A)

    AL,  L, _ =  left_canonical(A)
     R, AR, _ = right_canonical(A)
    alg = VUMPS(ifsimple_eig = true)
    λL, FL    =  leftenv(AL, conj(AL), M; ifobs=ifobs, alg=alg)
    λR, FR    = rightenv(AR, conj(AR), M; ifobs=ifobs, alg=alg)

    C = LRtoC(L,R)
    AC = ALCtoAC(AL,C)
    
    λAC, AC = ACenv(AC, FL, M, FR; alg=alg)
     λC,  C =  Cenv( C, FL,    FR; alg=alg)
    AL, AR, errL, errR = ACCtoALAR(AC, C)
    @test errL + errR !== nothing
end