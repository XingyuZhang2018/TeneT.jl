@testset "Tensorkit convention" begin
    A = rand(ComplexF64, ℂ^2 * ℂ^3 ← ℂ^4)
    
    @test space(A, 1) == ℂ^2
    @test space(A, 2) == ℂ^3
    @test space(A, 3) == (ℂ^4)'

    @test dot(A, A) ≈ (@tensor B = conj(A[1 2; 3]) * A[1 2; 3]) ≈ (@tensor B = adjoint(A)[3; 1 2] * A[1 2; 3])
    @test A'*A ≈ (@tensor B[-1; -2] := conj(A[1 2; -1]) * A[1 2; -2]) ≈ (@tensor B[-1; -2] := adjoint(A)[-1; 1 2] * A[1 2; -2])
end

@testset "initialize A C" for M in Ms, (d, D, χ) in zip(ds, Ds, χs)
    Random.seed!(42)

    A = initial_A(M, χ)
    C = initial_C(A)
    @test size(A) == (2, 2)
    @test size(C) == (2, 2)
    @test A[1,1] == A[2,2]
    @test A[1,2] == A[2,1]
    @test C[1,1] == C[2,2]
    @test C[1,2] == C[2,1]

    if M isa ipeps
        @test all(i -> space(i) == (χ*D*D' ← χ), A)
    else
        @test all(i -> space(i) == (χ*D ← χ), A)
    end
    @test all(i -> space(i) == (χ ← χ), C)
end

@testset "getL!, getAL and getLsped" for M in Ms, (d, D, χ) in zip(ds, Ds, χs)
    Random.seed!(42)
    A = initial_A(M, χ)
    C = initial_C(A)
    L = getL!(A, C)
    @test L[1,1] == L[2,2]
    @test L[1,2] == L[2,1]

    @test all(i -> all(j -> real(j) > 0, diag(i).values[]), L)
    @test all(i -> all(j -> imag(j) ≈ 0, diag(i).values[]), L)
    @test all(i -> space(i) == (χ ← χ), L)

    AL, Le, λ = getAL(A, L)
    @test AL[1,1] == AL[2,2]
    @test AL[1,2] == AL[2,1]
    @test Le[1,1] == Le[2,2]
    @test Le[1,2] == Le[2,1]
    @test λ[1,1] == λ[2,2]
    @test λ[1,2] == λ[2,1]

    @test all(i -> space(i) == (χ ← χ), Le)
    if M isa ipeps
        @test all(i -> space(i) == (χ*D*D' ← χ), AL)
        @test all(map((λ, AL, Le, A, L) -> λ * AL * Le ≈ transpose(L*transpose(A, ((1,),(4,3,2))), ((1,4,3),(2,))), λ, AL, Le, A, L))
    else
        @test all(i -> space(i) == (χ*D ← χ), AL)
        @test all(map((λ, AL, Le, A, L) -> λ * AL * Le ≈ transpose(L*transpose(A, ((1,),(3,2))), ((1,3),(2,))), λ, AL, Le, A, L))
    end

    L = getLsped(Le, A, AL)
    @test L[1,1] == L[2,2]
    @test L[1,2] == L[2,1]
    @test all(i -> space(i) == (χ ← χ), L)
    @test all(i -> all(j -> real(j) > 0, diag(i).values[]), L)
    @test all(i -> all(j -> imag(j) ≈ 0, diag(i).values[]), L)
end

@testset "canonical form for unitcell" for M in Ms, (d, D, χ) in zip(ds, Ds, χs)
    Random.seed!(42)
    A = initial_A(M, χ)

    AL, L, λ = left_canonical(A)
    @test AL[1,1] == AL[2,2]
    @test AL[1,2] == AL[2,1]
    @test L[1,1] == L[2,2]
    @test L[1,2] == L[2,1]
    @test λ[1,1] == λ[2,2]
    @test λ[1,2] == λ[2,1]

    if M isa ipeps
        @test all(i -> space(i) == (χ * D * D' ← χ), AL)
    else
        @test all(i -> space(i) == (χ * D ← χ), AL)
    end
    @test all(i -> space(i) == (χ ← χ), L)
    @test all(AL -> (AL' * AL ≈ isomorphism(χ, χ)), AL)
    @test all(map((A, AL, L, λ) -> λ * AL * L ≈ _to_front(L * _to_tail(A)), A, AL, L, λ))

    R, AR, λ = right_canonical(A)
    @test AR[1,1] == AR[2,2]
    @test AR[1,2] == AR[2,1]
    @test R[1,1] == R[2,2]
    @test R[1,2] == R[2,1]
    @test λ[1,1] == λ[2,2]
    @test λ[1,2] == λ[2,1]
    if M isa ipeps
        @test all(i -> space(i) == (χ * D * D' ← χ), AR)
    else
        @test all(i -> space(i) == (χ * D ← χ), AR)
    end
    @test all(i -> space(i) == (χ ← χ), R)
    @test all(AR -> (_to_tail(AR) * _to_tail(AR)' ≈ isomorphism(χ, χ)), AR)
    @test all(map((A, R, AR, λ) -> _to_front(λ * R * _to_tail(AR)) ≈ A * R, A, R, AR, λ))
end

@testset "initialize FL FR" for M in Ms, (d, D, χ) in zip(ds, Ds, χs)
    Random.seed!(42)
    A = initial_A(M, χ)

    AL, L, λ = left_canonical(A)
    R, AR, λ = right_canonical(A)

    FL = initial_FL(AL, M)
    FR = initial_FR(AR, M)
    if M isa ipeps
        @test all(i -> space(i) == (χ * D' * D ← χ), FL)
        @test all(i -> space(i) == (χ * D * D' ← χ), FR)
    else
        @test all(i -> space(i) == (χ * D' ← χ), FL)
        @test all(i -> space(i) == (χ * D ← χ), FR)
    end
end

@testset "leftenv and rightenv for ifsimple_eig=$ifsimple_eig ifobs=$ifobs" for M in Ms, (d, D, χ) in zip(ds, Ds, χs), ifsimple_eig in [true, false], ifobs in [true, false]
    Random.seed!(42)

    A = initial_A(M, χ)
    AL, L, λ = left_canonical(A)
    R, AR, λ = right_canonical(A)
    alg = VUMPS(ifsimple_eig = true)

    λL, FL =  leftenv(AL, adjoint(AL), M; ifobs, ifvalue=true, alg)
    λR, FR = rightenv(AR, adjoint(AR), M; ifobs, ifvalue=true, alg)

    @test FL[1,1] == FL[2,2]
    @test FL[1,2] == FL[2,1]
    @test λL[1,1] == λL[2,2]
    @test λL[1,2] == λL[2,1]
    @test FR[1,1] == FR[2,2]
    @test FR[1,2] == FR[2,1]
    @test λR[1,1] == λR[2,2]
    @test λR[1,2] == λR[2,1]
    if M isa ipeps
        @test all(i -> space(i) == (χ * D' * D ← χ), FL)
        @test all(i -> space(i) == (χ * D * D' ← χ), FR)
    else
        @test all(i -> space(i) == (χ * D' ← χ), FL)
        @test all(i -> space(i) == (χ * D ← χ), FR)
    end

    Ni, Nj = size(A)
    for i in 1:Ni
        ir = ifobs ? Ni + 1 - i : mod1(i + 1, Ni)
        for j in 1:Nj
            @test λL[i,j] * FL[i,j] ≈ FLmap(j, FL[i,j], AL[i,:], adjoint.(AL[ir,:]), M[i,:]) rtol = 1e-12
            @test λR[i,j] * FR[i,j] ≈ FRmap(j, FR[i,j], AR[i,:], adjoint.(AR[ir,:]), M[i,:]) rtol = 1e-12
        end
    end
end


@testset "ACenv and Cenv for ifsimple_eig=$ifsimple_eig" for M in Ms, (d, D, χ) in zip(ds, Ds, χs), ifsimple_eig in [true, false]
    Random.seed!(42)

    A = initial_A(M, χ)
    AL, L, λ = left_canonical(A)
    R, AR, λ = right_canonical(A)

    alg = VUMPS(ifsimple_eig = true)
    λL, FL =  leftenv(AL, adjoint(AL), M; ifsimple_eig, alg)
    λR, FR = rightenv(AR, adjoint(AR), M; ifsimple_eig, alg)

     C = LRtoC(L, R)
    AC = ALCtoAC(AL, C)
    λAC, AC = ACenv(AC, FL, FR, M; ifsimple_eig, ifvalue=true, alg)
     λC,  C =  Cenv( C, FL, FR;    ifsimple_eig, ifvalue=true, alg) 
    if M isa ipeps
        @test all(i -> space(i) == (χ * D * D' ← χ), AC)
    else
        @test all(i -> space(i) == (χ * D ← χ), AC)
    end
    @test all(i -> space(i) == (χ ← χ),  C)

    Ni, Nj = size(A)
    for j in 1:Nj
        jr = mod1(j + 1, Nj)
        for i in 1:Ni
            ir = mod1(i + 1, Ni)
            @test λAC[j] * AC[i,j] ≈ ACmap(i, AC[i,j], FL[:,j], FR[:,j], M[:,j]) rtol = 1e-12
            @test  λC[j] *  C[i,j] ≈  Cmap(i,  C[i,j], FL[:,jr], FR[:,j]) rtol = 1e-10
        end
    end
end

@testset "ACCtoALAR for ifsimple_eig=$ifsimple_eig" for M in Ms, (d, D, χ) in zip(ds, Ds, χs), ifsimple_eig in [true, false]
    Random.seed!(42)

    A = initial_A(M, χ)

    AL, L, λ =  left_canonical(A)
    R, AR, λ = right_canonical(A)

       alg = VUMPS(ifsimple_eig = true)
    λL, FL =  leftenv(AL, adjoint(AL), M; alg, ifsimple_eig)
    λR, FR = rightenv(AR, adjoint(AR), M; alg, ifsimple_eig)

     C = LRtoC(L, R)
    AC = ALCtoAC(AL, C)

    λAC, AC = ACenv(AC, FL, FR, M; alg, ifsimple_eig)
     λC,  C =  Cenv( C, FL, FR; alg, ifsimple_eig) 

    AL, AR, errL, errR = ACCtoALAR(AC, C)
    if M isa ipeps
        @test all(i -> space(i) == (χ * D * D' ← χ), AL)
        @test all(i -> space(i) == (χ * D * D' ← χ), AR)
    else
        @test all(i -> space(i) == (χ * D ← χ), AL)
        @test all(i -> space(i) == (χ * D ← χ), AR)
    end
    @test all(AL -> (AL' * AL ≈ isomorphism(χ, χ)), AL)
    @test all(AR -> (_to_tail(AR) * _to_tail(AR)' ≈ isomorphism(χ, χ)), AR)
    @test errL isa Real
    @test errR isa Real
end

@testset "leftCenv for ifsimple_eig=$ifsimple_eig ifobs=$ifobs" for M in Ms, (d, D, χ) in zip(ds, Ds, χs), ifsimple_eig in [true, false], ifobs in [true, false]
    Random.seed!(42)

    A = initial_A(M, χ)
    AL, L, λ = left_canonical(A)
    R, AR, λ = right_canonical(A)
    alg = VUMPS(ifsimple_eig = true)

    λL, L =  leftCenv(AL, adjoint(AL); ifobs, ifvalue=true, alg)

    @test L[1,1] == L[2,2]
    @test L[1,2] == L[2,1]
    @test λL[1,1] == λL[2,2]
    @test λL[1,2] == λL[2,1]
    @test all(i -> space(i) == (χ ← χ), L)

    Ni, Nj = size(A)
    for i in 1:Ni
        ir = ifobs ? Ni + 1 - i : mod1(i + 1, Ni)
        for j in 1:Nj
            @test λL[i,j] * L[i,j] ≈ Lmap(j, L[i,j], AL[i,:], adjoint.(AL[ir,:])) rtol = 1e-12
        end
    end
end
