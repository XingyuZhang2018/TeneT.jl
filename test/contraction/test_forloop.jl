@testset "forloop" for atype in [Array, CuArray]
    d, D, χ = 2, 3, 10
    Au = atype(rand(ComplexF64, χ,D,D,χ))
    Ad = atype(rand(ComplexF64, χ,D,D,χ))
    # M  = atype(rand(ComplexF64, D,D,D,D,d))
    M  = (atype(rand(ComplexF64, D,D,D,D,d)), atype(rand(ComplexF64, D,D,D,D,d)))
    FL = atype(rand(ComplexF64, χ,D,D,χ))
    FR = atype(rand(ComplexF64, χ,D,D,χ))

    # FLmap
    FLm1 = FLmap(FL, Au, Ad, M)
    for i in 1:5
        FLm2 = FLmap_forloop(FL, Au, Ad, M; forloop_iter=i)
        @test FLm1 ≈ FLm2
    end

    # FRmap
    FRm1 = FRmap(FR, Au, Ad, M)
    for i in 1:5
        FRm2 = FRmap_forloop(FR, Au, Ad, M; forloop_iter=i)
        @test FRm1 ≈ FRm2
    end

    # ACmap
    ACm1 = ACmap(Au, FL, FR, M)
    for i in 1:5
        ACm2 = ACmap_forloop(Au, FL, FR, M; forloop_iter=i)
        @test ACm1 ≈ ACm2
    end

    # ACdmap
    ACdm1 = ACdmap(Ad, FL, FR, M)
    for i in 1:5
        ACdm2 = ACdmap_forloop(Ad, FL, FR, M; forloop_iter=i)
        @test ACdm1 ≈ ACdm2
    end

    # Mmap
    FL3 = reshape(FL, (χ, D^2, χ))
    FR3 = reshape(FR, (χ, D^2, χ))
    Au3 = reshape(Au, (χ, D^2, χ))
    Ad3 = reshape(Ad, (χ, D^2, χ))
    M1 = Mmap(Au3, Ad3, FL3, FR3)
    for i in 1:5
        M2 = Mmap_forloop(Au3, Ad3, FL3, FR3; forloop_iter=i)
        @test M1 ≈ M2
    end

    M  = atype(rand(ComplexF64, D,D,D,D,d))
    # Mumap
    M1 = Mumap(Au, Ad, FL, FR, M)
    for i in 1:5
        M2 = Mumap_forloop(Au, Ad, FL, FR, M; forloop_iter=i)
        @test M1 ≈ M2
    end

    # Mdmap
    M1 = Mdmap(Au, Ad, FL, FR, M)
    for i in 1:5
        M2 = Mdmap_forloop(Au, Ad, FL, FR, M; forloop_iter=i)
        @test M1 ≈ M2
    end
end

@testset "backward forloop" for atype in [Array, CuArray]
    d, D, χ = 2, 3, 20
    Au = atype(randn(ComplexF64, χ,D,D,χ))
    Ad = atype(randn(ComplexF64, χ,D,D,χ))
    # M  = atype(randn(ComplexF64, D,D,D,D,d))
    M  = (atype(randn(ComplexF64, D,D,D,D,d)), atype(randn(ComplexF64, D,D,D,D,d)))
    FL = atype(randn(ComplexF64, χ,D,D,χ))
    FR = atype(randn(ComplexF64, χ,D,D,χ))

    # FLmap
    dFL1, dAu1, dAd1, dM1 = pullback(FLmap, FL, Au, Ad, M)[2](FL)
    for i in 1:1
        _, dFL2, dAu2, dAd2, dM2 = ChainRulesCore.rrule(FLmap_forloop, FL, Au, Ad, M; forloop_iter=i)[2](FL)
        @test dFL1 ≈ dFL2
        @test dAu1 ≈ dAu2
        @test dAd1 ≈ dAd2
        @test all(dM1 .≈ dM2)
    end

    # FRmap
    dFR1, dAu1, dAd1, dM1 = pullback(FRmap, FR, Au, Ad, M)[2](FR)
    for i in 1:5
        _, dFR2, dAu2, dAd2, dM2 = ChainRulesCore.rrule(FRmap_forloop, FR, Au, Ad, M; forloop_iter=i)[2](FR)
        @test dFR1 ≈ dFR2
        @test dAu1 ≈ dAu2
        @test dAd1 ≈ dAd2
        @test all(dM1 .≈ dM2)
    end

    # ACmap
    dAu1, dFL1, dFR1, dM1 = pullback(ACmap, Au, FL, FR, M)[2](Au)
    for i in 1:5
        _, dAu2, dFL2, dFR2, dM2 = ChainRulesCore.rrule(ACmap_forloop, Au, FL, FR, M; forloop_iter=i)[2](Au)
        @test dAu1 ≈ dAu2
        @test dFL1 ≈ dFL2
        @test dFR1 ≈ dFR2
        @test all(dM1 .≈ dM2)
    end
end