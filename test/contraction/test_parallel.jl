function test_parallel_iters(ref, par_fn, args...)
    for i in 1:5
        @test ref ≈ par_fn(args...; ifparallel=false, forloop_iter=i)
    end
end

@testset "forloop" for atype in [Array, CuArray]
    d, D, χ = 2, 3, 10
    Au = atype(rand(ComplexF64, χ,D,D,χ))
    Ad = atype(rand(ComplexF64, χ,D,D,χ))
    M  = (atype(rand(ComplexF64, D,D,D,D,d)), atype(rand(ComplexF64, D,D,D,D,d)))
    FL = atype(rand(ComplexF64, χ,D,D,χ))
    FR = atype(rand(ComplexF64, χ,D,D,χ))

    @testset "$name" for (name, ref_fn, par_fn, args) in [
        ("FLmap",  FLmap,  FLmap_parallel,  (FL, Au, Ad, M)),
        ("FRmap",  FRmap,  FRmap_parallel,  (FR, Au, Ad, M)),
        ("ACmap",  ACmap,  ACmap_parallel,  (Au, FL, FR, M)),
        ("ACdmap", ACdmap, ACdmap_parallel, (Ad, FL, FR, M)),
    ]
        test_parallel_iters(ref_fn(args...), par_fn, args...)
    end

    M_single = atype(rand(ComplexF64, D,D,D,D,d))
    @testset "Mumap" begin
        test_parallel_iters(Mumap(Au, Ad, FL, FR, M_single), Mumap_parallel, Au, Ad, FL, FR, M_single)
    end
end

@testset "backward forloop" for atype in [Array, CuArray]
    d, D, χ = 2, 3, 20
    Au = atype(randn(ComplexF64, χ,D,D,χ))
    Ad = atype(randn(ComplexF64, χ,D,D,χ))
    M  = (atype(randn(ComplexF64, D,D,D,D,d)), atype(randn(ComplexF64, D,D,D,D,d)))
    FL = atype(randn(ComplexF64, χ,D,D,χ))
    FR = atype(randn(ComplexF64, χ,D,D,χ))

    function test_pullback_iters(ref_fn, par_fn, args, seed_arg)
        refs = pullback(ref_fn, args...)[2](seed_arg)
        for i in 1:5
            grads = pullback((args...)->par_fn(args...; ifparallel=false, forloop_iter=i), args...)[2](seed_arg)
            for (g1, g2) in zip(refs, grads)
                g1 isa Tuple ? (@test all(g1 .≈ g2)) : (@test g1 ≈ g2)
            end
        end
    end

    test_pullback_iters(FLmap,  FLmap_parallel,  (FL, Au, Ad, M), FL)
    test_pullback_iters(FRmap,  FRmap_parallel,  (FR, Au, Ad, M), FR)
    test_pullback_iters(ACmap,  ACmap_parallel,  (Au, FL, FR, M), Au)
    test_pullback_iters(ACdmap, ACdmap_parallel, (Ad, FL, FR, M), Ad) 
end
