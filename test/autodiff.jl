begin "test utils"
    function num_grad(f, K; δ::Real=1e-5)
        if eltype(K) == ComplexF64
            (f(K + δ / 2) - f(K - δ / 2)) / δ + 
                (f(K + δ / 2 * 1.0im) - f(K - δ / 2 * 1.0im)) / δ * 1.0im
        else
            (f(K + δ / 2) - f(K - δ / 2)) / δ
        end
    end
    
    function num_grad(f, a::AbstractArray; δ::Real=1e-5)
        b = Array(copy(a))
        df = map(CartesianIndices(b)) do i
            foo = x -> (ac = copy(b); ac[i] = x; f(_arraytype(a)(ac)))
            num_grad(foo, b[i], δ=δ)
        end
        return _arraytype(a)(df)
    end

    function num_grad(f, a::StructArray; δ::Real=1e-5)
        b = copy(a)
        df = map(1:length(b.data)) do i
            foo = x -> (ac = copy(b); ac[i] = x; f(ac))
            num_grad(foo, b[i], δ=δ)
        end
        return df
    end
end

@testset "zygote mutable arrays with $atype{$dtype}" for atype in [Array], dtype in [ComplexF64]
    Random.seed!(100)
    function foo(F) 
        buf = Zygote.Buffer(F) # https://fluxml.ai/Zygote.jl/latest/utils/#Zygote.Buffer
        @inbounds @views for j in 1:2, i in 1:2 
            buf[:,:,:,i,j] = F[:,:,:,i,j]./norm(F[:,:,:,i,j]) 
        end
        return norm(copy(buf))
    end
    F = atype(rand(dtype, 3,2,3,2,2))
    @test Zygote.gradient(foo, F)[1] ≈ num_grad(foo, F) atol = 1e-8
end

@testset "loop_einsum mistake with $atype{$dtype}" for atype in [Array], dtype in [ComplexF64]
    Random.seed!(100)
    D = 5
    A = atype(rand(dtype, D,D,D))
    B = atype(rand(dtype, D,D))
    function foo(x)
        C = A * x
        D = B * x
        E = sum(ein"abc,abc -> "(C,C))
        F = sum(ein"ab,ab -> "(D,D))
        return norm(E/F)
        # E = ein"abc,abc -> "(C,C)[]
        # F = ein"ab,ab -> "(D,D)[]
        # return norm(E/F) mistake for GPU
    end 
    @test Zygote.gradient(foo, 1)[1] ≈ num_grad(foo, 1) atol = 1e-8
end

@testset "structarray AD for $atype" for atype in [Array, CuArray]
    Random.seed!(100)
    M = randSA(atype, [1 2; 2 1], [(1,2), (1,2)])
    function foo(M)
        return norm(M)
    end

    @test Zygote.gradient(foo, M)[1].data ≈ num_grad(foo, M) atol = 1e-8
end

@testset "structarray buffer for $atype" for atype in [Array]
    Random.seed!(100)
    M = randSA(atype, [1 2; 2 1], [(1,2), (1,2)])
    function foo(M)
        buff = Zygote.Buffer(M)
        buff[1,1] = M[1,1]
        buff[1,2] = M[1,2]
        return norm(copy(buff))
    end

    # @show Zygote.gradient(foo, M)[1]
    @test Zygote.gradient(foo, M)[1].data ≈ num_grad(foo, M) atol = 1e-8
end


@testset "QR factorization with $atype{$dtype}" for atype in [Array], dtype in [ComplexF64]
    Random.seed!(100)
    M = atype(rand(dtype, 3, 3))

    function foo(M)
        Q, R = qrpos(M)
        return norm(Q) + norm(R)
    end
    @test Zygote.gradient(foo, M)[1] ≈ num_grad(foo, M) atol = 1e-8
end

@testset "LQ factorization with $atype{$dtype}" for atype in [Array], dtype in [ComplexF64]
    Random.seed!(100)
    M = atype(rand(dtype, 3, 3))
    function foo(M)
        L, Q = lqpos(M)
        return  norm(Q) + norm(L)
    end
    @test Zygote.gradient(foo, M)[1] ≈ num_grad(foo, M) atol = 1e-8
end

@testset "leftenv and rightenv with $atype" for atype in [Array], (A, M, S) in zip(test_As, test_Ms, test_S1s), ifobs in [false]
    Random.seed!(100)

       ALu, =  left_canonical(A) 
       ALd, =  left_canonical(A)
    _, ARu, = right_canonical(A)
    _, ARd, = right_canonical(A)
    alg = VUMPS(ifsimple_eig = false)
    Ni, Nj = size(M)

    function foo1(M)
        _, FL = leftenv(ALu, conj(ALd), M; ifobs, alg)
        s = 0.0
        for p in 1:length(M.data)
            i, j = Tuple(findfirst(==(p), M.pattern))
            A = sum(ein"(abc,abcdef),def -> "(FL[i,j], S[i,j], FL[i,j]))
            B = sum(ein"(abc,abc) -> "(FL[i,j], FL[i,j]))
            s += norm(B)
        end
        return s
    end 
    @test Zygote.gradient(foo1, M)[1].data ≈ num_grad(foo1, M) atol = 1e-7

    function foo2(M)
        _,FR = rightenv(ARu, conj(ARd), M; ifobs, alg)
        s = 0
        for p in 1:length(M.data)
            i, j = Tuple(findfirst(==(p), M.pattern))
            A  = Array(ein"(abc,abcdef),def -> "(FR[i,j], S[i,j], FR[i,j]))[]
            B  = Array(ein"abc,abc -> "(FR[i,j], FR[i,j]))[]
            s += norm(A/B)
        end
        return s
    end 
    @test Zygote.gradient(foo2, M)[1].data ≈ num_grad(foo2, M) atol = 1e-7
end

@testset "ACenv and Cenv with $atype" for atype in [Array], (A, M, S1, S2) in zip(test_As, test_Ms, test_S1s, test_S2s), ifobs in [false]
    Random.seed!(100)
    AL, L, _ =  left_canonical(A) 
    R, AR, _ = right_canonical(A)
    alg = VUMPS(ifsimple_eig = false)
    _, FL    =  leftenv(AL, conj(AL), M; ifobs, alg)
    _, FR    = rightenv(AR, conj(AR), M; ifobs, alg)

     C =   LRtoC( L, R)
    AC = ALCtoAC(AL, C)
    function foo1(M)
        _, AC = ACenv(AC, FL, M, FR; alg)
        s = 0
        for p in 1:length(M.data)
            i, j = Tuple(findfirst(==(p), M.pattern))
            A = sum(ein"(abc,abcdef),def -> "(AC[i,j], S1[i,j], AC[i,j]))
            B = sum(ein"abc,abc -> "(AC[i,j], AC[i,j]))
            s += norm(A/B)
        end
        return s
    end
    # @test Zygote.gradient(foo1, M)[1].data ≈ num_grad(foo1, M) atol = 1e-7

    function foo2(M)
        _, FL = leftenv(AL, conj(AL), M, FL; ifobs, alg)
        _, FR = rightenv(AR, conj(AR), M, FR; ifobs, alg)
        _, C = Cenv(C, FL, FR; alg)
        s = 0
        for p in 1:length(M.data)
            i, j = Tuple(findfirst(==(p), M.pattern))
            A = sum(ein"(ab,abcd),cd -> "(C[i,j], S2[i,j], C[i,j]))
            B = sum(ein"ab,ab -> "(C[i,j], C[i,j]))
            s += norm(A/B)
        end
        return s
    end
    @test Zygote.gradient(foo2, M)[1].data ≈ num_grad(foo2, M) atol = 1e-7
end

@testset "ACCtoALAR with $atype" for atype in [Array], (A, M, S1, S2) in zip(test_As, test_Ms, test_S1s, test_S2s), ifobs in [false]
    Random.seed!(42)

    AL, L, _ =  left_canonical(A) 
    R, AR, _ = right_canonical(A)
    alg = VUMPS(ifsimple_eig = false)
    _, FL    =  leftenv(AL, conj(AL), M; ifobs, alg)
    _, FR    = rightenv(AR, conj(AR), M; ifobs, alg)

     Co =   LRtoC( L, R)
    ACo = ALCtoAC(AL, Co)
    _, Co = Cenv(Co, FL, FR; alg)
    function foo1(M)
        _, AC = ACenv(ACo, FL, M, FR; alg)
        AL, AR = ACCtoALAR(AC, Co) 
        s = 0
        for p in 1:length(M.data)
            i, j = Tuple(findfirst(==(p), M.pattern))
            A  = sum(ein"(abc,abcdef),def -> "(AL[i,j], S1[i,j], AL[i,j]))
            B  = sum(ein"abc,abc -> "(AL[i,j], AL[i,j]))
            s += norm(A/B)
            A  = sum(ein"(abc,abcdef),def -> "(AR[i,j], S1[i,j], AR[i,j]))
            B  = sum(ein"abc,abc -> "(AR[i,j], AR[i,j]))
            s += norm(A/B)
            A  = sum(ein"(abc,abcdef),def -> "(AC[i,j], S1[i,j], AC[i,j]))
            B  = sum(ein"abc,abc -> "(AC[i,j], AC[i,j]))
            s += norm(A/B)
        end
        return s
    end
    @test Zygote.gradient(foo1, M)[1].data ≈ num_grad(foo1, M) atol = 1e-3
end

# @testset "$(Ni)x$(Nj) fix_gauge_vumps_step backward with $atype" for Ni in 1:1, Nj in 1:1, atype = [Array]
#     Random.seed!(100)

#     alg = VUMPS(maxiter=200, miniter=100, verbosity=3, ifupdown=false)
#     χ = 3
#     β = 0.2
#     model = Ising(Ni, Nj, β)
#     M = atype.(model_tensor(model, Val(:bulk)))
#     rt = VUMPSRuntime(M, χ, alg)
#     rt = leading_boundary(rt, M, alg)

#     function energy(β)
#         model = Ising(Ni, Nj, β)
#         M = atype.(model_tensor(model, Val(:bulk)))
#         rt′, _ = fix_gauge_vumps_step(rt, M, alg)
#         env = VUMPSEnv(rt′, M)
#         return real(observable(env, model, Val(:energy)))
#     end
#     @test Zygote.gradient(energy, β)[1] ≈ num_grad(energy, β)
# end

include("../example/exampletensors.jl")
include("../example/exampleobs.jl")

@testset "ising backward with $atype $ifupdown $pattern" for atype = [Array], ifupdown in [false, true], pattern in [[1 2 3; 2 3 1; 3 1 2]]
    # [1;;], [1 1; 1 1], [1 2; 2 1], [1 2; 3 4], [1 1; 2 2]
    # [1 3 2 2 3 1; 2 3 1 1 3 2]
    Random.seed!(100)
    alg = VUMPS(maxiter=10, 
                miniter=1, 
                maxiter_ad=3, 
                miniter_ad=3, 
                verbosity=2, 
                ifupdown=ifupdown, 
                ifdownfromup=true, 
                ifsimple_eig=false)
    χ = 10

    function energy(β)
        model = Ising(β)
        l = length(unique(pattern))
        data =[atype(model_tensor(model, Val(:bulk))) for _ in 1:l]
        M = StructArray(data, pattern)
        rt = VUMPSRuntime(M, χ, alg)
        rt′ = leading_boundary(rt, M, alg)
        env = VUMPSEnv(rt′, M, alg)
        return real(observable(env, model, pattern, Val(:energy)))
    end
    # @show energy(0.5)
    @test Zygote.gradient(energy, 0.3)[1] ≈ num_grad(energy, 0.3)
end