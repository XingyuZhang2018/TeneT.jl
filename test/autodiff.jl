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

    function num_grad(f, a::AbstractTensorMap; δ::Real=1e-5)
        b = Array(copy(a))
        df = map(1:length(b.data)) do i
            foo = x -> (ac = copy(b); ac[i] = x; f(_arraytype(a)(ac)))
            num_grad(foo, b[i], δ=δ)
        end
        return TensorMap(_arraytype(a)(df), a.space)
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

@testset "structarray AD for $atype" for atype in [Array, CuArray]
    Random.seed!(100)
    M = atype(rand(ComplexF64, [(1,2), (1,2)], [1 2; 2 1]))
    function foo(M)
        return norm(M)
    end

    @test Zygote.gradient(foo, M)[1].data ≈ num_grad(foo, M) atol = 1e-8
end

@testset "structarray buffer for $atype" for atype in [Array]
    Random.seed!(100)
    M = atype(rand(ComplexF64, [(1,2), (1,2)], [1 2; 2 1]))
    function foo(M)
        buff = Zygote.Buffer(M)
        buff[1,1] = M[1,1]
        buff[1,2] = M[1,2]
        return norm(copy(buff))
    end

    @test Zygote.gradient(foo, M)[1].data ≈ num_grad(foo, M) atol = 1e-8
end

@testset "leftenv and rightenv for ifsimple_eig=$ifsimple_eig ifobs=$ifobs" for M in Ms, (d, D, χ) in zip(ds, Ds, χs), ifsimple_eig in [true, false], ifobs in [true, false]
    Random.seed!(42)

    A = initial_A(M, χ)
    AL, L, λ = left_canonical(A)
    R, AR, λ = right_canonical(A)
    alg = VUMPS(ifsimple_eig=ifsimple_eig)

    if M isa ipeps
        S1 = rand(ComplexF64, [χ*D'*D*χ' ← χ*D'*D*χ',χ*D'*D*χ' ← χ*D'*D*χ'], M.pattern)
        S2 = rand(ComplexF64, [χ*D*D'*χ' ← χ*D*D'*χ',χ*D*D'*χ' ← χ*D*D'*χ'], M.pattern)
    else
        S1 = rand(ComplexF64, [χ*D'*χ' ← χ*D'*χ',χ*D'*χ' ← χ*D'*χ'], M.pattern) 
        S2 = rand(ComplexF64, [χ*D*χ' ← χ*D*χ',χ*D*χ' ← χ*D*χ'], M.pattern) 
    end
    function foo1(M)
        _, FL = leftenv(AL, adjoint(AL), M; ifobs, alg)
        s = 0.0
        for p in 1:length(M.data)
            i, j = Tuple(findfirst(==(p), M.pattern))
            if M isa ipeps
                @tensor A = conj(FL[i,j][1 2 3 4]) * S1[i,j][1 2 3 4; 5 6 7 8] * FL[i,j][5 6 7 8]
            else
                @tensor A = conj(FL[i,j][1 2 3]) * S1[i,j][1 2 3; 4 5 6] * FL[i,j][4 5 6]
            end
            B = dot(FL[i,j], FL[i,j])
            s += norm(A/B)
        end
        return s
    end 
    @test norm(Zygote.gradient(foo1, M)[1].data - num_grad(foo1, M)) < 1e-7

    function foo2(M)
        _, FR = rightenv(AR, adjoint(AR), M; ifobs, alg)
        s = 0
        for p in 1:length(M.data)
            i, j = Tuple(findfirst(==(p), M.pattern))
            if M isa ipeps
                @tensor A = conj(FR[i,j][1 2 3 4]) * S2[i,j][1 2 3 4; 5 6 7 8] * FR[i,j][5 6 7 8]
            else
                @tensor A = conj(FR[i,j][1 2 3]) * S2[i,j][1 2 3; 4 5 6] * FR[i,j][4 5 6]
            end
            B = dot(FR[i,j], FR[i,j])
            s += norm(A/B)
        end
        return s
    end 
    @test norm(Zygote.gradient(foo2, M)[1].data - num_grad(foo2, M)) < 1e-6
end

@testset "ACenv and Cenv for ifsimple_eig=$ifsimple_eig ifobs=$ifobs" for M in Ms, (d, D, χ) in zip(ds, Ds, χs), ifsimple_eig in [true, false], ifobs in [false]
    Random.seed!(100)
    A = initial_A(M, χ)
    AL, L, λ = left_canonical(A)
    R, AR, λ = right_canonical(A)
    alg = VUMPS(ifsimple_eig=ifsimple_eig)
    _, FL =  leftenv(AL, adjoint(AL), M; ifobs, alg)
    _, FR = rightenv(AR, adjoint(AR), M; ifobs, alg)

     C =   LRtoC( L, R)
    AC = ALCtoAC(AL, C)
    if M isa ipeps
        S1 = rand(ComplexF64, [χ*D*D'*χ' ← χ*D*D'*χ',χ*D*D'*χ' ← χ*D*D'*χ'], M.pattern)
    else
        S1 = rand(ComplexF64, [χ*D*χ' ← χ*D*χ',χ*D*χ' ← χ*D*χ'], M.pattern) 
    end
    S2 = rand(ComplexF64, [χ*χ' ← χ*χ',χ*χ' ← χ*χ'], M.pattern)
    function foo1(M)
        _, AC = ACenv(AC, FL, FR, M; alg)
        s = 0
        for p in 1:length(M.data)
            i, j = Tuple(findfirst(==(p), M.pattern))
            if M isa ipeps
                @tensor A = conj(AC[i,j][1 2 3 4]) * S1[i,j][1 2 3 4; 5 6 7 8] * AC[i,j][5 6 7 8]
            else
                @tensor A = conj(AC[i,j][1 2 3]) * S1[i,j][1 2 3; 4 5 6] * AC[i,j][4 5 6]
            end
            B = dot(AC[i,j], AC[i,j])
            s += norm(A/B)
        end
        return s
    end
    @test norm(Zygote.gradient(foo1, M)[1].data - num_grad(foo1, M)) < 1e-7

    function foo2(M)
        _, FL =  leftenv(AL, adjoint(AL), M, FL; ifobs, alg)
        _, FR = rightenv(AR, adjoint(AR), M, FR; ifobs, alg)
        _, C = Cenv(C, FL, FR; alg)
        s = 0
        for p in 1:length(M.data)
            i, j = Tuple(findfirst(==(p), M.pattern))
            @tensor A = conj(C[i,j][1 2]) * S2[i,j][1 2; 3 4] * C[i,j][3 4]
            B = dot(C[i,j], C[i,j])
            s += norm(A/B)
        end
        return s
    end
    @test norm(Zygote.gradient(foo2, M)[1].data - num_grad(foo2, M)) < 1e-6
end

@testset "ACCtoALAR for ifsimple_eig=$ifsimple_eig ifobs=$ifobs" for M in Ms, (d, D, χ) in zip(ds, Ds, χs), ifsimple_eig in [true, false], ifobs in [false]
    Random.seed!(42)

    A = initial_A(M, χ)
    AL, L, λ = left_canonical(A)
    R, AR, λ = right_canonical(A)
    alg = VUMPS(ifsimple_eig=ifsimple_eig)
    _, FL =  leftenv(AL, adjoint(AL), M; ifobs, alg)
    _, FR = rightenv(AR, adjoint(AR), M; ifobs, alg)

     Co =   LRtoC( L, R)
    ACo = ALCtoAC(AL, Co)
    _, Co = Cenv(Co, FL, FR; alg)
    if M isa ipeps
        S = rand(ComplexF64, [χ*D*D'*χ' ← χ*D*D'*χ',χ*D*D'*χ' ← χ*D*D'*χ'], M.pattern)
    else
        S = rand(ComplexF64, [χ*D*χ' ← χ*D*χ',χ*D*χ' ← χ*D*χ'], M.pattern) 
    end

    function foo(M)
        _, AC = ACenv(ACo, FL, FR, M; alg)
        AL, AR = ACCtoALAR(AC, Co) 
        s = 0
        for p in 1:length(M.data)
            i, j = Tuple(findfirst(==(p), M.pattern))
            if M isa ipeps
                @tensor A = conj(AL[i,j][1 2 3 4]) * S[i,j][1 2 3 4; 5 6 7 8] * AL[i,j][5 6 7 8]
            else
                @tensor A = conj(AL[i,j][1 2 3]) * S[i,j][1 2 3; 4 5 6] * AL[i,j][4 5 6]
            end
            B = dot(AL[i,j], AL[i,j])
            s += norm(A/B)
            if M isa ipeps
                @tensor A = conj(AR[i,j][1 2 3 4]) * S[i,j][1 2 3 4; 5 6 7 8] * AR[i,j][5 6 7 8]
            else
                @tensor A = conj(AR[i,j][1 2 3]) * S[i,j][1 2 3; 4 5 6] * AR[i,j][4 5 6]
            end
            B = dot(AR[i,j], AR[i,j])
            s += norm(A/B)
            if M isa ipeps
                @tensor A = conj(AC[i,j][1 2 3 4]) * S[i,j][1 2 3 4; 5 6 7 8] * AC[i,j][5 6 7 8]
            else
                @tensor A = conj(AC[i,j][1 2 3]) * S[i,j][1 2 3; 4 5 6] * AC[i,j][4 5 6]
            end
            B = dot(AC[i,j], AC[i,j])
            s += norm(A/B)
        end
        return s
    end
    # @show foo1(M)
    @test norm(Zygote.gradient(foo, M)[1].data - num_grad(foo, M)) < 1e-3
end

@testset "ising backward with $atype $ifupdown $pattern" for atype = [Array], ifupdown in [true], pattern in [[1;;], [1 1; 1 1], [1 2; 2 1], [1 2; 3 4], [1 1; 2 2], [1 2; 3 4; 5 6],
    [1 2 3 4 5 6; 4 5 6 1 2 3], [1 4; 2 5; 3 6; 4 1;5 2; 6 3]]
    Random.seed!(100)
    alg = VUMPS(maxiter=10, 
                miniter=1, 
                maxiter_ad=10, 
                miniter_ad=5, 
                verbosity=1, 
                ifupdown=ifupdown, 
                ifdownfromup=true, 
                ifsimple_eig=false)
    χ = ℂ^10

    function logZ(β)
        model = Ising(β)
        l = length(unique(pattern))
        data = [TensorMap(atype(model_tensor(model, Val(:bulk))), ℂ^2*ℂ^2 ← ℂ^2*ℂ^2) for _ in 1:l]
        M = StructArray(data, pattern)
        rt = VUMPSRuntime(M, χ, alg)
        rt′ = leading_boundary(rt, M, alg)
        env = VUMPSEnv(rt′, M, alg)
        return log(real(observable(env, model, pattern, Val(:Z); alg)))
    end

    function energy(β)
        model = Ising(β)
        l = length(unique(pattern))
        data = [TensorMap(atype(model_tensor(model, Val(:bulk))), ℂ^2*ℂ^2 ← ℂ^2*ℂ^2) for _ in 1:l]
        M = StructArray(data, pattern)
        rt = VUMPSRuntime(M, χ, alg)
        rt′ = leading_boundary(rt, M, alg)
        env = VUMPSEnv(rt′, M, alg)
        return real(observable(env, model, pattern, Val(:energy)))
    end

    β = 0.3
    @test Zygote.gradient(β->-logZ(β), β)[1] ≈ num_grad(β->-logZ(β), β) ≈ energy(β)
    @test Zygote.gradient(energy, β)[1] ≈ num_grad(energy, β)
end                            