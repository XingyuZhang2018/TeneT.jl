# M2 map-chain tests. Run: julia --project=. test/test_chain_maps.jl
using Test, LinearAlgebra, Random, Zygote
using TeneT
using TeneT: Chain, chain_interlabels, tensor_chain, tensor_pinned_inters,
             conj_variant, chain_apply, use_chain_engine, CHAIN_ENGINE
using TensorOperations: tensorcontract

@testset "tensor_pinned_inters matches @tensor temp layouts" begin
    # FLmap leg4 — probe-verified:
    ops = ((:a,:d,:f), (:f,:g,:h), (:d,:g,:e,:b), (:a,:b,:c))
    @test tensor_pinned_inters(ops, (:c,:e,:h)) == ((:a,:h,:d,:g), (:h,:e,:a,:b))
    # ACmap leg5 — probe-verified layouts (review round, two independent runs):
    ops5 = ((:a,:b,:c,:d), (:d,:g,:h,:l), (:e,:j,:g,:b,:p), (:f,:k,:h,:c,:p), (:a,:e,:f,:i))
    @test tensor_pinned_inters(ops5, (:i,:j,:k,:l)) ==
          ((:a,:c,:h,:l,:b,:g), (:a,:l,:e,:j,:c,:h,:p), (:l,:j,:k,:a,:e,:f))
    # N=3 edge case (one intermediate), N=2 → nothing
    @test tensor_pinned_inters(((:a,:b), (:b,:c), (:c,:d)), (:a,:d)) isa NTuple{1,Tuple}
    @test tensor_pinned_inters(((:a,:b), (:b,:c)), (:a,:c)) === nothing
end

@testset "toggle + chainability guard" begin
    @test CHAIN_ENGINE[] == false                       # M2 default until Task 11
    A = rand(ComplexF64, 2, 2); V = [A, A]
    @test TeneT._chainable(A)
    @test TeneT._chainable(view(A, :, 1:1))
    @test !TeneT._chainable(V)                          # Vector-of-arrays excluded
    @test TeneT._chainable((A, A)) && !TeneT._chainable((A, V))
    try
        TeneT.set_chain_engine!(true)
        @test use_chain_engine(A, (A, A))
        @test !use_chain_engine(A, V)
    finally
        TeneT.set_chain_engine!(false)
    end
    @test !use_chain_engine(A)
end

@testset "rrule(chain_apply) under Zygote == Zygote over @tensor" begin
    Random.seed!(31)
    χ, D, d = 8, 3, 2
    FL  = rand(ComplexF64, χ, D, D, χ); ALu = rand(ComplexF64, χ, D, D, χ)
    ALd = rand(ComplexF64, χ, D, D, χ)
    M1  = rand(ComplexF64, D, D, D, D, d); M2 = rand(ComplexF64, D, D, D, D, d)
    loss_chain(t...) = sum(abs2, chain_apply(TeneT.FLMAP_LEG5_CHAIN, t))
    loss_tensor(fl, ald, m1, m2, alu) = sum(abs2, TeneT.FLmap(fl, alu, ald, m1, m2))
    gc = Zygote.gradient(loss_chain, FL, ALd, M1, M2, ALu)
    gt = Zygote.gradient(loss_tensor, FL, ALd, M1, M2, ALu)
    for i in 1:5; @test gc[i] ≈ gt[i] rtol = 1e-10; end
end

@testset "_chain_map inner_etype cast path" begin
    Random.seed!(32)
    χ, D = 8, 3
    # FLmap leg4 geometry (basic.jl:56 / test_contraction.jl:68-74):
    FL  = rand(ComplexF64, χ, D, χ); ALd = rand(ComplexF64, χ, D, χ)
    M   = rand(ComplexF64, D, D, D, D); ALu = rand(ComplexF64, χ, D, χ)
    ch  = tensor_chain(((:a,:d,:f), (:f,:g,:h), (:d,:g,:e,:b), (:a,:b,:c)), (:c,:e,:h))
    r64 = TeneT._chain_map(ch, (FL, ALd, M, ALu), nothing)
    r32 = TeneT._chain_map(ch, (FL, ALd, M, ALu), Float32)
    @test eltype(r32) == ComplexF64                     # upcast at exit
    @test r32 ≈ r64 rtol = 1e-5                         # F32 accuracy
    @test TeneT._chain_map(ch, (FL, ALd, M, ALu), Float64) ≈ r64 rtol = 1e-12  # identity branch: inner_etype == real(eltype) skips the cast
end

@testset "FLmap chains: parity over all variants" begin
    Random.seed!(41)
    χ, D, d = 8, 3, 2
    FL4  = rand(ComplexF64, χ, D, χ);    ALu4 = rand(ComplexF64, χ, D, χ)
    ALd4 = rand(ComplexF64, χ, D, χ);    M4   = rand(ComplexF64, D, D, D, D)
    FL5  = rand(ComplexF64, χ, D, D, χ); ALu5 = rand(ComplexF64, χ, D, D, χ)
    ALd5 = rand(ComplexF64, χ, D, D, χ); M5   = rand(ComplexF64, D, D, D, D, d)
    M8   = rand(ComplexF64, D, D, D, D, D, D, D, D)
    cases = [
        ((FL4, ALu4, ALd4, M4),        "leg4"),
        ((FL5, ALu5, ALd5, M5),        "leg5 single-M"),
        ((FL5, ALu5, ALd5, (M5, conj(M5))), "leg5 tuple"),
        ((FL5, ALu5, ALd5, M8),        "leg8"),
    ]
    for (args, name) in cases
        TeneT.set_chain_engine!(false)
        ref = TeneT.FLmap(args...)
        gref = Zygote.gradient((a...) -> sum(abs2, TeneT.FLmap(a...)), args...)
        geng = try                       # toggle hygiene: never leak ON state
            TeneT.set_chain_engine!(true)
            @test TeneT.FLmap(args...) ≈ ref rtol = 1e-12
            Zygote.gradient((a...) -> sum(abs2, TeneT.FLmap(a...)), args...)
        finally
            TeneT.set_chain_engine!(false)
        end
        for i in 1:4
            if gref[i] isa Tuple
                for j in 1:2; @test geng[i][j] ≈ gref[i][j] rtol = 1e-10; end
            else
                @test geng[i] ≈ gref[i] rtol = 1e-10
            end
        end
        # engine_backward registry (the forloop-reroute entry point):
        dOut = rand(ComplexF64, size(ref))
        _, bk = Zygote.pullback((a...) -> TeneT.FLmap(a...), args...)
        gz = bk(dOut)
        ge = TeneT.engine_backward(TeneT.FLmap, args, dOut)
        @test ge !== nothing
        for i in 1:4
            if gz[i] isa Tuple
                for j in 1:2; @test ge[i][j] ≈ gz[i][j] rtol = 1e-10; end
            else
                @test ge[i] ≈ gz[i] rtol = 1e-10
            end
        end
    end
    # inner_etype path survives the reroute (mirrors test_contraction.jl):
    r32 = try
        TeneT.set_chain_engine!(true)
        TeneT.FLmap(FL5, ALu5, ALd5, M5; inner_etype=Float32)
    finally
        TeneT.set_chain_engine!(false)
    end
    @test r32 ≈ TeneT.FLmap(FL5, ALu5, ALd5, M5; inner_etype=Float32) rtol = 1e-5
end

println("test_chain_maps done")
