# M2 map-chain tests. Run: julia --project=. test/test_chain_maps.jl
using Test, LinearAlgebra, Random, Zygote
using TeneT
using TeneT: Chain, chain_interlabels, tensor_chain, tensor_pinned_inters,
             conj_variant, chain_apply, use_chain_engine, CHAIN_ENGINE
using TensorOperations: tensorcontract

# Compare two gradient collections in map-arg order; tuple-M slots compare
# elementwise (a tuple/non-tuple slot mismatch fails the isa test and skips
# the elementwise loop instead of double-reporting as a MethodError). A
# nonempty `label` wraps the comparisons in their own @testset.
function grads_match(ga, gb; rtol=1e-10, label::String="")
    body = () -> begin
        @test length(ga) == length(gb)
        for (a, b) in zip(ga, gb)
            if b isa Tuple
                if a isa Tuple
                    @test length(a) == length(b)
                    for (ai, bi) in zip(a, b); @test ai ≈ bi rtol = rtol; end
                else
                    @test a isa Tuple              # slot mismatch: bail early
                end
            else
                @test a ≈ b rtol = rtol
            end
        end
    end
    if isempty(label)
        body()
    else
        @testset "$label" begin body() end
    end
    return nothing
end

# One parity case: engine-OFF reference (fwd + Zygote grads) vs engine-ON
# (fwd 1e-12, Zygote grads 1e-10) plus the engine_backward registry entry vs
# a toggle-OFF Zygote pullback (1e-10). Both regions set the toggle
# EXPLICITLY (OFF for references, ON for engine paths) and restore the
# ambient value — so the references remain @tensor-path results even after
# Task 11 flips the global default ON. Toggle always restored.
function chain_parity_case(f, args; check_engine_backward=true)
    old = TeneT.CHAIN_ENGINE[]
    ref, gref = try
        TeneT.set_chain_engine!(false)
        r = f(args...)
        (r, Zygote.gradient((a...) -> sum(abs2, f(a...)), args...))
    finally
        TeneT.set_chain_engine!(old)
    end
    geng = try
        TeneT.set_chain_engine!(true)
        @test f(args...) ≈ ref rtol = 1e-12
        Zygote.gradient((a...) -> sum(abs2, f(a...)), args...)
    finally
        TeneT.set_chain_engine!(old)
    end
    grads_match(geng, gref; label="zygote on/off")
    if check_engine_backward
        dOut = rand(ComplexF64, size(ref))
        # gz must REMAIN a toggle-OFF @tensor-path pullback regardless of the
        # ambient default: engine_backward is validated against pure Zygote.
        gz = try
            TeneT.set_chain_engine!(false)
            _, bk = Zygote.pullback((a...) -> f(a...), args...)
            bk(dOut)
        finally
            TeneT.set_chain_engine!(old)
        end
        ge = TeneT.engine_backward(f, args, dOut)
        @test ge !== nothing
        grads_match(ge, gz; label="engine_backward")
    end
    return nothing
end

# Engine-ON inner_etype result must match the toggle-OFF kernel's cast path.
function inner_etype_survives(f, args; rtol=1e-5)
    old = TeneT.CHAIN_ENGINE[]
    r32 = try
        TeneT.set_chain_engine!(true)
        f(args...; inner_etype=Float32)
    finally
        TeneT.set_chain_engine!(old)
    end
    ref = try
        TeneT.set_chain_engine!(false)
        f(args...; inner_etype=Float32)
    finally
        TeneT.set_chain_engine!(old)
    end
    @test r32 ≈ ref rtol = rtol
end

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
    # FRmap leg5 — helper output consistent with the constructed chain's first
    # three inter layouts (chain_interlabels = (I₁, I₂, I₃, out) for N=5):
    @test tensor_pinned_inters(((:i,:j,:k,:l), (:d,:g,:h,:l), (:e,:j,:g,:b,:p), (:f,:k,:h,:c,:p), (:a,:b,:c,:d)), (:a,:e,:f,:i)) ==
          chain_interlabels(TeneT.FRMAP_LEG5_CHAIN)[1:3]
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
        @testset "$name" begin chain_parity_case(TeneT.FLmap, args) end
    end
    # inner_etype path survives the reroute (mirrors test_contraction.jl):
    inner_etype_survives(TeneT.FLmap, (FL5, ALu5, ALd5, M5))
end

@testset "FRmap chains: parity over all variants" begin
    Random.seed!(42)
    χ, D, d = 8, 3, 2
    # Geometry mirrors test_contraction.jl:275-292:
    FR4  = rand(ComplexF64, χ, D, χ);    ARu4 = rand(ComplexF64, χ, D, χ)
    ARd4 = rand(ComplexF64, χ, D, χ);    M4   = rand(ComplexF64, D, D, D, D)
    FR5  = rand(ComplexF64, χ, D, D, χ); ARu5 = rand(ComplexF64, χ, D, D, χ)
    ARd5 = rand(ComplexF64, χ, D, D, χ); M5   = rand(ComplexF64, D, D, D, D, d)
    M8   = rand(ComplexF64, D, D, D, D, D, D, D, D)
    cases = [
        ((FR4, ARu4, ARd4, M4),        "leg4"),
        ((FR5, ARu5, ARd5, M5),        "leg5 single-M"),
        ((FR5, ARu5, ARd5, (M5, conj(M5))), "leg5 tuple"),
        ((FR5, ARu5, ARd5, M8),        "leg8"),
    ]
    for (args, name) in cases
        @testset "$name" begin chain_parity_case(TeneT.FRmap, args) end
    end
    # inner_etype path survives the reroute (mirrors test_contraction.jl):
    inner_etype_survives(TeneT.FRmap, (FR5, ARu5, ARd5, M5))
end

@testset "ACmap chains: parity over all variants" begin
    Random.seed!(43)
    χ, D, d = 8, 3, 2
    # Geometry mirrors test_contraction.jl's ACmap testsets (physical leg d=2):
    AC4 = rand(ComplexF64, χ, D, χ);    FL4 = rand(ComplexF64, χ, D, χ)
    FR4 = rand(ComplexF64, χ, D, χ);    M4  = rand(ComplexF64, D, D, D, D)
    AC5 = rand(ComplexF64, χ, D, D, χ); FL5 = rand(ComplexF64, χ, D, D, χ)
    FR5 = rand(ComplexF64, χ, D, D, χ); M5  = rand(ComplexF64, D, D, D, D, d)
    M8  = rand(ComplexF64, D, D, D, D, D, D, D, D)
    cases = [
        ((AC4, FL4, FR4, M4),        "leg4"),
        ((AC5, FL5, FR5, M5),        "leg5 single-M"),
        ((AC5, FL5, FR5, (M5, conj(M5))), "leg5 tuple"),
        ((AC5, FL5, FR5, M8),        "leg8"),
    ]
    for (args, name) in cases
        @testset "$name" begin chain_parity_case(TeneT.ACmap, args) end
    end
    # inner_etype path survives the reroute (mirrors test_contraction.jl):
    inner_etype_survives(TeneT.ACmap, (AC5, FL5, FR5, M5))
end

@testset "ACdmap chains: parity over all variants" begin
    Random.seed!(44)
    χ, D, d = 8, 3, 2
    # Geometry mirrors test_contraction.jl's ACdmap leg4/leg5 testsets
    # (physical leg d=2; no leg8 — ACdmap has no leg8 method):
    ACd4 = rand(ComplexF64, χ, D, χ);    FL4 = rand(ComplexF64, χ, D, χ)
    FR4  = rand(ComplexF64, χ, D, χ);    M4  = rand(ComplexF64, D, D, D, D)
    ACd5 = rand(ComplexF64, χ, D, D, χ); FL5 = rand(ComplexF64, χ, D, D, χ)
    FR5  = rand(ComplexF64, χ, D, D, χ); M5  = rand(ComplexF64, D, D, D, D, d)
    cases = [
        ((ACd4, FL4, FR4, M4),        "leg4"),
        ((ACd5, FL5, FR5, M5),        "leg5 single-M"),
        ((ACd5, FL5, FR5, (M5, conj(M5))), "leg5 tuple"),
    ]
    for (args, name) in cases
        @testset "$name" begin chain_parity_case(TeneT.ACdmap, args) end
    end
    # inner_etype path survives the reroute (mirrors test_contraction.jl):
    inner_etype_survives(TeneT.ACdmap, (ACd5, FL5, FR5, M5))
end

@testset "Cmap chains: parity (leg3/leg4 FL)" begin
    Random.seed!(45)
    χ, D = 8, 3
    # Geometry mirrors test_contraction.jl's "ACmap and Cmap" testset. Cmap
    # has no engine_backward entry (never goes through forloop/parallel) and
    # no inner_etype kwarg, so no inner_etype_survives call either.
    Cm  = rand(ComplexF64, χ, χ)
    FL3 = rand(ComplexF64, χ, D, χ);    FR3 = rand(ComplexF64, χ, D, χ)
    FL4 = rand(ComplexF64, χ, D, D, χ); FR4 = rand(ComplexF64, χ, D, D, χ)
    cases = [
        ((Cm, FL3, FR3), "leg3 FL"),
        ((Cm, FL4, FR4), "leg4 FL"),
    ]
    for (args, name) in cases
        @testset "$name" begin
            chain_parity_case(TeneT.Cmap, args; check_engine_backward=false)
        end
    end
end

println("test_chain_maps done")
