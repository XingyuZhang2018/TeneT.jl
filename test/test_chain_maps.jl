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

@testset "Mmap/Mumap/Mdmap chains: parity" begin
    Random.seed!(46)
    χ, D, d = 8, 3, 2
    # Probe-pinned tree layouts (Task 8 Step 1 @macroexpand probe). Y (the
    # inner chains' out) is a B-side temp — its layout CANNOT be derived by
    # tensor_pinned_inters and is pinned to the probe values; the outer
    # chains' single inter must be the probe's X = AC*FR temp, and their
    # ops[3] must equal the inner chains' out (the composed-glue contract).
    @test TeneT.MUMAP_INNER_CHAIN.out == (:a,:b,:g,:l,:f,:k,:p)
    @test TeneT.MDMAP_INNER_CHAIN.out == (:a,:c,:h,:l,:e,:j,:p)
    @test chain_interlabels(TeneT.MUMAP_INNER_CHAIN)[1] == (:a,:f,:k,:l,:e,:j)
    @test chain_interlabels(TeneT.MDMAP_INNER_CHAIN)[1] == (:a,:e,:j,:l,:f,:k)
    @test chain_interlabels(TeneT.MUMAP_OUTER_CHAIN)[1] == (:c,:h,:a,:b,:g,:l)
    @test chain_interlabels(TeneT.MDMAP_OUTER_CHAIN)[1] == (:b,:g,:a,:c,:h,:l)
    @test TeneT.MUMAP_OUTER_CHAIN.ops[3] == TeneT.MUMAP_INNER_CHAIN.out
    @test TeneT.MDMAP_OUTER_CHAIN.ops[3] == TeneT.MDMAP_INNER_CHAIN.out

    # Mmap geometry mirrors test_contraction.jl's "Mmap" testset (all four
    # boundary tensors (χ,D,χ), output (D,D,D,D)); Mumap/Mdmap geometry from
    # the production call sites (precondition.jl): AC/ACd/FL/FR (χ,D,D,χ),
    # Mu/Md (D,D,D,D,d). None of the three has an engine_backward entry —
    # they reach production only through forloop_sum/parallel_sum, which
    # have NO rrule (forward-only); Zygote-gradability here comes from the
    # chain_apply rrule (Mmap) and the composed-glue rrules (Mumap/Mdmap).
    AC3 = rand(ComplexF64, χ, D, χ);    ACd3 = rand(ComplexF64, χ, D, χ)
    FL3 = rand(ComplexF64, χ, D, χ);    FR3  = rand(ComplexF64, χ, D, χ)
    AC5 = rand(ComplexF64, χ, D, D, χ); ACd5 = rand(ComplexF64, χ, D, D, χ)
    FL5 = rand(ComplexF64, χ, D, D, χ); FR5  = rand(ComplexF64, χ, D, D, χ)
    M5  = rand(ComplexF64, D, D, D, D, d)
    @testset "Mmap" begin
        chain_parity_case(TeneT.Mmap, (AC3, ACd3, FL3, FR3); check_engine_backward=false)
    end
    @testset "Mumap" begin
        chain_parity_case(TeneT.Mumap, (AC5, ACd5, FL5, FR5, M5); check_engine_backward=false)
    end
    @testset "Mdmap" begin
        chain_parity_case(TeneT.Mdmap, (AC5, ACd5, FL5, FR5, M5); check_engine_backward=false)
    end
end

@testset "corner maps LD/DR/RU/LU chains: parity" begin
    Random.seed!(47)
    χ, D, d = 6, 3, 2
    # Geometry from oc_Q_22_getQ_CBE (observable.jl:108-136): the corner
    # inputs L/D/R/U are all 4-leg (χ,D,D,χ) boundary tensors, M1/M2 are
    # leg5 (D,D,D,D,d), and only the OUTPUTS are 6-leg (χ,D,D,D,D,χ).
    # Label-derived sizes (basic.jl integer labels), e.g. LDmap: label 9
    # joins L leg 4 with D leg 1 (χ); labels 5,6 join L legs 2,3 with
    # M1/M2 legs 1 (D); labels 10,11 join D legs 2,3 with M1/M2 legs 2 (D);
    # label 13 joins M1/M2 legs 5 (d). These maps are dead in src (their
    # only caller is commented out) and never go through forloop/parallel —
    # NO engine_backward entries, gradability via the chain_apply rrule, so
    # check_engine_backward=false throughout. The tests ARE the spec.
    Lc = rand(ComplexF64, χ, D, D, χ); Dc = rand(ComplexF64, χ, D, D, χ)
    Rc = rand(ComplexF64, χ, D, D, χ); Uc = rand(ComplexF64, χ, D, D, χ)
    M1 = rand(ComplexF64, D, D, D, D, d)        # pair case: two INDEPENDENT M's
    M2 = rand(ComplexF64, D, D, D, D, d)

    # Chain declarations transcribed verbatim from the kernels' integer labels:
    @test TeneT.LDMAP_CHAIN.ops == ((1,5,6,9), (9,10,11,12), (5,10,7,2,13), (6,11,8,3,13))
    @test TeneT.LDMAP_CHAIN.out == (1,2,3,7,8,12)
    @test TeneT.DRMAP_CHAIN.ops == ((9,10,11,12), (4,7,8,12), (5,10,7,2,13), (6,11,8,3,13))
    @test TeneT.DRMAP_CHAIN.out == (9,5,6,2,3,4)
    @test TeneT.RUMAP_CHAIN.ops == ((1,2,3,4), (4,7,8,12), (5,10,7,2,13), (6,11,8,3,13))
    @test TeneT.RUMAP_CHAIN.out == (12,10,11,5,6,1)
    @test TeneT.LUMAP_CHAIN.ops == ((1,5,6,9), (1,2,3,4), (5,10,7,2,13), (6,11,8,3,13))
    @test TeneT.LUMAP_CHAIN.out == (9,10,11,7,8,4)
    for ch1m in (TeneT.LDMAP_CHAIN_1M, TeneT.DRMAP_CHAIN_1M,
                 TeneT.RUMAP_CHAIN_1M, TeneT.LUMAP_CHAIN_1M)
        @test ch1m.conjs == (false, false, false, true)
    end

    # Kernel-first geometry sanity (engine OFF — the @tensor originals):
    @test size(TeneT.LDmap(Lc, Dc, M1, M2)) == (χ, D, D, D, D, χ)
    @test size(TeneT.DRmap(Dc, Rc, M1, M2)) == (χ, D, D, D, D, χ)
    @test size(TeneT.RUmap(Rc, Uc, M1, M2)) == (χ, D, D, D, D, χ)
    @test size(TeneT.LUmap(Lc, Uc, M1, M2)) == (χ, D, D, D, D, χ)

    # RUmap's case args are (R, U, M1, M2) — the MAP signature; its chain
    # tensors are (U, R, M1, M2), the @tensor written order.
    for (f, A, B, name) in ((TeneT.LDmap, Lc, Dc, "LDmap"),
                            (TeneT.DRmap, Dc, Rc, "DRmap"),
                            (TeneT.RUmap, Rc, Uc, "RUmap"),
                            (TeneT.LUmap, Lc, Uc, "LUmap"))
        @testset "$name pair" begin
            chain_parity_case(f, (A, B, M1, M2); check_engine_backward=false)
        end
        @testset "$name single-M" begin
            chain_parity_case(f, (A, B, M1); check_engine_backward=false)
        end
        @testset "$name tuple" begin
            chain_parity_case(f, (A, B, (M1, conj(M1))); check_engine_backward=false)
        end
    end
end

@testset "reroute stays live on views (forloop/parallel slice forms)" begin
    Random.seed!(47)
    χ, D, d = 8, 3, 2
    # The forloop/parallel rrules hand engine_backward SubArray slices of
    # arg 3 (FLmap/ACmap: last dim; FRmap/ACdmap: dim 1 — the N_in
    # conventions of forloop_parallel_MPI.jl:543-632) plus a view of dOut
    # (out dim 4 tracks the split in all four leg5 cases). `nothing` here
    # would silently send production back to per-slice Zygote tapes.
    A  = rand(ComplexF64, χ, D, D, χ); B  = rand(ComplexF64, χ, D, D, χ)
    S  = rand(ComplexF64, χ, D, D, χ); M5 = rand(ComplexF64, D, D, D, D, d)
    dO = rand(ComplexF64, χ, D, D, χ)
    dOv = @view dO[:, :, :, 1:3]                 # sliced output cotangent
    cases = (
        (TeneT.FLmap,  (A, B, (@view S[:, :, :, 1:3]), M5)),  # ALd last-dim slice
        (TeneT.FRmap,  (A, B, (@view S[1:3, :, :, :]), M5)),  # ARd dim-1 slice
        (TeneT.ACmap,  (A, B, (@view S[:, :, :, 1:3]), M5)),  # FR last-dim slice
        (TeneT.ACdmap, (A, B, (@view S[1:3, :, :, :]), M5)),  # FR dim-1 slice
    )
    for (f, va) in cases
        @test TeneT.engine_backward(f, va, dOv) !== nothing
    end
end

@testset "forloop rrule reroute: engine == Zygote path" begin
    Random.seed!(51)
    χ, D, d = 8, 3, 2
    FL  = rand(ComplexF64, χ, D, D, χ); ALu = rand(ComplexF64, χ, D, D, χ)
    ALd = rand(ComplexF64, χ, D, D, χ); M5  = rand(ComplexF64, D, D, D, D, d)
    # All four wrappers share the (χ,D,D,χ)³ + M5 geometry, so the same three
    # boundary tensors serve every map; only the wrapper arg ORDER differs:
    #   FLmap_parallel(FL, ALu, ALd, M)   — splits arg 3 (ALd) on its LAST dim
    #   FRmap_parallel(FR, ARu, ARd, M)   — splits arg 3 (ARd) on dim 1
    #   ACmap_parallel(AC, FL, FR, M)     — splits arg 3 (FR)  on its LAST dim
    #   ACdmap_parallel(ACd, FL, FR, M)   — splits arg 3 (FR)  on dim 1
    # χ = 8 is uneven under forloop_iter = 3 (split_ranges → 3,3,2 chunks).
    wrappers = (TeneT.FLmap_parallel, TeneT.FRmap_parallel,
                TeneT.ACmap_parallel, TeneT.ACdmap_parallel)
    for fmap in wrappers, Mform in (M5, (M5, conj(M5))), n in (1, 3)
        loss(a, b, c, m) = sum(abs2, fmap(a, b, c, m; ifparallel=false, forloop_iter=n))
        old = TeneT.CHAIN_ENGINE[]
        gref = try
            TeneT.set_chain_engine!(false)
            Zygote.gradient(loss, FL, ALu, ALd, Mform)
        finally
            TeneT.set_chain_engine!(old)
        end
        geng = try
            TeneT.set_chain_engine!(true)
            Zygote.gradient(loss, FL, ALu, ALd, Mform)
        finally
            TeneT.set_chain_engine!(old)
        end
        mname = Mform isa Tuple ? "tuple" : "single-M"
        grads_match(geng, gref; label="$(nameof(fmap)) $mname n=$n")
    end

    # inner_etype (do_cast) branch — real production traffic (leftenv/rightenv/
    # ACenv thread it under Zygote); cast tolerance is F32-level. n=1 covers
    # the iter==1 engineback upcast path, n=3 the chunked-back upcast path:
    for n in (1, 3)
        loss32 = (fl, alu, ald, m) -> sum(abs2, TeneT.FLmap_parallel(fl, alu, ald, m;
            ifparallel=false, forloop_iter=n, inner_etype=Float32))
        old = TeneT.CHAIN_ENGINE[]
        g32ref = try
            TeneT.set_chain_engine!(false)
            Zygote.gradient(loss32, FL, ALu, ALd, M5)
        finally
            TeneT.set_chain_engine!(old)
        end
        g32eng = try
            TeneT.set_chain_engine!(true)
            Zygote.gradient(loss32, FL, ALu, ALd, M5)
        finally
            TeneT.set_chain_engine!(old)
        end
        grads_match(g32eng, g32ref; rtol=1e-5, label="inner_etype Float32 forloop_iter=$n")
    end
end

println("test_chain_maps done")
