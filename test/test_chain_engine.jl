# Chain-engine (M1) local tests. Run: julia --project=. test/test_chain_engine.jl
# Plain serial julia — NO MPI (mirrors the standalone test_cannon.jl convention).
using Test, LinearAlgebra, Random, Zygote
using TeneT
using TeneT: Chain, chain_interlabels, FLMAP_LEG5_CHAIN
using TensorOperations: tensorcontract, tensorcontract!

@testset "chain label analysis" begin
    ops = ((:a,:e,:f,:i), (:i,:j,:k,:l), (:e,:j,:g,:b,:p), (:f,:k,:h,:c,:p), (:a,:b,:c,:d))
    out = (:d,:g,:h,:l)

    # un-pinned chain: derived left-assoc concat layouts
    ch = Chain(ops, out)
    ils = chain_interlabels(ch)
    @test ils[1] == (:a,:e,:f,:j,:k,:l)        # H
    @test ils[2] == (:a,:f,:k,:l,:g,:b,:p)     # T  (I_{k-1}-minus-shared, then op-minus-shared)
    @test ils[3] == (:a,:l,:g,:b,:h,:c)        # G
    @test ils[4] == out

    # FLMAP_LEG5_CHAIN pins the hand-kernel layouts (_cannon_stage1/_fold1/_fold2)
    pls = chain_interlabels(FLMAP_LEG5_CHAIN)
    @test pls[1] == (:a,:e,:f,:j,:k,:l)        # H (hand == derived)
    @test pls[2] == (:a,:f,:k,:g,:b,:p,:l)     # T (hand layout, ≠ derived)
    @test pls[3] == (:a,:b,:c,:g,:h,:l)        # G (hand layout, ≠ derived)
    @test pls[4] == out
    @test FLMAP_LEG5_CHAIN.out == out

    # constructor validation: each pinned tuple must be a set-permutation of
    # the derived label set at its position.
    bad_inters = ((:a,:e,:f,:j,:k,:l), (:a,:f,:k,:g,:b,:p,:q), (:a,:b,:c,:g,:h,:l))
    @test_throws AssertionError Chain(ops, out, bad_inters)
    @test_throws AssertionError Chain(ops, out, ((:a,:e,:f,:j,:k,:l),))  # wrong count
end

@testset "chain_apply == FLmap == hand pipeline" begin
    Random.seed!(11)
    χ, D, d = 12, 3, 2
    FL  = rand(ComplexF64, χ, D, D, χ)
    ALu = rand(ComplexF64, χ, D, D, χ)
    ALd = rand(ComplexF64, χ, D, D, χ)
    M1  = rand(ComplexF64, D, D, D, D, d)
    M2  = rand(ComplexF64, D, D, D, D, d)
    ref = TeneT.FLmap(FL, ALu, ALd, M1, M2)
    out = TeneT.chain_apply(FLMAP_LEG5_CHAIN, (FL, ALd, M1, M2, ALu))
    @test out ≈ ref rtol = 1e-12
end
@testset "chain_link1_add! accumulates the first link" begin
    Random.seed!(7)
    χ, D = 12, 3
    FL  = rand(ComplexF64, χ, D, D, χ)
    ALd = rand(ComplexF64, χ, D, D, χ)
    Href = TeneT._cannon_stage1(FL, ALd)
    # zero-init + two complementary i-block adds == full first link.
    # Engine H labels (:a,:e,:f,:j,:k,:l) equal the hand layout — direct compare.
    H2 = zero(Href)
    TeneT.chain_link1_add!(H2, FLMAP_LEG5_CHAIN, FL[:, :, :, 1:6], ALd[1:6, :, :, :])
    TeneT.chain_link1_add!(H2, FLMAP_LEG5_CHAIN, FL[:, :, :, 7:12], ALd[7:12, :, :, :])
    @test H2 ≈ Href rtol = 1e-12
end

@testset "_index2tuples toy validation" begin
    Random.seed!(8)
    # (1) transposed case: output labels permuted w.r.t. (openA..., openB...),
    #     so pAB is a genuine (non-identity) permutation.
    A = rand(ComplexF64, 4, 5)          # labels (:i, :a)
    B = rand(ComplexF64, 6, 4)          # labels (:b, :i)
    IA, IB, IC = (:i, :a), (:b, :i), (:b, :a)
    ref = tensorcontract(IC, A, IA, false, B, IB, false)
    pA, pB, pAB = TeneT._index2tuples(IA, IB, IC)
    C = zeros(ComplexF64, 6, 5)
    tensorcontract!(C, A, pA, false, B, pB, false, pAB, 1, 1)
    @test C ≈ ref rtol = 1e-14

    # (2) non-contiguous contracted positions: contracted labels sit at
    #     positions (1, 3) in A and (2, 4) in B; output also permuted.
    A2 = rand(ComplexF64, 3, 4, 5, 2)   # labels (:c1, :o1, :c2, :o2)
    B2 = rand(ComplexF64, 6, 3, 7, 5)   # labels (:o3, :c1, :o4, :c2)
    IA2, IB2, IC2 = (:c1, :o1, :c2, :o2), (:o3, :c1, :o4, :c2), (:o3, :o1, :o4, :o2)
    ref2 = tensorcontract(IC2, A2, IA2, false, B2, IB2, false)
    pA2, pB2, pAB2 = TeneT._index2tuples(IA2, IB2, IC2)
    C2 = zeros(ComplexF64, 6, 4, 7, 2)
    tensorcontract!(C2, A2, pA2, false, B2, pB2, false, pAB2, 1, 1)
    @test C2 ≈ ref2 rtol = 1e-14
end

@testset "chain_backward vs Zygote and hand adjoints" begin
    Random.seed!(13)
    χ, D, d = 12, 3, 2
    FL  = rand(ComplexF64, χ, D, D, χ)
    ALu = rand(ComplexF64, χ, D, D, χ)
    ALd = rand(ComplexF64, χ, D, D, χ)
    M1  = rand(ComplexF64, D, D, D, D, d)
    M2  = rand(ComplexF64, D, D, D, D, d)
    tensors = (FL, ALd, M1, M2, ALu)     # chain operand order
    out  = TeneT.chain_apply(FLMAP_LEG5_CHAIN, tensors)
    dOut = rand(ComplexF64, size(out))

    grads = TeneT.chain_backward(FLMAP_LEG5_CHAIN, tensors, dOut)
    @test grads isa Tuple && length(grads) == 5

    # (a) vs Zygote pullback of the serial FLmap — NOTE the operand-order
    #     difference: FLmap takes (FL, ALu, ALd, M1, M2).
    _, back = Zygote.pullback((fl, alu, ald, m1, m2) -> TeneT.FLmap(fl, alu, ald, m1, m2),
                              FL, ALu, ALd, M1, M2)
    dFL_z, dALu_z, dALd_z, dM1_z, dM2_z = back(dOut)
    @test grads[1] ≈ dFL_z  rtol = 1e-10   # FL
    @test grads[2] ≈ dALd_z rtol = 1e-10   # ALd
    @test grads[3] ≈ dM1_z  rtol = 1e-10   # M1
    @test grads[4] ≈ dM2_z  rtol = 1e-10   # M2
    @test grads[5] ≈ dALu_z rtol = 1e-10   # ALu

    # (b) vs the hand-adjoint chain, assembled exactly as the FLmap_cannon
    #     rrule walks it on a single full chunk.
    H = TeneT._cannon_stage1(FL, ALd)
    T = TeneT._cannon_fold1(H, M1)
    G = TeneT._cannon_fold2(T, M2)
    dG     = TeneT._cannon_stage2_dG(dOut, ALu)
    dALu_h = TeneT._cannon_stage2_dALu(dOut, G)
    dM2_h  = TeneT._cannon_fold2_dM2(dG, T)
    dT     = TeneT._cannon_fold2_dT(dG, M2)
    dM1_h  = TeneT._cannon_fold1_dM1(dT, H)
    dH     = TeneT._cannon_fold1_dH(dT, M1)
    dFL_h  = TeneT._cannon_stage1_dFL(dH, ALd)
    dALd_h = TeneT._cannon_stage1_dALd(dH, FL)
    @test grads[1] ≈ dFL_h  rtol = 1e-12
    @test grads[2] ≈ dALd_h rtol = 1e-12
    @test grads[3] ≈ dM1_h  rtol = 1e-12
    @test grads[4] ≈ dM2_h  rtol = 1e-12
    @test grads[5] ≈ dALu_h rtol = 1e-12

    # Caller-owned arrays must survive chain_backward: reuse tensors AND dOut
    # afterwards (on CUDA a freed array would error/corrupt here).
    out2 = TeneT.chain_apply(FLMAP_LEG5_CHAIN, tensors)
    @test out2 ≈ out rtol = 1e-12
    grads2 = TeneT.chain_backward(FLMAP_LEG5_CHAIN, tensors, dOut)
    @test all(g2 ≈ g1 for (g2, g1) in zip(grads2, grads))
end

@testset "chain engine reproduces the cannon local pipeline" begin
    # Single 2×2-grid rank's LOCAL workload (cf. _cannon_forward_sliced and the
    # FLmap_cannon rrule chunk body): a-block χ/2 row slices, full-i ALd column
    # slice, ring i-blocks delivered as views, l range processed in chunks.
    Random.seed!(17)
    χ, D, d = 12, 3, 2
    nl = χ ÷ 2                                    # local l extent
    FL_row  = rand(ComplexF64, χ ÷ 2, D, D, χ)    # a-block, FULL i (ring-assembled row)
    ALu_row = rand(ComplexF64, χ ÷ 2, D, D, χ)    # a-block, full d
    ALd_col = rand(ComplexF64, χ, D, D, nl)       # full i, l-block
    M1 = rand(ComplexF64, D, D, D, D, d)
    M2 = rand(ComplexF64, D, D, D, D, d)
    i_rs = TeneT.split_ranges(χ, 2)               # the N2 = 2 ring i-blocks

    # Unchunked engine run == the full local map (parity anchor for both dirs).
    full_engine = TeneT.chain_apply(FLMAP_LEG5_CHAIN, (FL_row, ALd_col, M1, M2, ALu_row))
    @test full_engine ≈ TeneT.FLmap(FL_row, ALu_row, ALd_col, M1, M2) rtol = 1e-12

    dpartial = rand(ComplexF64, size(full_engine))

    # Zygote reference for the backward: pullback of the full local map driven
    # by dpartial (NOTE FLmap operand order: FL, ALu, ALd, M1, M2).
    _, back = Zygote.pullback((fl, alu, ald, m1, m2) -> TeneT.FLmap(fl, alu, ald, m1, m2),
                              FL_row, ALu_row, ALd_col, M1, M2)
    dFL_z, dALu_z, dALd_z, dM1_z, dM2_z = back(dpartial)

    for n in (1, 3)
        l_chunks = TeneT.split_ranges(nl, n)

        # ── forward: per-chunk ring accumulate + tail links ────────────────
        partial_hand   = zeros(ComplexF64, size(full_engine))
        partial_engine = zeros(ComplexF64, size(full_engine))
        for ch in l_chunks
            # hand staged pipeline (the _cannon_forward_sliced chunk body)
            Hh = TeneT._cannon_stage1(view(FL_row, :, :, :, i_rs[1]),
                                      view(ALd_col, i_rs[1], :, :, ch))
            TeneT._cannon_stage1_add!(Hh, view(FL_row, :, :, :, i_rs[2]),
                                      view(ALd_col, i_rs[2], :, :, ch))
            G = TeneT._cannon_fold(Hh, M1, M2)
            P = TeneT._cannon_stage2(G, ALu_row)
            view(partial_hand, :, :, :, ch) .= P
            # engine: zero H (chunk dims from the hand kernel — H layouts are
            # equal, proven in the link1 testset), accumulate the i-blocks as
            # the ring would deliver them, then run links 2..N from H.
            H = zero(Hh)
            for t in 1:2
                TeneT.chain_link1_add!(H, FLMAP_LEG5_CHAIN,
                                       view(FL_row, :, :, :, i_rs[t]),
                                       view(ALd_col, i_rs[t], :, :, ch))
            end
            Pe = TeneT.chain_apply_from1(FLMAP_LEG5_CHAIN, H, (M1, M2, ALu_row))
            view(partial_engine, :, :, :, ch) .= Pe
        end
        @test partial_engine ≈ partial_hand rtol = 1e-12
        @test partial_engine ≈ full_engine rtol = 1e-12

        # ── backward: hand reference = the FLmap_cannon rrule chunk body ───
        dFL_h  = zeros(ComplexF64, size(FL_row))
        dALd_h = zeros(ComplexF64, size(ALd_col))
        dALu_h = zeros(ComplexF64, size(ALu_row))
        dM1_h  = zeros(ComplexF64, size(M1))
        dM2_h  = zeros(ComplexF64, size(M2))
        for ch in l_chunks
            Hc = TeneT._cannon_stage1(view(FL_row, :, :, :, i_rs[1]),
                                      view(ALd_col, i_rs[1], :, :, ch))
            TeneT._cannon_stage1_add!(Hc, view(FL_row, :, :, :, i_rs[2]),
                                      view(ALd_col, i_rs[2], :, :, ch))
            Tc = TeneT._cannon_fold1(Hc, M1)
            Gc = TeneT._cannon_fold2(Tc, M2)
            dPc = dpartial[:, :, :, ch]
            dGc = TeneT._cannon_stage2_dG(dPc, ALu_row)
            dALu_h .+= TeneT._cannon_stage2_dALu(dPc, Gc)
            dM2_h .+= TeneT._cannon_fold2_dM2(dGc, Tc)
            dTc = TeneT._cannon_fold2_dT(dGc, M2)
            dM1_h .+= TeneT._cannon_fold1_dM1(dTc, Hc)
            dHc = TeneT._cannon_fold1_dH(dTc, M1)
            for t in 1:2
                view(dFL_h, :, :, :, i_rs[t]) .+=
                    TeneT._cannon_stage1_dFL(dHc, view(ALd_col, i_rs[t], :, :, ch))
                view(dALd_h, i_rs[t], :, :, ch) .+=
                    TeneT._cannon_stage1_dALd(dHc, view(FL_row, :, :, :, i_rs[t]))
            end
        end

        # ── backward: engine path (NO _cannon_* kernels) ───────────────────
        dFL_e  = zeros(ComplexF64, size(FL_row))
        dALd_e = zeros(ComplexF64, size(ALd_col))
        dALu_e = zeros(ComplexF64, size(ALu_row))
        dM1_e  = zeros(ComplexF64, size(M1))
        dM2_e  = zeros(ComplexF64, size(M2))
        for ch in l_chunks
            H = zeros(ComplexF64, χ ÷ 2, D, D, D, D, length(ch))   # engine H labels (:a,:e,:f,:j,:k,:l)
            for t in 1:2
                TeneT.chain_link1_add!(H, FLMAP_LEG5_CHAIN,
                                       view(FL_row, :, :, :, i_rs[t]),
                                       view(ALd_col, i_rs[t], :, :, ch))
            end
            dPc = view(dpartial, :, :, :, ch)
            dH_c, dM1_c, dM2_c, dALu_c = TeneT.chain_backward_from1(
                FLMAP_LEG5_CHAIN, H, (M1, M2, ALu_row), dPc)
            dM1_e .+= dM1_c
            dM2_e .+= dM2_c
            dALu_e .+= dALu_c
            for t in 1:2
                dFL_t, dALd_t = TeneT.chain_link1_back(FLMAP_LEG5_CHAIN, dH_c,
                                    view(FL_row, :, :, :, i_rs[t]),
                                    view(ALd_col, i_rs[t], :, :, ch))
                view(dFL_e, :, :, :, i_rs[t]) .+= dFL_t
                view(dALd_e, i_rs[t], :, :, ch) .+= dALd_t
            end
        end

        @test dFL_e  ≈ dFL_h  rtol = 1e-12
        @test dALd_e ≈ dALd_h rtol = 1e-12
        @test dALu_e ≈ dALu_h rtol = 1e-12
        @test dM1_e  ≈ dM1_h  rtol = 1e-12
        @test dM2_e  ≈ dM2_h  rtol = 1e-12
        @test dFL_e  ≈ dFL_z  rtol = 1e-10
        @test dALd_e ≈ dALd_z rtol = 1e-10
        @test dALu_e ≈ dALu_z rtol = 1e-10
        @test dM1_e  ≈ dM1_z  rtol = 1e-10
        @test dM2_e  ≈ dM2_z  rtol = 1e-10
    end
end

println("test_chain_engine done")
