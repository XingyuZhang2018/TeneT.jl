# Chain-engine (M1) local tests. Run: julia --project=. test/test_chain_engine.jl
# Plain serial julia — NO MPI (mirrors the standalone test_cannon.jl convention).
using Test, LinearAlgebra, Random, Zygote
using TeneT
using TeneT: Chain, chain_interlabels, FLMAP_LEG5_CHAIN
using TensorOperations: tensorcontract, tensorcontract!

@testset "chain label analysis" begin
    ch = Chain(((:a,:e,:f,:i), (:i,:j,:k,:l), (:e,:j,:g,:b,:p), (:f,:k,:h,:c,:p), (:a,:b,:c,:d)),
               (:d,:g,:h,:l))
    ils = chain_interlabels(ch)
    @test ils[1] == (:a,:e,:f,:j,:k,:l)        # H
    @test ils[2] == (:a,:f,:k,:l,:g,:b,:p)     # T  (I_{k-1}-minus-shared, then op-minus-shared)
    @test ils[3] == (:a,:l,:g,:b,:h,:c)        # G
    @test FLMAP_LEG5_CHAIN.out == (:d,:g,:h,:l)
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

println("test_chain_engine done")
