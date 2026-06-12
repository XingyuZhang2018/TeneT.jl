# Chain-engine (M1) local tests. Run: julia --project=. test/test_chain_engine.jl
# Plain serial julia — NO MPI (mirrors the standalone test_cannon.jl convention).
using Test, LinearAlgebra, Random, Zygote
using TeneT
using TeneT: Chain, chain_interlabels, FLMAP_LEG5_CHAIN

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
println("test_chain_engine done")
