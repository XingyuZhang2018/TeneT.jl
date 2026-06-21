# parallel() rrule reroute under MPI: engine vs Zygote gradient parity.
# Run via: julia --project=. test/run_test_parallel_engine.jl
# CPU Arrays only (no CUDA-aware MPI on local machines).
using Test
using MPI
using LinearAlgebra
using Random
using Zygote
using TeneT
using TeneT: FLmap_parallel, FRmap_parallel, set_chain_engine!, CHAIN_ENGINE

MPI.Init()
const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
@assert MPI.Comm_size(comm) == 4 "test_parallel_engine.jl expects exactly 4 ranks"

# Identical tensors on every rank: fixed seed immediately before each rand group.
function make_leg5(χ, D; d=2, seed=61)
    Random.seed!(seed)
    F  = rand(ComplexF64, χ, D, D, χ)
    Au = rand(ComplexF64, χ, D, D, χ)
    Ad = rand(ComplexF64, χ, D, D, χ)
    M  = rand(ComplexF64, D, D, D, D, d)
    return F, Au, Ad, M
end

# Engine-ON vs engine-OFF gradients of sum(abs2, fmap(...; ifparallel=true,
# forloop_iter=2)). After the rrule's collectives (allgatherv for a last-dim
# split arg, allreduce for everything else) the gradients are identical on
# every rank, so EVERY rank asserts the full set at 1e-10.
function engine_parity(fmap, args; rtol=1e-10, label="")
    loss(a...) = sum(abs2, fmap(a...; ifparallel=true, forloop_iter=2))
    old = CHAIN_ENGINE[]
    gref = try
        set_chain_engine!(false)
        Zygote.gradient(loss, args...)
    finally
        set_chain_engine!(old)
    end
    geng = try
        set_chain_engine!(true)
        Zygote.gradient(loss, args...)
    finally
        set_chain_engine!(old)
    end
    @testset "$label" begin
        @test length(geng) == length(gref)
        for (a, b) in zip(geng, gref)
            if b isa Tuple
                @test a isa Tuple
                for (ai, bi) in zip(a, b); @test ai ≈ bi rtol = rtol; end
            else
                @test a ≈ b rtol = rtol
            end
        end
    end
    return nothing
end

# FLmap_parallel splits arg 3 (ALd) on its LAST dim → N_in[2] == ndims ⇒
# has_split_gather = true: the split-arg gradient travels via allgatherv.
@testset "parallel rrule reroute: FLmap (allgatherv split path)" begin
    χ, D = 8, 3
    FL, ALu, ALd, M = make_leg5(χ, D; seed=61)
    engine_parity(FLmap_parallel, (FL, ALu, ALd, M); label="single-M")
    engine_parity(FLmap_parallel, (FL, ALu, ALd, (M, conj(M))); label="tuple-M")
    rank == 0 && println("rank 0: FLmap parallel reroute done")
end

# FRmap_parallel splits arg 3 (ARd) on dim 1 → has_split_gather = false: ALL
# gradients (split arg included) travel via allreduce.
@testset "parallel rrule reroute: FRmap (allreduce path)" begin
    χ, D = 8, 3
    FR, ARu, ARd, M = make_leg5(χ, D; seed=62)
    engine_parity(FRmap_parallel, (FR, ARu, ARd, M); label="single-M")
    engine_parity(FRmap_parallel, (FR, ARu, ARd, (M, conj(M))); label="tuple-M")
    rank == 0 && println("rank 0: FRmap parallel reroute done")
end

println("rank $rank: test_parallel_engine.jl done")
