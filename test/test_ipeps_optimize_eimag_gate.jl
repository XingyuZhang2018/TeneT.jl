using Test
using TeneT

@testset "GradientOptimize Eimag gate stage handoff" begin
    names = fieldnames(TeneT.GradientOptimize)
    @test :bond_checkpoint in names
    @test :imag_tol in names
    @test :last_stop_reason in names
    @test :last_stop_χ in names
    @test :last_stop_chi ∉ names
    @test :last_stop_eimag in names
    @test :maxiter_restart ∉ names

    @test TeneT._normalize_χlist([16, 20, 24]) == [16, 20, 24]
    @test TeneT._normalize_χlist(16:4:24) == [16, 20, 24]
    @test_throws ArgumentError TeneT._normalize_χlist(Int[])
    @test_throws ArgumentError TeneT._normalize_χlist([16, 0])
    @test !isdefined(TeneT, :_normalize_chilist)
    @test hasmethod(TeneT.optimise_ipeps, Tuple{Any, Vector{Int}, TeneT.GradientOptimize})

    @test default_χlist(4; nstage=5) == [16, 24, 32, 48, 64]
    @test default_χlist(5; nstage=5) == [25, 32, 48, 64, 96]
    @test default_χlist(2; χmin=16, nstage=5) == [16, 24, 32, 48, 64]
    @test default_χlist(4; nstage=5, scheme=:sqrt2) == [16, 23, 32, 45, 64]
    @test default_χlist(5; nstage=4, maxχ=64) == [25, 32, 48, 64]
    @test !isdefined(TeneT, :default_chilist)
    @test_throws ArgumentError default_χlist(0)
    @test_throws ArgumentError default_χlist(4; nstage=0)
    @test_throws ArgumentError default_χlist(4; scheme=:unknown)

    @test !TeneT._imag_gate_hit(1e-10, 1e-8)
    @test TeneT._imag_gate_hit(1e-7, 1e-8)
    @test TeneT._imag_gate_hit(Inf, 1e-8)
    @test TeneT._imag_gate_hit(NaN, 1e-8)
end
