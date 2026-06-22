using Test
using TeneT

@testset "GradientOptimize Eimag gate stage handoff" begin
    names = fieldnames(TeneT.GradientOptimize)
    @test :bond_checkpoint in names
    @test :imag_tol in names
    @test :last_stop_reason in names
    @test :last_stop_chi in names
    @test :last_stop_eimag in names

    @test !TeneT._imag_gate_hit(1e-10, 1e-8)
    @test TeneT._imag_gate_hit(1e-7, 1e-8)
    @test TeneT._imag_gate_hit(Inf, 1e-8)
    @test TeneT._imag_gate_hit(NaN, 1e-8)
end
