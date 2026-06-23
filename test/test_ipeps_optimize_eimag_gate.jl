using Test
using TeneT

@testset "GradientOptimize Eimag gate stage handoff" begin
    names = fieldnames(TeneT.GradientOptimize)
    @test :bond_checkpoint in names
    @test :imag_tol in names
    @test :opt_obs_maxiter in names
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

    model = Heisenberg(lattice=Square(), S=0.5, Jx=1.0, Jy=1.0, Jz=1.0,
                       ifrotate=false, couplingtype=:uniform, bondratio=1.0)
    boundary_alg = VUMPS{General}(; maxiter=7, verbosity=0)
    params_default = GradientOptimize(; model, pattern=[1;;], boundary_alg,
                                      optimizer=nothing, ifplot=false)
    @test params_default.opt_obs_maxiter == 21
    @test TeneT._observable_params(params_default).boundary_alg.maxiter == 21

    params_override = GradientOptimize(; model, pattern=[1;;], boundary_alg,
                                       optimizer=nothing, ifplot=false,
                                       opt_obs_maxiter=11)
    @test params_override.opt_obs_maxiter == 11
    @test TeneT._observable_params(params_override).boundary_alg.maxiter == 11
end
