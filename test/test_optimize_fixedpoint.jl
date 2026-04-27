# Tests for src/ipeps_optimize/optimize_fixedpoint.jl
# Built up incrementally per docs/plans/2026-04-27-ipeps-fixedpoint-mcf-plan.md.

using TeneT: iPEPSFixedPointConfig

@testset "iPEPSFixedPointConfig" begin
    cfg = iPEPSFixedPointConfig()
    @test cfg.env_mode == :A
    @test cfg.H_eff_mode == :a
    @test cfg.decompose_method == :X
    @test cfg.mcf_ifignore_gauge == false
    @test cfg.outer_maxiter == 200

    cfg2 = iPEPSFixedPointConfig(env_mode=:C, mcf_ifignore_gauge=true)
    @test cfg2.env_mode == :C
    @test cfg2.mcf_ifignore_gauge == true
end
