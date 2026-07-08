# timing/vumps_alternatives/test_gate_easy.jl
#
# Onsager gate for the counted B0-power baseline (easy beta = 0.43).
#
# NOTE on chi (measured 2026-07-08, seed=42, pi=10, max_outer=200):
# The gate was originally planned at chi=64, but the chi=64 VUMPS fixed point
# at beta=0.43 has a finite-chi truncation floor |f - f_Onsager| = 1.400e-9 —
# the STOCK example (examples/2D_Classical/Ising_Square_VUMPS_C4v.jl) itself
# prints |delta f| = 1.400e-09 and cancels at maxiter=200 with err = 9.173e-9.
# step_power was verified BIT-IDENTICAL to TeneT.vumps_step per outer step
# (200/200 steps), so no correct implementation can pass 1e-10 at chi=64.
# Truncation floor scaling: chi=32 -> 1.2e-7, 64 -> 1.400e-9, 96 -> 6.5e-11,
# 128 -> 7.3e-12.  The gate keeps the ORIGINAL thresholds (err < 1e-9,
# |df| < 1e-10) and runs at chi=128 (a design-doc study cell), where the
# physics floor sits 14x below the |df| threshold.
using Test
include(joinpath(@__DIR__, "algorithms.jl"))
@testset "B0-power reproduces Onsager at beta=0.43 chi=128" begin
    S = setup_ising(beta=BETA_EASY, chi=128)
    cnt = MapCounter(); traj = TrajectoryLog()
    rt, err = run_b0_power!(cnt, traj, S; pi=10, max_outer=200, tol=1e-10)
    f = diag_free_energy(cnt, rt.AL, rt.C, rt.FL, S.M, BETA_EASY)
    @test err < 1e-9
    @test abs(f - S.f_exact) < 1e-10
    @test cnt.fl > 0 && cnt.ac > 0 && cnt.c > 0
    @test length(traj.rows) > 2
end
println("GATE_EASY OK")
