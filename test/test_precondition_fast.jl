"""
    Test: precondition_fast vs precondition_invese_single_envir

Verify that the optimised preconditioner produces *exactly* the same output
as the original for the Heisenberg Square VUMPS General case (smallest D/χ).

Run with:
    julia --project test/test_precondition_fast.jl
"""

using TeneT
using Random
using OptimKit
using LinearAlgebra
using Zygote
using Test

# ── include the fast version into TeneT's namespace so it can access internals ──
Base.include(TeneT, joinpath(pkgdir(TeneT), "src", "ipeps_optimize", "precondition_fast.jl"))

# ── reproducible setup ──
seed = 42
Random.seed!(seed)
atype = Array
etype = ComplexF64
D, χ = 2, 8            # small for fast testing
pattern = [1;;]

model = Heisenberg(lattice=Square(),
                   S=0.5, Jx=-1.0, Jy=-1.0, Jz=1.0,
                   ifrotate=true,
                   couplingtype=:uniform, bondratio=1.0)

boundary_alg = VUMPS{:General}(ifupdown=true,
                               ifdownfromup=false,
                               ifsimple_eig=true,
                               ifparallel=false,
                               ifcheckpoint=false,
                               forloop_iter=1,
                               maxiter=30,
                               miniter=0,
                               maxiter_ad=4,
                               miniter_ad=4,
                               power_iter=1,
                               power_iter_ad=5,
                               power_iter_obs=40,
                               show_every=100,
                               tol=1e-10,
                               verbosity=0)

params = GradientOptimize(model=model,
                          pattern=pattern,
                          boundary_alg=boundary_alg,
                          optimizer=LBFGS(10; maxiter=5, verbosity=0, gradtol=1e-7,
                                         linesearch=HagerZhangLineSearch(maxfg=5)),
                          ifcheckpoint=false,
                          forloop_iter=1,
                          maxiter_restart=1,
                          verbosity=0,
                          folder=mktempdir(),
                          ifSU=false,
                          SUτ=0,
                          ifprecondition=true,
                          iter_precond=0,
                          reuse_env=true,
                          ifsave_env=false,
                          ifload_env=false,
                          ifsave_lbfgs=false,
                          ifload_lbfgs=false)

restriction_ipeps(A) = TeneT.C4v_restriction(A)

# ── initialise iPEPS and environment ──
A = TeneT.init_ipeps(;atype, etype, No=0, D, χ, params)
rt = TeneT.initialize_env(A, D, χ, params; restriction_ipeps)

# Build a fake gradient (same shape as A, random)
Random.seed!(123)
grad = randn(etype, size(A))

# fδEi: [energy, delta, iteration, imag]
# iteration > iter_precond so preconditioner is active
fδEi = [0.5, 0.1, 5.0, 0.0]

# ── run original preconditioner ──
println("Running original preconditioner...")
TeneT._G_cache[] = nothing
t1 = time()
result_orig = TeneT.precondition_invese_single_envir(
    A, grad, rt, params, restriction_ipeps, fδEi, params.iter_precond)
time_orig = time() - t1
println("  Original: $(round(time_orig, digits=2)) s")

# ── run fast preconditioner (precond_every=1, no caching) ──
println("Running fast preconditioner (precond_every=1)...")
TeneT._precond_cache[] = nothing
TeneT._G_cache[] = nothing
t2 = time()
result_fast = TeneT.precondition_fast(
    A, grad, rt, params, restriction_ipeps, fδEi, params.iter_precond; precond_every=1)
time_fast = time() - t2
println("  Fast:     $(round(time_fast, digits=2)) s")

# ── compare ──
diff = norm(result_orig - result_fast)
relnorm = diff / max(norm(result_orig), 1e-15)

println("\nResults:")
println("  ‖orig‖  = $(norm(result_orig))")
println("  ‖fast‖  = $(norm(result_fast))")
println("  ‖diff‖  = $diff")
println("  rel err = $relnorm")
println("  speedup = $(round(time_orig / time_fast, digits=2))×")

@testset "precondition_fast correctness" begin
    @test relnorm < 1e-10
end

# ── test lazy caching (precond_every=3) ──
println("\nTesting lazy caching (precond_every=3)...")
TeneT._precond_cache[] = nothing
TeneT._G_cache[] = nothing

# First call: should compute
fδEi_lazy = [0.5, 0.1, 5.0, 0.0]
r1 = TeneT.precondition_fast(
    A, grad, rt, params, restriction_ipeps, fδEi_lazy, params.iter_precond; precond_every=3)

# Second call at iter=6: should return cached
fδEi_lazy2 = [0.5, 0.1, 6.0, 0.0]
t3 = time()
r2 = TeneT.precondition_fast(
    A, grad, rt, params, restriction_ipeps, fδEi_lazy2, params.iter_precond; precond_every=3)
time_cached = time() - t3

# Third call at iter=8: should recompute (8 - 5 = 3 >= precond_every)
fδEi_lazy3 = [0.5, 0.1, 8.0, 0.0]
r3 = TeneT.precondition_fast(
    A, grad, rt, params, restriction_ipeps, fδEi_lazy3, params.iter_precond; precond_every=3)

@testset "precondition_fast lazy caching" begin
    @test r1 === r2
    @test time_cached < 0.1
    # r3 is recomputed so may differ (different fδEi[2]=δ could matter, but here δ is same)
    @test norm(r1 - r3) / max(norm(r1), 1e-15) < 1e-10
end

println("\nAll tests passed!")
