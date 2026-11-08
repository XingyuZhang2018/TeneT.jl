# Local real-physics validation of the distributed (slice2d) Plaquette energy + full opt loop.
# Run: julia --project=. test/run_local_slice2d_realphys.jl   (4 ranks, 2×2 grid, CPU)
#
# Phase 1 — converged-A energy: load J2=0.5 D2 χ23 No.10 (history.log converged E=-0.470392432614),
#   serial vs slice2d (2×2) leading_boundary + block ObsEnv + DISTRIBUTED energy_value. Both must
#   reproduce the history energy; slice2d must match serial to machine precision (real-physics M6).
#   (NB: -0.4704 is the Plaquette single-env value; -0.4713 was the General-updown estimate.)
# Phase 2 — random-init slice2d optimization: optimise_ipeps with grid set, random A, a few LBFGS
#   iters (distributed energy + the gather-measure observable fix). Confirms the full production
#   loop runs end-to-end (energy descends, observable computes M/ξ without crashing).
using TeneT, Random, LinearAlgebra, OptimKit, Zygote, MPI, Printf
using TeneT: J1J2, Square, VUMPS, Plaquette, GradientOptimize, init_ipeps, optimise_ipeps,
             slice2d_grid, build_A, initialize_env, ObsEnv, energy_value, leading_boundary
MPI.Init()
const comm   = MPI.COMM_WORLD
const rank   = MPI.Comm_rank(comm)
const nprocs = MPI.Comm_size(comm)
@assert nprocs == 4 "local_slice2d_realphys.jl expects 4 ranks (2×2 grid)"
const N = isqrt(nprocs)
const atype = Array
say(s) = (rank == 0 && (println(s); flush(stdout)))

const pattern = [1 3; 2 4]
const model = J1J2(lattice=Square(), S=0.5, J1=1.0, J2=0.5, ifrotate=true, couplingtype=:uniform, bondratio=1.0)

function restriction_ipeps(A)
    Ar = Zygote.Buffer(A)
    Ar[:, :, :, :, :, 1] = A[:, :, :, :, :, 1]
    Ar[:, :, :, :, :, 1] += permutedims(Ar[:, :, :, :, :, 1], (4, 3, 2, 1, 5))
    Ar[:, :, :, :, :, 2] = permutedims(Ar[:, :, :, :, :, 1], (1, 4, 3, 2, 5))
    Ar[:, :, :, :, :, 3] = permutedims(Ar[:, :, :, :, :, 1], (3, 2, 1, 4, 5))
    Ar[:, :, :, :, :, 4] = permutedims(Ar[:, :, :, :, :, 1], (3, 4, 1, 2, 5))
    return copy(Ar)
end

mkalg(grid) = VUMPS{Plaquette{Square}}(grid=grid, ifsimple_eig=true, ifparallel=false, forloop_iter=1,
                                       maxiter=50, miniter=0, maxiter_ad=4, miniter_ad=4,
                                       power_iter=2, power_iter_ad=5, power_iter_obs=20, show_every=100,
                                       tol=1e-10, verbosity=0,
                                       step_checkpoint=TeneT.Recompute())   # R1: collapse the AD-loop tape (slice2d-safe: only inner must be Plain)
mkparams(grid, folder; maxit=0) = GradientOptimize(model=model, pattern=pattern, boundary_alg=mkalg(grid),
    optimizer=LBFGS(20; maxiter=maxit, verbosity=(rank == 0 ? 2 : 0), gradtol=1e-7,
                    linesearch=HagerZhangLineSearch(maxfg=5)),
    folder=folder, verbosity=0, ifSU=false, ifprecondition=false, iter_precond=0,
    reuse_env=true, ifsave_env=false, ifload_env=false, ifsave_lbfgs=false, ifload_lbfgs=false, ifplot=false)

# ── Phase 1: real-physics energy reproduction (converged A) ──────────────────────────
const FOLDER = joinpath(pkgdir(TeneT), "data", "$model", "$pattern", "VUMPS_Plaquette", "Float64", "seed42")
function phys_energy(grid)
    params = mkparams(grid, FOLDER)
    A = init_ipeps(; atype, etype=Float64, No=10, D=2, χ=23, params)
    M = build_A(restriction_ipeps(A), params)
    rt = initialize_env(A, 2, 23, params; restriction_ipeps)
    rt, err = leading_boundary(rt, M, params.boundary_alg)
    env = ObsEnv(rt, M, params.boundary_alg, model)
    e, _ = energy_value(model, M, env, params)
    return real(e), err
end
say("=== Phase 1: converged A (J2=0.5 D2 χ23 No.10) energy — serial vs slice2d (2×2) ===")
es, errs = phys_energy(nothing)
say(@sprintf("  SERIAL Plaquette : energy = %.12f   (vumps err %.2e)", es, errs))
ec, errc = phys_energy(slice2d_grid(N, N))
say(@sprintf("  SLICE2D Plaquette : energy = %.12f   (vumps err %.2e)", ec, errc))
say(@sprintf("  Δ(serial,slice2d) = %.2e    history.log χ23 = -0.470392432614", abs(es - ec)))

# ── Phase 2: random-init slice2d optimization (distributed energy + observable fix) ────
say("=== Phase 2: random-init slice2d optimization (3 LBFGS iters, distributed energy + observable) ===")
Random.seed!(2024)   # identical RNG on every rank → replicated random A (slice2d needs A replicated)
const TMP = joinpath(pkgdir(TeneT), "data", "_local_slice2d_smoketest",
                     "$model", "$pattern", "VUMPS_Plaquette", "Float64", "seed2024")
pc = mkparams(slice2d_grid(N, N), TMP; maxit=3)
A0 = init_ipeps(; atype, etype=Float64, No=0, D=2, χ=16, params=pc)
Aopt, e_final, eg, fgnum, history = optimise_ipeps(A0, [16], pc; restriction_ipeps)
say(@sprintf("  Phase 2 done: final energy = %.10f  (random→3 iters; descending, not converged)", real(e_final)))
say("  history (energy per LBFGS iter): $(round.(real.(history), digits=8))")
say("=== local slice2d real-physics validation done ===")
