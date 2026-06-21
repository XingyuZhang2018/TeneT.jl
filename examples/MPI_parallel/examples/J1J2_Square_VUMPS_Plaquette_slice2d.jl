# Slice2D (block-distributed χ) Plaquette J1J2 Square optimization — Sofia multi-GPU.
#
# Loads a converged D iPEPS at CHI_LOAD and optimizes at CHI_OPT on an N×N Slice2D grid
# (N = isqrt(nprocs); 16 ranks → 4×4). The whole leading_boundary + ObsEnv + energy_value +
# gradient run block-distributed: each GPU holds χ/N × χ/N blocks, so χ that OOMs the
# replicated forloop model (χ624 step-1 OOM on a 140 GB H200) becomes feasible.
#
# Parameters via ENV (submit script sets them):
#   D, J2, CHI_LOAD, NO_LOAD, CHI_OPT, OPT_MAXITER, POWER_ITER, VUMPS_MAXITER,
#   POWER_ITER_OBS, ENV_TOL, FORLOOP_ITER, DATA_ROOT, SEED
#   OPT_MAXITER=0  → SMOKE: env build + one forward + one (production) backward, no LBFGS step.
#
# Slice2D constraints (asserted in the solvers): square grid, ifsimple_eig=true, leg5 single
# layer, inner_checkpoint=Plain (NO ifcheckpoint=true), no mixed precision. ifprecondition is
# OFF — the preconditioner is not slice2d-ized (it gathers the full χ env → would OOM at χ768).
using TeneT, Random, LinearAlgebra, OptimKit, Zygote, MPI
using TeneT: slice2d_grid
using CUDA

MPI.Init()
const COMM   = MPI.COMM_WORLD
const RANK   = MPI.Comm_rank(COMM)
const NPROCS = MPI.Comm_size(COMM)
const NGRID  = isqrt(NPROCS)
@assert NGRID * NGRID == NPROCS "Slice2D needs a square process count; got nprocs=$NPROCS"

const atype = CuArray
geti(k, d) = parse(Int, get(ENV, k, string(d)))
getf(k, d) = parse(Float64, get(ENV, k, string(d)))
const D              = geti("D", 16)
const J2             = getf("J2", 0.5)
const CHI_LOAD       = geti("CHI_LOAD", 512)
const NO_LOAD        = geti("NO_LOAD", 50)
const CHI_OPT        = geti("CHI_OPT", 768)
const OPT_MAXITER    = geti("OPT_MAXITER", 0)        # 0 = smoke (init grad only, no step)
const POWER_ITER     = geti("POWER_ITER", 2)
const VUMPS_MAXITER  = geti("VUMPS_MAXITER", 10)
const POWER_ITER_OBS = geti("POWER_ITER_OBS", 20)
const ENV_TOL        = getf("ENV_TOL", 1e-6)
const FORLOOP_ITER   = geti("FORLOOP_ITER", 16)
const SEED           = geti("SEED", 42)
const DATA_ROOT      = get(ENV, "DATA_ROOT", joinpath(pkgdir(TeneT), "data"))
# R1 finding: the χ768 AD OOM is the un-checkpointed maxiter_ad-step AD-loop tape, NOT the seam
# gathers. step_checkpoint=Recompute collapses that tape (slice2d-safe: only inner_checkpoint must
# be Plain; step/subop are free). Exact (CPU-validated: identical gradient to Plain).
const STEP_CKPT      = get(ENV, "STEP_CKPT", "recompute")   # leading_boundary AD-loop tape lever
const BOND_CKPT      = get(ENV, "BOND_CKPT", "recompute")   # energy_value per-bond tape lever (production setting)
_ckpt(s) = s == "offload" ? TeneT.OffloadRecompute() : s == "plain" ? TeneT.Plain() : TeneT.Recompute()

Random.seed!(SEED)
const pattern = [1 3; 2 4]
say(s) = (RANK == 0 && (println(s); flush(stdout)))

say("=== Slice2D Plaquette J1J2 Square | nprocs=$NPROCS grid=$(NGRID)x$(NGRID) | D=$D J2=$J2 ===")
say("    load χ$CHI_LOAD No.$NO_LOAD → optimize χ$CHI_OPT | OPT_MAXITER=$OPT_MAXITER forloop=$FORLOOP_ITER power=$POWER_ITER vumps_maxiter=$VUMPS_MAXITER tol=$ENV_TOL")
RANK == 0 && CUDA.functional() && say("    rank0 GPU = $(CUDA.name(CUDA.device()))  CUDA_LAUNCH_BLOCKING=$(get(ENV,"CUDA_LAUNCH_BLOCKING","unset"))")

model = J1J2(lattice=Square(), S=0.5, J1=1.0, J2=J2,
             ifrotate=true, couplingtype=:uniform, bondratio=1.0)
folder = joinpath(DATA_ROOT, "$model", "$pattern", "VUMPS_Plaquette", "Float64", "seed$SEED")
say("    folder = $folder")

boundary_alg = VUMPS{Plaquette{Square}}(ifsimple_eig=true, ifparallel=false, forloop_iter=FORLOOP_ITER,
                                        maxiter=VUMPS_MAXITER, miniter=0, maxiter_ad=4, miniter_ad=4,
                                        power_iter=POWER_ITER, power_iter_ad=5, power_iter_obs=POWER_ITER_OBS,
                                        show_every=1, tol=ENV_TOL, step_checkpoint=_ckpt(STEP_CKPT),
                                        verbosity=(RANK == 0 ? 3 : 0))
boundary_alg.grid = slice2d_grid(NGRID, NGRID)
say("    step_checkpoint = $STEP_CKPT (AD-loop tape control)")

params = GradientOptimize(model=model, pattern=pattern, boundary_alg=boundary_alg,
                          optimizer=LBFGS(200; maxiter=OPT_MAXITER, verbosity=(RANK == 0 ? 4 : 0),
                                          gradtol=1e-7, linesearch=HagerZhangLineSearch(maxfg=5)),
                          maxiter_restart=1, folder=folder, verbosity=(RANK == 0 ? 4 : 0),
                          ifSU=false, ifprecondition=false, iter_precond=0,
                          bond_checkpoint=_ckpt(BOND_CKPT),
                          reuse_env=true, ifsave_env=false, ifload_env=false,
                          ifsave_lbfgs=false, ifload_lbfgs=false, ifplot=false)
say("    bond_checkpoint = $BOND_CKPT (energy_value per-bond tape control)")

A = init_ipeps(; atype, etype=Float64, No=NO_LOAD, D=D, χ=CHI_LOAD, params)

# Same C4v-like symmetrizer the χ512 reference was optimized with (site 1 independent +
# diagonal-symmetric; sites 2/3/4 are its permuted copies). Applied inside AD.
function restriction_ipeps(A)
    Ar = Zygote.Buffer(A)
    Ar[:, :, :, :, :, 1] = A[:, :, :, :, :, 1]
    Ar[:, :, :, :, :, 1] += permutedims(Ar[:, :, :, :, :, 1], (4, 3, 2, 1, 5))
    Ar[:, :, :, :, :, 2] = permutedims(Ar[:, :, :, :, :, 1], (1, 4, 3, 2, 5))
    Ar[:, :, :, :, :, 3] = permutedims(Ar[:, :, :, :, :, 1], (3, 2, 1, 4, 5))
    Ar[:, :, :, :, :, 4] = permutedims(Ar[:, :, :, :, :, 1], (3, 4, 1, 2, 5))
    Ar = copy(Ar)
    return Ar
end

if OPT_MAXITER == 0
    # SMOKE: one production forward + one production backward at χ_OPT, no LBFGS step and NO
    # observable() (magnetization_value / cor_len_value are NOT slice2d-ized yet — they'd run serial
    # ALCtoAC on a BLOCK AL → crash). This is exactly `energy()` (build_A → leading_boundary →
    # ObsEnv → distributed energy_value) differentiated once — the production fwd+bwd path. It
    # answers the smoke's questions: does χ_OPT fit memory on the grid, and does M6's distributed
    # energy + gradient run correctly on GPU.
    rt  = TeneT.initialize_env(A, D, CHI_OPT, params; restriction_ipeps)
    rt′ = deepcopy(rt)
    fδEierr = [1.0, 1.0, 0.0, 0.0]
    say("=== SMOKE: χ$CHI_OPT forward + production backward (no LBFGS step, no observable) ===")
    # Wall-clock split (rank 0). CUDA_LAUNCH_BLOCKING=1 already serializes kernels; the explicit
    # MPI.Barrier + CUDA.synchronize pin the timer to the collective-complete boundary so t_fwd /
    # t_bwd are the true forward(record) and backward(grad) wall times, not async-launch artifacts.
    MPI.Barrier(COMM); CUDA.synchronize()
    t0 = time()
    e, back = Zygote.pullback(A) do x
        real(TeneT.energy(restriction_ipeps(x), rt, rt′, fδEierr, params))
    end
    CUDA.synchronize(); t_fwd = time() - t0
    say("SMOKE forward : energy_χ$CHI_OPT = $e   (χ$CHI_LOAD reference ≈ -0.496682)   Eimag = $(fδEierr[4])   | forward(record) = $(round(t_fwd, digits=1)) sec")
    MPI.Barrier(COMM); t1 = time()
    g = back(one(e))[1]
    CUDA.synchronize(); t_bwd = time() - t1
    say("SMOKE backward: gradient computed, |g| = $(norm(g))   | backward = $(round(t_bwd, digits=1)) sec")
    say("=== SMOKE timing: forward(record) = $(round(t_fwd, digits=1)) sec | backward(grad) = $(round(t_bwd, digits=1)) sec ===")
    say("=== SMOKE done (rank 0) — χ$CHI_OPT slice2d forward+backward succeeded ===")
else
    # Full optimization. NOTE: observable() (post-LBFGS) is not slice2d-ready for mag/ξ; that must be
    # addressed (slice2d-ize magnetization_value/cor_len_value, or gather-measure) before relying on
    # the M/ξ outputs of a full run.
    optimise_ipeps(A, CHI_OPT, 0, params; restriction_ipeps)
    say("=== slice2d Plaquette full run finished (rank 0) ===")
end
