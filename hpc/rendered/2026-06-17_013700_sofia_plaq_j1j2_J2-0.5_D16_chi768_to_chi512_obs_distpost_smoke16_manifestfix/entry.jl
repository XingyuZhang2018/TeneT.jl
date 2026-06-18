using CUDA
using MPI
using TeneT
using Random
using LinearAlgebra
using OptimKit
using Zygote
using Printf
using TeneT: slice2d_grid

MPI.Init()
const COMM = MPI.COMM_WORLD
const RANK = MPI.Comm_rank(COMM)
const NPROCS = MPI.Comm_size(COMM)
const NGRID = isqrt(NPROCS)
@assert NGRID * NGRID == NPROCS "Slice2D needs a square process count; got nprocs=$NPROCS"

say(msg) = (RANK == 0 && (println(msg); flush(stdout)))
geti(k, default) = parse(Int, get(ENV, k, string(default)))
getf(k, default) = parse(Float64, get(ENV, k, string(default)))
getb(k, default) = lowercase(get(ENV, k, string(default))) in ("1", "true", "yes", "y")

const D = geti("D", 16)
const J2 = getf("J2", 0.5)
const CHI_LOAD = geti("CHI_LOAD", 768)
const NO_LOAD = geti("NO_LOAD", 10)
const CHI_OBS = geti("CHI_OBS", 512)
const SEED = geti("SEED", 42)
const VUMPS_MAXITER = geti("VUMPS_MAXITER", 30)
const SHOW_EVERY = geti("SHOW_EVERY", 1)
const POWER_ITER = geti("POWER_ITER", 5)
const POWER_ITER_OBS = geti("POWER_ITER_OBS", 40)
const FORLOOP_ITER = geti("FORLOOP_ITER", 64)
const ENV_TOL = getf("ENV_TOL", 1e-8)
const SAVE_ENV = getb("SAVE_ENV", false)
const DISTRIBUTED_QR = getb("DISTRIBUTED_QR", true)
const DATA_ROOT = get(ENV, "DATA_ROOT", joinpath(pkgdir(TeneT), "data"))
const OUTPUT_ROOT = get(ENV, "OUTPUT_ROOT", DATA_ROOT)

Random.seed!(SEED)
const atype = CuArray
const etype = Float64
const pattern = [1 3; 2 4]

model = J1J2(lattice=Square(), S=0.5, J1=1.0, J2=J2,
             ifrotate=true, couplingtype=:uniform, bondratio=1.0)
load_folder = joinpath(DATA_ROOT, "$model", "$pattern", "VUMPS_Plaquette", "Float64", "seed$SEED")
folder = joinpath(OUTPUT_ROOT, "$model", "$pattern", "VUMPS_Plaquette", "Float64", "seed$SEED")

boundary_alg = VUMPS{Plaquette{Square}}(
    ifsimple_eig=true,
    distributed_qr=DISTRIBUTED_QR,
    ifparallel=false,
    forloop_iter=FORLOOP_ITER,
    maxiter=VUMPS_MAXITER,
    miniter=0,
    maxiter_ad=0,
    miniter_ad=0,
    power_iter=POWER_ITER,
    power_iter_ad=5,
    power_iter_obs=POWER_ITER_OBS,
    show_every=SHOW_EVERY,
    tol=ENV_TOL,
    verbosity=(RANK == 0 ? 3 : 0),
)
boundary_alg.grid = slice2d_grid(NGRID, NGRID)

params = GradientOptimize(
    model=model,
    pattern=pattern,
    boundary_alg=boundary_alg,
    optimizer=LBFGS(200; maxiter=1, verbosity=0, gradtol=1e-7,
                    linesearch=HagerZhangLineSearch(maxfg=5)),
    forloop_iter=FORLOOP_ITER,
    maxiter_restart=1,
    verbosity=(RANK == 0 ? 4 : 0),
    folder=folder,
    ifSU=false,
    SUτ=0,
    ifprecondition=false,
    iter_precond=0,
    reuse_env=true,
    ifsave_env=SAVE_ENV,
    ifload_env=false,
    ifsave_lbfgs=false,
    ifload_lbfgs=false,
    save_every=0,
    ifplot=false,
)

function restriction_ipeps(A)
    Ar = Zygote.Buffer(A)
    Ar[:, :, :, :, :, 1] = A[:, :, :, :, :, 1]
    Ar[:, :, :, :, :, 1] += permutedims(Ar[:, :, :, :, :, 1], (4, 3, 2, 1, 5))
    Ar[:, :, :, :, :, 2] = permutedims(Ar[:, :, :, :, :, 1], (1, 4, 3, 2, 5))
    Ar[:, :, :, :, :, 3] = permutedims(Ar[:, :, :, :, :, 1], (3, 2, 1, 4, 5))
    Ar[:, :, :, :, :, 4] = permutedims(Ar[:, :, :, :, :, 1], (3, 4, 1, 2, 5))
    return copy(Ar)
end

function main()
    say("=== Slice2D Plaquette J1J2 Square obs distributed-post smoke ===")
    say("nprocs=$NPROCS grid=$(NGRID)x$(NGRID) D=$D J2=$J2 seed=$SEED")
    say("load chi=$CHI_LOAD No.$NO_LOAD -> obs chi=$CHI_OBS")
    say("distributed_qr=$DISTRIBUTED_QR maxiter=$VUMPS_MAXITER maxiter_ad=$(boundary_alg.maxiter_ad) miniter_ad=$(boundary_alg.miniter_ad)")
    say("show_every=$SHOW_EVERY power_iter=$POWER_ITER power_iter_obs=$POWER_ITER_OBS tol=$ENV_TOL forloop=$FORLOOP_ITER")
    say("save_env=$SAVE_ENV load_folder=$load_folder")
    say("output folder=$folder")
    RANK == 0 && CUDA.functional() && say("rank0 GPU=$(CUDA.name(CUDA.device())) CUDA_LAUNCH_BLOCKING=$(get(ENV, "CUDA_LAUNCH_BLOCKING", "unset"))")

    params_load = deepcopy(params)
    params_load.folder = load_folder
    A = init_ipeps(; atype, etype, No=NO_LOAD, D, χ=CHI_LOAD, params=params_load)
    MPI.Barrier(COMM)
    CUDA.synchronize()
    t0 = time()
    e, mag, xi = observable(A, CHI_OBS, params; restriction_ipeps)
    CUDA.synchronize()
    MPI.Barrier(COMM)
    say(@sprintf("=== obs distributed-post smoke done in %.1f sec ===", time() - t0))
    say("energy = $(e[1])")
    say("magnetization = $(mag[1])")
    say("correlation_length = $xi")
end

try
    main()
catch err
    RANK == 0 && (showerror(stdout, err, catch_backtrace()); println(); flush(stdout))
    MPI.Abort(COMM, 1)
finally
    MPI.Finalize()
end
