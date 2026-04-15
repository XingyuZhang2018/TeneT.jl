# MPI Multi-GPU Parallel iPEPS Optimization
#
# Use the provided submit.sh for Slurm clusters, which sets:
#   - CUDA_VISIBLE_DEVICES=$LOCAL_RANK  (isolates each process to its own GPU)
#   - UCX_MEMTYPE_CACHE=n               (avoids cudaMalloc interception crash)
#
# Key code settings for multi-GPU:
#   - ifparallel=true enables MPI-parallel tensor contractions
#   - ifcheckpoint=true enables AD checkpointing (required for large D/chi)
#   - forloop_iter=total_splits/nprocs splits work across GPUs

using Random, CUDA, TeneT, OptimKit, LinearAlgebra, Zygote, MPI

# ── MPI initialization ──────────────────────────────────────────────
MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nprocs = MPI.Comm_size(comm)
CUDA.device!(0)  # CUDA_VISIBLE_DEVICES already selects the right GPU per process

rank == 0 && @info "MPI initialized: nprocs=$nprocs, hostname=$(gethostname())"

# ── Verbosity: only rank 0 prints ───────────────────────────────────
if rank == 0
    verbosity_contract = 3
    verbosity_optim = 4
    ifsave_env = true
    ifsave_lbfgs = true
    ifplot = true
else
    verbosity_contract = 0
    verbosity_optim = 0
    ifsave_env = false
    ifsave_lbfgs = false
    ifplot = false
end

# ── Problem parameters ──────────────────────────────────────────────
seed = 42
Random.seed!(seed)
atype = CuArray
etype = Float64

D = 10                          # iPEPS bond dimension
χ = 400                         # boundary bond dimension
χshift = 16                     # chi increment per restart
maxiter_restart = 100           # number of restarts with increasing chi
total_splits = 128              # total forloop splits (fixed, independent of nprocs)
forloop_iter = total_splits ÷ nprocs  # each rank processes this many splits

rank == 0 && @info "D=$D chi=$chi forloop_iter=$forloop_iter total_splits=$total_splits"

# ── Model ────────────────────────────────────────────────────────────
pattern = [1 3; 2 4]
model = J1J2(lattice=Square(),
             S=0.5, J1=1.0, J2=0.5,
             ifrotate=true,
             couplingtype=:uniform, bondratio=1.0)
No = 0
folder = joinpath(pkgdir(TeneT), "data/$model/$pattern/VUMPS_Plaquette/$etype/D$D/seed$seed/")

# ── Algorithm ────────────────────────────────────────────────────────
boundary_alg = VUMPS{Plaquette{Square}}(
    ifsimple_eig  = true,
    ifparallel    = true,          # enable MPI parallel contractions
    ifcheckpoint  = true,          # required for large D/chi to fit in GPU memory
    forloop_iter  = forloop_iter,
    maxiter       = 30,
    miniter       = 0,
    maxiter_ad    = 4,
    miniter_ad    = 4,
    power_iter    = 5,             # power iterations for non-AD environment solve
    power_iter_ad = 5,             # power iterations for AD-tracked environment solve
    power_iter_obs = 40,           # power iterations for observation environment
    show_every    = 10,
    tol           = 1e-10,
    verbosity     = verbosity_contract,
)

params = GradientOptimize(
    model           = model,
    pattern         = pattern,
    boundary_alg    = boundary_alg,
    optimizer       = LBFGS(200; maxiter=10, verbosity=verbosity_optim, gradtol=1e-7,
                            linesearch=HagerZhangLineSearch(maxfg=5)),
    ifcheckpoint    = false,         # outer checkpoint for energy (separate from boundary_alg checkpoint)
    forloop_iter    = forloop_iter,  # must match boundary_alg setting
    maxiter_restart = maxiter_restart,
    verbosity       = verbosity_optim,
    folder          = folder,
    ifSU            = false,
    SUτ             = 0,
    ifprecondition  = true,
    iter_precond    = 0,
    reuse_env       = true,
    ifsave_env      = ifsave_env,
    ifload_env      = false,
    ifsave_lbfgs    = ifsave_lbfgs,
    ifload_lbfgs    = false,
    ifplot          = ifplot,
)

# ── iPEPS symmetry restriction (C4v for J1-J2 square) ───────────────
function restriction_ipeps(A)
    Ar = Zygote.Buffer(A)
    Ar[:,:,:,:,:,1] = A[:,:,:,:,:,1]
    Ar[:,:,:,:,:,1] += permutedims(Ar[:,:,:,:,:,1], (4,3,2,1,5))

    Ar[:,:,:,:,:,2] = permutedims(Ar[:,:,:,:,:,1], (1,4,3,2,5))
    Ar[:,:,:,:,:,3] = permutedims(Ar[:,:,:,:,:,1], (3,2,1,4,5))
    Ar[:,:,:,:,:,4] = permutedims(Ar[:,:,:,:,:,1], (3,4,1,2,5))

    return copy(Ar)
end

# ── Run optimization ────────────────────────────────────────────────
# Starts at chi, increases by chishift each restart up to maxiter_restart times
A = init_ipeps(; atype, etype, No, D, χ, params)
optimise_ipeps(A, χ, χshift, params; restriction_ipeps)

MPI.Finalize()
