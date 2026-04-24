# Per-iteration fg benchmark for D=10 χ=400 J1-J2 Plaquette VUMPS.
#
# Measures Forward (`leading_boundary → energy`) and fg (forward+backward via
# `Zygote.pullback`) times for a single iteration after one warmup call each.
# Matches the timing methodology of `benchmarks/*_fg` tables — STEADY-STATE
# per-iteration numbers, not cold-started. The env is primed once between
# the cold and timed calls with `update!(rt, rt′)` so every timed call starts
# from a converged environment (what `optimise_ipeps`'s LBFGS `finalize!`
# does between iterations).
#
# Each system's `submit_bench.sh` (or equivalent) loops over nprocs = 1/2/4/8.

using Random, CUDA, TeneT, OptimKit, LinearAlgebra, Zygote, MPI, Printf

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nprocs = MPI.Comm_size(comm)
CUDA.device!(0)

rank == 0 && println("=" ^ 70)
rank == 0 && println("iPEPS fg Benchmark (D=10 χ=400 J1-J2 Plaquette VUMPS)")
rank == 0 && println("=" ^ 70)
rank == 0 && println("nprocs=$nprocs  device=$(CUDA.device())  hostname=$(gethostname())")
rank == 0 && println("GPU: $(CUDA.name(CUDA.device()))  Memory: $(round(CUDA.total_memory()/1024^3, digits=1)) GiB")
rank == 0 && println("CLB=", get(ENV, "CUDA_LAUNCH_BLOCKING", "<unset>"))
rank == 0 && println()

# ── Problem setup (mirror of MPI_parallel.jl) ──
seed = 42
Random.seed!(seed)
atype = CuArray
etype = Float64

D = 10
χ = 400
total_splits = 128
forloop_iter = total_splits ÷ nprocs

pattern = [1 3; 2 4]
model = J1J2(lattice=Square(), S=0.5, J1=1.0, J2=0.5,
             ifrotate=true, couplingtype=:uniform, bondratio=1.0)
folder = joinpath(pkgdir(TeneT), "data/bench_fg/")

boundary_alg = VUMPS{Plaquette{Square}}(
    ifsimple_eig     = true,
    ifparallel       = true,
    inner_checkpoint = Recompute(),
    step_checkpoint  = Recompute(),
    forloop_iter     = forloop_iter,
    maxiter          = 30,
    miniter          = 0,
    maxiter_ad       = 4,
    miniter_ad       = 4,
    power_iter       = 5,
    power_iter_ad    = 5,
    power_iter_obs   = 40,
    show_every       = 10,
    tol              = 1e-10,
    verbosity        = 0,
)

params = GradientOptimize(
    model           = model,
    pattern         = pattern,
    boundary_alg    = boundary_alg,
    optimizer       = LBFGS(200; maxiter=1, verbosity=0, gradtol=1e-7,
                            linesearch=HagerZhangLineSearch(maxfg=1)),
    forloop_iter    = forloop_iter,
    maxiter_restart = 1,
    verbosity       = 0,
    folder          = folder,
    ifSU            = false,
    SUτ             = 0,
    ifprecondition  = false,
    iter_precond    = 0,
    reuse_env       = true,
    ifsave_env      = false,
    ifload_env      = false,
    ifsave_lbfgs    = false,
    ifload_lbfgs    = false,
    ifplot          = false,
)

function restriction_ipeps(A)
    Ar = Zygote.Buffer(A)
    Ar[:,:,:,:,:,1]  = A[:,:,:,:,:,1]
    Ar[:,:,:,:,:,1] += permutedims(Ar[:,:,:,:,:,1], (4,3,2,1,5))
    Ar[:,:,:,:,:,2]  = permutedims(Ar[:,:,:,:,:,1], (1,4,3,2,5))
    Ar[:,:,:,:,:,3]  = permutedims(Ar[:,:,:,:,:,1], (3,2,1,4,5))
    Ar[:,:,:,:,:,4]  = permutedims(Ar[:,:,:,:,:,1], (3,4,1,2,5))
    return copy(Ar)
end

# ── Initialize ──
rank == 0 && println("--- Initializing iPEPS + environment ---")
A = init_ipeps(; atype, etype, No=0, D, χ, params)
rt  = TeneT.initialize_env(A, D, χ, params; restriction_ipeps)
rt′ = deepcopy(rt)
fδEierr = [1.0, 1.0, 0.0, 0.0]

function fenergy(A)
    TeneT._G_cache[] = nothing
    A = restriction_ipeps(A)
    return real(TeneT.energy(A, rt, rt′, fδEierr, params))
end

# `energy(A, rt, rt', ...)` internally writes the converged env into rt'.
# Copy rt' back into rt between calls so the NEXT call starts from converged
# environment (what JSC/BSC's full optimise_ipeps flow does between LBFGS steps).
reuse_env!() = TeneT.update!(rt, rt′)

# ── Warmup forward (cold start: 30 VUMPS iters + JIT compile) ──
rank == 0 && println("--- Warmup forward (cold) ---")
MPI.Barrier(comm)
t = time()
e = fenergy(A); reuse_env!()
CUDA.synchronize(); MPI.Barrier(comm)
rank == 0 && @printf("cold forward  : %8.2f s   e=%.10f\n", time() - t, e)

# ── Timed forward (warm: env reused, should converge in few iters) ──
MPI.Barrier(comm)
t = time()
e = fenergy(A); reuse_env!()
CUDA.synchronize(); MPI.Barrier(comm)
t_fwd = time() - t
rank == 0 && @printf("TIMED forward : %8.2f s   e=%.10f\n", t_fwd, e)

# ── Warmup fg (forward + backward via Zygote) ──
rank == 0 && println("\n--- Warmup fg (warm env) ---")
MPI.Barrier(comm)
t1 = time()
e, vjp = Zygote.pullback(fenergy, A)
CUDA.synchronize(); MPI.Barrier(comm)
t2 = time()
g = vjp(1)[1]
CUDA.synchronize(); MPI.Barrier(comm)
reuse_env!()
rank == 0 && @printf("warmup fg     : fwd=%8.2f s  bwd=%8.2f s  total=%8.2f s\n",
                     t2-t1, time()-t2, time()-t1)
e = nothing; vjp = nothing; g = nothing
GC.gc(true); CUDA.reclaim()
MPI.Barrier(comm)

# ── Timed fg ──
MPI.Barrier(comm)
t1 = time()
e, vjp = Zygote.pullback(fenergy, A)
CUDA.synchronize(); MPI.Barrier(comm)
t2 = time()
g = vjp(1)[1]
CUDA.synchronize(); MPI.Barrier(comm)
reuse_env!()
t_fwd_in_fg = t2 - t1
t_bwd = time() - t2
t_fg = time() - t1
rank == 0 && @printf("TIMED fg      : fwd=%8.2f s  bwd=%8.2f s  total=%8.2f s\n",
                     t_fwd_in_fg, t_bwd, t_fg)
rank == 0 && @printf("gnorm         : %.15e\n", norm(Array(g)))
rank == 0 && println()
rank == 0 && @printf("=== SUMMARY nprocs=%d: Forward=%.2fs  fg=%.2fs  gnorm=%.10e ===\n",
                     nprocs, t_fwd, t_fg, norm(Array(g)))

MPI.Finalize()
