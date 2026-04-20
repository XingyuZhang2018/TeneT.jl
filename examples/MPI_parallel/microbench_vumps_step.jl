# Micro-benchmark: isolate vumps_step timing at production config.
# Runs 4 MPI ranks, D=10 chi=400 forloop_iter=32.
# Times a single vumps_step (leftenv + ACenv + Cenv + QR + norm) in F64 vs F32,
# comparing against the isolated FLmap_parallel cost to see where the F32
# slowdown comes from.

using Random, CUDA, MPI, TeneT, LinearAlgebra, Zygote, Printf, Statistics

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nprocs = MPI.Comm_size(comm)
CUDA.device!(0)

seed = 42; Random.seed!(seed)
D = parse(Int, get(ENV, "D", "10"))
chi = parse(Int, get(ENV, "CHI", "400"))
forloop_iter = parse(Int, get(ENV, "FORLOOP_ITER", "32"))
N_call = parse(Int, get(ENV, "N", "10"))

rank == 0 && @printf("=== vumps_step micro-benchmark  nprocs=%d D=%d chi=%d forloop_iter=%d N=%d ===\n",
                     nprocs, D, chi, forloop_iter, N_call)

# Set up pattern and model, same as profile_fg.jl
pattern = [1 3; 2 4]
model = J1J2(lattice=Square(), S=0.5, J1=1.0, J2=0.5, ifrotate=true, couplingtype=:uniform, bondratio=1.0)

function restriction_ipeps(A)
    Ar = Zygote.Buffer(A)
    Ar[:,:,:,:,:,1] = A[:,:,:,:,:,1]
    Ar[:,:,:,:,:,1] += permutedims(Ar[:,:,:,:,:,1], (4,3,2,1,5))
    Ar[:,:,:,:,:,2] = permutedims(Ar[:,:,:,:,:,1], (1,4,3,2,5))
    Ar[:,:,:,:,:,3] = permutedims(Ar[:,:,:,:,:,1], (3,2,1,4,5))
    Ar[:,:,:,:,:,4] = permutedims(Ar[:,:,:,:,:,1], (3,4,1,2,5))
    return copy(Ar)
end

function make_alg(prec)
    VUMPS{TeneT.Plaquette{Square}}(
        ifsimple_eig=true, ifparallel=true, ifcheckpoint=true,
        forloop_iter=forloop_iter,
        maxiter=1, miniter=0, maxiter_ad=0, miniter_ad=0,  # ONE vumps_step only
        power_iter=1, power_iter_ad=5, power_iter_obs=40,
        show_every=10, tol=1e-10, verbosity=0,
        inner_etype=prec,
        inner_etype_final_steps=0,
        simple_eig_polish_steps=0,
        whole_vumps_etype=nothing,
        ifoffload_eig=false,
        ifoffload_step=false,
    )
end

params = GradientOptimize(
    model=model, pattern=pattern, boundary_alg=make_alg(nothing),
    optimizer=Zygote.gradient,
    ifcheckpoint=false, forloop_iter=forloop_iter, maxiter_restart=1,
    verbosity=0, folder="./_mb_tmp/", ifSU=false, SUτ=0,
    ifprecondition=false, iter_precond=0,
    reuse_env=false, ifsave_env=false, ifload_env=false,
    ifsave_lbfgs=false, ifload_lbfgs=false, ifplot=false,
)

A = TeneT.init_ipeps(; atype=CuArray, etype=Float64, No=0, D, χ=chi, params)
Ar = restriction_ipeps(A)
M = TeneT.build_A(Ar, params)

# Pre-allocate environment (fresh every run to avoid caching effects)
function run_leading_boundary(prec)
    alg = make_alg(prec)
    rt = TeneT.init_env(M, chi, alg)
    CUDA.synchronize(); MPI.Barrier(comm)
    t0 = time_ns()
    TeneT.leading_boundary(rt, M, alg)
    CUDA.synchronize(); MPI.Barrier(comm)
    return (time_ns() - t0) / 1e6  # ms
end

# Warmup
rank == 0 && println("Warmup...")
run_leading_boundary(nothing)
run_leading_boundary(Float32)

rank == 0 && println("Timing...")
ts_f64 = [run_leading_boundary(nothing) for _ in 1:N_call]
ts_f32 = [run_leading_boundary(Float32) for _ in 1:N_call]

m(v) = median(length(v) > 3 ? v[3:end] : v)

if rank == 0
    m64 = m(ts_f64); m32 = m(ts_f32)
    @printf("\nPer vumps_step (1 iter, leftenv+ACenv+Cenv+QR):\n")
    @printf("  F64:   %7.2f ms\n", m64)
    @printf("  F32:   %7.2f ms   ratio F32/F64: %.3f (%+.1f%%)\n",
            m32, m32/m64, 100*(m32/m64 - 1))

    # Compare to isolated FLmap_parallel (from mbmpi_380627.out):
    flmap_f64 = 210.0; flmap_f32 = 163.0
    @printf("\nReference (isolated FLmap_parallel MPI at same config):\n")
    @printf("  FLmap F64: %7.2f ms\n", flmap_f64)
    @printf("  FLmap F32: %7.2f ms (ratio %.3f)\n", flmap_f32, flmap_f32/flmap_f64)
    @printf("\nvumps_step / FLmap_parallel ratio:\n")
    @printf("  F64: %.2fx (implies ~%.1f FLmap-equivalent calls)\n", m64/flmap_f64, m64/flmap_f64)
    @printf("  F32: %.2fx (implies ~%.1f FLmap-equivalent calls)\n", m32/flmap_f32, m32/flmap_f32)
    @printf("\nIf vumps_step = 3 env calls + QR + norm, and each env ~= FLmap:\n")
    @printf("  Expected: ~3x FLmap time + small QR ~\n")
    @printf("  F64 overhead beyond 3x FLmap: %.2f ms\n", m64 - 3*flmap_f64)
    @printf("  F32 overhead beyond 3x FLmap: %.2f ms\n", m32 - 3*flmap_f32)
end

MPI.Finalize()
