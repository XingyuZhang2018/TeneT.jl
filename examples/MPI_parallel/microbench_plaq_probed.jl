# Probed Plaquette vumps_step: time leftenv/ACenv/Cenv separately.
# Reuses profile_fg.jl setup (same model/pattern) so measurements reflect
# production path. Each env internally loops over 2x2 pattern → 2 simple_eig
# + 2 inner-loop ACmap_parallel per env. This benchmark times env-level
# granularity (not the inner-loop individually) — since leftenv/ACenv each
# drive ~4 kernel calls in Plaquette, this is the level the F32 savings
# should manifest or disappear.

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
N = parse(Int, get(ENV, "N", "6"))

rank == 0 && @printf("=== Plaquette vumps_step PROBED  nprocs=%d D=%d chi=%d forloop_iter=%d N=%d ===\n",
                     nprocs, D, chi, forloop_iter, N)

# Match profile_fg.jl
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
        maxiter=1, miniter=0, maxiter_ad=0, miniter_ad=0,
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
M_struct = TeneT.build_A(Ar, params)

# Need runtime struct matching Plaquette — use init_env
rt0 = TeneT.init_env(M_struct, chi, make_alg(nothing))

sync() = (CUDA.synchronize(); MPI.Barrier(comm))

function probed_plaq_step(rt, M, alg)
    times = Float64[]
    push_t(t0) = push!(times, (time_ns() - t0) / 1e6)

    AL, C, FL = rt.AL, rt.C, rt.FL

    sync(); t0 = time_ns()
    AC = TeneT.ALCtoAC(AL, C)           # bulk map
    sync(); push_t(t0)   # (a) ALCtoAC

    t0 = time_ns()
    _, FL_new = TeneT.leftenv(AL, conj(AL), M, FL; alg)
    sync(); push_t(t0)   # (b) leftenv (general, with 2x2 inner-loop pattern)

    t0 = time_ns()
    _, AC_new = TeneT.ACenv_plaq(AC, FL_new, M; alg)
    sync(); push_t(t0)   # (c) ACenv_plaq

    t0 = time_ns()
    _, C_new = TeneT.Cenv_plaq(C, FL_new; alg)
    sync(); push_t(t0)   # (d) Cenv_plaq

    t0 = time_ns()
    _, err = TeneT.ACCtoAL(AC_new, C_new)
    sync(); push_t(t0)   # (e) ACCtoAL (QR + reconstruct)

    return times
end

# Warmup
rank == 0 && println("Warmup...")
probed_plaq_step(rt0, M_struct, make_alg(nothing))
probed_plaq_step(rt0, M_struct, make_alg(Float32))

step_names = ["ALCtoAC", "leftenv", "ACenv_plaq", "Cenv_plaq", "ACCtoAL"]

function time_avg(prec, N)
    alg = make_alg(prec)
    all_t = [Float64[] for _ in step_names]
    for _ in 1:N
        ts = probed_plaq_step(rt0, M_struct, alg)
        for (i, t) in enumerate(ts); push!(all_t[i], t); end
    end
    m(v) = median(length(v) > 2 ? v[2:end] : v)
    return map(m, all_t)
end

rank == 0 && println("Timing F64...")
t64 = time_avg(nothing, N)
rank == 0 && println("Timing F32...")
t32 = time_avg(Float32, N)

if rank == 0
    @printf("\n%-12s | %-10s %-10s %-10s %-10s\n",
            "step", "F64 ms", "F32 ms", "delta ms", "ratio")
    println("-"^60)
    local total64 = 0.0
    local total32 = 0.0
    for (i, name) in enumerate(step_names)
        Δ = t32[i] - t64[i]
        r = t32[i] / t64[i]
        total64 += t64[i]
        total32 += t32[i]
        @printf("%-12s | %-10.2f %-10.2f %+10.2f %-10.3f\n", name, t64[i], t32[i], Δ, r)
    end
    println("-"^60)
    @printf("%-12s | %-10.2f %-10.2f %+10.2f %-10.3f\n",
            "TOTAL", total64, total32, total32 - total64, total32/total64)
    println()

    # Each Plaquette env = 2 simple_eig + 2 inner-loop kernel calls = 4 heavy calls
    # If each call saved ~50ms in F32, each env should save ~200ms
    expected_env_saving = 4 * 50  # ms
    @printf("Expected F32 saving per env (4 calls × ~50ms): ~%d ms\n", expected_env_saving)
    @printf("Actual leftenv saving: %.1f ms (%.0f%% of expected)\n",
            t64[2] - t32[2], (t64[2] - t32[2]) / expected_env_saving * 100)
    @printf("Actual ACenv saving: %.1f ms (%.0f%% of expected)\n",
            t64[3] - t32[3], (t64[3] - t32[3]) / expected_env_saving * 100)
end

MPI.Finalize()
