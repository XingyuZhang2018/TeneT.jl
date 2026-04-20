# Instrumented vumps_step benchmark: manually unfold vumps_step and time each
# sub-step to locate the ~105 ms F32-specific overhead observed in mbvumps_380631.
#
# Runs C4v-style vumps_step (structurally identical to Plaquette but simpler
# setup — leftenv + ACenv + Cenv + QR is the same in both). Times:
#   (a) ALCtoAC_map
#   (b) leftenv_c4v  (FLmap_parallel inside)
#   (c) ACenv_c4v    (ACmap_parallel inside)
#   (d) Cenv_c4v     (Cmap — always F64)
#   (e) qrpos(AC)
#   (f) qrpos(C)
#   (g) QAC * QC' + reshape
#   (h) norm(RAC - RC)
#   Total
# Compares F64 vs F32 per step.

using Random, CUDA, MPI, TeneT, LinearAlgebra, Printf, Statistics
using TeneT: C4vVUMPSEnv, ALCtoAC_map, leftenv_c4v, ACenv_c4v, Cenv_c4v,
             qrpos, _to_front

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nprocs = MPI.Comm_size(comm)
CUDA.device!(0)

seed = 42; Random.seed!(seed)
D = parse(Int, get(ENV, "D", "10"))
chi = parse(Int, get(ENV, "CHI", "400"))
forloop_iter = parse(Int, get(ENV, "FORLOOP_ITER", "32"))
N = parse(Int, get(ENV, "N", "10"))
D_phys = 2

rank == 0 && @printf("=== vumps_step PROBED  nprocs=%d D=%d chi=%d forloop_iter=%d N=%d ===\n",
                     nprocs, D, chi, forloop_iter, N)

# Build random C4v-style env.
#   M: ComplexF64 leg5 (D, D, D, D, D_phys)
#   FL, AL: (chi, D, D, chi) ComplexF64
#   C: (chi, chi) ComplexF64
M  = CUDA.rand(ComplexF64, D, D, D, D, D_phys)
FL = CUDA.rand(ComplexF64, chi, D, D, chi)
AL = CUDA.rand(ComplexF64, chi, D, D, chi)
C  = CUDA.rand(ComplexF64, chi, chi)

function make_alg(prec; ifcheckpoint=false)
    VUMPS{TeneT.C4v}(
        ifsimple_eig=true, ifparallel=true, ifcheckpoint=ifcheckpoint,
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

# Probe one vumps_step and record per-step times.
function probed_vumps_step!(rt, M, alg)
    times = Float64[]
    push_t(t0) = push!(times, (time_ns() - t0) / 1e6)
    sync() = (CUDA.synchronize(); MPI.Barrier(comm))

    sync(); t0 = time_ns()
    AC = ALCtoAC_map(rt.AL, rt.C)
    sync(); push_t(t0)    # (a)

    t0 = time_ns()
    _, FL_new = leftenv_c4v(rt.AL, conj(rt.AL), M, rt.FL; alg)
    sync(); push_t(t0)    # (b)

    t0 = time_ns()
    _, AC_new = ACenv_c4v(AC, FL_new, M; alg)
    sync(); push_t(t0)    # (c)

    t0 = time_ns()
    _, C_new  = Cenv_c4v(rt.C, FL_new; alg)
    sync(); push_t(t0)    # (d)

    t0 = time_ns()
    QAC, RAC = qrpos(_to_front(AC_new))
    sync(); push_t(t0)    # (e)

    t0 = time_ns()
    QC, RC = qrpos(C_new)
    sync(); push_t(t0)    # (f)

    t0 = time_ns()
    AL_new = reshape(QAC*QC', size(AC_new))
    sync(); push_t(t0)    # (g)

    t0 = time_ns()
    err = norm(RAC - RC)
    sync(); push_t(t0)    # (h)

    return (C4vVUMPSEnv(AL_new, C_new, FL_new), err, times)
end

# Warmup (compile both precisions and checkpoint modes)
rank == 0 && println("Warmup...")
for ifck in [false, true]
    for prec in [nothing, Float32]
        rt = C4vVUMPSEnv(AL, C, FL)
        probed_vumps_step!(rt, M, make_alg(prec; ifcheckpoint=ifck))
    end
end

# Measure
step_names = ["ALCtoAC", "leftenv", "ACenv", "Cenv", "qr(AC)", "qr(C)", "Q*Q'+resh", "norm"]

function time_avg(prec, N; ifcheckpoint=false)
    alg = make_alg(prec; ifcheckpoint=ifcheckpoint)
    all_t = [Float64[] for _ in step_names]
    for _ in 1:N
        rt = C4vVUMPSEnv(AL, C, FL)
        _, _, ts = probed_vumps_step!(rt, M, alg)
        for (i, t) in enumerate(ts); push!(all_t[i], t); end
    end
    m(v) = median(length(v) > 3 ? v[3:end] : v)
    return map(m, all_t)
end

function print_table(title, t64, t32)
    @printf("\n===== %s =====\n", title)
    @printf("%-12s | %-10s %-10s %-10s %-10s\n",
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
end

for ifck in [false, true]
    rank == 0 && @printf("\nTiming F64 (ifcheckpoint=%s)...\n", ifck)
    t64 = time_avg(nothing, N; ifcheckpoint=ifck)
    rank == 0 && @printf("Timing F32 (ifcheckpoint=%s)...\n", ifck)
    t32 = time_avg(Float32, N; ifcheckpoint=ifck)
    if rank == 0
        print_table("ifcheckpoint=$ifck", t64, t32)
    end
end

MPI.Finalize()
