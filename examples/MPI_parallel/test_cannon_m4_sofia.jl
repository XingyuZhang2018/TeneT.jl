# Sofia GPU M4 Batch-A validation: distributed `leftenv_cannon` (env-level Cannon
# with gather hoisting) parity vs the SERIAL `leftenv`, on GPU, at multi-card scale.
# The NEW value over the 4-rank CPU gate (test/test_cannon_m4.jl, already green) is:
#   (1) the whole hoisted env path runs on CuArray + MPI-CUDA (gather wrappers +
#       reduce-scatter adjoints on-device — exercises the GPU-densify fix in the
#       cannon_gather_row/col rrules);
#   (2) cross-node (16-GPU 4×4 = 2 nodes) collectives over IB + the col_comm axis;
#   (3) device-memory headroom at the validation cell.
# Grid is AUTO-DERIVED from nprocs: 4 ranks → 2×2 (1 node), 16 ranks → 4×4 (2 nodes).
# Design: docs/2026-06-15-m4-env-cannon-integration-design.md (Batch A / Gate 5).
#
# Local CPU smoke (no GPU, tiny dims; confirms the driver parses + runs end-to-end
# under 4 ranks on plain Arrays before it reaches the cluster):
#   TENET_BENCH_CPU=1 julia --project=. examples/MPI_parallel/run_test_cannon_m4_sofia_cpu.jl

const BENCH_CPU = get(ENV, "TENET_BENCH_CPU", "0") == "1"

using MPI, Zygote, LinearAlgebra, Random, Printf
using TeneT
using TeneT: cannon_grid, CannonGrid, cannon_scatter, cannon_gather, split_ranges,
             leftenv, leftenv_cannon, rightenv, rightenv_cannon,
             ACenv, ACenv_cannon, Cenv, Cenv_cannon, VUMPS, General, StructArray
if !BENCH_CPU
    using CUDA
end

MPI.Init()
const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
const nprocs = MPI.Comm_size(comm)
const N = isqrt(nprocs)
@assert N * N == nprocs "test_cannon_m4_sofia.jl needs a square nprocs (4→2×2, 16→4×4); got $nprocs"

const CHI = parse(Int, get(ENV, "TENET_CANNON_CHI", "256"))
const D   = parse(Int, get(ENV, "TENET_CANNON_D",   "8"))
const d_phys = 2
const POWER = parse(Int, get(ENV, "TENET_CANNON_POWER", "40"))   # forward convergence
const POWER_AD = 4                                               # small tape for grad parity

report(s...) = rank == 0 && println(s...)

# ── device/array shim (identical harness on CPU and GPU) ─────────────────────
if BENCH_CPU
    _arr(x) = x
    _sync() = nothing
    _reclaim() = nothing
    mem_used_gb() = 0.0
    _devname() = "CPU"
else
    _arr(x) = CuArray(x)
    _sync() = CUDA.synchronize()
    _reclaim() = CUDA.reclaim()
    mem_used_gb() = (CUDA.total_memory() - CUDA.available_memory()) / 2^30
    _devname() = CUDA.name(CUDA.device())
end

function mem_line(label)
    m = mem_used_gb()
    mmax = MPI.Allreduce(m, MPI.MAX, comm)
    rank == 0 && @printf("[mem]  %-40s rank0 %6.2f GB   max %6.2f GB\n", label, m, mmax)
    return mmax
end

relerr(a, b) = norm(a - b) / max(norm(b), eps())
allreduce_max(x) = MPI.Allreduce(Float64(x), MPI.MAX, comm)
const FWD_RTOL  = 1e-10
const GRAD_RTOL = 1e-8

# Identical full tensors on every rank (fixed seed), as TeneT StructArrays.
function build_cell(Ni, Nj, χ, Dc; seed)
    Random.seed!(seed)
    pat = reshape(collect(1:Ni*Nj), Ni, Nj)
    nu = Ni * Nj
    ALu = StructArray([_arr(rand(ComplexF64, χ, Dc, Dc, χ)) for _ in 1:nu], pat)
    ALd = StructArray([_arr(rand(ComplexF64, χ, Dc, Dc, χ)) for _ in 1:nu], pat)
    M   = StructArray([_arr(rand(ComplexF64, Dc, Dc, Dc, Dc, d_phys)) for _ in 1:nu], pat)
    FL  = StructArray([_arr(rand(ComplexF64, χ, Dc, Dc, χ)) for _ in 1:nu], pat)
    return ALu, ALd, M, FL
end
scatter_sa(SA, g) = StructArray([cannon_scatter(t, g) for t in SA.data], SA.pattern)
gather_sa(SA, g)  = StructArray([cannon_gather(t, g) for t in SA.data], SA.pattern)

# global-phase-insensitive eigenvector rel err. GPU-safe: the phase is the global
# overlap coefficient ph = ⟨b,a⟩/⟨b,b⟩ (dot + broadcast — NO scalar indexing, which a
# CuArray disallows; the earlier argmax+a[imax] crashed on GPU). If a ∥ b up to a
# complex scalar c, then ph=c and a-b·ph=0; otherwise relerr captures the residual.
function phase_relerr(a, b)
    ph = dot(b, a) / dot(b, b)
    return relerr(a, b .* ph)
end

# leg2 C cell for Cenv (C is the replicated χ×χ center matrix)
function build_C_cell(Ni, Nj, χ, Dc; seed)
    Random.seed!(seed)
    pat = reshape(collect(1:Ni*Nj), Ni, Nj); nu = Ni * Nj
    C  = StructArray([_arr(rand(ComplexF64, χ, χ)) for _ in 1:nu], pat)
    FL = StructArray([_arr(rand(ComplexF64, χ, Dc, Dc, χ)) for _ in 1:nu], pat)
    FR = StructArray([_arr(rand(ComplexF64, χ, Dc, Dc, χ)) for _ in 1:nu], pat)
    return C, FL, FR
end
algf() = VUMPS(General(); ifsimple_eig=true, power_iter=POWER, forloop_iter=1, verbosity=0)
algad() = VUMPS(General(); ifsimple_eig=true, power_iter=POWER_AD, forloop_iter=1, verbosity=0)

# forward eigenpair parity: gather (or compare directly if replicated) up to phase.
function fwd_check(name, λref, eref, λc, ec, χ, Dc, g; gather)
    _sync()
    ef = gather ? gather_sa(ec, g) : ec
    eλ = maximum(abs(λc.data[i] - λref.data[i]) / max(abs(λref.data[i]), eps()) for i in 1:length(λref.data))
    ev = maximum(phase_relerr(ef.data[i], eref.data[i]) for i in 1:length(eref.data))
    eλ = allreduce_max(eλ); ev = allreduce_max(ev)
    report(@sprintf("[fwd %-6s] χ=%d D=%d  λ rel=%.2e  env rel=%.2e", name, χ, Dc, eλ, ev))
    return eλ <= FWD_RTOL && ev <= FWD_RTOL
end

# gradient parity: per-rank weighted-sum loss; `blockmask[k]` true → compare block-of,
# false → compare full (replicated, e.g. dC). gref/gc are 3-tuples over the env inputs.
function grad_check(name, gref, gc, blockmask, χ, g)
    rs = split_ranges(χ, g.N1)
    blkof(t) = t[rs[g.r1+1], :, :, rs[g.r2+1]]
    e = 0.0
    for k in 1:3
        ref = blockmask[k] ? blkof(gref[k].data[1]) : gref[k].data[1]
        e = max(e, relerr(gc[k].data[1], ref))
    end
    e = allreduce_max(e)
    report(@sprintf("[grad %-5s] χ=%d  grad max rel=%.2e", name, χ, e))
    return e <= GRAD_RTOL
end

# ── per-env forward (1×1) ─────────────────────────────────────────────────────
function val_left_fwd(χ, Dc, g)
    ALu, ALd, M, FL = build_cell(1, 1, χ, Dc; seed=4001); alg = algf()
    λr, er = leftenv(ALu, ALd, M, FL; alg)
    λc, ec = leftenv_cannon(scatter_sa(ALu, g), scatter_sa(ALd, g), M, scatter_sa(FL, g), g; alg)
    fwd_check("left", λr, er, λc, ec, χ, Dc, g; gather=true)
end
function val_right_fwd(χ, Dc, g)
    ARu, ARd, M, FR = build_cell(1, 1, χ, Dc; seed=5001); alg = algf()
    λr, er = rightenv(ARu, ARd, M, FR; alg)
    λc, ec = rightenv_cannon(scatter_sa(ARu, g), scatter_sa(ARd, g), M, scatter_sa(FR, g), g; alg)
    fwd_check("right", λr, er, λc, ec, χ, Dc, g; gather=true)
end
function val_ac_fwd(χ, Dc, g)
    AC, FL, M, FR = build_cell(1, 1, χ, Dc; seed=6001); alg = algf()
    λr, er = ACenv(AC, FL, M, FR; alg)
    λc, ec = ACenv_cannon(scatter_sa(AC, g), scatter_sa(FL, g), M, scatter_sa(FR, g), g; alg)
    fwd_check("AC", λr, er, λc, ec, χ, Dc, g; gather=true)
end
function val_c_fwd(χ, Dc, g)
    C, FL, FR = build_C_cell(1, 1, χ, Dc; seed=7001); alg = algf()
    λr, er = Cenv(C, FL, FR; alg)
    λc, ec = Cenv_cannon(C, scatter_sa(FL, g), scatter_sa(FR, g), g; alg)   # C replicated
    fwd_check("C", λr, er, λc, ec, χ, Dc, g; gather=false)                  # no gather
end

# ── per-env gradient (1×1) ────────────────────────────────────────────────────
function val_left_grad(χ, Dc, g)
    ALu, ALd, M, FL = build_cell(1, 1, χ, Dc; seed=9100)
    Random.seed!(9101); W = _arr(rand(ComplexF64, χ, Dc, Dc, χ)); Wb = cannon_scatter(W, g)
    alg = algad()
    lr(au, ad, fl) = real(sum(conj(W)  .* leftenv(au, ad, M, fl; alg)[2].data[1]))
    lc(au, ad, fl) = real(sum(conj(Wb) .* leftenv_cannon(au, ad, M, fl, g; alg)[2].data[1]))
    gr = Zygote.gradient(lr, ALu, ALd, FL)
    gc = Zygote.gradient(lc, scatter_sa(ALu, g), scatter_sa(ALd, g), scatter_sa(FL, g))
    grad_check("left", gr, gc, (true, true, true), χ, g)
end
function val_right_grad(χ, Dc, g)
    ARu, ARd, M, FR = build_cell(1, 1, χ, Dc; seed=9200)
    Random.seed!(9201); W = _arr(rand(ComplexF64, χ, Dc, Dc, χ)); Wb = cannon_scatter(W, g)
    alg = algad()
    lr(au, ad, fr) = real(sum(conj(W)  .* rightenv(au, ad, M, fr; alg)[2].data[1]))
    lc(au, ad, fr) = real(sum(conj(Wb) .* rightenv_cannon(au, ad, M, fr, g; alg)[2].data[1]))
    gr = Zygote.gradient(lr, ARu, ARd, FR)
    gc = Zygote.gradient(lc, scatter_sa(ARu, g), scatter_sa(ARd, g), scatter_sa(FR, g))
    grad_check("right", gr, gc, (true, true, true), χ, g)
end
function val_ac_grad(χ, Dc, g)
    AC, FL, M, FR = build_cell(1, 1, χ, Dc; seed=9300)
    Random.seed!(9301); W = _arr(rand(ComplexF64, χ, Dc, Dc, χ)); Wb = cannon_scatter(W, g)
    alg = algad()
    lr(ac, fl, fr) = real(sum(conj(W)  .* ACenv(ac, fl, M, fr; alg)[2].data[1]))
    lc(ac, fl, fr) = real(sum(conj(Wb) .* ACenv_cannon(ac, fl, M, fr, g; alg)[2].data[1]))
    gr = Zygote.gradient(lr, AC, FL, FR)
    gc = Zygote.gradient(lc, scatter_sa(AC, g), scatter_sa(FL, g), scatter_sa(FR, g))
    grad_check("AC", gr, gc, (true, true, true), χ, g)
end
function val_c_grad(χ, Dc, g)
    C, FL, FR = build_C_cell(1, 1, χ, Dc; seed=9400)
    Random.seed!(9401); W = _arr(rand(ComplexF64, χ, χ))      # FULL (replicated output)
    alg = algad()
    lr(c, fl, fr) = real(sum(conj(W) .* Cenv(c, fl, fr; alg)[2].data[1]))
    lc(c, fl, fr) = real(sum(conj(W) .* Cenv_cannon(c, fl, fr, g; alg)[2].data[1]))
    gr = Zygote.gradient(lr, C, FL, FR)
    gc = Zygote.gradient(lc, C, scatter_sa(FL, g), scatter_sa(FR, g))
    grad_check("C", gr, gc, (false, true, true), χ, g)         # dC full, dFL/dFR block
end

function main()
    g = cannon_grid(N, N)
    χ, Dc = BENCH_CPU ? (8, 2) : (CHI, D)
    report("=== M4 A–D Sofia validation: ", N, "×", N, " grid (", nprocs, " ranks), ",
           _devname(), ", CLB=", get(ENV, "CUDA_LAUNCH_BLOCKING", "0"),
           ", power_iter=", POWER, "/", POWER_AD, " ===")
    report("# gate: forward eigenpair rel ≤ ", FWD_RTOL, ", gradient max rel (allreduced) ≤ ", GRAD_RTOL)
    mem_line("start")
    all_pass = true
    function rec(name, ok)
        all_pass &= ok
        report(@sprintf("=== RESULT %-12s : %s ===", name, ok ? "PASS" : "FAIL"))
    end

    rec("fwd left",  val_left_fwd(χ, Dc, g))
    rec("fwd right", val_right_fwd(χ, Dc, g))
    rec("fwd AC",    val_ac_fwd(χ, Dc, g))
    rec("fwd C",     val_c_fwd(χ, Dc, g))
    mem_line("after forward")
    rec("grad left",  val_left_grad(χ, Dc, g))
    rec("grad right", val_right_grad(χ, Dc, g))
    rec("grad AC",    val_ac_grad(χ, Dc, g))
    rec("grad C",     val_c_grad(χ, Dc, g))
    GC.gc(); _reclaim(); mem_line("end (after reclaim)")

    report("")
    report(all_pass ? "=== RESULT: PASS ===" : "=== RESULT: FAIL ===")
    return all_pass
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
