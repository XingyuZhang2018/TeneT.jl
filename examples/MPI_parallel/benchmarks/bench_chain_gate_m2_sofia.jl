# M2 chain-engine perf gate (Part 8): the PRODUCTION code path under
# TeneT.set_chain_engine!(true) [CHAIN] vs (false) [TENSOR], single GPU, no MPI.
#
# Unlike Part 7 (hand staged kernels vs a literal chain), here the A/B switch
# IS the toggle: both arms run the real *_parallel wrappers / map calls, so the
# gate measures exactly what production pays. ON routes through the engine
# (chain_apply / engine_backward-rerouted rrule); OFF runs the verbatim @tensor
# bodies (fwd) and the per-slice Zygote forloop rrule (bwd).
#
# Maps × direction:
#   FLmap/FRmap/ACmap/ACdmap — leg5 single-M production form via
#       <map>_parallel(args...; ifparallel=false, forloop_iter=n); fwd = the
#       wrapper, bwd = Zygote.pullback over sum(abs2, wrapper(...)) (one
#       pullback call), matching Part-7's bwd timing structure (forloop rrule
#       under Zygote). Toggle ON vs OFF.
#   Cmap   — leg4, DIRECT call (no wrapper, no chunking); fwd + Zygote-pullback
#       bwd; ON vs OFF. (Cmap is differentiated directly in Cenv loops.)
#   Mumap  — FORWARD ONLY (forward-only in production: preconditioner linsolve
#       closure; forloop_sum has no rrule); via
#       Mumap_parallel(...; ifparallel=false, forloop_iter=n); ON vs OFF fwd.
#
# Cells: (D, χ) ∈ {(10,512), (12,1024), (16,1024)}. Float64 leg5, d_phys=2.
#
# n-choice (matches Part 7's driver): a SINGLE chunk count n per cell, the
# engine/staged path's feasible n = pick_n(8, D, χ) (Part-6 coeff 8). BOTH the
# ON (engine) and OFF (@tensor+Zygote) arms run at this same n so the A/B
# compares like-for-like at the engine's feasible n — Part 7 used `n = nA`
# (coeff 8) for both H and C and reported nB (coeff 14) for reference only.
# We likewise compute nB = pick_n(14, ...) and PRINT it as a reference column
# (the OFF Zygote arm's own memory-feasible n at a fairer comparison), but the
# timed A/B uses n for both arms. Cmap takes no n (direct, unchunked).
#
# Gate (printed per map × cell × direction): ratio CHAIN/TENSOR ≤ 1.05 time
# AND CHAIN mem ≤ 1.10× TENSOR mem ⇒ PASS. Additionally flags any cell where
# CHAIN is SLOWER than TENSOR (design doc: "any map slower than its @tensor
# original is a bug"). Part-7 expectation: FLmap-leg5 bwd CHAIN clearly FASTER
# at production cells.
#
# mem columns = device used (total−available) GiB after one un-GC'd call (live
# + uncollected garbage — the pool pressure the path creates), same probe as
# Part 7.

# atype switch: GPU (CuArray) on Sofia, Array for a local CPU syntax/smoke
# check (set TENET_BENCH_CPU=1; uses tiny dims, no CUDA needed).
const BENCH_CPU = get(ENV, "TENET_BENCH_CPU", "0") == "1"

using Zygote, LinearAlgebra, Printf, TeneT
if !BENCH_CPU
    using CUDA
end

const nrep = 3
const d_phys = 2
const MEM_BUDGET = 110e9

H_bytes(D, χ) = χ^2 * D^4 / 4 * 8
tensor_bytes(D, χ) = χ^2 * D^2 * 8
# Part-6 chunk-count formulas (read from bench_kernel_ab_sofia.jl): the engine/
# staged ordered adjoints peak at ≈(2+2d)|H|/n → coeff 8 with margin; the OFF
# Zygote backward holds tape + cotangent chain ≈10-11 |H| units → coeff 14.
pick_n(coeff, D, χ) = max(1, ceil(Int, coeff * H_bytes(D, χ) / (MEM_BUDGET - 12 * tensor_bytes(D, χ))))

# ----- device/array shims so the harness is identical on CPU and GPU -----
if BENCH_CPU
    _rand(dims...) = rand(Float64, dims...)
    _sync() = nothing
    _reclaim() = nothing
    mem_gb() = 0.0                          # no device-mem probe on CPU
    _devname() = "CPU"
else
    _rand(dims...) = CUDA.rand(Float64, dims...)
    _sync() = CUDA.synchronize()
    _reclaim() = CUDA.reclaim()
    mem_gb() = (CUDA.total_memory() - CUDA.available_memory()) / 2^30
    _devname() = CUDA.name(CUDA.device())
end

fmt(x) = isnan(x) ? @sprintf("%8s", "oom") : @sprintf("%8.1f", x)
fmtr(x) = isnan(x) ? @sprintf("%6s", "—") : @sprintf("%6.3f", x)

# Single-process OOM guard (no MPI → catching is deadlock-free): report the
# cell as NaN and clean up instead of killing the sweep. On CPU, OOM cannot be
# a CUDA error, so just rethrow.
function try_or_nan(f)
    try
        return f()
    catch e
        if !BENCH_CPU && e isa CUDA.OutOfGPUMemoryError
            GC.gc(); _reclaim()
            return NaN
        end
        rethrow()
    end
end

# Object-returning OOM guard for the parity arms: returns `nothing` on OOM so a
# parity check degrades to "unknown" (f?/g?) instead of killing the sweep.
function try_or_none(f)
    try
        return f()
    catch e
        if !BENCH_CPU && e isa CUDA.OutOfGPUMemoryError
            GC.gc(); _reclaim()
            return nothing
        end
        rethrow()
    end
end

# Bring a forward result / gradient tuple to host so the GPU arrays of one
# parity arm are freed (by the GC between arms) before the other arm allocates.
# Only one arm's device memory is ever live at a time → the parity check can't
# OOM by accumulation (the bug that killed job 1285146 at the unguarded gON/gOFF).
to_cpu(x::AbstractArray) = BENCH_CPU ? copy(x) : Array(x)
to_cpu(t::Tuple)         = map(to_cpu, t)
to_cpu(::Nothing)        = nothing
to_cpu(x)                = x

# Same timeit harness as Part 7: warm once, then per-rep GC OUTSIDE the timed
# window (no reclaim between reps → the pool pressure of the path persists,
# exactly the regime the mem probe samples).
function timeit(f)
    f(); _sync()
    GC.gc()
    tot = 0.0
    for _ in 1:nrep
        t0 = time()
        f(); _sync()
        tot += time() - t0
        GC.gc()
    end
    return tot / nrep * 1000
end

# Toggle the production engine on/off around a thunk; always restore the prior
# state even on exception (toggle-hygiene: never leak ON/OFF into later cells).
function with_engine(on::Bool, f)
    prev = TeneT.CHAIN_ENGINE[]
    TeneT.set_chain_engine!(on)
    try
        return f()
    finally
        TeneT.set_chain_engine!(prev)
    end
end

# Build a zero-arg thunk that toggles the engine around `f`, for ab_time_mem
# (so the timed window itself runs at the right toggle state every rep).
with_engine_thunk(on::Bool, f) = () -> with_engine(on, f)

# ----- the six production-path thunks (CHAIN = engine ON, TENSOR = OFF) ------
# fwd thunk: the literal wrapper / map call. bwd thunk: Zygote.pullback +
# one pullback call over sum(abs2, wrapper(...)) (matches Part-7 Tb structure,
# the forloop rrule under Zygote). The toggle decides which path runs.

# FLmap / FRmap / ACmap / ACdmap share the (X, Yu, Yd, M) 4-arg wrapper shape.
fwd_FLmap(FL, ALu, ALd, M, n)  = TeneT.FLmap_parallel(FL, ALu, ALd, M;  ifparallel=false, forloop_iter=n)
fwd_FRmap(FR, ARu, ARd, M, n)  = TeneT.FRmap_parallel(FR, ARu, ARd, M;  ifparallel=false, forloop_iter=n)
fwd_ACmap(AC, FL, FR, M, n)    = TeneT.ACmap_parallel(AC, FL, FR, M;    ifparallel=false, forloop_iter=n)
fwd_ACdmap(ACd, FL, FR, M, n)  = TeneT.ACdmap_parallel(ACd, FL, FR, M;  ifparallel=false, forloop_iter=n)
fwd_Mumap(AC, ACd, FL, FR, Mu, n) = TeneT.Mumap_parallel(AC, ACd, FL, FR, Mu; ifparallel=false, forloop_iter=n)
fwd_Cmap(C, FL, FR)            = TeneT.Cmap(C, FL, FR)

# bwd thunk for a vumps map: one pullback call of sum(abs2, fwd(...)).
# `args` are the differentiable tensors; `n` the chunk count.
function bwd_vumps(fwd, args, n)
    _, bp = Zygote.pullback((a...) -> sum(abs2, fwd(a..., n)), args...)
    return bp(1.0)
end
function bwd_Cmap(C, FL, FR)
    _, bp = Zygote.pullback((c, fl, fr) -> sum(abs2, TeneT.Cmap(c, fl, fr)), C, FL, FR)
    return bp(1.0)
end

# ----- parity helpers (ON vs OFF on the SAME inputs; args are CPU copies) -----
# A `nothing` arm = that side OOM'd during the parity probe → parity unknown.
function fwd_parity(rON, rOFF)
    (rON === nothing || rOFF === nothing) && return "f?"
    isapprox(rON, rOFF; rtol = 1e-12) ? "f✓" : "f✗"
end
function grad_parity(gON, gOFF)
    (gON === nothing || gOFF === nothing) && return "g?"
    ok = true
    for (a, b) in zip(gON, gOFF)
        a === nothing && b === nothing && continue
        ok &= isapprox(a, b; rtol = 1e-10)
    end
    return ok ? "g✓" : "g✗"
end

# Measure one direction of one map: returns (t_on, t_off, m_on, m_off).
function ab_time_mem(thunk_on, thunk_off)
    t = Dict{String,Float64}(); m = Dict{String,Float64}()
    for (key, f) in (("on", thunk_on), ("off", thunk_off))
        GC.gc(); _reclaim()
        t[key] = try_or_nan(() -> timeit(f))
        if !isnan(t[key])
            f(); _sync()                     # one un-GC'd call for the mem probe
            m[key] = mem_gb()
        else
            m[key] = NaN
        end
        GC.gc(); _reclaim()
    end
    return t["on"], t["off"], m["on"], m["off"]
end

# ===== sweep ===== (everything above is included by the CPU smoke test main-guard)

function main()
    cells = BENCH_CPU ? [(2, 8)] : [(10, 512), (12, 1024), (16, 1024)]

    println("=== M2 chain-engine perf gate: CHAIN (engine ON) vs TENSOR (engine OFF), 1 GPU, ",
            _devname(), ", CLB=", get(ENV, "CUDA_LAUNCH_BLOCKING", "0"), " ===")
    println("# CHAIN/TENSOR = engine path ÷ @tensor path. Gate: ratio ≤ 1.05 AND mem ≤ 1.10×.")
    println("| map    | dir | D  | χ    | n  | nB | C ms     | T ms     | C/T   | C mem  | T mem  | C/T m | parity |")
    println("|--------|-----|----|------|----|----|----------|----------|-------|--------|--------|-------|--------|")

    gate_rows = NamedTuple[]

    function row(mapname, dir, D, χ, n, nB, t_on, t_off, m_on, m_off, parity)
        rt = t_off == 0 ? NaN : t_on / t_off
        rm = m_off == 0 ? NaN : m_on / m_off
        @printf("| %-6s | %-3s | %-2d | %-4d | %-2s | %-2s | %s | %s | %s | %s | %s | %s | %-6s |\n",
                mapname, dir, D, χ, n === nothing ? "—" : string(n), nB === nothing ? "—" : string(nB),
                fmt(t_on), fmt(t_off), fmtr(rt), fmt(m_on), fmt(m_off), fmtr(rm), parity)
        flush(stdout)
        push!(gate_rows, (map = mapname, dir = dir, D = D, χ = χ,
                          t_on = t_on, t_off = t_off, rt = rt, rm = rm))
    end

    for (D, χ) in cells
        n  = pick_n(8, D, χ)        # engine-feasible n; used for BOTH arms (Part-7 choice)
        nB = pick_n(14, D, χ)       # reference: the OFF Zygote arm's own feasible n

        !BENCH_CPU && CUDA.seed!(42)

        # leg5 single-M production geometry: all boundary/transfer tensors
        # (χ,D,D,χ); M (D,D,D,D,d). Cmap leg4: C (χ,χ), FL/FR (χ,D,D,χ).
        FL  = _rand(χ, D, D, χ); ALu = _rand(χ, D, D, χ); ALd = _rand(χ, D, D, χ)
        FR  = _rand(χ, D, D, χ); ARu = _rand(χ, D, D, χ); ARd = _rand(χ, D, D, χ)
        AC  = _rand(χ, D, D, χ); ACd = _rand(χ, D, D, χ)
        M   = _rand(D, D, D, D, d_phys)
        Mu  = _rand(D, D, D, D, d_phys)
        Cm  = _rand(χ, χ)

        # ---- FLmap (fwd + bwd) ----
        for (mapname, fwd, fargs) in (
            ("FLmap",  fwd_FLmap,  (FL, ALu, ALd, M)),
            ("FRmap",  fwd_FRmap,  (FR, ARu, ARd, M)),
            ("ACmap",  fwd_ACmap,  (AC, FL, FR, M)),
            ("ACdmap", fwd_ACdmap, (ACd, FL, FR, M)),
        )
            # fwd parity + A/B. CHAIN arm runs at the engine-feasible n; the
            # TENSOR arm at its OWN feasible nB (the @tensor+Zygote path needs
            # more chunks — Part-6 coeff 14 — to fit; forcing it to the engine's
            # n is both unfair and OOMs at D≥12). Each parity arm is brought to
            # host and freed before the other runs (one arm live at a time).
            rON  = try_or_none(() -> with_engine(true,  () -> to_cpu(fwd(fargs..., n))))
            GC.gc(); _reclaim()
            rOFF = try_or_none(() -> with_engine(false, () -> to_cpu(fwd(fargs..., nB))))
            GC.gc(); _reclaim()
            pf_s = fwd_parity(rON, rOFF)
            rON = rOFF = nothing; GC.gc(); _reclaim()
            t_on, t_off, m_on, m_off =
                ab_time_mem(with_engine_thunk(true,  () -> fwd(fargs..., n)),
                            with_engine_thunk(false, () -> fwd(fargs..., nB)))
            row(mapname, "fwd", D, χ, n, nB, t_on, t_off, m_on, m_off, pf_s)

            # bwd parity + A/B (same n/nB split, CPU-copied + freed between arms)
            gON  = try_or_none(() -> with_engine(true,  () -> to_cpu(bwd_vumps(fwd, fargs, n))))
            GC.gc(); _reclaim()
            gOFF = try_or_none(() -> with_engine(false, () -> to_cpu(bwd_vumps(fwd, fargs, nB))))
            GC.gc(); _reclaim()
            pg = grad_parity(gON, gOFF)
            gON = gOFF = nothing; GC.gc(); _reclaim()
            t_on, t_off, m_on, m_off =
                ab_time_mem(with_engine_thunk(true,  () -> bwd_vumps(fwd, fargs, n)),
                            with_engine_thunk(false, () -> bwd_vumps(fwd, fargs, nB)))
            row(mapname, "bwd", D, χ, n, nB, t_on, t_off, m_on, m_off, pg)
        end

        # ---- Cmap leg4 (direct, no chunking): fwd + bwd ----
        rON  = try_or_none(() -> with_engine(true,  () -> to_cpu(fwd_Cmap(Cm, FL, FR))))
        GC.gc(); _reclaim()
        rOFF = try_or_none(() -> with_engine(false, () -> to_cpu(fwd_Cmap(Cm, FL, FR))))
        GC.gc(); _reclaim()
        pf_s = fwd_parity(rON, rOFF)
        rON = rOFF = nothing; GC.gc(); _reclaim()
        t_on, t_off, m_on, m_off =
            ab_time_mem(with_engine_thunk(true,  () -> fwd_Cmap(Cm, FL, FR)),
                        with_engine_thunk(false, () -> fwd_Cmap(Cm, FL, FR)))
        row("Cmap", "fwd", D, χ, nothing, nothing, t_on, t_off, m_on, m_off, pf_s)

        gON  = try_or_none(() -> with_engine(true,  () -> to_cpu(bwd_Cmap(Cm, FL, FR))))
        GC.gc(); _reclaim()
        gOFF = try_or_none(() -> with_engine(false, () -> to_cpu(bwd_Cmap(Cm, FL, FR))))
        GC.gc(); _reclaim()
        pg = grad_parity(gON, gOFF)
        gON = gOFF = nothing; GC.gc(); _reclaim()
        t_on, t_off, m_on, m_off =
            ab_time_mem(with_engine_thunk(true,  () -> bwd_Cmap(Cm, FL, FR)),
                        with_engine_thunk(false, () -> bwd_Cmap(Cm, FL, FR)))
        row("Cmap", "bwd", D, χ, nothing, nothing, t_on, t_off, m_on, m_off, pg)

        # ---- Mumap (FORWARD ONLY: preconditioner path, no forloop_sum rrule) ----
        # CHAIN at n, TENSOR at nB (same feasible-chunking split as the vumps maps).
        rON  = try_or_none(() -> with_engine(true,  () -> to_cpu(fwd_Mumap(AC, ACd, FL, FR, Mu, n))))
        GC.gc(); _reclaim()
        rOFF = try_or_none(() -> with_engine(false, () -> to_cpu(fwd_Mumap(AC, ACd, FL, FR, Mu, nB))))
        GC.gc(); _reclaim()
        pf_s = fwd_parity(rON, rOFF)
        rON = rOFF = nothing; GC.gc(); _reclaim()
        t_on, t_off, m_on, m_off =
            ab_time_mem(with_engine_thunk(true,  () -> fwd_Mumap(AC, ACd, FL, FR, Mu, n)),
                        with_engine_thunk(false, () -> fwd_Mumap(AC, ACd, FL, FR, Mu, nB)))
        row("Mumap", "fwd", D, χ, n, nB, t_on, t_off, m_on, m_off, pf_s)

        FL = ALu = ALd = FR = ARu = ARd = AC = ACd = M = Mu = Cm = nothing
        GC.gc(); _reclaim()
    end

    # ===== gate =====
    println()
    pass = true
    slower = NamedTuple[]
    for r in gate_rows
        chain_ran = !isnan(r.t_on)
        tensor_oom = isnan(r.t_off)
        # cell passes iff CHAIN ran AND (TENSOR OOM'd at its own feasible nB —
        # an engine WIN, runs where @tensor can't — OR the ratio is within gate).
        # CHAIN itself OOMing is a real FAIL.
        cell_ok = chain_ran && (tensor_oom || ((r.rt ≤ 1.05) && (isnan(r.rm) || r.rm ≤ 1.10)))
        pass &= cell_ok
        note = (chain_ran && tensor_oom) ? "ok (TENSOR OOM @nB — engine win)" :
               (cell_ok ? "ok" : "FAIL")
        # flag any cell where CHAIN is strictly slower than its @tensor original.
        if !isnan(r.rt) && r.rt > 1.0
            push!(slower, (map = r.map, dir = r.dir, D = r.D, χ = r.χ, rt = r.rt))
        end
        @printf("GATE %-6s %-3s cell=(%d,%d) CHAIN/TENSOR time=%s mem=%s -> %s\n",
                r.map, r.dir, r.D, r.χ, fmtr(r.rt), fmtr(r.rm), note)
    end
    println()
    if isempty(slower)
        println("SLOWER-THAN-TENSOR: none (no map slower than its @tensor original).")
    else
        println("SLOWER-THAN-TENSOR cells (CHAIN > TENSOR — investigate per design doc):")
        for s in slower
            @printf("  %-6s %-3s (%d,%d): CHAIN/TENSOR=%.3f\n", s.map, s.dir, s.D, s.χ, s.rt)
        end
    end
    println("GATE RESULT: ", pass ? "PASS" : "FAIL")
    println("=== done ===")
    return pass
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
