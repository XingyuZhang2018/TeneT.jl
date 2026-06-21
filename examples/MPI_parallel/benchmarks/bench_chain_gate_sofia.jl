# Single-GPU chain-engine perf gate: CHAIN (generic chain engine) vs HAND
# (staged slice2d kernels) vs TENSOR (monolithic FLmap+forloop) on the
# IDENTICAL local workload of one 2×2-grid rank (no MPI).
#
# Workload per (D, χ): FL_row[χ/2, D, D, χ] (a-block, full i — the cached
# ring row, contiguous), ALu_row[χ/2, D, D, χ], ALd_col[χ, D, D, χ/2];
# output partial[χ, D, D, χ/2].
#
#   HAND:   per chunk stage1 → fold1 → fold2 → stage2, owned intermediates
#           with eager _free!; backward = the six hand-written adjoints
#           (Part 6's path A, copied verbatim from bench_kernel_ab_sofia.jl).
#   CHAIN:  per chunk chain_apply / chain_backward of FLMAP_LEG5_CHAIN at the
#           SAME n as HAND — same per-chunk structure (one full-i first link
#           per chunk), runtime tensorcontract instead of @tensor kernels.
#   TENSOR: TeneT.forloop(TeneT.FLmap, ...) — the original 5-tensor @tensor
#           kernel; backward = the existing forloop rrule (Part 6's path B).
#
# Gate (printed after the table): CHAIN/HAND time ratio per cell per
# direction; PASS iff every ratio ≤ 1.05 AND CHAIN mem ≤ 1.10× HAND mem.
# Memory columns: device used (total−available) right AFTER the timed calls.
using CUDA, Zygote, LinearAlgebra, Printf, TeneT

const nrep = 3
const d_phys = 2
const MEM_BUDGET = 110e9
const BWD_COEFF = 4 + 2 * d_phys

H_bytes(D, χ) = χ^2 * D^4 / 4 * 8
tensor_bytes(D, χ) = χ^2 * D^2 * 8
# Per-path chunk counts (Part 6 formulas): the ordered hand/chain adjoints
# peak at ≈(2+2d)|H|/n (use 8 with margin); TENSOR's Zygote backward holds
# tape + cotangent chain — use 14.
pick_n(coeff, D, χ) = max(1, ceil(Int, coeff * H_bytes(D, χ) / (MEM_BUDGET - 12 * tensor_bytes(D, χ))))

# Single-process OOM guard (no MPI → catching is deadlock-free): report the
# cell as NaN and clean up instead of killing the sweep.
function try_or_nan(f)
    try
        return f()
    catch e
        e isa CUDA.OutOfGPUMemoryError || rethrow()
        GC.gc(); CUDA.reclaim()
        return NaN
    end
end

mem_gb() = (CUDA.total_memory() - CUDA.available_memory()) / 2^30
fmt(x) = isnan(x) ? @sprintf("%8s", "oom") : @sprintf("%8.1f", x)

function timeit(f)
    f(); CUDA.synchronize()
    GC.gc()
    tot = 0.0
    for _ in 1:nrep
        t0 = time()
        f(); CUDA.synchronize()
        tot += time() - t0
        GC.gc()
    end
    return tot / nrep * 1000
end

# HAND: staged forward — full-i stage1 per l-chunk (row is contiguous, so the
# i-sum happens inside the single stage1 contraction; no per-block loop).
function staged_fwd(FLr, ALur, ALdc, M1, M2, n)
    χ = size(ALur, 4); χl = size(ALdc, 4)
    partial = similar(FLr, χ, size(M1, 3), size(M2, 3), χl)
    for ch in TeneT.split_ranges(χl, n)
        H = TeneT._slice2d_stage1(FLr, view(ALdc, :, :, :, ch))
        T = TeneT._slice2d_fold1(H, M1)
        G = TeneT._slice2d_fold2(T, M2); TeneT._free!(T)
        P = TeneT._slice2d_stage2(G, ALur); TeneT._free!(G)
        view(partial, :, :, :, ch) .= P
        TeneT._free!(P); TeneT._free!(H)
    end
    return partial
end

# HAND: staged backward — the six hand adjoints with eager frees (mirrors the
# FLmap_slice2d rrule chunk body).
function staged_bwd(FLr, ALur, ALdc, M1, M2, dout, n)
    dFL = zero(FLr); dALu = zero(ALur); dALd = zero(ALdc)
    dM1 = zero(M1); dM2 = zero(M2)
    χl = size(ALdc, 4)
    for ch in TeneT.split_ranges(χl, n)
        ALd_ch = view(ALdc, :, :, :, ch)
        H = TeneT._slice2d_stage1(FLr, ALd_ch)
        T = TeneT._slice2d_fold1(H, M1)
        G = TeneT._slice2d_fold2(T, M2)
        dP = dout[:, :, :, ch]
        dG = TeneT._slice2d_stage2_dG(dP, ALur)
        tmp = TeneT._slice2d_stage2_dALu(dP, G); dALu .+= tmp
        TeneT._free!(tmp); TeneT._free!(dP); TeneT._free!(G)
        tmp = TeneT._slice2d_fold2_dM2(dG, T); dM2 .+= tmp; TeneT._free!(tmp)
        dT = TeneT._slice2d_fold2_dT(dG, M2); TeneT._free!(dG); TeneT._free!(T)
        tmp = TeneT._slice2d_fold1_dM1(dT, H); dM1 .+= tmp; TeneT._free!(tmp)
        dH = TeneT._slice2d_fold1_dH(dT, M1); TeneT._free!(dT)
        tmp = TeneT._slice2d_stage1_dFL(dH, ALd_ch); dFL .+= tmp; TeneT._free!(tmp)
        tmp = TeneT._slice2d_stage1_dALd(dH, FLr)
        view(dALd, :, :, :, ch) .+= tmp
        TeneT._free!(tmp); TeneT._free!(dH); TeneT._free!(H)
    end
    return (dFL, dALu, dALd, dM1, dM2)
end

# CHAIN: engine forward — identical per-chunk structure to staged_fwd (the
# row is contiguous, so chain_apply's first link is one full-i contraction
# per chunk, exactly like staged_fwd's single stage1 call).
function chain_fwd(FLr, ALur, ALdc, M1, M2, n)
    χ = size(ALur, 4); χl = size(ALdc, 4)
    partial = similar(FLr, χ, size(M1, 3), size(M2, 3), χl)
    for ch in TeneT.split_ranges(χl, n)
        P = TeneT.chain_apply(TeneT.FLMAP_LEG5_CHAIN,
                              (FLr, view(ALdc, :, :, :, ch), M1, M2, ALur))
        view(partial, :, :, :, ch) .= P
        TeneT._free!(P)
    end
    return partial
end

# CHAIN: engine backward — chain_backward per chunk (recompute-style inside),
# accumulation mirroring staged_bwd 1:1: dFL/dM1/dM2/dALu accumulate across
# chunks, dALd writes its chunk slice. chain_backward returns grads in chain
# operand order (FL, ALd, M1, M2, ALu); all five are owned → _free! after
# accumulation. Return order matches staged_bwd: (dFL, dALu, dALd, dM1, dM2).
function chain_bwd(FLr, ALur, ALdc, M1, M2, dout, n)
    dFL = zero(FLr); dALu = zero(ALur); dALd = zero(ALdc)
    dM1 = zero(M1); dM2 = zero(M2)
    χl = size(ALdc, 4)
    for ch in TeneT.split_ranges(χl, n)
        g = TeneT.chain_backward(TeneT.FLMAP_LEG5_CHAIN,
                                 (FLr, view(ALdc, :, :, :, ch), M1, M2, ALur),
                                 view(dout, :, :, :, ch))
        dFL .+= g[1]
        view(dALd, :, :, :, ch) .+= g[2]
        dM1 .+= g[3]; dM2 .+= g[4]; dALu .+= g[5]
        foreach(TeneT._free!, g)
    end
    return (dFL, dALu, dALd, dM1, dM2)
end

# ===== sweep ===== (everything above is included by the CPU smoke test)

println("=== Chain-engine perf gate: CHAIN vs HAND vs TENSOR (1 GPU, ", CUDA.name(CUDA.device()), ", CLB=", get(ENV, "CUDA_LAUNCH_BLOCKING", "0"), ") ===")
println("| D  | χ    | n  | nB | H fwd ms | C fwd ms | T fwd ms | H bwd ms | C bwd ms | T bwd ms | Hf mem | Cf mem | Tf mem | Hb mem | Cb mem | Tb mem | parity |")
println("|----|------|----|----|----------|----------|----------|----------|----------|----------|--------|--------|--------|--------|--------|--------|--------|")

gate_rows = NamedTuple[]

for (D, χ) in [(10, 512), (12, 1024), (16, 1024)]
    nA = pick_n(8, D, χ)
    nB = pick_n(14, D, χ)
    χa = χ ÷ 2; χl = χ ÷ 2
    CUDA.seed!(42)
    FLr  = CUDA.rand(Float64, χa, D, D, χ)
    ALur = CUDA.rand(Float64, χa, D, D, χ)
    ALdc = CUDA.rand(Float64, χ, D, D, χl)
    M1   = CUDA.rand(Float64, D, D, D, D, d_phys)
    M2   = CUDA.rand(Float64, D, D, D, D, d_phys)
    dout = CUDA.rand(Float64, χ, D, D, χl)
    fl_kwA = (forloop_iter = nA, N_in = (3, 4), N_out = 4, size_out = (χ, D, D, χl))
    fl_kwB = (forloop_iter = nB, N_in = (3, 4), N_out = 4, size_out = (χ, D, D, χl))

    # parity first (also warms all paths); GC between steps so one path's
    # dead results never stack under the other's working set
    rH = staged_fwd(FLr, ALur, ALdc, M1, M2, nA)
    rC = chain_fwd(FLr, ALur, ALdc, M1, M2, nA)
    pf = isapprox(rC, rH; rtol = 1e-11) ? "F✓" : "F✗"
    rC = nothing
    GC.gc(); CUDA.reclaim()
    rT = try_or_nan() do
        TeneT.forloop(TeneT.FLmap, FLr, ALur, ALdc, (M1, M2); fl_kwA...)
    end
    tf = rT isa AbstractArray ? (isapprox(rT, rH; rtol = 1e-11) ? "Tf✓" : "Tf✗") : "Tf–oom"
    rH = rT = nothing
    GC.gc(); CUDA.reclaim()
    gH = staged_bwd(FLr, ALur, ALdc, M1, M2, dout, nA)
    GC.gc(); CUDA.reclaim()
    gC = chain_bwd(FLr, ALur, ALdc, M1, M2, dout, nA)
    pb = all(isapprox(gC[i], gH[i]; rtol = 1e-10) for i in 1:5) ? "B✓" : "B✗"
    gC = nothing
    GC.gc(); CUDA.reclaim()
    gT = try_or_nan() do
        _, bpT = Zygote.pullback((a, b, c, m) -> TeneT.forloop(TeneT.FLmap, a, b, c, m; fl_kwB...),
                                 FLr, ALur, ALdc, (M1, M2))
        bpT(dout)
    end
    tb = gT isa Tuple ?
         ((isapprox(gH[1], gT[1]; rtol = 1e-10) && isapprox(gH[2], gT[2]; rtol = 1e-10) &&
           isapprox(gH[3], gT[3]; rtol = 1e-10) && isapprox(gH[4], gT[4][1]; rtol = 1e-10) &&
           isapprox(gH[5], gT[4][2]; rtol = 1e-10)) ? "Tb✓" : "Tb✗") : "Tb–oom"
    gH = gT = nothing
    GC.gc(); CUDA.reclaim()

    # timing + memory per section: GC+reclaim → timed (per-rep GC, no reclaim)
    # → record used (live+garbage high-water proxy) → reclaim
    t = Dict{String, Float64}(); m = Dict{String, Float64}()
    for (key, f) in (
        ("Hf", () -> staged_fwd(FLr, ALur, ALdc, M1, M2, nA)),
        ("Cf", () -> chain_fwd(FLr, ALur, ALdc, M1, M2, nA)),
        ("Tf", () -> TeneT.forloop(TeneT.FLmap, FLr, ALur, ALdc, (M1, M2); fl_kwA...)),
        ("Hb", () -> staged_bwd(FLr, ALur, ALdc, M1, M2, dout, nA)),
        ("Cb", () -> chain_bwd(FLr, ALur, ALdc, M1, M2, dout, nA)),
        ("Tb", () -> Zygote.pullback((a, b, c, mm) -> TeneT.forloop(TeneT.FLmap, a, b, c, mm; fl_kwB...),
                                     FLr, ALur, ALdc, (M1, M2))[2](dout)),
    )
        GC.gc(); CUDA.reclaim()
        t[key] = try_or_nan() do
            timeit(f)
        end
        if !isnan(t[key])
            f(); CUDA.synchronize()      # one un-GC'd call for the memory probe
            m[key] = mem_gb()
        else
            m[key] = NaN
        end
        GC.gc(); CUDA.reclaim()
    end

    @printf("| %-2d | %-4d | %-2d | %-2d | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s %s %s %s |\n",
            D, χ, nA, nB, fmt(t["Hf"]), fmt(t["Cf"]), fmt(t["Tf"]), fmt(t["Hb"]), fmt(t["Cb"]), fmt(t["Tb"]),
            fmt(m["Hf"]), fmt(m["Cf"]), fmt(m["Tf"]), fmt(m["Hb"]), fmt(m["Cb"]), fmt(m["Tb"]), pf, pb, tf, tb)
    flush(stdout)

    push!(gate_rows, (D = D, χ = χ,
                      rf = t["Cf"] / t["Hf"], rb = t["Cb"] / t["Hb"],
                      mf = m["Cf"] / m["Hf"], mb = m["Cb"] / m["Hb"]))

    FLr = ALur = ALdc = M1 = M2 = dout = nothing
    GC.gc(); CUDA.reclaim()
end

# ===== gate =====
pass = true
for r in gate_rows
    @printf("GATE cell=(%d,%d) fwd CHAIN/HAND=%.3f bwd CHAIN/HAND=%.3f fwd-mem CHAIN/HAND=%.3f bwd-mem CHAIN/HAND=%.3f\n",
            r.D, r.χ, r.rf, r.rb, r.mf, r.mb)
    # NaN comparisons are false → an OOM'd HAND or CHAIN cell fails the gate.
    global pass &= (r.rf ≤ 1.05) && (r.rb ≤ 1.05) && (r.mf ≤ 1.10) && (r.mb ≤ 1.10)
end
println("GATE RESULT: ", pass ? "PASS" : "FAIL")

println("=== done ===")
