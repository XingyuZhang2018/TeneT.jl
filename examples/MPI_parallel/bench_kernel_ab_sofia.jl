# Single-GPU A/B: staged cannon kernels vs monolithic FLmap+forloop on the
# IDENTICAL local workload of one 2×2-grid rank (no MPI — isolates kernel
# organization from communication).
#
# Workload per (D, χ): FL_row[χ/2, D, D, χ] (a-block, full i — the cached
# ring row, contiguous), ALu_row[χ/2, D, D, χ], ALd_col[χ, D, D, χ/2];
# output partial[χ, D, D, χ/2]. Both paths sub-slice the local l range into
# the same n chunks (backward-peak formula).
#
#   A (staged):    per chunk stage1 → fold1 → fold2 → stage2, owned
#                  intermediates with eager _free!; backward = the six
#                  hand-written adjoints, mirroring the FLmap_cannon rrule.
#   B (monolithic): TeneT.forloop(TeneT.FLmap, ...) — the original 5-tensor
#                  @tensor kernel, TensorOperations-managed temporaries;
#                  backward = the existing forloop rrule (per-slice Zygote).
#
# Memory columns: device used (total−available) right AFTER the timed calls
# (live + uncollected garbage — the pool pressure the algorithm creates) and
# after GC+reclaim (retained floor). Same CLB=1 env as Part 5 for relevance.
using CUDA, Zygote, LinearAlgebra, Printf, TeneT

const nrep = 3
const d_phys = 2
const MEM_BUDGET = 110e9
const BWD_COEFF = 4 + 2 * d_phys

H_bytes(D, χ) = χ^2 * D^4 / 4 * 8
tensor_bytes(D, χ) = χ^2 * D^2 * 8
# Per-path chunk counts: A's ordered hand adjoints peak at ≈(2+2d)|H|/n (use
# 8 with margin); B's Zygote backward holds tape + cotangent chain ≈10-11
# units (job 1275726 OOMed at (10,768) n=1 with 130 GiB live) — use 14.
# Each path runs at its own memory-feasible n, both reported.
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

# A: staged forward — full-i stage1 per l-chunk (row is contiguous, so the
# i-sum happens inside the single stage1 contraction; no per-block loop).
function staged_fwd(FLr, ALur, ALdc, M1, M2, n)
    χ = size(ALur, 4); χl = size(ALdc, 4)
    partial = similar(FLr, χ, size(M1, 3), size(M2, 3), χl)
    for ch in TeneT.split_ranges(χl, n)
        H = TeneT._cannon_stage1(FLr, view(ALdc, :, :, :, ch))
        T = TeneT._cannon_fold1(H, M1)
        G = TeneT._cannon_fold2(T, M2); TeneT._free!(T)
        P = TeneT._cannon_stage2(G, ALur); TeneT._free!(G)
        view(partial, :, :, :, ch) .= P
        TeneT._free!(P); TeneT._free!(H)
    end
    return partial
end

# A: staged backward — the six hand adjoints with eager frees (mirrors the
# FLmap_cannon rrule chunk body).
function staged_bwd(FLr, ALur, ALdc, M1, M2, dout, n)
    dFL = zero(FLr); dALu = zero(ALur); dALd = zero(ALdc)
    dM1 = zero(M1); dM2 = zero(M2)
    χl = size(ALdc, 4)
    for ch in TeneT.split_ranges(χl, n)
        ALd_ch = view(ALdc, :, :, :, ch)
        H = TeneT._cannon_stage1(FLr, ALd_ch)
        T = TeneT._cannon_fold1(H, M1)
        G = TeneT._cannon_fold2(T, M2)
        dP = dout[:, :, :, ch]
        dG = TeneT._cannon_stage2_dG(dP, ALur)
        tmp = TeneT._cannon_stage2_dALu(dP, G); dALu .+= tmp
        TeneT._free!(tmp); TeneT._free!(dP); TeneT._free!(G)
        tmp = TeneT._cannon_fold2_dM2(dG, T); dM2 .+= tmp; TeneT._free!(tmp)
        dT = TeneT._cannon_fold2_dT(dG, M2); TeneT._free!(dG); TeneT._free!(T)
        tmp = TeneT._cannon_fold1_dM1(dT, H); dM1 .+= tmp; TeneT._free!(tmp)
        dH = TeneT._cannon_fold1_dH(dT, M1); TeneT._free!(dT)
        tmp = TeneT._cannon_stage1_dFL(dH, ALd_ch); dFL .+= tmp; TeneT._free!(tmp)
        tmp = TeneT._cannon_stage1_dALd(dH, FLr)
        view(dALd, :, :, :, ch) .+= tmp
        TeneT._free!(tmp); TeneT._free!(dH); TeneT._free!(H)
    end
    return (dFL, dALu, dALd, dM1, dM2)
end

println("=== Kernel A/B: staged cannon kernels vs FLmap+forloop (1 GPU, ", CUDA.name(CUDA.device()), ", CLB=", get(ENV, "CUDA_LAUNCH_BLOCKING", "0"), ") ===")
println("| D  | χ    | nA | nB | A fwd ms | B fwd ms | A bwd ms | B bwd ms | A fwd mem | B fwd mem | A bwd mem | B bwd mem | parity |")
println("|----|------|----|----|----------|----------|----------|----------|-----------|-----------|-----------|-----------|--------|")

for (D, χ) in [(8, 256), (8, 512), (10, 512), (10, 768), (12, 768), (12, 1024), (14, 1024), (16, 1024)]
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

    # parity first (also warms both paths); GC between steps so one path's
    # dead results never stack under the other's working set
    rA = staged_fwd(FLr, ALur, ALdc, M1, M2, nA)
    rB = TeneT.forloop(TeneT.FLmap, FLr, ALur, ALdc, (M1, M2); fl_kwA...)
    pf = isapprox(rA, rB; rtol = 1e-11) ? "F✓" : "F✗"
    rA = rB = nothing
    GC.gc(); CUDA.reclaim()
    gA = staged_bwd(FLr, ALur, ALdc, M1, M2, dout, nA)
    GC.gc(); CUDA.reclaim()
    gB = try_or_nan() do
        _, bpB = Zygote.pullback((a, b, c, m) -> TeneT.forloop(TeneT.FLmap, a, b, c, m; fl_kwB...),
                                 FLr, ALur, ALdc, (M1, M2))
        bpB(dout)
    end
    pb = gB isa Tuple ?
         ((isapprox(gA[1], gB[1]; rtol = 1e-10) && isapprox(gA[2], gB[2]; rtol = 1e-10) &&
           isapprox(gA[3], gB[3]; rtol = 1e-10) && isapprox(gA[4], gB[4][1]; rtol = 1e-10) &&
           isapprox(gA[5], gB[4][2]; rtol = 1e-10)) ? "B✓" : "B✗") : "B–oom"
    rA = rB = gA = gB = nothing
    GC.gc(); CUDA.reclaim()

    # timing + memory per section: GC+reclaim → timed (per-rep GC, no reclaim)
    # → record used (live+garbage high-water proxy) → reclaim
    t = Dict{String, Float64}(); m = Dict{String, Float64}()
    for (key, f) in (
        ("Af", () -> staged_fwd(FLr, ALur, ALdc, M1, M2, nA)),
        ("Bf", () -> TeneT.forloop(TeneT.FLmap, FLr, ALur, ALdc, (M1, M2); fl_kwA...)),
        ("Ab", () -> staged_bwd(FLr, ALur, ALdc, M1, M2, dout, nA)),
        ("Bb", () -> Zygote.pullback((a, b, c, mm) -> TeneT.forloop(TeneT.FLmap, a, b, c, mm; fl_kwB...),
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

    @printf("| %-2d | %-4d | %-2d | %-2d | %s | %s | %s | %s | %s | %s | %s | %s | %s %s |\n",
            D, χ, nA, nB, fmt(t["Af"]), fmt(t["Bf"]), fmt(t["Ab"]), fmt(t["Bb"]),
            fmt(m["Af"]), fmt(m["Bf"]), fmt(m["Ab"]), fmt(m["Bb"]), pf, pb)
    flush(stdout)

    FLr = ALur = ALdc = M1 = M2 = dout = nothing
    GC.gc(); CUDA.reclaim()
end

println("=== done ===")
