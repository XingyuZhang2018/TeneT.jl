# timing/vumps_alternatives/common.jl
# Shared infra for the VUMPS fixed-point alternatives study.
# Design: docs/2026-07-08-vumps-fixed-point-alternatives-design.md
using TeneT, Random, LinearAlgebra, Printf, KrylovKit

const BETA_EASY = 0.43
const BETA_CRIT = log(1 + sqrt(2)) / 2   # 0.4406867935097715

mutable struct MapCounter
    fl::Int; ac::Int; c::Int; nrm::Int      # nrm = norm-channel maps (family C)
    diag::Bool
    dfl::Int; dac::Int; dc::Int; dnrm::Int
end
MapCounter() = MapCounter(0,0,0,0,false,0,0,0,0)

function bump!(cnt::MapCounter, kind::Symbol)
    if cnt.diag
        kind === :fl  && (cnt.dfl += 1);  kind === :ac && (cnt.dac += 1)
        kind === :c   && (cnt.dc  += 1);  kind === :nrm && (cnt.dnrm += 1)
    else
        kind === :fl  && (cnt.fl  += 1);  kind === :ac && (cnt.ac  += 1)
        kind === :c   && (cnt.c   += 1);  kind === :nrm && (cnt.nrm += 1)
    end
    return nothing
end

counted(cnt::MapCounter, kind::Symbol, f) = x -> (bump!(cnt, kind); f(x))

function with_diagnostics(g, cnt::MapCounter)
    old = cnt.diag; cnt.diag = true
    try; return g(); finally; cnt.diag = old; end
end

total_maps(cnt::MapCounter) = cnt.fl + cnt.ac + cnt.c + cnt.nrm
# Cost weights: FLmap≈ACmap dominate (extra D² legs vs Cmap); norm channel ~D cheaper.
weighted_maps(cnt::MapCounter; D::Int) =
    cnt.fl * D^2 + cnt.ac * D^2 + cnt.c * 1 + cnt.nrm * D

struct TrajectoryLog
    rows::Vector{NamedTuple}
end
TrajectoryLog() = TrajectoryLog(NamedTuple[])
push_row!(t::TrajectoryLog; kw...) = push!(t.rows, (; kw...))

function save_traj(path::AbstractString, t::TrajectoryLog; meta::Dict)
    open(path, "w") do io
        println(io, "# " * join(("$k=$(meta[k])" for k in sort!(collect(keys(meta)))), " "))
        println(io, "outer,fl,ac,c,nrm,err,f,t")
        for r in t.rows
            @printf(io, "%d,%d,%d,%d,%d,%.6e,%.14f,%.3f\n",
                    r.outer, r.fl, r.ac, r.c, r.nrm, r.err, r.f, r.t)
        end
    end
end

# ── physics setup ────────────────────────────────────────────────────────────
function setup_ising(; beta::Float64, chi::Int, seed::Int=42)
    Random.seed!(seed)
    model = Ising(lattice=Square(), beta=beta)
    M0 = MPO(model, C4v; atype=Array)
    M  = TeneT._c4v_local_tensor(M0)
    alg = VUMPS{C4v}(; ifsimple_eig=true, ifparallel=false, power_iter=5,
                       maxiter=1, maxiter_ad=1, miniter_ad=1, verbosity=0)
    rt = init_env(M0, chi, alg)          # C4vVUMPSEnv(AL, C, FL), deterministic via seed
    f_exact = exact_free_energy(model; npts=2000)
    return (; model, M0, M, rt, alg, f_exact)
end

# Counted raw maps (serial path, mirrors c4v.jl kernels exactly)
flmap(cnt, FL, AL, M) = (bump!(cnt, :fl);
    TeneT.FLmap_parallel(FL, AL, conj(AL), M; ifparallel=false, forloop_iter=1))
acmap(cnt, AC, FL, M) = (bump!(cnt, :ac);
    TeneT.ACmap_parallel(AC, FL, FL, M; ifparallel=false, forloop_iter=1))
cmap(cnt, C, FL)      = (bump!(cnt, :c); TeneT.Cmap(C, FL, FL))

# Cheap diagnostic free energy from current (AL, C, FL): Rayleigh quotients,
# one ACmap + one Cmap, charged to the diagnostic buckets.
function diag_free_energy(cnt, AL, C, FL, M, beta)
    return with_diagnostics(cnt) do
        AC = TeneT.ALCtoAC_map(AL, C)
        λac = real(dot(AC, acmap(cnt, AC, FL, M))) / real(dot(AC, AC))
        λc  = real(dot(C,  cmap(cnt, C, FL)))      / real(dot(C, C))
        -log(λac / λc) / (2beta)
    end
end

# Gauge residual, same definition as c4v.jl vumps_step
function gauge_err(AL, C, M, FL, cnt)
    return with_diagnostics(cnt) do
        AC = TeneT.ALCtoAC_map(AL, C)
        AC2 = acmap(cnt, AC, FL, M); AC2 /= norm(AC2)
        C2  = cmap(cnt, C, FL);      C2  /= norm(C2)
        _, RAC = TeneT.qrpos(TeneT._to_front(AC2))
        _, RC  = TeneT.qrpos(C2)
        norm(RAC - RC)
    end
end

# AL from AC and C (same as c4v.jl vumps_step tail)
function accto_al(AC, C)
    QAC, RAC = TeneT.qrpos(TeneT._to_front(AC))
    QC,  RC  = TeneT.qrpos(C)
    AL = reshape(QAC * QC', size(AC))
    return AL, norm(RAC - RC)
end
