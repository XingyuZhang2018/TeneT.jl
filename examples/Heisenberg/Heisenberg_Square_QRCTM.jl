# Heisenberg QRCTM example / production-size smoke.
#
# Runs a single LBFGS iter at D=7 χ=256 on CuArray with
# `step_checkpoint=Offload()` (pb-capture walker: GPU→CPU on forward, reload
# + pb once on backward — no forward recomputation). Wraps in memory-info +
# wall-time instrumentation to make it usable as a Phase 2-D regression
# smoke on a 24 GB RTX 4090.
#
# To turn this into a real optimization run, bump `LBFGS(; maxiter=...)` and
# `maxiter_restart`. To scale down for a quick smoke on small GPUs, drop
# `D, χ, forloop_iter` (e.g. D=2 χ=16 forloop_iter=1).
#
# Usage:
#   julia --project=. examples/Heisenberg/Heisenberg_Square_QRCTM.jl

using TeneT
using Random
using CUDA
using OptimKit
using LinearAlgebra

seed = 42
Random.seed!(seed)
atype = CuArray
etype = Float64
D, χ, χshift = 7, 256, 0
pattern = [1;;]
model = Heisenberg(lattice=Square(),
                   S=0.5, Jx=1.0, Jy=1.0, Jz=1.0,
                   ifrotate=true,
                   couplingtype=:uniform, bondratio=1.0)
No = 0
folder = joinpath(pkgdir(TeneT), "data/$model/$pattern/QRCTM/$etype/seed$seed/")

# forloop_iter=2: required on 24 GB 4090 at D=7 χ=256 to avoid observable-
# backward OOM (see claude/heuristic-pike-f16f59 commit b484911).
boundary_alg = QRCTM(ifparallel=false,
                     step_checkpoint=Offload(),
                     forloop_iter=2,
                     maxiter=30, miniter=0,
                     maxiter_ad=20, miniter_ad=20,
                     show_every=1, tol=1e-10,
                     verbosity=3)

params = GradientOptimize(model=model,
                          pattern=pattern,
                          boundary_alg=boundary_alg,
                          optimizer=LBFGS(200; maxiter=1, verbosity=4, gradtol=1e-7,
                                           linesearch=HagerZhangLineSearch(maxfg=1)),
                          forloop_iter=2,
                          maxiter_restart=1,
                          verbosity=4, folder=folder,
                          ifSU=false, SUτ=0,
                          ifprecondition=true, iter_precond=0,
                          reuse_env=true,
                          ifsave_env=false, ifload_env=false,
                          ifsave_lbfgs=false, ifload_lbfgs=false)

A = init_ipeps(; atype, etype, No, D, χ, params)
function restriction_ipeps(A)
    return C4v_restriction(A)
end

CUDA.reclaim(); GC.gc()
t0 = time()
free_before, total = CUDA.memory_info()
println("Mem before: free=$(free_before/1024^2) MB, total=$(total/1024^2) MB")

optimise_ipeps(A, χ, χshift, params; restriction_ipeps)

free_after, _ = CUDA.memory_info()
used_delta = (free_before - free_after) / 1024^2
wall = time() - t0
@info "QRCTM Offload() smoke" wall used_delta_MB=used_delta
