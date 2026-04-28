# LBFGS warmup at 2×2 unit cell (pattern = [1 2; 2 1] — bipartite AFM stagger).
# No C4v restriction, just plain LBFGS via optimise_ipeps.
# Uses χ=16 + ifupdown=true (the standard production setup) for robust convergence.

using TeneT
using JLD2
using Random
using LinearAlgebra
using OptimKit

Random.seed!(42)

const D = 2
const χ = 16
const atype = Array
const etype = Float64
const pattern = [1 2; 2 1]

const boundary_alg = VUMPS{General}(maxiter=30, miniter=1, tol=1e-10,
                                    ifupdown=true,
                                    ifsimple_eig=true,
                                    ifparallelupdown=false,
                                    ifparallel=false,
                                    forloop_iter=1,
                                    verbosity=0,
                                    show_every=10)

const model = Heisenberg(lattice=Square(), S=0.5,
                         Jx=1.0, Jy=1.0, Jz=1.0, ifrotate=true,
                         couplingtype=:uniform, bondratio=1.0)

const folder = joinpath(@__DIR__, "..", "data", "lbfgs_warmup_2x2")
mkpath(folder)

const params = GradientOptimize(
    model = model,
    pattern = pattern,
    boundary_alg = boundary_alg,
    optimizer = LBFGS(20; maxiter=80, verbosity=2, gradtol=1e-7,
                      linesearch=HagerZhangLineSearch(maxfg=8)),
    forloop_iter = 1,
    maxiter_restart = 1,
    verbosity = 3,
    folder = folder,
    ifSU = false,
    SUτ = 0,
    ifprecondition = false,
    iter_precond = 0,
    reuse_env = true,
    ifsave_env = false,
    ifload_env = false,
    ifsave_lbfgs = false,
    ifload_lbfgs = false,
)

println("="^60)
println("LBFGS warmup 2×2: D=$D, χ=$χ, pattern=$(pattern)")
println("model = $model")
println("Output folder: $folder")
println("="^60)

A = init_ipeps(; atype=atype, etype=etype, No=0, D=D, χ=χ, params=params)
println("Initial A shape: ", size(A))
println("Initial A norm:  ", norm(A))

# No C4v restriction — only normalisation via local_min_norm.
function restriction_ipeps(A)
    A = local_min_norm(A, params)
    return A
end

const t_start = time()
optimise_ipeps(A, χ, 0, params; restriction_ipeps)
const t_total = time() - t_start

println("="^60)
println("LBFGS warmup 2×2 done in $(round(t_total, digits=1)) sec")
println("="^60)
