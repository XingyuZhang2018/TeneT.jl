using TeneT
using Random
using CUDA
using OptimKit
using LinearAlgebra

seed = 42
Random.seed!(seed)
atype = Array
etype = Float64
D, χ_init = 2, 16
χlist_opt = [χ_init]
pattern = [1;;]
model = CubicDimer()
No = 0
folder = joinpath(pkgdir(TeneT), "data", "CubicDimer", "VUMPS_C4v", "$etype", "seed$seed")

boundary_alg = VUMPS{C4v}(ifsimple_eig=true,
                           ifparallel=false,
                           ifupdown=false,
                           forloop_iter=1,
                           maxiter=30,
                           miniter=0,
                           maxiter_ad=4,
                           miniter_ad=4,
                           power_iter=10,
                           power_iter_ad=5,
                           power_iter_obs=40,
                           show_every=10,
                           tol=1e-10,
                           verbosity=3,
)

params = GradientOptimize(model=model,
                          pattern=pattern,
                          boundary_alg=boundary_alg,
                          optimizer=LBFGS(200; maxiter=100, verbosity=4, gradtol=1e-7,
                                          linesearch=HagerZhangLineSearch(maxfg=5)),
                          forloop_iter=1,
                          verbosity=4,
                          folder=folder,
                          ifprecondition=true,
                          iter_precond=0,
                          reuse_env=true,
                          ifsave_env=true,
                          ifload_env=true,
                          ifsave_lbfgs=true,
                          ifload_lbfgs=false,
                          ifplot=false,
                          save_every=1,
                          show_every=1,
)

A = init_ipeps(; atype, etype, No, D, χ=χ_init, params)

function restriction_ipeps(A)
    return C4v_restriction(A)
end

A, entropy, grad, fgnum, history = optimise_cubic_dimer(A, χlist_opt, params; restriction_ipeps)
obs = cubic_dimer_observable(A, χ_init, params; restriction_ipeps)

@show entropy
@show norm(grad)
@show obs.entropy
@show obs.xi
@show obs.err_norm
@show obs.err_transfer
