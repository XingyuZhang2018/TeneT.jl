using TeneT
using Random
using CUDA
using OptimKit
using LinearAlgebra
using Zygote

seed = 42
Random.seed!(seed)
atype = Array
etype = Float64
D, χ, χshift = 2, 8, 0
pattern = [1;;]
model = Heisenberg(lattice=Square(),
                   S=0.5, Jx=1.0, Jy=1.0, Jz=1.0,
                   ifrotate=true,
                   couplingtype=:uniform, bondratio=1.0)
No = 0
folder = joinpath(pkgdir(TeneT), "data/smoke_f32/$model/$pattern/VUMPS_C4v/$etype/seed$seed/")
boundary_alg = VUMPS{C4v}(ifsimple_eig=true,
                           ifparallel=false,
                           inner_checkpoint = Recompute(),
                           step_checkpoint  = Recompute(),
                           forloop_iter=1,
                           maxiter=3,
                           miniter=0,
                           maxiter_ad=4,
                           miniter_ad=4,
                           power_iter=1,
                           power_iter_ad=5,
                           power_iter_obs=40,
                           show_every=10,
                           tol=1e-10,
                           verbosity=1,
                           inner_etype=Float32,
)
params = GradientOptimize(model=model,
                          pattern=pattern,
                          boundary_alg=boundary_alg,
                          optimizer=LBFGS(200; maxiter=2, verbosity=1, gradtol=1e-7, linesearch=HagerZhangLineSearch(maxfg=5)),
                          forloop_iter=1,
                          maxiter_restart=1,
                          verbosity=1,
                          folder=folder,
                          ifSU=false,
                          SUτ=0,
                          ifprecondition=true,
                          iter_precond=0,
                          reuse_env=true,
                          ifsave_env=true,
                          ifload_env=false,
                          ifsave_lbfgs=true,
                          ifload_lbfgs=false
)
A = init_ipeps(;atype, etype, No, D, χ, params)
# A = TeneT_demo.init_ipeps_to_D(;atype, No, D, D_new=3, params)


function restriction_ipeps(A)
    return C4v_restriction(A)
end

optimise_ipeps(A, χ, χshift, params; restriction_ipeps);
# observable(A, χ, parasms; restriction_ipeps)
