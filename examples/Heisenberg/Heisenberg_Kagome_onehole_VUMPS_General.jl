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
D, χ, χshift, maxiter_restart = 2, 16, 4, 10
pattern = [1 3;
           2 4]
model = Heisenberg(lattice=Kagome(:onehole),
                   S=0.5, Jx=1.0, Jy=1.0, Jz=1.0,
                   ifrotate=false,
                   couplingtype=:uniform, bondratio=1.0)
No = 0
SUτ = 0.01
folder = joinpath(pkgdir(TeneT), "data/$model/$pattern/VUMPS_General/$etype/seed$seed/")
boundary_alg = VUMPS{General}(ifupdown=true,
                               ifdownfromup=false,
                               ifsimple_eig=true,
                               ifparallelupdown=false,
                               ifparallel=false,
                               forloop_iter=5,
                               maxiter=30,
                               miniter=0,
                               maxiter_ad=4,
                               miniter_ad=4,
                               power_iter=5,
                               power_iter_ad=5,
                               power_iter_obs=40,
                               show_every=10,
                               tol=1e-10,
                               verbosity=3,
)
params = GradientOptimize(model=model,
                          pattern=pattern,
                          boundary_alg=boundary_alg,
                          optimizer=LBFGS(200; maxiter=10, verbosity=4, gradtol=1e-7, linesearch=HagerZhangLineSearch(maxfg=5)),
                          forloop_iter=1,
                          maxiter_restart=maxiter_restart,
                          verbosity=4,
                          folder=folder,
                          ifSU=false,
                          SUτ=SUτ,
                          ifprecondition=true,
                          iter_precond=0,
                          reuse_env=true,
                          ifsave_env=false,
                          ifload_env=false,
                          ifsave_lbfgs=true,
                          ifload_lbfgs=false
)
A = init_ipeps(;atype, etype, No, D, χ, params)

function restriction_ipeps(A)
   A = local_min_norm(A, params)
   return A
end

optimise_ipeps(A, χ, χshift, params; restriction_ipeps);
