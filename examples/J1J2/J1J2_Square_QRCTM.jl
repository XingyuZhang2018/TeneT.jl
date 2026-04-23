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
D, χ, χshift = 2, 16, 0
# pattern = [1 2;
#            2 1]
pattern = [1;;]
# pattern = [1 3;
#            2 4]
model = J1J2(lattice=Square(), 
             S=0.5, J1=1.0, J2=0.5,
             ifrotate=true, 
             couplingtype=:uniform, bondratio=1.0)
No = 0
folder = joinpath(pkgdir(TeneT), "data/$model/$pattern/QRCTM/$etype/seed$seed/")
boundary_alg = QRCTM(ifparallel=false,
                     forloop_iter=1,
                     maxiter=3, 
                     miniter=0, 
                     maxiter_ad=20,
                     miniter_ad=20,
                     show_every=10,
                     tol=1e-10,
                     verbosity=3,
)
params = GradientOptimize(model=model,
                          pattern=pattern,
                          boundary_alg=boundary_alg, 
                          optimizer=LBFGS(200; maxiter=100, verbosity=4, gradtol=1e-7, linesearch=HagerZhangLineSearch(maxfg=5)),
                          forloop_iter=1,
                          maxiter_restart=1,
                          verbosity=4, 
                          folder=folder,
                          ifSU=false,
                          SUτ=0,
                          ifprecondition=true,
                          iter_precond=0,
                          reuse_env=true, 
                          ifsave_env=false,
                          ifload_env=false,
                          ifsave_lbfgs=true,
                          ifload_lbfgs=false
)
A = init_ipeps(;atype, etype, No, D, χ, params)
# A = TeneT_demo.init_ipeps_to_D(;atype, No, D, D_new=3, params)

function restriction_ipeps(A)
   A = C4v_restriction(A)
   # A /= norm(A)
   # A = local_min_norm(A, params)
   return A
end

optimise_ipeps(A, χ, χshift, params; restriction_ipeps);
# observable(A, χ, params; restriction_ipeps)
