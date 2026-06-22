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
D, χ_init = 2, 16
χlist_opt = default_χlist(D; χmin=χ_init, nstage=10)
pattern = [1 2;
           2 1]
# pattern = [1;;]
# pattern = [1 3;
#            2 4]
model = Heisenberg(lattice=Honeycomb(:brickwall_h),
                   S=0.5, Jx=1.0, Jy=1.0, Jz=1.0,
                   ifrotate=true,
                   couplingtype=:uniform, bondratio=1.0)
No = 0
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
A = init_ipeps(;atype, etype, No, D, χ=χ_init, params)
# A = TeneT_demo.init_ipeps_to_D(;atype, No, D, D_new=3, params)

function restriction_ipeps(A)
   # A = C4v_restriction(A)
   # A /= norm(A)
   # A = local_min_norm(A, params)
   B = Zygote.Buffer(A)
   B[:,:,:,:,:,1] = A[:,:,:,:,:,1]
   B[:,:,:,:,:,1] = (B[:,:,:,:,:,1] +
         permutedims(B[:,:,:,:,:,1], (3, 2, 4, 1, 5)) +
         permutedims(B[:,:,:,:,:,1], (4, 2, 1, 3, 5)) +
         permutedims(B[:,:,:,:,:,1], (1, 2, 4, 3, 5)) +
         permutedims(B[:,:,:,:,:,1], (3, 2, 1, 4, 5)) +
         permutedims(B[:,:,:,:,:,1], (4, 2, 3, 1, 5))) / 6
   B[:,:,:,:,:,2] = B[:,:,:,:,:,1]
   B = copy(B)
   return B/norm(B)
end

optimise_ipeps(A, χlist_opt, params; restriction_ipeps);
# observable(A, χ_init, params; restriction_ipeps)
