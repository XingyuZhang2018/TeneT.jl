using TeneT
using Random
using CUDA
using OptimKit
using LinearAlgebra
using Zygote


seed = 88
Random.seed!(seed)
atype = Array
etype = Float64
D, χ, χshift, maxiter_restart = 3, 8, 1, 100
# pattern = [1 2;
#            2 1]
# pattern = [1 3;
#            2 4]
pattern = [1 3 5 2 4 6;
           2 4 6 1 3 5]
# pattern = [1 3 5 7  9 11;
#            2 4 6 8 10 12]
lattice = Honeycomb(:brickwall)
model = J1J2J3(lattice=lattice, 
               S=0.5, J1=1.0, J2=0.6, J3=0.4,
               ifrotate=true, 
               couplingtype=:plaquette, bondratio=1)
No = 0
folder = joinpath(pkgdir(TeneT), "data/$model/$pattern/VUMPS_Plaquette/$etype/seed$seed/")
boundary_alg = VUMPS(Plaquette(lattice); ifsimple_eig=true,
                                         ifparallel=false,
                                         ifcheckpoint=false,
                                         forloop_iter=1,
                                         maxiter=30, 
                                         miniter=0, 
                                         maxiter_ad=4,
                                         miniter_ad=4,
                                         power_iter=1,
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
                          ifcheckpoint=false,
                          forloop_iter=1,
                          maxiter_restart=maxiter_restart,
                          verbosity=4, 
                          folder=folder,
                          ifSU=false,
                          SUτ=0,
                          ifprecondition=true,
                          iter_precond=0,
                          reuse_env=true, 
                          ifsave_env=true,
                          ifload_env=true,
                          ifsave_lbfgs=true,
                          ifload_lbfgs=false
)
A = init_ipeps(;atype, etype, No, D, χ, params)
# A = init_ipeps_SU(; atype, No, D, D_new=3, χ, params)
# A = init_ipeps_perturbation(;atype, No, D, D_new=4, χ, ϵ=1e-2, params)

function restriction_ipeps(A)
   # A /= norm(A)
   # A = local_min_norm(A, params)
   B = Zygote.Buffer(A)
   B[:,:,:,:,:,1] = A[:,:,:,:,:,1]
   B[:,:,:,:,:,1] += permutedims(B[:,:,:,:,:,1],(4,2,3,1,5))
   
   B[:,:,:,:,:,2] = permutedims(B[:,:,:,:,:,1],(3,2,1,4,5))
   B[:,:,:,:,:,3] = B[:,:,:,:,:,1]
   B[:,:,:,:,:,4] = permutedims(B[:,:,:,:,:,1],(3,2,4,1,5))
   B[:,:,:,:,:,5] = permutedims(B[:,:,:,:,:,1],(4,2,1,3,5))
   B[:,:,:,:,:,6] = permutedims(B[:,:,:,:,:,1],(4,2,1,3,5))

   B = copy(B)
   return B/norm(B)
   # return A
end

# observable(A, 23, params; restriction_ipeps)
optimise_ipeps(A, 8, χshift, params; restriction_ipeps);
