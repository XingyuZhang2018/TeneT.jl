using TeneT
using Random
using CUDA
using OptimKit
using LinearAlgebra
using Zygote

seed = 44
Random.seed!(seed)
atype = Array
etype = Float64
D, χ_init = 2, 16
χlist_opt = default_χlist(D; χmin=χ_init, nstage=100)
# pattern = [1 2;
#            2 1]
# pattern = [1 3;
#            2 4]
# pattern = [1 3 5 2 4 6;
#            2 4 6 1 3 5]
pattern = [1 3 5 7  9 11;
           2 4 6 8 10 12]
model = FWavePRVB(lattice=Honeycomb(:brickwall_h), 
                  S=0.5, J1=0.5, K=1)
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
                          ifsave_env=true,
                          ifload_env=true,
                          ifsave_lbfgs=true,
                          ifload_lbfgs=true
)
A = init_ipeps(;atype, etype, No, D, χ=χ_init, params)
# A = init_ipeps_SU(; atype, No, D, D_new=3, χ=χ_init, params)
# A = init_ipeps_perturbation(;atype, No, D, D_new=4, χ=χ_init, ϵ=1e-2, params)

function restriction_ipeps(A)
   # A /= norm(A)
   A = local_min_norm(A, params)
#    B = Zygote.Buffer(A)
#    for i in 1:length(A)
#        if i in [1,6]
#            B[:,:,:,:,:,i] = A[:,:,:,:,:,1]
#        elseif i in [2,5]
#            B[:,:,:,:,:,i] = A[:,:,:,:,:,2]
#        elseif i in [3,4]
#            B[:,:,:,:,:,i] = A[:,:,:,:,:,3]
#        end
#    end
#    B = copy(B)
#    return B/norm(B)
   # return A
end

# observable(A, χ_init, params; restriction_ipeps)
optimise_ipeps(A, χlist_opt, params; restriction_ipeps);
# 
