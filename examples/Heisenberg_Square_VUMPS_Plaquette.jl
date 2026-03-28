using TeneT
using Random
using CUDA
using OptimKit
using LinearAlgebra
using Zygote

seed = 42
Random.seed!(seed)
atype = Array
etype = ComplexF64
D, χ, χshift = 2, 16, 0
pattern = [1 3;
           2 4]
model = Heisenberg(Square(), 0.5,-1.0,-1.0,1.0, true)
No = 0
folder = joinpath(pkgdir(TeneT), "data/$model/$pattern/VUMPS_Plaquette/$etype/seed$seed/")
boundary_alg = VUMPS{:Plaquette}(ifsimple_eig=true,
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
                          optimizer=LBFGS(200; maxiter=100, verbosity=4, gradtol=1e-7, linesearch=HagerZhangLineSearch(maxfg=5)),
                          ifcheckpoint=false,
                          forloop_iter=1,
                          maxiter_restart=1,
                          verbosity=4, 
                          folder=folder,
                          ifSU=false,
                          SUτ=0,
                          ifprecondition=true,
                          ifMCF=false,
                          iter_precond=0,
                          reuse_env=true, 
                          ifsave_env=true,
                          ifload_env=false,
                          ifsave_lbfgs=true,
                          ifload_lbfgs=false
)
A = init_ipeps(;atype, etype, No, d=2, D, χ, params)
# A = TeneT_demo.init_ipeps_to_D(;atype, No, D, D_new=3, params)


function restriction_ipeps(A)
    Ar = Zygote.Buffer(A)
    Ar[:,:,:,:,:,1] = A[:,:,:,:,:,1]
    # Ar[:,:,:,:,:,1] += permutedims(Ar[:,:,:,:,:,1],(4,3,2,1,5))

    Ar[:,:,:,:,:,2] = permutedims(Ar[:,:,:,:,:,1], (1,4,3,2,5))
    Ar[:,:,:,:,:,3] = permutedims(Ar[:,:,:,:,:,1], (3,2,1,4,5))
    Ar[:,:,:,:,:,4] = permutedims(Ar[:,:,:,:,:,1], (3,4,1,2,5))

    Ar = copy(Ar)
    return Ar
end

optimise_ipeps(A, χ, χshift, params; restriction_ipeps);
# observable(A, χ, parasms; restriction_ipeps)
