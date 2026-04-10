using Random
using CUDA
using TeneT
using OptimKit
using LinearAlgebra
using Zygote

using MPI
MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
size = MPI.Comm_size(comm)
comm_l = MPI.Comm_split_type(comm, MPI.COMM_TYPE_SHARED, rank)
rank_l = MPI.Comm_rank(comm_l)
CUDA.device!(rank_l)

println("Hostname: ", gethostname())
println("in rankl $rank_l")

if rank == 0
    ifsave_env = true
    ifsave_lbfgs = true
    verbosity_contract = 3
    verbosity_optim = 4
    ifplot = true
else
     ifsave_env = false
     ifsave_lbfgs = false
     verbosity_contract = 0
     verbosity_optim = 0
     ifplot = false
end

seed = 42
Random.seed!(seed)
atype = Array
etype = Float64
D, χ, χshift, maxiter_restart = 2, 16, 1, 10
pattern = [1 3;
           2 4]
model = J1J2(lattice=Square(), 
             S=0.5, J1=1.0, J2=0.5,
             ifrotate=true, 
             couplingtype=:uniform, bondratio=1.0)
No = 0
folder = joinpath(pkgdir(TeneT), "data/$model/$pattern/VUMPS_Plaquette/$etype/seed$seed/")
boundary_alg = VUMPS{Plaquette{Square}}(ifsimple_eig=true,
                                 ifparallel=true,
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
                                 verbosity=verbosity_contract,
)
params = GradientOptimize(model=model,
                          pattern=pattern,
                          boundary_alg=boundary_alg, 
                          optimizer=LBFGS(200; maxiter=10, verbosity=4, gradtol=1e-7, linesearch=HagerZhangLineSearch(maxfg=5)),
                          ifcheckpoint=false,
                          forloop_iter=1,
                          maxiter_restart=maxiter_restart,
                          verbosity=verbosity_optim, 
                          folder=folder,
                          ifSU=false,
                          SUτ=0,
                          ifprecondition=true,
                          iter_precond=0,
                          reuse_env=true, 
                          ifsave_env=ifsave_env,
                          ifload_env=false,
                          ifsave_lbfgs=ifsave_lbfgs,
                          ifload_lbfgs=false,
                          ifplot=ifplot
)
A = init_ipeps(;atype, etype, No, D, χ, params)
# A = TeneT_demo.init_ipeps_to_D(;atype, No, D, D_new=3, params)

function restriction_ipeps(A)
   Ar = Zygote.Buffer(A)
   Ar[:,:,:,:,:,1] = A[:,:,:,:,:,1]
   Ar[:,:,:,:,:,1] += permutedims(Ar[:,:,:,:,:,1],(4,3,2,1,5))

   Ar[:,:,:,:,:,2] = permutedims(Ar[:,:,:,:,:,1], (1,4,3,2,5))
   Ar[:,:,:,:,:,3] = permutedims(Ar[:,:,:,:,:,1], (3,2,1,4,5))
   Ar[:,:,:,:,:,4] = permutedims(Ar[:,:,:,:,:,1], (3,4,1,2,5))

   Ar = copy(Ar)
   return Ar
end

optimise_ipeps(A, χ, χshift, params; restriction_ipeps);