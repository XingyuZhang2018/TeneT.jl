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

# Keep the default light enough to run through optimisation and observables.
# Increase chi_list_opt and LBFGS maxiter for production scans.
D, chi_init = 2, 16
chi_list_opt = [chi_init]
pattern = [1;;]
# pattern = [1 2;
#            2 1]
# pattern = [1 3;
#            2 4]

# Honeycomb(:merge) stores two honeycomb physical sites in one square iPEPS
# tensor with physical dimension d^2. The merge energy path currently supports
# uniform couplings and no sublattice rotation.
model = J1J2(lattice=Honeycomb(:merge),
             S=0.5, J1=1.0, J2=0.5,
             ifrotate=false,
             couplingtype=:uniform, bondratio=1.0)
No = 0
SUtau = 0
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
                          optimizer=LBFGS(200; maxiter=1, verbosity=4, gradtol=1e-7, linesearch=HagerZhangLineSearch(maxfg=1)),
                          forloop_iter=1,
                          verbosity=4,
                          folder=folder,
                          ifSU=false,
                          SUτ=SUtau,
                          ifprecondition=true,
                          iter_precond=0,
                          reuse_env=true,
                          ifsave_env=false,
                          ifload_env=false,
                          ifsave_lbfgs=false,
                          ifload_lbfgs=false,
                          ifplot=true
)
A = init_ipeps(;atype, etype, No, D, χ=chi_init, params)
# A = init_ipeps_perturbation(;atype, No, D, D_new=3, χ=chi_init, ε=1e-2, params)
# A = init_ipeps_SU(; atype, No, D, D_new=3, χ=chi_init, params)

function restriction_ipeps(A)
    A = local_min_norm(A, params)
    return A
end

optimise_ipeps(A, chi_list_opt, params; restriction_ipeps)
# observable(A, chi_init, params; restriction_ipeps)
