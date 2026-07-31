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
D, chi_init = 3, 32
chilist_opt = default_χlist(D; χmin=chi_init, nstage=10)
pattern = [1;;]

# H = -J sum_<ij> sigma_i^z sigma_j^z - h sum_i sigma_i^x.
# With Pauli matrices (not spin operators), h/J = 3.0 is close to the
# square-lattice quantum critical point h/J ~ 3.044.
model = TFIsing(lattice=Square(), J=1.0, h=3.0)

No = 38
folder = joinpath(pkgdir(TeneT), "data/$model/$pattern/VUMPS_C4v/$etype/seed$seed/")
boundary_alg = VUMPS{C4v}(ifsimple_eig=true,
                          ifparallel=false,
                          ifcheckpoint=false,
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
                          optimizer=LBFGS(50;
                              maxiter=200,
                              verbosity=4,
                              gradtol=1e-7,
                              linesearch=HagerZhangLineSearch(maxfg=5),
                          ),
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
                          ifload_lbfgs=false,
)
A = init_ipeps(; atype, etype, No, D, χ=chi_init, params)

function restriction_ipeps(A)
    A = C4v_restriction(A)
    return A/norm(A)
end

optimise_ipeps(A, chilist_opt, params; restriction_ipeps)
# observable(A, chi_init, params; restriction_ipeps)
