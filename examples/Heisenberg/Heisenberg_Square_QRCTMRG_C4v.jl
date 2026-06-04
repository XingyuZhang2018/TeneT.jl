using TeneT
using Random
using CUDA
using OptimKit
using LinearAlgebra

seed = 42
Random.seed!(seed)
atype = Array
etype = Float64
D, χ, χshift = 2, 16, 0
pattern = [1;;]
model = Heisenberg(lattice=Square(),
                   S=0.5, Jx=1.0, Jy=1.0, Jz=1.0,
                   ifrotate=true,
                   couplingtype=:uniform, bondratio=1.0)
No = 0
folder = joinpath(pkgdir(TeneT), "data/$model/$pattern/QRCTMRG_C4v/$etype/seed$seed/")

boundary_alg = QRCTMRG{C4v}(ifparallel=false,
                            step_checkpoint=Offload(),
                            forloop_iter=1,
                            maxiter=30, miniter=0,
                            maxiter_ad=20, miniter_ad=20,
                            show_every=10, tol=1e-10,
                            verbosity=3)

params = GradientOptimize(model=model,
                          pattern=pattern,
                          boundary_alg=boundary_alg,
                          optimizer=LBFGS(200; maxiter=100, verbosity=4, gradtol=1e-7,
                                           linesearch=HagerZhangLineSearch(maxfg=5)),
                          forloop_iter=1,
                          maxiter_restart=10,
                          verbosity=4, folder=folder,
                          ifSU=false, SUτ=0,
                          ifprecondition=true, iter_precond=0,
                          reuse_env=true,
                          ifsave_env=false, ifload_env=false,
                          ifsave_lbfgs=false, ifload_lbfgs=false)

A = init_ipeps(; atype, etype, No, D, χ, params)

function restriction_ipeps(A)
    return C4v_restriction(A)
end

optimise_ipeps(A, χ, χshift, params; restriction_ipeps)