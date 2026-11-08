#doesn't work currently, the symmetrization is not working well, and the optimization is not converging.
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
χlist_opt = [χ_init]
pattern = [1 2;
           2 1]

model = Heisenberg(lattice=Honeycomb(:c3v),
                   S=0.5, Jx=1.0, Jy=1.0, Jz=1.0,
                   ifrotate=true, 
                   couplingtype=:uniform, bondratio=1.0)
No = 0
folder = joinpath(pkgdir(TeneT), "data/$model/$pattern/QRCTMRG_C3vTwoSite/$etype/seed$seed/")

boundary_alg = QRCTMRG{C3vTwoSite}(ifparallel=false,
                                   step_checkpoint=Plain(),
                                   forloop_iter=1,
                                   maxiter=30, miniter=0,
                                   maxiter_ad=20, miniter_ad=20,
                                   show_every=5, tol=1e-10,
                                   verbosity=3)

params = GradientOptimize(model=model,
                          pattern=pattern,
                          boundary_alg=boundary_alg,
                          optimizer=LBFGS(200; maxiter=100, verbosity=4, gradtol=1e-7,
                                           linesearch=HagerZhangLineSearch(maxfg=5)),
                          forloop_iter=1,
                          verbosity=4,
                          folder=folder,
                          ifSU=false, SUτ=0,
                          ifprecondition=false, iter_precond=0,
                          reuse_env=true,
                          ifsave_env=true, ifload_env=true,
                          ifsave_lbfgs=false, ifload_lbfgs=false)

A = init_ipeps(; atype, etype, No, D, χ=χ_init, params)

function _c3v_symmetrize_site(A1)
    return (A1 +
            permutedims(A1, (2, 3, 1, 4)) +
            permutedims(A1, (3, 1, 2, 4)) +
            permutedims(A1, (1, 3, 2, 4)) +
            permutedims(A1, (2, 1, 3, 4)) +
            permutedims(A1, (3, 2, 1, 4))) / 6
end

function restriction_ipeps(A)
    B = Zygote.Buffer(A)
    A1 = _c3v_symmetrize_site(A[:,:,:,:,1])
    A1 /= norm(A1)
    A2 = _c3v_symmetrize_site(A[:,:,:,:,2])
    A2 /= norm(A2)
    B[:,:,:,:,1] = A1
    B[:,:,:,:,2] = A2
    B = copy(B)
    return B / norm(B)
end

optimise_ipeps(A, χlist_opt, params; restriction_ipeps)
