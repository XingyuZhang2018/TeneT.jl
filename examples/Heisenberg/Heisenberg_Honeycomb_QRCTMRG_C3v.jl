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

model = Heisenberg(lattice=Honeycomb(:c3v),
                   S=0.5, Jx=1.0, Jy=1.0, Jz=1.0,
                   ifrotate=false,
                   couplingtype=:uniform, bondratio=1.0)
No = 0
folder = joinpath(pkgdir(TeneT), "data/$model/$pattern/QRCTMRG_C3v/$etype/seed$seed/")

boundary_alg = QRCTMRG{C3v}(ifparallel=false,
                            step_checkpoint=Offload(),
                            forloop_iter=1,
                            maxiter=10, miniter=0,
                            maxiter_ad=1, miniter_ad=1,
                            show_every=5, tol=1e-10,
                            verbosity=3)

params = GradientOptimize(model=model,
                          pattern=pattern,
                          boundary_alg=boundary_alg,
                          optimizer=LBFGS(200; maxiter=1, verbosity=4, gradtol=1e-7,
                                           linesearch=HagerZhangLineSearch(maxfg=5)),
                          forloop_iter=1,
                          maxiter_restart=1,
                          verbosity=4,
                          folder=folder,
                          ifSU=false, SUτ=0,
                          ifprecondition=false, iter_precond=0,
                          reuse_env=true,
                          ifsave_env=false, ifload_env=false,
                          ifsave_lbfgs=false, ifload_lbfgs=false)

A = init_ipeps(; atype, etype, No, D, χ, params)

function restriction_ipeps(A)
    A1 = A[:,:,:,:,1]
    A1 = (A1 +
          permutedims(A1, (2, 3, 1, 4)) +
          permutedims(A1, (3, 1, 2, 4)) +
          permutedims(A1, (1, 3, 2, 4)) +
          permutedims(A1, (2, 1, 3, 4)) +
          permutedims(A1, (3, 2, 1, 4))) / 6
    A1 /= norm(A1)
    return reshape(A1, size(A))
end

optimise_ipeps(A, χ, χshift, params; restriction_ipeps)
