using TeneT
using Random
using CUDA
using OptimKit
using LinearAlgebra
using Zygote

seed = 42
Random.seed!(seed)
atype = CuArray
etype = Float64
D, χ, χshift = 7, 256, 0
pattern = [1;;]
model = Heisenberg(lattice=Square(),
                   S=0.5, Jx=1.0, Jy=1.0, Jz=1.0,
                   ifrotate=true,
                   couplingtype=:uniform, bondratio=1.0)
No = 0
folder = joinpath(pkgdir(TeneT), "data/$model/$pattern/QRCTM/$etype/seed$seed/")
boundary_alg = QRCTM(ifparallel=false,
                     ifcheckpoint=false,
                     ifcheckpoint_wengert=true,
                     forloop_iter=1,
                     maxiter=30,
                     miniter=0,
                     maxiter_ad=20,
                     miniter_ad=20,
                     show_every=1,
                     tol=1e-10,
                     verbosity=3,
)
params = GradientOptimize(model=model,
                          pattern=pattern,
                          boundary_alg=boundary_alg,
                          optimizer=LBFGS(200; maxiter=2, verbosity=4, gradtol=1e-7, linesearch=HagerZhangLineSearch(maxfg=5)),
                          ifcheckpoint=false,
                          forloop_iter=1,
                          maxiter_restart=1,
                          verbosity=4,
                          folder=folder,
                          ifSU=false,
                          SUτ=0,
                          ifprecondition=true,
                          iter_precond=0,
                          reuse_env=true,
                          ifsave_env=false,
                          ifload_env=false,
                          ifsave_lbfgs=false,
                          ifload_lbfgs=false
)
A = init_ipeps(;atype, etype, No, D, χ, params)


function restriction_ipeps(A)
    return C4v_restriction(A)
end

CUDA.reclaim()
GC.gc()
t0 = time()
mem_before = CUDA.memory_info()

optimise_ipeps(A, χ, χshift, params; restriction_ipeps);

mem_peak = CUDA.memory_info()
@info "WENGERT CHECKPOINT SMOKE" wall = time() - t0 mem_before mem_peak
