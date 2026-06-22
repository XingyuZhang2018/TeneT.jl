using TeneT
using Random
using CUDA
using Printf

seed = 42
Random.seed!(seed)
atype = Array
beta = 0.43
χ = 64
npts = 2000

model = Ising(lattice=Square(), beta=beta)
M = MPO(model, C4v; atype)

boundary_alg = VUMPS{C4v}(;
    ifsimple_eig = true,
    ifparallel = false,
    ifupdown = false,
    forloop_iter = 5,
    maxiter = 200,
    miniter = 1,
    maxiter_ad = 1,
    miniter_ad = 1,
    power_iter = 10,
    show_every = 10,
    tol = 1e-10,
    verbosity = 3,
)

println("=" ^ 80)
println("# 2D classical Ising square lattice: VUMPS{C4v}")
println("atype=$atype  chi=$χ  beta=$beta")
println("=" ^ 80)

rt = init_env(M, χ, boundary_alg)
t0 = time()
rt, err = leading_boundary(rt, M, boundary_alg)
elapsed = time() - t0

fr = free_energy(rt, M, boundary_alg, model)
f_exact = exact_free_energy(model; npts)

@printf("VUMPS err       = %.3e\n", err)
@printf("Z per site      = %.12f\n", fr.Z_per_site)
@printf("f_VUMPS         = %.12f\n", fr.f)
@printf("f_Onsager       = %.12f  (npts=%d)\n", f_exact, npts)
@printf("|delta f|       = %.3e\n", abs(fr.f - f_exact))
@printf("elapsed         = %.2f s\n", elapsed)
