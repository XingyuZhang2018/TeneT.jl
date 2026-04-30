# Quick inspection: generate one :onehole lattice plot to a persistent folder
# (no LBFGS, no chi-shift). Run after editing plot_obs.jl to verify visuals.
using TeneT
using Random
using OptimKit
using LinearAlgebra
Random.seed!(42)

D, χ = 2, 4
pattern = [1 3; 2 4]
mode = length(ARGS) >= 1 ? Symbol(ARGS[1]) : :onehole
@assert mode in (:onehole, :onehole_real) "mode must be :onehole or :onehole_real"
model = Heisenberg(lattice=Kagome(mode), S=0.5, Jx=1.0, Jy=1.0, Jz=1.0,
                   ifrotate=false, couplingtype=:uniform, bondratio=1.0)
folder = joinpath(@__DIR__, "..", "..", ".claude", "onehole_inspect_$(mode)")
isdir(folder) && rm(folder; recursive=true)
mkpath(folder)
boundary_alg = VUMPS{TeneT.General}(ifupdown=true, ifsimple_eig=true,
                                    maxiter=4, tol=1e-4, verbosity=0)
params = GradientOptimize(model=model, pattern=pattern, boundary_alg=boundary_alg,
                          optimizer=LBFGS(10; maxiter=1, gradtol=1e-3, verbosity=0),
                          maxiter_restart=1, verbosity=0, folder=folder,
                          ifSU=false, SUτ=0.0, ifprecondition=false,
                          reuse_env=true, ifsave_env=false, ifload_env=false,
                          ifsave_lbfgs=false, ifload_lbfgs=false)

A_raw = init_ipeps(; atype=Array, etype=Float64, No=0, D=D, χ=χ, params=params)
A = TeneT.build_A(A_raw, params)
rt = TeneT.initialize_env(A_raw, D, χ, params)
rt, _ = TeneT.leading_boundary(rt, A, params.boundary_alg)
env = TeneT.ObsEnv(rt, A, params.boundary_alg)
e, e_dict = TeneT.energy_value(model, A, env, params)
M_mean, m_dict = TeneT.magnetization_value(model, A, env, params)

println("e = ", e, "  |M| = ", M_mean)
TeneT.plot_lattice_obs(e_dict, m_dict, model.lattice, pattern;
                       save_path=folder, save_format="png", χ=χ,
                       e_scalar=real(e), mag_scalar=M_mean)
println("plot: ", joinpath(folder, "lattice_χ$χ.png"))
