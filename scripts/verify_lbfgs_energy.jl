# Verify the loaded LBFGS-A energy under our framework's params.
# Tries multiple boundary configurations (ifupdown true vs false, χ=8 vs 16)
# to localize any discrepancy.

using TeneT
using JLD2
using LinearAlgebra
using OptimKit

const D = 2
const LBFGS_JLD2 = "D:/1 - research/1.26 - iPEPS_opt/TeneT.jl/.claude/worktrees/stupefied-cartwright/data/Heisenberg_Square(S=0.5,Jx=-1.0,Jy=-1.0,Jz=1.0,ifrotate=true,couplingtype=uniform)/[1;;]/VUMPS_General/Float64/seed42/D2/ipeps/χ16/No.20.jld2"

A_raw = load(LBFGS_JLD2, "bcipeps"; iotype=IOStream)
println("Loaded A shape:  ", size(A_raw))
println("Loaded A norm:   ", norm(A_raw))
println()

function eval_E(A_raw, χ; ifupdown::Bool)
    boundary_alg = VUMPS{General}(maxiter=30, miniter=1, tol=1e-10,
                                  ifupdown=ifupdown, ifsimple_eig=true,
                                  ifparallel=false, forloop_iter=1, verbosity=0)
    model = Heisenberg(lattice=Square(), S=0.5,
                       Jx=-1.0, Jy=-1.0, Jz=1.0, ifrotate=true,
                       couplingtype=:uniform, bondratio=1.0)
    params = GradientOptimize(model=model, pattern=ones(Int, 1, 1),
                              boundary_alg=boundary_alg, verbosity=0,
                              ifSU=false, ifprecondition=false, forloop_iter=1)
    A = TeneT.build_A(A_raw, params)

    rt = ifupdown ? TeneT.init_env(A, χ, params.boundary_alg) :
                    TeneT.init_VUMPSRuntime(A, χ, params.boundary_alg)
    rt, err = TeneT.leading_boundary(rt, A, params.boundary_alg)
    env = TeneT.ObsEnv(rt, A, params.boundary_alg)
    E, e_dict = TeneT.energy_value(params.model, A, env, params)
    return real(E), e_dict, err
end

for χ in (8, 16, 32), ifupdown in (false, true)
    println("--- χ=$χ, ifupdown=$ifupdown ---")
    try
        E, e_dict, err = eval_E(A_raw, χ; ifupdown=ifupdown)
        bH = e_dict["bond_H_energy"]["1,1"]
        bV = e_dict["bond_V_energy"]["1,1"]
        println("  VUMPS err: ", err)
        println("  E_total:   ", E)
        println("  bond_H:    ", bH)
        println("  bond_V:    ", bV)
    catch e
        println("  ERROR: ", e)
    end
    println()
end

println("Reference (LBFGS at χ=16, ifupdown=true): -0.660231093546424")
