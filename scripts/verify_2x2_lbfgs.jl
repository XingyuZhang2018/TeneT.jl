# Verify the 2×2 LBFGS-converged A under multiple boundary configs.
using TeneT
using JLD2
using LinearAlgebra
using OptimKit

const A_FILE = "D:/1 - research/1.26 - iPEPS_opt/TeneT.jl/.claude/worktrees/ipeps-fixedpoint-mcf/data/lbfgs_warmup_2x2/D2/ipeps/χ16/No.80.jld2"
A_raw = load(A_FILE, "bcipeps"; iotype=IOStream)
println("Loaded A shape:  ", size(A_raw))
println("Loaded A norm:   ", norm(A_raw))
println()

const pattern = [1 2; 2 1]

function eval_E(A_raw, χ; ifupdown::Bool)
    boundary_alg = VUMPS{General}(maxiter=30, miniter=1, tol=1e-10,
                                  ifupdown=ifupdown, ifsimple_eig=true,
                                  ifparallel=false, forloop_iter=1, verbosity=0)
    model = Heisenberg(lattice=Square(), S=0.5,
                       Jx=1.0, Jy=1.0, Jz=1.0, ifrotate=true,
                       couplingtype=:uniform, bondratio=1.0)
    params = GradientOptimize(model=model, pattern=pattern,
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

for χ in (8, 16), ifupdown in (false, true)
    println("--- χ=$χ, ifupdown=$ifupdown ---")
    try
        E, e_dict, err = eval_E(A_raw, χ; ifupdown=ifupdown)
        println("  VUMPS err: ", err)
        println("  E_total:   ", E)
        for k in sort(collect(keys(e_dict)))
            for site in sort(collect(keys(e_dict[k])))
                println("    $k[$site] = $(e_dict[k][site])")
            end
        end
    catch e
        println("  ERROR: ", e)
    end
    println()
end

println("Reference (LBFGS @ χ=16 ifupdown=true): E = -0.66251")
