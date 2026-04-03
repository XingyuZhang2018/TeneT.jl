# CAPEPS: Clifford Augmented iPEPS for Kitaev Honeycomb Model
# Compares standard iPEPS vs Clifford-pre-optimized iPEPS

using TeneT
using Random
using OptimKit
using LinearAlgebra
using Printf

seed = 42
Random.seed!(seed)
atype = Array
etype = Float64
D, χ, χshift = 2, 16, 0
pattern = [1 3;
           2 4]

# ============================================================================
# Step 1: Define the Kitaev model
# ============================================================================
model = Kitaev(lattice=Honeycomb(:brickwall),
               S=0.5, Jx=-1.0, Jy=-1.0, Jz=-1.0,
               couplingtype=:uniform, bondratio=1.0)

println("="^60)
println("CAPEPS: Clifford Augmented iPEPS for Kitaev Model")
println("="^60)

# ============================================================================
# Step 2: Clifford pre-optimization
# ============================================================================
println("\n--- Phase 1: Clifford Circuit Optimization ---")
result = optimize_clifford(model; n_layers=3, max_sweeps=10, verbosity=2)

h_orig = TeneT.hamiltonian(model)
h_transformed = result[:h_transformed]

println("\nOriginal bond entanglement entropies:")
for (i, label) in enumerate(["x-bond", "y-bond", "z-bond"])
    @printf("  %s: %.6f\n", label, bond_entanglement_entropy(h_orig[i]))
end
println("Transformed bond entanglement entropies:")
for (i, label) in enumerate(["x-bond", "y-bond", "z-bond"])
    @printf("  %s: %.6f\n", label, bond_entanglement_entropy(h_transformed[i]))
end

# ============================================================================
# Step 3: iPEPS optimization with transformed Hamiltonian (CAPEPS)
# ============================================================================
println("\n--- Phase 2a: iPEPS AD Optimization (CAPEPS) ---")

tmodel = TransformedKitaev(model, h_transformed, result[:circuit])

boundary_alg = VUMPS{:General}(ifupdown=true,
                               ifdownfromup=false,
                               ifsimple_eig=true,
                               ifparallel=false,
                               ifcheckpoint=false,
                               forloop_iter=1,
                               maxiter=30,
                               miniter=0,
                               maxiter_ad=4,
                               miniter_ad=4,
                               power_iter=1,
                               power_iter_ad=5,
                               power_iter_obs=40,
                               show_every=10,
                               tol=1e-10,
                               verbosity=3)

No = 0
folder_capeps = joinpath(pkgdir(TeneT), "data/$tmodel/$pattern/VUMPS_General/$etype/seed$seed/")
params_capeps = GradientOptimize(model=tmodel,
                                 pattern=pattern,
                                 boundary_alg=boundary_alg,
                                 optimizer=LBFGS(200; maxiter=100, verbosity=4, gradtol=1e-7,
                                                 linesearch=HagerZhangLineSearch(maxfg=5)),
                                 ifcheckpoint=false,
                                 forloop_iter=1,
                                 maxiter_restart=4,
                                 verbosity=4,
                                 folder=folder_capeps,
                                 ifSU=false,
                                 SUτ=0,
                                 ifprecondition=true,
                                 iter_precond=0,
                                 reuse_env=true,
                                 ifsave_env=false,
                                 ifload_env=false,
                                 ifsave_lbfgs=true,
                                 ifload_lbfgs=false)

A_capeps = init_ipeps(;atype, etype, No, D, χ, params=params_capeps)

function restriction_ipeps(A)
    A /= norm(A)
    return A
end

optimise_ipeps(A_capeps, χ, χshift, params_capeps; restriction_ipeps)

# ============================================================================
# Step 4: Standard iPEPS optimization (baseline)
# ============================================================================
println("\n--- Phase 2b: iPEPS AD Optimization (Standard) ---")

folder_std = joinpath(pkgdir(TeneT), "data/$model/$pattern/VUMPS_General/$etype/seed$seed/")
params_std = GradientOptimize(model=model,
                              pattern=pattern,
                              boundary_alg=boundary_alg,
                              optimizer=LBFGS(200; maxiter=100, verbosity=4, gradtol=1e-7,
                                              linesearch=HagerZhangLineSearch(maxfg=5)),
                              ifcheckpoint=false,
                              forloop_iter=1,
                              maxiter_restart=4,
                              verbosity=4,
                              folder=folder_std,
                              ifSU=false,
                              SUτ=0,
                              ifprecondition=true,
                              iter_precond=0,
                              reuse_env=true,
                              ifsave_env=false,
                              ifload_env=false,
                              ifsave_lbfgs=true,
                              ifload_lbfgs=false)

A_std = init_ipeps(;atype, etype, No, D, χ, params=params_std)
optimise_ipeps(A_std, χ, χshift, params_std; restriction_ipeps)

println("\n" * "="^60)
println("Comparison complete. Check data/ folder for results.")
println("="^60)
