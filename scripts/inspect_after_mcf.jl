# Trace what happens to A after H sweep + decompose + MCF.
# Why does build_phi(A, A, :V) give zero?

using TeneT
using JLD2
using LinearAlgebra
using KrylovKit

D, χ = 2, 8
A_raw = load("D:/1 - research/1.26 - iPEPS_opt/TeneT.jl/.claude/worktrees/ipeps-fixedpoint-mcf/data/lbfgs_warmup/D2/ipeps/χ8/No.2.jld2", "bcipeps"; iotype=IOStream)
params = TeneT.make_default_params(; D=D, χ=χ)
A = TeneT.build_A(A_raw, params)
rt = TeneT.init_VUMPSRuntime(A, χ, params.boundary_alg)
rt, _ = TeneT.leading_boundary(rt, A, params.boundary_alg)

println("--- Initial ---")
println("A[1,1] norm: ", norm(A[1,1]))
println("A[1,1] sample slice [:,:,:,1,1]: ", reshape(A[1,1][:,:,:,1,1], :)[1:8])

# Manual H sweep
φ_init = TeneT.build_phi(A[1,1], A[1,1], Val(:H))
H_op = TeneT.make_H_op(rt, A, Val(:H), params; mode=:a)
N_op = TeneT.make_N_op(rt, A, Val(:H), params)
λs, φs, _ = geneigsolve(x -> (H_op(x), N_op(x)), φ_init, 1, :SR;
                         krylovdim=20, tol=1e-10, maxiter=100,
                         ishermitian=true, isposdef=true)
φ_new = φs[1]

A_new_central, trunc_err = TeneT.decompose_phi(φ_new, Val(:H); method=:X, D_max=D)
println("\n--- After :X decompose ---")
println("A_new norm: ", norm(A_new_central))
println("A_new sample slice [:,:,:,1,1]: ", reshape(A_new_central[:,:,:,1,1], :)[1:8])

# Apply MCF (mimic the driver)
A_struct = deepcopy(A)
A_struct[1, 1] = A_new_central
A_c = A_struct[1, 1]
A_c6 = reshape(A_c, size(A_c)..., 1)
A_c6_mcf = TeneT.local_min_norm(A_c6, params; ifignore_gauge=false)
A_post_mcf = reshape(A_c6_mcf, size(A_c))
A_struct[1, 1] = A_post_mcf

println("\n--- After MCF ---")
println("A norm: ", norm(A_struct[1,1]))
println("A sample slice [:,:,:,1,1]: ", reshape(A_struct[1,1][:,:,:,1,1], :)[1:8])
println("A sample slice [:,:,:,2,1]: ", reshape(A_struct[1,1][:,:,:,2,1], :)[1:8])

# Try build_phi V
φ_V_postmcf = TeneT.build_phi(A_struct[1,1], A_struct[1,1], Val(:V))
println("\nbuild_phi V (post-MCF) norm: ", norm(φ_V_postmcf))

# Driver actually does V sweep BEFORE MCF — using post-H-sweep, post-:X A
println("\n--- IMPORTANT: driver does V sweep BEFORE MCF ---")
A_post_H_only = deepcopy(A_new_central)
println("A_post_H norm: ", norm(A_post_H_only))
φ_V_pre_mcf = TeneT.build_phi(A_post_H_only, A_post_H_only, Val(:V))
println("build_phi V (post-H, pre-MCF) norm: ", norm(φ_V_pre_mcf))
println("φ_V (pre-MCF) max abs: ", maximum(abs.(φ_V_pre_mcf)))
println("φ_V (pre-MCF) min abs: ", minimum(abs.(φ_V_pre_mcf)))

# Now also re-run rt env update after H sweep changed A
A_after_H = deepcopy(A)
A_after_H[1, 1] = A_post_H_only
println("\nNow re-converging rt for A_after_H (env_mode=:A in cfg)...")
rt_new, _ = TeneT.leading_boundary(rt, A_after_H, params.boundary_alg)
println("rt re-conv done. ACu[1,1] norm: ", norm(rt_new.AL[1,1]))

# Now try V sweep manually
H_op_V = TeneT.make_H_op(rt_new, A_after_H, Val(:V), params; mode=:a)
N_op_V = TeneT.make_N_op(rt_new, A_after_H, Val(:V), params)
println("Try V sweep N_op on φ_V_pre_mcf:")
try
    Nφ = N_op_V(φ_V_pre_mcf)
    println("  N_op result norm: ", norm(Nφ))
    val_N = real(sum(conj(φ_V_pre_mcf) .* Nφ))
    println("  <φ|N|φ>: ", val_N)
catch e
    println("  ERROR: ", e)
end
