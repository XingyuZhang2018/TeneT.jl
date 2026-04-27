# Inspect what eigenvalues come out of :X decompose, to diagnose why
# A_new becomes near-zero in the V sweep after H sweep + :X + MCF.

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

φ_init = TeneT.build_phi(A[1,1], A[1,1], Val(:H))
H_op = TeneT.make_H_op(rt, A, Val(:H), params; mode=:a)
N_op = TeneT.make_N_op(rt, A, Val(:H), params)
λs, φs, _ = geneigsolve(x -> (H_op(x), N_op(x)), φ_init, 1, :SR;
                         krylovdim=20, tol=1e-10, maxiter=100,
                         ishermitian=true, isposdef=true)
φ_new = φs[1]
println("φ_new norm: ", norm(φ_new))

# Replicate :X
φ_refl = permutedims(φ_new, (7, 5, 6, 8, 2, 3, 1, 4))
φ_sym  = (φ_new + φ_refl) / 2
φ_perm = permutedims(φ_sym, (1, 2, 3, 4, 7, 5, 6, 8))
M = reshape(φ_perm, 16, 16)
Msym = (M + M') / 2
F = eigen(Hermitian(Msym))
perm = sortperm(abs.(F.values), rev=true)
sorted_vals = F.values[perm]

println("Eigenvalues (sorted by abs):")
for (i, v) in enumerate(sorted_vals)
    sign_str = v >= 0 ? "+" : "-"
    println("  $i: $sign_str$(abs(v))")
end

println("\nTop $D kept: ", sorted_vals[1:D])
sqrt_kept = sqrt.(complex.(sorted_vals[1:D]))
println("sqrt(complex(kept)): ", sqrt_kept)
println("Imag parts: ", imag.(sqrt_kept))

# Run :X to get A_new
A_new, trunc_err = TeneT.decompose_phi(φ_new, Val(:H); method=:X, D_max=D)
println("\nA_new norm:    ", norm(A_new))
println("real(A_new) norm: ", norm(real.(A_new)))
println("imag(A_new) norm: ", norm(imag.(A_new)))
