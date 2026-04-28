# Test the user's reframing via a 5-leg generalized eigenvalue problem.
#
# Idea: instead of 8-leg geneigsolve(H_eff, N_eff, φ) which escapes the
# PEPS-rank-D manifold, run 5-leg geneigsolve with one site held fixed.
# By construction this stays in manifold. Its fixed-point is per-bond
# local g=0 (with env fixed) — the FU fixed-point.
#
# Test: at LBFGS-converged A, the 5-leg eigsolver should (approximately)
# return A itself, with eigenvalue ≈ bond energy = -0.330.

using TeneT
using JLD2
using LinearAlgebra
using KrylovKit
using TensorOperations: @tensor

const A_FILE = "D:/1 - research/1.26 - iPEPS_opt/TeneT.jl/.claude/worktrees/ipeps-fixedpoint-mcf/data/lbfgs_vumps_c4v_seed42/D2/ipeps/χ16/No.13.jld2"

A_raw = load(A_FILE, "bcipeps"; iotype=IOStream)
A_raw = C4v_restriction(A_raw)
A_raw = A_raw / norm(A_raw)
println("Loaded A, norm = ", norm(A_raw))

const D = 2
const χ = 16
const eval_alg = VUMPS{General}(maxiter=30, miniter=1, tol=1e-10,
                                ifupdown=true, ifsimple_eig=true,
                                ifparallel=false, forloop_iter=1, verbosity=0)
const model = Heisenberg(lattice=Square(), S=0.5,
                         Jx=1.0, Jy=1.0, Jz=1.0, ifrotate=true,
                         couplingtype=:uniform, bondratio=1.0)
const eval_params = GradientOptimize(model=model, pattern=ones(Int,1,1),
                                     boundary_alg=eval_alg, verbosity=0,
                                     ifSU=false, ifprecondition=false, forloop_iter=1)

A_struct = TeneT.build_A(A_raw, eval_params)
rt_eval = TeneT.init_env(A_struct, χ, eval_alg)
rt_eval, _ = TeneT.leading_boundary(rt_eval, A_struct, eval_alg)
const env = TeneT.ObsEnv(rt_eval, A_struct, eval_alg)

E_total, e_dict = TeneT.energy_value(model, A_struct, env, eval_params)
println("E_total = ", real(E_total), "  bond_H = ", e_dict["bond_H_energy"]["1,1"])

const A_old = A_struct[1, 1]
println("A_old shape: ", size(A_old))

# Env tensors for horizontal bond at site (1,1)
const FL    = env.FLo[1, 1]
const FR    = env.FRo[1, 1]
const AC_l  = env.ACu[1, 1]
const AR_r  = env.ARu[1, 1]
const ACd_l = env.ACd[1, 1]
const ARd_r = env.ARd[1, 1]
const h_bond = TeneT._build_h_bond_2site(model, Array, 2)

# 8-leg N_op (existing infrastructure)
function N_op_8leg(φ)
    @tensor opt = true Nφ[l, dl, ul, pl, dr, ur, r, pr] :=
        FL[χTL, e, l, χBL] *
        AC_l[χTL, b, ul, χTM] *
        ACd_l[χBL, j, dl, χBM] *
        AR_r[χTM, bp, ur, χTR] *
        ARd_r[χBM, jp, dr, χBR] *
        FR[χTR, gp, r, χBR] *
        φ[e, j, b, pl, jp, bp, gp, pr]
    return Nφ
end

function H_op_8leg(φ)
    @tensor opt = true φh[l, dl, ul, plp, dr, ur, r, prp] :=
        φ[l, dl, ul, pl, dr, ur, r, pr] * h_bond[plp, prp, pl, pr]
    return N_op_8leg(φh)
end

# 5-leg lift M_l: x_5leg → φ_8leg via build_phi(x, A_old, :H)
function lift_left(x)
    @tensor φ[l, dl, ul, pl, dr, ur, r, pr] :=
        x[l, dl, c, ul, pl] * A_old[c, dr, r, ur, pr]
    return φ
end

# Adjoint M_l^†: 8-leg v → 5-leg by contracting with conj(A_old) on 4 outer right-site legs
function adjoint_left(v)
    @tensor opt = true y[l, dl, c, ul, pl] :=
        v[l, dl, ul, pl, dr, ur, r, pr] * conj(A_old)[c, dr, r, ur, pr]
    return y
end

# 5-leg effective operators: H_eff_5(x) = M^† H M x, N_eff_5(x) = M^† N M x
H_eff_5(x) = adjoint_left(H_op_8leg(lift_left(x)))
N_eff_5(x) = adjoint_left(N_op_8leg(lift_left(x)))

# Sanity: <A_old | H_eff_5 | A_old> / <A_old | N_eff_5 | A_old> should equal bond_H_energy
val_H_5 = real(sum(conj(A_old) .* H_eff_5(A_old)))
val_N_5 = real(sum(conj(A_old) .* N_eff_5(A_old)))
λ_at_A_old = val_H_5 / val_N_5
println()
println("<A_old | H_eff_5 | A_old> = ", val_H_5)
println("<A_old | N_eff_5 | A_old> = ", val_N_5)
println("λ at A_old (should = bond_H = -0.330): ", λ_at_A_old)

# Test: residual ‖H_eff_5 A - λ N_eff_5 A‖ at LBFGS-converged A
HA = H_eff_5(A_old)
NA = N_eff_5(A_old)
residual_5 = HA .- λ_at_A_old .* NA
res_5_norm = norm(residual_5)
println()
println("5-leg residual ‖H_eff·A - λ·N_eff·A‖   = ", res_5_norm)
println("relative to ‖H_eff·A‖                  = ", res_5_norm / norm(HA))
println("(if small ⇒ A IS the 5-leg eigenvector ⇒ in-manifold fixed-point)")

# Run KrylovKit on 5-leg space
println()
println("Running KrylovKit geneigsolve on 5-leg space...")
λs, As, info = geneigsolve(x -> (H_eff_5(x), N_eff_5(x)), A_old, 1, :SR;
                            krylovdim=20, tol=1e-10, maxiter=100,
                            ishermitian=true, isposdef=true)
A_new = As[1]
λ_new = real(λs[1])
println("KrylovKit 5-leg λ = ", λ_new)
println("info: ", info)
overlap = abs(dot(A_old, A_new)) / (norm(A_old) * norm(A_new))
println("|<A_old, A_new>| / norms = ", overlap, "  (1 ⇒ same direction)")
println()

# Also try without isposdef in case N_eff_5 is not strictly PSD
if info.converged < 1
    println("Retrying without isposdef constraint...")
    λs2, As2, info2 = geneigsolve(x -> (H_eff_5(x), N_eff_5(x)), A_old, 1, :SR;
                                   krylovdim=20, tol=1e-10, maxiter=100,
                                   ishermitian=true, isposdef=false)
    println("Without isposdef: λ = ", real(λs2[1]), "  info: ", info2)
end
