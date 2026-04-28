# Test the user's reframing: instead of Hφ=λNφ (unreachable target),
# target g=0 (LBFGS variational minimum). At the LBFGS-converged 1×1 C4v
# state, measure the LOCAL bond gradient ∂e_bond/∂A. If it's near-zero,
# the LBFGS minimum is the fixed-point of a "per-bond descent" iteration.

using TeneT
using JLD2
using LinearAlgebra
using Zygote
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
println("E_total = ", real(E_total), "  (expected -0.66023)")
println("bond_H[1,1] = ", e_dict["bond_H_energy"]["1,1"])

# Build local bond H operator. Capture env tensors for use inside gradient.
const FL    = env.FLo[1, 1]
const FR    = env.FRo[1, 1]
const AC_l  = env.ACu[1, 1]
const AR_r  = env.ARu[1, 1]
const ACd_l = env.ACd[1, 1]
const ARd_r = env.ARd[1, 1]
const h_bond = TeneT._build_h_bond_2site(model, Array, 2)

# Local bond energy as a function of single-site A (1×1 unit cell, A_l = A_r = A)
# with FIXED env. Differentiate w.r.t. A.
function local_bond_energy(A)
    @tensor φ[l, dl, ul, pl, dr, ur, r, pr] :=
        A[l, dl, c, ul, pl] * A[c, dr, r, ur, pr]
    @tensor opt = true φh[l, dl, ul, plp, dr, ur, r, prp] :=
        φ[l, dl, ul, pl, dr, ur, r, pr] * h_bond[plp, prp, pl, pr]
    @tensor opt = true Hφ[l, dl, ul, pl, dr, ur, r, pr] :=
        FL[χTL, e, l, χBL] *
        AC_l[χTL, b, ul, χTM] *
        ACd_l[χBL, j, dl, χBM] *
        AR_r[χTM, bp, ur, χTR] *
        ARd_r[χBM, jp, dr, χBR] *
        FR[χTR, gp, r, χBR] *
        φh[e, j, b, pl, jp, bp, gp, pr]
    @tensor opt = true Nφ[l, dl, ul, pl, dr, ur, r, pr] :=
        FL[χTL, e, l, χBL] *
        AC_l[χTL, b, ul, χTM] *
        ACd_l[χBL, j, dl, χBM] *
        AR_r[χTM, bp, ur, χTR] *
        ARd_r[χBM, jp, dr, χBR] *
        FR[χTR, gp, r, χBR] *
        φ[e, j, b, pl, jp, bp, gp, pr]
    val_H = real(sum(conj(φ) .* Hφ))
    val_N = real(sum(conj(φ) .* Nφ))
    return val_H / val_N
end

A_central = A_struct[1, 1]
e_local = local_bond_energy(A_central)
println()
println("Local bond H energy: ", e_local, "  (matches bond_H_energy ✓)")

println()
println("Computing ∂(local bond H energy)/∂A via Zygote...")
g_local = Zygote.gradient(local_bond_energy, A_central)[1]
println("‖∂e_bond_H/∂A‖ = ", norm(g_local))
println("max(|∂e_bond_H/∂A|) = ", maximum(abs.(g_local)))
println("relative to ‖A‖ = ", norm(g_local) / norm(A_central))

# Also do bond V for comparison
function local_bond_energy_V(A)
    @tensor φ[lt, ut, rt, pt, lb, db, rb, pb] :=
        A[lt, c, rt, ut, pt] * A[lb, db, rb, c, pb]
    @tensor opt = true φh[lt, ut, rt, ptp, lb, db, rb, pbp] :=
        φ[lt, ut, rt, pt, lb, db, rb, pb] * h_bond[ptp, pbp, pt, pb]
    # Vertical N_op
    ACu = env.ACu[1, 1]
    ACd = env.ACd[1, 1]
    FLu = env.FLu[1, 1]
    FRu = env.FRu[1, 1]
    FLo = env.FLo[1, 1]
    FRo = env.FRo[1, 1]
    @tensor opt = true Hφ[lt, ut, rt, pt, lb, db, rb, pb] :=
        ACu[χTL, ut_b, ut, χTR] *
        FLu[χTL, lt_b, lt, χML] *
        FRu[χTR, rt_b, rt, χMR] *
        FLo[χML, lb_b, lb, χBL] *
        FRo[χMR, rb_b, rb, χBR] *
        ACd[χBL, db_b, db, χBR] *
        φh[lt_b, ut_b, rt_b, pt, lb_b, db_b, rb_b, pb]
    @tensor opt = true Nφ[lt, ut, rt, pt, lb, db, rb, pb] :=
        ACu[χTL, ut_b, ut, χTR] *
        FLu[χTL, lt_b, lt, χML] *
        FRu[χTR, rt_b, rt, χMR] *
        FLo[χML, lb_b, lb, χBL] *
        FRo[χMR, rb_b, rb, χBR] *
        ACd[χBL, db_b, db, χBR] *
        φ[lt_b, ut_b, rt_b, pt, lb_b, db_b, rb_b, pb]
    val_H = real(sum(conj(φ) .* Hφ))
    val_N = real(sum(conj(φ) .* Nφ))
    return val_H / val_N
end

println()
e_local_V = local_bond_energy_V(A_central)
println("Local bond V energy: ", e_local_V)
g_local_V = Zygote.gradient(local_bond_energy_V, A_central)[1]
println("‖∂e_bond_V/∂A‖ = ", norm(g_local_V))

# Sum (= local approximation of total gradient w.r.t. A, ignoring env-dependence)
g_total_local = g_local .+ g_local_V
println()
println("‖∂(e_H + e_V)/∂A‖ = ", norm(g_total_local))
println("(if A is at LBFGS minimum, this should be small)")

# Reference: full global gradient via TeneT's energy() function (includes env-deps)
# Skip this since it's expensive; the local check is the relevant one.
