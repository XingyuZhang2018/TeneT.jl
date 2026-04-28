# Diagnostic: 1×1 Heisenberg C4v LBFGS to convergence, then measure
# the eigenvalue-residual ‖Hφ - λ·Nφ‖ at the converged A.
#
# Logic: at a LBFGS variational minimum, we have ∂E/∂A = 0 in the PEPS
# manifold. The framework's fixed-point requires Hφ = λ·Nφ exactly
# (eigenvector). If LBFGS-A satisfies Hφ ≈ λ·Nφ within reasonable tol,
# then the two fixed-points coincide and the framework should iterate
# stably from this state. If not, it means the framework's geneigsolve
# eigenvector is in a direction NOT reachable by PEPS-rank-D — which is
# the "manifold escape" risk from the design doc.

using TeneT
using JLD2
using Random
using LinearAlgebra
using OptimKit
using KrylovKit
using TensorOperations: @tensor

Random.seed!(42)

const D = 2
const χ = 16   # use χ=16 for robust LBFGS convergence
const atype = Array
const etype = Float64

const boundary_alg = VUMPS{General}(maxiter=30, miniter=1, tol=1e-10,
                                    ifupdown=true, ifsimple_eig=true,
                                    ifparallel=false, forloop_iter=1,
                                    verbosity=0, show_every=10)

const model = Heisenberg(lattice=Square(), S=0.5,
                         Jx=1.0, Jy=1.0, Jz=1.0, ifrotate=true,
                         couplingtype=:uniform, bondratio=1.0)

const folder = joinpath(@__DIR__, "..", "data", "lbfgs_c4v_warmup")
mkpath(folder)

const params = GradientOptimize(
    model = model,
    pattern = ones(Int, 1, 1),
    boundary_alg = boundary_alg,
    optimizer = LBFGS(20; maxiter=80, verbosity=2, gradtol=1e-7,
                      linesearch=HagerZhangLineSearch(maxfg=8)),
    forloop_iter = 1,
    maxiter_restart = 1,
    verbosity = 3,
    folder = folder,
    ifSU = false,
    SUτ = 0,
    ifprecondition = false,
    iter_precond = 0,
    reuse_env = true,
    ifsave_env = false,
    ifload_env = false,
    ifsave_lbfgs = false,
    ifload_lbfgs = false,
)

function restriction_ipeps(A)
    A = C4v_restriction(A)
    A = local_min_norm(A, params)
    return A
end

const SKIP_LBFGS = isfile(joinpath(folder, "D$D", "ipeps", "χ$χ", "No.2.jld2"))
if !SKIP_LBFGS
    println("="^60)
    println("Step 1: LBFGS warmup at 1×1 with C4v restriction")
    println("        D=$D, χ=$χ, model=$model")
    println("="^60)

    A = init_ipeps(; atype=atype, etype=etype, No=0, D=D, χ=χ, params=params)
    println("Initial A norm: ", norm(A))

    const t0 = time()
    optimise_ipeps(A, χ, 0, params; restriction_ipeps)
    println("LBFGS done in $(round(time()-t0, digits=1)) sec")
    println()
else
    println("Skipping LBFGS (output already exists at $folder)")
end

# Find the converged jld2 (highest No.X)
const ipeps_dir = joinpath(folder, "D$D", "ipeps", "χ$χ")
files = filter(f -> startswith(f, "No.") && endswith(f, ".jld2"), readdir(ipeps_dir))
isempty(files) && error("No iPEPS jld2 in $ipeps_dir")
nums = [parse(Int, replace(replace(f, "No." => ""), ".jld2" => "")) for f in files]
last_no = maximum(nums)
const A_FILE = joinpath(ipeps_dir, "No.$(last_no).jld2")
println("Loading converged A from: $A_FILE")
A_raw = load(A_FILE, "bcipeps"; iotype=IOStream)
A_raw = restriction_ipeps(A_raw)   # ensure C4v + MCF state at the point we test
println("Converged A norm (post-C4v+MCF): ", norm(A_raw))

println()
println("="^60)
println("Step 2: Measure eigenvalue residual ‖Hφ - λ·Nφ‖ at converged A")
println("="^60)

# Build env at the SAME χ as LBFGS used
A_struct = TeneT.build_A(A_raw, params)
rt = TeneT.init_env(A_struct, χ, params.boundary_alg)
rt, vumps_err = TeneT.leading_boundary(rt, A_struct, params.boundary_alg)
println("VUMPS err: ", vumps_err)

env = TeneT.ObsEnv(rt, A_struct, params.boundary_alg)
E_total, e_dict = TeneT.energy_value(params.model, A_struct, env, params)
println("E_total (per site): ", real(E_total))
println("bond_H[1,1]: ", e_dict["bond_H_energy"]["1,1"])
println("bond_V[1,1]: ", e_dict["bond_V_energy"]["1,1"])

# Build φ on horizontal bond
φ = TeneT.build_phi(A_struct[1, 1], A_struct[1, 1], Val(:H))
println("φ shape: ", size(φ))
println("φ norm:  ", norm(φ))

# Need to make_N_op / make_H_op accept Tuple rt. The current signature is
# `rt::VUMPSRuntime`. Inline the contraction here using the env (works for
# both single and Tuple rt since env is always VUMPSEnv).

FL    = env.FLo[1, 1]
FR    = env.FRo[1, 1]
AC_l  = env.ACu[1, 1]
AR_r  = env.ARu[1, 1]
ACd_l = env.ACd[1, 1]
ARd_r = env.ARd[1, 1]

function N_op(φ)
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

# Build h (4-leg bond H) using the framework's helper
const h_bond = TeneT._build_h_bond_2site(model, atype, 2)

function H_op(φ)
    @tensor opt = true φh[l, dl, ul, plp, dr, ur, r, prp] :=
        φ[l, dl, ul, pl, dr, ur, r, pr] * h_bond[plp, prp, pl, pr]
    return N_op(φh)
end

# Compute Hφ, Nφ, eigenvalue λ ≈ <φ|H|φ>/<φ|N|φ>, residual
Hφ = H_op(φ)
Nφ = N_op(φ)

val_H = real(sum(conj(φ) .* Hφ))
val_N = real(sum(conj(φ) .* Nφ))
λ_est = val_H / val_N
println()
println("<φ|H|φ>            = ", val_H)
println("<φ|N|φ>            = ", val_N)
println("λ_est ≈ <H>/<N>    = ", λ_est)
println("(should match bond_H_energy = $(e_dict["bond_H_energy"]["1,1"]))")
println()

# Residual norm — measures how far φ is from being an eigenvector of (H, N)
residual = Hφ .- λ_est .* Nφ
res_norm = norm(residual)
ratio_to_Hφ = res_norm / norm(Hφ)
ratio_to_Nφ = res_norm / norm(Nφ)
println("Residual ‖Hφ - λ·Nφ‖ = ", res_norm)
println("relative to ‖Hφ‖     = ", ratio_to_Hφ, "  (small ⇒ near eigenvector)")
println("relative to ‖Nφ‖     = ", ratio_to_Nφ)
println()

# Compare λ_est to what geneigsolve would actually find
println("Step 3: What does geneigsolve actually return at this φ_init?")
λs, φs, info = geneigsolve(x -> (H_op(x), N_op(x)), φ, 1, :SR;
                            krylovdim=20, tol=1e-10, maxiter=100,
                            ishermitian=true, isposdef=true)
λ_kry = real(λs[1])
φ_kry = φs[1]
println("KrylovKit λ:        ", λ_kry)
println("KrylovKit info:     ", info)
overlap = abs(dot(φ, φ_kry)) / (norm(φ) * norm(φ_kry))
println("|<φ_LBFGS, φ_kry>| / norms = ", overlap, "  (1 ⇒ same direction; near-zero ⇒ moved away)")

println()
println("="^60)
println("INTERPRETATION:")
println("  - If ratio_to_Hφ is small (< ~1e-3): LBFGS-A is a fixed-point")
println("    of geneigsolve too — framework's manifold escape is a non-issue")
println("    when starting from a converged state.")
println("  - If ratio_to_Hφ is O(1): geneigsolve would move φ to a direction")
println("    NOT in PEPS-rank-D manifold — confirming fundamental escape.")
println("  - The overlap with KrylovKit's φ_kry tells the same story.")
println("="^60)
