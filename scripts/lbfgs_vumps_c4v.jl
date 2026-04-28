# LBFGS with VUMPS{C4v} boundary alg + ifprecondition=true.
# This is the user's recommended setup that should produce a properly
# converged 1×1 C4v iPEPS, then we measure the eigenvalue residual.

using TeneT
using Random
using LinearAlgebra
using OptimKit
using JLD2
using KrylovKit
using TensorOperations: @tensor

const seed = 42
Random.seed!(seed)
const atype = Array
const etype = Float64
const D, χ, χshift = 2, 16, 0
const pattern = [1;;]

const model = Heisenberg(lattice=Square(),
                         S=0.5, Jx=1.0, Jy=1.0, Jz=1.0,
                         ifrotate=true,
                         couplingtype=:uniform, bondratio=1.0)
const No = 0
const folder = joinpath(@__DIR__, "..", "data",
                        "lbfgs_vumps_c4v_seed$(seed)")
mkpath(folder)

const boundary_alg = VUMPS{C4v}(ifsimple_eig=true,
                                ifparallel=false,
                                inner_checkpoint = Recompute(),
                                step_checkpoint  = Recompute(),
                                forloop_iter=1,
                                maxiter=3,
                                miniter=0,
                                maxiter_ad=4,
                                miniter_ad=4,
                                power_iter=1,
                                power_iter_ad=5,
                                power_iter_obs=40,
                                show_every=10,
                                tol=1e-10,
                                verbosity=0)

const params = GradientOptimize(
    model = model,
    pattern = pattern,
    boundary_alg = boundary_alg,
    optimizer = LBFGS(200; maxiter=200, verbosity=2, gradtol=1e-7,
                      linesearch=HagerZhangLineSearch(maxfg=5)),
    forloop_iter = 1,
    maxiter_restart = 1,
    verbosity = 3,
    folder = folder,
    ifSU = false,
    SUτ = 0,
    ifprecondition = true,
    iter_precond = 0,
    reuse_env = true,
    ifsave_env = true,
    ifload_env = false,
    ifsave_lbfgs = true,
    ifload_lbfgs = false,
)

println("="^60)
println("LBFGS with VUMPS{C4v} + preconditioning")
println("D=$D, χ=$χ, model=$model, seed=$seed")
println("="^60)

# Skip if already done
const SAVED_FILE = joinpath(folder, "D$D", "ipeps", "χ$χ")
if isdir(SAVED_FILE) && length(filter(f -> startswith(f, "No.") && endswith(f, ".jld2"), readdir(SAVED_FILE))) > 5
    println("Saved iPEPS already exists; skipping LBFGS.")
else
    A0 = init_ipeps(; atype=atype, etype=etype, No=No, D=D, χ=χ, params=params)
    println("Initial A norm: ", norm(A0))

    function restriction_ipeps(A)
        return C4v_restriction(A)
    end

    t_start = time()
    optimise_ipeps(A0, χ, χshift, params; restriction_ipeps)
    println("LBFGS done in $(round(time()-t_start, digits=1)) sec")
end

# Find the latest converged jld2
const ipeps_dir = joinpath(folder, "D$D", "ipeps", "χ$χ")
files = filter(f -> startswith(f, "No.") && endswith(f, ".jld2"), readdir(ipeps_dir))
isempty(files) && error("No iPEPS jld2 in $ipeps_dir")
nums = [parse(Int, replace(replace(f, "No." => ""), ".jld2" => "")) for f in files]
last_no = maximum(nums)
const A_FILE = joinpath(ipeps_dir, "No.$(last_no).jld2")
println()
println("Loading converged A from: $A_FILE")
A_raw = load(A_FILE, "bcipeps"; iotype=IOStream)
A_raw = C4v_restriction(A_raw)   # ensure C4v
A_raw = A_raw / norm(A_raw)
println("Converged A norm: ", norm(A_raw))

println()
println("="^60)
println("Step 2: Measure residual ‖Hφ - λ·Nφ‖ at converged A")
println("        (using VUMPS{General} for env so make_N_op works)")
println("="^60)

# Use VUMPS{General} for the env evaluation since make_N_op expects VUMPSEnv.
const eval_alg = VUMPS{General}(maxiter=30, miniter=1, tol=1e-10,
                                ifupdown=true, ifsimple_eig=true,
                                ifparallel=false, forloop_iter=1, verbosity=0)
const eval_params = GradientOptimize(model=model, pattern=pattern,
                                     boundary_alg=eval_alg, verbosity=0,
                                     ifSU=false, ifprecondition=false, forloop_iter=1)

A_struct = TeneT.build_A(A_raw, eval_params)
rt_eval = TeneT.init_env(A_struct, χ, eval_alg)
rt_eval, vumps_err = TeneT.leading_boundary(rt_eval, A_struct, eval_alg)
println("VUMPS err (eval): ", vumps_err)

env = TeneT.ObsEnv(rt_eval, A_struct, eval_alg)
E_total, e_dict = TeneT.energy_value(eval_params.model, A_struct, env, eval_params)
println("E_total (per site): ", real(E_total))
println("bond_H[1,1]: ", e_dict["bond_H_energy"]["1,1"])
println("bond_V[1,1]: ", e_dict["bond_V_energy"]["1,1"])

φ = TeneT.build_phi(A_struct[1,1], A_struct[1,1], Val(:H))
println("φ norm: ", norm(φ))

# Inline the same contraction my make_N_op does
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

const h_bond = TeneT._build_h_bond_2site(model, atype, 2)

function H_op(φ)
    @tensor opt = true φh[l, dl, ul, plp, dr, ur, r, prp] :=
        φ[l, dl, ul, pl, dr, ur, r, pr] * h_bond[plp, prp, pl, pr]
    return N_op(φh)
end

Hφ = H_op(φ)
Nφ = N_op(φ)

val_H = real(sum(conj(φ) .* Hφ))
val_N = real(sum(conj(φ) .* Nφ))
λ_est = val_H / val_N
println()
println("<φ|H|φ>          = ", val_H)
println("<φ|N|φ>          = ", val_N)
println("λ_est = <H>/<N>  = ", λ_est)
println("(should match bond_H_energy = $(e_dict["bond_H_energy"]["1,1"]))")
println()

residual = Hφ .- λ_est .* Nφ
res_norm = norm(residual)
ratio_to_Hφ = res_norm / norm(Hφ)
println("Residual ‖Hφ - λ·Nφ‖ = ", res_norm)
println("relative to ‖Hφ‖     = ", ratio_to_Hφ)
println()

println("Step 3: KrylovKit eigenvalue + overlap")
λs, φs, info = geneigsolve(x -> (H_op(x), N_op(x)), φ, 1, :SR;
                            krylovdim=20, tol=1e-10, maxiter=100,
                            ishermitian=true, isposdef=true)
λ_kry = real(λs[1])
φ_kry = φs[1]
println("KrylovKit λ:        ", λ_kry)
overlap = abs(dot(φ, φ_kry)) / (norm(φ) * norm(φ_kry))
println("|<φ_LBFGS, φ_kry>| / norms = ", overlap)
