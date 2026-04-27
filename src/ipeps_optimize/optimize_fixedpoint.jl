# iPEPS fixed-point eigenvalue iteration with MCF gauge fix.
# PoC for the alternative to gradient-based optimization.
# See docs/plans/2026-04-27-ipeps-fixedpoint-mcf-design.md.

export iPEPSFixedPointConfig, optimize_ipeps_fixedpoint

"""
    iPEPSFixedPointConfig

Configuration for the fixed-point eigenvalue iteration optimizer.

Fields:
- `env_mode`           — `:A` full VUMPS, `:B` warm restart, `:C` 1-step
- `env_warm_steps`     — for `:B`, number of warm-restart VUMPS steps
- `H_eff_mode`         — `:a` h on central bond only, `:b` all 7 bonds (v2)
- `decompose_method`   — `:X` symmetrize-first, `:Y` avg, `:Z` lazy SVD
- `geneig_*`           — KrylovKit.geneigsolve knobs
- `mcf_ifignore_gauge` — pass to `local_min_norm`; if `true`, MCF disabled (control)
- `outer_maxiter` / `outer_tol_λ` / `outer_tol_A` — outer-loop convergence
- `log_every` / `save_every` — logging cadence
"""
@kwdef mutable struct iPEPSFixedPointConfig
    env_mode::Symbol           = :A
    env_warm_steps::Int        = 5
    H_eff_mode::Symbol         = :a
    decompose_method::Symbol   = :X
    geneig_krylovdim::Int      = 20
    geneig_tol::Float64        = 1e-10
    geneig_maxiter::Int        = 100
    mcf_ifignore_gauge::Bool   = false
    outer_maxiter::Int         = 200
    outer_tol_λ::Float64       = 1e-8
    outer_tol_A::Float64       = 1e-7
    log_every::Int             = 1
    save_every::Int            = 10
end

# build_phi, make_N_op, make_H_op, decompose_phi, sweep_bond, optimize_ipeps_fixedpoint
# are added in subsequent tasks.

# ============================================================================
# build_phi: form 2-site bond tensor from two 5-leg iPEPS tensors
# A is (l, d, r, u, p). φ has 8 legs.
# ============================================================================

"""
    build_phi(A_l, A_r, ::Val{:H})

Form a 2-site horizontal-bond tensor `φ` by contracting the right leg of `A_l`
with the left leg of `A_r`.

Output legs: `(l, d_l, u_l, p_l, d_r, u_r, r, p_r)`.
"""
function build_phi(A_l, A_r, ::Val{:H})
    @tensor φ[l, dl, ul, pl, dr, ur, r, pr] :=
        A_l[l, dl, c, ul, pl] * A_r[c, dr, r, ur, pr]
    return φ
end

"""
    build_phi(A_t, A_b, ::Val{:V})

Form a 2-site vertical-bond tensor `φ` by contracting the down leg of `A_t`
with the up leg of `A_b`.

Output legs: `(l_t, u_t, r_t, p_t, l_b, d_b, r_b, p_b)`.
"""
function build_phi(A_t, A_b, ::Val{:V})
    @tensor φ[lt, ut, rt, pt, lb, db, rb, pb] :=
        A_t[lt, c, rt, ut, pt] * A_b[lb, db, rb, c, pb]
    return φ
end

# ============================================================================
# make_default_params: minimal GradientOptimize for testing make_N_op / make_H_op
# ============================================================================

"""
    make_default_params(; D, χ)

Build a minimal `GradientOptimize` params object suitable for fixed-point
PoC testing on a 1×1-cell rotated AFM Heisenberg model. χ is consumed at
runtime by `init_VUMPSRuntime`/`init_ipeps`; the boundary_alg itself stores
no χ field.
"""
function make_default_params(; D::Int, χ::Int)
    boundary_alg = VUMPS{General}(maxiter=20, miniter=1, tol=1e-10,
                                  ifupdown=false,
                                  ifsimple_eig=true,
                                  ifparallel=false,
                                  forloop_iter=1,
                                  verbosity=0)
    model = Heisenberg(lattice=Square(), S=0.5,
                       Jx=-1.0, Jy=-1.0, Jz=1.0,
                       ifrotate=true,
                       couplingtype=:uniform, bondratio=1.0)
    return GradientOptimize(model=model,
                            pattern=ones(Int, 1, 1),
                            boundary_alg=boundary_alg,
                            verbosity=0,
                            ifSU=false,
                            ifprecondition=false,
                            forloop_iter=1)
end

# ============================================================================
# make_N_op: 2-site norm operator on horizontal / vertical bond
# ============================================================================

"""
    make_N_op(rt, A, ::Val{:H}, params) -> Function

Return `N_op(φ) -> Nφ` applying the 2-site horizontal-bond norm operator.
For 1×1 unit cell: env tensors are FLo[1,1], FRo[1,1], ACu[1,1], ARu[1,1],
ACd[1,1], ARd[1,1].

Convention (matching `Mumap`/`FLmap(::leg5)`): the input φ is plugged into
the M1 position (env leg-2), the output Nφ dangles at M2 position (env leg-3).
Physical legs (pl, pr) pass through. Leg ordering of φ matches `build_phi`:
`(l, dl, ul, pl, dr, ur, r, pr)`.

Consistency: `dot(conj(φ), N_op(φ)) ≈ contract_n_12(...)` when
`φ = build_phi(A, A, Val(:H))` and the env tensors come from a converged
VUMPS environment around the same A.
"""
function make_N_op(rt::VUMPSRuntime, A, ::Val{:H}, params)
    env = ObsEnv(rt, A, params.boundary_alg)
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
    return N_op
end

"""
    make_N_op(rt, A, ::Val{:V}, params) -> Function

Vertical 2-site bond version of `make_N_op`. Env tensors: ACu (top),
ACd (bot), FLu/FRu (upper-row left/right), FLo/FRo (lower-row).
φ legs match `build_phi(_, _, Val(:V))`: `(lt, ut, rt, pt, lb, db, rb, pb)`.
"""
function make_N_op(rt::VUMPSRuntime, A, ::Val{:V}, params)
    env = ObsEnv(rt, A, params.boundary_alg)
    ACu = env.ACu[1, 1]
    ACd = env.ACd[1, 1]
    FLu = env.FLu[1, 1]
    FRu = env.FRu[1, 1]
    FLo = env.FLo[1, 1]
    FRo = env.FRo[1, 1]

    function N_op(φ)
        @tensor opt = true Nφ[lt, ut, rt, pt, lb, db, rb, pb] :=
            ACu[χTL, ut_b, ut, χTR] *
            FLu[χTL, lt_b, lt, χML] *
            FRu[χTR, rt_b, rt, χMR] *
            FLo[χML, lb_b, lb, χBL] *
            FRo[χMR, rb_b, rb, χBR] *
            ACd[χBL, db_b, db, χBR] *
            φ[lt_b, ut_b, rt_b, pt, lb_b, db_b, rb_b, pb]
        return Nφ
    end
    return N_op
end

# ============================================================================
# decompose_phi: φ → A_new via SVD/eigh + truncation
# ============================================================================

"""
    decompose_phi(φ, dir::Val; method=:X, D_max::Int) -> (A_new, trunc_err)

Decompose a 2-site bond tensor `φ` into an updated 1-site iPEPS tensor `A_new`
of shape `(D, D, D, D, d)` (legs `l, d, r, u, p`), truncating the central
bond to dimension `D_max`. Returns truncation error (sum of squared discarded
singular/eigenvalues).

Methods:
- `:Z` lazy:   SVD φ across central bipartition, take left factor as A_new.
- `:Y` avg:    SVD then average left and right factors.
- `:X` symmetrize-first: enforce reflection symmetry on φ, eigendecompose, take left factor.

`dir = Val(:H)` for horizontal bond (φ legs `(l, dl, ul, pl, dr, ur, r, pr)`),
`Val(:V)` for vertical (φ legs `(lt, ut, rt, pt, lb, db, rb, pb)`).
"""
function decompose_phi(φ, dir::Val; method::Symbol=:X, D_max::Int)
    if method == :Z
        return _decompose_phi_lazy(φ, dir, D_max)
    elseif method == :Y
        return _decompose_phi_avg(φ, dir, D_max)
    elseif method == :X
        return _decompose_phi_symmetrize(φ, dir, D_max)
    else
        error("Unknown decompose method: $method")
    end
end

# ----- :Z lazy SVD ------------------------------------------------------

function _decompose_phi_lazy(φ, ::Val{:H}, D_max::Int)
    Dl, Ddl, Dul, dl_sz, Ddr, Dur, Dr, dr_sz = size(φ)
    M = reshape(φ, Dl * Ddl * Dul * dl_sz, Ddr * Dur * Dr * dr_sz)
    U, S, _ = svd(M)
    keep = min(D_max, length(S))
    trunc_err = sum(abs2, @view S[keep+1:end])
    Ut = U[:, 1:keep] .* sqrt.(S[1:keep]')   # (left_dim, keep)
    # axes (l, dl, ul, pl, central) → (l, d=dl, r=central, u=ul, p=pl)
    A = reshape(Ut, Dl, Ddl, Dul, dl_sz, keep)
    return permutedims(A, (1, 2, 5, 3, 4)), trunc_err
end

function _decompose_phi_lazy(φ, ::Val{:V}, D_max::Int)
    Dlt, Dut, Drt, pt_sz, Dlb, Ddb, Drb, pb_sz = size(φ)
    M = reshape(φ, Dlt * Dut * Drt * pt_sz, Dlb * Ddb * Drb * pb_sz)
    U, S, _ = svd(M)
    keep = min(D_max, length(S))
    trunc_err = sum(abs2, @view S[keep+1:end])
    Ut = U[:, 1:keep] .* sqrt.(S[1:keep]')
    # axes (lt, ut, rt, pt, central) → (l=lt, d=central, r=rt, u=ut, p=pt)
    A = reshape(Ut, Dlt, Dut, Drt, pt_sz, keep)
    return permutedims(A, (1, 5, 3, 2, 4)), trunc_err
end

# ----- :Y avg ------------------------------------------------------------

function _decompose_phi_avg(φ, ::Val{:H}, D_max::Int)
    Dl, Ddl, Dul, dl_sz, Ddr, Dur, Dr, dr_sz = size(φ)
    M = reshape(φ, Dl * Ddl * Dul * dl_sz, Ddr * Dur * Dr * dr_sz)
    U, S, V = svd(M)
    keep = min(D_max, length(S))
    trunc_err = sum(abs2, @view S[keep+1:end])
    Ut = U[:, 1:keep] .* sqrt.(S[1:keep]')             # (left_dim, keep)
    Vt = sqrt.(S[1:keep]) .* V[:, 1:keep]'              # (keep, right_dim)

    # A_left from left factor, axes (l, dl, ul, pl, central) → (l, d, r=central, u, p)
    A_left  = permutedims(reshape(Ut, Dl, Ddl, Dul, dl_sz, keep), (1, 2, 5, 3, 4))
    # A_right from right factor, axes (central, dr, ur, r, pr) → (l=central, d=dr, r, u=ur, p=pr)
    A_right = permutedims(reshape(Vt, keep, Ddr, Dur, Dr, dr_sz), (1, 2, 4, 3, 5))
    @assert size(A_left) == size(A_right) ":Y horizontal requires symmetric leg dims"
    return (A_left .+ A_right) ./ 2, trunc_err
end

function _decompose_phi_avg(φ, ::Val{:V}, D_max::Int)
    Dlt, Dut, Drt, pt_sz, Dlb, Ddb, Drb, pb_sz = size(φ)
    M = reshape(φ, Dlt * Dut * Drt * pt_sz, Dlb * Ddb * Drb * pb_sz)
    U, S, V = svd(M)
    keep = min(D_max, length(S))
    trunc_err = sum(abs2, @view S[keep+1:end])
    Ut = U[:, 1:keep] .* sqrt.(S[1:keep]')
    Vt = sqrt.(S[1:keep]) .* V[:, 1:keep]'

    # A_top from top factor, axes (lt, ut, rt, pt, central) → (l=lt, d=central, r=rt, u=ut, p=pt)
    A_top    = permutedims(reshape(Ut, Dlt, Dut, Drt, pt_sz, keep), (1, 5, 3, 2, 4))
    # A_bot from bot factor, axes (central, lb, db, rb, pb) → (l=lb, d=db, r=rb, u=central, p=pb)
    A_bottom = permutedims(reshape(Vt, keep, Dlb, Ddb, Drb, pb_sz), (2, 3, 4, 1, 5))
    @assert size(A_top) == size(A_bottom) ":Y vertical requires symmetric leg dims"
    return (A_top .+ A_bottom) ./ 2, trunc_err
end

# ----- :X symmetrize-first -----------------------------------------------

function _decompose_phi_symmetrize(φ, ::Val{:H}, D_max::Int)
    # Reflection (swap L and R sites): permutation (7, 5, 6, 8, 2, 3, 1, 4)
    φ_refl = permutedims(φ, (7, 5, 6, 8, 2, 3, 1, 4))
    φ_sym  = (φ .+ φ_refl) ./ 2

    # Reorder right half so M = reshape(φ, left, right) is symmetric:
    # left = (l, dl, ul, pl), right = (r, dr, ur, pr) which in φ are at (7, 5, 6, 8).
    φ_perm = permutedims(φ_sym, (1, 2, 3, 4, 7, 5, 6, 8))
    Dl, Ddl, Dul, dl_sz, Dr, Ddr, Dur, dr_sz = size(φ_perm)
    @assert (Dl, Ddl, Dul, dl_sz) == (Dr, Ddr, Dur, dr_sz) ":X horizontal requires L/R-symmetric leg dims"

    M = reshape(φ_perm, Dl * Ddl * Dul * dl_sz, Dr * Ddr * Dur * dr_sz)
    Msym = (M .+ M') ./ 2
    F = eigen(Hermitian(Msym))
    perm = sortperm(abs.(F.values), rev=true)
    vals = F.values[perm]
    vecs = F.vectors[:, perm]
    keep = min(D_max, length(vals))
    trunc_err = sum(abs2, @view vals[keep+1:end])
    Σ_kept = sqrt.(complex.(vals[1:keep]))
    Ut = vecs[:, 1:keep] .* Σ_kept'
    A = reshape(Ut, Dl, Ddl, Dul, dl_sz, keep)
    A_new = permutedims(A, (1, 2, 5, 3, 4))
    if eltype(φ) <: Real
        A_new = real(A_new)
    end
    return A_new, real(trunc_err)
end

function _decompose_phi_symmetrize(φ, ::Val{:V}, D_max::Int)
    # Reflection (swap top and bot sites): permutation (5, 6, 7, 8, 1, 2, 3, 4).
    # Top legs (lt, ut, rt, pt) reflect into bot ordering directly — no inner permutation needed.
    φ_refl = permutedims(φ, (5, 6, 7, 8, 1, 2, 3, 4))
    φ_sym  = (φ .+ φ_refl) ./ 2

    Dlt, Dut, Drt, pt_sz, Dlb, Ddb, Drb, pb_sz = size(φ_sym)
    @assert (Dlt, Dut, Drt, pt_sz) == (Dlb, Ddb, Drb, pb_sz) ":X vertical requires top/bot-symmetric leg dims"

    M = reshape(φ_sym, Dlt * Dut * Drt * pt_sz, Dlb * Ddb * Drb * pb_sz)
    Msym = (M .+ M') ./ 2
    F = eigen(Hermitian(Msym))
    perm = sortperm(abs.(F.values), rev=true)
    vals = F.values[perm]
    vecs = F.vectors[:, perm]
    keep = min(D_max, length(vals))
    trunc_err = sum(abs2, @view vals[keep+1:end])
    Σ_kept = sqrt.(complex.(vals[1:keep]))
    Ut = vecs[:, 1:keep] .* Σ_kept'
    A = reshape(Ut, Dlt, Dut, Drt, pt_sz, keep)
    A_new = permutedims(A, (1, 5, 3, 2, 4))
    if eltype(φ) <: Real
        A_new = real(A_new)
    end
    return A_new, real(trunc_err)
end

# ============================================================================
# make_H_op: 2-site bond Hamiltonian as a linear operator on φ
# ============================================================================

"""
    _build_h_bond_2site(model, atype, d) -> 4-leg tensor h[pl', pr', pl, pr]

Assemble the 2-site bond Hamiltonian as a 4-leg tensor from
`_heisenberg_bond_terms` (or analogous for other models). Leg convention:
  h[pl_new, pr_new, pl_old, pr_old] = sum_i coeff_i * OL_i[pl_new, pl_old] * OR_i[pr_new, pr_old]
"""
function _build_h_bond_2site(model, atype, d::Int)
    terms = _heisenberg_bond_terms(model, atype)
    h_arr = zeros(Float64, d, d, d, d)
    for (coeff, OL, OR) in terms
        OL_a = Array(OL)
        OR_a = Array(OR)
        @tensor t[plp, prp, pl, pr] := OL_a[plp, pl] * OR_a[prp, pr]
        h_arr .+= coeff .* real.(t)
    end
    return atype(h_arr)
end

"""
    make_H_op(rt, A, dir, params; mode=:a) -> Function

Return `H_op(φ) -> Hφ` applying the effective Hamiltonian on a 2-site bond.

Mode `:a` includes only the bond Hamiltonian acting on the central bond
between the two sites of φ. Mode `:b` (sum over all 7 bonds) is for v2.
"""
function make_H_op(rt::VUMPSRuntime, A, dir::Val, params; mode::Symbol=:a)
    if mode == :a
        return _make_H_op_central(rt, A, dir, params)
    elseif mode == :b
        error("make_H_op mode :b not yet implemented (v2)")
    else
        error("Unknown H_eff mode: $mode")
    end
end

function _make_H_op_central(rt::VUMPSRuntime, A, dir::Val{:H}, params)
    N_op = make_N_op(rt, A, dir, params)
    atype = _arraytype(A[1, 1])
    d = size(A[1, 1], 5)
    h = _build_h_bond_2site(params.model, atype, d)

    function H_op(φ)
        @tensor opt = true φh[l, dl, ul, plp, dr, ur, r, prp] :=
            φ[l, dl, ul, pl, dr, ur, r, pr] * h[plp, prp, pl, pr]
        return N_op(φh)
    end
    return H_op
end

function _make_H_op_central(rt::VUMPSRuntime, A, dir::Val{:V}, params)
    N_op = make_N_op(rt, A, dir, params)
    atype = _arraytype(A[1, 1])
    d = size(A[1, 1], 5)
    h = _build_h_bond_2site(params.model, atype, d)

    function H_op(φ)
        @tensor opt = true φh[lt, ut, rt, ptp, lb, db, rb, pbp] :=
            φ[lt, ut, rt, pt, lb, db, rb, pb] * h[ptp, pbp, pt, pb]
        return N_op(φh)
    end
    return H_op
end

# ============================================================================
# sweep_bond: build φ, geneigsolve, decompose back to A
# ============================================================================

"""
    sweep_bond(rt, A, dir, params, cfg) -> (λ, A_new, trunc_err)

One bond sweep step: build φ from A·A on the chosen bond, build H_op/N_op
from boundary env, run KrylovKit.geneigsolve to find the lowest generalized
eigenvalue/eigenvector pair, decompose φ_new back to a 1-site A_new.

Does NOT update boundary env or apply MCF (those are outer-loop concerns).
1×1 unit cell only.
"""
function sweep_bond(rt::VUMPSRuntime, A, dir::Val, params, cfg::iPEPSFixedPointConfig)
    A_central = A[1, 1]
    φ_old = build_phi(A_central, A_central, dir)
    H_op  = make_H_op(rt, A, dir, params; mode=cfg.H_eff_mode)
    N_op  = make_N_op(rt, A, dir, params)

    λs, φs, info = geneigsolve(
        x -> (H_op(x), N_op(x)), φ_old, 1, :SR;
        krylovdim   = cfg.geneig_krylovdim,
        tol         = cfg.geneig_tol,
        maxiter     = cfg.geneig_maxiter,
        ishermitian = true,
        isposdef    = true,
    )
    if info.converged < 1
        @warn "geneigsolve did not converge" info=info dir=dir
    end
    λ_new = real(λs[1])
    φ_new = φs[1]

    D = size(A_central, 1)
    A_new_central, trunc_err = decompose_phi(φ_new, dir;
                                             method = cfg.decompose_method,
                                             D_max  = D)
    A_new = deepcopy(A)
    A_new[1, 1] = A_new_central
    return λ_new, A_new, trunc_err
end

# ============================================================================
# optimize_ipeps_fixedpoint: top-level driver
# ============================================================================

"""
    optimize_ipeps_fixedpoint(A_raw, χ, model, params, cfg) -> history

Top-level driver for the fixed-point eigenvalue iteration optimizer.
- `A_raw` : 6D parameter array (D×D×D×D×d×Nsites).
- `χ`     : boundary bond dim.
- `model` : Hamiltonian model (used for diagnostics; bond H from `params.model`).
- `params`: GradientOptimize/iPEPSOptimize params (provides boundary_alg, pattern).
- `cfg`   : `iPEPSFixedPointConfig`.

Returns per-step history `Vector` of `NamedTuple` with keys
`(iter, λ, E, dλ, dA, trunc_err, t_total, t_env, t_eig, t_mcf)`.
"""
function optimize_ipeps_fixedpoint(A_raw, χ::Int, model, params, cfg::iPEPSFixedPointConfig)
    A = build_A(A_raw, params)
    rt = init_VUMPSRuntime(A, χ, params.boundary_alg)
    rt, _ = leading_boundary(rt, A, params.boundary_alg)

    history = NamedTuple[]
    λ_prev = NaN
    A_prev = deepcopy(A)

    for k in 1:cfg.outer_maxiter
        t0 = time()

        # Env update by mode
        t_env0 = time()
        if cfg.env_mode == :A
            rt, _ = leading_boundary(rt, A, params.boundary_alg)
        elseif cfg.env_mode == :B
            warm_alg = deepcopy(params.boundary_alg)
            warm_alg.maxiter = cfg.env_warm_steps
            rt, _ = leading_boundary(rt, A, warm_alg)
        elseif cfg.env_mode == :C
            warm_alg = deepcopy(params.boundary_alg)
            warm_alg.maxiter = 1
            rt, _ = leading_boundary(rt, A, warm_alg)
        else
            error("Unknown env_mode: $(cfg.env_mode)")
        end
        t_env = time() - t_env0

        # Sweep H + V
        t_eig0 = time()
        trunc_errs = Float64[]
        λs = Float64[]
        for dir in (Val(:H), Val(:V))
            λ_dir, A, te = sweep_bond(rt, A, dir, params, cfg)
            push!(λs, λ_dir)
            push!(trunc_errs, te)
        end
        t_eig = time() - t_eig0

        # MCF gauge fix on A[1,1] (wrap 5D → 6D for local_min_norm signature)
        t_mcf0 = time()
        if !cfg.mcf_ifignore_gauge
            A_c = A[1, 1]
            A_c6 = reshape(A_c, size(A_c)..., 1)
            A_c6 = local_min_norm(A_c6, params; ifignore_gauge=cfg.mcf_ifignore_gauge)
            A[1, 1] = reshape(A_c6, size(A_c))
        end
        t_mcf = time() - t_mcf0

        # Energy diagnostic via converged env (re-converges if env_mode != :A)
        env = ObsEnv(rt, A, params.boundary_alg)
        E, _ = energy_value(model, A, env, params)
        E = real(E)
        λ_now = sum(λs) / length(λs)
        dλ = isnan(λ_prev) ? Inf : abs(λ_now - λ_prev)
        dA = norm(A[1, 1] .- A_prev[1, 1])

        rec = (
            iter      = k,
            λ         = λ_now,
            E         = E,
            dλ        = dλ,
            dA        = dA,
            trunc_err = maximum(trunc_errs),
            t_total   = time() - t0,
            t_env     = t_env,
            t_eig     = t_eig,
            t_mcf     = t_mcf,
        )
        push!(history, rec)

        if k % cfg.log_every == 0
            @info "outer step $k" λ=λ_now E=E dλ=dλ dA=dA trunc_err=rec.trunc_err
        end

        if dλ < cfg.outer_tol_λ && dA < cfg.outer_tol_A
            @info "converged at iter $k"
            break
        end
        λ_prev = λ_now
        A_prev = deepcopy(A)
    end
    return history
end
