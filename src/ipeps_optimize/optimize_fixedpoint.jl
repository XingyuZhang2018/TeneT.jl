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
