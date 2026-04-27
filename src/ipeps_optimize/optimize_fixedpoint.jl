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
"""
function make_N_op(rt::VUMPSRuntime, A, dir::Val{:H}, params)
    error("make_N_op horizontal: not yet implemented (filled in Task 7)")
end
