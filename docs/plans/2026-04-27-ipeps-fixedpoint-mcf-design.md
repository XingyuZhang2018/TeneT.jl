# Fixed-Point Eigenvalue Iteration for iPEPS Ground State with MCF

Date: 2026-04-27
Status: Brainstorming complete; ready for implementation plan
Branch target: based off `iPEPS-unified` (NOT this worktree, which is on the older VUMPS-only `claude/mystifying-tesla-242a2c`)

## Context and Motivation

iPEPS ground-state optimization in TeneT.jl currently uses energy-gradient descent (LBFGS, optionally preconditioned by `N⁻¹` from boundary VUMPS environment, see `src/ipeps_optimize/optimize.jl` and `precondition.jl`). Robust but slow.

This design explores a different approach inspired by 2-site iDMRG: solve a generalized eigenvalue problem `H_eff φ = λ N_eff φ` for the ground state on a 2-site bond, decompose back to local tensors, gauge-fix via Minimal Canonical Form (MCF), iterate.

**Key new ingredient:** `local_min_norm` (Acuaviva–Bondarenko–Christandl–Sgarbazzi style MCF, alternating-SVD) is now implemented in TeneT.jl at `src/ipeps_optimize/restriction.jl:439`, providing a unique gauge for any iPEPS. Hypothesis: MCF stabilizes the fixed-point iteration that would otherwise be ill-conditioned due to gauge ambiguity.

**Critical clarification:** MCF does NOT make `N_eff = I`. The eigenvalue problem remains generalized `Hφ = λNφ`. MCF's role is to fix gauge so the iteration is uniquely determined between outer steps.

This is a **proof-of-concept** to answer:
1. Does the iteration converge at all?
2. Does MCF actually make it stable (vs. without MCF)?
3. Does it find the same ground state as LBFGS, with comparable accuracy?

If PoC fails, write up findings and stop. Not a production replacement for LBFGS.

## Setup

| Parameter | Value |
|---|---|
| Model | Heisenberg + sublattice rotation |
| Lattice | Square, **1×1 unit cell** |
| Bond dimension | D = 2–3 |
| Boundary χ | 8–16 |
| Implementation | `TeneT.jl/src/ipeps_optimize/optimize_fixedpoint.jl` (new file, on `iPEPS-unified` branch) |

## Algorithm

### High-level loop

```
input: A (initial 5-leg tensor, 1×1 unit cell)
loop k = 1, 2, ...:
    # === env step (mode-dependent) ===
    rt = update_env(A, rt; mode = config.env_mode)

    # === alternating bond sweep ===
    for direction in [:H, :V]:
        φ_old = build_phi(A, direction)                       # 8-leg
        H_op  = make_H_op(rt, A, direction; mode = config.H_eff_mode)
        N_op  = make_N_op(rt, A, direction)
        λ, φ_new = geneigsolve((H_op, N_op), φ_old, 1, :SR;
                               krylovdim = config.geneig_krylovdim,
                               tol       = config.geneig_tol,
                               maxiter   = config.geneig_maxiter,
                               ishermitian = true, isposdef = true)
        A = decompose_phi(φ_new, direction;
                          method = config.decompose_method,
                          D_max  = D)

    # === MCF gauge fix ===
    A = local_min_norm(A, params; ifignore_gauge = config.mcf_ifignore_gauge)

    # === diagnostics + convergence ===
    log_step(k, λ, energy(A, rt), ...)
    if dλ < config.outer_tol_λ && dA < config.outer_tol_A: break
```

Two bond sweeps (H + V) = one outer step. A/B/C modes change ONLY the env update step; everything else is identical.

### Iteration mode (env interleaving)

| Mode | Env update per outer step | Cost | Stability hypothesis |
|---|---|---|---|
| `:A` | full VUMPS to convergence | high | most stable |
| `:B` | K-step warm restart (default K=5) | medium | iDMRG-like |
| `:C` | 1-step interleaved | low | most aggressive, may oscillate |

All three exposed as runtime config; PoC explores all three experimentally.

### H_eff construction modes

H_eff is the only operator that requires summing multiple bond contributions; N_eff has only one form (the boundary norm contraction around the 2-site region).

| Mode | Description | Status |
|---|---|---|
| `:a` | h on central bond only; sweep through other bonds via direction alternation | **v1: implement** |
| `:b` | sum over all 7 bonds touching the 2-site region | v2: extend |

`:a` is closer to FU-update style (one bond's h at a time). `:b` is closer to iDMRG 2-site update spirit (full local H_eff). `:b` evaluation cost is ~7× of `:a`.

### φ → A decomposition methods (1×1 symmetrization)

The φ → A step is the only fundamentally novel piece (no 1D analogue). 8-leg φ_new is generally NOT in the variational manifold {φ : φ = A·A}; the projection back is lossy and method-dependent.

| Method | Description | Recommended |
|---|---|---|
| `:X` | Symmetrize-first: `φ_sym = (φ_new + reflect(φ_new))/2`, then `eigh` across central bipartition, truncate to D, `A_new = U·√Σ` reshaped | **default** |
| `:Y` | SVD φ_new, average A_l and A_r (with leg permutation) | for comparison |
| `:Z` | SVD φ_new, take A_l only (lazy) | for comparison |

`:X` is principled because H, N are reflection-symmetric → ground state φ should be too; symmetrize cleans numerical noise. The reflection operator implementation must be careful for horizontal vs vertical bonds.

### MCF

Reuse existing `local_min_norm(A, params; ifignore_gauge=false)`. Implementation already has Zygote rrule (not needed here since no AD through fixed-point loop, but available).

### Generalized eigenvalue solver

`KrylovKit.geneigsolve` with `(H_op, N_op)`, `:SR`, `ishermitian=true`, `isposdef=true`. **Fail loud** on non-convergence (no silent fallback to power iteration). If routinely unstable, v2 will switch to `linsolve(N, H·x)` power iteration based on existing `precondition_invese_single_envir` machinery.

## File structure

```
TeneT.jl/src/ipeps_optimize/
├── optimize.jl                      EXISTING (gradient-based LBFGS)
├── restriction.jl                   EXISTING (contains local_min_norm)
├── precondition.jl                  EXISTING (N⁻¹ machinery, 1-site)
└── optimize_fixedpoint.jl           NEW
    Contains:
      struct iPEPSFixedPointConfig
      build_phi(A, dir)
      make_H_op(rt, A, dir; mode)
      make_N_op(rt, A, dir)
      decompose_phi(φ_new, dir; method, D_max)
      sweep_bond!(A, rt, dir, config)
      optimize_ipeps_fixedpoint(A, χ, model, params, config)

TeneT.jl/examples/Heisenberg/
└── Heisenberg_Square_FixedPoint.jl  NEW (PoC entry point)
```

## Config struct

```julia
@kwdef struct iPEPSFixedPointConfig
    env_mode::Symbol           = :A          # :A full / :B warm / :C 1-step
    env_warm_steps::Int        = 5           # for :B
    H_eff_mode::Symbol         = :a          # :a central bond / :b all 7
    decompose_method::Symbol   = :X          # :X symmetrize / :Y avg / :Z lazy
    geneig_krylovdim::Int      = 20
    geneig_tol::Float64        = 1e-10
    geneig_maxiter::Int        = 100
    mcf_ifignore_gauge::Bool   = false       # if true, MCF disabled (control)
    outer_maxiter::Int         = 200
    outer_tol_λ::Float64       = 1e-8
    outer_tol_A::Float64       = 1e-7
    log_every::Int             = 1
    save_every::Int            = 10
end
```

All experimental knobs exposed → 3·2·3 = 18 mode combinations + MCF on/off control.
PoC does NOT exhaustively sweep these — runs targeted experiments below.

## Diagnostics (per outer step)

CSV-like log line:
```
iter | dir | λ_eig | E_iPEPS | dλ | dA | trunc_err | env_iter | mcf_iter | t_total | t_env | t_eig | t_mcf
```

- `λ_eig`: generalized eigenvalue from `geneigsolve` (local quantity)
- `E_iPEPS`: energy recomputed with converged env (global; cross-check that `λ_eig` is close to true energy)
- `trunc_err`: sum of squared discarded singular values from `decompose_phi`
- `dλ`, `dA`: outer convergence monitors
- per-step timing: identify bottleneck and quantify A/B/C cost difference

## Initial state

- **Primary**: load existing LBFGS-optimized `A` from jld2 (same D, χ) to enable direct energy comparison.
- **Fallback**: random init + brief LBFGS warmup (50 steps) — avoids conflating "algorithm fails" with "init terrible".
- **Avoid**: pure random.

## Success criteria (PoC deliverable)

| # | Criterion | Target | Failure interpretation |
|---|---|---|---|
| 1 | **Convergence** | at least one (env_mode × decompose_method) combo reaches `dλ < 1e-6` in ≤ 200 outer steps | concept fails |
| 2 | **Physical correctness** | final `E_iPEPS` within `1e-3 / site` of LBFGS baseline | bug or conceptual error |
| 3 | **Stability** | `dλ` mostly monotone-decreasing, no sustained oscillation | MCF not stabilizing as hoped |
| 4 | **MCF value (key science Q)** | `mcf_ifignore_gauge=true` control run shows clear degradation OR fails to converge | MCF unnecessary → reconsider hypothesis |

Criterion #4 is the real PoC deliverable: does MCF actually do the stabilization we hypothesized?

## Targeted experiment matrix (PoC)

Not all 18 combos. Run these:

| Experiment | env_mode | H_eff | decompose | MCF | Purpose |
|---|---|---|---|---|---|
| E1 baseline | :A | :a | :X | on | does it work at all? |
| E2 MCF off | :A | :a | :X | off | the science question |
| E3 cheap env | :C | :a | :X | on | do A/B/C differ? |
| E4 lazy decomp | :A | :a | :Z | on | does decompose method matter? |
| E5 avg decomp | :A | :a | :Y | on | ditto |

If E1 works: continue. If E1 fails: re-evaluate before more experiments.

## Out of scope (v2 candidates)

- Plaquette (4-site) φ
- H_eff mode `:b` (all 7 bonds at once)
- 2×N unit cells (no sublattice rotation; for AFM Heisenberg without rotation, J1J2, etc.)
- Adaptive env step count
- Replacing `optimise_ipeps` public interface
- AD differentiability through the fixed-point loop (irrelevant; this is direct optimization)

## Risks

- **N ill-conditioning**: gauge null space leaks through MCF → `geneigsolve` slow / unstable. Fallback: `N⁻¹H` power iteration (already-existing `linsolve(N+δ, ...)` machinery, extended from 1-site to 2-site).
- **Symmetrization × MCF interaction**: method `:X` enforces reflection symmetry, MCF enforces minimum-norm gauge — these may fight. Mitigation: config knobs allow disabling either; control experiments E2, E4, E5 separate effects.
- **Gapless Heisenberg slow convergence**: outer iteration may converge very slowly due to gapless Goldstone modes. PoC accepts "slow but converging" as criterion 1 success.
- **Worktree migration**: this design doc lives on old `mystifying-tesla-242a2c` branch. Implementation must move to a new worktree based on `iPEPS-unified`. Will copy / re-commit on switch.

## Open questions deferred to implementation

- Exact contraction order for `make_H_op` and `make_N_op` (need 2-site versions of existing 1-site contractions in `precondition.jl`)
- Exact reflection-leg-permutation for `:X` decompose (horizontal vs vertical)
- Whether to symmetrize across H and V passes within an outer step
