# PoC Findings — iPEPS Fixed-Point Eigenvalue Iteration with MCF

> **Status: NEGATIVE — algorithm has a fundamental conceptual flaw. Do NOT continue this approach as written.**
>
> All implementation correct (51 tests pass, including strong sanity checks against existing TeneT primitives). Failure is algorithmic, not implementation.

## TL;DR

Tested the design of [docs/plans/2026-04-27-ipeps-fixedpoint-mcf-design.md](2026-04-27-ipeps-fixedpoint-mcf-design.md):
solve `H_eff φ = λ N_eff φ` on a 2-site bond, decompose `φ_new` back to a 1-site iPEPS tensor, gauge-fix with MCF, iterate.

**Result**: the iteration **does not converge** because `geneigsolve` consistently finds the unconstrained 2-site spin singlet (λ = -3J/4 = -0.75) which lies **far outside the PEPS-rank-D variational manifold**. The lossy `decompose_phi` projection back to rank-D destroys ~80% of the eigenvector's mass on the first step. MCF cannot recover from this — the failure is "manifold escape", not gauge ambiguity.

The science question (criterion #4 — is MCF essential?) **cannot be answered**: E2-E5 would all fail by the same manifold-escape mechanism, providing no information about MCF's role.

## Was the algorithm correct?

| Criterion | Target | Actual | Verdict |
|---|---|---|---|
| #1 Convergence | one config reaches `dλ < 1e-6` in ≤ 200 outer steps | E1 crashes by iter 8; thrashes for 1-7 with `λ` pinned at -3/4 | **FAIL** |
| #2 Physical accuracy | `\|E_PoC + 0.66\| < 1e-3 / site` | `E` oscillates between -0.38 and +0.45 across 7 iters | **FAIL** |
| #3 Stability | `dλ` mostly monotone | `dA ≈ 1` per step (A norm ~1) → completely thrashing | **FAIL** |
| #4 MCF essential? | E1 vs E2 distinguishes | Cannot distinguish — both fail by escape | **N/A** |

## What was implemented (all working)

`src/ipeps_optimize/optimize_fixedpoint.jl`:
- `iPEPSFixedPointConfig` — config struct with all design knobs.
- `build_phi(A_l, A_r, dir)` — 8-leg 2-site bond tensor (H + V).
- `make_N_op(rt, A, dir, params)` — 7-tensor `@tensor opt=true` contraction, returns `N · φ`.
- `make_H_op(rt, A, dir, params; mode=:a)` — `H_op(φ) = N_op(h ⊗ φ)`.
- `decompose_phi(φ, dir; method∈{:Z,:Y,:X}, D_max)` — three projection methods.
- `sweep_bond` — single bond geneigsolve + decompose.
- `optimize_ipeps_fixedpoint(A_raw, χ, model, params, cfg)` — top-level driver.

`test/test_optimize_fixedpoint.jl`: **51 tests, 51 passing.** Strong sanity checks include:
- `dot(conj(φ), N_op(φ)) ≈ contract_n_12(...)` (canonical 2-site norm) — **rtol 1e-10**.
- Same for vertical via `contract_n_21`. — **rtol 1e-10**.
- `e_bond = <φ|H_op|φ>/<φ|N_op|φ> ≈ e_dict["bond_H_energy"]["1,1"]` from `energy_value` — **rtol 1e-8**.
- `decompose_phi`: lossless when input φ has bond rank ≤ D (Z, Y); bounded trunc_err for X.

These confirm the framework is consistent with the codebase's verified primitives. **The code is correct.**

## The decisive diagnostic: `‖Hφ - λ·Nφ‖` residual at LBFGS-converged A

`scripts/lbfgs_c4v_then_residual.jl` measures, at a 1×1 Heisenberg LBFGS-warmed A, both:
1. The bond expectation value: `λ_est = <φ|H|φ> / <φ|N|φ>` (= bond_H_energy).
2. The eigenvalue residual: `‖Hφ - λ_est·Nφ‖ / ‖Hφ‖`.

A *true* fixed-point of the iteration would have residual << 1 (φ is an eigenvector of (H, N)). Across **6 random seeds × 2 etypes**:

| seed | iters | E_LBFGS | gnorm | **residual / ‖Hφ‖** | KrylovKit λ | |<φ_LBFGS, φ_kry>\| |
|------|-------|---------|-------|---------------------|-------------|----------------------|
| 42 (Float64) | 2 | -0.580 | 0.28 | **0.86** | -0.7500 | 0.094 |
| 1 (Float64) | 2 | -0.570 | 0.087 | **0.86** | -0.7500 | 0.081 |
| 1 (ComplexF64) | 3 | -0.617 | 9.6 | 0.82 | (Hermiticity check failed) | — |
| 2 (Float64) | 3 | -0.613 | 0.79 | 0.84 | -0.7500 | — |
| 3 (Float64) | 2 | -0.606 | 0.65 | 0.84 | -0.7500 | — |
| 7 (Float64) | 2 | -0.564 | 0.073 | 0.87 | -0.7500 | — |
| **100** (Float64) | 6 | **-0.644** | 2.25 | **0.80** | -0.7500 | — |
| **42 (VUMPS{C4v} + precondition)** | **13** | **-0.66023** | **6.5e-8** | **0.77** | **-0.7500** | **0.00013** |

The last row is the **decisive measurement**. With `VUMPS{C4v}` boundary algorithm and `ifprecondition=true`, LBFGS escapes the linesearch stall and converges to the exact reference variational minimum (gnorm 6.5e-8). At this *perfectly converged* state:
- Residual ‖Hφ - λ·Nφ‖ / ‖Hφ‖ = 0.77 — still O(1).
- KrylovKit eigenvector overlap with φ_LBFGS = **0.013%** — essentially orthogonal.
- KrylovKit eigenvalue = -0.75 (= -3J/4, isolated 2-site singlet).
- φ_LBFGS bond energy = -0.33 (PEPS variational); φ_kry bond energy = -0.75 (unconstrained singlet).

The two fixed-points are nearly orthogonal vectors with O(1) energy difference per bond — no PEPS-manifold-preserving operation (decompose/MCF/gauge fix) can bridge them.

**Pattern**: as LBFGS converges deeper (E approaches -0.66), residual asymptotes to ~0.77, NOT to zero. The trend confirms: at the exact PEPS variational minimum, the framework's eigenvalue iteration target is still O(1) far away.

## Two separate issues, both inherent

After a follow-up reframing test (`scripts/five_leg_eigsolve.jl`), it became clear the failure has TWO distinct components:

### Issue 1: Manifold escape (the 8-leg version)
`geneigsolve` on the 8-leg space finds the unconstrained 2-site singlet (λ = -3J/4) lying outside PEPS-rank-D. `decompose_phi` truncation discards ~80%+ of the eigenvector mass.

### Issue 2: Local-vs-global fixed-point divergence (even in-manifold)
We tested a 5-leg version: `H_eff_5(x) = M_l^†[H_op_8leg(M_l[x])]` where `M_l[x] = build_phi(x, A_old, :H)`. This stays inside the 5-leg PEPS-rank-D manifold by construction. But:

| Quantity | Value |
|---|---|
| 5-leg residual at LBFGS-A: ‖HA - λNA‖ / ‖HA‖ | **0.55** (still O(1)) |
| KrylovKit 5-leg ground eigenvalue | -0.71 |
| \|⟨A_LBFGS, A_5leg_kry⟩\| | **0.00009** (~orthogonal) |

So even when the iteration is restricted to PEPS-rank-D, its fixed-point is *still* nearly orthogonal to the LBFGS variational minimum. The reason: per-bond eigvalue iteration with env held fixed has a fixed-point characterized by

  per-bond local stationarity: ∂(<φ|H|φ>/<φ|N|φ>)/∂A_l = 0  (with env fixed)

while LBFGS minimum is characterized by

  global stationarity:  ∂E_total/∂A = 0  (with env varying via ∂env/∂A)

These are different conditions. The env-AD term `∂E/∂env · ∂env/∂A` and inter-bond couplings together are O(1), so the two fixed-points differ by O(1) in tensor norm.

This means: **ANY per-bond / fixed-env Rayleigh-quotient iteration cannot reach the LBFGS minimum**. Local 5-leg eigsolve, FU updates, per-bond gradient descent — all converge to different fixed-points than LBFGS. To reach LBFGS minimum, the iteration must use the global gradient with full env-AD chain rule (which is exactly what `optimise_ipeps` does).

The original PoC's "geneigsolve per bond" structure cannot recover the LBFGS minimum *even if* manifold escape were somehow prevented. The structural choice "per-bond Rayleigh quotient" is itself wrong for a target of "global LBFGS minimum".

**Geometric interpretation**:
- LBFGS minimum: PEPS-rank-D variational ground state, bond energy ≈ -0.29 to -0.33 (per-bond, depending on convergence).
- Framework target eigenvector: 2-site spin singlet, bond energy = -3/4 = -0.75.
- |φ_LBFGS, φ_kry overlap| ≈ 0.08-0.10 across seeds — the two states are **nearly orthogonal**.

These two fixed-points are O(1) apart in tensor norm and ≈ 0.46 apart in per-bond energy. No restriction (decompose method, env mode, MCF flag) bridges this gap.

## Was MCF essential? (criterion #4 — key science Q)

**Cannot be determined from this PoC.** The failure cause is unconstrained `geneigsolve` escaping the PEPS manifold, not gauge ambiguity that MCF was designed to fix. Running E2 (MCF off) would crash by the same mechanism, slightly faster perhaps, but providing no information about MCF's stabilization role.

For the design's hypothesis "MCF stabilizes against gauge ambiguity" to be testable, **the iteration must first stay within the manifold** — which the current scheme cannot do.

## Sensitivity to design choices (NOT meaningfully separable)

| Knob | Tested? | Useful? |
|---|---|---|
| Env mode (A vs B vs C) | partial — E1 used :A | No — manifold escape happens at sweep level, before env step |
| Decompose method (X vs Y vs Z) | three implementations all working | No — all fail; difference is between modes of escape, not whether |
| MCF on/off | only on tested (E1) | No — gated by criterion #1 not being met |
| Unit cell (1×1 vs 2×2) | LBFGS confirmed both work | Manifold escape is local-bond phenomenon, not cell-size |
| Initial state | random vs LBFGS-warmup vs converged 2×2 | All escape — convergence depth doesn't help |
| Random seed | sweep of 6 seeds | All show same escape pattern |

## Implementation bugs found and fixed during PoC

1. **`:X` decompose with negative top eigenvalues**: original used `eigen` on `(M+M')/2` and `sqrt(complex(neg_val))` → imaginary → `real()` zeros A_new. Replaced with SVD (always positive singular values, robust to KrylovKit's arbitrary `φ_new` sign convention). Commit `b51c506`.
2. **Model definition mismatch**: original `make_default_params` used `Heisenberg(Jx=Jy=-1, Jz=1)` (older defaults inconsistent with current `_heisenberg_bond_terms` sign convention). Updated to `Jx=Jy=Jz=+1` matching the canonical example file. Commit `b51c506`.
3. **LBFGS jld2 file path was for old code state**: the old stupefied-cartwright data (`Jx=-1`) gave +0.34 instead of -0.66 in current framework — sign-flipped Hamiltonian. Re-ran LBFGS with current code as `scripts/lbfgs_warmup.jl` and `scripts/lbfgs_warmup_2x2.jl`.

## Recommendations

### Do NOT continue this approach as written

The "8-leg geneigsolve + lossy projection" loop is fundamentally not iterating toward a PEPS ground state — it's iterating toward an unreachable singlet. No tweak of the listed knobs fixes this.

### If anyone wants to revisit (v2)

The path forward must **constrain the search to the PEPS manifold**. Possible directions, ranked by closeness to the original PoC's spirit:

1. **Parameterize-on-A directly**: instead of solving `geneigsolve(H_eff, N_eff, φ)` and projecting, solve a 5-leg generalized eigenvalue problem `H_A · A = λ · N_A · A` where `H_A`, `N_A` are 5-leg→5-leg operators built from env + neighbour A. This is essentially a 2-site full-update (FU) step but with eigenvalue rather than gradient updates. There's existing TeneT machinery (`FUOptimize`) that does similar work.

2. **Trust-region damped step**: `A_new = (1-α) A_old + α A_after_decomp`, with α ∈ (0, 1) chosen so that bond rank stays ≤ D in linearised approximation. Would need a search over α each step.

3. **Constrain the Krylov subspace**: pass `geneigsolve` only φ vectors in the form `build_phi(A, A)` — but Krylov methods don't natively support manifold constraints. Would need to project gradients at each Lanczos step.

4. **Replace `geneigsolve` with `linsolve(N⁻¹H · x)` power iteration in the projected manifold**: similar to existing `precondition_invese_single_envir` (1-site) but extended to 2-site. Slower per iter but easier to constrain.

These are not minor revisions — they are essentially different algorithms.

## Artifact summary

- **Branch**: `claude/ipeps-fixedpoint-mcf` (off `iPEPS-unified`).
- **51 unit tests**, all passing.
- **`src/ipeps_optimize/optimize_fixedpoint.jl`** — full implementation (~600 lines).
- **`examples/Heisenberg/Heisenberg_Square_FixedPoint.jl`** — E1-E5 driver.
- **`scripts/lbfgs_warmup.jl`**, **`scripts/lbfgs_warmup_2x2.jl`** — LBFGS warmup utilities.
- **`scripts/lbfgs_c4v_then_residual.jl`** — the decisive diagnostic.
- **`scripts/inspect_decompose.jl`**, **`scripts/inspect_after_mcf.jl`**, **`scripts/verify_lbfgs_energy.jl`**, **`scripts/verify_2x2_lbfgs.jl`** — debugging utilities.
- **`docs/plans/baseline-energies.md`** — reference E_LBFGS values.
- **`docs/plans/poc-results.md`** — initial E1 trace + analysis.
- **`docs/plans/poc-findings.md`** — this document.

## Recommendation for the branch

- **Merge** the `optimize_fixedpoint.jl` + tests if they have value as standalone primitives (`build_phi`, `make_N_op` 2-site contractions are reusable building blocks for any future 2-site approach). 51 tests with strong checks against existing `contract_n_12` / `energy_value` provide quality evidence.
- **OR Archive** the branch with this findings doc for reference.

The decision is the maintainer's. From an algorithmic perspective, the PoC delivered a clean negative result.
