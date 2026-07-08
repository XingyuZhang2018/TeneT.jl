# Design: Alternatives to Nested Self-Consistent Iteration in VUMPS (Ising C4v testbed)

**Date:** 2026-07-08
**Status:** Approved (brainstorming session)
**Testbed:** `examples/2D_Classical/Ising_Square_VUMPS_C4v.jl`, `src/boundary_algorithm/vumps/c4v.jl`

## 1. Research question and evaluation framework

VUMPS finds the leading boundary MPS as the fixed point where the tangent-space
gradient vanishes. Each outer step solves inner eigenvalue problems
(FL / AC / C) plus QR. The inner eigensolver can be swapped from Arnoldi to a
plain power method (`ifsimple_eig=true`) — still correct at outer convergence,
but the outer iteration count grows, catastrophically so near criticality
(documented in `docs/benchmarks/chi128_D2_ising_solver_comparison.md`: at β_c
with `power_iter<50`, err stalls at 1e-5 and silently hits `maxiter=500`).

**Question:** where is the optimal inner/outer balance for the nested
self-consistent structure, and is there a non-self-consistent scheme that is
better overall?

**Primary metric: total map applications** (FLmap / ACmap / Cmap counted
separately; report both raw counts and cost-weighted views since Cmap is much
cheaper). This metric is implementation-independent — the Julia side compares
only this. Wall clock is secondary; true kernel speed is delegated to
KrylovKit.c (§5).

**Unified convergence accounting** (key pitfall): VUMPS `err = ‖R_AC − R_C‖`
and a Riemannian gradient norm are not directly comparable. Acceptance uses
two curves per algorithm:

- `|f − f_Onsager|` vs cumulative map applications (physics accuracy; note the
  finite-χ plateau ~1e-8 at χ=128, β_c — compare *speed to plateau*).
- Internal residual vs cumulative map applications (who stalls, who doesn't).

## 2. Shared experiment infrastructure

- **Standalone research scripts, no `src/` changes:** new directory
  `timing/vumps_alternatives/`. Scripts call internal maps
  (`TeneT.FLmap_parallel` etc.) directly and implement their own `vumps_step`
  variants. Map counting via closure wrappers around each `f(x)`. Rationale:
  research freedom without touching production hot paths
  (REPOSITORY_RULES.md).
- **Physics setup:** Ising Square C4v, β ∈ {0.43 (easy), β_c = 0.4406867935
  (hard)}, χ ∈ {64, 128}, Float64 CPU. `exact_free_energy` as ground truth.
- **Same seed, same initial state** for all algorithms; one Julia process per
  algorithm run (same-process A/B pollution rule, applied on CPU too to
  isolate JIT state).

## 3. Candidate algorithm specifications

### Baselines (existing code)

- `B0-power-pi`: existing `simple_eig`, pi ∈ {1, 5, 10, 20, 50, 100}.
  pi=1 is the "fully collapsed single-loop power method" limit.
- `B0-arnoldi`: KrylovKit `eigsolve` to tol (strongest inner solver for the
  self-consistent structure).

### Family A — upgraded inner solvers (self-consistency done right)

- `A1 adaptive inner tolerance` (Eisenstat–Walker): inner stop at
  `max(η·outer_err, tol_floor)`, η ∈ {0.1, 0.01}, same rule for all three
  fixed points. Power residual `‖f(v) − λv‖` estimated from already-computed
  quantities (no extra map). Expected: kills early over-solving waste and the
  per-problem `power_iter` tuning.
- `A2 momentum power iteration` (heavy ball):
  `v_{k+1} = f(v_k)/‖·‖ − β v_{k−1}`, β from a running λ₂ estimate (spectrum
  real in the hermitian case). Convergence rate gap → √gap at zero extra map
  cost; largest payoff near criticality.
- `A3 hermitian Lanczos`: first numerically verify self-adjointness of the
  C4v Ising FLmap/ACmap (`⟨x, f(y)⟩ ≈ ⟨f(x), y⟩`); if it holds, run
  `eigsolve` with `ishermitian=true` (3-term recurrence) and compare iteration
  counts/stability vs the current `ishermitian=false` Arnoldi.

### Family B — outer fixed-point acceleration

- `B1 Anderson(m)`: treat `vumps_step` (inner pi ∈ {1, 5}) as a black-box
  fixed-point map; Anderson mixing on `vec(AL, C)`, m ∈ {2, 5}. Gauge handling
  escalation: (i) mix raw iterates (qrpos already fixes a positive-diagonal-R
  gauge); (ii) if that fails, polar-project mixed AL back onto the isometry
  manifold; (iii) if that fails, record the failure mode — a documented
  negative result is a valid outcome.

### Family C — Riemannian direct optimization (non-self-consistent route)

- `C1 Riemannian GD + line search`: variable = isometric AL (Stiefel
  manifold; symmetrize the gradient to preserve the C4v constraint).
  Objective = −log λ(AL). The gradient needs the FL environment — solved with
  the A1 adaptive inner solver + warm starts, and **those maps count honestly
  toward the total**. Retraction = polar. Gradient validated by finite
  differences.
- `C2 Riemannian CG / L-BFGS`: C1 skeleton + direction memory (vector
  transport = projection). Literature expectation (Hauru–Van Damme–Haegeman,
  SciPost 2021): remains monotone near criticality where VUMPS-type schemes
  stall.

(Newton–Krylov on the tangent-space residual considered and cut — L-BFGS
covers the curvature idea more robustly; YAGNI.)

## 4. Evaluation protocol and deliverables

- Per algorithm × (β, χ) cell: map-count trajectory, internal residual
  trajectory, f-error trajectory, wall clock.
- Core plots/tables: f-error vs cumulative maps (log-log overlay); table of
  maps-to-reach |Δf| < {1e-6, 1e-8, plateau}; dedicated critical-point stall
  analysis.
- Report: `docs/benchmarks/vumps_fixed_point_alternatives_c4v.md`, with a
  conclusion matrix (which scenario → which method) and a qualitative outlook
  section on AD-ability of each candidate (this study is forward-only by
  decision).
- Execution order: infra → B0 baselines → A → C → B → report → KrylovKit.c
  validation.

## 5. KrylovKit.c wall-clock validation layer (phase 2)

After the Julia-side iteration-count comparison picks winners, benchmark the
winning kernels (e.g. Lanczos, momentum power) in
[KrylovKit.c](https://github.com/qiyang-ustc/KrylovKit.c) (C++/CUDA) at
χ = 64–256 for true wall-clock, answering whether the iteration-count
advantage survives real kernel costs. Kernel-level only (single fixed-point
solve); the full VUMPS loop is not rewritten in C++.

## 6. Risks and expected failure modes

- **Anderson gauge trap:** mixing tensors in different gauges may simply
  diverge — the failure mode goes in the report as a result.
- **Riemannian methods still need environments:** not zero nesting; under
  honest map accounting the advantage may be smaller than the literature
  impression — quantifying this is part of the point.
- **Critical-point f plateau:** at χ=128 the finite-χ error floor (~1e-8)
  caps |Δf|; convergence claims must also rest on internal residuals.
- **λ₂ estimation noise for momentum:** clustered spectra make the estimate
  jumpy; use a sliding window / conservative lower bound.

## Session decisions

- Goal: open-ended algorithm research (comparison report), not a production
  feature.
- Forward-only exploration; AD compatibility is an outlook section, not a
  constraint.
- Julia = iteration-count science; KrylovKit.c = wall-clock ground truth
  (user judgment: Julia Krylov wall-clock is not trustworthy as a speed
  reference).
- Scope: all three families A + B + C approved.
