# CPU benchmark — inner_etype=Float32 on Heisenberg_Square_VUMPS_C4v

**Design:** [docs/2026-04-17-inner-float32-vumps-design.md](../2026-04-17-inner-float32-vumps-design.md)
**Plan:** [docs/2026-04-17-inner-float32-vumps-plan.md](../2026-04-17-inner-float32-vumps-plan.md)

Environment: Julia 1.11.1, Windows 11, CPU (Array backend). BLAS thread count not pinned for L1 (single-call latency is negligible).

## L1 — single FLmap_parallel call (2026-04-17, CPU)

| D | χ | seed | err_forloop | err_float32 | ratio |
|---|---|------|-------------|-------------|-------|
| 2 | 16 | 42 | 0.000e+00 | 1.814e-07 | 816927568.642 |
| 2 | 16 | 43 | 0.000e+00 | 1.787e-07 | 804964928.781 |
| 2 | 16 | 44 | 0.000e+00 | 1.708e-07 | 769167917.341 |
| 2 | 32 | 42 | 0.000e+00 | 2.057e-07 | 926291736.316 |
| 2 | 32 | 43 | 0.000e+00 | 2.213e-07 | 996576141.510 |
| 2 | 32 | 44 | 0.000e+00 | 2.462e-07 | 1108995398.870 |
| 3 | 16 | 42 | 2.743e-16 | 2.347e-07 | 855621886.817 |
| 3 | 16 | 43 | 2.973e-16 | 2.551e-07 | 858257341.640 |
| 3 | 16 | 44 | 2.916e-16 | 2.422e-07 | 830843509.946 |
| 3 | 32 | 42 | 1.369e-16 | 2.674e-07 | 1204429069.647 |
| 3 | 32 | 43 | 1.462e-16 | 2.868e-07 | 1291426723.152 |
| 3 | 32 | 44 | 1.368e-16 | 2.753e-07 | 1239966954.866 |
| 3 | 64 | 42 | 2.088e-16 | 2.899e-07 | 1305600581.725 |
| 3 | 64 | 43 | 2.057e-16 | 2.833e-07 | 1275659594.925 |
| 3 | 64 | 44 | 1.905e-16 | 2.874e-07 | 1294400542.196 |

### L1 interpretation

- **Raw `err_float32`** is consistently **1.7e-7 to 2.9e-7** — at Float32 machine epsilon (~1.19e-7),
  scaling mildly with problem size. Expected behavior for one FLmap contraction in Float32.
- **Raw `err_forloop`** is `0` (D=2) or `~1e-16` (D=3) — essentially zero. Float64 summation
  reordering only flips the lowest bit or two at these modest (D, χ), well below Float32 noise.
- **Ratio `err_float32 / err_forloop` is ~1e9**, which misleadingly tripped the original
  `ratio < 10` gate to "STOP". But the denominator is vanishingly small, not large — so the
  ratio-based gate is the wrong heuristic here.
- **Correct heuristic**: is the raw `err_float32` small enough to be tolerable for iPEPS
  convergence? At ~3e-7, yes — it sits at the level of `gradtol=1e-7` but L4 will measure how
  that propagates through LBFGS.

**Decision: proceed to L4** (GATE override, based on raw-magnitude criterion).

> Note on the user's original observation (large `FLmap_forloop` differences at `forloop=1` vs
> `forloop=16`): this was presumably at larger D/χ than tested here, or after VUMPS iteration
> accumulated the noise. At single-call D≤3, χ≤64, Float64 reordering noise is effectively zero.

## L4 D=2 — Run 1: no polish (all AD iters in Float32)

**Config:** `inner_etype=Float32`, `inner_etype_final_steps=0` equivalent (field didn't exist yet).
All 4 AD iterations use Float32 inner contractions.

BLAS threads: 4

| D | χ | seed | precision | E_final | n_steps | wall (s) | ΔRSS (MB) |
|---|---|------|-----------|---------|---------|----------|-----------|
| 2 | 16 | 42 | Float64 | -0.660231093480 | 13 | 129.8 | 457 |
| 2 | 16 | 42 | Float32 | -0.660231080697 | 80 | 33.0 | 90 |
| 2 | 16 | 43 | Float64 | -0.660231093480 | 18 | 2.5 | 227 |
| 2 | 16 | 43 | Float32 | -0.660231106837 | 165 | 17.4 | 7 |
| 2 | 16 | 44 | Float64 | -0.660231093480 | 13 | 2.2 | 1 |
| 2 | 16 | 44 | Float32 | -0.660231070504 | 200 | 24.3 | 0 |
| 2 | 32 | 42 | Float64 | -0.660231093474 | 18 | 10.6 | 60 |
| 2 | 32 | 42 | Float32 | -0.660231167517 | 200 | 108.2 | 0 |
| 2 | 32 | 43 | Float64 | -0.660231093466 | 24 | 16.2 | 0 |
| 2 | 32 | 43 | Float32 | -0.660231137510 | 163 | 94.2 | 0 |
| 2 | 32 | 44 | Float64 | -0.660231093479 | 17 | 10.7 | 0 |
| 2 | 32 | 44 | Float32 | -0.660231166019 | 200 | 111.8 | 0 |

**Run 1 verdict:** energy |ΔE| ~1e-8 PASSES but n_steps blows up from 13–24 (Float64) to 80–200
(Float32 hitting `maxiter=200`), making Float32 wall-clock **6–10× SLOWER** than Float64 at χ=32.
Root cause: `gradtol=1e-7` equals Float32 epsilon, so LBFGS's `‖∇f‖` never drops below tol and
grinds until iteration cap. The clean energy passes because of LBFGS's robustness to small
gradient noise — not because precision is adequate for the convergence criterion.

## L4 D=2 — Run 2: with polish (`inner_etype_final_steps=2`)

**Config:** `inner_etype=Float32`, `inner_etype_final_steps=2`. First 2 AD iterations use
Float32 inner contractions; last 2 switch to Float64 to give LBFGS a clean gradient. Warmup
(non-AD) loop stays Float32 throughout.

BLAS threads: 4

| D | χ | seed | precision | E_final | n_steps | wall (s) | ΔRSS (MB) |
|---|---|------|-----------|---------|---------|----------|-----------|
| 2 | 16 | 42 | Float64 | -0.660231093480 | 13 | 62.6 | 439 |
| 2 | 16 | 42 | Float32 | -0.660231093480 | 13 | 12.5 | 98 |
| 2 | 16 | 43 | Float64 | -0.660231093480 | 18 | 1.2 | 231 |
| 2 | 16 | 43 | Float32 | -0.660231093480 | 18 | 1.2 | 13 |
| 2 | 16 | 44 | Float64 | -0.660231093480 | 13 | 0.9 | 0 |
| 2 | 16 | 44 | Float32 | -0.660231093480 | 13 | 0.9 | 0 |
| 2 | 32 | 42 | Float64 | -0.660231093474 | 18 | 3.9 | 58 |
| 2 | 32 | 42 | Float32 | -0.660231093478 | 21 | 5.3 | 0 |
| 2 | 32 | 43 | Float64 | -0.660231093466 | 24 | 5.5 | 1 |
| 2 | 32 | 43 | Float32 | -0.660231093478 | 22 | 4.1 | 0 |
| 2 | 32 | 44 | Float64 | -0.660231093479 | 17 | 3.6 | 0 |
| 2 | 32 | 44 | Float32 | -0.660231093479 | 20 | 4.0 | 0 |

**Run 2 verdict:** ✅ **both energy AND n_steps match Float64**.
- `|ΔE|` drops from 1e-8 → **3.3e-15 (χ=16)** and **4.7e-12 (χ=32)** — indistinguishable from Float64.
- `n_steps`: Float32-with-polish 13/18/13 (χ=16) and 21/22/20 (χ=32), within ±3 of Float64
  counterparts — no more `maxiter=200` grinding.
- wall-clock (excluding first-run JIT): parity with Float64 at χ=16 (~1s each) and near-parity
  at χ=32 (Float32 ~4-5s vs Float64 ~3.5-5.5s).

At D=2 the absolute savings are small because the contraction work is tiny. The real test of
"does Float32 save time" must be at larger (D, χ) — the polish eliminates the step-count
blowup so timing now actually reflects the per-call compute-cost difference. D=3 next.
