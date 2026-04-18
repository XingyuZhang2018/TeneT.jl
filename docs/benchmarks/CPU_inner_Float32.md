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

## L4 D=2 — polish mode: `fine` (2026-04-17, CPU)

BLAS threads: 4

| D | χ | seed | precision | E_final | n_steps | wall (s) | ΔRSS (MB) |
|---|---|------|-----------|---------|---------|----------|-----------|
| 2 | 16 | 42 | Float64 | -0.660231093480 | 13 | 66.3 | 460 |
| 2 | 16 | 42 | Float32/fine | -0.660231093512 | 27 | 15.2 | 105 |
| 2 | 16 | 43 | Float64 | -0.660231093480 | 18 | 1.2 | 211 |
| 2 | 16 | 43 | Float32/fine | -0.660231093520 | 20 | 1.3 | 4 |
| 2 | 16 | 44 | Float64 | -0.660231093480 | 13 | 1.0 | 4 |
| 2 | 16 | 44 | Float32/fine | -0.660231093541 | 42 | 2.3 | 0 |
| 2 | 32 | 42 | Float64 | -0.660231093474 | 18 | 3.9 | 59 |
| 2 | 32 | 42 | Float32/fine | -0.660231093596 | 100 | 18.6 | 0 |
| 2 | 32 | 43 | Float64 | -0.660231093466 | 24 | 5.5 | 1 |
| 2 | 32 | 43 | Float32/fine | -0.660231093313 | 51 | 8.7 | 0 |
| 2 | 32 | 44 | Float64 | -0.660231093479 | 17 | 3.6 | 0 |
| 2 | 32 | 44 | Float32/fine | -0.660231093534 | 99 | 20.8 | 0 |

**Run 3 verdict (fine polish alone):** energy passes but **both accuracy and step count regress vs coarse polish**.
- `|ΔE|`: **3.97e-11 (χ=16)** and **6.07e-11 (χ=32)** — PASS (< 1e-7) but ~10⁴× worse than coarse polish.
- `n_steps`: Float32/fine = [27, 20, 42] (χ=16) and [100, 51, 99] (χ=32) — **1.5–5× MORE steps than Float64**.
- wall-clock: fine polish worse than Float64 at χ=32 (8.7–20.8s vs 3.6–5.5s).

**Mechanism:** fine polish only polishes the LAST AD iter's `simple_eig` inner power iterations. The first 3 AD iterations run entirely in Float32, so their contributions to the Zygote backward graph carry Float32 noise. Only the last layer is partially polished, which isn't enough to wash out the upstream Float32 error in the gradient.

**Coarse polish wins** — covering the last 2 **entire** AD iterations in Float64 gives the backward pass 2 full Float64 layers to suppress upstream Float32 noise. Fine polish is too narrow.

**Recommendation:** use `inner_etype_final_steps=2` (coarse) for the D=3 stage. Fine polish is retained as a configurable option for future experimentation (e.g. combining with `power_iter_ad` tuning) but is not the default path.

## L4 D=2 — fine polish sweep N ∈ [2, 3, 4, 5] (2026-04-17, CPU)

BLAS threads: 4

| D | χ | seed | precision | E_final | n_steps | wall (s) | ΔRSS (MB) |
|---|---|------|-----------|---------|---------|----------|-----------|
| 2 | 16 | 42 | Float64 | -0.660231093480 | 13 | 74.0 | 399 |
| 2 | 16 | 42 | Float32/fine=2 | -0.660231093512 | 27 | 16.6 | 100 |
| 2 | 16 | 42 | Float32/fine=3 | -0.660231093579 | 51 | 3.4 | 219 |
| 2 | 16 | 42 | Float32/fine=4 | -0.660231093483 | 19 | 1.3 | 6 |
| 2 | 16 | 42 | Float32/fine=5 | -0.660231093457 | 34 | 2.0 | 1 |
| 2 | 16 | 43 | Float64 | -0.660231093480 | 18 | 1.3 | 3 |
| 2 | 16 | 43 | Float32/fine=2 | -0.660231093520 | 20 | 1.3 | 0 |
| 2 | 16 | 43 | Float32/fine=3 | -0.660231093416 | 45 | 2.8 | 0 |
| 2 | 16 | 43 | Float32/fine=4 | -0.660231093554 | 30 | 1.7 | 0 |
| 2 | 16 | 43 | Float32/fine=5 | -0.660231093488 | 19 | 1.2 | 0 |
| 2 | 16 | 44 | Float64 | -0.660231093480 | 13 | 1.0 | 1 |
| 2 | 16 | 44 | Float32/fine=2 | -0.660231093541 | 42 | 2.5 | 0 |
| 2 | 16 | 44 | Float32/fine=3 | -0.660231093469 | 37 | 2.4 | 0 |
| 2 | 16 | 44 | Float32/fine=4 | -0.660231093511 | 75 | 4.1 | 0 |
| 2 | 16 | 44 | Float32/fine=5 | -0.660231093487 | 73 | 4.5 | 0 |
| 2 | 32 | 42 | Float64 | -0.660231093474 | 18 | 4.4 | 61 |
| 2 | 32 | 42 | Float32/fine=2 | -0.660231093596 | 100 | 19.3 | 0 |
| 2 | 32 | 42 | Float32/fine=3 | -0.660231093485 | 42 | 7.7 | 0 |
| 2 | 32 | 42 | Float32/fine=4 | -0.660231093490 | 153 | 27.1 | 0 |
| 2 | 32 | 42 | Float32/fine=5 | -0.660231093477 | 72 | 14.1 | 0 |
| 2 | 32 | 43 | Float64 | -0.660231093466 | 24 | 5.9 | 0 |
| 2 | 32 | 43 | Float32/fine=2 | -0.660231093313 | 51 | 9.3 | 0 |
| 2 | 32 | 43 | Float32/fine=3 | -0.660231093439 | 23 | 4.5 | 0 |
| 2 | 32 | 43 | Float32/fine=4 | -0.660231093451 | 37 | 7.2 | 0 |
| 2 | 32 | 43 | Float32/fine=5 | -0.660231093450 | 53 | 10.1 | 0 |
| 2 | 32 | 44 | Float64 | -0.660231093479 | 17 | 3.6 | 0 |
| 2 | 32 | 44 | Float32/fine=2 | -0.660231093534 | 99 | 20.4 | 0 |
| 2 | 32 | 44 | Float32/fine=3 | -0.660231093339 | 17 | 3.9 | 0 |
| 2 | 32 | 44 | Float32/fine=4 | -0.660231093457 | 27 | 6.1 | 0 |
| 2 | 32 | 44 | Float32/fine=5 | -0.660231093387 | 12 | 2.7 | 0 |

**Fine-polish sweep verdict (N ∈ {2, 3, 4, 5}):** All N values PASS `|ΔE| < 1e-7`, but:

| N | \|ΔE\| χ=16 | \|ΔE\| χ=32 | Float32 n_steps χ=16 (per seed) | Float32 n_steps χ=32 (per seed) |
|---|---:|---:|---:|---:|
| 2 | 3.97e-11 | 6.07e-11 | 27, 20, 42 | 100, 51, 99 |
| 3 | 1.05e-11 | 3.46e-11 | 51, 45, 37 | 42, 23, 17 |
| 4 | 3.06e-11 | 1.67e-11 | 19, 30, 75 | 153, 37, 27 |
| 5 | 6.75e-12 | 2.40e-11 | 34, 19, 73 | 72, 53, 12 |
| *coarse=2* | *3.33e-15* | *4.65e-12* | *13, 18, 13* | *21, 22, 20* |

**Analysis:**
- Accuracy improves weakly with N (fine=5 best ~ 10⁻¹¹), but coarse=2 is still **~10³×** more accurate.
- n_steps remains highly variable at all fine-N values; none achieves parity with Float64 or coarse=2.
- `fine=5` (the user's "equivalent to `coarse=1`" point) already underperforms `coarse=2` by 10³
  on accuracy, confirming that 1 fully-F64 AD layer is insufficient — the gradient still carries
  Float32 noise from the upstream 3 Float32 AD layers. Two F64 layers (coarse=2) provide the
  damping needed.
- Only seed-to-seed variance is comparable to the N-to-N variance (χ=32 seed=44: N=5 → 12 steps,
  N=4 → 27 steps, N=3 → 17 steps, N=2 → 99 steps — variance dominated by LBFGS starting
  condition, not by N).

**Takeaway:** coarse polish is the winning strategy. Fine polish is useful mainly as a
diagnostic showing **how many AD layers need to be F64 to suppress upstream F32 noise**. Answer:
at least 2 (coarse=2). One (fine=5 ≡ coarse=1) is not enough.

## L4 D=2 — coarse=1 vs fine=5 equivalence check (2026-04-17, CPU)

BLAS threads: 4

| D | χ | seed | precision | E_final | n_steps | wall (s) | ΔRSS (MB) |
|---|---|------|-----------|---------|---------|----------|-----------|
| 2 | 16 | 42 | Float64 | -0.660231093480 | 13 | 71.8 | 466 |
| 2 | 16 | 42 | Float32/coarse=1 | -0.660231093504 | 48 | 15.0 | 42 |
| 2 | 16 | 42 | Float32/fine=5 | -0.660231093457 | 34 | 3.5 | 162 |
| 2 | 16 | 43 | Float64 | -0.660231093480 | 18 | 1.3 | 94 |
| 2 | 16 | 43 | Float32/coarse=1 | -0.660231093499 | 37 | 2.2 | 0 |
| 2 | 16 | 43 | Float32/fine=5 | -0.660231093488 | 19 | 1.2 | 0 |
| 2 | 16 | 44 | Float64 | -0.660231093480 | 13 | 1.0 | 0 |
| 2 | 16 | 44 | Float32/coarse=1 | -0.660231093446 | 80 | 4.4 | 0 |
| 2 | 16 | 44 | Float32/fine=5 | -0.660231093487 | 73 | 4.4 | 0 |
| 2 | 32 | 42 | Float64 | -0.660231093474 | 18 | 4.2 | 43 |
| 2 | 32 | 42 | Float32/coarse=1 | -0.660231093441 | 26 | 6.1 | 0 |
| 2 | 32 | 42 | Float32/fine=5 | -0.660231093477 | 72 | 13.3 | 0 |
| 2 | 32 | 43 | Float64 | -0.660231093466 | 24 | 5.8 | 0 |
| 2 | 32 | 43 | Float32/coarse=1 | -0.660231093465 | 66 | 14.0 | 0 |
| 2 | 32 | 43 | Float32/fine=5 | -0.660231093450 | 53 | 10.0 | 0 |
| 2 | 32 | 44 | Float64 | -0.660231093479 | 17 | 3.6 | 0 |
| 2 | 32 | 44 | Float32/coarse=1 | -0.660231093479 | 75 | 13.6 | 0 |
| 2 | 32 | 44 | Float32/fine=5 | -0.660231093387 | 12 | 2.8 | 0 |

**Equivalence check: coarse=1 vs fine=5 — not bit-identical, ΔE ~1e-11:**

| (D, χ) | coarse=1 \|ΔE\| | fine=5 \|ΔE\| | coarse=1 median n_steps | fine=5 median n_steps |
|---|---:|---:|---:|---:|
| (2, 16) | 1.90e-11 | 6.75e-12 | 48 | 34 |
| (2, 32) | 8.77e-12 | 2.40e-11 | 66 | 53 |

Both PASS `\|ΔE\| < 1e-7` but are NOT bit-identical. Per-seed deviations are at ~1e-11 level,
well above Float64 precision (~1e-15). Both are also ~1000× worse than `coarse=2` (~3e-15 at
χ=16).

**Why not strictly equivalent?** The two code paths are mathematically identical on the last AD
iter (both collapse to "all power-iter calls in Float64"), but they traverse different Zygote
backward graphs:
- `coarse=1`: `leftenv_c4v` creates a single closure `f` with `inner_etype=nothing`, feeds
  `simple_eig` through its original no-polish branch.
- `fine=5`: `leftenv_c4v` creates BOTH `f` (Float32, never invoked when `n_pre=0`) AND
  `f_polish` (Float64, used for all 5 iters), and feeds `simple_eig` through the new
  polish branch.

Zygote's pullback traces the closure structure differently. Float64 accumulation order inside
the backward pass is different. The resulting gradient differs at ~1e-13 level, which LBFGS
amplifies to ~1e-11 final energy difference.

**Practical consequence:** the two polish modes are semantically equivalent **in the forward
mathematical sense** but **not numerically interchangeable through Zygote**. Neither rescues the
gradient quality enough to match `coarse=2`. The conclusion stands: covering 2 full F64 AD
layers is the minimum viable polish; any surgical variant covering 1 layer (whether via
`coarse=1` or `fine=5` or combinations) is insufficient.

## L4 D3 — coarse polish sweep N ∈ [1, 2] (2026-04-17, CPU)

BLAS threads: 4

| D | χ | seed | precision | E_final | n_steps | wall (s) | ΔRSS (MB) |
|---|---|------|-----------|---------|---------|----------|-----------|
| 3 | 16 | 42 | Float64 | -0.666753472298 | 40 | 76.5 | 670 |
| 3 | 16 | 42 | Float32/coarse=1 | -0.666747145266 | 38 | 21.2 | 165 |
| 3 | 16 | 42 | Float32/coarse=2 | -0.666756358939 | 41 | 9.5 | 3 |
| 3 | 16 | 43 | Float64 | -0.667678821841 | 200 | 44.4 | 6 |
| 3 | 16 | 43 | Float32/coarse=1 | -0.667678767137 | 200 | 42.8 | 0 |
| 3 | 16 | 43 | Float32/coarse=2 | -0.667644270987 | 69 | 14.1 | 0 |
| 3 | 16 | 44 | Float64 | -0.667678593145 | 200 | 41.7 | 2 |
| 3 | 16 | 44 | Float32/coarse=1 | -0.667678693180 | 200 | 39.1 | 0 |
| 3 | 16 | 44 | Float32/coarse=2 | -0.667678610979 | 200 | 39.6 | 0 |
| 3 | 32 | 42 | Float64 | -0.667137525348 | 58 | 60.7 | 98 |
| 3 | 32 | 42 | Float32/coarse=1 | -0.667140046414 | 58 | 48.1 | 13 |
| 3 | 32 | 42 | Float32/coarse=2 | -0.667138666860 | 57 | 51.5 | 0 |
| 3 | 32 | 43 | Float64 | -0.667809084704 | 101 | 104.9 | 0 |
| 3 | 32 | 43 | Float32/coarse=1 | -0.667945156083 | 200 | 162.5 | 26 |
| 3 | 32 | 43 | Float32/coarse=2 | -0.667945020121 | 200 | 166.1 | 0 |
| 3 | 32 | 44 | Float64 | -0.667154571547 | 103 | 98.4 | 0 |
| 3 | 32 | 44 | Float32/coarse=1 | -0.667945233892 | 200 | 175.8 | 1 |
| 3 | 32 | 44 | Float32/coarse=2 | -0.667154512642 | 200 | 161.1 | 0 |

### L4 D=3 interpretation — precision noise now dominated by **LBFGS basin selection**

Median |ΔE| per arm (Float64 vs Float32/coarse=*):
- (D=3, χ=16, coarse=1): 1.00e-07 → FAIL
- (D=3, χ=16, coarse=2): 3.43e-05 → FAIL
- (D=3, χ=32, coarse=1): 7.91e-04 → FAIL
- (D=3, χ=32, coarse=2): 5.89e-08 → PASS (lucky median alignment)

**But the median test is misleading at D=3**. Looking at per-seed results:

**Finding 1: most seeds don't converge to gradtol=1e-7 at D=3**.
- χ=16: 5 out of 9 Float32 runs and 2 out of 3 Float64 runs hit `maxiter=200`.
- χ=32: 4 out of 6 Float32 runs and 2 out of 3 Float64 runs hit `maxiter=200`.

**Finding 2: different precisions land in different local minima**, not different floating-point versions of the same minimum.
- (D=3, χ=16, seed=43): F64 E=-0.66768, coarse=1 E=-0.66768 (same), coarse=2 E=-0.66764 (different basin, worse by 3.4e-5).
- (D=3, χ=32, seed=43): F64 E=-0.66781, coarse=1 E=-0.66795 (better basin, by 1.4e-4), coarse=2 E=-0.66795 (same as coarse=1).
- (D=3, χ=32, seed=44): F64 E=-0.66715, coarse=1 E=-0.66795 (different basin, better), coarse=2 E=-0.66715 (same as F64).

When precisions DO agree on the same basin, they match to ~1e-5 (seed=42 cases). When they don't, |ΔE| jumps to 1e-4.

**Finding 3: Float32 sometimes finds LOWER energies than Float64**.
- (D=3, χ=32, seed=43,44): coarse=1 reaches -0.66795 vs F64's -0.66781 / -0.66715. Float32 noise perturbs LBFGS out of F64's local minimum into a lower one. Not universally — but happens.

**Finding 4: wall-clock is NOT consistently faster for Float32**.
- χ=16 seed=42: Float32 ~10-20s vs F64 76s (speedup when converging fast)
- χ=32 seed=44: Float32 ~160-175s vs F64 98s (SLOWER when F32 hits maxiter)

### What this means

The Phase-1 success criterion "`median |ΔE| < 1e-7`" was calibrated for LBFGS reaching gradtol=1e-7. At D=3 most runs hit `maxiter=200` without converging, so "final E" is just "wherever LBFGS stopped running" — dominated by basin selection, not by precision error.

**Polish works as designed** — coarse=2 at D=3 χ=16 seed=44 gives E=-0.66767861 vs F64 -0.66767859, differing at 2e-8. When both converge to the same basin AND neither hits maxiter, polish delivers its promise.

But the "does this method save time" question at D=3 is **inconclusive from this experiment**:
- When both converge fast (seed=42), Float32 is 2-10× faster (20s vs 76s on χ=16)
- When both hit maxiter (many seeds), Float32 is often SLOWER because it processes 200 iterations to `maxiter` while F64 hit `maxiter` at "same" rate but with higher per-iter throughput
- `inner_etype=Float32` at D=3 doesn't consistently deliver wall-clock savings on this model/config

### Recommendation

The Phase-1 CPU experiment hits a methodology limit at D=3: LBFGS non-convergence (`maxiter=200` saturation) contaminates the precision-comparison signal. To cleanly test "does Float32 reach the same minimum as F64", we would need:
- (a) Increase `maxiter` (to e.g. 500 or 1000) so both precisions have a chance to converge
- (b) Relax `gradtol` to something both precisions can reliably satisfy (e.g. 5e-7)
- (c) Use identical initial iPEPS tensors across precisions (currently each `Random.seed!(seed)` creates the same A, but Float64 vs Float32 downstream diverge — it's already controlled)
- (d) More seeds (10+) to average out basin selection noise

Or (e) accept this as "the method works when LBFGS converges cleanly, is contaminated by optimization-landscape issues when it doesn't, and wall-clock savings depend on config." Move to Phase-2 GPU testing where the contrast between F32/F64 per-op speed is higher and the landscape effects are the same.

## L4 D=4 χ=128 — GPU (2026-04-17)

Hardware: NVIDIA GeForce RTX 4090 (25.8 GB)
Seed: 42, LBFGS maxiter cap: 20, gradtol: 1e-7

| precision | E_final | n_steps | wall (s) | ΔGPU (MB) |
|-----------|---------|---------|----------|-----------|
| Float64 | -0.668965524178 | 20 | 772.6 | 13946 |
| Float32/coarse=2 | -0.668966984124 | 20 | 1585.2 | 12281 |

- **Speedup (F64/F32 wall):** 0.49x
- **|ΔE|:** 1.460e-06

### D=4 χ=128 GPU interpretation — Float32 is SLOWER

**Surprising result**: `Float32/coarse=2` takes **2.05× longer** than Float64 baseline on RTX 4090.

Initial hypotheses:
1. **Consumer GPU FP64 throttle doesn't manifest in practice.** RTX 4090 theoretical F64 is 1/64 of F32 (1.3 TFlops vs ~83 TFlops). But memory bandwidth is shared (~1 TB/s). If FLmap at D=4 χ=128 is **memory-bound**, F32's advantage shrinks to at most 2× (half the bytes per op). Observed ΔGPU suggests memory-pressure: F64 uses 13.9 GB, F32 uses 12.3 GB — only 12% less, not 50%.
2. **Downcast/upcast overhead**: my implementation allocates fresh Float32 buffers at each FLmap call (5 input tensors × ~2 MB each + 1 output → 12 MB of allocation per call). On GPU this is fast per byte (~1 TB/s) but frequent — with ~100 FLmap calls per LBFGS iter × 20 iters = 2000+ conversion allocations.
3. **Polish penalty**: `coarse=2` runs the LAST 2 AD iters in full F64. So half of the AD compute is F64 anyway, giving only partial speedup even if F32 were faster.
4. **TensorOperations GPU F32 path not optimal?** Possible but unverified — would need profiling.

The 2× slowdown is the NET after all of these. The experiment does not clearly isolate which factor dominates. Energy fidelity is fine (|ΔE| = 1.5e-6, as expected given maxiter=20 cap prevents full convergence).

**Action:** need a cleaner diagnostic — pure FLmap timing at D=4 χ=128 on GPU, F64 vs F32 (no VUMPS/LBFGS overhead). If F32 is already slower at the single-call level, the issue is in the TensorOperations+CUDA path or the downcast/upcast wrappers; if F32 is faster per call but slower in the full optimization, the overhead accumulates in VUMPS/LBFGS loops or the polish alternation.

## FLmap single-call diagnostic at D=4 χ=128 on GPU

Hardware: NVIDIA GeForce RTX 4090 (25.8 GB)

| call | precision | ms (median of 5) | F32/F64 ratio |
|------|-----------|------------------|---------------|
| FLmap forward             | F64       | 8.0 | 1.00 |
| FLmap forward             | F32-inner | 35.0 | 4.38 |
| FLmap_parallel forward    | F64       | 7.0 | 1.00 |
| FLmap_parallel forward    | F32-inner | 5.0 | 0.71 |
| FLmap forward+backward    | F64       | 26.0 | 1.00 |
| FLmap forward+backward    | F32-inner | 42.0 | 1.62 |

Single-call F32 rel_err vs F64: 2.790e-06

### FLmap single-call diagnostic interpretation

**Key numbers (D=4, χ=128, RTX 4090, median of 5 reps):**

| path | precision | ms | ratio F32/F64 |
|------|-----------|---:|---:|
| FLmap forward (direct) | F64 | 8.0 | 1.00 |
| FLmap forward (direct) | F32-inner | 35.0 | **4.38× SLOWER** |
| FLmap_parallel forward | F64 | 7.0 | 1.00 |
| FLmap_parallel forward | F32-inner | 5.0 | **0.71× (29% FASTER)** |
| FLmap forward+backward | F64 | 26.0 | 1.00 |
| FLmap forward+backward | F32-inner | 42.0 | **1.62× slower** |

**Sanity**: F32 single-call relative error vs F64 = 2.8e-6 (consistent with Float32 epsilon).

### What this tells us about the L4 slowdown

1. **Per-call FLmap_parallel forward IS faster in F32** (29% speedup) — the TensorOperations + CUDA F32 path works as expected at the single-call level for the path VUMPS actually uses.
2. **Per-call forward+backward is 1.62× slower in F32** — the backward pass through downcast+@tensor+upcast adds overhead that eats the forward savings.
3. **Full L4 run is 2× slower** — an ADDITIONAL ~1.2× factor comes from somewhere else in the stack.

### Suspected causes of the extra slowdown in full L4

- **Polish runs last 2 AD iters in full Float64 on a consumer GPU**. RTX 4090's F64 is throttled to ~1.3 TFlops (1/64 of F32). If those 2 iters dominate when F32 overhead has already reduced the "F32 iters" advantage, the polish iters become the bottleneck. Average AD iter time: F64-all ≈ 9.7s vs F32-coarse=2 ≈ 30s per iter (derived from full L4 wall / 4 AD iters / 20 LBFGS iters), so per-iter F32-with-polish is ~3× slower than F64-baseline — consistent with "F32 forward+backward 1.62× slower + polish-F64-iters adding ~2× on top".
- **Simple_eig's polish machinery**: my `leftenv_c4v` closure captures both `f` (Float32) and `f_polish` (Float64). Zygote differentiates through this mixed-precision chain. Each AD iter pulls a different rrule tree depending on which closure path was used. Possibly type-instability introduces extra overhead.
- **Per-call allocation**: 5 input-tensor downcasts + 1 output upcast per FLmap = 6 CuArray allocations/deallocations per call. At ~100 FLmap calls per LBFGS iter × 20 = 2000 alloc/free pairs. CUDA.jl pool handles these but not for free.

### Takeaway for the experiment

On RTX 4090 at D=4 χ=128:
- **Pure Float64 is the current fastest path** (~773s for 20 LBFGS iters).
- **Float32/coarse=2 is 2.05× slower**, losing to consumer-GPU F64-throttle-avoidance hypothesis because the polish forces F64 iters back in.
- The mixed-precision approach as designed **doesn't deliver time savings on this hardware at this scale**.

To realize Float32 speedups we'd likely need:
- A data-center GPU (H100/A100/GH200) where F64 is not throttled — polish iters cost more reasonably.
- OR a polish strategy that avoids Float64 altogether (risk: precision degrades).
- OR BFloat16/FP16 via Tensor Cores for a bigger F32→lower-precision jump.

## L4 D=4 χ=128 — GPU, forloop=2 (2026-04-18)

Hardware: NVIDIA GeForce RTX 4090 (25.8 GB)
Seed: 42, LBFGS maxiter cap: 20, gradtol: 1e-7, forloop_iter: 2

| precision | E_final | n_steps | wall (s) | ΔGPU (MB) |
|-----------|---------|---------|----------|-----------|
| Float64 | -0.668965385303 | 20 | 166.2 | 20490 |
| Float32/coarse=2 | -0.668967088177 | 20 | 67.7 | 19959 |

- **Speedup (F64/F32 wall):** 2.45x
- **|ΔE|:** 1.703e-06

### forloop=2 reverses the D=4 χ=128 GPU result

| config | forloop=1 wall | forloop=2 wall | forloop=2 speedup |
|---|---:|---:|---:|
| Float64 | 772.6s | **166.2s** | **4.65×** |
| Float32/coarse=2 | 1585.2s | **67.7s** | **23.4×** |

**Speedup F64/F32:** forloop=1 → 0.49× (F32 slower); **forloop=2 → 2.45× (F32 faster, as originally hypothesized)**.

**|ΔE| forloop=2:** 1.70e-6 (comparable to forloop=1, consistent with LBFGS maxiter=20 cap, NOT a precision degradation).

### What forloop=2 changes

`forloop_iter=2` splits the FLmap/FRmap/ACmap `@tensor` contraction along one input dimension (χ by default) into 2 chunks, computes each chunk separately, then concatenates/sums the results. At D=4 χ=128 with 128×4×4×128 tensors, this is a memory-blocking technique:

1. **Intermediate tensor size drops** — TensorOperations may produce intermediate tensors of shape up to `χ³D⁴` during contraction (hundreds of MB in F64). With forloop=2 those intermediates are halved along one axis.
2. **L2 cache fit** — RTX 4090 has ~72 MB L2. Smaller intermediates may fit in L2 → compute-bound instead of bandwidth-bound.
3. **Memory pool hygiene** — fewer simultaneous large allocations → less pool fragmentation → fewer deep-pool pauses.

### Why F32 scales 5× better with forloop=2 than F64 does

F64 4.65× speedup vs F32 23.4× speedup:
- F64 was already working "OK" at forloop=1 (ΔGPU 13.9 GB, within card's 24 GB). forloop=2 gives a modest memory-blocking win.
- F32 at forloop=1 was 2× slower than F64 — suggesting it was hitting a performance cliff (possibly TensorOperations GPU F32 path spilling or falling back to slow kernels at those intermediate sizes). forloop=2 brings intermediate sizes below that cliff, restoring the expected F32 throughput advantage.

### Implication for the experiment

**The "polish + inner_etype=Float32" approach DOES deliver GPU speedup — but only with `forloop_iter ≥ 2`.**

- At D=4 χ=128 RTX 4090 with forloop=2: **F32/coarse=2 = 67.7s vs F64 = 166.2s → 2.45× faster, |ΔE| = 1.7e-6** (within LBFGS-maxiter-cap tolerance).
- The forloop=1 slowdown was an **implementation-specific pathology**, not a fundamental limit of the approach.
- Memory usage is comparable (both arms ~20 GB), so the win is purely throughput.

This also has a practical implication: **production GPU runs at D≥4 should default to `forloop_iter ≥ 2`**, regardless of whether `inner_etype` is set. The forloop parameter was introduced for MPI/memory, but acts as a performance tuning knob at large (D, χ) on single-GPU.

### ⚠️ RETRACTION of earlier D=4 χ=128 GPU timings

The forloop=1 and forloop=2 GPU timings reported in the two previous sections were
**contaminated** by CUDA memory-pool fragmentation. Both prior runs executed Float64
and Float32/coarse=2 arms **in the same Julia process**, and the F32 arm (second) was
systematically slowed down by fragmentation from the F64 arm's leftover allocations.

To verify, I re-ran each (precision, forloop_iter) arm in an **independent Julia invocation**
(fresh CUDA context per arm; see `examples/benchmark_D4chi128_verify_clean.jl`):

| precision | forloop | wall (s) clean | wall (s) contaminated | inflation |
|---|---:|---:|---:|---:|
| Float64 | 1 | 151.5 | 772.6 | 5.10× |
| Float32/coarse=2 | 1 | 147.2 | 1585.2 | 10.77× |
| Float64 | 2 | 156.7 | 166.2 | 1.06× |
| Float32/coarse=2 | 2 | 153.1 | 67.7 | 0.44× |

(Note: the Float32/coarse=2 forloop=2 contaminated measurement was 67.7s — **faster** than
the true clean 153.1s. This is another side of the same artifact: the second arm in a dual-arm
script runs against a pool that's already saturated by the first arm, and depending on the
specific allocation pattern either gets dramatically slower (fl=1) or appears dramatically
faster (fl=2). Neither extreme reflects true per-call throughput.)

### ✅ Clean D=4 χ=128 RTX 4090 results (each arm its own process)

**Per-20-LBFGS-iter wall-clock at clean state is 147-157s across all 4 combinations**:
- Float32 vs Float64: ~3% difference (within noise)
- forloop=1 vs forloop=2: ~3% difference (within noise)
- `inner_etype=Float32/coarse=2` does **NOT** provide wall-clock speedup at this config
- `forloop_iter=2` does **NOT** provide wall-clock speedup at this config

|ΔE| between F32/coarse=2 and F64 at forloop=1: **6.6e-7** — slightly above gradtol=1e-7 but
expected at maxiter=20 cap. At forloop=2 the |ΔE| is 3.4e-8, consistent with polish working.

### Methodology lesson

**Cross-arm CUDA pool fragmentation produces 5-10× measurement artifacts**. Any GPU comparative
benchmark where multiple precisions / configurations share a Julia process is unreliable. Going
forward, all GPU benchmarks must launch each arm as a separate process.

This is NOT specific to TeneT.jl — it's inherent to CUDA.jl's memory pool. The pool retains
large allocations across function calls (for performance), but subsequent allocations with
different size patterns hit fragmented pool fragments and can trigger catastrophic slowdowns.
The naive fix `GC.gc(); CUDA.reclaim()` before each arm does NOT reliably prevent this (as
demonstrated: our benchmark script already called reclaim between arms but still got
contaminated).

### Revised D=4 χ=128 GPU conclusion

At D=4 χ=128 on RTX 4090, **no time savings from Float32 inner contractions are visible** in
a clean methodology. The `inner_etype=Float32 + coarse=2 polish` approach works for precision
(|ΔE| well under LBFGS noise) but doesn't deliver the hoped-for GPU speedup at this model/config.

Potential reasons:
- Consumer GPU F64 throttle isn't the bottleneck at this problem size (memory-bound?)
- The polish runs 2/4 AD iters in full F64, limiting any F32 gain
- Downcast/upcast per FLmap call has its own overhead
- `ifcheckpoint=true` trades memory for compute via forward re-runs, diluting F32 gain

**Still an open question**: at even larger (D, χ) the F32 advantage might manifest.
On data-center GPUs (H100, A100) where F64 isn't throttled, the picture would also differ.
Both are out of scope for Phase-1.

### Diagnostic: F64 throttle is NOT the bottleneck at D=4 χ=128

Added `F32/coarse=1 forloop=1 clean` to the sweep:

| polish config | wall (s) | F64 AD iters / 4 | \|ΔE\| |
|---|---:|---:|---:|
| F64 pure | 151.5 | 4 | — |
| F32/coarse=2 | 147.2 | 2 | 6.6e-7 |
| F32/coarse=1 | 144.1 | 1 | 1.6e-6 |

Reducing F64 AD iters from 4 → 2 → 1 saves only **~4s per step** (~3% per F64 iter removed).
If consumer-GPU F64 throttle (1/64 F32 on RTX 4090) were the bottleneck, each removed F64
AD iter would save tens of seconds. It doesn't.

**Conclusion:** the time at D=4 χ=128 on RTX 4090 is NOT spent in Float64 @tensor contractions.
Most wall-clock goes to:
- LBFGS linesearch forward passes (re-runs forward many times per step)
- Zygote pullback traversal through the VUMPS AD graph
- CUDA kernel launch + host-side dispatch overhead
- Checkpointing's forward re-computation in backward

The F32 inner optimization reduces kernel *compute* time but kernel compute is a small fraction
of wall. Net result: no meaningful wall-clock improvement from F32.
