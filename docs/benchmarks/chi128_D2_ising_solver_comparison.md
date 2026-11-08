# Final TeneT Benchmark: 2D Classical Ising at χ=128 (corrected)

**Hardware:** NVIDIA GeForce RTX 4090
**Date:** 2026-05-07
**Branch:** `iPEPS-unified` of TeneT.jl 1.26 + GPUKrylov.jl v1.5

## Pre-existing bugs found and fixed

### Bug 1: `shermitian` typo in c4v.jl
`src/boundary_algorithm/vumps/c4v.jl` lines 46 & 67 had `shermitian=false` (should be `ishermitian=false`). This made `ifsimple_eig=false` unreachable on c4v path. Fixed.

### Bug 2: `ising_mpo` constructed wrong tensor
The original `ising_mpo` in `test/runtests.jl` built `M[s1,s2,s3,s4] = B(s1,s3)·B(s2,s4)` — **two decoupled 1D Ising chains, not 2D Ising**. Verified: M's dominant eigenvalue exactly equals `(2cosh β)²` (1D Ising²). Replaced with the standard 2D tensor `M[a,b,c,d] = sum_s W(s,a)·W(s,b)·W(s,c)·W(s,d)` using a δ tensor. With this fix, all VUMPS methods converge to Onsager 2D exact f.

### Bug 3 (in my own bench): `Base.delete_method` failed silently
Initial bench used `Base.delete_method` to remove GPUKrylov type-piracy before benching "native" KrylovKit. The deletion FAILED (silently — no error), so what I called "KrylovKit native" was actually still GPUKrylov. **The two reported numbers being identical was the smoking gun.** Fixed by phase-separating: KK native runs FIRST (before any GPUKrylov import), then type-piracy is registered, GPUKrylov runs.

## Phase-isolated benchmark results

### Off-critical: χ=128, β=0.43, tol=1e-6 (correlation length finite, easy)

| Method | Time (s) | err | f | Δf vs Onsager |
|---|---|---|---|---|
| simple_eig pi=5 | 1.13 | 9.4e-7 | -2.1285104706 | 2.90e-13 |
| simple_eig pi=10 | 0.61 | 7.4e-7 | -2.1285104706 | 2.18e-12 |
| **simple_eig pi=20** | **0.58** | 4.4e-7 | -2.1285104706 | 5.6e-13 |
| simple_eig pi=50 | 0.65 | 9.3e-8 | -2.1285104706 | 3.06e-14 |
| simple_eig pi=100 | 1.07 | 4.2e-7 | -2.1285104706 | 9.19e-14 |
| KrylovKit (native) | **8.76** | 8.3e-8 | -2.1285104706 | **1.15e-14** |
| **GPUKrylov** | **0.68** | 7.5e-8 | -2.1285104706 | **1.15e-14** |

**Headline:** GPUKrylov **13× faster** than KrylovKit native. Power_iter=20 sweet spot at 0.58s is fastest overall but slightly less accurate (Δf=5.6e-13 vs 1.2e-14).

Onsager `f_exact = -2.1285104706`.

### At critical: χ=128, β_c = 0.4406867935, tol=1e-6

| Method | Time (s) | err | f | Δf | Status |
|---|---|---|---|---|---|
| simple_eig pi=5 | 12.05 | 8.3e-5 | -2.1096511516 | 9.17e-10 | ⚠ no-conv |
| simple_eig pi=10 | 16.01 | 1.9e-5 | -2.1096511438 | 8.65e-9 | ⚠ no-conv |
| simple_eig pi=20 | 23.98 | 8.3e-6 | -2.1096511445 | 7.94e-9 | ⚠ no-conv |
| simple_eig pi=50 | 29.02 | 1.0e-6 | -2.1096511444 | 8.05e-9 | ✓ |
| **simple_eig pi=100** | **18.76** | 8.9e-7 | -2.1096511441 | 8.37e-9 | ✓ |
| KrylovKit (native) | **1069.23** | 8.4e-7 | -2.1096511446 | 7.90e-9 | ✓ |
| **GPUKrylov** | **25.25** | 8.8e-7 | -2.1096511446 | 7.90e-9 | ✓ |

**Headline:** GPUKrylov **42× faster** than KrylovKit native (25s vs 17.8 minutes!). Power_iter ≤ 20 silently hit VUMPS maxiter=500 without reaching tol=1e-6.

Onsager `f_exact = -2.1096511525`.

## Key Findings

### 1. GPUKrylov speedup vs KrylovKit native
- **β=0.43**: 13× faster (off-critical, easy convergence)
- **β_c**: 42× faster (critical, hard convergence)

The factor grows at the critical point because:
- VUMPS needs more outer iterations (~80 vs ~5)
- Each iteration calls leftenv 3× → ~240 vs ~15 KK eigsolve invocations
- KK's per-call cost is ~4s due to sync overhead (precision-independent)
- GPUKrylov's per-call cost is ~0.1s (CGS2, batched cuBLAS)

### 2. Finite-bond-dim error at χ=128 caps Δf at ~1e-8 to 1e-9 at criticality
At critical β_c, correlation length diverges → MPS at any finite chi has irreducible error. χ=128 gives Δf ~ 8e-9 regardless of solver. To get better f, must increase χ.

### 3. Power iteration "silent non-convergence" at critical
At β_c with `power_iter < 50`, VUMPS err saturates at 1e-5 to 1e-6 (above `tol=1e-6`) and hits `maxiter=500`. The reported Δf is small ONLY because finite-bond-dim caps it. **No safety against this** — the user has no signal that they should increase `power_iter`.

GPUKrylov's eigsolve has internal tol-based convergence and never silently fails.

### 4. simple_eig optimal `power_iter` is problem-dependent
- β=0.43: pi=20 sweet spot
- β_c: pi=100 needed (pi=50 barely scrapes by)

Without knowing the spectral gap a priori, users must run multiple times to tune. This is a real weakness in production.

### 5. The previous "leftenv micro-bench: KK 200-400× slower" result still holds
For c4v leg5 contractions (the actual iPEPS workflow path), the bra/ket doubling makes KK's sync wall even more dominant. The 13-42× factors here are for General leg4 (smaller contraction, sync less dominant).

## Final recommendations (honest assessment)

The earlier "GPUKrylov 200-400× faster" came from a `KrylovKit vs GPUKrylov` per-call microbenchmark. In actual VUMPS loops, with `simple_eig` available as a third option, the picture is different:

**Tuned `simple_eig` wins on raw wall-clock in both regimes:**
- β=0.43: simple_eig pi=20 → 0.58s, GPUKrylov → 0.68s (17% faster)
- β_c: simple_eig pi=100 → 18.76s, GPUKrylov → 25.25s (35% faster)

GPUKrylov's real practical value is **not speed-vs-simple_eig** but:
1. **Replacing KrylovKit on its own path** (10-42× win) — for code that requires `eigsolve` (e.g. AD `rrule` via GMRES adjoint)
2. **Robustness near critical / gapless regions** — power iteration silently hits maxiter without converging
3. **No `power_iter` tuning per (J, β, χ) combination**
4. **tol-based convergence guarantee** — `info.converged` is honest

| Scenario | Best method | Why |
|---|---|---|
| **iPEPS optimization, gapped phase (typical)** | **simple_eig pi ∈ [10, 50]** | Fastest; gapped transfer matrix needs few inner iters |
| **iPEPS optimization at quantum critical / near phase boundary** | **GPUKrylov** | simple_eig silently fails to converge at small spectral gap |
| **Replace KrylovKit `eigsolve` calls (e.g. AD adjoint via GMRES)** | **GPUKrylov** | KK native is sync-bound on GPU; 13-42× speedup |
| **Production code as a safety net** | **GPUKrylov** as fallback when simple_eig stalls | tol-based convergence catches what power can't |
| **Tuning-free, "just works" path** | **GPUKrylov** | No `power_iter` to set per problem |

## Files

- `timing/bench_general_ising_chi128.jl` — phase-isolated bench script
- `timing/bench_general_chi128_results.md` — β=0.43 raw output
- `timing/bench_general_critical_results.md` — β_c raw output
- `timing/sanity_general_ising_freeenergy.jl` — sanity check (Δf vs Onsager)
- `timing/debug_dispatch.jl` — diagnostic for the delete_method bug
- `src/boundary_algorithm/vumps/c4v.jl` — `shermitian` typo fix
- `test/runtests.jl` — `ising_mpo` correctness fix
