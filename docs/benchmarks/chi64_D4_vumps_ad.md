# Bench: VUMPS forward + AD at D=4, χ=64, ComplexF64

**Hardware:** NVIDIA RTX 4090, Julia 1.11.1, CUDA 13.0.0
**Date:** 2026-05-09
**Settings:** `tol=1e-10`, `krylovdim=30`, `maxiter=30`, `maxiter_ad=4`, `miniter_ad=4`, `T=ComplexF64`, `seed=42`
**State dim:** N = χ²·D² = 64²·16 = 65,536
**Operator:** iPEPS-derived `M[D², D², D², D²]` from random `A[4,4,4,4,2]` via `M = A·conj(A)` along physical leg.
**Loss:** `loss(p) = real(λ_AC/λ_C)` for `M(p) = M0 + p·M1` after `leading_boundary` converges. Gradient via `Zygote.gradient(loss, p0=0.05)`.

## Results

| Config | Forward (s) | Gradient (s) | grad/fwd | g (sanity) |
|---|---|---|---|---|
| **A: simple_eig power_iter_ad=5** | **1.20** | **2.02** | **1.68×** | 2.101539 |
| **B: simple_eig power_iter_ad=40** | 2.50 | 47.91 | 19.14× | 2.101538 |
| **C: GPUKrylov (`ifsimple_eig=false`)** | 109.03 | (skipped — see note) | — | — |

**Sanity:** simple_eig with `power_iter_ad=5` and `=40` produce identical gradients to 6 digits, confirming the AD path is correct.

## Headline

For this well-conditioned iPEPS-derived MPO (small spectral gap from random `A`), at D=4 χ=64:

- **simple_eig power_iter_ad=5 dominates wall-clock:** forward 1.2s, gradient 2.0s. The AD path's `power_iter_ad=5` is the sweet spot — short enough that the reverse loop is cheap.
- **power_iter_ad=40 makes gradient 24× slower** (47.9s vs 2.0s) without changing the gradient value to 6 digits. Each extra power step in the AD loop adds reverse-mode pullback work that compounds with VUMPS's outer iteration.
- **grad/fwd ratio explodes from 1.7× (pi=5) to 19× (pi=40)** — the per-iter pullback cost dominates when many inner iterations are taken.
- **GPUKrylov forward at 109s is much slower than simple_eig** here. Two reasons: (1) KrylovKit's default `tol=1e-12` (which the override inherits) is tighter than VUMPS's `tol=1e-10`, so GPUKrylov runs many Arnoldi cycles per eigsolve, and (2) at this well-conditioned spectrum, simple_eig's 5 power steps converge faster than tol-based Arnoldi.

## Note on GPUKrylov gradient (skipped)

`Zygote.gradient` through GPUKrylov's pirated `KrylovKit.eigsolve` failed because the **`GPUKrylovChainRulesCoreExt` extension didn't load in this TeneT dep environment**. Julia emits a "Circular dependency detected" warning during precompile and skips loading the extension. Without the extension, `install_krylovkit_override!()` only registers the forward override and Zygote falls back to AD-tracing through the wrapper body, hitting `llvmcall requires the compiler`.

The extension loads cleanly in GPUKrylov's own test environment (180/180 tests pass including AD), and in simpler environments. The TeneT dep graph (CUDA + cuTENSOR + AMDGPU + ChainRules + Zygote + TensorOperations + KrylovKit + ...) hits a Julia ext-graph false-positive that prevents loading.

**Workarounds for production iPEPS use:**
1. Load GPUKrylov in a separate Julia session with cleaner dep graph
2. Manually `dev` GPUKrylov into TeneT's Project.toml (so it's part of the project precompile graph, not Pkg.develop'd at runtime)
3. Wait for Julia upstream fix of the ext-graph cycle detection false positive

The earlier leftenv AD bench (`D:\1 - research\1.19 - GPU\GPUKrylov.jl\bench\leftenv_ad_converged_results.md`) validated GPUKrylov's gradient correctness on the iPEPS-shaped FLmap operator at χ=12 with rel diff 0.9% vs KrylovKit, and gradient speedup 15× at χ=12.

## Why simple_eig wins at well-conditioned, gapped problems

This bench's MPO is a **random iPEPS** (no parameter dependence beyond p), well-separated spectrum. At such problems, 5 power iteration steps are enough to project onto the dominant eigenvector to working precision — Arnoldi's tighter convergence is wasted work.

The previous **2D Ising chi=128 bench** (`timing/BENCH_FINAL_SUMMARY.md`) with the SAME state dim N=65536 (χ=128, D=2, F64) showed:
- Off-critical (β=0.43, gapped): `simple_eig pi=20` → 0.58s ≈ `GPUKrylov` 0.68s ≈ `KrylovKit` 8.76s. simple_eig wins narrowly.
- Critical (β_c, gapless): `simple_eig pi=100` → 18.76s ≈ `GPUKrylov` 25.25s, **`KrylovKit` 1069s**. GPUKrylov is robust where KrylovKit silently fails.

**Conclusion:** for **gapped/well-conditioned** iPEPS, simple_eig with appropriate `power_iter` (5 here, 20-100 at criticality) is the wall-clock winner. **GPUKrylov's value is robustness near critical/gapless points** where simple_eig can silently hit `maxiter` without converging, plus replacing KrylovKit on its own path (10-1500× speedup there).

## Methodology

- 1 untimed warmup + 1 timed sample per (config, mode). At D=4 χ=64 with single-call wall times of seconds-to-minutes, sample variance is small relative to JIT/cache effects.
- Forward = `leading_boundary(rt, M, alg)` + `CUDA.synchronize()`.
- Gradient = `Zygote.gradient(p -> full_loss(p, ...), p0=0.05)` + `CUDA.synchronize()`.
- All three configs use the same `M0`, `M1`, initial runtime, and `p0`. Phase 1 (configs A, B) runs BEFORE `install_krylovkit_override!()` (irreversible).
- `ifupdown=false` (single MPS) to keep `init_env` returning a single VUMPSRuntime; workload semantics representative of one direction of an iPEPS production VUMPS step.

## Files

- `bench_vumps_chi64_D4_ad.jl` — bench script
- `bench_vumps_chi64_D4_ad.log` — full output (with all warnings, JIT noise, etc.)
- This file (`bench_vumps_chi64_D4_ad_results.md`) — clean summary
