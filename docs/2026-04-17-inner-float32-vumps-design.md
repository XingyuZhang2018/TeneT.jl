# Inner Float32 in FLmap/FRmap/ACmap — Experiment Design

Date: 2026-04-17
Branch base: `iPEPS-unified`
Phase: 1 (CPU) — Phase 2 (GPU) out of scope here

## §1 Motivation & Goal

### Observation
In `Heisenberg_Square_VUMPS_C4v.jl`, running `FLmap_parallel` with `forloop_iter=1` vs
`forloop_iter=16` produces noticeably different FLmap results (driven by floating-point
summation order), yet the final iPEPS optimization energy is effectively unaffected.

### Hypothesis
VUMPS is a fixed-point iteration + eigensolver. It appears robust to numerical noise in
its dominant inner tensor contractions (`FLmap` / `FRmap` / `ACmap`, each O(χ³D⁴)–O(χ²D⁶)).
If the `forloop=16` reordering noise is tolerated, then Float32's ~1.2e-7 relative
error is likely also tolerated.

### Goal (Path C — experiment-first)
1. On `Heisenberg_Square_VUMPS_C4v.jl`, compare final iPEPS energy between:
   - Baseline: all-Float64
   - Experimental: `FLmap / FRmap / ACmap` internal `@tensor` in Float32,
     all other VUMPS machinery in Float64
2. If D=2 passes (|ΔE| < 1e-7), extend to D=3.
3. If D=3 also passes, output as "CPU experiment success" and feed into Phase-2
   (GPU) planning.
4. If D=2 fails, use L1 diagnostic to locate whether single-FLmap precision is already
   bad, or iteration amplifies it.

### Non-goals
- Do not change precision of QR / eigsolve / norm / tol — the boundary stays inside
  FLmap/FRmap/ACmap.
- Do not hand-write rrules — Zygote auto-differentiates `convert.(T, x)` + `@tensor`.
- Do not extend to other models (J1J2, Honeycomb, Kagome, Kitaev).
- Do not benchmark on GPU in this phase.

## §2 Architecture & Precision Boundary

```
iPEPS tensor A (Float64)
      ▼
  M = contract(A, conj(A))            Float64
      ▼
  VUMPS loop (Float64 state)
      │  per step:
      │  ├─ leftenv_c4v ── eigsolve/simple_eig ── FLmap_parallel ── forloop/parallel
      │  │                                                           └─ FLmap(...)
      │  │                                                              ├─ convert.(Float32, ...)
      │  │                                                              ├─ @tensor  (Float32)  ★
      │  │                                                              └─ convert.(Float64, result)
      │  ├─ ACenv_c4v  (same pattern for ACmap)
      │  ├─ Cenv_c4v   (unchanged — Cmap is O(χ³), not dominant)
      │  └─ QR, norm, tol                                             Float64
      ▼
  environments AL, C, FL              Float64
      ▼
  E = energy(A, M, FL, FR)            Float64
      ▼
  Zygote pullback → grad(A)           Float64 (precision capped ~1e-7 through Float32 segment)
      ▼
  LBFGS step                          Float64
```

### Design choices
- **Boundary inside FLmap (not at FLmap_parallel entry).** `forloop` slices inputs
  along a dimension; keeping the cast inside FLmap means forloop/parallel code and
  their rrules need no precision awareness.
- **Cmap / Lmap / Rmap / Mmap untouched.** Not dominant cost; keeping minimal scope.
- **Same treatment for all leg-type dispatches** (leg4 / leg5 / Tuple{leg5,leg5} / leg8)
  of each map.

### Subtle points
1. `convert.(Float32, A)` and `Float64.(A)` are ChainRules-native and Zygote handles
   them automatically; no manual rrule needed.
2. Per-call conversion means extra transient Float32 buffers of slice size
   (~1/forloop_iter of full). Small, acceptable overhead.

## §3 API Changes

### §3.1 VUMPS struct — new optional field
```julia
# src/boundary_algorithm/interface.jl
@kwdef mutable struct VUMPS{F <: ContractionMode} <: Algorithm
    # ... existing fields ...
    inner_etype::Union{Nothing, Type} = nothing   # nothing = keep current behavior
end
```

### §3.2 FLmap_parallel / FRmap_parallel / ACmap_parallel — new kwarg, closure dispatch
```julia
function FLmap_parallel(FL, ALu, ALd, M; ifparallel, forloop_iter, inner_etype=nothing)
    # N_in / N_out / size_out unchanged
    f = inner_etype === nothing ? FLmap :
        (args...) -> FLmap(args...; inner_etype)
    if ifparallel
        return parallel(f, FL, ALu, ALd, M; forloop_iter, N_in, N_out, size_out)
    else
        return forloop(f, FL, ALu, ALd, M; forloop_iter, N_in, N_out, size_out)
    end
end
```
`FRmap_parallel`, `ACmap_parallel`, `ACdmap_parallel` follow the same pattern.

`Mmap_parallel` / `Mumap_parallel` / `Mdmap_parallel` are left untouched (observable
path, not VUMPS-dominant — preserves minimal-boundary scope).

### §3.3 FLmap / FRmap / ACmap — precision branch in each method
```julia
# src/contraction/basic.jl
function FLmap(FL, ALu, ALd, M::leg4; inner_etype=nothing)
    if inner_etype === nothing || inner_etype == eltype(FL)
        @tensor result[c,e,h] := FL[a,d,f] * ALd[f,g,h] * M[d,g,e,b] * ALu[a,b,c]
        return result
    else
        T_out, T = eltype(FL), inner_etype
        FL_t  = T.(FL);  ALu_t = T.(ALu);  ALd_t = T.(ALd);  M_t = T.(M)
        @tensor result_t[c,e,h] := FL_t[a,d,f] * ALd_t[f,g,h] * M_t[d,g,e,b] * ALu_t[a,b,c]
        return T_out.(result_t)
    end
end
```
Same pattern for the `leg5`, `Tuple{leg5,leg5}`, `leg8` dispatches, and for FRmap / ACmap.
Count: 3 maps × ~3–4 leg-type dispatches ≈ 9–12 method bodies get the Float32 branch.

### §3.4 leftenv_c4v / ACenv_c4v — thread `inner_etype` through
```julia
# src/boundary_algorithm/vumps/c4v.jl
function leftenv_c4v(ALu, ALd, M, FL; alg, kwargs...)
    @unpack power_iter, ifparallel, forloop_iter, ifcheckpoint, inner_etype = alg
    f(FL) = ifcheckpoint ?
        checkpoint(FLmap_parallel, FL, ALu, ALd, M; ifparallel, forloop_iter, inner_etype) :
        FLmap_parallel(FL, ALu, ALd, M; ifparallel, forloop_iter, inner_etype)
    # remainder unchanged
end
# ACenv_c4v same pattern; Cenv_c4v unchanged
```

### §3.5 Unchanged
- `forloop` / `parallel` wrappers and their rrules
- `Cmap` / `Lmap` / `Rmap` / `Mmap`
- `simple_eig` / `eigsolve` / QR / `norm` / convergence judgments
- `init_ipeps` / `optimise_ipeps` / LBFGS / energy functions

### §3.6 Files touched
1. `src/boundary_algorithm/interface.jl`
2. `src/contraction/basic.jl`
3. `src/contraction/forloop_parallel_MPI.jl`
4. `src/boundary_algorithm/vumps/c4v.jl`

## §4 Experiment Design

### §4.1 L1 — single FLmap diagnostic
File: `examples/benchmark_inner_Float32_L1.jl`

Question: is single-call FLmap Float32 error comparable to (or smaller than) the
forloop=16 reordering noise?

```
Configs  : (D, χ) ∈ {(2,16), (2,32), (3,16), (3,32), (3,64)}
Seeds    : 3
Leg type : leg5 (bilayer iPEPS, matches Heisenberg_Square_VUMPS_C4v)
Compare  :
  R0  = FLmap_parallel(...; forloop_iter=1,  inner_etype=nothing)   # Float64 reference
  R16 = FLmap_parallel(...; forloop_iter=16, inner_etype=nothing)   # forloop reorder noise
  R32 = FLmap_parallel(...; forloop_iter=1,  inner_etype=Float32)   # precision under test
Metrics  :
  err_forloop = ‖R16 - R0‖ / ‖R0‖
  err_float32 = ‖R32 - R0‖ / ‖R0‖
  ratio       = err_float32 / err_forloop
Output   : Markdown table (D, χ, seed, err_forloop, err_float32, ratio)
Runtime  : < 1 minute on CPU
Gate     : ratio < 10 → proceed to L4
           10 ≤ ratio < 100 → proceed to L4 with caution
           ratio ≥ 100 → stop; reconsider precision strategy
```

### §4.2 L4 — full iPEPS energy comparison
File: `examples/benchmark_inner_Float32_L4.jl`

Question: does the final iPEPS energy under inner-Float32 match the all-Float64 baseline
to within `gradtol` (1e-7)?

#### Stage A — D=2 (must pass first)
```
Configs   : (D, χ) ∈ {(2, 16), (2, 32)}
Seeds     : 3
Axis      : precision ∈ {Float64, Float32}   (single-noise; forloop=1 only for L4)
            → 2 configs × 3 seeds × 2 precisions = 12 optimizations
Fixed     : tol=1e-10, gradtol=1e-7, maxiter=3, maxiter_ad=4,
            LBFGS(200; maxiter=200), HagerZhangLineSearch(maxfg=5)
Runtime   : ~10s per run → ~2 min total
```

#### Stage B — D=3 (only if Stage A passes)
```
Configs   : (D, χ) ∈ {(3, 16), (3, 32)}
Seeds     : 3
Axis      : same {Float64, Float32}
            → 12 optimizations
Runtime   : ~10 min per run → ~2 h total
```

#### Per-run metrics
- `E_final` (final energy)
- `n_lbfgs_steps`
- `wall_clock` (seconds)
- `peak_rss` (via `Sys.maxrss`; opportunistic — report if easy)
- `grad_norm_trajectory` (optional diagnostic)

#### Environment controls
- Fix `BLAS.set_num_threads(N)` to the same N for all configs to prevent thread-count
  differences masquerading as precision speedups.
- Same Julia session for all configs (eliminates startup / GC variance).

#### Seed statistics
3 seeds is the minimum to see mean + spread. If seed-to-seed spread on one (D, χ)
exceeds the median |E_F32 − E_F64|, expand to 5–10 seeds before judging.

## §5 Deliverables & Success Criteria

### §5.1 Phase-1 deliverables
1. **Source changes** (4 files in §3.6) with `inner_etype=nothing` default → zero impact
   on existing callers.
2. **Two benchmark scripts**:
   - `examples/benchmark_inner_Float32_L1.jl`
   - `examples/benchmark_inner_Float32_L4.jl`
3. **Result report**: `docs/benchmarks/CPU_inner_Float32.md`, containing:
   - Environment info (Julia version, BLAS threads, CPU model, memory)
   - L1 table: `(D, χ, seed) × err_forloop / err_float32 / ratio`
   - L4 table: `(D, χ, seed, precision) × E_final / n_steps / wall_clock [/ peak_rss]`
   - Decision section per §5.2
4. **This design doc**: `docs/2026-04-17-inner-float32-vumps-design.md`.
5. **Implementation plan**: `docs/2026-04-17-inner-float32-vumps-plan.md`
   (produced next by the writing-plans skill).

### §5.2 Staged success criteria

| Stage  | Criterion                                                      | On pass        | On fail                                                          |
|--------|----------------------------------------------------------------|----------------|------------------------------------------------------------------|
| L1     | `err_float32 / err_forloop < 10`                               | Proceed to L4  | If ratio ≥ 100, reconsider strategy (e.g. Float32×Float64 accum) |
| L4 D=2 | `median_seeds |E_F32 − E_F64| < 1e-7` and optimization does not diverge | Proceed to D=3 | Fine-grained L2/L3 diagnostics; write negative-result report     |
| L4 D=3 | same as above                                                  | Proceed to Phase-2 GPU planning | Record "Float32 fine for D=2 only on CPU"          |

"Does not diverge" = LBFGS step count ≤ 2× baseline, and energy trajectory is monotonic
or contains only noise-level oscillations.

### §5.3 Known limitations
- Under boundary (i) (precision cast inside FLmap/FRmap/ACmap only), **peak memory is
  expected to barely decrease** — the persistent AL/FL/C/M arrays stay Float64 between
  map calls; only transient @tensor buffers are Float32. If peak RSS shows no drop,
  this is expected, not a bug. To materially reduce memory, boundary (ii) (whole VUMPS
  inner loop Float32) would be needed — deferred as Phase-2 option.
- Only C4v Heisenberg square is tested. Conclusions do **not** automatically transfer to
  General / Plaquette / non-square / Kagome / Kitaev.
- CPU only. GPU Tensor Cores and FP64 throttling (e.g. RTX 4090's 1:64 ratio) can yield
  entirely different precision/speed tradeoffs — Phase-2 will re-measure.
- `maxiter=3, maxiter_ad=4` are optimization-practice values, not "converge VUMPS to
  tol=1e-10" values. Testing whether Float32 can converge to tol=1e-10 is a separate
  experiment, out of scope.

### §5.4 Phase-2 roadmap (not in current deliverables)
If Phase-1 passes:
- Run identical code on GPU (RTX 4090 first, then H100 / GH200).
- Evaluate boundary (ii) (whole VUMPS in Float32) to materialize memory savings.
- Evaluate BFloat16 / FP16 on Tensor Cores for large speedups.

## §6 Branch Strategy

### §6.1 Base
This work is based off `origin/iPEPS-unified` — `master` lacks `FLmap_parallel`,
`forloop_iter`, and `VUMPS{C4v}`.

The worktree branch `claude/thirsty-heisenberg-6de4de` was reset (hard) to
`origin/iPEPS-unified` before any code changes — prior commits on the branch were
generic fixtures, not protected work.

### §6.2 PR strategy
Deferred to implementation time. Default: prepare all Phase-1 deliverables as one
coherent PR targeted at `iPEPS-unified` (not `master`). User may override.
