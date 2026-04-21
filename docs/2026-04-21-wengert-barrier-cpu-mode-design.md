# Wengert barriers — switch `:recompute` → `:cpu` mode

Date: 2026-04-21
Scope: One follow-up experiment to the Wengert CPU-offload integration.
Change 6 `Wengert.barrier(…; checkpoint=:recompute)` calls in
`qrctm_step_split` to `checkpoint=:cpu`. Measure GPU memory + wall time
impact at D=6 χ=192 and attempt D=7 χ=256 (which OOMed in prior round).

## Motivation — three orthogonal mechanisms

Prior round confused two distinct offload mechanisms. Re-stated precisely:

| Mechanism | What it offloads | Where it lives |
|---|---|---|
| `Wengert.@checkpoint expr` (tape flag) | **tape slot values** on `push_slot!` (intermediate storage) | `src/tape_ops.jl:20-27` |
| `barrier(…; checkpoint=:recompute)` | nothing; just skips zpb caching. Raw_args stay on device in zpb closure. | `src/api.jl:345-350` |
| `barrier(…; checkpoint=:cpu)` | **raw_args** (barrier inputs) to CPU; zpb closure also absent | `src/api.jl:352-362` |

Current code has the `@checkpoint` flag on (outer loop) and `:recompute`
mode on inner barriers. That leaves the barriers' `raw_args` alive on
device for the whole tape lifetime. At D=7 χ=256, with 20 AD steps × 6
barriers × ~3 args (e.g. env.C, env.T, M, U, R, C_new, T_new — ~26 MB each),
this pins roughly `20 × 6 × 3 × 26 MB ≈ 9 GB` of GPU memory just in zpb
closures — well before the observable backward runs.

D=7 χ=256 OOMed in `energy_value → contract_o_12 → FLmap` backward with
pool at ~45 GB. Releasing the ~9 GB held by QRCTM barrier closures may be
enough to let the observable backward fit.

## Change

`src/boundary_algorithm/qrctm.jl:71-90` — 6 `Wengert.barrier(f, Zygote.pullback, args...; checkpoint=:recompute)` calls become `checkpoint=:cpu`. No other code touched.

## Tradeoffs

- **Forward cost**: `:cpu` adds a `Array(raw_arg)` copy per barrier (GPU→CPU). At D=7 χ=256, ~30 MB/arg × several args × 6 barriers × 20 steps ≈ 20 GB GPU→CPU traffic per forward pass. At PCIe Gen 4 bandwidth (~25 GB/s), that's ~1 second extra. Small relative to ~39 s forward.
- **Backward cost**: each barrier's backward reloads raw_args CPU→GPU before calling `Zygote.pullback`. Symmetric cost to forward.
- **Memory**: frees ~9 GB of device memory that is currently pinned in zpb closures at D=7 χ=256. May unblock the observable backward.
- **Numerical**: none — `:cpu` just shifts storage location, does not change math.

## Test plan

Run the existing `examples/Heisenberg/Heisenberg_Square_QRCTM_wengert_smoke.jl` script (CuArray, 4090) under three conditions:

1. **D=6 χ=192, `:cpu` mode** — compare wall time to the prior-round `:recompute` numbers (39.3 fwd / 29.1 bwd / 68.4 total).
2. **D=7 χ=256, `:cpu` mode** — see if fg eval completes or still OOMs.
3. If still OOM: report where (likely the same observable backward). If it completes: capture wall time + `mem_peak`.

No unit test changes this round. Existing `test_autodiff.jl` CPU gradient equivalence already verifies `:cpu` produces the same gradient as `:recompute` (both are just storage-location variants of `:none`, ChainRules semantics preserved) — but we'll do a quick CPU-suite run after the change as a sanity check.

## Out of scope

- Extending Wengert to the observable backward path (that's the "Task C" option from prior session).
- Per-barrier tuning (e.g. `:cpu` on the heavy FLmap barrier only, `:recompute` on small ones).
- New dependency or API surface changes.
