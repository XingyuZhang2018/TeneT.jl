# Distributed `energy_value` — Plaquette J1J2 Square (M6)

Date: 2026-06-16. Branch `claude/ecstatic-golick-e3f6f0`. Process: research → design →
adversarial review → implement → dual review, parity-gated.

## Goal & scope

Make `energy_value(J1J2{Square}, A, env::PlaquetteVUMPSEnv, params)` compute the physical
energy **block-distributed**, calling the slice2d transfer maps (`FLmap_slice2d_dist` etc.)
on χ-block tensors directly — instead of the current v1 path that gathers the whole
observation environment to FULL on every rank and runs the serial contraction. This is the
user's explicit request ("energy_value 也需要像之前那样直接调用 FLmap_slice2d 来并行,先把
Plaquette Square J1J2 做了").

In scope: `energy_value(J1J2{Square}, ::PlaquetteVUMPSEnv)`, its sibling `imag_error(::Plaquette
VUMPSEnv)` (same env, called in the same `energy()`), and `ObsEnv(::PlaquetteVUMPSRuntime)`
(stop gathering). General `VUMPSEnv` path and all other models: UNCHANGED (still gather; v1).

## Why this is mostly reuse

Every expectation kernel is already built on the `*map_parallel` family + `dot`:

| bond | `energy_value` call | kernel chain (observable.jl) | slice2d replacement |
|------|--------------------|------------------------------|--------------------|
| J1H  | `contract_*_12`    | `oc_12` = 2× `FLmap_parallel` + `dot(conj(l),FRo)` | `FLmap_slice2d_dist` ×2 + `slice2d_dot` |
| J1V  | `contract_*_21`    | `oc_21` = 2× `ACmap_parallel` + `dot(conj(u),ACd)` | `ACmap_slice2d_dist` ×2 + `slice2d_dot` |
| J2\\ | `contract_*_22`    | `oc_22` = `oc_Q_22` + **`qrpos`(full χ)** + `dot(Q,QQ)` | see §oc_22 |
| diag (imag) | `contract_*_11` | `oc_11` = 1× `FLmap_parallel` + `dot(conj(l),FRo)` | `FLmap_slice2d_dist` + `slice2d_dot` |

`oc_Q_22` = `FLmap` → `ACdmap` → `FRmap` → `ACmap` (4-map chain, observable.jl:99-105). All
four have `*_slice2d_dist` variants (slice2d.jl 528/740/827/920), and every one already
accepts a `(Au,Ad)` tuple as `M` via `M1,M2 = M isa Tuple ? M : (M,conj(M))` — exactly the
double-layer A the observables pass. `slice2d_dot` (slice2d.jl:212) + its rrule
(rules.jl:487) exist. `ALCtoAC_slice2d` (slice2d.jl:349) exists.

So no new contraction primitive is needed. The work is **threading a `grid` through the
existing kwarg path** and the one genuinely-new piece: the oc_22 full-χ qrpos gather seam.

## Architecture — grid threading (4 levels, mirrors `ifparallel`/`forloop_iter`)

`ifparallel` and `forloop_iter` already flow kernel-ward as kwargs. `grid` rides the same
rails, defaulting to `nothing` (serial, bit-identical to today):

1. **oc kernels** `oc_11/oc_12/oc_21/oc_22/oc_Q_22` — add `grid = nothing` kwarg. Inside,
   `grid === nothing` keeps `FLmap_parallel`/`dot`; else `FLmap_slice2d_dist`/`slice2d_dot`.
2. **contract wrappers** `contract_{n,o}_{11,12,21,22…}` — add `grid = nothing`, forward to oc.
3. **`_contract_barebones`/`_contract_one`** (the `params::iPEPSOptimize` overloads,
   basic_interactions.jl:26-51) — additionally read `grid = params.boundary_alg.grid` and
   pass it as a kwarg alongside `ifparallel`/`forloop_iter`.
4. **callers** `energy_value(::PlaquetteVUMPSEnv)` and `imag_error(::PlaquetteVUMPSEnv)` —
   `AC = grid === nothing ? ALCtoAC(AL,C) : ALCtoAC_slice2d(AL,C,grid)`. (`imag_error` calls
   `contract_*_11` directly, so it passes `grid` explicitly; `energy_value` gets it for free
   through `_contract_*`.)

`A`, `O`, `M=(Au,Ad)` are replicated full tensors (no χ leg blocked) — passed unchanged. Only
the χ-tensors (FLo/FLu/AL/AC, and intermediate `l`/`u`/`Q`) are blocks.

## ObsEnv stops gathering (Plaquette only)

`ObsEnv(::PlaquetteVUMPSRuntime)` slice2d path currently computes the block obs env then
`gather_env` → full. Change: **return the BLOCK `PlaquetteVUMPSEnv`** (drop the gather).
energy_value/imag_error then consume blocks. The General `VUMPSEnv`/`Tuple` ObsEnv paths are
left gathering (their energy_value is still serial-on-full).

Block-tiling invariant: `FLo`,`FLu` come from `leftenv_slice2d` (standard convention: first χ
leg by r1, last by r2). `AL` from the runtime (same). `C` replicated. `AC = ALCtoAC_slice2d`
gathers AL → serial `ALCtoAC` → scatters → standard block convention. So every χ-tensor fed
to a slice2d map / `slice2d_dot` shares the identical disjoint tiling → `slice2d_dot` is a single
`+`-allreduce of local pieces (slice2d.jl:213), exactly right.

## oc_22 — the one non-trivial kernel

```
Q0   = rand block (χ,D1,D2,χ)         # Zygote.@ignore, see "rank-consistent Q0"
Q1   = oc_Q_22_slice2d(Q0, env_blocks…, grid)        # 4 slice2d maps, block→block
Q1f  = slice2d_gather(Q1, grid)                       # full χ (replicated)
Q,_  = qrpos(reshape(Q1f, χ*D1*D2, χ)); Q = reshape(Q,χ,D1,D2,χ)   # serial, replicated
Qb   = slice2d_scatter(Q, grid)                       # back to block
QQ   = oc_Q_22_slice2d(Qb, env_blocks…, grid)
return slice2d_dot(Qb, QQ, grid)                      # note: dot(Q,QQ), NOT dot(conj…)
```

The `gather → serial full-χ factorization → scatter` seam mirrors `ACCtoALAR_slice2d`
(slice2d.jl:370-374) exactly — a known, AD-correct pattern (gather take-my-block adjoint →
qrpos rrule replicated → scatter allreduce adjoint). `slice2d_gather`/`slice2d_scatter` are the
single-tensor differentiable wrappers (not the `*_struct` StructArray shims; Q is one tensor).

**Rank-consistent Q0 (correctness-critical).** All ranks MUST hold blocks of the *same* full
Q0, else the distributed maps contract mismatched blocks → garbage (not noise). Generate full
Q0 on rank 0, `MPI.Bcast` over `grid.comm`, slice to this rank's block — all inside
`Zygote.@ignore`. (A per-rank independent randn would be silently wrong.)

**Q0-noise finding (probe, 2026-06-16).** `oc_22` is one power-iteration of a qrpos-projected
plaquette transfer, so its value depends on the random Q0 at the **~1e-10** level: two
unseeded serial `contract_n_22` calls differ by 1.76e-10 (relative); same-seed → bit-exact
(0.0). This is a pre-existing property of the serial algorithm. Consequences:
- Slice2D inherits the same ~1e-10 floor (its own Q0 realization), exactly as a serial re-run
  would. Not a regression.
- The parity gate must not demand machine precision on the J2 term (see §gate).

## AD

energy_value is differentiated (checkpointed in `energy()`, ipeps_optimize/optimize.jl:31).
All slice2d pieces are AD-complete:
- `FLmap/ACmap/FRmap/ACdmap_slice2d_dist` — rrules (M3/M4).
- `slice2d_dot` — rrule (rules.jl:487): `dx=y·conj(d), dy=x·d`, **no cotangent allreduce**
  (consumed scalar). The observables consume the scalar exactly this way → correct.
- oc_22 seam: `slice2d_gather` (take-my-block adj) → `qrpos` rrule → `slice2d_scatter`
  (allreduce adj). Q0 is `@ignore` (no grad), as in serial.

`bond_checkpoint` recompute caveat: if `params.bond_checkpoint != Plain`, oc_22's forward is
re-run in backward, regenerating a *different* Q0 → ~1e-10 forward/backward mismatch. This is
pre-existing in serial; slice2d's only added requirement is that each (re)generation stays
rank-consistent (the Bcast guarantees it). The gate uses `bond_checkpoint=Plain` to keep the
parity clean.

## Parity gate (4-rank CPU, 2×2 grid)

On a real physical converged Plaquette A (or a well-conditioned random env), serial
energy_value (grid=nothing) vs slice2d energy_value (grid set), bond-by-bond from `e_dict`:
- **J1H, J1V** per bond: ≤ 1e-12 (exact kernels; CPU map reassociation only).
- **J2\\** per bond: ≤ 1e-8 (Q0 noise floor ≈ 2e-10 from two independent realizations;
  a wrong contraction would be O(1), so 1e-8 still proves algebraic correctness).
- **total energy** |E_slice2d − E_serial| ≤ 1e-8, and physical value reproduces the history
  −0.4713509925 to ≤ 1e-7 (the end-to-end check: slice2d leading_boundary + block ObsEnv +
  distributed energy_value, ifupdown via Plaquette path).
- ObsEnv block-vs-serial parity (existing M5o-3) preserved by test-side `gather_env`.

## Files

- `src/contraction/observable.jl` — `grid=nothing` on oc_11/12/21/22, oc_Q_22, and the
  contract_{n,o}_{11,12,21,22…} wrappers; slice2d branches.
- `src/models/basic_interactions.jl` — `_contract_barebones`/`_contract_one` (params) read &
  forward `params.boundary_alg.grid`.
- `src/models/J1J2/energy.jl` — `energy_value(J1J2{Square}, ::PlaquetteVUMPSEnv)`: slice2d AC.
- `src/boundary_algorithm/vumps/plaquette.jl` — `imag_error(::PlaquetteVUMPSEnv)` slice2d AC +
  grid; `ObsEnv(::PlaquetteVUMPSRuntime)` returns block.
- `src/boundary_algorithm/vumps/slice2d.jl` — oc_22 qrpos seam helper if needed; Q0 Bcast.
- `test/test_slice2d_energy_plaq.jl` (+ runner) — the parity gate.

## Risks / adversarial-review targets

1. **slice2d_dot conj convention** — oc_11/12/21 use `dot(conj(l),FRo)`; oc_22 uses
   `dot(Q,QQ)`. Must pass the matching args to `slice2d_dot` (it computes Σconj(x)·y). Verify
   each call site reproduces the serial `dot` argument order.
2. **Q0 rank-consistency** — Bcast is load-bearing; a missed Bcast is silent corruption, not
   a crash. Gate must run ≥4 ranks with a non-trivial pattern.
3. **block tiling match for slice2d_dot** — `l`/`u`/`Q` (map outputs) vs the FRo/ACd/QQ they
   dot against must share tiling. Map outputs follow input convention; verify FRo (=FLo slot)
   and ACd are standard-tiled.
4. **checkpoint recompute × Q0** — gate forces Plain; confirm production behavior is no worse
   than serial.
5. **imag_error inside ignore_derivatives** — no AD, but still does MPI collectives
   (slice2d_dot/maps) → must be rank-uniform (it is; called on every rank).
6. **double gather** — ALCtoAC_slice2d gathers AL, and each slice2d map gathers rows/cols
   internally; confirm this is the intended M4 "gather hoisting" cost, not accidental O(P²).
```

## Adversarial review R1 (opus) — fixes applied

Two BLOCKERs + SHOULD-FIX items found. Resolutions folded into the plan above:

- **[BLOCKER 1] grid leak via the shared `_contract_*` params overload.** During a *General*
  slice2d run `params.boundary_alg.grid` is also set; if the shared
  `_contract_barebones`/`_contract_one` auto-read `grid` from params, the General/Heisenberg
  `energy_value(::VUMPSEnv)` (which calls the same overload on a GATHERED-full env) would route
  to slice2d maps on full tensors → wrong/deadlock. **Fix:** do NOT touch basic_interactions.jl
  and do NOT auto-read grid there. The overload *already forwards `kwargs...`* to
  `checkpoint → contract_fn` (that's how `ifparallel`/`forloop_iter` reach the kernels today).
  So `energy_value(J1J2{Square},::PlaquetteVUMPSEnv)` passes `grid=grid` **explicitly** as a
  kwarg; every other caller passes nothing → serial. No shared-layer change.

- **[BLOCKER 2] model-blind `ObsEnv` block-return breaks Honeycomb-Plaquette.**
  `ObsEnv(::PlaquetteVUMPSRuntime)` dispatches on runtime only, but is shared by J1J2{Honeycomb}
  (uses `oc_13`, NOT slice2d-ized) and Heisenberg Plaquette. **Fix:** `ObsEnv` already receives
  `model` (optimize.jl:30). Gate the no-gather strictly to `model isa J1J2{Square}`; all other
  Plaquette models keep gathering to full (v1, untouched). Define ONE predicate
  `_dist_energy(model, alg) = alg.grid !== nothing && model isa J1J2{Square}` and use it in
  ObsEnv / energy_value / imag_error so "env is block" ⟺ "use slice2d contractions" everywhere.
  (For Honeycomb+grid: ObsEnv gathers → full env → imag_error/energy_value must take the SERIAL
  branch even though grid is set — hence the predicate guards on model, not just grid.)

- **[SHOULD-FIX 3] pin slice2d_dot conj per site.** `slice2d_dot(x,y)=Σconj(x)·y`. Serial
  `dot(conj(l),FRo)=Σl·FRo` ⇒ `slice2d_dot(conj(l),FRo,grid)`. `oc_21`'s `dot(conj(u),ACd)` ⇒
  `slice2d_dot(conj(u),ACd,grid)`. `oc_22`'s `dot(Q,QQ)=Σconj(Q)·QQ` ⇒ `slice2d_dot(Q,QQ,grid)`
  (no extra conj). conj is AD-safe.

- **[SHOULD-FIX 4] oc_22 seam wording.** Same *pattern* as ACCtoALAR_slice2d but single-tensor
  `slice2d_gather`/`slice2d_scatter` (NOT `*_struct`; Q is one tensor). Adjoints: gather
  take-my-block, qrpos rrule replicated, scatter allreduce — no Q-grad double-count.

- **[SHOULD-FIX 5] Q0 distribution.** Generate full Q0 on rank 0, `MPI.Bcast` over `grid.comm`,
  then `slice2d_scatter(Q0_full, grid)` to block (reuse the audited slicer, don't hand-roll the
  r1/r2 partition). All inside `Zygote.@ignore`.

- **[SHOULD-FIX 6] imag_error** must also use `ALCtoAC_slice2d` + pass `grid`, gated by the same
  `_dist_energy` predicate. It bypasses `_contract_*` (calls `contract_*_11` directly), so pass
  `grid` explicitly there.

- **[NIT 7] tolerances** made relative; J1H/J1V ≤1e-9, J2 ≤1e-8 (probe actual floor in gate);
  a wrong contraction is O(1) so these still prove correctness. **[NIT 8/9]** e/n replicated
  ratio + double-gather confirmed benign.

VERDICT (R1): sound after the two blocker fixes (env-typed gating, no shared-layer change).
