# Deterministic Contraction Engine + Full-VUMPS Cannon Roadmap — Design

**Date**: 2026-06-12
**Status**: Direction approved by user ("cannon 实际落地本身也需要把所有的 map 都写好");
detailed design pending section approvals
**Evidence base**: `benchmarks/Sofia_VUB_H200.md` Part 6 (job 1275909) — the
staged + eager-free + hand-adjoint organization beats monolithic `@tensor` +
`forloop` + Zygote on every cell: fwd 1.2–1.5×, bwd 1.5–1.8×, ~40–55% lower
pool pressure, and the Zygote path needs ~2× the chunk count to fit at all.
**Predecessors**: `docs/2026-06-10-cannon-flmap-design.md` (FLmap v1–v3),
`docs/2026-05-11-2d-distributed-vumps-runtime-design.md` (per-map dataflow
analysis, partially superseded).

---

## Why one effort, not two

Landing Cannon in production (full VUMPS + iPEPS optimization) requires
distributed versions of every map `vumps_step` touches — FRmap, ACmap,
ACdmap, Cmap — and each one's local compute wants exactly the staged
organization FLmap v3 proved out. Meanwhile the A/B shows that organization
wins on a single GPU too. So the deterministic contraction engine is the
shared substrate: write the mechanism once, declare each map as data, get
both the single-GPU win across all production paths and the local-compute
layer of every cannon map.

Hand-writing per-map kernels does not scale: 21 map methods
(`src/contraction/basic.jl`) × ~3–5 pairwise stages × forward/accumulate/two
adjoints ≈ 200+ kernels. FLmap leg5 alone took 4 forward + 8 adjoint kernels.

## Layered architecture

```
L4  leftenv/rightenv/ACenv/Cenv (cannon) → vumps_step → VUMPS → iPEPS ∂E/∂A
L3  cannon distributed wrappers per map (grid comm + comm adjoints)
L2  forloop / parallel slicing (existing semantics, over chains)
L1  contraction-chain engine (this design)
```

### L1 — the engine

A map's kernel is declared as a **pairwise chain** — data, not code:

```julia
# FLmap leg5: result[d,g,h,l] = FL·ALd·M1·M2·ALu, current left-assoc order
const FLMAP_LEG5 = Chain(
    (:FL,  (:a,:e,:f,:i)), (:ALd, (:i,:j,:k,:l)),   # → H[a,e,f,j,k,l]
    (:M1,  (:e,:j,:g,:b,:p)),                       # → T
    (:M2,  (:f,:k,:h,:c,:p)),                       # → G
    (:ALu, (:a,:b,:c,:d));  out = (:d,:g,:h,:l))
```

- **Forward executor**: walks the chain via TensorOperations' runtime API
  (`tensorcontract!` with index tuples — routes to the same cuTENSOR calls
  as `@tensor`); every intermediate is an owned array, `unsafe_free!`d after
  its last use. Optional β=1 accumulation into a caller buffer for the first
  link (the ring/forloop accumulation pattern).
- **Generic adjoints**: for each link `C = A·B`, `dA = dC·conj(B)` and
  `dB = conj(A)·dC` with index sets derived mechanically from the link's
  labels. The backward walks the reversed chain recompute-style (rebuild
  intermediates, never capture χ²D⁴-class arrays in the rrule closure),
  freeing each array after its last consumer — the FLmap v3 ordering
  generalized.
- **One generic rrule** replaces the per-slice Zygote pullback inside the
  `forloop`/`parallel` rrules.

**Perf gate (must pass before migration)**: runtime `tensorcontract!` must
match `@tensor` within ~5% on representative shapes (cuTENSOR plans are
cached by descriptor, so parity is expected; if it fails, fallback is
`@generated` per-chain kernels emitted from the same chain tables — same
declarations, compile-time lowering).

**Order**: each chain initially fixes today's left-assoc order
(bit-compatible-ish parity, simple review); per-chain order overrides are a
later optimization.

### L2 — slicing unchanged

`forloop`/`parallel` keep their public semantics and slicing logic; the
kernel call inside becomes a chain execution, and their rrules delegate the
per-slice backward to the engine's generic adjoint walk instead of
`Zygote.pullback(f, ...)`.

### L3 — cannon wrappers per map

- **FRmap**: mirror of FLmap with directions flipped (ring along the row
  with FR blocks i↔l, a↔d swapped roles; column reduce-scatter on the first
  leg ↔ existing helpers cover both axes already).
- **ACmap / ACdmap**: AC is the sandwich center (both χ legs contracted) but
  the output legs come from FL's and FR's *outer* legs — under the uniform
  block convention the output lands cross-axis (the AllToAll problem the
  2026-05-11 design identified). **This is the one genuinely open dataflow
  design**; candidate resolutions (to be settled in the implementation-plan
  round with FLmap-level rigor): (a) a row/col block transpose after the
  map, (b) an alternate blocking for FL/FR inside ACmap, (c) the old
  design's AllToAll. Cmap is small (C is χ×χ, kept replicated per the
  2026-05-11 analysis).
- All wrappers reuse the existing comm primitives + adjoints (ring shift,
  row/col allgather + reduce-scatter, tags 700–760) and return block
  gradients.

### L4 — VUMPS integration

- `leftenv`/`rightenv` via distributed power iteration: `cannon_dot`/
  `cannon_norm` + `simple_eig` hooks are DONE and parity-proven (FP-level
  agreement with the serial eigenpair). Required before production: hoist
  the AL slice gathers out of the iteration loop (kill the per-call
  duplicate captures), persistent comm buffers (the `_comm_sendbuf`
  pattern), double-buffered iterate.
- `ACenv`/`Cenv`, then `vumps_step` (ACCtoALAR via gather+QR per the
  2026-05-11 v1 approach), `init_VUMPSRuntime`, ObsEnv, and finally the
  iPEPS energy/gradient chain with the existing checkpoint machinery
  (`Recompute()` segments compose with cannon and the segment boundaries
  become 1/P-sized blocks — cheaper than serial).

## Migration & testing

Map-by-map, parity-gated, old paths retained until verified:

| Milestone | Deliverable | Gate |
|-----------|-------------|------|
| M1 | Engine + FLmap leg5 chain (port of proven kernels) | == hand kernels bitwise-ish; perf ≥ hand kernels; 4-rank suite green |
| M2 | All FLmap/FRmap/ACmap/ACdmap/Cmap leg variants as chains; forloop/parallel rerouted | per-map parity vs @tensor 1e-12 fwd / 1e-10 grad; existing serial test suite green |
| M3 | cannon wrappers: FRmap, Cmap; ACmap dataflow design + impl | distributed parity per map |
| M4 | leftenv/rightenv/ACenv/Cenv cannon; gather hoisting; buffer reuse | env-level eigenpair parity; zero-allocation steady-state iteration |
| M5 | vumps_step → full VUMPS distributed | fixed-point trajectory match vs serial |
| M6 | iPEPS ∂E/∂A end-to-end | gradient parity 1e-8 (repo PR standard); Sofia production benchmark D=10+ χ≥512 |

Single-GPU regression guard throughout: the Part 6 A/B driver re-run per
milestone; any map slower than its @tensor original is a bug.

## Out of scope (unchanged from FLmap design)

NCCL send/recv fast path for the p2p primitives, 8×8/12×12 cross-node
campaigns (after M5), TSQR, leg8/C3v chains beyond what production needs
first.
