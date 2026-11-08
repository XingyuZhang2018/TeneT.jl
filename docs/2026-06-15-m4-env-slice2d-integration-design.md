# M4: env-level Slice2D integration + gather hoisting (distributed leftenv/rightenv/ACenv/Cenv)

Status: **Batch A COMPLETE — implemented, 4-rank CPU parity GREEN, dual-reviewed (both
approve-with-nits, all nits addressed).** leftenv_slice2d gates: Gate 1 eigenpair parity
**36/36** (χ∈{16,18} incl. uneven blocks; 1×1/1×2/2×2 + a repeated-pattern dedup cell),
Gate 2 hoisting equivalence **5/5** (forward bitwise + an AD sliced==dist single-application
equivalence), Gate 3a rrules **6/6** (complex; slice2d_norm tested with distinct per-rank
cotangent pieces), Gate 3b distributed-AD gradient parity **12/12** (1×1 Plain, **1×1
Recompute()** eig+segment checkpoint [R1-M2], 2×2 multi-cell dedup with nothing/near-zero
handling); existing run_test_slice2d.jl no-regression. Design adversarially reviewed (8-lens,
R1) + revised; the **slice2d_norm cotangent-allreduce** was a Batch-A discovery (§5.3). Dual
review applied: GPU-densify fix in the gather-wrapper rrules (collect→device-aware similar),
`_free!(dpartial)`, CRLF→LF on misc.jl/TeneT.jl (clean diff). **Batch B (rightenv_slice2d) + C
(ACenv_slice2d) ALSO COMPLETE + 4-rank CPU GREEN** (Gate B1 36/36, B3 3/3, C1 pass incl. Ni≠Nj,
C3 9/9 incl. the non-leading-cell `slice2d_norm` normalize under AD; dual review running). The
sliced-map surgery generalized: iterate keeps its per-call gather+reduce-scatter adjoint (FR
col / AC row), fixed partners hoisted (ARu/ARd, FL/FR) with the gather-wrapper reduce-scatters.
ACenv is the axis-transposed case (outer cell-col, chain cell-row, `selectpos(…,Ni)`, global
`slice2d_norm` on non-leading cells). **Batch D (Cenv_slice2d, replicated-output, take-my-block via
the existing `slice2d_gather` + serial Cmap + plain dot/norm) DONE + green.** **Sofia 16-GPU 4×4
(2-node) GPU validation PASSED** (job 1287306; all four env fns fwd+grad parity at χ256 D8 — fwd
~1e-15, grad ~1e-10, gates fwd 1e-10 / grad 1e-8; the test driver's `phase_relerr` needed a
GPU-safe `dot(b,a)/dot(b,b)` phase — CuArray disallows the `argmax`+scalar-index). **Batch E
(zero-alloc steady-state) gate MET by measurement** (Gate E: hoisted sliced map allocates LESS
per step than per-call dist — saved = the χ²D²/N gathers, ~67 MB/iter at production — AND the
per-iter alloc is FLAT/no-growth); the persistent-buffer pool + forward-only double-buffer are
**DEFERRED** (marginal — comm ≈5-11%, chain transients dominate + are recycled — and they mutate
the GPU-validated collectives). **Deferred (quality):** a shared chunk-loop helper (6 near-dup
slice2d rrules). **Remaining: F (NCCL χ·D/√P gating). M5 = vumps_step + QR gather seam.**
Branch: `claude/ecstatic-golick-e3f6f0` (M1+M2+M3+M3.5+NCCL landed; `CHAIN_ENGINE`
default ON; all 4 slice2d maps single-l-chunk ring-class with identical comm).
Milestone row: `docs/2026-06-12-deterministic-contraction-engine-design.md` L125 —
**M4 = leftenv/rightenv/ACenv/Cenv slice2d; gather hoisting; buffer reuse; gate =
env-level eigenpair parity + zero-allocation steady-state iteration.** (vumps_step
assembly is **M5**, deliberately out of scope here — see §11.)

Predecessors:
- `docs/2026-06-13-acmap-slice2d-dataflow-design.md` (M3 cross-axis dataflow),
- `docs/2026-06-15-m35-slice2d-ring-reorder-design.md` (M3.5 all-four-ring-class),
- `benchmarks/Sofia_VUB_H200.md` Parts 5/10/11/12 (perf evidence).

---

## 0. Evidence base (the seven-reader research sweep, 2026-06-15)

The current state was mapped by a fan-out research pass. The load-bearing facts
the design rests on (with the verifying site):

1. **Env layer is greenfield for slice2d.** No `Slice2DGrid`/`slice2d_*`/`grid::` anywhere
   in `src/boundary_algorithm/`. `leftenv`/`rightenv`/`ACenv`/`Cenv`/`vumps_step`
   (`src/boundary_algorithm/vumps/general.jl`) operate on full per-cell tensors in
   `StructArray`s; the only distribution is the **1D-slice** `parallel()`/`forloop()`
   path (`src/contraction/forloop_parallel_MPI.jl`), which splits ONE leg, replicates χ,
   and `allgatherv_p2p!`-stitches the full result. That is **orthogonal and
   incompatible** with the 2D-block Slice2D scheme (block-in/block-out, reduce-scatter).
2. **The slice2d maps are an AD-complete library with no production consumer.** Every
   `*_slice2d_dist`/`Cmap_slice2d` reference is in `slice2d.jl`, its rrule in `rules.jl`,
   or tests/benchmarks. M4 builds the env→map wiring from scratch.
3. **`simple_eig` distribution hooks exist but are unwired.** `simple_eig(f, v; …,
   inner_product=dot, norm_fn=norm)` (`src/utils/misc.jl:23-26`) accepts the hooks;
   `slice2d_dot`/`slice2d_norm` (`slice2d.jl:212-218`) are built for them; **no caller
   passes them.** The `_simple_eig_*map` wrappers (`general.jl:34-69`) forward only
   `power_iter, segment_checkpoint, f_final, final_polish_steps`.
4. **The gather-hoist hook is already clean.** `_slice2d_forward_sliced` (and the FR/AC/ACd
   `_*_slice2d_forward_sliced`) take **already-gathered** slices; the `*_slice2d_dist`
   wrappers are the only place the per-call gathers run. `FLmap_slice2d_dist`'s own
   docstring (`slice2d.jl:524-526`) flags hoisting as "the natural optimization
   (leftenv round)."
5. **The rrule structure splits cleanly along AD lines** (verified in every `*_dist`
   rrule, `rules.jl:589-1056`): the forward input gathers run **once** in the rrule body
   and are **captured**; the pullback re-gathers **only `dresult`** (the output-reduce-
   scatter adjoint, type A); the input-gather **reduce-scatters** (slice→block gradient,
   type B) run at the **end** of the pullback. Slice grads already accumulate with `.+=`
   into zeroed slice buffers.
6. **AD differentiates the env via `pullback(vumps_itr)`** (`rules.jl:1074`, the
   `leading_boundary` rrule) — the env's `simple_eig` power loop is unrolled by Zygote,
   recompute-style under `checkpoint(eig_checkpoint)`. So slice2d map rrules compose
   automatically; any hoisting must stay AD-transparent.
7. **The AD pass uses `power_iter_ad`** (`alg_ad.power_iter = alg.power_iter_ad`,
   `general.jl:945`), default 1, while the forward convergence loop uses `power_iter`
   (production C4v ≈ 5; under `ignore_derivatives`). **The big hoisting win is in the
   non-differentiated forward loop.**
8. **No env-level / hoisting / `simple_eig`-with-slice2d-hooks test exists** — only a
   hand-rolled FLmap power loop (`test/test_slice2d.jl:248-282`). A hoisting bug (stale
   slice across iterates, double-buffer aliasing) would pass every current test.
9. **Persistent comm buffers exist for the base collectives** (`_ensure_buf!` +
   `_comm_sendbuf/_comm_recvbuf/_comm_hostbuf`, `forloop_parallel_MPI.jl:24-51`) but the
   **slice2d collectives allocate fresh per call** (`slice2d.jl` ring/gather/RS helpers).
   No double buffering anywhere. `update!` (`environment.jl:209`) is the in-place
   double-buffer hook for runtime fields.
10. **Runtime types need no change.** `StructArray = {data::Vector{dense}, pattern}`; a
    block is just a smaller dense array. `randSA`/`ISA` already take per-entry block
    shapes. Obstacles are semantic: `norm`/`dot` are **local** on blocks (→ slice2d
    hooks); `FLint`/`FRint`/`cellones` infer shapes from global χ; QR (`ACCtoALAR`) and
    `C` are full/replicated (the M5 gather seam). Leg conventions already match the
    slice2d block convention exactly.
11. **Chain engine allocs intermediates per call** (`chain_apply`); only
    `chain_link1_add!` writes into a caller buffer. Literal zero-alloc of H/T/G needs
    into-buffer fold variants. `_free!` + CUDA-pool recycling already bound peak memory.
12. **FR/ACd distributed backward MUST use the `_SLICE2D` reordered chains** via
    `chain_backward`, NOT `engine_backward` (which uses the serial chains). FL/AC use the
    serial (already-ring) chains; Cmap carries FL (not C) and has no `engine_backward`.
13. **Benchmarks:** gather amortization is a *hypothesis* — measured **un-amortized**
    (2.7–5.5× fwd penalty at 16 GPU cross-node IB; "a few %" intra-node at 4 GPU).
    Backward is compute-bound (1.47–1.78× at 16 GPU). Production comm fraction ≈ 5% fwd /
    11% bwd (< 20% → no async overlap). NCCL crossover χ512@16GPU → χ1024@64GPU; gate on
    per-rank message χ·D/√P; hard precondition `χ%N==0` (`_equal_blocks`). 2³¹ cuTENSOR
    floor (`_ring_l_chunks`) must be preserved. Clean grids 16=4×4, 64=8×8 (32 no square).

---

## R1 — Adversarial review revisions (2026-06-15, 8-lens skeptic pass)

The 8-lens adversarial review **confirmed the core architecture** (hoisting via
differentiable primitives; the §5.2 linearity proof — FLmap term accounting exact,
reduce-scatters genuinely linear; the iterate-vs-fixed classification; take-my-block
for Cmap; the gather-once tag/sequencing safety). It also found concrete defects.
**Where R1 conflicts with any later section, R1 governs** (later sections are patched
inline where load-bearing; R1 is the binding delta for the rest).

### Blockers (wrong as written — fixed here)

- **R1-B1 `slice2d_dot` rrule formula was wrong for complex.** Julia `dot(x,y)=Σ conj(x)·y`
  is antilinear in x. Correct (ChainRules dense.jl convention; `ds` is the replicated
  allreduce-output cotangent → no extra comm):
  `rrule(slice2d_dot)(ds) = (NoTangent, y_blk·conj(ds), x_blk·ds, NoTangent)`.
- **R1-B2 `slice2d_norm` rrule formula was wrong** (dropped the real-projection). Correct
  form: `rrule(slice2d_norm)(dn) = (NoTangent, x_blk·(real(c̄)/n), NoTangent)` with **global**
  `n`. **Batch-A refinement (see §5.3):** `dn` is NOT replicated — it arrives as a per-rank
  PIECE from the `v/=norm` broadcast division, so `c̄ = allreduce(dn)`. (R1's "replicated,
  no comm" was wrong here; verified by Gate 3b — FP-exact only with the allreduce.)
  `slice2d_dot` stays local (consumed as a scalar). VERIFIED GREEN.
- **R1-B3 `orth_for_ad` uses a LOCAL `dot` (rules.jl:99: `_dv - dot(v,_dv)·v`) — wrong on
  blocks.** `simple_eig` returns `orth_for_ad(v1)` unconditionally (misc.jl:74); its
  backward projection is global on the full eigenvector but `dot` only sums the local
  block → the ⊥v projection under-subtracts the cross-rank term. **This is a third
  required hook, not just dot/norm.** Fix: add a grid-aware `orth_for_ad_slice2d(v, grid)`
  whose rrule uses `_dv - slice2d_dot(v,_dv,grid)·v`, and give `simple_eig` an `orth_fn`
  kwarg (defaulting to `orth_for_ad`) threaded by `leftenv_slice2d`/`rightenv_slice2d`/
  `ACenv_slice2d`. **Cenv is exempt** (C replicated → local dot == global). Verified
  first-hand at rules.jl:95-103.
- **R1-B4 Slice2D maps are leg5-ONLY; "General-mode core" wrongly implied leg4/leg8-M
  parity.** `_slice2d_stage1`/the `*_SLICE2D`/`ACMAP_LEG5` chains are hard-wired to leg5 FL
  + leg4 AL + leg5 M; only `Cmap_slice2d` handles leg3/leg4. The serial General env also
  serves **leg4 single-layer M (2D-classical Ising, leg3 FL)** and **leg8** — unachievable
  with today's slice2d maps. **Fix:** §1 restricts M4 to **leg5 single-layer-pair M (the
  iPEPS General path)**; leg4/leg8-M → §11 deferred (slice2d leg4/leg8 maps do not exist).
  M is passed as a single leg5 tensor; `*_slice2d_sliced` materializes `conj(M)` (no `_1M`
  slice2d chain — a perf, not correctness, gap, flagged).

### Majors

- **R1-M1 The sliced-core surgery was mischaracterized for FR/AC/ACd.** Their
  `_*_slice2d_forward_sliced` cores take **all** operands *already gathered, including the
  iterate* (`_frmap_slice2d_forward_sliced(ARd_g, FR_g, ARu_g, …)`, slice2d.jl:612). So
  `FRmap_slice2d_sliced` is **not** "reuse the core unchanged": it is a thin wrapper that
  gathers **only the iterate** per-call (`FR_g = _slice2d_col_allgather(FR_blk,…)`) then
  calls the existing core with the hoisted fixed slices. Its rrule keeps the iterate-gather
  adjoint (the iterate reduce-scatter, e.g. `dFR_blk`) per-call and returns the fixed
  operands' **slice** grads. **FLmap is the only clean case** (iterate FL is the ring
  carried block, never gathered → `FLmap_slice2d_sliced` calls `_slice2d_forward_sliced`
  directly). See patched §2.2b/§4.
- **R1-M2 The `eig_checkpoint=Recompute()` boundary was dropped.** The serial env wraps the
  whole solve in `checkpoint(eig_checkpoint, _simple_eig_FLmap, …)` (general.jl:296). For
  "gather fires once" to survive Recompute (which re-runs `f` under Zygote on backward),
  **the hoisted gathers must sit OUTSIDE the checkpoint**, and a new
  `_simple_eig_*_slice2d(iterate_blk, ALu_row, ALd_col, M, grid; inner_product, norm_fn,
  orth_fn)` wrapper (taking pre-gathered slices) must be the checkpointed target.
  `checkpoint` does splat kwargs through, so the hooks forward. Gate 3 gains a
  `Recompute()` variant to prove gather-once survives. See patched §2.3.
- **R1-M3 `slice2d_gather_full` duplicates the existing `slice2d_gather`** (rules.jl:450
  already has the take-my-block rrule). **Reuse `slice2d_gather`** for Cenv's full gather
  (or make `slice2d_gather_full` a documented alias); its rrule must slice **both** grid
  axes (`dfull[a_rs[r1+1], …, e_rs[r2+1]]`). The naming separation from
  `slice2d_gather_row/col` is by-convention only — add a **negative test** (feed Cenv's FL
  through the distributed wrapper, assert the ×P-wrong gradient) to pin the trap.
- **R1-M4 Cenv's replicated-C invariant is undefended.** Plain `dot`/`norm` on C is correct
  only if the replicas stay bit-identical; the hoist removes the per-step gather, so the C
  iterate runs collective-free per rank. **Precondition (state it):** the C iterate path
  must run only deterministic local ops (no reductions, no NCCL). Gate adds a per-solve
  cross-rank assertion `Allreduce(max|C - bcast(C,0)|, max) ≈ 0`.
- **R1-M5/M6 NCCL gating.** (a) The gate predicate must take the **GLOBAL χ** (the
  `MPI.Allreduce(size(blk,1),+,col_comm)` value, slice2d.jl:540) + grid P + replicated D
  — never a local block size — or NCCL-vs-MPI selection diverges per rank and **deadlocks**;
  assert the predicate returns the same value on all ranks. (b) The correct gate form is
  **χ·D/√P** (constant at both measured crossovers: 512/4 = 1024/8 = 128, ×D); the
  "benefit ~ χ²D²/√P" prose is the *payload*, not the crossover law. Calibrate
  `χ/√P ≳ 128` (Sofia H200, Part 12 jobs 1287248/1287257), default conservative. See
  patched §7.
- **R1-M7 Gate 1 (eigenpair parity) is wrong for the replicated C** — do NOT `slice2d_gather`
  C; compare the replicated `C_full` directly (after the cross-rank assertion). Add an
  **eigenvalue-gap guard** |λ1/λ2| > 1+ε so the eigenvector is non-degenerate (selectpos'
  degeneracy branch is dead under `ifsimple_eig`; a true degeneracy breaks phase-fix). See
  patched §9.
- **R1-M8 Gate 2 must split.** 2a: fresh-alloc (Batch A), **CPU/MPI same-backend bitwise**.
  2b: buffers-ON (Batch E) — the actual stale-slice/aliasing catch. Cross-backend (NCCL vs
  MPI) only 1e-10 (NCCL reduces in a different order). Add an Nj>1 non-leading-cell variant
  to cover hoisted-slice index reuse. Gate 2 must run **under AD** (gradient compare), not
  just forward, to catch the double-buffer aliasing (R1-M9).
- **R1-M9 Double-buffered iterate is UNSAFE under AD.** The forward aliases the iterate into
  the rrule-captured `blocks` (slice2d.jl:440-443; rrule reads `blocks[t+1]` in the
  pullback). Recycling the iterate storage overwrites a captured value before backward.
  **Permit double-buffering ONLY in the `ignore_derivatives` forward convergence loop;
  FORBID it on the AD-recorded `power_iter_ad` loop** (or defensively `copy` the iterate
  into `blocks[r2+1]` — one χ²D²/P copy). Implement as a separate forward-only recycle
  segment the AD path never enters. See patched §6.
- **R1-M10 The persistent buffer pool cannot be keyed by (role, shape).** Uneven-χ
  allgathers post N−1 **concurrent** Irecvs before one Waitall; same-shape peers (e.g.
  χ=18,N=4 → d_rs=[5,5,4,4]) need **distinct** live buffers. Key by **(role, peer-slot)**:
  a per-call Vector of grow-only pooled buffers, one per concurrent recv/send slot
  (sendbufs in the reduce-scatters are per-peer non-contiguous getindex copies → own slots
  too). Within a solve χ/D are fixed → slots stabilize after the first call. See patched §6.
- **R1-M11 The zero-alloc gate using gross `@allocated` is meaningless** — it counts every
  pool event and never nets out `unsafe_free!`, so it is a per-iteration constant dominated
  by the chain transients regardless of whether hoisting/buffer reuse is wired. **Replace
  with:** (a) net peak via `CUDA.memory_status()` reserved high-water before vs after
  warm-up, and (b) a **direct counter** of slice2d-collective `similar`/pool reallocations
  asserting it drops to 0 after warm-up (that is what "persistent buffers" must prove).
  Report gross `@allocated` only as a residual. See patched §9 Gate 4.
- **R1-M12 Mixed precision must fail LOUD, not silent.** `leftenv_slice2d` gets the same
  `alg`; if `alg.inner_etype`/`whole_vumps_etype`/`simple_eig_polish_steps` are set, the
  slice2d path would silently run full precision while serial downcasts. Add
  `@assert alg.inner_etype === nothing && alg.whole_vumps_etype === nothing` (+ polish) at
  each `*_slice2d` entry. Also `@assert alg.ifsimple_eig` (the eigsolve/GMRES branch is an
  implicit-adjoint that bypasses the slice2d_dot/norm/orth rrules → out of scope) and
  `@assert !ifobs && model === nothing` (obs/obs_index deferred — prevents silent wrong-row
  contraction from the omitted scaffolding).
- **R1-M13 ACenv axis transposition must be written out.** ACenv loops the unit-cell
  **column j** (outer), chains over the **row i**, uses `selectpos(…, Ni)`, non-leading
  `for i in 2:Ni` — the *transpose* of leftenv. The AC tensor's leg `i` (grid r1 split) is
  a **different `i`** from the unit-cell row index. Explicit `ACenv_slice2d` and
  `Cenv_slice2d` pseudocode added (§2.3'). Gate 1 adds an Ni≠Nj unit cell (e.g. 1×2) so a
  wrong axis cannot pass.

### Minors / nits folded in

- **R1-m1** Mixed-precision M5 trap: the slice-grad **upcast** must move to the gather
  wrapper (after its reduce-scatter), not stay in the sliced rrule, else the reduce-scatter
  sums at full precision instead of `inner_etype` → wrong mixed-precision gradient. Flagged
  in §11; do NOT copy the `*_dist` upcast verbatim.
- **R1-m2** Capture the hoisted slices as **immutable per-j locals / a Tuple**, NOT a
  mutated `Zygote.Buffer` — so Zygote accumulates the cotangent across the K reuses
  correctly. Micro-test: hoisted dALu_blk == K-summed dALu_row reduce-scattered once.
- **R1-m3** The §3 ACdmap row is **moot for M4** — no env function power-iterates ACdmap
  (its only consumer is the deferred obs env, applied once). No `ACdmap_slice2d_sliced`/
  `ACdenv_slice2d` in M4. Carried for the M5 obs path.
- **R1-m4** The FD micro-test of slice2d_dot/slice2d_norm/orth_for_ad_slice2d MUST use
  **complex** inputs + a complex upstream cotangent (real inputs hide the conj/real bugs).
- **R1-m5** Hard precondition: **`ifsimple_eig=true`** on the slice2d env (folded into R1-M12).
- **R1-m6** `slice2d_gather`'s take-my-block adjoint slices **both** grid axes (§ R1-M3).
- **R1-m7** The hoisted gathers run **unconditionally and identically on every grid rank**
  (the dedup decision is rank-uniform because the pattern is replicated) — required so the
  first-use NCCL comm-init (`_get_nccl_comm` Bcast, collective) cannot deadlock. Gate 5 adds
  a 16-GPU `TENET_USE_NCCL=1` leftenv+ACenv parity case to flush a comm-init ordering bug.
- **R1-m8** Gate 1 init: FL/FR/AC/C must be the **same fixed-seed full tensor scattered to
  blocks** (not an independent `randSA` per path). FLint determinism is not load-bearing
  (power method is start-invariant up to phase under a spectral gap).
- **R1-m9** Distinguish the **unit cell (Ni×Nj)** from the **process grid (N1×N2)**
  everywhere; Gate 1 must use a non-trivial unit cell (e.g. 2×2 with a 2-value pattern) so
  `processed_indices` dedup, `ir=mod1(i+1,Ni)`, and the Cenv `jr=mod1(j+1,Nj)` offset are
  actually exercised (today only the 2×2 process grid is in the matrix).

---

## 1. Scope (deliberately narrow)

M4 delivers **four distributed env functions** that find the boundary fixed points on
block-distributed tensors, with the gathers of the *fixed* boundary operands hoisted out
of the power iteration:

```
leftenv_slice2d (ALu_blk, ALd_blk, M, FL_blk, grid; alg)  →  (λL, FL_blk')
rightenv_slice2d(ARu_blk, ARd_blk, M, FR_blk, grid; alg)  →  (λR, FR_blk')
ACenv_slice2d   (AC_blk, FL_blk, M, FR_blk, grid; alg)    →  (λAC, AC_blk')
Cenv_slice2d    (C_full, FL_blk, FR_blk, grid; alg)       →  (λC, C_full')
```

**New functions, NOT a modification of the serial env functions.** The serial path
(`leftenv`/…) is byte-untouched (constraint: "旧路径保留到 parity 验证通过"). The
slice2d functions live alongside, dispatched explicitly by the (future M5) slice2d
`vumps_step`. M4's gate is per-function eigenpair parity vs the serial env + a
zero-allocation steady-state probe — **not** a full distributed `vumps_step`.

**In scope:** General-mode core (the eigenpair-parity gate): square grid N1=N2, the
power-method (`ifsimple_eig=true`) path, the unit-cell pattern dedup, gather hoisting,
persistent comm buffers, double-buffered iterate, size-gated NCCL, and the AD path
(sliced-map rrules + gather-wrapper rrules + slice2d_dot/slice2d_norm rrules) so the
functions are differentiable and ready for M5.

**Explicitly deferred to M5+ (noted, not built):** the QR gather seam
(`ACCtoALAR` on full tensors → where AC/C are gathered between Cenv and QR),
`vumps_step`/`vumps_itr`/`leading_boundary` slice2d assembly, `init_VUMPSRuntime`
block scatter, mixed-precision (`inner_etype`/`whole_vumps_etype`) on the slice2d env
path, the `ifobs` observation env, C4v/Plaquette modes, rectangular grids (M3 v2),
and literal-zero-alloc of chain intermediates (into-buffer fold variants).

Rationale for the narrow boundary: [[feedback_scope_choice]] — minimum-viable
optimization boundary beats covering all stated goals. The four env functions + their
AD are a coherent, independently-gateable unit; bolting on vumps_step/init/obs now would
balloon the diff and entangle the QR-gather seam (a genuinely new sub-design) into the
parity gate.

---

## 2. The hoisting architecture

### 2.1 The forward/AD asymmetry that de-risks everything

The env function is called in two contexts (`general.jl` vumps_itr):
- **Forward convergence loop** — `ChainRulesCore.ignore_derivatives`, `power_iter`
  iterations (production ≈5). **Not differentiated.** Here hoisting saves
  `power_iter × Nj` redundant gathers of the fixed boundary blocks per env solve — the
  dominant cross-node cost (Part 10's 2.7–5.5× fwd penalty).
- **AD loop** — recorded by Zygote, `power_iter_ad` iterations (default 1). Hoisting here
  is correct and gives a modest extra saving; the heavy lifting was already done in the
  forward loop.

A **single** `leftenv_slice2d` must serve both. The design achieves this with one uniform
mechanism (below) that is AD-transparent: in the forward context the rrules simply don't
fire; in the AD context they compose correctly.

### 2.2 Three differentiable building blocks (Option A: differentiable primitives)

The hoist = "gather once, iterate the already-sliced map, scatter the input gradients
once." Made AD-transparent by three new differentiable entry points, each with a
**hand-written rrule (no Zygote inside, eager `_free!`, densify guard)** — the established
slice2d discipline:

**(a) Gather wrappers** — `slice2d_gather_row(blk, grid, rs)` and
`slice2d_gather_col(blk, grid, rs)`, thin differentiable wrappers over the existing
`_slice2d_row_allgather` / `_slice2d_col_allgather`. Their rrule adjoints are the existing
reduce-scatters:
```
rrule(slice2d_gather_row): forward = _slice2d_row_allgather;  back(d) = _slice2d_row_reduce_scatter_last(d, grid, rs)
rrule(slice2d_gather_col): forward = _slice2d_col_allgather;  back(d) = _slice2d_col_reduce_scatter(d, grid, rs)
```
These are the **type-B input-gather adjoints** relocated from the per-call `*_dist`
rrule to a standalone primitive that Zygote runs **once** (the gather is called once,
outside the power loop). Pure comm, scalar-free, no χ²D⁴ tape → safe under the OOM
constraint. **Named distinctly on purpose:** their adjoint (reduce-scatter) is correct
only for the *distributed-output* maps (FL/FR/AC/ACd). The *replicated-output* Cmap uses
a different adjoint (take-my-block, §2.4) — keeping these as named distributed-only
wrappers prevents the wrong adjoint from being applied to Cmap (a latent cross-leak the
research flagged).

**(b) Sliced map entry points** — `FLmap_slice2d_sliced`, `FRmap_slice2d_sliced`,
`ACmap_slice2d_sliced` (NO `ACdmap_slice2d_sliced` — R1-m3: no M4 env iterates ACdmap). Each
takes the **hoisted fixed slices** and the **block iterate**. **Two structural cases
(R1-M1):**
- **FLmap (clean):** the iterate FL is the ring carried block, never gathered.
  `FLmap_slice2d_sliced(FL_blk, ALu_row, ALd_col, M, grid; forloop_iter)` calls
  `_slice2d_forward_sliced` directly.
- **FRmap/ACmap (iterate-gather):** the existing `_*_slice2d_forward_sliced` cores take ALL
  operands *already gathered including the iterate* (`_frmap_slice2d_forward_sliced(ARd_g,
  FR_g, ARu_g, …)`, slice2d.jl:612). So the sliced wrapper gathers **only the iterate**
  per-call (`FR_g = _slice2d_col_allgather(FR_blk,…)`; `AC_g = _slice2d_row_allgather(AC_blk,…)`)
  then calls the core with the hoisted fixed slices. It is **NOT** "reuse the core unchanged."

Each has a hand rrule = the `*_dist` rrule body MINUS the **fixed**-operand gathers (now
args) MINUS the **fixed**-operand type-B reduce-scatters (now in the gather wrappers), but
**KEEPING the per-call iterate gather + its reduce-scatter adjoint** (e.g. `dFR_blk =
_slice2d_col_reduce_scatter(dFR_g)` stays per-call). It returns **slice** gradients for the
hoisted fixed operands and the block gradient for the iterate:
```
FLmap_slice2d_sliced rrule returns:  (NoTangent, dFL_blk, dALu_row, dALd_col, dM, NoTangent)
                                      (^iterate)   (^slice)   (^slice)
```
The type-A output-adjoint allgather of `dresult` (`_slice2d_col_allgather(d_c,…)` /
`_slice2d_row_allgather(d_c,…)`) **stays inside** this rrule — it is per-iterate (a
different `v` each call) and cannot be hoisted. The per-l-chunk hand/`chain_backward`
adjoint walk, the `_ring_l_chunks` 2³¹ floor, the M `allreduce_p2p!`, the FillArrays
densify guard — all unchanged from the `*_dist` rrule. FR/ACd use the `_SLICE2D`
reordered chains via `chain_backward` (research finding #12); FL uses its hand kernels.

**(c) Distributed inner-product / norm** — `slice2d_dot`/`slice2d_norm` gain rrules so the
differentiated normalization (`v /= norm_fn(v)`) and eigenvalue
(`λ = inner_product(v, v1)`) are correct on blocks (§5.3).

### 2.3 What the env function does (`leftenv_slice2d`; the checkpoint boundary is load-bearing)

The hoisted gathers MUST sit **OUTSIDE** `checkpoint(eig_checkpoint, …)`; the checkpointed
target is a new `_simple_eig_FLmap_slice2d` that takes the **already-gathered** slices, so
`Recompute()` re-runs only the chain (not the gathers), and each gather's reduce-scatter
pullback fires **once** on the outer tape (R1-M2). Hoisted slices are captured as
**immutable per-j locals / a Tuple**, NOT a mutated `Zygote.Buffer` (R1-m2). The
`processed_indices` dedup mirrors the serial loop and is **rank-uniform** (the pattern is
replicated → the data-dependent `break` is identical on every rank → the collectives inside
do not desync; R1 deadlock). All gathers run **unconditionally on every grid rank** (R1-m7).

```
leftenv_slice2d(ALu_blk, ALd_blk, M, FL_blk, grid; alg):
  @assert grid.N1 == grid.N2                                  # square (FR/AC/ACd maps)
  @assert alg.ifsimple_eig                                    # eigsolve branch bypasses slice2d rrules
  @assert alg.inner_etype===nothing && alg.whole_vumps_etype===nothing && alg.simple_eig_polish_steps==0
  @assert !ifobs && model===nothing                           # obs/obs_index deferred (R1-M12, R1-m10)
  @assert eltype-leg(M)==5                                    # leg5 single-layer-pair M only (R1-B4)
  Ni, Nj = size(M);  processed = Set{Int}()
  for i in 1:Ni
    ir = mod1(i+1, Ni)                                        # serial down-partner (no ifobs in M4)
    p = FL.pattern[i,1];  p in processed && continue
    χ = MPI.Allreduce(size(ALu_blk[i,1], 1), +, grid.col_comm)
    a_rs = split_ranges(χ, grid.N1);  l_rs = split_ranges(χ, grid.N2)
    # HOIST (OUTSIDE checkpoint): gather the FIXED ALu/ALd slices once per column. Tuples.
    ALu_row = ntuple(j -> slice2d_gather_row(ALu_blk[i,j],  grid, l_rs), Nj)   # local-a, full-d
    ALd_col = ntuple(j -> slice2d_gather_col(ALd_blk[ir,j], grid, a_rs), Nj)   # full-i, local-l
    λLs, FLs = checkpoint(eig_checkpoint, _simple_eig_FLmap_slice2d,
                          FL_blk[i,1], ALu_row, ALd_col, M[i,:], grid;
                          power_iter, segment_checkpoint, forloop_iter)
    λL[i,1], FL_blk'[i,1] = selectpos(λLs, FLs, Nj);  push!(processed, p)
    for j in 2:Nj                                             # non-leading cycle cells: one sliced step
      p = FL.pattern[i,j];  p in processed && continue
      FL_blk'[i,j] = FLmap_slice2d_sliced(FL_blk'[i,j-1], ALu_row[j-1], ALd_col[j-1], M[i,j-1], grid; forloop_iter)
      λL[i,j] = λL[i,1];  push!(processed, p)
    end
  end;  return λL, FL_blk'

_simple_eig_FLmap_slice2d(FLij_blk, ALu_row, ALd_col, M_i, grid; power_iter, segment_checkpoint, forloop_iter):
  f(v) = (for j in 1:Nj; v = FLmap_slice2d_sliced(v, ALu_row[j], ALd_col[j], M_i[j], grid; forloop_iter); end; v)
  simple_eig(f, FLij_blk; power_iter, segment_checkpoint,
             inner_product = (x,y)->slice2d_dot(x,y,grid),
             norm_fn       = x   ->slice2d_norm(x,grid),
             orth_fn       = v   ->orth_for_ad_slice2d(v,grid))     # R1-B3 global ⊥ projection
```

- **In the forward loop** (`ignore_derivatives`): the gather wrappers and sliced maps run
  as plain functions → full hoisting benefit, zero AD overhead.
- **In the AD loop:** Zygote records. `slice2d_gather_row/col` fire once per (i,j) → their
  reduce-scatter adjoints run **once** in the backward. `FLmap_slice2d_sliced` fires
  `power_iter_ad × Nj` times; Zygote **accumulates** the cotangents flowing into each
  `ALu_row[j]`/`ALd_col[j]` across all those uses, then hands the summed slice cotangent
  to the gather wrapper's single reduce-scatter. This is exactly the **sum-then-scatter ≡
  scatter-then-sum** linearity the `*_dist` rrules already rely on (research #5) — the
  gradient is identical to the un-hoisted per-call `*_dist` path. §5.2 proves it.

### 2.4 Cenv / Cmap is the replicated-output special case

`Cmap_slice2d` gathers FL/FR to **full** (row+col allgather) and returns the full χ×χ
**replicated** C (no reduce-scatter). In `Cenv` the map iterates over FL/FR which are
**fixed**, so the FL_full/FR_full gathers hoist too:
```
Cenv_slice2d: FL_full[j] = slice2d_gather_full(FL_blk[i,jr], grid)   # hoisted, replicated
             FR_full[j] = slice2d_gather_full(FR_blk[i,j],  grid)
             f(C) = chain over i of Cmap_slice2d_sliced(C, FL_full, FR_full, grid)
             simple_eig(f, C_full[1,j]; inner_product=dot, norm_fn=norm)   # C replicated → LOCAL dot/norm OK
```
Crucial differences from §2.2:
- C is **replicated** (full χ×χ on every rank), so `simple_eig` uses **plain `dot`/`norm`**
  (every rank holds the identical full C — local == global). No slice2d hooks for Cenv.
- The hoisted gather is `slice2d_gather_full` (both axes to full), and its **adjoint is
  take-my-block, NOT reduce-scatter** (the replicated-output rule, `Cmap_slice2d` rrule
  comment `rules.jl:700-711`; reduce-scatter would over-count by P). This is why the
  distributed wrappers (§2.2a) are named separately — Cenv must not reuse them.
- `Cmap_slice2d_sliced` rrule returns `(dC_replicated, dFL_full, dFR_full)`;
  `slice2d_gather_full`'s rrule turns `dFL_full` → `dFL_blk` by take-my-block.

This is the one place a wrong adjoint would silently corrupt gradients; it gets its own
dedicated wrapper + a dedicated replicated-output gradient test (§9).

---

## 3. Per-function dataflow summary

**Notation (R1-m9):** the **unit cell** is Ni×Nj (lattice sites, `.pattern`); the **process
grid** is N1×N2 (the χ-bond MPI grid). They are independent — every cell tensor's χ legs are
block-distributed over the *same* N1×N2 grid. Below, `[j]`/`[i']` index the **unit cell**;
the tensor-leg labels (a/i/l on r1/r2) are the **grid** split. Do not conflate the unit-cell
row index `i` with the AC tensor's leg `i`.

| function | iterate (per-call gather) | fixed boundary (HOISTED once) | slice2d map (sliced) | inner_product / norm / orth | output |
|----------|---------------------------|-------------------------------|---------------------|------------------------------|--------|
| `leftenv_slice2d`  | FL_blk — **not gathered** (ring block) | ALu_row[j], ALd_col[j] (j=1:Nj) | `FLmap_slice2d_sliced` | slice2d_dot / slice2d_norm / orth_slice2d | FL_blk |
| `rightenv_slice2d` | FR_blk — col-gather per-call | ARu_g[j], ARd_g[j] (j=1:Nj) | `FRmap_slice2d_sliced` | slice2d_dot / slice2d_norm / orth_slice2d | FR_blk |
| `ACenv_slice2d`    | AC_blk — row-gather per-call | FL_g[i'], FR_g[i'] (i'=1:Ni, fixed column j) | `ACmap_slice2d_sliced` | slice2d_dot / slice2d_norm / orth_slice2d | AC_blk |
| `Cenv_slice2d`     | C_full — replicated, not gathered | FL_full[i'], FR_full[i'] (full gather, jr=mod1(j+1,Nj) offset) | `Cmap_slice2d_sliced` | dot / norm (replicated; no orth hook) | C_full |

\* FRmap/ACmap/ACdmap gather **three** operands (two cross-axis legs); these are the
ones whose forward gathers the `*_dist` rrule captures (research #5). For the iterate
itself (FR for rightenv, AC for ACenv) **only the fixed partners are hoisted** — the
iterate is re-fed each power step and is NOT gathered (it stays block-distributed; the
map gathers only its fixed neighbours). Concretely:
- `rightenv_slice2d` hoists `ARu_g`, `ARd_g`, **and `FR_g` is the iterate** — wait: in
  `FRmap_slice2d_dist(FR_blk, ARu_blk, ARd_blk, M, grid)` the **iterate is FR_blk** and
  the **fixed** operands are ARu/ARd. The forward gathers are on ARd (col), ARu (row),
  **and FR (col)** — but FR is the iterate, so its gather is **per-call, not hoisted.**
  Hoist only ARu_g (row, full-d) and ARd_g (col, full-i). `FRmap_slice2d_sliced` then
  takes (FR_blk_iterate, ARu_g, ARd_g, M, grid) and internally col-gathers FR_blk each
  call (the per-iterate gather, analogous to the type-A output adjoint — cannot hoist).
- Symmetrically `ACenv_slice2d`: iterate AC, fixed FL/FR → hoist FL_g (row, full-i) and
  FR_g (col, full-d); the map per-call row-gathers the iterate AC (AC_g, full-d).

**This is a refinement of "hoist all three gathers":** only the gathers of *fixed*
operands hoist; the gather of the *iterate* operand stays per-call. The implementation
must split each `*_slice2d_dist` into (fixed-operand gathers → hoist) + (iterate gather +
local chain + output RS → per-call `*_slice2d_sliced`). For FLmap the iterate FL is **not
gathered at all** (it is the ring's carried block) — both ALu and ALd are fixed and fully
hoisted, the cleanest case and the first to implement.

> **Design note (verify in Batch A):** confirm per map exactly which forward gathers are
> over *fixed* operands (hoistable) vs the *iterate* (per-call). From the `*_dist`
> rrules: FLmap gathers ALu+ALd (both fixed) → both hoist, FL never gathered.
> FRmap gathers ARd+ARu+FR → ARu/ARd fixed (hoist), FR=iterate (per-call). ACmap gathers
> AC+FL+FR → FL/FR fixed (hoist), AC=iterate (per-call). ACdmap gathers ACd+FL+FR →
> FL/FR fixed (hoist), ACd=iterate (per-call). Cmap gathers FL+FR (both fixed → hoist),
> C=iterate but replicated (no gather). This table is the contract for §2.2b's signatures.

---

## 4. The sliced-map rrules (mechanical derivation from the `*_dist` rrules)

For each map, the `*_slice2d_sliced` rrule is obtained from the `*_slice2d_dist` rrule by a
**mechanical surgery** (no new math), preserving every numerical step:

1. **Delete the forward input-gathers of the FIXED operands** — they become function
   args (`ALu_row`, `ALd_col`, etc.).
2. **Keep the forward iterate gather** if any (FR/AC/ACd col/row gather of the iterate),
   inside the sliced forward (per-call, like today).
3. **Keep the local chain forward** (`_*_slice2d_forward_sliced`) and the **type-A output
   reduce-scatter** unchanged.
4. In the pullback: **keep** the type-A output-adjoint allgather of `dresult`, the
   per-l-chunk `chain_backward`/hand-adjoint walk (with the `_SLICE2D` chains for FR/ACd),
   the iterate gather's adjoint (the iterate's reduce-scatter, e.g. dFR_blk for rightenv),
   and the M `allreduce_p2p!`.
5. **Delete the type-B reduce-scatters of the FIXED operands** (`dALu_blk =
   reduce_scatter(dALu_row)` etc.). **Return the slice gradient `dALu_row`/`dALd_col`
   directly.** The matching reduce-scatter now lives in `slice2d_gather_row/col`'s rrule
   and runs once at the hoist boundary.

The result is provably the same gradient (linearity, §5.2). Each sliced rrule keeps the
do-cast scaffolding shape (but mixed precision on the slice2d env path is deferred — §11,
so initially `inner_etype=nothing`; the scaffold is retained for M5).

**Equivalent fallback (if the surgery proves fiddly): keep the `*_dist` rrule, hoist by
capture.** A `*_slice2d_sliced` could instead accept the slices, and its rrule could still
run the type-B reduce-scatters but the env caches the gathered slices across calls and
discards the redundant block grads — *but that re-introduces a per-call reduce-scatter*,
defeating half the hoist. **Rejected** in favour of the clean surgery (§4.1-5), which
moves the type-B comm to once-per-solve. Recorded so the reviewer can weigh it.

---

## 5. AD correctness

### 5.1 The composition

`leftenv_slice2d` is plain Julia that calls `slice2d_gather_{row,col}` (once) and
`simple_eig` (which calls `FLmap_slice2d_sliced` `power_iter[_ad]·Nj` times, and
`slice2d_dot`/`slice2d_norm`). Zygote differentiates this composition (the status-quo
env-AD path via `pullback(vumps_itr)`); each building block contributes its hand rrule.
**No Zygote inside any map rrule** (constraint preserved): the rrules are hand-written;
Zygote only sees the env-level glue, exactly as it sees the serial env glue today.

### 5.2 Why hoisting gives the identical gradient (the linearity proof)

Let `g = slice2d_gather_row` (a linear map blk ↦ row-slice) with adjoint `gᵀ =
_slice2d_row_reduce_scatter_last`. Un-hoisted (`FLmap_slice2d_dist`), one power step is
`y = m(g(ALu_blk), …)` where `m` is the sliced compute; its pullback yields
`dALu_blk = gᵀ(∂m/∂ALu_row · dy)`. Over `K` power steps with the **same** `ALu_blk`,
the un-hoisted total is `Σ_k gᵀ(s_k)` where `s_k = ∂m_k/∂ALu_row · dy_k` is the per-step
slice cotangent. Hoisted: `g(ALu_blk)` is computed once and reused; Zygote accumulates
`s = Σ_k s_k` (a value used K times accumulates K cotangents), then runs `gᵀ` **once**:
`dALu_blk = gᵀ(Σ_k s_k)`. Since `gᵀ` is **linear**, `gᵀ(Σ_k s_k) = Σ_k gᵀ(s_k)`. ∎
Identical to the un-hoisted path. The same argument covers `slice2d_gather_col` and the
Cmap take-my-block adjoint (also linear). This is the formal version of research #5's
"sum-then-scatter ≡ scatter-then-sum."

### 5.3 slice2d_dot / slice2d_norm rrules

`slice2d_dot(x,y,grid) = MPI.Allreduce(dot(x_blk,y_blk), +, grid.comm)` and
`slice2d_norm(x,grid) = sqrt(MPI.Allreduce(sum(abs2,x_blk), +, grid.comm))` are
differentiated in the normalization/eigenvalue of `simple_eig`. The scalar allreduce-sum
adjoint is **identity on the local contribution** (∂(Σ_r s_r)/∂s_myrank = 1; other ranks'
locals don't depend on my block), and `ds`/`dn` are the **replicated** allreduce-output
cotangents (same on every rank → no extra comm). **The CORRECT formulas** (R1-B1/B2 — the
earlier draft's were wrong for complex tensors; pinned to ChainRules `dot`/`_norm2_back`):
```
rrule(slice2d_dot)(ds)  = (NoTangent, y_blk·conj(ds), x_blk·ds, NoTangent)            # consumed-as-scalar: NO allreduce
rrule(slice2d_norm)(dn) = (NoTangent, x_blk·(real(allreduce(dn,grid.comm))/n), NoTangent)  # denominator: allreduce dn
```
The **global** `n` and the **`real(·)`** projection are both load-bearing: a naive local
`norm(x_blk)` divides by the per-rank norm (wrong), and dropping `real(·)` leaks the
imaginary part into the block gradient when the upstream cotangent is complex.

**Batch-A finding (refines R1-B2) — `slice2d_norm` must allreduce its cotangent; `slice2d_dot`
must not.** This is a role asymmetry the original design + R1 both missed (caught only by the
distributed Gate 3b, which the per-rrule Gate 3a cannot see — Gate 3a tests the formula in
isolation). `slice2d_norm` is the **normalization denominator**: `simple_eig` does
`v /= slice2d_norm(v)`, broadcasting the replicated scalar `n` into a **per-rank division**
`v_blk/n`, so each rank's backward yields only its **piece** `dn_r = ∂loss/∂(use on rank r)`;
the true scalar cotangent is `c̄ = Σ_r dn_r = allreduce(dn)`, and `dx_blk = real(c̄)·x_blk/n`.
Omitting the allreduce drops the cross-rank term (measured ≈ 1–10 % gradient error at
χ=12 D=2; FP-exact once added). `slice2d_dot` is the **inner-product / eigenvalue** hook,
**consumed as a scalar** (λ stored, used once) — its `ds` is the genuine replicated scalar
cotangent, so allreducing it would over-count by P. Each hook has exactly one role in
`simple_eig`, so the asymmetry is well-defined (not fragile). The allreduce in
`slice2d_norm`'s backward is collective on `grid.comm` and rank-uniform. **General rule for
M5+:** an allreduce-produced replicated scalar that is *broadcast into a per-rank elementwise
op* needs its cotangent allreduced; one *consumed as a scalar* does not (the dual of the
gather's reduce-scatter-vs-take-my-block context dependence).

**Third required hook — `orth_for_ad` (R1-B3).** `simple_eig` ends with `v1 =
orth_for_ad(v1)` (misc.jl:74), whose rrule (rules.jl:99) is `_dv - dot(v,_dv)·v` with a
**LOCAL `dot`** — wrong on blocks (per-rank partial projection). Add a grid-aware
`orth_for_ad_slice2d(v, grid)` whose rrule is `_dv - slice2d_dot(v,_dv,grid)·v`, and give
`simple_eig` an `orth_fn` kwarg (default `orth_for_ad`) threaded by the distributed-iterate
envs (leftenv/rightenv/ACenv). **Cenv is exempt** (replicated C → local dot == global).

All three rrules are scalar/projection, pure-comm, no χ²D⁴ tape → safe. **Validation
(corrected, R1-m4):** a wrong *adjoint* is invisible to the forward eigenpair gate (Gate 1)
— it is caught ONLY by **Gate 3 gradient parity + a finite-difference micro-test with
COMPLEX inputs and a complex upstream cotangent** (real inputs hide the conj/real bugs).
**Precondition:** the slice2d env requires `ifsimple_eig=true` (the eigsolve/GMRES branch is
an implicit linear-solve adjoint that bypasses these rrules — out of scope).

> **Open question for review:** does the production energy gradient actually backprop
> through `simple_eig`'s normalization (vs. treating the converged eigenvector as a
> fixed point with an implicit-function adjoint)? `leading_boundary` unrolls
> `vumps_itr` (finite `power_iter_ad`), so YES the normalization is on the AD path and
> these rrules are required. Confirm there is no separate implicit-diff path that would
> make them dead code.

---

## 6. Persistent comm buffers, double buffering, zero-alloc steady state

The gate is "zero-allocation steady-state iteration." Interpreted operationally as:
**after warm-up, per-power-iteration device allocation does not grow and is bounded by
the recyclable chain transients** (literal-zero of H/T/G is deferred — research #11).
Three levers, in priority order:

1. **Gather hoisting (the dominant win, §2).** The χ²D²/N gathered slices (the large
   allocations) move out of the power loop — allocated once per env solve. This alone
   removes `power_iter·Nj` large allocations per solve.

2. **Persistent comm buffers for the slice2d collectives.** Today every slice2d collective
   `similar`-allocates recv/send buffers per call (research #9). Add a **shaped buffer
   pool** mirroring `_ensure_buf!` (grow-only, eltype-matched) but returning correctly-
   shaped views — the slice2d helpers use multi-dim `similar(acc)` and non-contiguous leg
   views, so a flat `_comm_sendbuf` is not a drop-in; use flat-buffer + `reshape`, keyed
   per (role, shape). Wire it into `_slice2d_row_shift` (the ring recv) and the four
   gather/RS helpers behind a `grid`-scoped pool. **Tag-collision guard (research #12):**
   the slice2d collectives use a single fixed tag per type; double-buffering or
   overlapping two in-flight collectives of the same class would collide. M4 keeps the
   iterates **non-overlapping** (one collective completes via Waitall before the next) —
   so the single tag stays safe; if a future overlap is added, parameterize the tag
   (the `_slice2d_row_shift` `tag` kwarg precedent; 720 band is free).

3. **Double-buffered iterate.** `simple_eig` does `v = f(v); v /= norm`. The default
   allocates a fresh `v` each step. Provide a buffer-reuse variant of the power loop
   (ping-pong two iterate buffers via the `update!`/in-place pattern, `environment.jl:209`)
   so the iterate is not reallocated. This is a `simple_eig` extension (a `recycle=true`
   path or a slice2d-specific `_power_iter_segment` that writes into a held buffer) — it
   must not perturb the FP trajectory (same arithmetic, just reused storage).

**Honest scope:** literal zero-alloc additionally needs `tensorcontract!`-into-buffer
fold variants for the chain links 3..N (only `chain_link1_add!` exists today, research
#11). That is an engine-level sub-task; M4 targets **"no allocation growth at steady
state"** measured by a per-iteration `CUDA.@allocated` (GPU) / `@allocated` (CPU) delta
that is flat after warm-up, and reports the residual transient (the eagerly-freed,
pool-recycled H/T/G). The into-buffer fold variants are an M4.5 stretch flagged here.

---

## 7. NCCL gating (χ·D/√P)

The slice2d collectives already have an NCCL fast path (`TENET_USE_NCCL=1`, gated
`_use_nccl() && CuArray && _equal_blocks`). M4 does **not** add an NCCL path; it adds
**per-call size gating** so NCCL fires only where it wins (research #13): the benefit is
set by per-rank message size ≈ χ²D²/√P, which **shrinks** under strong scaling. Implement
a predicate `_slice2d_nccl_worth(χ, D, P)` (threshold calibrated from Part 12: crossover
χ512@16GPU → χ1024@64GPU, i.e. roughly χ·D/√P above a constant) consulted at the slice2d
collective call sites in addition to the existing hard `_equal_blocks` precondition.
Default conservative (off below the crossover); env-overridable. This keeps the hand ring
for the strong-scaling tail and small cells (where Part 12 measured up to 5.6× slowdown
under blanket NCCL). The gate is a perf knob — **parity is unaffected** (NCCL and MPI
paths are bit-equivalent up to reduction order).

---

## 8. Grid threading + block representation

- **Pass the `Slice2DGrid` explicitly** to each env function (research flag: avoid the
  module-global `_slice2d_grid_cache` re-derivation; the cache stays as the builder).
- **Block-distributed StructArrays need no type change** (research #10): each `.data`
  entry is a block (smaller dense array); `.pattern` unchanged. The env functions
  allocate output env blocks via `randSA`/Zygote.Buffer of block shapes (already
  supported). `C` stays replicated (full χ×χ) in `Cenv_slice2d`.
- **χ recovery** is via `MPI.Allreduce(size(blk,1), +, col_comm)` (the established
  idiom), since a block only knows its local extent.
- **Square-grid `@assert N1==N2`** at each env entry (inherited from FR/AC/ACd maps).
- **Init scatter is M5** — for M4 tests, build full tensors with a fixed seed and
  `slice2d_scatter` to blocks (the existing harness pattern), exactly as the map tests do.

---

## 9. Parity gates + tests (the M4 acceptance criteria)

New file `test/test_slice2d_m4.jl` + launcher `test/run_test_slice2d_m4.jl` (4-rank,
`mpiexec -n 4`, mirroring `run_test_slice2d_m3.jl`). Grids: 2×2 (square; FR/AC/ACd skip
rectangular as M3 v2). χ ∈ {16, 18} (18 = uneven blocks), D ∈ {3}, plus Db≠Dc for
ACdmap-bearing paths. Tolerances: forward/eigenpair `rtol 1e-10` (CPU; GPU 1e-8),
gradient `rtol 1e-10` (CPU; GPU 1e-8). Reference = the **serial env function** on full
tensors (fixed seed → reproducible on every rank).

**Gate 1 — env-level eigenpair parity (the headline gate).** For each of the four env
functions: build full ALu/ALd/M/FL (fixed seed), run serial `leftenv` → `(λ_ref,
FL_ref)`; scatter to blocks, run `leftenv_slice2d` → `(λ_c, FL_c_blk)`; `slice2d_gather`
→ full; assert `λ_c ≈ λ_ref` (rtol 1e-8) and `FL_full ≈ FL_ref` up to global phase
(rtol 1e-8) — the existing phase-fix idiom (`test_slice2d.jl:268-275`). This is the M4
deliverable's correctness proof.

**Gate 2 — hoisting equivalence (catches the untested hoisting bug, research #8).**
Iterate `K` (≥3) power steps with the **hoisted** sliced map (fixed gathered slices) and
assert **bitwise/1e-12** agreement with `K` iterations of the per-call `*_slice2d_dist`
map (re-gathering each step). A stale-slice or double-buffer-aliasing bug fails here and
nowhere else.

**Gate 3 — gradient parity (readies M5).** Weighted-sum loss through `leftenv_slice2d`
vs serial `leftenv`; compare block cotangents (block-of) at rtol 1e-10; M cotangents
full. Plus a **finite-difference micro-test** of `slice2d_dot`/`slice2d_norm` rrules.
Plus the Cmap **replicated-output** gradient test (dC full, dFL/dFR block-of) — the
take-my-block-vs-reduce-scatter trap (§2.4).

**Gate 4 — zero-alloc steady state.** After warm-up, assert per-power-iteration
allocation delta is flat (no growth) and report the bounded transient (§6).

**Gate 5 — GPU validation (Sofia, deferred to a benchmark batch).** 4×4 (16-GPU) env
parity at χ=256 D=8, and the **gather-amortization hypothesis (research #13/#6.7):** a
leftenv power-iteration timing showing the hoisted forward beats the per-call `*_dist`
forward by ≈ the Part 10 penalty (2.7–5.5× at scale). This converts the un-amortized
penalty measurement into the M4 payoff — the one empirically-unproven claim. Recorded as
a new benchmark Part (13).

CI note: like the existing slice2d tests, M4 tests are launcher-driven (out-of-band of
`runtests.jl`). Decide separately whether to add an mpiexec-spawning include to CI.

---

## 10. Implementation task list (batched, each parity-gated, double-reviewed)

Order chosen so each batch is independently gateable and the lowest-risk map (FLmap, both
boundary operands fixed and hoistable, hand-kernel adjoints already proven) lands first.

- **Batch A — primitives + FLmap (the template).**
  1. `slice2d_gather_row`/`slice2d_gather_col` + rrules (reduce-scatter adjoints).
  2. `slice2d_dot`/`slice2d_norm` rrules.
  3. `FLmap_slice2d_sliced` + rrule (surgery from `FLmap_slice2d_dist` rrule, §4).
  4. `leftenv_slice2d` (General core, hoist ALu/ALd, slice2d hooks, no obs/mixed-prec).
  5. Tests: Gate 1+2+3 for leftenv; Gate 4 alloc probe. **Gate before Batch B.**
- **Batch B — rightenv (iterate-gather case).** `FRmap_slice2d_sliced` (hoist ARu/ARd,
  per-call FR gather; `FRMAP_LEG5_SLICE2D_CHAIN` backward) + `rightenv_slice2d` + tests.
- **Batch C — ACenv.** `ACmap_slice2d_sliced` (hoist FL/FR, per-call AC gather;
  `ACMAP_LEG5_CHAIN`) + `ACenv_slice2d` + tests.
- **Batch D — Cenv (replicated-output).** `slice2d_gather_full` (take-my-block adjoint) +
  `Cmap_slice2d_sliced` + `Cenv_slice2d` (plain dot/norm) + the replicated-output gradient
  test (§9 Gate 3). **Highest-risk adjoint — extra review.**
- **Batch E — persistent buffers + double-buffered iterate** (§6), behind a flag;
  re-run Gates 1-4; alloc-flatness gate.
- **Batch F — NCCL size gating** (§7) + Sofia GPU validation (Gate 5, benchmark Part 13).
  Includes the gather-amortization hypothesis test.

Each batch: per-map parity (1e-12 fwd / 1e-10 grad CPU) + double review (spec + quality,
**opus reviewers — `claude-fable-5` not accessible in this env**). Old serial path
untouched throughout.

---

## 11. Out of scope (→ M5/M6)

- `vumps_step`/`vumps_itr`/`leading_boundary` slice2d assembly + the **QR gather seam**
  (`ACCtoALAR` is full-χ per-cell QR; AC must be gathered between `ACenv_slice2d` and the
  QR, C is already replicated). This is a genuine new sub-design (distributed-or-gathered
  QR) and belongs to M5 ("fixed-point trajectory match vs serial").
- `init_VUMPSRuntime` block scatter (replace the `ifparallel` bcast with `slice2d_scatter`).
- Mixed precision (`inner_etype`/`whole_vumps_etype`) on the slice2d env path — the
  do-cast scaffold is retained in the sliced rrules but not exercised in M4.
- `ifobs` observation env (`leftenv(…; ifobs=true)`), C4v/Plaquette modes.
- Rectangular grids (N1≠N2) — M3 v2 block transpose.
- Literal zero-alloc of chain intermediates (into-buffer fold variants, M4.5).
- iPEPS ∂E/∂A end-to-end (M6).

---

## 12. Risks

1. **Hoisting AD equivalence** (§5.2) is the central correctness claim — Gate 2+3 are
   designed to catch any deviation; the linearity proof is the safety net.
2. **Cmap replicated-output adjoint** (§2.4) — take-my-block vs reduce-scatter; isolated
   by a dedicated wrapper + dedicated test (Batch D extra review).
3. **`slice2d_dot`/`slice2d_norm` rrule on the AD path** — confirm the normalization is
   actually differentiated (§5.3 open question) before relying on it; FD micro-test.
4. **Iterate-gather classification** (§3 design note) — FRmap/ACmap/ACdmap gather the
   iterate per-call; mis-hoisting the iterate (treating it as fixed) would be a
   correctness bug. The §3 table is the contract; Batch A/B verify it against the rrules.
5. **Zero-alloc gate is operational, not literal** (§6) — honest about the chain-
   intermediate residual; the dominant gathers ARE eliminated.
6. **`_ring_l_chunks` 2³¹ floor** must survive any sliced-map restructuring (research #13).
7. **Tag collisions** if double-buffering overlaps same-class collectives (§6.2) — M4
   keeps collectives non-overlapping; flagged for any future overlap.
8. **Gather-amortization is an unproven hypothesis** (research #13) — Gate 5 validates it
   on GPU; if it fails to amortize, the forward-perf rationale weakens (correctness
   unaffected).
