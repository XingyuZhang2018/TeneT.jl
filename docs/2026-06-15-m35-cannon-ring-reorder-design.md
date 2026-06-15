# M3.5: make FRmap / ACdmap cannon RING-class by reordering the local chain

Status: **IMPLEMENTED + VALIDATED** (commits 03df56b reorder, 334575d 2³¹ floor;
benchmarks Part 11). 4-rank CPU parity green (multi-chunk accumulate + off-diag
trap + Db≠Dc), opus review CLEAN, GPU parity `F✓ B✓` at 4×4/8×8, FR/ACd dropped
from 3–6× FLmap to ≈FLmap. Author response to the question "why aren't
FRmap/ACdmap the same architecture as FLmap/ACmap — can't we just reorder the
contraction?". **Answer: yes.** The gather-class 2-level i/d chunk
of `FRmap_cannon_dist`/`ACdmap_cannon_dist` is **not fundamental** — it is an
artifact of reusing the M2 serial chain order. Reordering the *local* chain so
the cross-axis *contracted* leg dies at link 1 (exactly what makes `ACmap`
ring-class) collapses the full-i×full-d intermediate plane back to χ²D⁴/P and
makes both maps single-l-chunk ring-class — **with identical communication**.

Predecessor: `docs/2026-06-13-acmap-cannon-dataflow-design.md` (the gather-class
design; correct given the chain order it fixed, but it never explored
reordering). Evidence the gather-class is slower: `benchmarks/Sofia_VUB_H200.md`
Part 11 (jobs 1287202/1287203) — at 4 GPU FRmap/ACdmap run **2–3.5×** FLmap
while ACmap runs **1.0–1.2×**. This doc's reorder is predicted to close that gap;
Part 11 is the "before" baseline.

---

## 1. The root cause is the CHAIN ORDER, not the block convention

All four maps share the uniform Cannon block convention (first χ leg → N1/r1,
last χ leg → N2/r2). Under it, ACmap and ACdmap have the SAME cross-axis legs
(one contracted, one output) — the difference is purely which operand carries
which, and therefore which contraction order kills the cross-axis contracted leg
early.

Per-leg census (χ-legs only; D-legs/p omitted):

```
ACmap   result[i,j,k,l] = AC[a,b,c,d] FR[d,g,h,l] M1 M2 FL[a,e,f,i]
  a co-distributed N1 (contracted) | d cross-axis (contracted) | i cross-axis (output) | l aligned (output)

ACdmap  result[a,b,c,d] = ACd[i,j,k,l] FR[d,g,h,l] M1 M2 FL[a,e,f,i]
  i cross-axis (contracted) | l aligned (contracted) | a aligned (output) | d cross-axis (output)

FLmap   result[d,g,h,l] = FL[a,e,f,i] ALd[i,j,k,l] M1 M2 ALu[a,b,c,d]    (fully on-axis — no cross-axis leg)
FRmap   result[a,e,f,i] = ARd[i,j,k,l] FR[d,g,h,l] M1 M2 ARu[a,b,c,d]
  d cross-axis (contracted) | l aligned (contracted) | a aligned (output) | i cross-axis (output)
```

**The ring-class condition.** A map's local intermediates stay bounded
(χ²D⁴/P, no full-χ plane) **iff** the cross-axis *contracted* leg is killed at
the first link AND the cross-axis *output* leg is born at the last link (so the
two never coexist as live intermediate legs). The cross-axis output leg is then
handled exactly like ACmap's: gather it to full on its input operand, produce it
full in the (bounded χ²D²/N) output buffer, and reduce-scatter it to its owning
axis.

`ACmap` satisfies this **with the M2 order** (`AC,FR,…,FL`): `AC·FR` kills the
cross-axis contracted `d` at link 1; the cross-axis output `i` is born from the
last operand `FL`. That is the entire reason ACmap was already ring-class while
ACdmap/FRmap were not — the M2 order, reused verbatim, happens to satisfy the
condition for ACmap but not for its twins.

`ACdmap`/`FRmap` **violate it with the M2 order**:

```
ACdmap M2 order (ACd,FR,M1,M2,FL): ACd·FR contracts l (aligned) FIRST
  → the cross-axis contracted i (from ACd, full after gather) and the cross-axis
    output d (from FR, full after gather) BOTH survive into I1=(i,k,d,h,j,g) …
  → full-i × full-d plane  → 2-level i/d chunk (the gather-class cost)

FRmap M2 order (ARd,FR,M1,M2,ARu): ARd·FR contracts l (aligned) FIRST
  → cross-axis output i (from ARd) and cross-axis contracted d (from FR) both survive
  → full-i × full-d plane  → 2-level chunk
```

## 2. The reorder

Permute the LOCAL chain so the cross-axis contracted leg is the link-1
contraction; keep the cross-axis output leg's operand last. (Tensor contraction
is order-independent in value — any order yields the same result.)

```
ACdmap → (FL, ACd, M1, M2, FR),  out (a,b,c,d)     # FL·ACd kills cross-axis contracted i at link 1
FRmap  → (FR, ARu, M1, M2, ARd), out (a,e,f,i)     # FR·ARu kills cross-axis contracted d at link 1
```

These become new **distributed-only** chain consts (the single-GPU M2 chains and
their proven layout pins are untouched):

```julia
const ACDMAP_LEG5_CANNON_CHAIN = tensor_chain(
    ((:a,:e,:f,:i), (:i,:j,:k,:l), (:e,:j,:g,:b,:p), (:f,:k,:h,:c,:p), (:d,:g,:h,:l)), (:a,:b,:c,:d))
const FRMAP_LEG5_CANNON_CHAIN  = tensor_chain(
    ((:d,:g,:h,:l), (:a,:b,:c,:d), (:e,:j,:g,:b,:p), (:f,:k,:h,:c,:p), (:i,:j,:k,:l)), (:a,:e,:f,:i))
```

(Plus their `conj_variant(…,?)` single-M twins, slot-index TBD by the
operand carrying M2 in the new order — see §9.)

## 3. Per-link memory proof (ACdmap reorder; FRmap is the i↔d mirror)

On rank `(r1,r2)`, square grid `N1=N2=N`, `p_rs = split_ranges(χ,N)`. The
gathers (§4) deliver `FL_g[a∈p_rs[r1+1], e,f, i∈1:χ]` (full i),
`ACd_g[i∈1:χ, j,k, l∈p_rs[r2+1]]` (full i), `FR_g[d∈1:χ, g,h, l∈p_rs[r2+1]]`
(full d). Walk the reordered chain:

| link | contract | intermediate | χ-legs | size |
|------|----------|--------------|--------|------|
| FL_g·ACd_g | `i` (FULL, local Σ) | I1=(a,e,f,j,k,l) | a∈p_rs[r1+1], l∈p_rs[r2+1] | **χ²D⁴/P** |
| I1·M1 | e,j | I2=(a,f,k,l,g,b,p) | a-block, l-block | χ²D⁴·d_phys/P |
| I2·M2 | f,k | I3=(a,l,g,b,h,c) | a-block, l-block | χ²D⁴/P |
| I3·FR_g | g,h,l | out=(a,b,c,d) | a-block, **d FULL** | χ²D²/N (output buffer) |

(Index tuples above are label **SETS**; the engine's `tensor_pinned_inters`
derives an equivalent layout order, e.g. pinned I1=(a,f,k,l,e,j) — same set, same
size, both χ-legs local. The bound depends only on the set, not the pin order.)

**No intermediate carries a full-χ leg.** `i` is full only inside the *input*
slices `FL_g`/`ACd_g` (χ²D²/N each) and is summed away at link 1; `d` is full
only inside `FR_g` (χ²D²/N) and the *output buffer* (χ²D²/N, a-block × d-full).
The carried intermediates I1/I2/I3 are χ²D⁴/P — **identical to FLmap's H/T/G**.

**Chunk axis = the aligned contracted leg `l`** (single axis, like FLmap/ACmap;
any local-block χ-leg present in I1/I2/I3 would work — `a` too — but `l` is the
natural choice). `l` lives on the local r2-block; slice it into `forloop_iter`
sub-chunks, run the chain per sub-chunk, **accumulate** the partial (`Σ_l` over
the local block — `l` is CONTRACTED, so each sub-chunk adds into the same
output buffer), and let F5's row reduction finish `Σ_l` over r2. Per-chunk peak =
χ²D⁴/(P·forloop_iter) — **the FLmap/ACmap bound, single-axis**. The output `d`
(full, χ²D²/N) need not be chunked (already bounded), but MAY be co-chunked if the
χ²D²/N buffer is large.

**Forward chunk semantics (do NOT copy ACmap's loop).** ACmap's single-l loop is
ASSIGN (its `l` is an OUTPUT leg, sliced on the output buffer). Here `l` is a
CONTRACTED leg absent from `out=(a,b,c,d)`: the loop ACCUMULATES `Σ_l`, slicing
the FORWARD operands `ACd_g`/`FR_g` on `l` and adding each sub-chunk's result into
the one full-d output buffer (FRmap: slice `FR_g`/`ARd_g`... — i.e. the two
operands that carry `l`).

Contrast the gather-class M2 order: its I1=(i,k,d,h,j,g) is full-i × full-d =
χ²D⁴ (NOT /P), forcing the 2-level i/d chunk to N²·⌈√floop⌉² ≈ P·floop pieces.

## 4. Communication is UNCHANGED (the surgical claim)

The reorder needs the **same gathered tensors** the current gather-class forward
already builds, and the **same output collective**. Compare current ACdmap F1–F5
(`acmap-cannon-dataflow-design.md` §5.1) to the reorder:

| step | current (gather-class) | reorder (ring-class) |
|------|------------------------|----------------------|
| χ recovery (F0) | `MPI.Allreduce` size over col_comm | **same** |
| gather i on ACd | `_cannon_col_allgather` (730) | **same** |
| gather i on FL | `_cannon_row_allgather` (750) | **same** |
| gather d on FR | `_cannon_col_allgather` (730) | **same** |
| local einsum | 2-level i/d chunk, `ACDMAP_LEG5_CHAIN` | **single-l chunk, `ACDMAP_LEG5_CANNON_CHAIN`** ← only this changes |
| output | `_cannon_row_reduce_scatter_last` (760): Σ_l over row + scatter d→r2 | **same** |

So the gathers (3), the reduce-scatter (1), all tags (730/750/760), and the
per-step comm adjoints are **identical**. **No new primitive; the change is
local-compute-only.** Same for FRmap (gathers: d on FR (730) + d on ARu (750) +
i on ARd (730); output row_reduce_scatter_last on `i`).

## 5. Backward

`chain_backward(ACDMAP_LEG5_CANNON_CHAIN, (FL_g, ACd_g, M1, M2, FR_g), dpartial)`
— recompute-style, eager `_free!`, **no Zygote** (unchanged discipline). The
**comm adjoints** (B5 allgather of dresult, the gathers' reduce-scatter adjoints,
the output reduce-scatter's allgather adjoint, the M allreduces) are
**byte-for-byte the current rrule** — same gathered-grad shapes (`dFL_g`/`dACd_g`
full-i, `dFR_g` full-d, `dpartial` full-d), since the comm is unchanged (§4). The
**final return** is map-arg order `(NoTangent, dACd_blk, dFL_blk, dFR_blk, dM,
NoTangent)`, as today.

**Two non-comm edits the implementer MUST make** (the local-compute change is not
"identical"):

1. **B4 unpack order follows the NEW ops order.** `chain_backward` returns grads
   in the reordered chain's ops order, so for ACdmap:
   `(dFL_c, dACd_c, dM1_c, dM2_c, dFR_c) = chain_backward(ACDMAP_LEG5_CANNON_CHAIN,
   (FL_g, ACd_g[l-ch], M1, M2, FR_g[l-ch]), dpartial)` — **not** the current
   `(dACd, dFR, dM1, dM2, dFL)` order. (FRmap: `(dFR_c, dARu_c, dM1_c, dM2_c,
   dARd_c)`.) Re-wire each to its grad buffer, then permute to the map-arg return
   above. Assert the unpack order against the chain's `ops` to prevent a silent
   mis-wire.
2. **B4 l-loop ACCUMULATES (mirror of the forward, not ACmap's assign).** `l` is
   contracted and absent from `dpartial`, so feed the **full `dpartial`** to every
   l-sub-chunk; slice the l-carrying operands (`ACd_g`/`FR_g`; FRmap `ARd_g`/`FR_g`)
   on `l`; the l-carrying grads (`dACd_g`/`dFR_g`) assign-disjoint into their
   l-slice while the non-l grads (`dFL_g`, `dM1`, `dM2`) accumulate over l-chunks.
   Copy-pasting ACmap's assign-style loop (which slices `dpartial` on `l`) would be
   wrong — `dpartial` has no `l` axis — and would silently drop all but one chunk
   (predecessor §9 risk-1).

Backward peak: (2+2d)·χ²D⁴/(P·forloop_iter) + the three gathered-slice grad
buffers — **the FLmap/ACmap backward bound** (was the gather-class 2-level
bound).

## 6. Completeness (diagonal trap still defeated)

The cross-axis **output** leg (`d` for ACdmap, `i` for FRmap) is made **full**
in the output buffer before the reduce-scatter (FR_g/ARd_g carry it full → the
last contraction produces it full). So every output block `(a∈p_rs[A],
d∈p_rs[B])` is materialized in the full-d buffer on rank `(A,·)` and the
row_reduce_scatter delivers block `B` to r2=B — off-diagonal included. This is
the ACmap §2.5 completeness argument verbatim (cross-axis output leg full before
the scatter). The cross-axis **contracted** leg is gathered full on both its
operands and summed locally, so its contraction is complete. ∎

## 7. Expected performance

Two independent wins over the gather-class order:

1. **Per-rank FLOPs ÷N.** The gather-class link-1 `ACd·FR` produces the full-i×
   full-d plane: FLOPs/rank ≈ χ³D⁴/N (full i, full d, local-l). The reorder's
   link-1 `FL·ACd` produces a-block × l-block: FLOPs/rank ≈ χ³D⁴/N² = χ³D⁴/P —
   **N× fewer** (2× at 4 GPU, 4× at 16 GPU).
2. **GEMM granularity.** Single-l chunk = `forloop_iter` larger GEMMs vs the
   gather-class ≈P·forloop_iter smaller GEMMs.

Predicted result: **FRmap/ACdmap drop from Part 11's 2–3.5× to ≈ACmap's
1.0–1.2× of FLmap**, and the N× FLOP factor predicts the gap *widens* with grid
size in the gather-class — i.e. the reorder should help 16 GPU even more than
4 GPU. Acceptance test = re-run the Part 11 driver and confirm parity with FLmap.

## 8. All four maps become one architecture

With the reorder, FLmap / FRmap / ACmap / ACdmap are all:
**3 gathers + single-l-chunk local chain (χ²D⁴/(P·forloop_iter)) + 1
reduce-scatter**, no full-χ plane, no new primitive. This is exactly the uniform
architecture the question anticipated; the M2-order reuse was the only thing
breaking the symmetry.

## 9. Implementation task list

1. **Chains** (`chain_maps.jl`): add `ACDMAP_LEG5_CANNON_CHAIN`,
   `FRMAP_LEG5_CANNON_CHAIN` (+ single-M `conj_variant` twins). **Guard against a
   transcription typo**: assert each new chain's per-operand leg tuples equal the
   existing chain's (only the operand ORDER permuted) — a changed leg tuple would
   silently change a gather requirement (§4 rests on the leg labels being
   identical). Verify the engine's `tensor_pinned_inters` auto-pins the new orders
   sanely; if a non-pinned layout costs the Part-7 5–9%, add a probe-verified pin
   (as `FLMAP_LEG5_CHAIN` has). Correctness is independent of pinning.
2. **Forward** (`cannon_2d.jl` `_acdmap_cannon_forward_sliced`,
   `_frmap_cannon_forward_sliced`): keep F1–F3 gathers + F5 reduce-scatter
   verbatim; replace the F4 2-level i/d loop with a single-l-chunk loop over the
   `_CANNON_CHAIN`. Drop the `n_i,n_d`/`n_chunk = N·⌈√floop⌉` machinery; chunk
   `l` like `_cannon_forward_sliced` (FLmap) does.
3. **Backward** (`rules.jl` FRmap/ACdmap rrules): swap `chain_backward` to the
   `_CANNON_CHAIN`; chunk loop 2-level→single-l; comm adjoints unchanged.
4. **forloop_iter semantics**: now identical to FLmap (= #l-chunks). Update the
   docstring/`@assert`; the `forloop_iter`→chunk-count gap (`N·⌈√·⌉`) is gone.
5. **Validate**: `test/test_cannon_m3.jl` 4-rank parity (1e-12/1e-10) must stay
   green; Sofia GPU parity at χ=256 D=8; re-run the Part 11 timing → expect
   FR/ACd ≈ FLmap. Record as Part 12 (and note Part 11 as the gather-class
   baseline).

## 10. Risks

- **Layout pinning** for the new orders may need a probe (perf only, not
  correctness) — risk 1, mitigated by step 1's probe check.
- **Single-M conj slot**: the M2 operand sits at a different chain position in
  the reordered order; the `conj_variant(…, slot)` index and the boundary
  `dM = dM1 + conj(dM2)` composition must target the right slot (pin to the
  `FLmap_cannon_dist` precedent). Covered by the single-M parity test.
- **FR's two d-gathers vs ACd's one i-gather**: FRmap gathers the cross-axis
  *contracted* `d` on BOTH FR and ARu (col+row allgather) and the cross-axis
  *output* `i` on ARd (col allgather) — confirm this matches the current FRmap
  forward's gather set (it does: FRmap is the i↔d twin of ACdmap). No change.
- **No regression for ACmap/FLmap**: untouched (already ring-class). Only
  FRmap/ACdmap code paths change.
