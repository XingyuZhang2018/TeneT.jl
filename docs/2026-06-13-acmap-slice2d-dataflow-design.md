# M3: ACmap / ACdmap Slice2D cross-axis dataflow (square grid)

Status: design, gating the M3 plan.
Branch: `claude/ecstatic-golick-e3f6f0` (M1+M2 chain engine landed, `CHAIN_ENGINE` default ON).
Template: `FLmap_slice2d_dist` rrule (`src/autodiff/rules.jl:589-698`), forward
`_slice2d_forward_sliced` (`src/contraction/slice2d.jl:417-464`).
Local einsum: chain engine `ACMAP_LEG5_CHAIN` / `ACDMAP_LEG5_CHAIN`
(`src/contraction/chain_maps.jl:106,139`), `chain_apply` / `chain_backward`
(`src/contraction/chain_engine.jl:121,231`).
Supersedes the hand-wavy `AllGather→einsum→AllReduce→AllToAll` sketch in
`docs/2026-05-11-2d-distributed-vumps-runtime-design.md` §2.3 and resolves the
open ACmap candidate flagged in `docs/2026-06-12-…` L3 (user chose square-grid
block transpose). M3 milestone row = line 124 of that doc.

**Status after review (2026-06-13).** Core dataflow PASSED adversarial review:
the diagonal-block trap is correctly defeated (gather the cross-axis FREE leg to
full before the local chain, then fuse the output redistribution into an existing
reduce-scatter), completeness/adjoints/MPI-safety/no-new-primitive all CONFIRMED.
Two issues were fixed in this revision and are reflected below:

- **ACdmap memory (was a sign-off BLOCKER).** The first draft chunked the
  contracted `l` for ACdmap, but `l` is contracted at link 2 (FR) and is **absent
  from every internal intermediate** `I1=(i,k,d,h,j,g)`, `I2=(i,d,e,b,k,h,p)`,
  `I3=(d,b,c,i,e,f)` — all of which carry **full `i`** (cross-chain contracted,
  gathered full as `ACd_g`/`FL_g`) **and full `d`** (output, computed full before
  the row scatter). Chunking `l` cannot bound them (χ²D⁴ ≈ 25 GB/intermediate at
  D=10 χ=400; I2 is χ²D⁵). The fix chunks the legs that actually appear in
  I1/I2/I3: a **2-level loop over `i` (contracted → accumulate) and `d` (output →
  assign)**, reaching the ACmap-class bound `χ²D⁴/(P·forloop_iter)` (§5.1, §6.2,
  §9 risk 3). ACmap was already correct and is **untouched** — its intermediates
  `I1=(a,c,h,l,b,g)`, `I2=(a,l,e,j,c,h,p)`, `I3=(l,j,k,a,e,f)` carry only the
  **local** `a`-block plus the **chunked** `l`; the output `i`/`d` never appear as
  intermediate legs because FL is the chain's LAST operand.

- **Single-M `dM` composition (was a MAJOR self-contradiction).** Pinned to the
  `FLmap_slice2d_dist` precedent (`rules.jl:592,572,690`): the single-M entry
  routes through the **2M tuple chain internally** (`M1,M2 = (M, conj(M))`) and
  composes `dM = dM1 + conj(dM2)` at the **rrule boundary** on exit. §§4.1/4/6.1
  now state this one convention (§9 risk 7).

---

## 1. Scope and the square-grid assumption

`ACmap_slice2d_dist` / `ACdmap_slice2d_dist` are the cross-axis Slice2D maps. The
serial kernels (verified `src/contraction/basic.jl:287,374`) are leg5 (tuple-M):

```
ACmap:  result[i,j,k,l] := AC[a,b,c,d] FR[d,g,h,l] M1[e,j,g,b,p] M2[f,k,h,c,p] FL[a,e,f,i]
ACdmap: result[a,b,c,d] := ACd[i,j,k,l] FR[d,g,h,l] M1[e,j,g,b,p] M2[f,k,h,c,p] FL[a,e,f,i]
```

All χ-legged operands follow the Slice2D block convention (`slice2d_scatter`,
`slice2d.jl:164`): **first χ leg split N1-ways by r1, last χ leg split N2-ways
by r2.** Rank `(r1,r2)`, `rank = r1*N2 + r2`. `row_comm`: fixed r1, internal
rank == r2, size N2. `col_comm`: fixed r2, internal rank == r1, size N1.

**M3 v1 assumes a SQUARE grid: `N1 = N2 = N`** (user decision). Assert at entry:

```julia
@assert grid.N1 == grid.N2 "ACmap_slice2d_dist: M3 v1 requires a square grid (N1==N2)"
```

Consequence (load-bearing). `split_ranges(χ, N1) == split_ranges(χ, N2)` because
`N1 == N2`: **the r1-partition and the r2-partition of any χ leg are literally
the same partition.** Call it `p_rs := split_ranges(χ, N)`, a length-N vector of
contiguous 1-based `UnitRange`s. So the block held by `(r1,r2)` of a χ×…×χ tensor
is `[p_rs[r1+1], …, p_rs[r2+1]]`, and "the t-th block of the first leg" and "the
t-th block of the last leg" use the *same* range `p_rs[t+1]`. This is what makes
a square block transpose (rank swap on the grid) move block `p_rs[A]×p_rs[B]` to
the rank that owns `p_rs[B]×p_rs[A]` with matching shapes. **Rectangular grids
(`N1≠N2`) are deferred to M3 v2** — there `p_rs` splits differ, the transpose is
no longer a rank swap, and an explicit re-block (alltoallv with per-(src,dst)
counts) is required.

Per-leg distribution census (the source of the cross-axis problem):

| index | operands | distribution | axis kind |
|-------|----------|--------------|-----------|
| `a`   | AC (1st, r1), FL (1st, r1)       | both N1/r1 | **co-distributed N1** (contract over col_comm) |
| `d`   | AC (last, r2), FR (1st, r1)      | AC on r2, FR on r1 | **cross-axis** (contracted) |
| `i`   | FL (last, r2), result (1st, r1)  | FL on r2, result on r1 | **cross-axis** (output) |
| `l`   | FR (last, r2), result (last, r2) | both N2/r2 | **aligned** |

Contrast with FLmap, which is fully on-axis: there `a` (AC↔FL analog) is
co-distributed N1, the contracted χ leg `i` is aligned on N2, and the output
χ legs land on the same axes as the inputs, so the map iterates with no
redistribution. ACmap's `d` (contracted) and `i` (output) are both cross-axis —
that is the entire content of this document.

---

## 2. The diagonal-block trap, and the correct forward scheme

### 2.1 Why the naive 4-step is INCOMPLETE (must read before anything else)

The orchestrator's hand-derivation flags a trap that any "transpose at the end"
scheme falls into. Consider the naive candidate:

> (1) col_allgather AC over its first leg `a` to recover full-`a` AC slice;
> (2) col_allgather FR over its first leg `d`; (3) single local `chain_apply`;
> (4) col-reduce/allreduce + transpose `i`.

Track the **output leg ranges** this produces on rank `(r1,r2)`. The two output
χ legs are `i` (from FL's last leg) and `l` (from FR's last leg). Under the
convention FL is `[a, e, f, i]` with `i` on the **last** leg → r2 → range
`p_rs[r2+1]`. FR is `[d,g,h,l]` with `l` on the **last** leg → r2 → range
`p_rs[r2+1]`. The local einsum can only produce output indices that the local
input blocks span, so the locally-computed partial is

```
partial[ i ∈ p_rs[r2+1], j, k, l ∈ p_rs[r2+1] ]
```

**The i-block and the l-block are the SAME range `p_rs[r2+1]`** (this is the
`p_rs == shared-partition` consequence biting). So across the whole grid only the
**DIAGONAL** result blocks `(i∈p_rs[A], l∈p_rs[A])` are ever computed. The
off-diagonal blocks `result[i∈p_rs[A], l∈p_rs[B]]`, `A≠B`, are **never produced
by any rank**. A post-hoc transpose of `i` permutes ranks but cannot synthesize a
block that no rank computed: transposing a strictly-diagonal set of blocks yields
a strictly-diagonal set. **The naive 4-step is incomplete — it silently computes
only the χ/N diagonal stripe of the χ×χ (i,l) output.** A "simple rank swap"
transpose is therefore wrong; the fix must make one of the cross-axis legs
*full* before the local einsum so off-diagonal pairs are actually contracted.

Root cause restated: `i` and `l` are **both** on r2. The local working set sees
only the r2-th block of each, so it can only fill the (r2,r2) tile of the (i,l)
plane.

### 2.2 Candidate evaluation

We must make the local einsum produce, for the fixed local `l`-block, the **full
`i` column** — every `i ∈ 1:χ` paired with the local `l`-block — and *then*
redistribute `i` to its owning r1 by a **scatter of a free output leg** (not a
reduce-scatter; nothing is summed over the scatter). Three ways to get there:

- **(i) Gather one free input leg to full before the einsum.** FL carries `i`
  on its last leg, distributed over r2 (range `p_rs[r2+1]`). The N row peers
  `(r1, 0..N-1)` collectively hold all χ values of `i` for the local `a`-block.
  `_slice2d_row_allgather(FL_blk, grid, p_rs)` (existing, tag 750, concatenates the
  LAST leg) assembles `FL[a∈p_rs[r1+1], e, f, i∈1:χ]` — local `a`-block, **full
  `i`**. The local einsum then produces full `i`, local `l`-block. A final
  **free-leg scatter** of `i` keeps block `p_rs[r1+1]` on rank r1. This is the
  chosen scheme — it reuses an existing primitive for the gather and needs one
  small new primitive (a free-leg scatter) whose adjoint is a free-leg
  allgather. PROVED complete in §2.4.

- **(ii) Re-block FL so `i` lands on N1.** Would let the einsum yield
  `i`-block r1 directly with no output transpose. But re-blocking FL means
  moving its last leg from r2-distribution to r1-distribution — that is itself a
  block transpose (the same cross-axis move), just relocated to the input. It
  buys nothing over (i) and still needs a transpose primitive. The `d`
  contraction is unaffected either way (see §2.3). Rejected: no simpler than (i),
  and it perturbs the input convention so the map no longer iterates cleanly.

- **(iii) Genuine all-to-all materializing full `i` or full `l`.** Equivalent in
  data volume to (i)'s row_allgather but with bespoke per-(src,dst) counts and a
  new tag; (i) already materializes full `i` via the proven concatenating
  allgather. Rejected: strictly more code for the same peak memory.

**Chosen: candidate (i).** Gather `i` to full on FL (row_allgather, free last
leg), run one local chain over the full-`i` FL, then free-leg-scatter the output
`i` to N1. The contracted cross-axis leg `d` is handled by a separate gather, §2.3.

### 2.3 Handling the contracted cross-axis leg `d`

`d` is contracted between AC (`d` on its last leg, r2, range `p_rs[r2+1]`) and FR
(`d` on its first leg, r1, range `p_rs[r1+1]`). For the contraction `Σ_d AC[…d]
FR[d…]` to be local, both operands must expose the **same full `d` axis** to the
local einsum.

- AC's `d` is on r2 → the N row peers `(r1, 0..N-1)` hold all χ values of `d`
  for the local `a`-block. `_slice2d_row_allgather(AC_blk, grid, p_rs)` gives
  `AC[a∈p_rs[r1+1], b, c, d∈1:χ]` — local `a`-block, **full `d`**.
- FR's `d` is on r1 → the N col peers `(0..N-1, r2)` hold all χ values of `d`
  for the local `l`-block. `_slice2d_col_allgather(FR_blk, grid, p_rs)` gives
  `FR[d∈1:χ, g, h, l∈p_rs[r2+1]]` — **full `d`**, local `l`-block.

After these two gathers, AC has full `d` and FR has full `d`, so `Σ_d` is local.
No new primitive: both are existing collectives (row_allgather tag 750,
col_allgather tag 730). Note AC and FL share the `a` axis (co-distributed N1) and
both end up `a`-blocked on r1 after their respective row_allgathers — `a` stays
local to the chain over the col_comm reduction (next).

### 2.4 The co-distributed leg `a` and the column reduction

`a` is contracted between AC and FL and is co-distributed on N1 (r1). After the
gathers of §2.2–2.3, on rank `(r1,r2)`:

```
AC_g  = AC[a∈p_rs[r1+1], b, c, d∈1:χ]      (row_allgather AC, full d)
FL_g  = FL[a∈p_rs[r1+1], e, f, i∈1:χ]      (row_allgather FL, full i)
FR_g  = FR[d∈1:χ, g, h, l∈p_rs[r2+1]]      (col_allgather FR, full d)
M1, M2 replicated (tiny)
```

Every operand exposes its local `a`-block `p_rs[r1+1]`. The chain
`ACMAP_LEG5_CHAIN` over `(AC_g, FR_g, M1, M2, FL_g)` contracts `a` between AC and
FL **inside the chain**, summing only over the **local** `a`-block. So the local
result is a **partial sum over `a ∈ p_rs[r1+1]`**:

```
partial_{r1}[ i∈1:χ, j, k, l∈p_rs[r2+1] ]
  = Σ_{a∈p_rs[r1+1]} Σ_{b,c,d,e,f,g,h,p}
      AC_g[a,b,c,d] FR_g[d,g,h,l] M1[e,j,g,b,p] M2[f,k,h,c,p] FL_g[a,e,f,i]
```

The full result needs `Σ_{a∈1:χ} = Σ_{r1=0}^{N-1} Σ_{a∈p_rs[r1+1]}`, i.e. a
**sum over the col_comm** (fixed r2, varying r1):

```
y[i∈1:χ, j, k, l∈p_rs[r2+1]] = Σ_{r1} partial_{r1}[i, j, k, l]   (allreduce over col_comm)
```

This `y` is **full `i`**, **full `j,k`**, local `l`-block — identical on every
col rank (allreduce, not reduce-scatter, because `i` is full and we still must
scatter it afterward; reduce-scattering `i` over col_comm would mix the
co-`a`-sum with the `i`-scatter incorrectly — see §2.6). Now scatter the free
output leg `i` to its owning r1:

```
result_blk[i∈p_rs[r1+1], j, k, l∈p_rs[r2+1]] = y[p_rs[r1+1], :, :, :]   (free-leg scatter, §3)
```

### 2.5 Completeness proof (all (i,l) block pairs are computed)

Fix any output block `(i∈p_rs[A], l∈p_rs[B])`, `A,B ∈ 0..N-1`. The rank that
must own it is `(r1=A, r2=B)`. On that rank:

1. FL row_allgather gives **full `i`** (`i∈1:χ ⊇ p_rs[A+1]`), local `a`-block
   `p_rs[A+1]`. ✓ — `i∈p_rs[A]` is present.
2. FR col_allgather gives **full `d`**, local `l`-block `p_rs[B+1]`. ✓ —
   `l∈p_rs[B]` is present.
3. AC row_allgather gives full `d`, local `a`-block `p_rs[A+1]`. The chain
   contracts `Σ_{a∈p_rs[A+1]}` locally and `Σ_d` (full) locally, producing
   `partial_A[i∈1:χ, j,k, l∈p_rs[B+1]]`.
4. col_comm allreduce sums `partial_{r1}` over all r1, completing
   `Σ_{a∈1:χ}` → `y[i∈1:χ, j,k, l∈p_rs[B+1]]`. The block `i∈p_rs[A]` is a slice
   of full `y`. ✓
5. Free-leg scatter keeps `i∈p_rs[A+1]` on r1=A. ✓

So `result[i∈p_rs[A], l∈p_rs[B]]` is computed correctly for **every** `(A,B)`,
including all off-diagonal `A≠B`. The diagonal trap is defeated because step 1
makes `i` **full** before the einsum, so the off-diagonal `(A,B≠A)` tile is
materialized in `y` on rank `(A,B)` and merely sliced out — never relying on a
transpose to invent it. ∎

### 2.6 Why allreduce-then-scatter, not reduce-scatter

Both `a` (the col_comm sum) and `i` (the scatter target) are on the col axis / r1.
But they are **different legs**: `a` is contracted (summed), `i` is a free output
leg (sliced). A reduce-scatter over col_comm conflates them — it would deliver to
rank r1 the sum-over-r1 of *its own i-block contribution only*, i.e.
`Σ_{r1'} partial_{r1'}[i∈p_rs[r1+1]]`, which is exactly the correct answer for
the value but is computed by routing the wrong source blocks: reduce-scatter
extracts chunk `p_rs[dest+1]` of leg 1 from each rank and sums into `dest`. Since
every `partial_{r1}` already holds **full `i`** (not an `i`-blocked partial), the
reduce-scatter's per-destination chunking on leg `i` is in fact valid and
identical to allreduce+slice. So either works numerically; we choose **explicit
allreduce + local slice** for clarity (the scatter is a pure local
`getindex`, no comm) — but note `_slice2d_col_reduce_scatter(partial, grid, p_rs)`
(existing, tag 710) computes `y[p_rs[r1+1]]` in a single fused collective and is
the **production choice** because it halves the i-leg bandwidth (each rank
receives only its `i`-block, not full `i`). Its adjoint is `col_allgather`
(tag 730), already wired. We adopt the fused form:

```
result_blk = _slice2d_col_reduce_scatter(partial_{r1}, grid, p_rs)   # tag 710
```

where `partial_{r1}` is `[i∈1:χ, j, k, l∈p_rs[r2+1]]` (full first leg `i`, local
last leg `l`) — exactly the shape `_slice2d_col_reduce_scatter` expects (it sums
the full first leg over col_comm and keeps the r1-block). **This makes the new
free-leg scatter primitive unnecessary in the forward** — the col_reduce_scatter
already delivers `i∈p_rs[r1+1]`. The only genuinely new operation needed is on
the **input gather of `i`** (FL row_allgather, existing) — so M3 ACmap forward
needs **NO new primitive at all**. (The free-leg scatter/allgather pair in §3 is
documented for completeness and for the ACdmap variant analysis, but is not on
the ACmap forward path.)

### 2.7 ACmap forward — final step list

On rank `(r1,r2)`, inputs block-stored (`AC_blk, FL_blk, FR_blk` per convention),
`M1,M2` replicated, `χ = N·blocklen` recovered by the usual col allreduce of
`size(AC_blk,1)`:

```
F0. χ = MPI.Allreduce(size(AC_blk, 1), +, col_comm);  p_rs = split_ranges(χ, N)
F1. AC_g = _slice2d_row_allgather(AC_blk, grid, p_rs)   # full d   (tag 750)  [a-block, full d]
F2. FL_g = _slice2d_row_allgather(FL_blk, grid, p_rs)   # full i   (tag 750)  [a-block, full i]
F3. FR_g = _slice2d_col_allgather(FR_blk, grid, p_rs)   # full d   (tag 730)  [full d, l-block]
F4. for each l-chunk ch (forloop_iter, §6):
        partial[:, :, :, ch] = chain_apply(ACMAP_LEG5_CHAIN,
            (AC_g, FR_g[:,:,:,ch], M1, M2, FL_g))      # local; full i, local l-block
F5. result_blk = _slice2d_col_reduce_scatter(partial, grid, p_rs)  # sum a over col, keep i-block r1 (tag 710)
return result_blk   # [i∈p_rs[r1+1], j, k, l∈p_rs[r2+1]]  — same convention as input
```

Output distribution == input distribution (first χ leg `i` on r1, last χ leg `l`
on r2), so the map **iterates** with no further redistribution — same property
FLmap_slice2d has.

Local chain (`ACMAP_LEG5_CHAIN`, `chain_maps.jl:106`): ops order
`((AC),(FR),(M1),(M2),(FL))` = `((:a,:b,:c,:d),(:d,:g,:h,:l),(:e,:j,:g,:b,:p),
(:f,:k,:h,:c,:p),(:a,:e,:f,:i))`, out `(:i,:j,:k,:l)`. The carried operand is AC;
`a` is contracted at the LAST link (FL), so `a` survives as an open intermediate
label through links 2–4 and is summed only when FL is contracted — and since
`AC_g`/`FL_g` carry only the local `a`-block, that final contraction is the local
`a`-partial. Exactly what §2.4 requires. (The slice2d rrule always runs this **2M**
`ACMAP_LEG5_CHAIN`; the single-M entry materializes `M2 = conj(M)` at the boundary
and composes `dM = dM1 + conj(dM2)` on exit, pinned to the `FLmap_slice2d_dist`
precedent `rules.jl:592,690` — see §4.1/§6.1. The serial `ACMAP_LEG5_CHAIN_1M =
conj_variant(…,4)` slot-sum path `chain_maps.jl:108,118-121` gives the same value
but is not the rrule path.)

**Peak memory.** Bounded. The gathers produce slices of size χ²D²/N (AC_g, FL_g:
local a-block × full χ; FR_g: full χ × local l-block) — the irreducible per-rank
working set, same order as FLmap's gathered AL slices. `partial` is
`χ·D²·(local l) = χ²D²/N` with **full `i`** but **local `l`** — bounded by
χ²D²/N, never χ². The chain's internal intermediates (analog of H,T,G) are
chunked by `forloop_iter` (§6). **No χ×χ (i,l) plane is ever materialized**: full
`i` coexists only with the *local* `l`-block. ∎

---

## 3. New primitive (only if ACdmap or a future variant needs it): free-leg scatter / allgather

The ACmap forward (§2.6) needs **no new primitive**. The ACdmap analysis (§5)
shows it too reduces to existing primitives. We nonetheless specify the free-leg
**scatter** and its adjoint here because (a) it is the conceptually-correct
operation behind "scatter a free output leg to N1" and (b) it is the fallback if
a future rectangular variant cannot fuse into col_reduce_scatter.

`_slice2d_col_scatter_first(full, grid, d_rs)`: input `full[a∈1:χ, mid…]`
**replicated across col_comm** (identical on all r1 for fixed r2); output the
r1-block `full[d_rs[r1+1], mid…]`. **Pure local `getindex` — NO communication**
(every col rank already holds identical `full`). Its **adjoint** is
`_slice2d_col_allgather(dblk, grid, d_rs)` (existing, tag 730): the cotangent of a
replicated→block slice is gather-the-blocks (each rank contributed a disjoint
slice of the replicated input; the upstream replicated cotangent is the
concatenation — identical to the `slice2d_gather` rrule's "downstream of gather is
replicated → adjoint is take-my-block", run in reverse). Because the forward is
communication-free, **no tag, no synchronize, no N==1 path, no
chi-divisibility logic** are needed for the scatter itself; all of that lives in
the adjoint's `_slice2d_col_allgather`, which already handles them
(`slice2d.jl:275`, N1==1 fast path at line 281, uneven χ via `d_rs` lengths).

The row-axis mirror (`_slice2d_row_scatter_last` / adjoint
`_slice2d_row_allgather`, tag 750) is defined symmetrically if a free **last** leg
must be scattered over r2; same comm-free-forward / allgather-adjoint structure.

**If a genuine communicating transpose is ever required** (rectangular M3 v2), it
would be `_slice2d_block_transpose` (NEW tag 770): on a square grid, rank
`(r1,r2)` `Sendrecv`s its block to/from rank `(r2,r1)` over `grid.comm` (a single
paired exchange, self-send elided when `r1==r2`), recv buffer shaped
`(len p_rs[r1+1], mid…, len p_rs[r2+1])` swapped to `(len p_rs[r2+1], mid…,
len p_rs[r1+1])`; `synchronize(blk)` before the `Isend`; **N==1 fast path**
returns `blk`; **χ-not-divisible** handled by `p_rs` range lengths (recv size =
sender's block size, computable from `p_rs` on both ends); **adjoint is the same
transpose** (it is an involution / orthogonal permutation: `(r2,r1)→(r1,r2)`).
**Not needed for M3 v1** — recorded so the tag is reserved and the square-grid
involution property is on record.

---

## 4. `ACmap_slice2d_dist` BACKWARD

Adjoint of every forward step in reverse. **No Zygote** (tapes OOMed job
1274256). Local adjoint via `chain_backward(ACMAP_LEG5_CHAIN, …)` (recompute
style, eager `_free!`, grads in ops order). Eager `_free!` discipline mirrors the
FLmap rrule exactly.

### 4.1 Adjoint of each forward step (reverse order)

Forward F1–F5; reverse is B5→B1.

**B5 — adjoint of `_slice2d_col_reduce_scatter` (F5).** Reduce-scatter's adjoint is
**allgather** (`slice2d.jl:275`, the documented pair). Given `dresult_blk`
(`[i∈p_rs[r1+1], j,k, l∈p_rs[r2+1]]`), produce `dpartial`:

```
dpartial = _slice2d_col_allgather(dresult_blk, grid, p_rs)   # [i∈1:χ, j,k, l∈p_rs[r2+1]]  (tag 730)
```

Derivation of the col_comm reduction adjoint from first principles. Forward is
`y[i] = Σ_{r1} partial_{r1}[i]` (full `i`), then `result_blk[i∈p_rs[r1+1]] =
y[p_rs[r1+1]]`. The fused col_reduce_scatter does both: rank r1 receives
`Σ_{r1'} partial_{r1'}[p_rs[r1+1]]`. For the pullback, the loss depends on
`result_blk` on each rank; `result_blk` on rank r1 is `Σ_{r1'}
partial_{r1'}[p_rs[r1+1]]`. So `∂loss/∂(partial_{r1'}[i])` is nonzero only for
`i∈p_rs[r1'... ]`? No — every source rank `r1'` contributed its full-`i`
`partial_{r1'}` to **every** destination's block (destination r1 took
`partial_{r1'}[p_rs[r1+1]]`). Therefore the cotangent that flows back to a given
source rank `r1'` for index `i∈p_rs[m+1]` equals `dresult` of the rank that
*owned* block `m`, i.e. rank `m`'s `dresult_blk`. Gathering all destinations'
`dresult_blk` over col_comm and concatenating on `i` reconstructs full
`dpartial[i∈1:χ]`, identical on every source r1' — which is exactly
`_slice2d_col_allgather`. ∎ (Equivalently: reduce-scatter = scatter∘broadcast-sum;
its transpose = sum-broadcast∘gather = allgather, since the per-rank partials all
fed the same sum.)

**B4 — adjoint of the local chain (F4).** For each l-chunk `ch`, run
`chain_backward` recompute-style over the **gathered slices**:

```
dAC_g_c, dFR_g_c, dM1_c, dM2_c, dFL_g_c =
    chain_backward(ACMAP_LEG5_CHAIN, (AC_g, FR_g[:,:,:,ch], M1, M2, FL_g),
                   view(dpartial, :, :, :, ch))
```

Grads come back in **ops order** `(dAC, dFR, dM1, dM2, dFL)` (`chain_engine.jl:231`
returns `Tuple(grads)` in `ch.ops` order). Accumulate into per-rank gradient
buffers (`dAC_g`, `dFR_g`, `dFL_g` over the gathered slices; `dM1`, `dM2`
replicated), with the same eager-free pattern as the FLmap rrule:

```
dAC_g .+= dAC_g_c
view(dFR_g, :, :, :, ch) .+= dFR_g_c
dFL_g .+= dFL_g_c
dM1 .+= dM1_c;  dM2 .+= dM2_c
_free!(dAC_g_c); _free!(dFR_g_c); _free!(dFL_g_c); _free!(dM1_c); _free!(dM2_c)
```

`chain_backward` itself frees its recomputed intermediates (the H/T/G analogs)
internally after their last consumer — no Zygote, no tape. This is the M2 payoff:
**the local einsum AND its adjoint are the chain engine; no new hand kernels.**

(Single-M leg5 entry — pin to the FLmap precedent, `rules.jl:592,690`. **Do NOT
route the rrule through the `_1M` chain.** Materialize `M1, M2 = (M, conj(M))` at
entry, run the **2M tuple chain** `ACMAP_LEG5_CHAIN` throughout F4/B4, and compose
`dM = dM1 + conj(dM2)` at the **rrule boundary** on exit (§4.1 "M gradients",
§6.1). Equivalence note: the chain-internal slot-sum `dM = g[3] + g[4]`
(`engine_backward(::typeof(ACmap),…)`, `chain_maps.jl:118-121`, run over
`ACMAP_LEG5_CHAIN_1M` with slot 4 conj-flagged) yields the **same value** —
because slot 4's conj flag makes `g[4]` already the `conj(dM2)`-equivalent, so
`g[3]+g[4]` ≡ `dM1+conj(dM2)`. The slice2d rrule nonetheless follows the FLmap
boundary convention `dM1 + conj(dM2)` over the 2M chain, for one consistent code
path with the tuple-M case.)

**B3 — adjoint of `_slice2d_col_allgather` on FR (F3).** Allgather's adjoint is
**reduce-scatter** (`slice2d.jl:242`, the documented pair). `dFR_g` is full `d`,
local `l`-block; reduce-scatter over col_comm on the first leg `d` returns the
r1-block:

```
dFR_blk = _slice2d_col_reduce_scatter(dFR_g, grid, p_rs)    # [d∈p_rs[r1+1], g,h, l∈p_rs[r2+1]]  (tag 710)
```

**B2 — adjoint of `_slice2d_row_allgather` on FL (F2).** Row_allgather (last leg
`i`) adjoint is `_slice2d_row_reduce_scatter_last` (`slice2d.jl:375`, tag 760,
the documented pair). `dFL_g` is local `a`-block, full `i`; reduce-scatter the
LAST leg `i` over row_comm returns the r2-block:

```
dFL_blk = _slice2d_row_reduce_scatter_last(dFL_g, grid, p_rs)  # [a∈p_rs[r1+1], e,f, i∈p_rs[r2+1]]  (tag 760)
```

**B1 — adjoint of `_slice2d_row_allgather` on AC (F1).** Same primitive: `dAC_g`
is local `a`-block, full `d`; reduce-scatter the LAST leg `d` over row_comm:

```
dAC_blk = _slice2d_row_reduce_scatter_last(dAC_g, grid, p_rs)  # [a∈p_rs[r1+1], b,c, d∈p_rs[r2+1]]  (tag 760)
```

**M gradients (replicated).** Both the tuple-M and the single-M entries run the
**2M chain** `ACMAP_LEG5_CHAIN` internally (single-M materializes `M1, M2 = (M,
conj(M))` at entry, exactly as `FLmap_slice2d_dist` does at `rules.jl:592`), so B4
always produces a `(dM1, dM2)` pair. `dM1`, `dM2` accumulated over chunks (and
over the local a-block via the chain); stitch the per-rank slices with a single
allreduce each (picks up the NCCL fast path), then compose at the boundary
following the FLmap precedent (`rules.jl:572,690`):

```
allreduce_p2p!(dM1, +, grid.comm);  allreduce_p2p!(dM2, +, grid.comm)
dM = is_tuple ? (dM1, dM2) : dM1 .+ conj(dM2)
```

### 4.2 Backward step list and return tuple

```
B0. densify dresult if !(dresult isa DenseArray) (FillArrays guard, like FLmap rrule);
    do_cast boundary cast.
B5. dpartial = _slice2d_col_allgather(dresult, grid, p_rs)                 # tag 730
B4. for each l-chunk ch:  (dAC_g_c,dFR_g_c,dM1_c,dM2_c,dFL_g_c) =
        chain_backward(ACMAP_LEG5_CHAIN, (AC_g, FR_g[:,:,:,ch], M1, M2, FL_g), dpartial[:,:,:,ch])
        accumulate into dAC_g, dFR_g(view ch), dFL_g, dM1, dM2; eager _free! each.
B3. dFR_blk = _slice2d_col_reduce_scatter(dFR_g, grid, p_rs)               # tag 710
B2. dFL_blk = _slice2d_row_reduce_scatter_last(dFL_g, grid, p_rs)          # tag 760
B1. dAC_blk = _slice2d_row_reduce_scatter_last(dAC_g, grid, p_rs)          # tag 760
BM. allreduce_p2p!(dM1,+,comm); allreduce_p2p!(dM2,+,comm)
    dM = is_tuple ? (dM1, dM2) : dM1 .+ conj(dM2)   # FLmap precedent (rules.jl:572,690)
    do_cast upcast of dAC_blk,dFL_blk,dFR_blk,dM at boundary exit.
return (NoTangent(), dAC_blk, dFL_blk, dFR_blk, dM, NoTangent())
```

Return tuple mirrors the **map arg order** `ACmap(AC, FL, FR, M)` →
`(NoTangent, dAC, dFL, dFR, dM, NoTangent)` — note the chain ops order is
`(AC,FR,M1,M2,FL)`, so the chain's `(dAC,dFR,dM1,dM2,dFL)` is permuted to map
order `(dAC, dFL, dFR, dM)` exactly as `engine_backward(::typeof(ACmap),…)` does
(`chain_maps.jl:116`). The slice2d rrule does this permutation when wiring B4's
accumulators to the returned blocks.

**Eager-free / live-set.** Same discipline as FLmap rrule: `dpartial` chunk
slices freed after the chain_backward consumes them; `dAC_g/dFR_g/dFL_g` (size
χ²D²/N) freed after their reduce-scatters; chain_backward's internal recomputed
intermediates freed inside the engine. Backward peak ≈ (2+2d)·χ²D⁴/(P·forloop_iter)
+ the three gathered-slice gradient buffers — the FLmap backward bound, plus the
extra cross-axis gather slices (AC_g, FR_g, FL_g and their grads coexist where
FLmap had only AL slices). Size `forloop_iter` by this bound.

---

## 5. `ACdmap_slice2d_dist` — derive forward + backward (SHOW THE WORK)

ACdmap kernel (`basic.jl:374`):

```
result[a,b,c,d] := ACd[i,j,k,l] FR[d,g,h,l] M1[e,j,g,b,p] M2[f,k,h,c,p] FL[a,e,f,i]
```

Same five operands and same M-contractions as ACmap; the difference is which
legs are **inputs** vs **outputs**. Here **`{i,l}` are inputs** (on ACd) and
**`{a,d}` are outputs** (the TOP row of the unit cell). Per-leg census under the
convention (ACd is `[i,j,k,l]`: first leg `i` r1, last leg `l` r2):

| index | operands | distribution | axis kind |
|-------|----------|--------------|-----------|
| `i`   | ACd (1st, r1), FL (last, r2) | ACd on r1, FL on r2 | **cross-axis** (contracted) |
| `l`   | ACd (last, r2), FR (last, r2) | both r2 | **aligned** (contracted) |
| `a`   | FL (1st, r1), result (1st, r1) | both r1 | **aligned** (output) |
| `d`   | FR (1st, r1), result (last, r2) | FR on r1, result on r2 | **cross-axis** (output) |

So ACdmap is cross-axis **the other way**: the cross-axis legs are `i`
(contracted: ACd.i on r1, FL.i on r2) and `d` (output: FR.d on r1, result.d on
r2). Work it leg by leg.

**Contracted `i` (cross-axis).** ACd.i on r1 (range `p_rs[r1+1]`), FL.i on r2
(range `p_rs[r2+1]`). To contract `Σ_i` locally both must expose full `i`:
- ACd.i is on r1 → col peers `(0..N-1, r2)` hold all `i` for the local `l`-block.
  `_slice2d_col_allgather(ACd_blk, grid, p_rs)` → `ACd[i∈1:χ, j, k, l∈p_rs[r2+1]]`.
- FL.i is on r2 → row peers `(r1, 0..N-1)` hold all `i` for the local `a`-block.
  `_slice2d_row_allgather(FL_blk, grid, p_rs)` → `FL[a∈p_rs[r1+1], e, f, i∈1:χ]`.
  (Same FL gather as ACmap F2.)

**Contracted `l` (aligned).** ACd.l and FR.l both on r2, same range `p_rs[r2+1]`.
After the ACd col_allgather, ACd has local `l`-block; FR keeps its block
`l∈p_rs[r2+1]`. So `Σ_l` is **already local** on the r2 block — but note `Σ_l`
runs only over the local l-block unless the output needs full l... it does not:
`l` is purely contracted and aligned, so summing the local l-block on each rank
and combining is NOT needed *if both operands carry the same l-block*, which they
do. ✓ No gather for `l`.

**Output `a` (aligned).** result.a on r1, FL.a on r1, same range. The local chain
produces `a∈p_rs[r1+1]` directly (FL carries `a`-block). ✓ No redistribution.

**Output `d` (cross-axis).** result.d on r2 (must be), FR.d on r1. The local
einsum produces `d` from FR's first leg. After we gather FR's `d`? — careful:
unlike ACmap we do **not** want full `d` here, because `d` is an **output**, not
contracted. We want the local chain to produce **full `d`** (so off-diagonal
`(a,d)` pairs exist — the SAME trap, transposed), then scatter/reduce-scatter `d`
to r2. So FR must expose **full `d`** to the einsum:
- FR.d on r1 → col peers `(0..N-1, r2)` hold all `d` for the local `l`-block.
  `_slice2d_col_allgather(FR_blk, grid, p_rs)` → `FR[d∈1:χ, g, h, l∈p_rs[r2+1]]`.
  (Same FR gather as ACmap F3.)

Now the **co-distribution / reduction axis**. The contracted `i` after gathers:
ACd carries full `i`, FL carries full `i`, but FL is `a`-blocked on r1 while ACd
is `l`-blocked on r2 — `i` is full on both, so `Σ_i` is fully local, **no col/row
sum needed for `i`**. What about the output `d` being full and needing scatter to
r2? The chain produces, on rank `(r1,r2)`:

```
partial[a∈p_rs[r1+1], b, c, d∈1:χ]   (full d, local a-block, local l-block summed in)
```

This is **full `d`**, local `a`-block. `d` must end up on r2 (range `p_rs[r2+1]`).
That is a scatter of a free **last** output leg over r2. But here is the key
difference from ACmap: in ACmap the scatter axis `i` **coincided** with the
col-sum axis `a`, letting us fuse into `col_reduce_scatter`. In ACdmap the
output cross-axis leg is `d` (must go to **r2**), and there is **no reduction**
to fuse with (the contractions `i`,`l` are both fully resolved locally — `i` by
the gathers, `l` by alignment; `a` is a free output, not summed). So ACdmap's
output redistribution is a **pure free-leg scatter of `d` to r2** with no reduce:

```
result_blk = _slice2d_row_scatter_last(partial, grid, p_rs)   # take d∈p_rs[r2+1]; comm-free local slice (§3)
```

**This is where ACdmap genuinely differs from ACmap.** ACmap fused output-scatter
into a col_reduce_scatter (because scatter-axis == sum-axis); ACdmap has no
matching reduction, so it uses the **pure free-leg row scatter** (§3) — a local
`getindex` keeping `d∈p_rs[r2+1]`, whose adjoint is `_slice2d_row_allgather`
(tag 750). Equivalently and preferably, **don't gather FR's `d` to full at all**;
instead recognize that since `partial` is full-`d` and identical-across-r2 only
if we summed nothing over r2 — and we did not — each r2 rank computes the **same**
full-`d` `partial` **only if** ACd and FR are replicated across r2, which they are
NOT (they are l-blocked). So `partial` on rank `(r1,r2)` is the contribution from
the **local l-block only**:

```
partial_{r2}[a∈p_rs[r1+1], b, c, d∈1:χ] = Σ_{l∈p_rs[r2+1]} Σ_{i,…} …
```

The full result needs `Σ_{l∈1:χ} = Σ_{r2} Σ_{l∈p_rs[r2+1]}` — a **sum over
row_comm** (fixed r1, varying r2). And the output `d` must land on r2. **Now the
scatter axis (`d`→r2) DOES coincide with the sum axis (row_comm/r2)** — so
ACdmap fuses exactly like ACmap, but on the **row** axis with the **last** leg:

```
result_blk = _slice2d_row_reduce_scatter_last(partial, grid, p_rs)   # sum l over row, keep d-block r2 (tag 760)
```

Wait — `_slice2d_row_reduce_scatter_last` reduces the LAST leg over row_comm. Here
the last leg of `partial` is `d` (full), and we want `Σ_{r2}` (sum over row) while
keeping `d∈p_rs[r2+1]`. That is precisely "reduce-scatter the last leg over
row_comm" = `_slice2d_row_reduce_scatter_last(partial, grid, p_rs)` (tag 760,
`slice2d.jl:375`). **So ACdmap also needs NO new primitive** — it is the
row-axis mirror of ACmap's col_reduce_scatter, and the existing
`_slice2d_row_reduce_scatter_last` is exactly the fused op. The "pure free-leg
scatter" detour above is subsumed; the correct operation is the fused
row_reduce_scatter_last. (The §3 free-leg scatter remains documented as the
non-fused fallback for rectangular v2.)

### 5.1 ACdmap forward step list

**Internal-intermediate memory (the BLOCKER, derived).** Run the engine's
`tensor_pinned_inters` on `ACDMAP_LEG5_CHAIN` (`chain_maps.jl:139`, ops
`(ACd,FR,M1,M2,FL)`, out `(a,b,c,d)`); it returns the pinned layouts

```
I1 = (i, k, d, h, j, g)        ~ χ² D⁴   (χ-legs i, d)
I2 = (i, d, e, b, k, h, p)     ~ χ² D⁵   (χ-legs i, d)
I3 = (d, b, c, i, e, f)        ~ χ² D⁴   (χ-legs d, i)
```

**Every internal intermediate carries FULL `i` AND FULL `d`.** `i` is the
cross-chain contracted leg (gathered full as `ACd_g`/`FL_g`); `d` is the output
leg (computed full before the row scatter). The earlier-draft "chunk the
contracted `l`" does **nothing** here: `l` is contracted at link 2 (FR) and is
**absent from I1/I2/I3** — chunking `l` shrinks nothing internal, leaving each
intermediate at the full `χ²D⁴` plane (`χ²D⁵` for I2) ≈ 25 GB at D=10 χ=400. This
is the genuine `χ×χ (i,d)` plane the ACmap derivation was careful never to
materialize. (Contrast ACmap: its intermediates `I1=(a,c,h,l,b,g)`,
`I2=(a,l,e,j,c,h,p)`, `I3=(l,j,k,a,e,f)` carry the **local** `a`-block and the
**chunked** `l`; the output `i`/`d` never appear because FL is the chain's LAST
operand — so ACmap's `l`-chunk bounds them. ACdmap's outputs `a`/`d` enter
**early** because FL is again last but now `a`/`d` are output legs that ride the
whole chain.)

**Fix — chunk `i` (contracted) and `d` (output), the legs that actually appear in
I1/I2/I3.** Both are full-`χ`; neither is `l`. Chunk each with its own loop:

- `i` is **contracted** (summed in the chain at the FL link). Slicing `i` slices
  `ACd_g` over its first leg and `FL_g` over its last leg; the per-`i`-chunk chain
  outputs **accumulate** into the same `partial` (`Σ_i`). Each chain run sees only
  `i ∈ i-chunk`, so I1/I2/I3 carry `(χ/n_i)` on `i`.
- `d` is the **output** (chain's last open leg, from FR's first leg). Slicing `d`
  slices `FR_g` over its first leg; the per-`d`-chunk chain output is the
  **disjoint** slice `partial[:,:,:,d-chunk]` — **assign**, not accumulate. Each
  chain run carries `(χ/n_d)` on `d` in I1/I2/I3.

```
F0. χ = MPI.Allreduce(size(ACd_blk, 1), +, col_comm);  p_rs = split_ranges(χ, N)
F1. ACd_g = _slice2d_col_allgather(ACd_blk, grid, p_rs)   # full i  (tag 730)  [full i, l-block]
F2. FL_g  = _slice2d_row_allgather(FL_blk,  grid, p_rs)   # full i  (tag 750)  [a-block, full i]
F3. FR_g  = _slice2d_col_allgather(FR_blk,  grid, p_rs)   # full d  (tag 730)  [full d, l-block]
F4. # 2-level memory chunk over the FULL legs i (contracted) and d (output);
    # the contracted local l-block is summed WHOLE inside each chain_apply.
    i_chunks = split_ranges(χ, n_i);  d_chunks = split_ranges(χ, n_d)  # §6.2 sizes n_i,n_d
    for dch in d_chunks                       # disjoint output slices -> ASSIGN
        acc = nothing
        for ich in i_chunks                   # summed input slices    -> ACCUMULATE (Σ_i)
            piece = chain_apply(ACDMAP_LEG5_CHAIN,
                (ACd_g[ich,:,:,:], FR_g[dch,:,:,:], M1, M2, FL_g[:,:,:,ich]))  # I1/I2/I3 ~ (χ/n_i)(χ/n_d)D⁴
            acc === nothing ? (acc = piece) : (acc .+= piece; _free!(piece))
        end
        partial[:, :, :, dch] = acc;  _free!(acc)         # disjoint d-slice assignment
    end
F5. result_blk = _slice2d_row_reduce_scatter_last(partial, grid, p_rs)  # sum l over row, keep d-block r2 (tag 760)
return result_blk   # [a∈p_rs[r1+1], b, c, d∈p_rs[r2+1]]  — convention-correct (a on r1, d on r2)
```

(`FR_g[dch,:,:,:]` slices `d`; `ACd_g[ich,:,:,:]`/`FL_g[:,:,:,ich]` slice `i`. The
local l-block is **not** chunked — it is contracted whole inside each
`chain_apply`, the same `Σ_{l∈p_rs[r2+1]}` as before; F5's `Σ_{r2}` completes
`Σ_l`. The `i`/`d` loops are a pure memory device and leave the value identical.)

Local chain `ACDMAP_LEG5_CHAIN` (`chain_maps.jl:139`): ops
`((:i,:j,:k,:l),(:d,:g,:h,:l),(:e,:j,:g,:b,:p),(:f,:k,:h,:c,:p),(:a,:e,:f,:i))`,
out `(:a,:b,:c,:d)` — carried operand is ACd; `i` is contracted at the LAST link
(FL), `l` is contracted at link 2 (FR). With ACd_g/FR_g full-`i`/full-`d` and FL_g
full-`i`, the chain sums `Σ_i` (over the local `i`-chunk, accumulated across
chunks) and `Σ_l` (over the local l-block, whole). The `Σ_{r2}` over the row
completes `Σ_l` in F5.

**Achieved internal-intermediate bound.** Each `chain_apply` run sees `i ∈ i-chunk`
(`χ/n_i`) and `d ∈ d-chunk` (`χ/n_d`), so

```
peak per intermediate ~ (χ/n_i)·(χ/n_d)·D⁴ = χ² D⁴ / (n_i · n_d)     (I2: ·D⁵)
```

To reach the **ACmap-class bound** `χ²D⁴/(P·forloop_iter)` (P = N1·N2 = N²; the
§6.2 ACmap figure), set `n_i · n_d = P · forloop_iter = N² · forloop_iter`. The
clean single-knob choice is `n_i = n_d = N·⌈√forloop_iter⌉` (so the product is
`N²·forloop_iter`); the even simpler `n_i = n_d = N·forloop_iter` over-chunks to
`χ²D⁴/(N²·forloop_iter²)`, strictly tighter than ACmap. Either way **no `χ×χ`
`(i,d)` plane is ever materialized** — the full `i` and full `d` coexist only as
disjoint chunk slices, never simultaneously at full extent. ∎

**Completeness (mirror of §2.5).** Fix output block `(a∈p_rs[A], d∈p_rs[B])`,
owner `(r1=A, r2=B)`. F2 gives FL full `i`, a-block `p_rs[A+1]` → `a∈p_rs[A]`
present. F1/F3 give ACd/FR full `i`/full `d`, l-block `p_rs[B+1]`. The chain on
`(A,B)` produces `partial_B[a∈p_rs[A+1], …, d∈1:χ]` (full `d` ⊇ `p_rs[B+1]`),
summing the local l-block. F5 sums over r2 (completing `Σ_l`) and keeps
`d∈p_rs[B+1]`. So `result[a∈p_rs[A], d∈p_rs[B]]` is computed for **every** `(A,B)`,
off-diagonal included. The trap (here on the `(a,d)` plane) is defeated because F2
makes `i`'s partner output `d` full before the row reduction. ∎

Note the **output buffer** `partial`'s shape is `[a∈p_rs[r1+1], b, c, d∈1:χ]` =
χ²D²/N (full d, local a-block) — itself bounded (no `χ²` plane: `a` is the local
block, only `d` is full, with `b,c` D-bonds). The danger was never `partial`; it
was the **internal chain intermediates** I1/I2/I3, which carry full `i` ×
full `d` and are bounded only by the `i`/`d` chunking above, not by `partial`'s
shape.

### 5.2 ACdmap backward

Reverse of F1–F5; same eager-free, `chain_backward(ACDMAP_LEG5_CHAIN,…)`,
no Zygote.

**B4 chunks the SAME `i`/`d` as F4.** `chain_backward` recompute-style rebuilds
I1/I2/I3 (`chain_engine.jl:231`), so the **identical** full-`i`×full-`d`
intermediates reappear in the adjoint pass; the bound is recovered by chunking
`i` and `d` exactly as in F4. The cotangent `dpartial` is full-`d`, local-a; its
`d`-slice `dpartial[:,:,:,dch]` is the cotangent for the `d`-chunk, while every
`i`-chunk shares it (`i` is contracted, so it has no slot in `dpartial`). Grads
accumulate over both loops: `dFR_g` is sliced by `d` (**assign-disjoint** per
`dch`, accumulate over `ich`); `dACd_g`/`dFL_g` are sliced by `i` (**accumulate**
over both loops). `dM1`/`dM2` accumulate over both loops.

```
B0. densify dresult; do_cast.
B5. dpartial = _slice2d_row_allgather(dresult, grid, p_rs)                 # adjoint of row_reduce_scatter_last; full d (tag 750)
B4. # SAME 2-level i/d chunk as F4 (chain_backward recomputes I1/I2/I3 → same full-i×full-d intermediates).
    i_chunks = split_ranges(χ, n_i);  d_chunks = split_ranges(χ, n_d)
    for dch in d_chunks
      for ich in i_chunks
        (dACd_g_c, dFR_g_c, dM1_c, dM2_c, dFL_g_c) =
            chain_backward(ACDMAP_LEG5_CHAIN,
                (ACd_g[ich,:,:,:], FR_g[dch,:,:,:], M1, M2, FL_g[:,:,:,ich]),
                dpartial[:,:,:,dch])
        view(dACd_g,ich,:,:,:) .+= dACd_g_c          # i-sliced (accumulate over both loops)
        view(dFR_g, dch,:,:,:) .+= dFR_g_c           # d-sliced disjoint per dch (accumulate over ich)
        view(dFL_g,:,:,:,ich)  .+= dFL_g_c           # i-sliced
        dM1 .+= dM1_c; dM2 .+= dM2_c
        _free!(dACd_g_c); _free!(dFR_g_c); _free!(dFL_g_c); _free!(dM1_c); _free!(dM2_c)
      end
    end
B3. dFR_blk  = _slice2d_col_reduce_scatter(dFR_g, grid, p_rs)              # adjoint of F3 col_allgather (tag 710)
B2. dFL_blk  = _slice2d_row_reduce_scatter_last(dFL_g, grid, p_rs)        # adjoint of F2 row_allgather (tag 760)
B1. dACd_blk = _slice2d_col_reduce_scatter(dACd_g, grid, p_rs)            # adjoint of F1 col_allgather (tag 710)
BM. allreduce_p2p!(dM1,+,comm); allreduce_p2p!(dM2,+,comm)
    dM = is_tuple ? (dM1, dM2) : dM1 .+ conj(dM2);  do_cast upcast.    # FLmap precedent (rules.jl:572,690)
return (NoTangent(), dACd_blk, dFL_blk, dFR_blk, dM, NoTangent())
```

**Backward intermediate bound** matches the forward: each `chain_backward` run
recomputes I1/I2/I3 over `i∈i-chunk`, `d∈d-chunk`, so its recomputed-plus-adjoint
working set is the backward multiple of `χ²D⁴/(n_i·n_d)` (≈ doubled for the
co-resident recomputed intermediate and its cotangent), i.e. the same
`χ²D⁴/(N²·forloop_iter)`-class figure as ACmap's backward when `n_i·n_d =
N²·forloop_iter`. `chain_backward` frees its own recomputed I1/I2/I3 internally.

Adjoint pairs used (all existing, all documented in `slice2d.jl`):
- F5 `row_reduce_scatter_last` (760) ↔ B5 `row_allgather` (750).
- F1/F3 `col_allgather` (730) ↔ B1/B3 `col_reduce_scatter` (710).
- F2 `row_allgather` (750) ↔ B2 `row_reduce_scatter_last` (760).

Chain grads come back in ops order `(dACd, dFR, dM1, dM2, dFL)`
(`chain_engine.jl`); permuted to map arg order `ACdmap(ACd, FL, FR, M)` →
`(dACd, dFL, dFR, dM)` exactly as `engine_backward(::typeof(ACdmap),…)`
(`chain_maps.jl:147`). **No new primitive; no new hand kernel.**

---

## 6. Signatures, rrule shape, chunking, AD capture

### 6.1 Signatures (mirror `FLmap_slice2d_dist`)

```julia
function ACmap_slice2d_dist(AC_blk, FL_blk, FR_blk, M, grid::Slice2DGrid;
                           forloop_iter = 1, inner_etype = nothing)
function ACdmap_slice2d_dist(ACd_blk, FL_blk, FR_blk, M, grid::Slice2DGrid;
                            forloop_iter = 1, inner_etype = nothing)
```

`M` is leg5 or `(M1,M2)` tuple. **Single-M convention (pinned to
`FLmap_slice2d_dist`, `rules.jl:592,572,690`):** at entry set
`is_tuple = M isa Tuple; M1, M2 = is_tuple ? M : (M, conj(M))`, then run the
**2M tuple chain** `ACMAP_LEG5_CHAIN` / `ACDMAP_LEG5_CHAIN` throughout the
forward and backward (the rrule never touches the `_1M` conj_variant chains).
B4 always yields `(dM1, dM2)`; the rrule composes
`dM = is_tuple ? (dM1, dM2) : dM1 .+ conj(dM2)` at the boundary. (This is byte-for-byte the
FLmap rrule's M-handling; the chain-internal `_1M` slot-sum `g[3]+g[4]` of
`engine_backward` gives the identical value because slot 4 is conj-flagged, but
the slice2d rrule follows the FLmap 2M-chain + boundary-compose path for one
code path.) `do_cast = inner_etype !== nothing && inner_etype != real(eltype(AC_blk))`;
`_downcast_eltype` every operand at entry, `T_orig.(…)` results/grads at exit —
byte-for-byte the FLmap_slice2d_dist boundary-cast block (`rules.jl:594-607,
691-694`). MPI travels in `inner_etype` (2× bandwidth for F32). Densify
non-`DenseArray` cotangents before any `MPI.Isend` (FillArrays guard,
`rules.jl:618-622`). Rank-uniform control flow (every rank runs the identical
collective sequence, else deadlock).

rrule return: `(NoTangent(), dAC_blk, dFL_blk, dFR_blk, dM, NoTangent())` /
`(NoTangent(), dACd_blk, dFL_blk, dFR_blk, dM, NoTangent())`, `dM = (dM1,dM2)` if
tuple else `dM1 .+ conj(dM2)`.

### 6.2 `forloop_iter` chunking — which leg(s)?

The chunk leg(s) must **appear in the internal chain intermediates I1/I2/I3** and
bound them. The two maps differ here because their intermediates carry different
χ legs (verified by `tensor_pinned_inters`, §2.7 / §5.1):

**ACmap — chunk the local `l`-block (one loop), unchanged from FLmap.** ACmap's
intermediates `I1=(a,c,h,l,b,g)`, `I2=(a,l,e,j,c,h,p)`, `I3=(l,j,k,a,e,f)` carry
the **local** `a`-block and the **output** `l`; the output `i` and contracted `d`
never appear as intermediate legs (FL is last; `d` is consumed at link 2). So
chunking the local `l`-block (the last leg of `FR_g` and of `partial`) bounds
them, identical to FLmap's "chunk the local l range". `l_chunks =
split_ranges(nl, min(forloop_iter, nl))`, `nl = length(p_rs[r2+1])`. The chunk is
over the **output** `l` leg: each chunk is an independent slice of `partial`'s
last leg → assembled by `view(partial,:,:,:,ch) .= …`. Per-chunk transient bounded
by ≈(1+d)·χ²D⁴/(P·forloop_iter) forward, ≈(2+2d)·… backward (P = N1·N2 = N²). The
`min(forloop_iter,nl)` clamp matches FLmap. **This is unchanged and correct.**

**ACdmap — chunk `i` (contracted) AND `d` (output), a 2-level loop.** ACdmap's
intermediates `I1=(i,k,d,h,j,g)`, `I2=(i,d,e,b,k,h,p)`, `I3=(d,b,c,i,e,f)` carry
**full `i` × full `d`** and **never `l`** (`l` is contracted at link 2 and gone
from I1 on). Chunking `l` therefore bounds **nothing internal** — that was the
BLOCKER. The legs that bound I1/I2/I3 are `i` and `d`; chunk both:
`i_chunks = split_ranges(χ, n_i)`, `d_chunks = split_ranges(χ, n_d)`. `i` is
contracted → its per-chunk chain outputs **accumulate** into `partial` (slices
`ACd_g`/`FL_g`); `d` is output → its per-chunk chain output is a **disjoint
assign** into `partial[:,:,:,dch]` (slices `FR_g`). The contracted local `l`-block
is **not** chunked — it is summed whole inside each `chain_apply` (it is not an
intermediate leg, so it costs nothing to keep whole). Per-chain transient bounded
by ≈(1+d)·χ²D⁴/(n_i·n_d) forward, ≈(2+2d)·… backward.

**Sizing `n_i`, `n_d` to the ACmap class.** To match ACmap's
`χ²D⁴/(P·forloop_iter)` set `n_i·n_d = P·forloop_iter = N²·forloop_iter`. Single
knob: `n_i = n_d = N·⌈√forloop_iter⌉` (product `≥ N²·forloop_iter`); or `n_i =
n_d = N·forloop_iter` for the simpler-to-derive (and strictly tighter)
`χ²D⁴/(N²·forloop_iter²)`. Clamp each `n` to `≤ χ` (the `min(·,χ)` analog of
FLmap's `min(forloop_iter,nl)`). The same `n_i`, `n_d` are used in the backward
(§5.2, B4) since `chain_backward` recomputes the same I1/I2/I3.

The **public knob stays `forloop_iter`** (same signature as ACmap/FLmap, §6.1):
`ACdmap_slice2d_dist` derives `n_i = n_d = N·⌈√forloop_iter⌉` internally from
`grid.N` and `forloop_iter`, so callers and the test harness pass a single
`forloop_iter` for both maps. (Exposing `n_i`/`n_d` separately is a possible
future refinement but unnecessary for v1.)

**Structural asymmetry to code carefully.** ACmap's single `l`-loop **assigns**
disjoint output slices; ACdmap's inner `i`-loop **accumulates** (`Σ_i`) while its
outer `d`-loop **assigns** disjoint `d`-slices. A copy-paste of the ACmap
assignment loop into the ACdmap `i`-loop would silently drop all but the last
`i`-chunk — use `acc .+= chain_apply(…)` / `view(dXXX_g, …) .+= …` for the
`i`-accumulation and `_free!` each per-chunk piece (the `chain_apply` API returns
a fresh array per call).

### 6.3 What is cached for AD (the FLmap `blocks` analog)

FLmap captures the ring-visited FL blocks (`blocks::Vector`, χ²D²/N per rank). The
cross-axis maps have **no ring** (the gathers replace the ring), so the AD capture
is the **gathered slices**: `AC_g, FR_g, FL_g` (ACmap) / `ACd_g, FR_g, FL_g`
(ACdmap), each χ²D²/N. These are caller-recomputable inputs to the local chain;
the rrule closure captures them (and `p_rs`, `M1`, `M2`, plus the chunk counts —
`forloop_iter` for ACmap, `n_i`/`n_d` for ACdmap) — **never a χ²D⁴ array, never a
χ×χ (i,l)/(a,d) plane.** `partial` (χ²D²/N) is forward-only and not captured;
`chain_backward` recomputes the chain's internal intermediate analogs from the
captured slices, exactly the recompute-style contract of `chain_backward`
(`chain_engine.jl:231`, "recomputes I₁..I_{N-2} but never the final output").

**The captured slices are bounded (χ²D²/N), but they are NOT what bounds the
recompute peak.** For ACmap the recomputed I1/I2/I3 carry local `a` + chunked
`l` → bounded by the `l`-chunk. For ACdmap the recomputed I1/I2/I3 carry **full
`i` × full `d`** — so the backward peak is bounded **only** by feeding
`chain_backward` the `i`/`d`-chunk slices (§5.2, B4), exactly as the forward feeds
them to `chain_apply`. The capture footprint is 3·χ²D²/N (same order as FLmap),
but the ACdmap backward transient is `χ²D⁴/(n_i·n_d)`-class, recovered by the
2-level chunk — capturing bounded slices alone would NOT bound the recompute.

Implementation choice mirroring FLmap_slice2d_dist: do the three gathers **once**
in the primal (outside the rrule's backward closure), capture the gathered
slices, and run `_acmap_forward_sliced` / `_acdmap_forward_sliced` cores shared
by the non-AD and rrule paths (the FLmap `_slice2d_forward_sliced` pattern). In an
iteration with fixed FL/FR (e.g. the ACenv power loop), hoisting the gathers out
of the loop is the natural optimization, as noted for FLmap_slice2d_dist.

---

## 7. Chain-engine composition (no new hand kernels)

The local einsum **and** its adjoint are the chain engine:
- forward local: `chain_apply(ACMAP_LEG5_CHAIN, …)` / `chain_apply(ACDMAP_LEG5_CHAIN, …)`.
- backward local: `chain_backward(ACMAP_LEG5_CHAIN, …)` / `chain_backward(ACDMAP_LEG5_CHAIN, …)`.

Both chains already exist and are parity-tested (`chain_maps.jl:106,139`;
`test_chain_maps.jl`). The pinned intermediate layouts (`tensor_chain` →
`tensor_pinned_inters`) give cuTENSOR the same permutation problems as the proven
hand kernels — the M1 Part-7 lesson, inherited for free. **The ONLY hand-written
code in M3 is communication** (and it is all existing primitives — gathers and
reduce-scatters; §2.6 and §5 prove no new comm primitive is required for v1).
Eager-free discipline is preserved end to end: `chain_apply`/`chain_backward`
free their owned intermediates internally; the rrule frees the per-chunk grad
contributions (`dAC_g_c` etc.) and the gathered-slice grad buffers
(`dAC_g`/`dFR_g`/`dFL_g`) right after their reduce-scatters. **NEVER Zygote inside
the map rrule or inside any map's chain backward** (MEMORY: slice2d+chain engine).

---

## 8. Test strategy (2×2 grid, 4 ranks, parity vs serial)

Extend `test/test_slice2d.jl` (4-rank harness, CPU `Array`, `@assert
Comm_size==4`). All comparisons via `slice2d_scatter`/`slice2d_gather` (+ their
rrules) and Zygote sum-loss, exactly like the FLmap blocks. **Square grid only:
`(N1,N2) = (2,2)`** (M3 v1 scope; the `(1,4)`/`(4,1)` rectangular cases are
explicitly skipped with an `@test_skip` noting M3 v2). χ ∈ (16, 18) to exercise
uneven `p_rs`; `forloop_iter ∈ (1, 3)` including the clamp path (`forloop_iter=99
> nl`).

```julia
using TeneT: ACmap, ACdmap, ACmap_slice2d_dist, ACdmap_slice2d_dist
make_leg5 reused (FL, ALu→FL/FR roles, AC/ACd via rand(ComplexF64, χ, D, D, χ))

# Forward parity (1e-12):
ref  = ACmap(AC, FL, FR, (M1, M2))
out  = slice2d_gather(ACmap_slice2d_dist(
          slice2d_scatter(AC,g), slice2d_scatter(FL,g), slice2d_scatter(FR,g), (M1,M2), g; forloop_iter=n), g)
@test out ≈ ref rtol = 1e-12
# CRITICAL off-diagonal assertion (the trap): with χ=18, N=2, p_rs=[1:9,10:18],
# explicitly check an off-diagonal (i,l) block is nonzero AND correct, e.g.
@test out[1:9, :, :, 10:18] ≈ ref[1:9, :, :, 10:18] rtol = 1e-12   # i-block 0, l-block 1 (A≠B)
@test out[10:18,:, :, 1:9]  ≈ ref[10:18,:, :, 1:9]  rtol = 1e-12

# Gradient parity (1e-10), block-wise for distributed args, replicated for M:
Wb = slice2d_scatter(W, g)
loss_ref(AC,FL,FR,M1,M2) = real(sum(W .* ACmap(AC,FL,FR,(M1,M2))))
loss_dist(ACb,FLb,FRb,M1,M2) = real(sum(Wb .* ACmap_slice2d_dist(ACb,FLb,FRb,(M1,M2),g; forloop_iter=n)))
g_ref  = Zygote.pullback(loss_ref, AC,FL,FR,M1,M2)[2](1.0)
g_dist = Zygote.pullback(loss_dist, ACb,FLb,FRb,M1,M2)[2](1.0)
blkof(x) = x[p_rs[g.r1+1], :, :, p_rs[g.r2+1]]
@test g_dist[1] ≈ blkof(g_ref[1]) rtol=1e-10   # dAC block
@test g_dist[2] ≈ blkof(g_ref[2]) rtol=1e-10   # dFL block
@test g_dist[3] ≈ blkof(g_ref[3]) rtol=1e-10   # dFR block
@test g_dist[4] ≈ g_ref[4]        rtol=1e-10   # dM1 replicated
@test g_dist[5] ≈ g_ref[5]        rtol=1e-10   # dM2
```

ACdmap test is identical with `ACdmap`/`ACdmap_slice2d_dist`, output convention
`[a,b,c,d]` (a on r1, d on r2 → `blkof` for the **output** uses
`[p_rs[r1+1],:,:,p_rs[r2+1]]` as well, since a is first / d is last). The
off-diagonal assertion targets the `(a,d)` plane (`out[1:9,:,:,10:18]` etc.).
Also include: single-M entry (`dM = dM1 + conj(dM2)` composition), bare-`sum`
loss (FillArrays densify guard), `inner_etype=Float32` boundary cast (fwd 1e-4,
grad 1e-3), and an **iterability** check for ACmap (output block feeds straight
back as input: `ACmap_slice2d_dist(out_blk, FLb, FRb, …)` matches the serial
double-apply) — ACdmap is not self-iterating (output is `{a,d}` top, input is
`{i,l}` bottom), so its "iterate" test instead feeds its output into a matching
`ACmap_slice2d_dist` and compares the composed serial map.

GPU coverage: extend `examples/MPI_parallel/test_slice2d_sofia.jl` (the
CUDA-aware path) once the CPU parity passes.

---

## 9. Open risks / edge cases

1. **ACdmap 2-level chunk: accumulate `i`, assign `d` (§5.1/§6.2).** ACdmap
   chunks **both** the contracted `i` (inner loop) and the output `d` (outer
   loop). The inner `i`-loop **accumulates** (`Σ_i`: `acc .+= chain_apply(…)`,
   `view(dXXX_g,…) .+= …`); the outer `d`-loop **assigns** disjoint output slices
   (`partial[:,:,:,dch] = …`). The contracted local `l`-block is summed whole
   inside each `chain_apply` (not chunked). A copy-paste of ACmap's single
   assignment loop would (a) drop all but the last `i`-chunk (missing the `Σ_i`
   accumulation) and (b) fail to bound the internal intermediates at all. The
   `chain_apply` API returns a fresh array per call; `_free!` each per-chunk
   piece. **Highest-risk detail; the off-diagonal + `forloop_iter=3` test (which
   forces `n_i,n_d ≥ 2` per §6.2) is the guard.** (This replaces the earlier-draft
   "chunk the contracted `l`" risk, which both under-bounded memory and described
   the wrong leg.)

2. **`d_rs` argument to col/row primitives is the OUTPUT block range, not the
   input's.** All existing reduce-scatter/allgather take a `*_rs` of the leg they
   act on. For ACmap B3 (`dFR_g` first leg `d`, reduce over col) the range is
   `p_rs` for `d` (= a_rs); for B2/B1 (last leg `i`/`d`, reduce over row) it is
   `p_rs` for `i`/`d` (= l_rs). On a square grid `a_rs == l_rs == p_rs`, so a
   single `p_rs` is correct everywhere — but this is **only** true because
   `N1==N2`. The `@assert N1==N2` is what licenses passing one `p_rs` to every
   primitive; drop it and the ranges diverge silently. Document inline.

3. **Memory: the dominant term is the ACdmap internal chain intermediates, NOT
   the gathered slices.** Two separate contributions:
   - *Gathered slices* (both maps): three coexist — AC_g/FR_g/FL_g (ACmap) or
     ACd_g/FR_g/FL_g (ACdmap), 3·χ²D²/N forward, +3 grad buffers backward — vs
     FLmap's two AL slices + ring blocks. Bounded, χ²D²/N each.
   - *Internal chain intermediates* (the BLOCKER, ACdmap-specific): ACdmap's
     I1/I2/I3 carry **full `i` × full `d`** ≈ χ²D⁴ (I2 χ²D⁵) ≈ **25 GB each** at
     D=10 χ=400 — a genuine `χ×χ` plane. **`forloop_iter` alone does NOT bound
     this**, because the first draft chunked `l`, which is absent from I1/I2/I3.
     The mitigation is the **2-level `i`/`d` chunk** (§5.1, §6.2): set
     `n_i·n_d = P·forloop_iter` to reach the ACmap-class
     `χ²D⁴/(P·forloop_iter)`. ACmap's intermediates carry local `a` + chunked `l`
     and are already bounded by the single `l`-chunk — ACmap is unaffected.

   At production (D=10, χ=400, P=16) ACdmap's intermediate term dominates the
   gathered slices by `~D²` — size `n_i`/`n_d` against the **backward** bound
   (`~(2+2d)·χ²D⁴/(n_i·n_d)`), not against the slice term. Verify the peak on
   Sofia before declaring M3 done (the FLmap OOM lesson, job 1274256 / 1265371).

4. **No transpose primitive lands in v1.** §2.6/§5 prove v1 fuses every cross-axis
   move into existing gather + reduce-scatter pairs. If a future consumer needs a
   raw block transpose (rectangular grid, or a map whose output cross-axis leg has
   no matching reduction to fuse with), implement `_slice2d_block_transpose`
   (tag 770, square-grid involution, §3) — reserved but unbuilt.

5. **`partial` first-leg fullness assumption in col_reduce_scatter (ACmap F5).**
   `_slice2d_col_reduce_scatter` extracts `partial[d_rs[j+1],…]` per destination
   (`slice2d.jl:245`). ACmap's `partial` first leg is **full `i`**, so the
   extraction is valid (it slices the full `i` into r1-blocks). This relies on F4
   producing full `i` — which the FL row_allgather (F2) guarantees. If F2 were
   ever skipped (optimization), F5 would silently read out-of-range. Keep F2→F5
   coupled; assert `size(partial,1) == χ`.

6. **Rectangular grid (`N1≠N2`) is genuinely different**, not a parameter tweak:
   `p_rs` for the first leg (N1) and last leg (N2) differ, the gathers produce
   mismatched ranges, and the output redistribution can no longer fuse into a
   single reduce-scatter (the scatter axis and sum axis use different partitions).
   v2 needs the real `_slice2d_block_transpose` with `alltoallv` per-(src,dst)
   counts. **Explicitly out of scope for M3 v1**; the square assertion is the gate.

7. **Single-M composition — PINNED to the FLmap precedent (no ambiguity).** The
   slice2d rrule routes the single-M case through the **2M tuple chain
   internally**: materialize `M1, M2 = (M, conj(M))` at entry (`rules.jl:592`),
   run `ACMAP_LEG5_CHAIN`/`ACDMAP_LEG5_CHAIN` (the 2M chains, never the `_1M`
   conj_variants) in F4/B4, and compose `dM = dM1 + conj(dM2)` at the **rrule
   boundary** on exit (`rules.jl:572,690`). The chain-internal slot-sum
   `dM = g[3] + g[4]` of `engine_backward(::typeof(ACmap/ACdmap),…)` over the
   `_1M` chain (`chain_maps.jl:118-121,150-153`) yields the **same value** —
   slot 4's conj flag makes `g[4]` the `conj(dM2)`-equivalent, so `g[3]+g[4] ≡
   dM1+conj(dM2)` — but that is the *serial* path, NOT the rrule path. One
   convention everywhere: 2M chain + `dM1 + conj(dM2)` boundary compose, byte-for-byte
   the FLmap rrule. Test both entry points (tuple-M and single-M), as the FLmap
   tests do.

---

## 10. Sign-off summary (post-review revision)

Both review blockers are resolved; the rest of the design is unchanged and was
confirmed correct (diagonal-trap fix, ACmap forward/backward, adjoint pairings,
no-new-primitive decision, square-grid assertion, test strategy).

- **ACdmap memory (was BLOCKER).** Root cause: `tensor_pinned_inters` on
  `ACDMAP_LEG5_CHAIN` gives I1=(i,k,d,h,j,g), I2=(i,d,e,b,k,h,p),
  I3=(d,b,c,i,e,f) — all carry **full `i` × full `d`** (`χ²D⁴`, I2 `χ²D⁵`);
  `l` (the first-draft chunk leg) is absent. Fix: 2-level chunk over `i`
  (contracted → accumulate) and `d` (output → assign), forward (§5.1) and
  backward (§5.2) identically. **Proven achieved per-intermediate bound:**
  `peak ~ χ²D⁴/(n_i·n_d)`; setting `n_i·n_d = P·forloop_iter` (P = N1·N2 = N²)
  reaches the ACmap-class `χ²D⁴/(P·forloop_iter)` — no `χ×χ (i,d)` plane ever
  materialized.
- **Single-M `dM` (was MAJOR).** Pinned to the `FLmap_slice2d_dist` precedent:
  2M tuple chain internally (`M1,M2 = (M, conj(M))`), compose
  `dM = dM1 + conj(dM2)` at the rrule boundary (§§4.1/4/6.1, risk 7). The
  chain-internal `_1M` slot-sum is noted as the equal-value serial path, not the
  rrule path.
- **Untouched (confirmed correct):** ACmap forward/backward and its single
  `l`-chunk (its intermediates I1=(a,c,h,l,b,g), I2=(a,l,e,j,c,h,p),
  I3=(l,j,k,a,e,f) carry local `a` + chunked `l`, never `i`/`d`); all adjoint
  pairings; no-new-primitive; the `@assert N1==N2` gate; the test strategy.
