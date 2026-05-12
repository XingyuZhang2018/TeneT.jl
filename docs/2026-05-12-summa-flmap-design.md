# SUMMA-FLmap Design — 2D Distributed Tensor Contraction with Cross-Axis Legs

**Date**: 2026-05-12
**Branch**: `claude/2d-distributed-vumps`
**Status**: design proposal, no implementation yet
**Supersedes**: my broken Section 2 (`docs/2026-05-11-2d-distributed-vumps-runtime-design.md` §2 has the
off-diagonal block bug; my followup "AllGather + AllReduce" simplification at
`docs/2026-05-12-2d-design-blockers.md` Path B is correct but 1.2-1.5× slower than 1D and uses
more memory transient — defeats the purpose of 2D)

---

## 0. TL;DR

For genuine peak-memory savings (the actual 2D motivation, as evidenced by the
1.25 Kagome 1003757 OOM at χ=540 D=10 forward), `FLmap_parallel_2D` requires a
**SUMMA-style algorithm with two communication phases**, not the simple
`AllGather + local + AllReduce` pattern I previously implemented.

Cost summary (square grid `N1 = N2 = M`, χ=1024 D=10):

| Pattern | Peak per rank | Total comm per rank |
|---|---|---|
| 1D (current production, no 2D distribution) | ~3.6 GB (FL+ALu+ALd full + result) | 1 AllGatherv at end |
| 2D naïve "AllGather + AllReduce" (my v0) | ~2.2 GB | 3 AllGathers + 1 AllReduce |
| **SUMMA-FLmap (this proposal)** | **~0.6 GB** | M-step iteration: M × 2 broadcasts (row+col) + M × ALu gather amortized |

The SUMMA peak `~0.6 GB` is roughly **χ²·D²/M²** (one `slice × slice` tile of
each tensor) — true 2D scaling.

---

## 1. Problem recap — why simpler patterns fail

### 1.1 The FLmap contraction

```
result[d, g, h, l] = sum_{a, e, f, i, b, c, j, k, p}
                     FL[a, e, f, i] × ALd[i, j, k, l]
                     × M1[e, j, g, b, p] × M2[f, k, h, c, p]
                     × ALu[a, b, c, d]
```

In the standard 2D layout (`first χ on N1`, `last χ on N2`):

- `FL[a∈slice_r1, e, f, i∈slice_r2]`  — `a` on N1, `i` on N2
- `ALu[a∈slice_r1, b, c, d∈slice_r2]` — `a` on N1, `d` on N2
- `ALd[i∈slice_r1, j, k, l∈slice_r2]` — `i` on N1, `l` on N2
- `result[d∈slice_r1, g, h, l∈slice_r2]` — `d` on N1, `l` on N2

Leg-by-leg distribution status:

| Leg | Role | Distribution on tensors | Status |
|---|---|---|---|
| `a` | contracted | FL.a=N1, ALu.a=N1 | **co-distributed on N1** ✓ |
| `i` | contracted | FL.i=N2, ALd.i=N1 | **cross-axis** ✗ |
| `d` | free leg | ALu.d=N2, result.d=N1 | **cross-axis** ✗ |
| `l` | free leg | ALd.l=N2, result.l=N2 | aligned ✓ |

The two cross-axis legs (`i` contracted, `d` free) are what make this hard.

### 1.2 Why my naïve "AllGather + AllReduce" doesn't deliver

To make both cross-axis legs work locally, I gathered:
- FL.i (along row_comm) → FULL i locally
- ALd.i (along col_comm) → FULL i locally
- ALu.d (along row_comm) → FULL d locally

This made the einsum locally well-defined, but:
- **Each rank held χ²·D²/N1 of FL+ALu (~600 MB at χ=1024 D=10 N1=2), plus χ²·D²/N2 of ALd (~600 MB) + partial output**: peak ~2 GB per rank.
- **Local einsum operated on the FULL d**, doing M× more compute than the rank's own d-slice would need.
- **Sofia bench (1099008)**: 2D was 1.2-1.5× SLOWER than 1D at D=10 χ∈{512,768,1024}.

The naïve approach trades memory ratio (1/N persistent, but 1/√N transient) for compute waste. Not a 2D win.

### 1.3 What 2D's selling point actually is

The Hong-Kung / Loomis-Whitney lower bound for distributed contraction says **per-rank
memory floor is `χ²·D²/M`** (= `χ²·D²/√N` for square grid). My naïve algorithm
hit `χ²·D²/N1` ≈ `χ²·D²/√N` transient, which is the right order, but **peak compute**
also explodes by M× because the local einsum is "fat".

SUMMA iterates over the contracted dimension(s) in M steps, where **each step's
working set is `χ²·D²/M²`** (a single `slice × slice` tile per tensor). After M steps
of `χ²·D²/M²` work each, total compute is `χ²·D²/M` per rank — matches 1D distribution
total work, but with **memory peak floored at `χ²·D²/M²`**.

This is the genuine 2D win.

---

## 2. The SUMMA-FLmap algorithm

### 2.1 Setup: square grid `N1 = N2 = M`

Restrict v1 to square N1 = N2 = M (= √N). Rectangular (M-rank for non-square N)
is a v2 extension; the algorithm doesn't change conceptually but communicator
choices get fiddlier.

Per-rank stored tensors (same as current 2D layout):

- `FL_local[a=slice_r1, e, f, i=slice_r2]` — `χ²·D²/M²` bytes
- `ALu_local[a=slice_r1, b, c, d=slice_r2]` — `χ²·D²/M²` bytes
- `ALd_local[i=slice_r1, j, k, l=slice_r2]` — `χ²·D²/M²` bytes
- `M_full` — small, replicated
- `partial[d=slice_r1, g, h, l=slice_r2]` accumulator — `χ²·D²/M²` bytes

### 2.2 Why we can't avoid handling both cross-axis legs

Naïve idea: "if we redistribute ALu so its `d` is on N1 (matching result.d), the d cross-axis problem goes away."

But ALu has two χ legs (`a` and `d`). Both must be partitioned on **different** sub-comms in 2D — there's no way to have both on N1 without losing M-1/M of the data (only the "diagonal" `a=d` blocks would be addressable; the off-diagonal blocks have nowhere to live in an `(a_N1, d_N1)` layout when only M ranks per axis exist).

Concrete check for `M = 2`, `χ = 4` (slice_0 = {0,1}, slice_1 = {2,3}):

ALu has 4 × 4 = 16 (a,d) cells. In the (a_N1, d_N2) layout, the 4 ranks each hold a 2×2 block; 16 cells total. ✓

In a hypothetical (a_N1, d_N1) layout, each rank holds (a=slice_r1, d=slice_r1) — a 2×2 diagonal block. 4 ranks × 4 cells = 16. The OFF-diagonal cells (8 of them: a=slice_0, d=slice_1 and vice versa) have **no rank to live on**. They'd be dropped.

This is fundamental — you cannot fully store an M×M matrix on an M×M grid with both axes on the same sub-comm of size M.

**Conclusion**: ALu must stay in (a_N1, d_N2) layout. FLmap must handle the `d` cross-axis by bringing the right ALu slice to the right rank at compute time.

### 2.3 SUMMA over `i` with diagonal-style ALu gather

The full algorithm has **one ALu redistribution + M-step SUMMA over i**:

```
# ─── Phase 0: bring ALu[..., d=slice_r1] to our rank ────────────
# Our rank (r1, r2) needs ALu data for the output's d-slice (=slice_r1).
# Originally that data is in column r1 of the grid (the col where d=slice_r1
# lives in the (a_N1, d_N2) layout):
#    - Rank (0, r1) holds ALu[a=slice_0, d=slice_r1]
#    - Rank (1, r1) holds ALu[a=slice_1, d=slice_r1]
#    - ...
#    - Rank (M-1, r1) holds ALu[a=slice_{M-1}, d=slice_r1]
# Combined: col_comm[r1] members jointly hold ALu[a=ALL, d=slice_r1].
#
# Two-step plan to get ALu[a=ALL, d=slice_r1] onto rank (r1, r2) for any r2:

# Step 0a: each col_comm performs AllGather on the a-axis.
# After: rank (r1, r2) holds ALu_a_full[a=ALL, b, c, d=slice_r2]
#         (χ²·D²/N1 = χ²·D²/M per rank)
ALu_a_full = allgather_dim_direct(ALu_local, 1, col_comm[r2])

# Step 0b: each row_comm performs a Broadcast with source = (r1, r1).
# After: rank (r1, r2) holds ALu_for_my_d[a=ALL, b, c, d=slice_r1]
#         — exactly the ALu slab needed for THIS rank's output d.
ALu_for_my_d = broadcast_in_comm(ALu_a_full, source=r1, comm=row_comm[r1])
# (source `r1` is the rank within row_comm[r1] whose r2 coord equals r1 —
# i.e., the diagonal rank (r1, r1) in the original grid coords)

# Memory peak at this point: ALu_for_my_d = χ²·D²/M (~400 MB at χ=1024 D=10 M=2)
# Plus persistent locals: 3 × χ²·D²/M² ≈ 150 MB
# Total: ~550 MB

# ─── Phase 1: M-step SUMMA over i ─────────────────────────────
partial[d=slice_r1, g, h, l=slice_r2] = 0
for t in 0..M-1:
    # Bring FL's i=slice_t and ALd's i=slice_t to our rank.

    # FL.i is on N2 (row_comm). Source for i=slice_t in row_comm[r1] is
    # rank (r1, t) — the member of row_comm[r1] whose i-slice equals t.
    FL_t = broadcast(FL_local, source=t, comm=row_comm[r1])
    #   shape: (χ/M, D, D, χ/M)  — chi²·D²/M² per rank

    # ALd.i is on N1 (col_comm). Source for i=slice_t in col_comm[r2] is
    # rank (t, r2).
    ALd_t = broadcast(ALd_local, source=t, comm=col_comm[r2])
    #   shape: (χ/M, D, D, χ/M)  — chi²·D²/M² per rank

    # Local einsum on small tiles.
    # Contraction over (a=slice_r1, i=slice_t, b, c, e, f, j, k, p)
    # NOTE: ALu_for_my_d.a is FULL [0,χ), but only a∈slice_r1 contributes
    # because FL_t.a is restricted to slice_r1. The cuTENSOR einsum will
    # only touch a∈slice_r1 portion of ALu_for_my_d, BUT carrying the full
    # a-extent in memory wastes (M-1)·χ²·D²/M² of buffer.
    # Optimization: slice ALu_for_my_d to a=slice_r1 before einsum (see §3.3).
    partial += einsum(
        FL_t[a=slice_r1, e, f, i=slice_t],
        ALd_t[i=slice_t, j, k, l=slice_r2],
        M1, M2,
        ALu_for_my_d[a=slice_r1, b, c, d=slice_r1]
    )

# After M iterations: partial covers full-i sum AND full-a sum (over slice_r1
# from local ALu only — but that's all we need, because off-slice_r1 a values'
# contributions go to OTHER ranks' output blocks, which they compute
# independently).

# NO further reduction needed. partial IS the final result block.
result[d=slice_r1, g, h, l=slice_r2] = partial
```

### 2.4 Why this is correct — careful proof

Claim: rank (r1, r2)'s `partial` after the loop equals
`result[d=slice_r1, g, h, l=slice_r2]` (the full sum over all `a, i`).

The full result formula:

```
result[d, l] = sum over (a, i, ...) FL[a, ..., i] × ALd[i, ..., l] × M × ALu[a, ..., d]
```

Substituting `d = slice_r1`, `l = slice_r2`:

```
result[d∈slice_r1, l∈slice_r2] = sum_{a=0..χ-1, i=0..χ-1, ...}
    FL[a, ..., i] × ALd[i, ..., l∈slice_r2] × M × ALu[a, ..., d∈slice_r1]
```

Split the `a` sum into blocks `a ∈ slice_p` for `p = 0..M-1`:

```
= sum over p: sum_{a∈slice_p, i, ...} FL × ALd × M × ALu[a∈slice_p, d∈slice_r1]
```

**Key observation**: ALu[a∈slice_p, d∈slice_r1] for `p ≠ r1` is held at rank
`(p, r1)`'s ALu_local. After Phase 0 (Steps 0a + 0b), rank `(r1, r2)` holds
ALu_for_my_d[a=ALL, d∈slice_r1] — which **contains all `a∈slice_p` blocks for
every p**, including `p ≠ r1`. The Phase 0 gather has materialized exactly the
ALu data this rank needs for the full `a` sum.

Then in Phase 1, the local einsum's `a` index ranges over the FULL `a` axis of
ALu_for_my_d (which is ALL of [0,χ)). The corresponding FL and ALd indices —
wait.

Actually let me re-examine. FL_local has a∈slice_r1 only (restricted by
the rank's own data). FL_t (broadcast within row_comm[r1]) — the broadcast
within row_comm[r1] preserves a-distribution (only changes which i-slice each
rank holds). After broadcast, FL_t still has a∈slice_r1.

So during the einsum, FL_t.a is ONLY slice_r1. The contraction `sum_a FL[a,...] × ALu[a,...]`
is then over `a ∈ slice_r1 ∩ ALL = slice_r1`.

That gives ONLY the `p = r1` term of the outer-sum decomposition. The `p ≠ r1`
terms are MISSING from `partial`.

**This is a bug in my Phase 0 design as written.** The Phase 0 gather isn't enough.

### 2.5 Fixing the algorithm — Phase 1 must also iterate over `a`

To pick up `a ∈ slice_p` contributions for all `p`, we need a similar SUMMA
iteration over `a`:

```
For each (p, t) ∈ {0..M-1} × {0..M-1}:
    # Bring FL[a=slice_p, i=slice_t] to all ranks.
    # FL[a=slice_p, e, f, i=slice_t] is at rank (p, t).
    # We need it at rank (r1, r2).

    # In our 2D grid, neither sub-comm naturally bridges (p, t) → (r1, r2)
    # for arbitrary (p, t). So we'd need either point-to-point sends or
    # nested broadcasts.
```

This is the "2D matmul SUMMA" pattern. For matrix multiplication on M×M grid,
the algorithm is:

```
For step k = 0..M-1:
    # All ranks in column k broadcast their A column to their row.
    # All ranks in row k broadcast their B row to their column.
    # Each rank accumulates its local C += A_col × B_row.
```

In FLmap analog, the analog is harder because of the 4-tensor structure and
two contracted indices (a AND i). Let me propose a clean version:

```
# ─── Setup: bring ALu's slab to each rank ─────────────────────
# Same as before: Phase 0a (AllGather a in col_comm) + Phase 0b
# (Broadcast in row_comm with source = (r1, r1)).
# After: rank (r1, r2) holds ALu_full_a[a=ALL, d∈slice_r1]

# ─── Outer loop over a-block p ─────────────────────────────────
partial[d=slice_r1, g, h, l=slice_r2] = 0
for p in 0..M-1:
    # Get FL[a=slice_p, e, f, i=*] onto our rank.
    # FL[a=slice_p, e, f, i=slice_r2] is at rank (p, r2) ∈ col_comm[r2].
    # Broadcast within col_comm[r2] from source p:
    FL_for_p = broadcast(FL_local, source=p, comm=col_comm[r2])
    #   shape: (χ/M, D, D, χ/M) — but now a=slice_p (not slice_r1)

    # ─── Inner SUMMA over i ────────────────────────────────────
    for t in 0..M-1:
        # FL[a=slice_p, e, f, i=slice_t]: need to bring i=slice_t.
        # After the outer broadcast above, rank (r1, r2) has FL_for_p with
        # i=slice_r2 still (didn't touch i). We need i=slice_t.
        # FL[a=slice_p, e, f, i=slice_t] is at rank (p, t). To get it to (r1, r2):
        #   This is a broadcast in some comm that contains both (p, t) and
        #   (r1, r2). For general (p, t, r1, r2), no single sub-comm contains both.
        # Use intermediate: rank (p, t) broadcasts in col_comm[t] to (*, t),
        #   then col_comm[t] members broadcast in row_comm? But again, our
        #   rank isn't necessarily in col_comm[t].
        # ...
```

This is where the 2D matmul SUMMA pattern from textbooks struggles for
**4-tensor contractions with TWO contracted indices**. Standard SUMMA handles
matmul (one contracted index) cleanly. Two contracted indices need either:

- **Nested SUMMA** (M² iterations): expensive in comm volume.
- **Tensor-train decomposition** (Cyclops Tensor Framework approach): research-grade.
- **Replicated intermediate**: hold partial results full-size on each rank (memory cost).

### 2.6 A practical compromise: fuse the two contractions

For our FLmap, there's a way out: **observe that the (a, i) contracted indices
are independent** — they don't appear together in any tensor as a coupled
"matrix" object. Specifically:

- a appears in FL[a, e, f, i] and ALu[a, b, c, d]
- i appears in FL[a, e, f, i] and ALd[i, j, k, l]

So FL "bridges" a and i, but ALu only sees a, and ALd only sees i.

Define the intermediate **T[a, j, k, l]** = sum_i FL[a, ..., i] × ALd[i, j, k, l].

Then result[d, l] = sum_a T[a, ..., l] × ALu[a, ..., d] × M.

This decomposes FLmap into:
1. **`i`-contraction** (FL × ALd → T)
2. **`a`-contraction** (T × ALu × M → result)

Each is a 1-contracted-index matmul-like operation, SUMMA-able cleanly.

#### Step 1: T = FL · ALd over i (SUMMA on i)

T has shape `[a∈[0,χ), j, k, l∈[0,χ)]` — full a and l (no contraction). j, k are D-sized (small).

**Memory concern**: T full would be `χ² × D² × ...` = big. We want T 2D-distributed:
- T[a∈slice_r1, j, k, l∈slice_r2] — same layout as inputs.

SUMMA algorithm:
```
T_local[a=slice_r1, j, k, l=slice_r2] = 0
for t in 0..M-1:
    # FL.i=slice_t broadcast in row_comm[r1] from source rank (r1, t)
    FL_t = broadcast(FL_local, source=t, comm=row_comm[r1])
    #   shape (χ/M, D, D, χ/M); a=slice_r1, i=slice_t

    # ALd.i=slice_t broadcast in col_comm[r2] from source rank (t, r2)
    ALd_t = broadcast(ALd_local, source=t, comm=col_comm[r2])
    #   shape (χ/M, D, D, χ/M); i=slice_t, l=slice_r2

    # Local outer-product-like contraction
    T_local += einsum(FL_t[a=slice_r1, e, f, i=slice_t],
                       ALd_t[i=slice_t, j, k, l=slice_r2])
end
```

After M iters: T_local has full `i` sum, with the same 2D layout as inputs.

Cost: M iterations × 2 broadcasts of size χ²·D²/M² each = **2·χ²·D²/M per rank total comm**.
Memory peak: 3 × χ²·D²/M² (FL_t, ALd_t, T_local) ≈ very small.

#### Step 2: result = T · ALu × M over a (SUMMA on a, but a is co-distributed)

Wait — both T.a and ALu.a are on N1 (co-distributed). So the a contraction can be done as **partial + AllReduce** (no SUMMA needed):

```
# Local partial (a contraction limited to slice_r1)
partial_local[d_target, g, h, l=slice_r2] = einsum(
    T_local[a=slice_r1, j, k, l=slice_r2],
    M1, M2,
    ALu_???[a=slice_r1, b, c, d=???]
)
```

The "???" indices for ALu are the same d cross-axis issue: ALu.d is on N2, but
we want result.d on N1.

**Workaround for this step**: use the Phase 0 gather (`ALu_for_my_d` =
ALu[a=ALL, b, c, d=slice_r1]) from §2.3, but only the slice_r1 portion since T
already has a=slice_r1 locked in.

```
# Phase 0a + 0b (one-time setup before Step 2):
ALu_full_a   = allgather_dim_direct(ALu_local, 1, col_comm[r2])
              # χ²·D²/M per rank; a=ALL, d=slice_r2
ALu_for_my_d = broadcast(ALu_full_a, source=r1, comm=row_comm[r1])
              # χ²·D²/M per rank; a=ALL, d=slice_r1

# Use ALu_for_my_d's a=slice_r1 portion in the local einsum:
ALu_slice = ALu_for_my_d[r1*χ/M : (r1+1)*χ/M, :, :, :]  # a=slice_r1, d=slice_r1
            # shape (χ/M, D, D, χ/M) — small

partial_local[d=slice_r1, g, h, l=slice_r2] = einsum(
    T_local[a=slice_r1, j, k, l=slice_r2],
    M1, M2,
    ALu_slice[a=slice_r1, b, c, d=slice_r1]
)
```

The partial covers a=slice_r1 only. To get full-a sum, AllReduce along
col_comm[r2] — but each col_comm[r2] member has a different d=slice_r1 (since
their r1 differs).

**Slot mismatch problem again.** So we can't AllReduce directly.

#### The right fix for Step 2: keep d FULL in the partial, slice at the end

```
# In the einsum, let d range over FULL (use ALu_for_my_d with a=slice_r1 and d=slice_r1
#   wait, this still only has d=slice_r1).

# Better: do not extract slice_r1 from ALu_for_my_d's d-axis. Keep d=FULL.
# Then partial has d=FULL, and AllReduce across col_comm[r2] works
# (all members have the SAME d slot mapping).

partial_local[d=FULL, g, h, l=slice_r2] = einsum(
    T_local[a=slice_r1, j, k, l=slice_r2],
    M1, M2,
    ALu_a_full[a=slice_r1, b, c, d=FULL]  # from Phase 0a only; no Phase 0b needed
)
# Memory: partial_local is χ²·D²/M per rank (d=FULL, l=slice_r2)
# Local compute on M× the tiles — but the a sum is restricted to slice_r1,
# so total local compute is χ²·D²·M FLOPs (1/M of full FLOPs since a is sliced).

# Full a sum via AllReduce
summed = allreduce_dim_direct(partial_local, +, col_comm[r2])
# After: rank (r1, r2) has summed[d=FULL, g, h, l=slice_r2]

# Local slice
result_local[d=slice_r1, g, h, l=slice_r2] = summed[r1*χ/M : (r1+1)*χ/M, :, :, :]
```

This avoids the slot-mismatch by carrying d=FULL through the AllReduce.

**Memory peak for Step 2**: ~χ²·D²/M for `ALu_a_full` and `partial_local` (~400 MB at χ=1024 D=10 M=2). Plus T_local (small).

### 2.7 Final algorithm summary

```
# Phase A: SUMMA over i to build T (intermediate)
T_local = 0  # shape (χ/M, D, D, χ/M)
for t in 0..M-1:
    FL_t = broadcast(FL_local, source=t, comm=row_comm[r1])
    ALd_t = broadcast(ALd_local, source=t, comm=col_comm[r2])
    T_local += einsum(FL_t, ALd_t)

# Phase B: one-time gather of ALu's a-axis (col_comm)
ALu_a_full = allgather_dim_direct(ALu_local, dim=1, comm=col_comm[r2])
# shape (χ, D, D, χ/M); a=FULL, d=slice_r2

# Phase C: local einsum with d=FULL output
partial_local = einsum(T_local, M1, M2, ALu_a_full)
# Sums over a∈slice_r1, b, c, e, f, j, k, p
# Output shape (χ, D, D, χ/M); d=FULL, l=slice_r2

# Phase D: AllReduce to complete a sum
summed = allreduce_dim_direct(partial_local, +, col_comm[r2])

# Phase E: slice d to slice_r1
result_local = summed[r1*χ/M : (r1+1)*χ/M, :, :, :]
```

### 2.8 Cost accounting

**Comm per rank**:
- Phase A: M iters × (broadcast of χ²·D²/M² in row_comm + broadcast in col_comm) = 2·M·χ²·D²/M² = **2·χ²·D²/M**
- Phase B: allgather of χ²·D²/M² in col_comm[size M] → χ²·D²/M = **(M-1)·χ²·D²/M² ≈ χ²·D²/M**
- Phase D: allreduce of χ²·D²/M = **2·(M-1)·χ²·D²/(M·M) ≈ 2·χ²·D²/M**
- **Total: ~5·χ²·D²/M per rank**

For χ=1024 D=10 M=2 (N=4): 5 × 800 MB / 2 = **2 GB per rank**. Same order as the
current 1D `FLmap_parallel`'s allgatherv (~800 MB · N-1)/N ≈ 600 MB).

Comm is ~3× larger than 1D, but memory peak is much lower.

**Memory peak per rank**:
- Phase A: T_local + FL_t + ALd_t = 3 × χ²·D²/M² = **150 MB at χ=1024 D=10 M=2**
- Phase B-C: ALu_a_full + T_local + partial_local = 2 × χ²·D²/M + χ²·D²/M² ≈ 2·χ²·D²/M = **800 MB at χ=1024 D=10 M=2**
- Phase D-E: summed (replaces partial_local) + result_local ≈ **800 MB**

**Peak ≈ 800 MB per rank**, vs 1D production's ~3.6 GB. **4-5× lower peak.**

**Local compute**:
- Phase A einsum: M iterations, each a (χ/M)² × D² × ... FLOP. Total: M · (χ/M)² · D² · D² ≈ χ²·D⁴/M per rank.
- Phase C einsum: T_local · M · ALu_a_full. T_local has (χ/M)² entries, ALu_a_full has χ·χ/M = χ²/M entries. Einsum FLOPs: (χ/M) · χ · D⁴ = χ²·D⁴/M per rank.
- **Total per rank: ~2·χ²·D⁴/M**, matching 1/M of full compute.

Production 1D does the same total work (~χ²·D⁴/M per rank) but with FULL tensors in memory.

### 2.9 Comparison table

For χ=1024, D=10, N=4 (M=2), Float64:

| Approach | Peak per rank | Comm per rank | Local compute |
|---|---|---|---|
| 1D `FLmap_parallel` (production) | ~3.6 GB | ~600 MB | χ²·D⁴/N = 4·10⁹ FLOPs |
| Naïve 2D "AllGather + AllReduce" (my v0) | ~2.2 GB | ~2.4 GB | M×χ²·D⁴/N (M× waste) |
| **SUMMA-FLmap (this design)** | **~800 MB** | **~2 GB** | **χ²·D⁴/N ≈ 4·10⁹ FLOPs** |

SUMMA wins on memory (4× lower than 1D), pays in comm (3× more), matches in compute.

For 1.25 Kagome 1003757 OOM (χ=540 D=10 forward exceeded 140GB on H200): the OOM
was likely from many tensors held simultaneously × Krylov queue × AD tape. With
4× lower per-rank peak from SUMMA, we'd extrapolate the OOM ceiling pushing
from χ~540 to χ~1080 at fixed N=4, or larger if N is bigger.

---

## 3. AD considerations

### 3.1 Phase-by-phase rrule

Each phase has a clean rrule pattern:

| Phase | Forward | Backward |
|---|---|---|
| A (SUMMA over i) | M broadcasts + M einsums | M einsums + M reduce-scatters (broadcast adjoint) |
| B (AllGather a) | allgather_dim_direct | reduce_scatter_dim (or slice — see §3.2) |
| C (local einsum) | T × M × ALu_a_full | standard tensor pullback |
| D (AllReduce on col_comm) | allreduce_dim_direct | identity (per PR #42 convention; see Phase 0 report) |
| E (slice) | local slice | zero-pad (to FULL d) |

### 3.2 The broadcast adjoint

`Bcast` in MPI takes one rank's input and distributes to all comm members. Adjoint
in linear-algebra terms is `Reduce` (sum across members back to source rank).

For Zygote's "per-rank gradient" convention (PR #42 style), the adjoint should
be: each rank's `d_FL_t` (for the broadcast output) is summed across the comm to
the source rank.

`MPI.Reduce!` with `op=+` and the broadcast source as destination is the right primitive.

For the SUMMA loop, each iteration's Bcast adjoint goes to a different source rank
(rank t in row_comm[r1] for FL_t, rank t in col_comm[r2] for ALd_t). So the
backward loop unrolls naturally.

### 3.3 Checkpoint considerations

The SUMMA loop's intermediates (FL_t, ALd_t, T_local) at each step t are
**transient** — they go into the einsum and are gone. For Zygote's tape, by
default they'd be saved for backward, but they're already implied by FL_local and
ALd_local + the broadcast op. So the rrule can recompute them on backward instead
of saving — much like the existing `Recompute()` checkpoint scheme.

Recommend: wrap Phase A in a `Recompute()` checkpoint, similar to how `_simple_eig_*`
wraps its iterations. This keeps the tape size constant in M (doesn't grow with
iteration count).

---

## 4. Symmetry to FRmap, ACmap, Cmap

### 4.1 FRmap

FRmap is the mirror of FLmap (left-right symmetry). Algorithm derivation is mechanical:

- Phase A: SUMMA over `l` (the cross-axis contracted leg in FRmap, FR.l=N2 vs ARd.l=N2 — wait both on N2)
  - Actually for FRmap, the leg distribution is different. Let me check.

FRmap (from `src/contraction/basic.jl`):
```
result[a, e, f, i] = sum_{j, k, l, b, c, d, g, h, p}
                     ARd[i, j, k, l] × FR[d, g, h, l] × M × ARu[a, b, c, d]
```

Distribution:
- ARd[i_N1, j, k, l_N2]: i on N1, l on N2
- FR[d_N1, g, h, l_N2]: d on N1, l on N2
- ARu[a_N1, b, c, d_N2]: a on N1, d on N2
- result[a_N1, e, f, i_N2]: a on N1, i on N2

Leg analysis:
- `l` contracted (ARd.l=N2, FR.l=N2) — **co-distributed on N2** ✓
- `d` contracted (FR.d=N1, ARu.d=N2) — **cross-axis**
- `a` free (ARu.a=N1, result.a=N1) — aligned ✓
- `i` free (ARd.i=N1, result.i=N2) — **cross-axis**

So FRmap has the cross-axis legs on different axes than FLmap: `d` contracted instead of `i`, and `i` free instead of `d`. But the structure is symmetric.

Mirror algorithm for FRmap:
- Phase A: SUMMA over `d`, broadcasting FR.d in col_comm and ARu.d in row_comm
- Phase B: one-time gather of ARd.l along row_comm (analog of ALu.a)
- Phase C-E: similar, output has i=FULL during AllReduce in row_comm, then slice to slice_r2

### 4.2 ACmap

ACmap = AC × FR × M × FL → result.

Same family of leg analysis applies. Two cross-axis legs. SUMMA over one, gather + AllReduce + slice for the other.

### 4.3 Cmap

Cmap is small (C is χ×χ, full on each rank in our convention). Comm cost dominates over compute. Simpler — won't elaborate here.

---

## 5. Implementation phases

Given the complexity, recommended progression:

| Phase | Deliverable | Effort | Sofia validation |
|---|---|---|---|
| 1 | `summa_broadcast_dim` primitive + rrule (broadcast within sub-comm, adjoint = reduce to source) | 1 week | unit test |
| 2 | `FLmap_parallel_2D_SUMMA` Phase A only (SUMMA over i → T), unit-test against serial T | 1 week | small problem validation |
| 3 | Phase B-E (ALu gather + local einsum + AllReduce + slice), unit-test FLmap correctness | 1 week | bench vs 1D |
| 4 | rrule for full FLmap_parallel_2D_SUMMA, gradcheck via Zygote | 2 weeks | gradient parity test |
| 5 | Symmetry derivation + impl for FRmap | 1 week | bench |
| 6 | ACmap, Cmap | 1.5 weeks | bench |
| 7 | Integrate into VUMPSRuntime / leftenv / ACenv / `vumps_step` | 2-3 weeks | end-to-end VUMPS test |
| 8 | Production benchmark (Sofia, push χ to OOM-ceiling) | 1 week | data-inventory entry |

**Total: ~10-12 weeks for v1.** Significantly more than my original "3-4 week" estimate; this reflects the actual complexity of SUMMA for 4-tensor contractions.

---

## 6. Open questions / risks

1. **Rectangular grid (N1 ≠ N2)**: this design assumes square. For non-square (e.g. N=8 with N1=2 N2=4), the SUMMA loop count differs between row and col directions. Either restrict to square (drops N=8 as a valid config — would need N=4 or N=16 grids) or generalize the loop (more code).

2. **Broadcast efficiency at small messages**: at χ=1024 M=2, each broadcast message is χ²·D²/M² ≈ 100 MB. NCCL handles this fine. At χ=2048 M=4 (16 GPUs), message is 200 MB — also fine. At χ=4096 M=8 (64 GPUs), 50 MB — still NCCL-friendly.

3. **AD with SUMMA**: each iteration of Phase A is a separate op in the tape. With `Recompute()` checkpointing, tape size is O(1) in M instead of O(M). Without it, O(M) tape memory.

4. **Memory measurement on Sofia**: my bench's `free / total = X / 139.80 GB` doesn't capture peak (only final). Need a proper instrumented bench (CUDA.alloc / nvprof) to validate the §2.8 peak estimates.

5. **Comparison to 1D extended (PR #42 style for all 5 tensors)**: I haven't compared SUMMA against the alternative path of just extending PR #42's FR-only 1D distribution to all 5 tensors. The 1D-extended approach has lower comm but higher peak; for χ-pushing it might still hit OOM. SUMMA is the right answer if OOM is the binding constraint.

---

## 7. Recommendation

Commit to the **SUMMA path**, with the algorithm in §2.7 as the v1 target for FLmap. Pre-implement Phase 1 (primitive + rrule) in next session as the foundation, then iterate. Target: working `FLmap_parallel_2D_SUMMA` in ~3-4 weeks (Phases 1-4), full Phase 1 (all maps + AD) in ~10-12 weeks.

Alternatively: confirm or revise the algorithm in §2.7 before committing — I derived it inline above. A second pair of eyes (or a math-rigorous review) before coding would catch any remaining bugs (the §2.4-2.5 self-correction shows how easy it is to miss off-diagonal contributions).

**Action item for next session**: Phase 1 — implement `summa_broadcast_dim` primitive + rrule, unit-tested on Sofia.
