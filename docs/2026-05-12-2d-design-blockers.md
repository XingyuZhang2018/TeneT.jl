# 2D Distributed VUMPS — Design Blockers Found at Task 1.7

**Date**: 2026-05-12
**Branch**: `claude/2d-distributed-vumps` at `99b0999` (after Task 1.6)
**Status**: Phase 1 implementation HALTED at Task 1.7

---

## Summary

While preparing to implement Task 1.7 (`alltoall_dim_swap` for square grid), an analyst subagent reviewing Section 2 of `docs/2026-05-11-2d-distributed-vumps-runtime-design.md` discovered a **mathematical bug** in the 4-step dataflow that applies to FLmap, FRmap, and ACmap. Separately, Sofia MPI validation of the existing Task 1.3 `allgather_dim` primitive (commit `396b55d`) revealed an implementation bug in the permutedims-based general-dim path.

Both issues block Phase 1 progress until resolved.

---

## Issue 1: `allgather_dim` MPI run fails (implementation bug)

**Sofia job 1098663** (`zen5_vis` partition, 4-rank MPI, distributed_primitives_test.jl) failed in the very first testset:

```
@testset "Gather along row_comm — last dim (direct path)" begin
    data = fill(Float64(grid.r1 + 1), 8, χ_local)
    full = allgather_dim(data, 2, grid.col_comm)
    @test all(full[:, 1:χ_local] .== 1.0)       # FAIL
    @test all(full[:, χ_local+1:2χ_local] .== 2.0)  # FAIL
end
```

The gathered tensor's per-rank slabs do not land in the expected slot positions. Hypothesis: the buffer layout for `allgatherv_p2p!` with `dim != ndims` doesn't behave as the Task 1.3 implementer assumed even in the "direct path" (no permutedims).

**Action needed**: Re-examine `allgather_dim` semantics. Likely the buffer layout invariant needs a more careful test in isolation; the Phase 0 sandbox `2d_allgather_sanity.jl` used a 3D recv buffer that bypassed this issue, so it didn't catch the bug.

---

## Issue 2: Section 2 dataflow off-diagonal block bug (design flaw)

### The trace (FLmap, square N1=N2=2)

Pre-call distributions:
- `FL[a∈slice_r1, e, f, i∈slice_r2]`
- `ALu[a∈slice_r1, b, c, d∈slice_r2]`
- `ALd[i∈slice_r1, j, k, l∈slice_r2]`
- result needed: `[d∈slice_r1, g, h, l∈slice_r2]`

Section 2.1 Step 1: `AllGather(ALd, dim=i, col_comm)` → ALd.i = full ✓

Section 2.1 Step 2: local einsum at rank (r1, r2):
- Uses FL[a=slice_r1, e, f, i=slice_r2] (FL.i still distributed on N2)
- Uses ALd_gathered[i=full, j, k, l=slice_r2]
- Uses ALu[a=slice_r1, b, c, d=slice_r2]
- Sum over i: FL has i∈slice_r2, ALd has i∈full → product nonzero only for i ∈ slice_r2
- **Bug**: `i` sum is restricted to slice_r2, missing contributions from i ∈ slice_r2' (r2' ≠ r2)

Section 2.1 Step 4: `AllToAll(partial_full_a, d:N2→N1, row_comm[r1])`
- Wants result[d=slice_r1, g, h, l=slice_r2]
- Across row_comm[r1] (varying r2'), available (d, l) blocks are diagonal: {(r2', r2') for r2' ∈ row_comm}
- Required (d=slice_r1, l=slice_r2) for r2 ≠ r1: NOT computed by any rank in row_comm[r1]
- **Bug**: Off-diagonal (d, l) blocks are absent from the entire grid

### Why this is structural, not a typo

Each rank's einsum couples `d` (from ALu.d, on N2) and `l` (from ALd.l, on N2) into the SAME row_comm member (same r2). Free-leg blocks (d=k, l=j) for k ≠ j cannot be computed by a 4-step "gather + einsum + reduce + transpose" pattern because:
- ALu has d on N2 only at the local r2
- ALd has l on N2 only at the local r2
- Therefore the combined (d, l) tensor at any rank has d=l=r2 (diagonal)

The same issue applies to:
- FRmap: (a, i) plane has the symmetric off-diagonal bug
- ACmap: (a, d) plane has the symmetric off-diagonal bug

### The correct algorithm requires SUMMA-style multi-step broadcasts

For square grid N=N1=N2=M, FLmap result on rank (r1, r2) requires:
```
result[d=slice_r1, g, h, l=slice_r2]
  = sum_{a, i, b, c, e, f, j, k, p}
      FL[a, e, f, i] × ALd[i, j, k, l=slice_r2]
      × M1 × M2 × ALu[a, b, c, d=slice_r1]
```

The "right" algorithm is SUMMA-like:
```
For t = 0..M-1:
  - Broadcast FL[a=slice_r1, ..., i=slice_t] in row_comm[r1] (from rank (r1, t))
  - Broadcast ALd[i=slice_t, ..., l=slice_r2] in col_comm[r2] (from rank (t, r2))
  - Broadcast ALu[a=slice_r1, ..., d=slice_r1] in row_comm[r1] (one-time, diagonal source)
  - partial += FL × ALd × M × ALu_diag (local einsum over a∈slice_r1, i∈slice_t)
```

Communication per FLmap call per rank: ≈ 2·χ²·D²/M (M iterations × 2 broadcasts/iter × χ²·D²/M² per broadcast). For χ=768, D=16, M=4: ≈ 2 × 1.2/4 = 0.6 GB per map call.

This is significantly more than the design's claimed ~300 MB per map, and requires fundamentally different code structure (not 4 sequential primitives, but a SUMMA loop).

---

## Path forward — three options

### Option A: Pause + redesign with SUMMA, ~3-4 weeks added

- Rewrite Section 2 of design doc with SUMMA-style algorithms for all 3 maps + Cmap
- Add a new primitive: `summa_step` or similar (or break into existing broadcast primitives)
- Re-do the rrule design (SUMMA backward is more involved)
- Re-do plan Tasks 1.7-1.18 timeline

Pro: Correct algorithm, full 2D memory benefit
Con: Significant delay; SUMMA AD is research-grade in places

### Option B: Switch v1 to 1D-only, defer 2D to v2

- Discard the (N1, N2) 2D grid for v1; use only 1D (N1×1 or 1×N2)
- 1D = effectively PR #42's distribution applied to all 5 tensors (FL/FR/AL/AR/AC)
- Memory savings: 1/N persistent per tensor (per the original design)
- Peak savings: limited (only 1/N of each tensor in worst case, not 1/√N)
- No off-diagonal bug since 1D distributions don't couple (a, i)/(d, l)

Pro: Quick path to a working distributed runtime; gradient parity verifiable
Con: Lower peak-memory benefit than 2D; need to redo the design doc Section 2 anyway

### Option C: Restrict 2D to "compatible" tensor distributions (research)

- Re-derive Section 2 with all χ-legs on the SAME 1D comm per tensor (so each tensor's 2 χ-legs are both on N1 or both on N2)
- This avoids the (a, i)/(d, l) coupling bug
- But each tensor is now effectively 1D, like Option B

Same trade-off as B; no extra benefit.

---

## Status of work completed so far (preserved)

**Code on branch `claude/2d-distributed-vumps`** (HEAD `99b0999`):

| Task | Commit | Status |
|------|--------|--------|
| Pre-flight: worktree | `9c3d956` | done |
| Phase 0 sandbox 0.1-0.4 | `bd517a2`..`d3402f1` | done, Sofia validated |
| Phase 1 Task 1.1: Cart2DGrid | `65eba73`, `b3803db` | done, Sofia validated |
| Phase 1 Task 1.2: VUMPS alg fields | `6a1accb` | done (code review approved) |
| Phase 1 Task 1.3: allgather_dim forward | `396b55d` | **FAILS on Sofia** — needs fix (Issue 1) |
| Phase 1 Task 1.4+1.5: rrules | `72cf523`, `e370eb9` | code approved but depends on 1.3 fix |
| Phase 1 Task 1.6: allreduce_dim | `99b0999` | code approved, untested on Sofia |

**Documents**:
- `docs/2026-05-11-2d-distributed-vumps-runtime-design.md`: **Section 2 contains the design flaw**. Section 1 (partition convention) and Section 3 (primitives) are largely OK; Section 2 needs full rewrite.
- `docs/2026-05-11-2d-distributed-vumps-runtime-plan.md`: Tasks 1.7+ blocked.
- `docs/2026-05-12-2d-distributed-phase0-report.md`: still valid.

---

## Recommendation

**Pause this session.** The user needs to decide:

1. Is the 2D scaling target (push χ by √N) critical enough to invest 3-4 more weeks in SUMMA redesign?
2. Is 1D distribution (PR #42 style applied to all 5 tensors) sufficient for the OOM problem at hand?
3. Is there appetite for a hybrid (1D for v1, 2D for v2)?

The user has already invested significant time. A clear directional pivot now is far better than continuing to build on a broken Section 2.
