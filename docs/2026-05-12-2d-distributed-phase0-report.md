# 2D Distributed VUMPS — Phase 0 Completion Report

**Date**: 2026-05-12
**Branch**: `claude/2d-distributed-vumps` at `d3402f1`
**Worktree**: `.claude/worktrees/2d-distributed-vumps/`
**Predecessors**: design `docs/2026-05-11-2d-distributed-vumps-runtime-design.md`, plan `docs/2026-05-11-2d-distributed-vumps-runtime-plan.md`

---

## What's done

| Task | Commit | Status |
|------|--------|--------|
| Pre-flight: worktree + branch | `9c3d956` | done |
| 0.1 grid sanity | `bd517a2` | sandbox PASS on Sofia |
| 0.2 allgather sanity | `83452ab` → fixed in `d3402f1` | sandbox PASS on Sofia |
| 0.3 alltoall N1=N2=2 (Sendrecv) | `232b19d` → fixed in `d3402f1` | sandbox PASS on Sofia |
| 0.4 Zygote rrule mock | `1ec262d` → fixed in `d3402f1` | sandbox PASS on Sofia |
| Driver `run_phase0_sanity.sh` | `6ce62f9` → fixed in `d3402f1` | runs all 4 tests |

**Sofia validation**: sbatch job `1098132` on partition `zen5_himem` (node `hm009`, 4 CPU tasks, no GPU), 18 seconds wall, exit 0. All 4 sandboxes PASS.

Re-run command for next session:

```bash
ssh -l vsc48503 sofia.hpc.vub.be
cd /sofia/scratch/pilot/pilot_2026_0002/xz/TeneT-2d-validation
git pull --ff-only origin claude/2d-distributed-vumps
sbatch sandbox/submit_phase0.slurm
```

---

## Bugs caught by Phase 0 (Phase 1 design implications)

### Bug 1: Julia top-level `for` soft-scope (Tasks 0.2, 0.3)

`all_ok = true` declared at top-level inside `if rank == 0` block, then `all_ok &= ...` inside a `for` body — Julia's soft-scope rule re-declared `all_ok` as a new local inside the `for`, making the `&=` read UndefVar at runtime. **Parse-checked clean; ran only failed at runtime.**

**Phase 1 implication**: every test driver and sandbox script with stateful loop variables must wrap them in `let ... end` or in a function. Pure top-level for-loops over global accumulators are broken in Julia ≥1.6 strict mode. Add this to the test-writing convention.

### Bug 2: Zygote complex gradient convention (Task 0.4)

For `f: ℂⁿ → ℝ, f(z) = ∑ |z|²`, Zygote returns `∂f/∂(conj z) = 2z`, **NOT** `2·conj(z)`. My initial sandbox test expected the wrong sign.

**Phase 1 implication (critical)**: the four 2D primitive rrules (`allgather_dim`, `reduce_scatter_dim`, `allreduce_dim`, `alltoall_dim_swap`) are linear operations — their backward passes are themselves linear primitives (the adjoints), and the upstream `d_y` is already in Zygote's convention. **Do NOT add manual `conj()` anywhere in the rrules.** The Wirtinger gradient flows through naturally. The plan's Section 3 rrule sketches already show this pattern (`back(d_result) = reduce_scatter_dim(unthunk(d_result), ...)` etc.) — keep it that way.

### Bug 3: UCX env var noise

Sofia's `OpenMPI/5.0.7-GCC-14.2.0` module sets `UCX_MEMTYPE_CACHE` and `UCX_ERROR_SIGNALS` which UCX 1.18 doesn't use, producing 4 warnings per MPI rank per Julia process. Pure cosmetic.

**Phase 1 implication**: every sbatch driver should set `export UCX_WARN_UNUSED_ENV_VARS=n`. Already in `sandbox/run_phase0_sanity.sh`; replicate in Phase 1 `test/run_*_test.sh` drivers as we write them.

---

## Phase 0 sandbox design notes (carry into Phase 1)

From the subagent reports:

1. **`alltoall_dim_swap` for rectangular grid (N1 ≠ N2)** is structurally different from the square case. Task 0.3's Sendrecv pattern works only because `(r1, r2) ↔ (r2, r1)` is a bijection on a square grid. For rectangular, each rank pairs with `gcd(N1, N2)` partners and `MPI.Alltoallv!` with non-trivial counts/displacements is needed. **Plan Task 1.7 (square case) is fine; Task 1.8 (rectangular) is genuinely harder — preserve the 1-week budget.**

2. **Send vs recv buffer shape**: for N1 ≠ N2, send-shape has extent `χ/N_src` on the swap dim while recv-shape has `χ/N_dst`. Cannot use `similar(local_tensor)` — must allocate recv buffer with the post-swap shape explicitly. Add to Phase 1 Task 1.7 implementation checklist.

3. **Non-swap axis distribution**: Task 0.3 had the j-axis fully replicated for simplicity. In production FLmap/FRmap, the "other" χ axis is itself distributed on the orthogonal grid axis. The `alltoall_dim_swap` primitive must preserve that distribution unchanged. Add a Phase 1 unit test covering this case.

---

## Resumption checkpoint for next session

**Next task**: Phase 1 Task 1.1 — `Cart2DGrid` struct + alg field (plan reference: `docs/2026-05-11-2d-distributed-vumps-runtime-plan.md` line ~120).

**Worktree state**: `claude/2d-distributed-vumps` at `d3402f1`, clean, in sync with origin.

**Files to touch (Task 1.1)**:
- Create `src/contraction/cart2d_grid.jl`
- Modify `src/boundary_algorithm/interface.jl` (VUMPS alg fields)
- Modify `src/TeneT.jl` (include new file)
- Create `test/cart2d_grid_test.jl`

**Sofia env (Phase 1 sets up here)**:
- Worktree on Sofia: `/sofia/scratch/pilot/pilot_2026_0002/xz/TeneT-2d-validation/`
- Julia: `/sofia/scratch/pilot/pilot_2026_0002/xz/julia-1.11.3`
- Depot: `/sofia/scratch/pilot/pilot_2026_0002/xz/.julia` (already populated, ~50s compile last run)
- Account flag for sbatch: `-A pilot_2026_0002`
- Recommended partition for CPU-only tests: `zen5_himem` (16 idle) or `zen5_vis` (4 idle)

**Don't forget when starting Phase 1**:
- Set `UCX_WARN_UNUSED_ENV_VARS=n` in driver
- Wrap stateful loop accumulators in `let` blocks
- Pass complex gradients through rrules unchanged (no `conj()`)
- Reserve full week budget for `alltoall_dim_swap` rectangular case (Task 1.8)
