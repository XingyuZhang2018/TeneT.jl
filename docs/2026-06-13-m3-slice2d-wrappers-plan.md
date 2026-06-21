# M3: Slice2D distributed map wrappers (Cmap / FRmap / ACmap / ACdmap)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to
> implement this plan task-by-task, with a double review (spec compliance +
> code quality) after every task.

**Goal:** Four new distributed cross-/co-axis Slice2D maps —
`Cmap_slice2d`, `FRmap_slice2d_dist`, `ACmap_slice2d_dist`,
`ACdmap_slice2d_dist` — each with a forward, a Zygote-free `rrule`, and a
4-rank (2×2) MPI distributed-parity gate (forward rtol 1e-12, gradient rtol
1e-10) vs the serial kernel. M3 is **additive**: `FLmap_slice2d_dist`,
`FLmap_slice2d`, the serial `basic.jl` kernels, and the M2 chain engine stay
untouched and green. Local einsums use the **chain engine** (M2 payoff), not
new hand kernels; the only hand-written code is the comm wrapper, built
entirely from the existing `slice2d.jl` primitives and their documented
adjoints. **No new comm primitive lands in M3** (design doc §2.6/§5/risk 4 —
every cross-axis move fuses into an existing gather + reduce-scatter pair).

**Architecture:** The maps follow two structural classes, both grounded in
`FLmap_slice2d_dist` (`src/contraction/slice2d.jl:516`, rrule
`src/autodiff/rules.jl:589-698`):

- **Replicated class — Cmap.** C stays replicated; only FL/FR are distributed.
  Allgather the FL/FR blocks to full, one local `chain_apply(CMAP_LEG*_CHAIN)`,
  returns the FULL χ×χ tensor (no block, identical on every rank). No ring, no
  chunk, no square-grid assertion. Simplest map — **implemented FIRST** as the
  end-to-end harness warm-up.

- **Cross-axis (gather) class — FRmap, ACmap, ACdmap.** Each has **two**
  cross-axis legs (one contracted, one output) that no single ring can serve
  simultaneously; the design doc (`docs/2026-06-13-acmap-slice2d-dataflow-design.md`)
  gathers the cross-axis legs to full before a single local `chain_apply`, then
  fuses the output redistribution into an existing reduce-scatter. **FRmap and
  ACdmap are the same structural shape** (their pinned chain intermediates carry
  full `i` × full `d` — verified identical label-sets via `tensor_chain`): both
  need the 2-level memory chunk over the contracted leg (accumulate) and the
  output leg (assign). FRmap is **ACdmap with the `i`↔`d` roles swapped** (see
  Batch B). ACmap is the simpler one-chunk variant (its intermediates carry the
  local `a`-block + chunked `l`; the output `i`/`d` never appear because FL is
  the chain's last operand). All three use the chain engine's whole-chain API
  (`chain_apply` / `chain_backward`) over `FRMAP_LEG5_CHAIN` /
  `ACMAP_LEG5_CHAIN` / `ACDMAP_LEG5_CHAIN`, and all three **require a square
  grid** (`@assert grid.N1 == grid.N2`).

**Tech Stack:** Julia 1.11, MPI.jl (4-rank CPU tests on this Windows machine;
4-GPU Sofia validation in Batch E), Zygote (reference + outer driver only —
NEVER inside any rrule backward; slice2d-OOM lesson), CUDA.jl (`_free!`),
TensorOperations 5.5.1 runtime API via the chain engine. Existing
`slice2d.jl` comm primitives + `chain_maps.jl` chains as the building
blocks.

**Branch:** `claude/ecstatic-golick-e3f6f0` (this worktree; M1+M2 landed here,
`CHAIN_ENGINE[] == true` by default).

---

## Context for implementers

### The chain-engine-vs-hand-kernel decision (load-bearing)

M3 writes **NO new `_slice2d_<map>_*` hand kernels.** The M2 chain engine is
the local einsum AND its adjoint, and **every map uses the whole-chain API** —
`chain_apply` / `chain_backward`
(`src/contraction/chain_engine.jl:121,231`) over a single fully-gathered
working set (`FRMAP_LEG5_CHAIN` / `ACMAP_LEG5_CHAIN` / `ACDMAP_LEG5_CHAIN` /
`CMAP_LEG*_CHAIN`). **No map uses the slice2d-ring API** (`chain_link1_add!` /
`chain_apply_from1` / `chain_backward_from1` / `chain_link1_back`): the earlier
"FRmap is a ring-class FLmap mirror" design was **wrong** — FRmap has two
cross-axis legs (one contracted `d`, one output `i`), so the ring cannot serve
both; FRmap is a **gather-class** map, structurally identical to ACdmap (Batch
B). The ring APIs and the M1 *"chain engine reproduces the slice2d local
pipeline"* testset are not part of M3 at all.

The ONLY hand-written code is the comm wrapper, and every comm op already
exists in `slice2d.jl` with a documented adjoint — **no new comm primitive
lands in M3** (design doc §2.6, §5, risk 4). The four existing primitives M3
needs are: `_slice2d_col_allgather` (730), `_slice2d_row_allgather` (750),
`_slice2d_col_reduce_scatter` (710), `_slice2d_row_reduce_scatter_last` (760),
each with its documented adjoint partner.

### Key file:line references

| What | Where |
|------|-------|
| `FLmap_slice2d_dist` forward (signature template) | `src/contraction/slice2d.jl:516-535` |
| `_slice2d_forward_sliced` (gather + chunk core) | `src/contraction/slice2d.jl:417-464` |
| `rrule(FLmap_slice2d_dist)` (rrule STRUCTURAL template) | `src/autodiff/rules.jl:589-698` |
| do_cast / densify / NoTangent / eager `_free!` boundary | `src/autodiff/rules.jl:591-607,614-623,690-695` |
| `Slice2DGrid`, `slice2d_grid`, `slice2d_scatter`/`gather` | `src/contraction/slice2d.jl:16-50,164-203` |
| `slice2d_gather` rrule (take-my-block adjoint — Cmap template) | `src/autodiff/rules.jl:450-461` |
| `slice2d_scatter` rrule (allreduce-stitch adjoint) | `src/autodiff/rules.jl:~430-447` |
| `_slice2d_col_reduce_scatter` (first leg, tag 710) | `src/contraction/slice2d.jl:242` |
| `_slice2d_col_allgather` (first leg, tag 730, adjoint of 710) | `src/contraction/slice2d.jl:275` |
| `_slice2d_row_allgather` (last leg, tag 750) | `src/contraction/slice2d.jl:339` |
| `_slice2d_row_reduce_scatter_last` (last leg, tag 760, adjoint of 750) | `src/contraction/slice2d.jl:375` |
| `_free!` (CuArray unsafe_free) | `src/contraction/slice2d.jl:71-72` |
| chain engine whole-chain API (`chain_apply` / `chain_backward`) | `src/contraction/chain_engine.jl:121,231` |
| `FRMAP_LEG5_CHAIN`, `ACMAP_LEG5_CHAIN`, `ACDMAP_LEG5_CHAIN`, `CMAP_LEG3/4_CHAIN` | `src/contraction/chain_maps.jl:71,106,139,169,170` |
| `engine_backward` grad-slot permutations (FR/AC/ACd) | `src/contraction/chain_maps.jl:75-97,110-132,142-160` |
| FRMAP/ACDMAP pinned inters (full `i`×full `d`, identical) | `tensor_chain` → `.inters` (verified: `(i,k,d,h,j,g),(i,d,e,b,k,h,p),(i,e,f,d,b,c)` ≡ ACDMAP) |
| serial `FRmap` leg5 (`(FR,ARu,ARd,M1,M2)`) | `src/contraction/basic.jl:170` |
| serial `Cmap` leg3/leg4 (`(C,FL,FR)`) | `src/contraction/basic.jl:336,341` |
| serial `ACmap` leg5 (`(AC,FL,FR,M1,M2)`) | `src/contraction/basic.jl:283` |
| serial `ACdmap` leg5 (`(ACd,FL,FR,M1,M2)`) | `src/contraction/basic.jl:370` |
| `split_ranges(N, n)` | `src/contraction/forloop_parallel_MPI.jl:19` |
| `_TAG_BASE = 1000`, `allreduce_p2p!`, `allgatherv_p2p!` | `src/contraction/forloop_parallel_MPI.jl:35`, `nccl_wrapper.jl` |
| 4-rank parity harness (CLONE this) | `test/test_slice2d.jl`, launcher `test/run_test_slice2d.jl` |
| Sofia 4-GPU slice2d driver + submit (Batch E template) | `examples/MPI_parallel/test_slice2d_sofia.jl`, `examples/MPI_parallel/Sofia/submit_test_slice2d.sh` |
| Design doc (AUTHORITATIVE for AC/ACdmap) | `docs/2026-06-13-acmap-slice2d-dataflow-design.md` |

### The FRmap gather-class decomposition (a decision this plan makes — read before Batch B)

> **REVISION (2026-06-14):** the earlier draft classified FRmap as a ring-class
> FLmap mirror. **That was wrong** — verified against source by two reviewers.
> FRmap has TWO cross-axis legs (contracted `d`, output `i`), so no single ring
> can serve both. FRmap is a **gather-class** map, structurally **identical to
> ACdmap** with the `i`↔`d` roles swapped. The corrected scheme below was
> verified to rel err 3.7e-16 on a 2×2 grid.

The serial kernels, side by side
(`basic.jl:170` FRmap, `basic.jl` FLmap):

```
FLmap:  result[d,g,h,l] := FL[a,e,f,i] ALd[i,j,k,l] M1[e,j,g,b,p] M2[f,k,h,c,p] ALu[a,b,c,d]
FRmap:  result[a,e,f,i] := ARd[i,j,k,l] FR[d,g,h,l] M1[e,j,g,b,p] M2[f,k,h,c,p] ARu[a,b,c,d]
ACdmap: result[a,b,c,d] := ACd[i,j,k,l] FR[d,g,h,l] M1[e,j,g,b,p] M2[f,k,h,c,p] FL[a,e,f,i]
```

Slice2D convention: first χ leg by r1, last χ leg by r2. Per-leg census for
FRmap (ARd `[i,j,k,l]`, FR `[d,g,h,l]`, ARu `[a,b,c,d]`, result `[a,e,f,i]`):

| leg | who | distribution | role |
|-----|-----|--------------|------|
| `a` | ARu.1=r1, result.1=r1 | co-distributed N1 | **output** (local on r1) |
| `l` | ARd.4=r2, FR.4=r2 | aligned N2 | contracted (local) |
| `d` | FR.1=r1, ARu.4=r2 | **cross-axis** | **contracted** → gather to full |
| `i` | ARd.1=r1, result.4=r2 | **cross-axis** | **output** → row reduce-scatter on last leg |

Two cross-axis legs (`d` contracted, `i` output) — exactly the gather-class
signature. The diagonal trap (the off-diagonal `(a,i)`-plane blocks are never
computed by a naive transpose) is defeated by making the cross-axis legs full
before the local chain. `FRMAP_LEG5_CHAIN` ops order is `(ARd, FR, M1, M2, ARu)`
(`chain_maps.jl:71`), out `(:a,:e,:f,:i)`; link 1 contracts `ARd ⋆ FR` over the
**aligned `l`** (NOT `d`), so its pinned intermediates carry full `i` × full `d`
(verified `(i,k,d,h,j,g),(i,d,e,b,k,h,p),(i,e,f,d,b,c)` — **identical label-sets
to `ACDMAP_LEG5_CHAIN`**). FRmap IS structurally ACdmap.

**FRmap ↔ ACdmap structural correspondence (clone Batch D's ACdmap, swap i↔d):**

```
FRmap.d (cross-axis CONTRACTED)  ==  ACdmap.i (cross-axis contracted)
FRmap.i (cross-axis OUTPUT)      ==  ACdmap.d (cross-axis output)
FRmap.a (co-dist OUTPUT, local)  ==  ACdmap.a (aligned output, local)
FRmap.l (aligned CONTRACTED)     ==  ACdmap.l (aligned contracted)
```

So the FRmap gather scheme is ACdmap's F1–F5 / B1–B5 with `i`↔`d` swapped. The
forward (on rank `(r1,r2)`, inputs block-stored, `M1,M2` replicated):

```
F0. χ = MPI.Allreduce(size(ARd_blk,1), +, col_comm);  p_rs = split_ranges(χ, N)
F1. ARd_g = _slice2d_col_allgather(ARd_blk, grid, p_rs)   # full i   (tag 730)  [full i, l-block]
F2. ARu_g = _slice2d_row_allgather(ARu_blk, grid, p_rs)   # full d   (tag 750)  [a-block, full d]
F3. FR_g  = _slice2d_col_allgather(FR_blk,  grid, p_rs)   # full d   (tag 730)  [full d, l-block]
F4. 2-level memory chunk over d (CONTRACTED → ACCUMULATE Σ_d) and i (OUTPUT →
    ASSIGN disjoint i-slice); local l-block summed WHOLE inside each chain_apply:
      for ich in i_chunks                      # disjoint output i-slices → ASSIGN
        acc = nothing
        for dch in d_chunks                    # summed contracted d-slices → ACCUMULATE Σ_d
          piece = chain_apply(FRMAP_LEG5_CHAIN,
              (ARd_g[ich,:,:,:], FR_g[dch,:,:,:], M1, M2, ARu_g[:,:,:,dch]))
          acc === nothing ? (acc = piece) : (acc .+= piece; _free!(piece))
        end
        view(partial, :, :, :, ich) .= acc; _free!(acc)   # partial[a-block, e, f, i∈1:χ]
F5. result_blk = _slice2d_row_reduce_scatter_last(partial, grid, p_rs)  # sum l over row, keep i-block r2 (tag 760)
return result_blk   # [a∈p_rs[r1+1], e, f, i∈p_rs[r2+1]]  — same convention as input
```

Note the cross-axis roles vs ACdmap: ACdmap contracts `i` and outputs `d`;
FRmap contracts `d` and outputs `i`. So **F1/F3 col_allgather** make the *first*
legs full (`ARd.i`, `FR.d`), **F2 row_allgather** makes ARu's *last* leg `d`
full, the **inner accumulate loop chunks the contracted `d`** (slicing `FR_g`
first leg and `ARu_g` last leg), the **outer assign loop chunks the output `i`**
(slicing `ARd_g` first leg), and **F5 row_reduce_scatter_last** lands the output
`i` (partial's last leg) on r2. (Compare ACdmap F5, which lands `d` on r2.) The
co-distributed output `a` is ARu/ARd's a-block (kept local on r1); the aligned
contracted `l` is summed whole inside each `chain_apply`, F5's `Σ_{r2}` over the
row completing `Σ_l`.

**Square grid REQUIRED** (`@assert grid.N1 == grid.N2`), the same as ACdmap:
the single `p_rs = split_ranges(χ, N)` is licensed only by `N1==N2`, when the
r1- and r2-partitions coincide. The rectangular `(1,4)`/`(4,1)` cases are M3 v2
(they need a real block transpose) and are **dropped** from FRmap's tests.

### Square-grid assumption (all gather-class maps: FRmap / ACmap / ACdmap)

`FRmap_slice2d_dist` / `ACmap_slice2d_dist` / `ACdmap_slice2d_dist` **assume
`N1 == N2`** (design doc §1, risk 2/6). The `@assert grid.N1 == grid.N2` at
entry licenses passing one `p_rs = split_ranges(χ, N)` to every primitive (the
r1- and r2-partitions coincide). **Only Cmap does NOT need the square
assertion** — Cmap allgathers full FL/FR so no cross-axis partition matters.
State this in each map's docstring.

### Run commands (Windows; ALWAYS use the Bash tool — PowerShell mangles quotes)

```bash
cd "/d/1 - research/1.26 - iPEPS_opt/TeneT.jl/.claude/worktrees/ecstatic-golick-e3f6f0"
julia --project=. test/run_test_slice2d.jl        # existing 4-rank FLmap suite (must STAY green)
julia --project=. test/run_test_slice2d_m3.jl      # NEW — 4-rank M3 maps suite (this plan)
julia --project=. test/test_chain_engine.jl       # M1/M2 engine suite (regression)
julia --project=. test/test_chain_maps.jl         # M2 map-chain suite (regression)
```

`Manifest.toml` is present in this worktree (copied from main during research —
do not delete). Tests are CPU; `CUDA.functional()` is false here. 4-rank MPI
runs green on this machine (proven repeatedly in M2). **Known transient:** an
occasional `libLLVM` / `0xc0000005` startup crash on `mpiexec` — retry the run
once before treating a failure as real.

### Test-file organization (a decision this plan makes)

Add the four new maps' parity tests to a **new** `test/test_slice2d_m3.jl` with
its own launcher `test/run_test_slice2d_m3.jl` (clone of
`test/run_test_slice2d.jl`, pointing at the new test file). Rationale: keeps the
existing `test_slice2d.jl` (FLmap, the M3-untouched gate) byte-stable so a green
`run_test_slice2d.jl` stays a clean regression signal, while the new file grows
batch by batch. Both launchers run under `mpiexec -n 4`; the new test file
clones `test_slice2d.jl`'s header verbatim (`MPI.Init()`, `comm`/`rank` consts,
`@assert Comm_size == 4`, no explicit `Finalize`).

**Per-batch imports (NIT — avoid importing undefined symbols).** A failing
batch must not `using TeneT: …` a symbol that does not exist yet (it errors at
load before any test runs). Each batch adds ONLY its own map symbols to the
`using TeneT:` line as it lands: Batch A creates the file with `Cmap,
Cmap_slice2d`; Batch B appends `FRmap, FRmap_slice2d_dist`; Batch C appends
`ACmap, ACmap_slice2d_dist`; Batch D appends `ACdmap, ACdmap_slice2d_dist`. The
shared helpers (`slice2d_grid`, `Slice2DGrid`, `slice2d_scatter`, `slice2d_gather`,
`split_ranges`) and the header are present from Batch A onward.

### Invariants (every reviewer checks every task)

- **No Zygote inside any `rrule` backward.** Recompute-style + eager `_free!`,
  exactly the `FLmap_slice2d_dist` rrule discipline. Zygote appears only as the
  test parity reference and the outer driver that calls the rrule.
- **Old paths untouched and green.** `basic.jl` serial kernels,
  `FLmap_slice2d`, `FLmap_slice2d_dist`, the chain engine, `test_slice2d.jl`.
- **Single-M convention = FLmap precedent** (`rules.jl:592,572,690`):
  `is_tuple = M isa Tuple; M1, M2 = is_tuple ? M : (M, conj(M))`; run the **2M
  tuple chain** throughout; compose `dM = is_tuple ? (dM1, dM2) : dM1 .+
  conj(dM2)` at the boundary. NEVER route the rrule through the `_1M`
  conj_variant chains.
- **do_cast boundary cast** = `inner_etype !== nothing && inner_etype !=
  real(eltype(...))`; `_downcast_eltype` at entry, `T_orig.(…)` at exit —
  byte-for-byte the FLmap_slice2d_dist block.
- **Densify non-`DenseArray` cotangents** before any `MPI.Isend` (FillArrays
  guard, `rules.jl:614-623`).
- **Rank-uniform control flow** — every rank runs the identical collective
  sequence (else deadlock).
- **rrule return tuple** mirrors the map arg order, padded with `NoTangent()`
  for the function-self slot and the `grid` slot.
- **NO new comm primitive.** M3 uses ONLY the four existing primitives
  (`_slice2d_col_allgather` 730, `_slice2d_row_allgather` 750,
  `_slice2d_col_reduce_scatter` 710, `_slice2d_row_reduce_scatter_last` 760) and
  their documented adjoint partners. No `_slice2d_col_shift`, no col-twin
  reduce-scatter, no `_slice2d_block_transpose` — all proven unnecessary for v1
  (design doc §2.6/§5/risk 4). No new `_slice2d_<map>_*` hand kernel either.
- **Include-order / late-bind note.** `slice2d.jl` is included BEFORE
  `chain_engine.jl` / `chain_maps.jl`, so the chain consts
  (`FRMAP_LEG5_CHAIN` etc.) are not yet defined when the `*_slice2d_dist`
  functions are parsed. This is fine: Julia resolves global names at CALL time,
  not parse time, so the forwards (which only *reference* the consts inside the
  function body) work as long as the consts are defined before the first call —
  which they always are (module fully loaded before any test runs). No reorder
  needed.
- The 4-rank parity gate (1e-12 fwd / 1e-10 grad) must pass BEFORE a batch is
  done; `run_test_slice2d.jl` must stay green throughout.

---

## Batch A — Cmap_slice2d (3 tasks)

The replicated class. FL/FR distributed (block-stored), C replicated; output is
the FULL χ×χ tensor (C stays replicated per the design). No ring, no chunk, no
output redistribution, no square-grid assertion. **Genuinely the simplest map**
— implemented FIRST as the end-to-end harness warm-up: once Batch A is green,
`test/test_slice2d_m3.jl` + `test/run_test_slice2d_m3.jl` are validated end to
end and every later batch only appends testsets.

### Task A1: Cmap_slice2d forward (allgather FL/FR → chain_apply, C replicated)

**Files:** modify `src/contraction/slice2d.jl`; create
`test/test_slice2d_m3.jl` + `test/run_test_slice2d_m3.jl`.

**Step 1 — failing forward test.** Create `test/test_slice2d_m3.jl` with the
`test_slice2d.jl` header, then a first testset (CPU `Array`, 4 ranks). Import
ONLY Batch A's map symbols (later batches append their own — NIT):

```julia
# M3 Slice2D distributed maps (Cmap/FR/AC/ACd) — distributed parity vs serial.
# Run via: julia --project=. test/run_test_slice2d_m3.jl  (mpiexec -n 4)
# CPU Arrays only; GPU is examples/MPI_parallel/test_slice2d_m3_sofia.jl (Batch E).
using Test, MPI, LinearAlgebra, Random, Zygote
using TeneT
using TeneT: slice2d_grid, Slice2DGrid, slice2d_scatter, slice2d_gather,
             split_ranges, Cmap, Cmap_slice2d
# Batch B appends: FRmap, FRmap_slice2d_dist
# Batch C appends: ACmap, ACmap_slice2d_dist
# Batch D appends: ACdmap, ACdmap_slice2d_dist

MPI.Init()
const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
@assert MPI.Comm_size(comm) == 4 "test_slice2d_m3.jl expects exactly 4 ranks"

function make_leg5(χ, D; d=2, seed=42)
    Random.seed!(seed)
    A   = rand(ComplexF64, χ, D, D, χ)   # AC / ACd / FR role tensor
    Bu  = rand(ComplexF64, χ, D, D, χ)   # ARu / FL role
    Bd  = rand(ComplexF64, χ, D, D, χ)   # ARd / FR role
    M1  = rand(ComplexF64, D, D, D, D, d)
    M2  = rand(ComplexF64, D, D, D, D, d)
    W   = rand(ComplexF64, χ, D, D, χ)
    return A, Bu, Bd, M1, M2, W
end

@testset "Cmap_slice2d forward parity" begin
    for (N1, N2) in ((2, 2), (1, 4), (4, 1)), χ in (16, 18)
        D = 3
        Random.seed!(2400 + χ + 10N1)
        FL = rand(ComplexF64, χ, D, D, χ)
        FR = rand(ComplexF64, χ, D, D, χ)
        C  = rand(ComplexF64, χ, χ)
        g = slice2d_grid(N1, N2)
        ref = Cmap(C, FL, FR)                 # leg4 Cmap: result[e,f] := FL[a,c,d,e] C[a,b] FR[b,c,d,f]
        out = Cmap_slice2d(C, slice2d_scatter(FL, g), slice2d_scatter(FR, g), g)
        @test size(out) == size(ref)          # FULL χ×χ, replicated (NOT a block)
        @test out ≈ ref rtol = 1e-12
    end
end

println("rank $rank: test_slice2d_m3.jl batch A done")
```

`test/run_test_slice2d_m3.jl` (clone of `run_test_slice2d.jl`):

```julia
# Launch test_slice2d_m3.jl under 4 MPI ranks. Usage: julia --project=. test/run_test_slice2d_m3.jl
using MPI
const julia_exe = first(Base.julia_cmd().exec)
const proj = dirname(Base.active_project())
run(`$(MPI.mpiexec()) -n 4 $julia_exe --project=$proj $(joinpath(@__DIR__, "test_slice2d_m3.jl"))`)
```

**Step 2 — run, expect FAIL** (`Cmap_slice2d` undefined):
`julia --project=. test/run_test_slice2d_m3.jl`

**Step 3 — implement** in `slice2d.jl` (after the FLmap block). Cmap leg4 is
`result[e,f] := FL[a,c,d,e] C[a,b] FR[b,c,d,f]` (`basic.jl:343`), chain
`CMAP_LEG4_CHAIN` ops `(FL, C, FR)` (`chain_maps.jl:170`). FL block is
`[a∈r1, c, d, e∈r2]`; FR block is `[b∈r1, c, d, f∈r2]`. C is `[a,b]`
replicated. To make the contraction local, allgather FL and FR to full on their
distributed legs:

```julia
# ─── Cmap (replicated class — C replicated, FL/FR distributed) ───────────────
# Cmap leg4 result[e,f] := FL[a,c,d,e] C[a,b] FR[b,c,d,f]; leg3 result[d,e] :=
# FL[a,c,d] C[a,b] FR[b,c,e]. C stays REPLICATED (per the M3 design — C is
# tiny χ×χ); only FL/FR are block-stored. Output is the FULL χ×χ tensor,
# identical on every rank (every rank gathers the SAME full FL/FR and runs the
# SAME full chain → identical replicated output, NO allreduce). No ring, no
# output scatter, no square-grid assertion. Local einsum is CMAP_LEG*_CHAIN via
# chain_apply (whole-chain engine API), NOT a hand kernel.
function Cmap_slice2d(C, FL_blk, FR_blk, grid::Slice2DGrid; inner_etype = nothing)
    χ = MPI.Allreduce(size(FL_blk, 1), +, grid.col_comm)   # full a/b extent (r1)
    a_rs = split_ranges(χ, grid.N1)
    e_rs = split_ranges(χ, grid.N2)
    # Make FL/FR fully local: gather the r1 (a/b) leg over col, the r2 (e/f)
    # leg over row. After both gathers each rank holds the IDENTICAL FULL FL and
    # FR, so the local chain produces the complete replicated χ×χ output — NO
    # allreduce needed (design (a) below).
    FL_full = _slice2d_col_allgather(_slice2d_row_allgather(FL_blk, grid, e_rs), grid, a_rs)
    FR_full = _slice2d_col_allgather(_slice2d_row_allgather(FR_blk, grid, e_rs), grid, a_rs)
    chain = ndims(FL_blk) == 3 ? CMAP_LEG3_CHAIN : CMAP_LEG4_CHAIN
    out = chain_apply(chain, (FL_full, C, FR_full))
    return out   # full χ×χ, replicated
end
```

> **Derivation / sub-decision (gather vs allreduce):** Two clean designs —
> (a) **fully gather** FL and FR (col_allgather on r1 leg, row_allgather on r2
> leg) so the local chain produces the complete χ×χ output with **no**
> allreduce (every rank holds identical full FL/FR); or (b) gather only the r2
> (e/f) leg, run the local chain over the local a-block (partial over r1), and
> **allreduce** over col_comm to complete the `Σ_a` / `Σ_b` contraction.
> Design (a) is simpler and matches "returns the FULL χ×χ tensor"; **pick (a)
> and drop the allreduce** — this is also what makes the BLOCKER-3 rrule
> (take-my-block, Task A2) correct: because every rank ran the identical full
> chain on identical full inputs, the forward is a replicated computation, so
> its adjoint is take-my-block (NOT reduce-scatter — see A2). **Decide and
> verify with the A1 parity test** — if (a)'s double-gather peak is acceptable
> at test χ (it is, χ small), keep it; the production memory tradeoff (a vs b)
> is a Batch-E note, not a v1 blocker. Confirm
> `_slice2d_col_allgather`/`_slice2d_row_allgather` compose correctly for a 4-leg
> (leg4) and 3-leg (leg3) tensor — the leg3 Cmap has FL `[a,c,d]` (a=r1, d=r2)
> and FR `[b,c,e]` (b=r1, e=r2).

**Step 4 — run, expect PASS** (m3 + slice2d suites). **Step 5 — commit:**
`feat: Cmap_slice2d forward (gather FL/FR → chain_apply, C replicated)`

### Task A2: Cmap_slice2d rrule (take-my-block FL/FR, dC replicated)

**Files:** modify `src/autodiff/rules.jl`; extend `test/test_slice2d_m3.jl`.

Cmap is differentiated **directly** (not through forloop/parallel) — its rrule
returns block gradients for FL/FR (their inputs were blocks) and a full gradient
for C (replicated).

> **BLOCKER-3 FIX (2026-06-14):** the earlier draft prescribed **reduce-scatter**
> for `dFL`/`dFR`. **That is WRONG** and over-counts by exactly `Px` (fails the
> gradient parity gate). In design (a) every rank gathered the SAME full FL/FR
> and computed the IDENTICAL replicated `out`; the gathers are the forward of an
> allgather of a replicated-into-blocks tensor, so the per-rank pullback adjoint
> is **TAKE-MY-BLOCK** (a `getindex` slice), exactly the `slice2d_gather` rrule
> (`rules.jl:459`: `gather_back = unthunk(dfull)[inds...]`). Reduce-scatter
> would SUM the identical peer cotangents → `P×` too large. `dC` from
> `chain_backward` is already correct and replicated (every rank ran the
> identical chain → identical `dC`; no allreduce, no slice). **This differs from
> `FLmap_slice2d_dist`, where reduce-scatter IS correct** — there the loss depends
> on the DISTRIBUTED output block, so each rank's cotangent is its own block's,
> and the AL-gather adjoints genuinely reduce-scatter. Cmap's output is
> replicated, so take-my-block; FLmap's output is distributed, so reduce-scatter.

**Step 1 — failing test:**

```julia
@testset "Cmap_slice2d gradient parity" begin
    for (N1, N2) in ((2, 2), (1, 4))
        χ, D = 16, 3
        Random.seed!(2500 + 10N1)
        FL = rand(ComplexF64, χ, D, D, χ); FR = rand(ComplexF64, χ, D, D, χ); C = rand(ComplexF64, χ, χ)
        W  = rand(ComplexF64, χ, χ)
        g = slice2d_grid(N1, N2)
        FLb = slice2d_scatter(FL, g); FRb = slice2d_scatter(FR, g)
        loss_ref(C,FL,FR)   = real(sum(W .* Cmap(C, FL, FR)))
        loss_dist(C,FLb,FRb) = real(sum(W .* Cmap_slice2d(C, FLb, FRb, g)))
        g_ref  = Zygote.pullback(loss_ref,  C, FL, FR)[2](1.0)
        g_dist = Zygote.pullback(loss_dist, C, FLb, FRb)[2](1.0)
        a_rs = split_ranges(χ, N1); e_rs = split_ranges(χ, N2)
        blkof(x) = x[a_rs[g.r1+1], :, :, e_rs[g.r2+1]]
        @test g_dist[1] ≈ g_ref[1] rtol = 1e-10              # dC full (replicated)
        @test g_dist[2] ≈ blkof(g_ref[2]) rtol = 1e-10       # dFL block (take-my-block)
        @test g_dist[3] ≈ blkof(g_ref[3]) rtol = 1e-10       # dFR block (take-my-block)
    end
end
```

**Step 2 — run, expect FAIL.**

**Step 3 — implement** the rrule in `rules.jl`. Forward design (a) gathers
FL/FR to full and runs the chain; capture `FL_full`, `FR_full`, `C`, the chain,
`a_rs`/`e_rs`, `grid`. Backward:

- **B0:** `dOut = unthunk(dresult)`; densify if not `DenseArray` (FillArrays
  guard, `rules.jl:618-622`) — the chain_backward feeds it locally, but keep the
  guard for the bare-`sum` loss. `dOut` is full χ×χ, identical on every rank
  (replicated forward output); NO allgather (the forward output was already full).
- **B1:** `(dFL_full, dC, dFR_full) = chain_backward(chain, (FL_full, C, FR_full), dOut)`.
  Grads in ops order `(FL, C, FR)`.
- **B2 — dC (replicated, NO stitch):** every rank ran the identical chain on
  identical full inputs and the identical `dOut`, so `dC` is already the same on
  every rank. Return it as-is — **no allreduce, no slice.** (Allreduce would
  multiply it by `P`; that is the same over-count BLOCKER-3 guards against for
  FL/FR.)
- **B3 — dFL/dFR take-my-block:** the gathers (col_allgather ∘ row_allgather)
  are an allgather of a replicated-into-blocks tensor; the downstream
  (`chain_apply` on the full gathered tensor) is a **replicated** computation,
  so the adjoint of "gather my block into the full" is "slice my block out of
  the full cotangent" — exactly the `slice2d_gather` rrule (`rules.jl:455-459`):

  ```julia
  # take-my-block: dFL_full / dFR_full are identical on every rank; slice the
  # block this rank owns (NOT reduce-scatter — that sums identical peers = P×).
  dFL_blk = dFL_full[a_rs[grid.r1 + 1], :, :, e_rs[grid.r2 + 1]]   # leg4
  dFR_blk = dFR_full[a_rs[grid.r1 + 1], :, :, e_rs[grid.r2 + 1]]
  ```

  (leg3: `dFL_full[a_rs[r1+1], :, e_rs[r2+1]]`, `dFR_full[a_rs[r1+1], :,
  e_rs[r2+1]]` — three legs `[a, c, d/e]`.) Free `dFL_full`/`dFR_full` after the
  slice. do_cast upcast at the boundary if used.
- return `(NoTangent(), dC, dFL_blk, dFR_blk, NoTangent())` — map order
  `Cmap_slice2d(C, FL_blk, FR_blk, grid)`.

**Step 4 — run, expect PASS.** **Step 5 — commit:**
`feat: rrule(Cmap_slice2d) (take-my-block FL/FR grads, dC replicated)`

### Task A3: Cmap_slice2d leg3 + replicated-C invariant

**Files:** extend `test/test_slice2d_m3.jl`.

**Step 1 — tests:** leg3 Cmap forward + grad parity (FL `[a,c,d]`, FR
`[b,c,e]`, C `[a,b]`, out `[d,e]`); the leg3 take-my-block slice is the 3-leg
`[a_rs[r1+1], :, e_rs[r2+1]]`. Assert the output is byte-identical across ranks
(replicated) — simplest: compare each rank's `out` to the serial `ref` (already
1e-12 in A1, extend to leg3); optionally an `MPI.Allreduce(norm(out -
out_bcast), MAX)` cross-rank equality check.

**Step 2-4 — run-fail-pass.** **Step 5 — commit:**
`test: Cmap_slice2d leg3 parity + replicated-output invariant`

---

## Batch B — FRmap_slice2d_dist (5 tasks)

The cross-axis (gather) class — **a clone of Batch D's ACdmap with the `i`↔`d`
roles swapped** (FRmap contracts `d` / outputs `i`; ACdmap contracts `i` /
outputs `d`). FRmap's pinned chain intermediates carry full `i` × full `d` —
**verified identical label-sets to `ACDMAP_LEG5_CHAIN`** (`(i,k,d,h,j,g)`,
`(i,d,e,b,k,h,p)`, `(i,e,f,d,b,c)`), so FRmap needs the **same 2-level memory
chunk** (contracted leg → accumulate, output leg → assign) and the **same
`@assert N1 == N2`** square-grid requirement. **NOT a ring, NOT rectangular-safe,
NOT a harness warm-up** — it is gather-class. Uses the whole-chain API
(`chain_apply` / `chain_backward`), NEVER the ring API. NO new comm primitive.

> **Implementer:** treat Batch D (ACdmap) as the literal template — write
> ACdmap first if you prefer (it is the canonical version), then derive FRmap by
> the i↔d swap table in "The FRmap gather-class decomposition" above. The F1–F5 /
> B1–B5 below ARE ACdmap's with `i`↔`d` and the output-leg scatter retargeted.

### Task B1: FRmap_slice2d_dist forward (F0-F5, 2-level d/i chunk, square grid)

**Files:** modify `src/contraction/slice2d.jl`; extend `test/test_slice2d_m3.jl`.

**Step 1 — failing forward test + off-diagonal `(a,i)` assertion** (the trap,
defeated by gathering the cross-axis legs full). FRmap output `[a,e,f,i]` has a
on r1, i on r2. Append to `test_slice2d_m3.jl` (and add `FRmap, FRmap_slice2d_dist`
to the `using TeneT:` line):

```julia
@testset "FRmap_slice2d_dist forward parity (square grid)" begin
    N1 = N2 = 2
    for χ in (16, 18), n in (1, 3)
        D = 3
        FR, ARu, ARd, M1, M2, _ = make_leg5(χ, D; seed=2100 + χ + n)
        g = slice2d_grid(N1, N2)
        ref = FRmap(FR, ARu, ARd, (M1, M2))   # result[a,e,f,i] := ARd[i,j,k,l] FR[d,g,h,l] M1 M2 ARu[a,b,c,d]
        FRb = slice2d_scatter(FR, g); ARub = slice2d_scatter(ARu, g); ARdb = slice2d_scatter(ARd, g)
        out = slice2d_gather(FRmap_slice2d_dist(FRb, ARub, ARdb, (M1, M2), g; forloop_iter=n), g)
        @test out ≈ ref rtol = 1e-12
        if χ == 18   # off-diagonal (a,i) plane — the trap (a-blk≠i-blk)
            @test out[1:9, :, :, 10:18] ≈ ref[1:9, :, :, 10:18] rtol = 1e-12   # a-blk 0, i-blk 1
            @test out[10:18, :, :, 1:9] ≈ ref[10:18, :, :, 1:9] rtol = 1e-12   # a-blk 1, i-blk 0
        end
        # single-M entry (M2 = conj(M1) internally)
        out1 = slice2d_gather(FRmap_slice2d_dist(FRb, ARub, ARdb, M1, g), g)
        @test out1 ≈ FRmap(FR, ARu, ARd, M1) rtol = 1e-12
    end
    @test_skip "FRmap_slice2d_dist rectangular grid (N1≠N2) deferred to M3 v2"
end
```

**Step 2 — run, expect FAIL** (`FRmap_slice2d_dist` undefined).

**Step 3 — implement** F0-F5 (the i↔d swap of ACdmap's §5.1). In `slice2d.jl`:

```julia
# ─── FRmap (cross-axis gather class — SQUARE grid, 2-LEVEL d/i chunk) ────────
# result[a,e,f,i] := ARd[i,j,k,l] FR[d,g,h,l] M1[e,j,g,b,p] M2[f,k,h,c,p] ARu[a,b,c,d]
# STRUCTURALLY ACdmap with i↔d swapped: cross-axis CONTRACTED d (FR.1=r1,
# ARu.4=r2) + cross-axis OUTPUT i (ARd.1=r1, result.4=r2); co-dist output a
# (local r1); aligned contracted l (local r2). Pinned chain intermediates
# I1/I2/I3 carry FULL i × FULL d (verified ≡ ACDMAP label-sets) — chunking l
# bounds NOTHING. Fix: 2-level loop over d (contracted → ACCUMULATE Σ_d) and i
# (output → ASSIGN disjoint i-slice). Local l-block summed WHOLE inside each
# chain_apply. Square grid REQUIRED (single p_rs). Local einsum is
# FRMAP_LEG5_CHAIN via chain_apply (whole-chain API), NOT a ring, NOT a hand
# kernel. Leg placement (slice accordingly — see loop): i is on ARd.1 (first leg)
# and result.4; d is on FR.1 (first leg) and ARu.4 (LAST leg). So the i-chunk
# `ich` slices ARd_g's FIRST leg; the d-chunk `dch` slices FR_g's FIRST leg AND
# ARu_g's LAST leg.
function _frmap_slice2d_forward_sliced(ARd_g, FR_g, ARu_g, M1, M2, grid, p_rs, n_d, n_i; forloop_iter = 1)
    # ARd_g = ARd[i∈1:χ, j, k, l∈p_rs[r2+1]]   (col_allgather, full i)
    # FR_g  = FR[d∈1:χ, g, h, l∈p_rs[r2+1]]    (col_allgather, full d)
    # ARu_g = ARu[a∈p_rs[r1+1], b, c, d∈1:χ]   (row_allgather, full d)
    χ = sum(length, p_rs)
    na = length(p_rs[grid.r1 + 1])             # local a extent
    # out (a,e,f,i): e = M1's FIRST leg (:e in (:e,:j,:g,:b,:p)) = size(M1,1);
    # f = M2's FIRST leg (:f in (:f,:k,:h,:c,:p)) = size(M2,1). (Verified with
    # non-uniform bonds; uniform-D test would mask a wrong index.)
    partial = similar(ARu_g, na, size(M1,1), size(M2,1), χ)   # [a-block, e, f, i∈1:χ]
    d_chunks = split_ranges(χ, min(n_d, χ))    # contracted → ACCUMULATE Σ_d
    i_chunks = split_ranges(χ, min(n_i, χ))    # output     → ASSIGN disjoint i-slice
    for ich in i_chunks                          # disjoint output i-slices → ASSIGN
        acc = nothing
        for dch in d_chunks                      # summed contracted d-slices → ACCUMULATE Σ_d
            piece = chain_apply(FRMAP_LEG5_CHAIN,
                (ARd_g[ich, :, :, :], FR_g[dch, :, :, :], M1, M2, ARu_g[:, :, :, dch]))
            if acc === nothing
                acc = piece
            else
                acc .+= piece; _free!(piece)
            end
        end
        view(partial, :, :, :, ich) .= acc; _free!(acc)   # disjoint i-slice assignment
    end
    # F5: sum l over row (full last leg i), keep i-block r2.
    @assert size(partial, 4) == χ "FRmap F5: partial last leg must be full i"
    result = _slice2d_row_reduce_scatter_last(partial, grid, p_rs)   # tag 760
    return result, ARd_g, FR_g, ARu_g
end

function FRmap_slice2d_dist(FR_blk, ARu_blk, ARd_blk, M, grid::Slice2DGrid;
                           forloop_iter = 1, inner_etype = nothing)
    @assert grid.N1 == grid.N2 "FRmap_slice2d_dist: M3 v1 requires a square grid (N1==N2)"
    M1, M2 = M isa Tuple ? M : (M, conj(M))
    T_orig = eltype(FR_blk)
    do_cast = inner_etype !== nothing && inner_etype != real(T_orig)
    if do_cast
        FR_blk  = _downcast_eltype(inner_etype, FR_blk)
        ARu_blk = _downcast_eltype(inner_etype, ARu_blk)
        ARd_blk = _downcast_eltype(inner_etype, ARd_blk)
        M1 = _downcast_eltype(inner_etype, M1); M2 = _downcast_eltype(inner_etype, M2)
    end
    χ = MPI.Allreduce(size(ARd_blk, 1), +, grid.col_comm)       # F0
    p_rs = split_ranges(χ, grid.N1)
    n_chunk = grid.N1 * ceil(Int, sqrt(forloop_iter))          # §6.2: n_d = n_i = N·⌈√forloop_iter⌉
    ARd_g = _slice2d_col_allgather(ARd_blk, grid, p_rs)         # F1: full i (tag 730)
    ARu_g = _slice2d_row_allgather(ARu_blk, grid, p_rs)         # F2: full d (tag 750)
    FR_g  = _slice2d_col_allgather(FR_blk,  grid, p_rs)         # F3: full d (tag 730)
    result, _, _, _ = _frmap_slice2d_forward_sliced(ARd_g, FR_g, ARu_g, M1, M2, grid, p_rs, n_chunk, n_chunk; forloop_iter)
    return do_cast ? T_orig.(result) : result
end
```

> **Slice cross-check (FRMAP_LEG5_CHAIN ops `(ARd, FR, M1, M2, ARu)`,
> `chain_maps.jl:71`):** the chain expects tensors in ops order. `ARd_g[ich,:,:,:]`
> slices `i` (ARd first leg); `FR_g[dch,:,:,:]` slices `d` (FR first leg);
> `ARu_g[:,:,:,dch]` slices `d` (ARu LAST leg). The inner `d`-loop accumulates
> (`Σ_d`, the cross-axis contracted leg, contracted at the ARu link); the outer
> `i`-loop assigns disjoint output slices. **`partial` middle-leg dims (verified
> with non-uniform bonds):** out `(:a,:e,:f,:i)` → `e = size(M1,1)` (`:e` is M1's
> first leg), `f = size(M2,1)` (`:f` is M2's first leg) — NOT `size(M1,2)`/
> `size(M2,2)` (those are `j`/`k`, the ACmap output legs). A uniform-D test masks
> a wrong index; the `na, size(M1,1), size(M2,1), χ` above is the checked form.
> (This is the FRmap analog of the Batch D MINOR dim fix — ACdmap's are
> `size(M1,4)`/`size(M2,4)`.)

**Step 4 — run, expect PASS** (m3 + slice2d). The off-diagonal assertion is the
diagonal-trap guard; the `forloop_iter=3` case forces `n_d,n_i ≥ 2N`. **Step 5
— commit:** `feat: FRmap_slice2d_dist forward (F0-F5, 2-level d/i chunk, square grid)`

### Task B2: FRmap_slice2d_dist rrule (B0-B5, same 2-level chunk)

**Files:** modify `src/autodiff/rules.jl` (clone `rrule(ACdmap_slice2d_dist)`
with i↔d swap); extend `test/test_slice2d_m3.jl`.

**Step 1 — failing gradient test** (block-of for output `[a,e,f,i]` is
`[p_rs[r1+1],:,:,p_rs[r2+1]]`, a first=r1 / i last=r2):

```julia
@testset "FRmap_slice2d_dist gradient parity (square grid)" begin
    N1 = N2 = 2
    for χ in (16, 18), n in (1, 3)
        D = 3
        FR, ARu, ARd, M1, M2, W = make_leg5(χ, D; seed=2200 + χ + n)
        g = slice2d_grid(N1, N2)
        FRb = slice2d_scatter(FR, g); ARub = slice2d_scatter(ARu, g); ARdb = slice2d_scatter(ARd, g)
        Wb = slice2d_scatter(W, g)
        loss_ref(FR,ARu,ARd,M1,M2)  = real(sum(W  .* FRmap(FR,ARu,ARd,(M1,M2))))
        loss_dist(FRb,ARub,ARdb,M1,M2) = real(sum(Wb .* FRmap_slice2d_dist(FRb,ARub,ARdb,(M1,M2),g; forloop_iter=n)))
        g_ref  = Zygote.pullback(loss_ref,  FR,ARu,ARd,M1,M2)[2](1.0)
        g_dist = Zygote.pullback(loss_dist, FRb,ARub,ARdb,M1,M2)[2](1.0)
        p_rs = split_ranges(χ, N1)
        blkof(x) = x[p_rs[g.r1+1], :, :, p_rs[g.r2+1]]
        @test g_dist[1] ≈ blkof(g_ref[1]) rtol = 1e-10   # dFR block
        @test g_dist[2] ≈ blkof(g_ref[2]) rtol = 1e-10   # dARu block
        @test g_dist[3] ≈ blkof(g_ref[3]) rtol = 1e-10   # dARd block
        @test g_dist[4] ≈ g_ref[4] rtol = 1e-10          # dM1 replicated
        @test g_dist[5] ≈ g_ref[5] rtol = 1e-10          # dM2
    end
    # single-M dM = dM1 + conj(dM2) composition
    FR, ARu, ARd, M1, M2, W = make_leg5(16, 3; seed=2250)
    g = slice2d_grid(2, 2)
    FRb=slice2d_scatter(FR,g); ARub=slice2d_scatter(ARu,g); ARdb=slice2d_scatter(ARd,g); Wb=slice2d_scatter(W,g)
    lr(FR,M) = real(sum(W  .* FRmap(FR,ARu,ARd,M)))
    ld(FRb,M)= real(sum(Wb .* FRmap_slice2d_dist(FRb,ARub,ARdb,M,g)))
    gr = Zygote.pullback(lr, FR, M1)[2](1.0); gd = Zygote.pullback(ld, FRb, M1)[2](1.0)
    p_rs = split_ranges(16, 2)
    @test gd[1] ≈ gr[1][p_rs[g.r1+1], :, :, p_rs[g.r2+1]] rtol = 1e-10
    @test gd[2] ≈ gr[2] rtol = 1e-10
    # bare-sum loss (FillArrays densify guard)
    back = Zygote.pullback(x -> real(sum(FRmap_slice2d_dist(x, ARub, ARdb, (M1,M2), g))), FRb)[2]
    dblk = back(1.0)[1]
    dFR_ref = Zygote.pullback(x -> real(sum(FRmap(x, ARu, ARd, M1, M2))), FR)[2](1.0)[1]
    @test dblk ≈ dFR_ref[p_rs[g.r1+1], :, :, p_rs[g.r2+1]] rtol = 1e-10
    # inner_etype Float32 boundary cast (fwd 1e-4, grad 1e-3)
    out32 = slice2d_gather(FRmap_slice2d_dist(FRb, ARub, ARdb, (M1,M2), g; inner_etype=Float32), g)
    @test eltype(out32) == ComplexF64
    @test out32 ≈ FRmap(FR, ARu, ARd, (M1,M2)) rtol = 1e-4
end
```

**Step 2 — run, expect FAIL** (no rrule → Zygote tries to differentiate through
the MPI forward and errors / mismatches).

**Step 3 — implement** B0-B5 (the i↔d swap of ACdmap's §5.2). The forward does
the three gathers in the primal (capture `ARd_g`, `FR_g`, `ARu_g` + `p_rs` +
`M1`/`M2` + the chunk counts; design doc §6.3). SAME 2-level d/i chunk
(`chain_backward` recomputes I1/I2/I3 → same full-i×full-d intermediates). Use
the **whole-chain backward** `chain_backward(FRMAP_LEG5_CHAIN, …)`, NOT the ring
API:

- **B0:** densify non-DenseArray cotangent (FillArrays guard, `rules.jl:618-622`);
  do_cast.
- **B5:** `dpartial = _slice2d_row_allgather(d_c, grid, p_rs)` (tag 750, adjoint
  of F5 row_reduce_scatter_last) → full i, local a.
- **B4 (2-level loop, Zygote-free):** zero-init the gathered-slice grad
  accumulators FIRST (mirror `rules.jl:632-639`):
  ```julia
  dARd_g = zero(ARd_g); dFR_g = zero(FR_g); dARu_g = zero(ARu_g)
  dM1 = zero(M1); dM2 = zero(M2)
  ```
  Then loop. `dpartial[:,:,:,ich]` is the cotangent for the i-chunk (every
  d-chunk shares it — d is contracted, no slot in dpartial). Chain grads come
  back in **ops order** `(dARd, dFR, dM1, dM2, dARu)` (`chain_engine.jl:231`):
  ```julia
  for ich in i_chunks
    for dch in d_chunks
      (dARd_c, dFR_c, dM1_c, dM2_c, dARu_c) =
          chain_backward(FRMAP_LEG5_CHAIN,
              (ARd_g[ich,:,:,:], FR_g[dch,:,:,:], M1, M2, ARu_g[:,:,:,dch]),
              dpartial[:,:,:,ich])
      view(dARd_g, ich,:,:,:) .+= dARd_c        # i-sliced disjoint per ich (accumulate over dch)
      view(dFR_g, dch,:,:,:)  .+= dFR_c         # d-sliced (accumulate over both loops)
      view(dARu_g,:,:,:,dch)  .+= dARu_c        # d-sliced (accumulate over both loops)
      dM1 .+= dM1_c; dM2 .+= dM2_c
      _free!(dARd_c); _free!(dFR_c); _free!(dARu_c); _free!(dM1_c); _free!(dM2_c)
    end
  end
  ```
- **B3 — adjoint of F3 col_allgather on FR:** `dFR_blk =
  _slice2d_col_reduce_scatter(dFR_g, grid, p_rs)` (tag 710) — `dFR_g` full d,
  local l-block → keep d-block r1.
- **B2 — adjoint of F2 row_allgather on ARu:** `dARu_blk =
  _slice2d_row_reduce_scatter_last(dARu_g, grid, p_rs)` (tag 760) — `dARu_g`
  a-block, full d → keep d-block r2.
- **B1 — adjoint of F1 col_allgather on ARd:** `dARd_blk =
  _slice2d_col_reduce_scatter(dARd_g, grid, p_rs)` (tag 710) — `dARd_g` full i,
  local l-block → keep i-block r1.
- **BM:** `allreduce_p2p!(dM1,+,grid.comm)`; `allreduce_p2p!(dM2,+,grid.comm)`;
  `dM = is_tuple ? (dM1,dM2) : dM1 .+ conj(dM2)`; do_cast upcast.
- return `(NoTangent(), dFR_blk, dARu_blk, dARd_blk, dM, NoTangent())` — map arg
  order `FRmap_slice2d_dist(FR, ARu, ARd, M, grid)`. The chain returns
  `(dARd, dFR, dM1, dM2, dARu)`; permute to map order `(dFR, dARu, dARd, dM)`
  exactly as `engine_backward(::typeof(FRmap),…)` does (`chain_maps.jl:75-97`,
  return `(g[2], g[5], g[1], (g[3],g[4]))`).

> **Adjoint pairs (all existing, all documented):** F5 row_reduce_scatter_last
> (760) ↔ B5 row_allgather (750); F1/F3 col_allgather (730) ↔ B1/B3
> col_reduce_scatter (710); F2 row_allgather (750) ↔ B2 row_reduce_scatter_last
> (760). The A2-grad parity test is the gate.

**Step 4 — run, expect PASS** (m3 + slice2d). **Step 5 — commit:**
`feat: rrule(FRmap_slice2d_dist) (B0-B5, same d/i chunk, chain_backward, no Zygote)`

### Task B3: FRmap forloop_iter clamp + 2-level chunk parity

**Files:** extend `test/test_slice2d_m3.jl`.

**Step 1 — parity test** (append): the `forloop_iter=99` clamp path (each `n_d`,
`n_i` clamps to `≤ χ` via `min(n, χ)`):

```julia
@testset "FRmap_slice2d_dist forloop_iter clamp (square grid)" begin
    N1 = N2 = 2
    χ, D = 18, 3
    FR, ARu, ARd, M1, M2, _ = make_leg5(χ, D; seed=2300)
    g = slice2d_grid(N1, N2)
    ref = FRmap(FR, ARu, ARd, (M1, M2))
    FRb=slice2d_scatter(FR,g); ARub=slice2d_scatter(ARu,g); ARdb=slice2d_scatter(ARd,g)
    out = slice2d_gather(FRmap_slice2d_dist(FRb, ARub, ARdb, (M1,M2), g; forloop_iter=99), g)
    @test out ≈ ref rtol = 1e-12
end
```

**Step 2-4 — run-expect-fail-then-pass** (likely passes immediately if B1 clamps
`n_d`/`n_i` with `min(n, χ)`; if not, add the clamp). **Step 5 — commit:**
`test: FRmap_slice2d_dist forloop_iter clamp parity`

### Task B4: FRmap distributed self-consistency (square-grid gate)

**Files:** none new; explicit **gate confirmation** — run the full
`test/run_test_slice2d_m3.jl` and `test/run_test_slice2d.jl`, confirm Batch B is
green on `(2,2)` at 1e-12 fwd / 1e-10 grad, the off-diagonal `(a,i)` assertion
passes, and the rectangular `@test_skip` is recorded. Confirm FRmap requires the
square assertion (NOT rectangular-safe — the corrected gather-class fact).

**Commit:** `docs: FRmap_slice2d_dist square-grid gate (gather-class, not a ring)`

### Task B5: FRmap rrule eager-free + capture audit

**Files:** review `src/autodiff/rules.jl` FRmap rrule against the ACdmap rrule's
eager-`_free!` and capture discipline (capture only the gathered slices `ARd_g`,
`FR_g`, `ARu_g` + `M1`/`M2` + `p_rs` + chunk counts, never a χ²D⁴ array, never a
χ×χ `(a,i)` plane). Add an inline comment stating the captured footprint
(3·χ²D²/N gathered slices, same order as ACdmap) and that the backward transient
is bounded by the 2-level d/i chunk (`chain_backward` recomputes the full-i×full-d
I1/I2/I3, bounded only by the chunk — capturing bounded slices alone does NOT
bound the recompute). No behavior change.

**Commit:** `chore: FRmap rrule eager-free + capture audit (matches ACdmap bound)`

---

## Batch C — ACmap_slice2d_dist (5 tasks)

The cross-axis (gather) class. Implement EXACTLY the design doc's ACmap
F1-F5 / B1-B5 (`docs/2026-06-13-acmap-slice2d-dataflow-design.md` §2.7, §4.2).
**Square grid required** (`@assert grid.N1 == grid.N2`). Single
`chain_apply(ACMAP_LEG5_CHAIN)` per l-chunk; col_reduce_scatter output. The
diagonal trap is defeated by gathering the cross-axis FREE leg `i` (on FL) to
full BEFORE the local chain.

### Task C1: ACmap_slice2d_dist forward (F0-F5)

**Files:** modify `src/contraction/slice2d.jl`; extend `test/test_slice2d_m3.jl`.

**Step 1 — failing forward test + the CRITICAL off-diagonal assertion** (design
doc §8). Append `ACmap, ACmap_slice2d_dist` to the `using TeneT:` line (NIT —
per-batch imports), then append to `test_slice2d_m3.jl`:

```julia
@testset "ACmap_slice2d_dist forward parity (square grid)" begin
    N1 = N2 = 2
    for χ in (16, 18), n in (1, 3)
        D = 3
        AC, FL, FR, M1, M2, _ = make_leg5(χ, D; seed=2600 + χ + n)
        g = slice2d_grid(N1, N2)
        ref = ACmap(AC, FL, FR, (M1, M2))      # result[i,j,k,l] := AC[a,b,c,d] FR[d,g,h,l] M1 M2 FL[a,e,f,i]
        ACb = slice2d_scatter(AC, g); FLb = slice2d_scatter(FL, g); FRb = slice2d_scatter(FR, g)
        out = slice2d_gather(ACmap_slice2d_dist(ACb, FLb, FRb, (M1,M2), g; forloop_iter=n), g)
        @test out ≈ ref rtol = 1e-12
        # CRITICAL off-diagonal (the diagonal trap): with χ=18, N=2 →
        # p_rs = [1:9, 10:18]; check off-diagonal (i,l) blocks A≠B explicitly.
        if χ == 18
            @test out[1:9, :, :, 10:18] ≈ ref[1:9, :, :, 10:18] rtol = 1e-12   # i-blk 0, l-blk 1
            @test out[10:18, :, :, 1:9] ≈ ref[10:18, :, :, 1:9] rtol = 1e-12   # i-blk 1, l-blk 0
        end
        # single-M entry
        out1 = slice2d_gather(ACmap_slice2d_dist(ACb, FLb, FRb, M1, g), g)
        @test out1 ≈ ACmap(AC, FL, FR, M1) rtol = 1e-12
    end
    # rectangular grids are M3 v2 — explicitly skipped:
    @test_skip "ACmap_slice2d_dist rectangular grid (N1≠N2) deferred to M3 v2"
end
```

**Step 2 — run, expect FAIL.**

**Step 3 — implement** F0-F5 verbatim from design doc §2.7. In `slice2d.jl`:

```julia
# ─── ACmap (cross-axis gather class — SQUARE grid only) ─────────────────────
# Design: docs/2026-06-13-acmap-slice2d-dataflow-design.md §2-§4.
# result[i,j,k,l] := AC[a,b,c,d] FR[d,g,h,l] M1[e,j,g,b,p] M2[f,k,h,c,p] FL[a,e,f,i]
# Two cross-axis legs: contracted d (AC.4=r2, FR.1=r1) + output i (FL.4=r2,
# result.1=r1). The diagonal trap (i and l both on r2) is defeated by gathering
# FL's free leg i to FULL before the local chain (F2). Square grid REQUIRED:
# p_rs = split_ranges(χ,N) is the single partition for every leg (risk 2).
# Local einsum is ACMAP_LEG5_CHAIN via chain_apply, NOT a hand kernel.
function _acmap_slice2d_forward_sliced(AC_g, FR_g, FL_g, M1, M2, grid, p_rs; forloop_iter = 1)
    # AC_g = AC[a∈p_rs[r1+1], b, c, d∈1:χ]   (row_allgather, full d)
    # FR_g = FR[d∈1:χ, g, h, l∈p_rs[r2+1]]   (col_allgather, full d)
    # FL_g = FL[a∈p_rs[r1+1], e, f, i∈1:χ]   (row_allgather, full i)
    χ = sum(length, p_rs)
    Dg, Dh = size(M1, 3), size(M2, 3)
    nl = size(FR_g, 4)                       # local l extent
    partial = similar(AC_g, χ, size(M1,2), size(M2,2), nl)   # [i∈1:χ, j, k, l-block]
    l_chunks = split_ranges(nl, min(forloop_iter, nl))
    for ch in l_chunks
        # F4: single local chain over full-i FL_g; full i, local l-chunk.
        Pc = chain_apply(ACMAP_LEG5_CHAIN, (AC_g, FR_g[:, :, :, ch], M1, M2, FL_g))
        view(partial, :, :, :, ch) .= Pc
        _free!(Pc)
    end
    # F5: sum a over col (full first leg i), keep i-block r1.
    @assert size(partial, 1) == χ "ACmap F5: partial first leg must be full i (risk 5)"
    result = _slice2d_col_reduce_scatter(partial, grid, p_rs)   # tag 710
    return result, AC_g, FR_g, FL_g
end

function ACmap_slice2d_dist(AC_blk, FL_blk, FR_blk, M, grid::Slice2DGrid;
                           forloop_iter = 1, inner_etype = nothing)
    @assert grid.N1 == grid.N2 "ACmap_slice2d_dist: M3 v1 requires a square grid (N1==N2)"
    M1, M2 = M isa Tuple ? M : (M, conj(M))
    T_orig = eltype(AC_blk)
    do_cast = inner_etype !== nothing && inner_etype != real(T_orig)
    if do_cast
        AC_blk = _downcast_eltype(inner_etype, AC_blk); FL_blk = _downcast_eltype(inner_etype, FL_blk)
        FR_blk = _downcast_eltype(inner_etype, FR_blk)
        M1 = _downcast_eltype(inner_etype, M1); M2 = _downcast_eltype(inner_etype, M2)
    end
    χ = MPI.Allreduce(size(AC_blk, 1), +, grid.col_comm)      # F0
    p_rs = split_ranges(χ, grid.N1)
    AC_g = _slice2d_row_allgather(AC_blk, grid, p_rs)          # F1: full d (tag 750)
    FL_g = _slice2d_row_allgather(FL_blk, grid, p_rs)          # F2: full i (tag 750)
    FR_g = _slice2d_col_allgather(FR_blk, grid, p_rs)          # F3: full d (tag 730)
    result, _, _, _ = _acmap_slice2d_forward_sliced(AC_g, FR_g, FL_g, M1, M2, grid, p_rs; forloop_iter)
    return do_cast ? T_orig.(result) : result
end
```

> **Verify the chain operand slot:** `ACMAP_LEG5_CHAIN` ops order is
> `(AC, FR, M1, M2, FL)` (`chain_maps.jl:106`). `chain_apply` takes tensors in
> ops order ⇒ `(AC_g, FR_g[:,:,:,ch], M1, M2, FL_g)`. The carried operand is
> AC; `a` is contracted at the LAST link (FL) so the chain sums only the local
> a-block — exactly what §2.4 requires for the col_reduce_scatter to complete
> `Σ_a`. Confirm `_slice2d_row_allgather`/`_slice2d_col_allgather` produce the
> shapes the docstrings claim (last-leg concat for row 750, first-leg concat
> for col 730).

**Step 4 — run, expect PASS** (m3 + slice2d). The off-diagonal assertion is the
diagonal-trap guard. **Step 5 — commit:**
`feat: ACmap_slice2d_dist forward (F0-F5, gather-i, square grid)`

### Task C2: ACmap_slice2d_dist rrule (B0-B5)

**Files:** modify `src/autodiff/rules.jl`; extend `test/test_slice2d_m3.jl`.

**Step 1 — failing gradient test** (clone the FLmap dist grad block; design doc
§8). Block-of for ACmap output `[i,j,k,l]` is `[p_rs[r1+1],:,:,p_rs[r2+1]]`
(i first=r1, l last=r2):

```julia
@testset "ACmap_slice2d_dist gradient parity (square grid)" begin
    N1 = N2 = 2
    for χ in (16, 18), n in (1, 3)
        D = 3
        AC, FL, FR, M1, M2, W = make_leg5(χ, D; seed=2700 + χ + n)
        g = slice2d_grid(N1, N2)
        ACb=slice2d_scatter(AC,g); FLb=slice2d_scatter(FL,g); FRb=slice2d_scatter(FR,g); Wb=slice2d_scatter(W,g)
        loss_ref(AC,FL,FR,M1,M2)  = real(sum(W  .* ACmap(AC,FL,FR,(M1,M2))))
        loss_dist(ACb,FLb,FRb,M1,M2) = real(sum(Wb .* ACmap_slice2d_dist(ACb,FLb,FRb,(M1,M2),g; forloop_iter=n)))
        g_ref  = Zygote.pullback(loss_ref,  AC,FL,FR,M1,M2)[2](1.0)
        g_dist = Zygote.pullback(loss_dist, ACb,FLb,FRb,M1,M2)[2](1.0)
        p_rs = split_ranges(χ, N1)
        blkof(x) = x[p_rs[g.r1+1], :, :, p_rs[g.r2+1]]
        @test g_dist[1] ≈ blkof(g_ref[1]) rtol = 1e-10   # dAC block
        @test g_dist[2] ≈ blkof(g_ref[2]) rtol = 1e-10   # dFL block
        @test g_dist[3] ≈ blkof(g_ref[3]) rtol = 1e-10   # dFR block
        @test g_dist[4] ≈ g_ref[4] rtol = 1e-10          # dM1 replicated
        @test g_dist[5] ≈ g_ref[5] rtol = 1e-10          # dM2
    end
    # single-M dM composition + bare-sum densify + inner_etype, as in the Batch B
    # FRmap grad-test block (Task B2).
end
```

**Step 2 — run, expect FAIL.**

**Step 3 — implement** B0-B5 verbatim from design doc §4.2. The forward does
the three gathers in the primal (capture `AC_g`, `FR_g`, `FL_g` + `p_rs` +
`M1`/`M2` + `forloop_iter` in the closure; design doc §6.3). Backward:

- **B0:** densify `dresult` if not DenseArray; do_cast.
- **B5:** `dpartial = _slice2d_col_allgather(d_c, grid, p_rs)` (tag 730) → full
  i, local l-block.
- **B4** (per l-chunk, **Zygote-free** `chain_backward`):
  `(dAC_g_c, dFR_g_c, dM1_c, dM2_c, dFL_g_c) = chain_backward(ACMAP_LEG5_CHAIN,
  (AC_g, FR_g[:,:,:,ch], M1, M2, FL_g), view(dpartial,:,:,:,ch))`. Grads come
  back in ops order `(dAC, dFR, dM1, dM2, dFL)`. Accumulate:
  `dAC_g .+= dAC_g_c`, `view(dFR_g,:,:,:,ch) .+= dFR_g_c`, `dFL_g .+= dFL_g_c`,
  `dM1 .+= dM1_c`, `dM2 .+= dM2_c`; eager `_free!` each `_c`.
- **B3:** `dFR_blk = _slice2d_col_reduce_scatter(dFR_g, grid, p_rs)` (tag 710).
- **B2:** `dFL_blk = _slice2d_row_reduce_scatter_last(dFL_g, grid, p_rs)` (760).
- **B1:** `dAC_blk = _slice2d_row_reduce_scatter_last(dAC_g, grid, p_rs)` (760).
- **BM:** `allreduce_p2p!(dM1,+,grid.comm)`; `allreduce_p2p!(dM2,+,grid.comm)`;
  `dM = is_tuple ? (dM1,dM2) : dM1 .+ conj(dM2)`; do_cast upcast.
- return `(NoTangent(), dAC_blk, dFL_blk, dFR_blk, dM, NoTangent())` — map order
  `ACmap_slice2d_dist(AC, FL, FR, M, grid)`.

> The chain returns `(dAC,dFR,dM1,dM2,dFL)`; the map arg order is
> `(AC, FL, FR, M)` so the return permutes to `(dAC, dFL, dFR, dM)` — exactly
> the `engine_backward(::typeof(ACmap),…)` permutation (`chain_maps.jl:116`).

**Step 4 — run, expect PASS** (m3 + slice2d). **Step 5 — commit:**
`feat: rrule(ACmap_slice2d_dist) (B0-B5, chain_backward, no Zygote)`

### Task C3: ACmap single-M + bare-sum + inner_etype

**Files:** extend `test/test_slice2d_m3.jl` with the single-M `dM = dM1 +
conj(dM2)` composition, bare-`sum` FillArrays densify, and `inner_etype=Float32`
(fwd 1e-4, grad 1e-3) cases — clone the Batch B FRmap grad-test block (Task B2).
**Commit:** `test: ACmap_slice2d_dist single-M / densify / inner_etype`

### Task C4: ACmap iterability (output feeds back as input)

**Files:** extend `test/test_slice2d_m3.jl`. ACmap output `[i,j,k,l]` has i on
r1, l on r2 — **same convention as its AC input** `[a,b,c,d]` (a=r1, d=r2)? AC
is `[a,b,c,d]` (a=r1, d=r2); output is `[i,j,k,l]` (i=r1, l=r2). The output
block convention matches the AC input convention, so the output block feeds
straight back as a new AC block (design doc §8 iterability check):

```julia
@testset "ACmap_slice2d_dist iterability" begin
    N1 = N2 = 2; χ, D = 16, 3
    AC, FL, FR, M1, M2, _ = make_leg5(χ, D; seed=2800)
    g = slice2d_grid(N1, N2)
    ACb=slice2d_scatter(AC,g); FLb=slice2d_scatter(FL,g); FRb=slice2d_scatter(FR,g)
    out_blk = ACmap_slice2d_dist(ACb, FLb, FRb, (M1,M2), g)
    out2 = slice2d_gather(ACmap_slice2d_dist(out_blk, FLb, FRb, (M1,M2), g), g)
    ref = ACmap(AC, FL, FR, (M1,M2))
    @test out2 ≈ ACmap(ref, FL, FR, (M1,M2)) rtol = 1e-11
end
```

**Commit:** `test: ACmap_slice2d_dist iterability (output block feeds back)`

### Task C5: ACmap square-grid assertion + memory-bound comment

**Files:** review `slice2d.jl` ACmap — confirm the `@assert N1==N2`, the
"single p_rs licensed by square grid" inline comment (risk 2), and the peak
memory comment (design doc §2.7: gathers χ²D²/N, partial χ²D²/N full-i local-l,
no χ×χ plane). Add the F2→F5 coupling assert (`size(partial,1)==χ`, risk 5).
No behavior change. **Commit:**
`chore: ACmap_slice2d_dist square-grid + memory-bound asserts (risks 2,5)`

---

## Batch D — ACdmap_slice2d_dist (5 tasks)

The cross-axis (gather) class with the **2-level i+d memory chunk** — the
BLOCKER the design fixed. ACdmap's internal intermediates carry full-i × full-d
and are bounded ONLY by chunking `i` (contracted → accumulate) and `d` (output
→ assign), NOT by chunking `l` (which is absent from them). Square grid
required. Implement EXACTLY design doc §5.1 (F0-F5) / §5.2 (B0-B5).

### Task D1: ACdmap_slice2d_dist forward (F0-F5, 2-level i/d chunk)

**Files:** modify `src/contraction/slice2d.jl`; extend `test/test_slice2d_m3.jl`.

**Step 1 — failing forward test + off-diagonal (a,d) assertion** (design doc
§5.1 completeness, §8). Append `ACdmap, ACdmap_slice2d_dist` to the `using TeneT:`
line (NIT — per-batch imports). ACdmap output `[a,b,c,d]` has a on r1, d on r2:

```julia
@testset "ACdmap_slice2d_dist forward parity (square grid)" begin
    N1 = N2 = 2
    for χ in (16, 18), n in (1, 3)
        D = 3
        ACd, FL, FR, M1, M2, _ = make_leg5(χ, D; seed=2900 + χ + n)
        g = slice2d_grid(N1, N2)
        ref = ACdmap(ACd, FL, FR, (M1, M2))   # result[a,b,c,d] := ACd[i,j,k,l] FR[d,g,h,l] M1 M2 FL[a,e,f,i]
        ACdb = slice2d_scatter(ACd, g); FLb = slice2d_scatter(FL, g); FRb = slice2d_scatter(FR, g)
        out = slice2d_gather(ACdmap_slice2d_dist(ACdb, FLb, FRb, (M1,M2), g; forloop_iter=n), g)
        @test out ≈ ref rtol = 1e-12
        if χ == 18   # off-diagonal (a,d) plane — the trap, transposed (§5.1)
            @test out[1:9, :, :, 10:18] ≈ ref[1:9, :, :, 10:18] rtol = 1e-12
            @test out[10:18, :, :, 1:9] ≈ ref[10:18, :, :, 1:9] rtol = 1e-12
        end
        out1 = slice2d_gather(ACdmap_slice2d_dist(ACdb, FLb, FRb, M1, g), g)
        @test out1 ≈ ACdmap(ACd, FL, FR, M1) rtol = 1e-12
    end
    @test_skip "ACdmap_slice2d_dist rectangular grid (N1≠N2) deferred to M3 v2"
end
```

**Step 2 — run, expect FAIL.**

**Step 3 — implement** F0-F5 verbatim from design doc §5.1. The chunk-leg
difference from ACmap is the headline (design doc §6.2, risk 1): ACmap chunks
`l` (one loop, assign); ACdmap chunks `i` (inner, accumulate) AND `d` (outer,
assign). `n_i = n_d = N·⌈√forloop_iter⌉` derived internally from `grid.N` and
`forloop_iter`, clamped to `≤ χ`.

```julia
# ─── ACdmap (cross-axis gather class — SQUARE grid, 2-LEVEL i/d chunk) ───────
# Design: docs/2026-06-13-acmap-slice2d-dataflow-design.md §5-§6.
# result[a,b,c,d] := ACd[i,j,k,l] FR[d,g,h,l] M1[e,j,g,b,p] M2[f,k,h,c,p] FL[a,e,f,i]
# Cross-axis the OTHER way vs ACmap: contracted i (ACd.1=r1, FL.4=r2) + output
# d (FR.1=r1, result.4=r2). Internal chain intermediates I1/I2/I3 carry FULL i
# × FULL d (absent l) — chunking l bounds NOTHING (THE BLOCKER §5.1). Fix:
# 2-level loop over i (contracted → ACCUMULATE Σ_i) and d (output → ASSIGN
# disjoint d-slice). n_i·n_d = N²·forloop_iter reaches the ACmap-class
# χ²D⁴/(P·forloop_iter) bound (§6.2). Local l-block summed WHOLE inside each
# chain_apply (not chunked). Square grid REQUIRED. Local einsum is
# ACDMAP_LEG5_CHAIN via chain_apply, NOT a hand kernel.
function _acdmap_slice2d_forward_sliced(ACd_g, FR_g, FL_g, M1, M2, grid, p_rs, n_i, n_d; forloop_iter = 1)
    # ACd_g = ACd[i∈1:χ, j, k, l∈p_rs[r2+1]]  (col_allgather, full i)
    # FR_g  = FR[d∈1:χ, g, h, l∈p_rs[r2+1]]   (col_allgather, full d)
    # FL_g  = FL[a∈p_rs[r1+1], e, f, i∈1:χ]   (row_allgather, full i)
    χ = sum(length, p_rs)
    na = length(p_rs[grid.r1 + 1])             # local a extent
    # out (a,b,c,d): b = M1's leg :b in (:e,:j,:g,:b,:p) = size(M1,4);
    # c = M2's leg :c in (:f,:k,:h,:c,:p) = size(M2,4). NOT size(M1,2)/size(M2,2)
    # (those are :j/:k, the ACmap output legs). Verified with non-uniform bonds
    # (Db≠Dc); a uniform-D=3 test masks the wrong index — D3 below adds a Db≠Dc
    # test so this cannot hide.
    partial = similar(FL_g, na, size(M1,4), size(M2,4), χ)   # [a-block, b, c, d∈1:χ]
    i_chunks = split_ranges(χ, min(n_i, χ))
    d_chunks = split_ranges(χ, min(n_d, χ))
    for dch in d_chunks                          # disjoint output d-slices → ASSIGN
        acc = nothing
        for ich in i_chunks                      # summed input i-slices → ACCUMULATE Σ_i
            piece = chain_apply(ACDMAP_LEG5_CHAIN,
                (ACd_g[ich, :, :, :], FR_g[dch, :, :, :], M1, M2, FL_g[:, :, :, ich]))
            if acc === nothing
                acc = piece
            else
                acc .+= piece; _free!(piece)
            end
        end
        view(partial, :, :, :, dch) .= acc; _free!(acc)
    end
    # F5: sum l over row (full last leg d), keep d-block r2.
    @assert size(partial, 4) == χ "ACdmap F5: partial last leg must be full d (mirror of ACmap risk-5)"
    result = _slice2d_row_reduce_scatter_last(partial, grid, p_rs)   # tag 760
    return result, ACd_g, FR_g, FL_g
end

function ACdmap_slice2d_dist(ACd_blk, FL_blk, FR_blk, M, grid::Slice2DGrid;
                            forloop_iter = 1, inner_etype = nothing)
    @assert grid.N1 == grid.N2 "ACdmap_slice2d_dist: M3 v1 requires a square grid (N1==N2)"
    M1, M2 = M isa Tuple ? M : (M, conj(M))
    T_orig = eltype(ACd_blk)
    do_cast = inner_etype !== nothing && inner_etype != real(T_orig)
    if do_cast
        ACd_blk = _downcast_eltype(inner_etype, ACd_blk); FL_blk = _downcast_eltype(inner_etype, FL_blk)
        FR_blk = _downcast_eltype(inner_etype, FR_blk)
        M1 = _downcast_eltype(inner_etype, M1); M2 = _downcast_eltype(inner_etype, M2)
    end
    χ = MPI.Allreduce(size(ACd_blk, 1), +, grid.col_comm)        # F0
    p_rs = split_ranges(χ, grid.N1)
    n_chunk = grid.N1 * ceil(Int, sqrt(forloop_iter))           # §6.2: n_i = n_d = N·⌈√forloop_iter⌉
    ACd_g = _slice2d_col_allgather(ACd_blk, grid, p_rs)          # F1: full i (tag 730)
    FL_g  = _slice2d_row_allgather(FL_blk, grid, p_rs)           # F2: full i (tag 750)
    FR_g  = _slice2d_col_allgather(FR_blk, grid, p_rs)           # F3: full d (tag 730)
    result, _, _, _ = _acdmap_slice2d_forward_sliced(ACd_g, FR_g, FL_g, M1, M2, grid, p_rs, n_chunk, n_chunk; forloop_iter)
    return do_cast ? T_orig.(result) : result
end
```

> **The copy-paste hazard (risk 1, design doc §6.2):** the inner `i`-loop
> **accumulates** (`acc .+= chain_apply(…)`, `_free!(piece)`); the outer
> `d`-loop **assigns** a disjoint slice (`view(partial,:,:,:,dch) .= acc`). A
> copy-paste of ACmap's single assignment loop would (a) drop all but the last
> `i`-chunk (missing `Σ_i`) and (b) fail to bound the intermediates. The
> `forloop_iter=3` test (forces `n_i,n_d ≥ N·2`) plus the off-diagonal
> assertion is the guard. `ACDMAP_LEG5_CHAIN` ops order is
> `(ACd, FR, M1, M2, FL)` (`chain_maps.jl:139`); `ACd_g[ich,:,:,:]` and
> `FL_g[:,:,:,ich]` slice `i`; `FR_g[dch,:,:,:]` slices `d`.

**Step 4 — run, expect PASS** (m3 + slice2d). **Step 5 — commit:**
`feat: ACdmap_slice2d_dist forward (F0-F5, 2-level i/d chunk, square grid)`

### Task D2: ACdmap_slice2d_dist rrule (B0-B5, same 2-level chunk)

**Files:** modify `src/autodiff/rules.jl`; extend `test/test_slice2d_m3.jl`.

**Step 1 — failing gradient test** (block-of for output `[a,b,c,d]` is
`[p_rs[r1+1],:,:,p_rs[r2+1]]`):

```julia
@testset "ACdmap_slice2d_dist gradient parity (square grid)" begin
    N1 = N2 = 2
    for χ in (16, 18), n in (1, 3)
        D = 3
        ACd, FL, FR, M1, M2, W = make_leg5(χ, D; seed=3000 + χ + n)
        g = slice2d_grid(N1, N2)
        ACdb=slice2d_scatter(ACd,g); FLb=slice2d_scatter(FL,g); FRb=slice2d_scatter(FR,g); Wb=slice2d_scatter(W,g)
        loss_ref(ACd,FL,FR,M1,M2)  = real(sum(W  .* ACdmap(ACd,FL,FR,(M1,M2))))
        loss_dist(ACdb,FLb,FRb,M1,M2) = real(sum(Wb .* ACdmap_slice2d_dist(ACdb,FLb,FRb,(M1,M2),g; forloop_iter=n)))
        g_ref  = Zygote.pullback(loss_ref,  ACd,FL,FR,M1,M2)[2](1.0)
        g_dist = Zygote.pullback(loss_dist, ACdb,FLb,FRb,M1,M2)[2](1.0)
        p_rs = split_ranges(χ, N1)
        blkof(x) = x[p_rs[g.r1+1], :, :, p_rs[g.r2+1]]
        @test g_dist[1] ≈ blkof(g_ref[1]) rtol = 1e-10   # dACd block
        @test g_dist[2] ≈ blkof(g_ref[2]) rtol = 1e-10   # dFL block
        @test g_dist[3] ≈ blkof(g_ref[3]) rtol = 1e-10   # dFR block
        @test g_dist[4] ≈ g_ref[4] rtol = 1e-10          # dM1 replicated
        @test g_dist[5] ≈ g_ref[5] rtol = 1e-10          # dM2
    end
end
```

**Step 2 — run, expect FAIL.**

**Step 3 — implement** B0-B5 verbatim from design doc §5.2. SAME 2-level i/d
chunk (chain_backward recomputes I1/I2/I3 → same full-i×full-d intermediates):

- **B0:** densify non-DenseArray cotangent (FillArrays guard, `rules.jl:618-622`);
  do_cast.
- **B5:** `dpartial = _slice2d_row_allgather(d_c, grid, p_rs)` (tag 750, adjoint
  of F5 row_reduce_scatter_last) → full d, local a.
- **B4** (2-level loop, **Zygote-free**): zero-init the gathered-slice grad
  accumulators FIRST (mirror `rules.jl:632-639`):
  ```julia
  dACd_g = zero(ACd_g); dFR_g = zero(FR_g); dFL_g = zero(FL_g)
  dM1 = zero(M1); dM2 = zero(M2)
  ```
  Then loop. `dpartial[:,:,:,dch]` is the cotangent for the d-chunk (every
  i-chunk shares it — i is contracted, no slot in dpartial).
  `(dACd_g_c, dFR_g_c, dM1_c, dM2_c, dFL_g_c) =
  chain_backward(ACDMAP_LEG5_CHAIN, (ACd_g[ich,:,:,:], FR_g[dch,:,:,:], M1, M2,
  FL_g[:,:,:,ich]), dpartial[:,:,:,dch])`. Accumulate:
  `view(dACd_g,ich,:,:,:) .+= dACd_g_c` (i-sliced, accumulate both loops);
  `view(dFR_g,dch,:,:,:) .+= dFR_g_c` (d-sliced disjoint per dch, accumulate
  over ich); `view(dFL_g,:,:,:,ich) .+= dFL_g_c` (i-sliced); `dM1 .+= dM1_c`;
  `dM2 .+= dM2_c`; eager `_free!` each `_c`.
- **B3:** `dFR_blk = _slice2d_col_reduce_scatter(dFR_g, grid, p_rs)` (710,
  adjoint of F3 col_allgather).
- **B2:** `dFL_blk = _slice2d_row_reduce_scatter_last(dFL_g, grid, p_rs)` (760,
  adjoint of F2 row_allgather).
- **B1:** `dACd_blk = _slice2d_col_reduce_scatter(dACd_g, grid, p_rs)` (710,
  adjoint of F1 col_allgather).
- **BM:** allreduce dM1/dM2; compose dM; do_cast upcast.
- return `(NoTangent(), dACd_blk, dFL_blk, dFR_blk, dM, NoTangent())` — map
  order `ACdmap_slice2d_dist(ACd, FL, FR, M, grid)`; chain `(dACd,dFR,...,dFL)`
  permutes to `(dACd, dFL, dFR, dM)` (chain_maps.jl:147).

**Step 4 — run, expect PASS** (m3 + slice2d). **Step 5 — commit:**
`feat: rrule(ACdmap_slice2d_dist) (B0-B5, same i/d chunk, chain_backward, no Zygote)`

### Task D3: ACdmap single-M + densify + inner_etype + non-uniform-bond (Db≠Dc)

**Files:** extend `test/test_slice2d_m3.jl` — single-M `dM` composition,
bare-sum FillArrays densify, `inner_etype=Float32`. Clone the Batch-C ACmap
single-M block.

**Non-uniform-bond test (MINOR — guards the `partial` dim fix).** The output
middle legs `b = size(M1,4)`, `c = size(M2,4)` are masked by a uniform `D=3`
test (every dim is 3). Add a `Db ≠ Dc` case so a wrong index (`size(M1,2)` etc.)
cannot hide. Build operands with distinct M-slot bonds and assert forward +
gradient parity:

```julia
@testset "ACdmap_slice2d_dist non-uniform bond (Db≠Dc) parity" begin
    N1 = N2 = 2; χ = 16
    # distinct M-slot bonds: M1=(e,j,g,b,p), M2=(f,k,h,c,p); pick b=5 (Db), c=7 (Dc), b≠c.
    De, Df, Dj, Dk, Dg, Dh, Db, Dc, Dp = 2, 2, 3, 3, 4, 4, 5, 7, 6
    Random.seed!(3050)
    M1 = rand(ComplexF64, De, Dj, Dg, Db, Dp)
    M2 = rand(ComplexF64, Df, Dk, Dh, Dc, Dp)
    ACd = rand(ComplexF64, χ, Dj, Dk, χ)         # (i,j,k,l)
    FL  = rand(ComplexF64, χ, De, Df, χ)         # (a,e,f,i)
    FR  = rand(ComplexF64, χ, Dg, Dh, χ)         # (d,g,h,l)
    W   = rand(ComplexF64, χ, Db, Dc, χ)         # out (a,b,c,d)
    g = slice2d_grid(N1, N2)
    ACdb=slice2d_scatter(ACd,g); FLb=slice2d_scatter(FL,g); FRb=slice2d_scatter(FR,g); Wb=slice2d_scatter(W,g)
    ref = ACdmap(ACd, FL, FR, (M1, M2))
    @test size(ref) == (χ, Db, Dc, χ)            # b=size(M1,4)=5, c=size(M2,4)=7 — the dim fix
    out = slice2d_gather(ACdmap_slice2d_dist(ACdb, FLb, FRb, (M1,M2), g), g)
    @test out ≈ ref rtol = 1e-12
    loss_ref(ACd,FL,FR,M1,M2)  = real(sum(W  .* ACdmap(ACd,FL,FR,(M1,M2))))
    loss_dist(ACdb,FLb,FRb,M1,M2) = real(sum(Wb .* ACdmap_slice2d_dist(ACdb,FLb,FRb,(M1,M2),g)))
    g_ref  = Zygote.pullback(loss_ref,  ACd,FL,FR,M1,M2)[2](1.0)
    g_dist = Zygote.pullback(loss_dist, ACdb,FLb,FRb,M1,M2)[2](1.0)
    p_rs = split_ranges(χ, N1); blkof(x) = x[p_rs[g.r1+1], :, :, p_rs[g.r2+1]]
    @test g_dist[1] ≈ blkof(g_ref[1]) rtol = 1e-10
    @test g_dist[4] ≈ g_ref[4] rtol = 1e-10
    @test g_dist[5] ≈ g_ref[5] rtol = 1e-10
end
```

(The analogous FRmap `Db≠Dc` guard — `e=size(M1,1)`, `f=size(M2,1)` — can be
added to Batch B Task B3 if desired; the ACdmap one is the canonical guard since
FRmap is its clone.)

**Commit:** `test: ACdmap_slice2d_dist single-M / densify / inner_etype / Db≠Dc dim guard`

### Task D4: ACdmap not-self-iterating (compose into ACmap)

**Files:** extend `test/test_slice2d_m3.jl`. ACdmap is NOT self-iterating
(output is `{a,d}` top, input is `{i,l}` bottom — design doc §8). Its iterate
test feeds its output into a matching `ACmap_slice2d_dist` and compares the
composed serial maps:

```julia
@testset "ACdmap_slice2d_dist composes into ACmap" begin
    N1 = N2 = 2; χ, D = 16, 3
    ACd, FL, FR, M1, M2, _ = make_leg5(χ, D; seed=3100)
    g = slice2d_grid(N1, N2)
    ACdb=slice2d_scatter(ACd,g); FLb=slice2d_scatter(FL,g); FRb=slice2d_scatter(FR,g)
    mid_blk = ACdmap_slice2d_dist(ACdb, FLb, FRb, (M1,M2), g)   # [a,b,c,d] block
    # the {a,d} output block convention matches ACmap's AC input {a,d} → feed in
    out = slice2d_gather(ACmap_slice2d_dist(mid_blk, FLb, FRb, (M1,M2), g), g)
    ref_mid = ACdmap(ACd, FL, FR, (M1,M2))
    @test out ≈ ACmap(ref_mid, FL, FR, (M1,M2)) rtol = 1e-11
end
```

**Commit:** `test: ACdmap_slice2d_dist composes into ACmap (iterate analog)`

### Task D5: ACdmap chunk-leg + memory-bound audit

**Files:** review `slice2d.jl` ACdmap — confirm the inline comment block
states the chunk-leg difference from ACmap (i+d not l, accumulate-i/assign-d),
the `n_i=n_d=N·⌈√forloop_iter⌉` derivation (§6.2), and the achieved bound
`χ²D⁴/(P·forloop_iter)` (risk 1,3). Confirm `chain_backward` recompute matches
the forward chunk. No behavior change. **Commit:**
`chore: ACdmap_slice2d_dist chunk-leg + memory-bound audit (risks 1,3)`

---

## Batch E — Sofia 4-GPU validation (3 tasks; cluster run gated separately)

A 4-GPU (2×2) Sofia submit + driver validating all four maps' distributed
parity + a memory check (esp. ACdmap's bounded intermediates). The implementer
builds the driver + submit script; the cluster run is gated separately (handled
by the user / `/hpc` flow). Clone the existing Sofia slice2d pattern
(`examples/MPI_parallel/test_slice2d_sofia.jl` +
`examples/MPI_parallel/Sofia/submit_test_slice2d.sh`; see
`examples/MPI_parallel/benchmarks/Sofia_VUB_H200.md` Part 5).

### Task E1: Sofia M3 validation driver

**Files:** create `examples/MPI_parallel/test_slice2d_m3_sofia.jl` (clone
`test_slice2d_sofia.jl`).

For each of Cmap / FRmap / ACmap / ACdmap (FRmap/AC/ACdmap square 2×2; Cmap any
grid) run the distributed map on `CuArray`s at production-ish cells (`χ` from
`TENET_SLICE2D_CHI` default 400, `D` from `TENET_SLICE2D_D` default 10), compare
forward (rel err ≤ 1e-10) and gradient (max rel err ≤ 1e-8, allreduced over
ranks) against the serial `Cmap`/`FRmap`/`ACmap`/`ACdmap` via `Zygote.pullback`,
with the `mem_line` / `timed` helpers verbatim from `test_slice2d_sofia.jl:17-31`.
Special memory check: print `mem_line` immediately after each **FRmap and
ACdmap** forward+bwd — both carry full-`i`×full-`d` chain intermediates and rely
on the 2-level chunk to stay bounded (NOT a χ×χ plane). Set `forloop_iter` so
`n_d=n_i ≥ 2N` (FRmap) / `n_i=n_d ≥ 2N` (ACdmap) and watch the peak vs ACmap
(single l-chunk). Emit a per-map `=== RESULT: PASS/FAIL ===` line like the FLmap
driver.

> Use `inner_etype` only if the serial reference also uses it; default to the
> ComplexF64 full-precision path matching the CPU parity tests. The driver runs
> the SAME `*_slice2d_dist` code the tests exercise — the GPU is the only new
> variable.

**Step — build only; do NOT run on cluster (gated).** **Commit:**
`feat: Sofia 4-GPU M3 validation driver (C/FR/AC/ACd parity + FRmap/ACdmap mem)`

### Task E2: Sofia M3 submit script

**Files:** create `examples/MPI_parallel/Sofia/submit_test_slice2d_m3.sh` (clone
`submit_test_slice2d.sh`). Same module loads, `WD`, `CLEAN_LD`, `BASE_ENVS`,
`mpirun -np 4` invocation; point the driver at `../test_slice2d_m3_sofia.jl`;
bump `--time` to `00:45:00` (four maps + ACdmap mem probe). Keep
`--gres=gpu:nvidia_h200:4`, `--ntasks-per-node=4`, the LD_PRELOAD libcuda fix
and `CUDA_LAUNCH_BLOCKING=1` (MEMORY: Sofia MPI+CUDA.jl needs both).

**Step — build only.** **Commit:** `feat: Sofia submit script for M3 4-GPU validation`

### Task E3: Sofia run record placeholder (Part 9)

**Files:** none until the run lands. After the gated cluster run, record the
parity table + PASS/FAIL + the FRmap/ACdmap memory peak (vs ACmap, confirming
bounded full-`i`×full-`d` intermediates) as **Part 9** in
`examples/MPI_parallel/benchmarks/Sofia_VUB_H200.md`, following the Part 5/7
format. **This task is the cluster run + write-up — do NOT mark M3 done until
it lands.** **Commit (after run):** `docs: Part 9 — M3 slice2d-wrapper 4-GPU Sofia validation`

---

## Done criteria for M3

- `Cmap_slice2d`, `FRmap_slice2d_dist`, `ACmap_slice2d_dist`,
  `ACdmap_slice2d_dist` each have a forward + a Zygote-free `rrule`.
- `julia --project=. test/run_test_slice2d_m3.jl` green: every map's 4-rank
  (2×2) distributed-parity gate passes at **forward rtol 1e-12 / gradient rtol
  1e-10**. Cmap is also green on `(1,4)`/`(4,1)` (no square assertion);
  FRmap/AC/ACdmap assert the square grid and `@test_skip` the rectangular case.
- The diagonal-trap off-diagonal assertions (FRmap `(a,i)` plane, AC `(i,l)`
  plane, ACd `(a,d)` plane, χ=18 N=2) pass.
- FRmap and ACdmap 2-level chunk (FRmap d/i, ACdmap i/d) and ACmap single
  l-chunk all parity-pass at `forloop_iter ∈ (1,3)` including the clamp; the
  accumulate/assign asymmetry is proven by the `forloop_iter=3` + off-diagonal
  combination.
- Single-M (`dM = dM1 + conj(dM2)`), bare-sum densify, and
  `inner_etype=Float32` paths pass for every map (Cmap has no `inner_etype`
  kwarg → only single-M/densify apply there).
- The ACdmap (and FRmap) output middle-leg dims are `size(M*,4)` (ACdmap) /
  `size(M*,1)` (FRmap), guarded by a non-uniform-bond (`Db≠Dc`) test so the
  uniform-D case cannot mask a wrong index.
- Local einsums route through the chain engine via the **whole-chain API**
  (`chain_apply` / `chain_backward`) for ALL FOUR maps — **no new
  `_slice2d_<map>_*` hand kernels, and NO slice2d-ring API** (FRmap is gather-class,
  not a ring).
- **NO new comm primitive** — M3 uses ONLY the four existing primitives
  (`_slice2d_col_allgather` 730, `_slice2d_row_allgather` 750,
  `_slice2d_col_reduce_scatter` 710, `_slice2d_row_reduce_scatter_last` 760) and
  their documented adjoint partners. No `_slice2d_col_shift`, no col-twin
  reduce-scatter, no `_slice2d_block_transpose`.
- Cmap's FL/FR gradients are **take-my-block** (`getindex` slice), NOT
  reduce-scatter (replicated output → take-my-block adjoint); `dC` is replicated
  with no allreduce. (Contrast FLmap_slice2d_dist, whose distributed output makes
  reduce-scatter the correct adjoint there.)
- NO Zygote inside any rrule backward; eager `_free!` discipline matches the
  FLmap rrule; captures are bounded slices (χ²D²/N), never a χ²D⁴ array or a
  χ×χ plane.
- `julia --project=. test/run_test_slice2d.jl`, `test/test_chain_engine.jl`,
  `test/test_chain_maps.jl` stay green throughout (M3 is additive).
- Batch E driver + submit built; the Sofia 4-GPU run (Part 9) lands with
  PASS and the ACdmap/FRmap bounded-memory confirmation (the FLmap OOM lesson) —
  M3 is NOT done until Part 9 records PASS.

## Out of scope (explicit)

- **Rectangular grids for FRmap/AC/ACdmap** (`N1≠N2`) — M3 v2; needs the real
  `_slice2d_block_transpose` (tag 770, design doc §3, risk 6). The square
  assertion is the gate. (Only Cmap is rectangular-safe in v1.)
- **Distributed leg4/leg8 variants** of the gather-class maps — only leg5 (tuple
  + single M) is wired (matches the FLmap_slice2d_dist scope and the AD census).
- Hoisting the AC/ACdmap gathers out of a fixed-FL/FR iteration loop (the
  natural ACenv-loop optimization, design doc §6.3) — recorded, not built.
- Wiring the new maps into the VUMPS/forloop production paths — M3 delivers the
  map + rrule + parity gate; production integration is a follow-up milestone.
- A free-leg scatter/allgather primitive (design doc §3) and
  `_slice2d_block_transpose` (tag 770) — proven unnecessary for M3 v1; reserved.
