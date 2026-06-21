# 2D Distributed VUMPS Runtime — Design

**Date**: 2026-05-11
**Status**: Brainstorming complete, awaiting implementation
**Predecessor**: PR #42 (1D FR-only distribution, closed in favor of this comprehensive approach)
**Author**: Brainstormed with Claude
**Target branch**: TBD (to be created as `feat/2d-distributed-runtime`)

---

## Executive Summary

Replace PR #42-style "FR-only 1D distribution" with a **full 2D block distribution** of all 4-leg tensors (FL, FR, AL, AR, AC). Each tensor stored as `(χ/N1, D, D, χ/N2)` on each rank, with total `N = N1 × N2` ranks in a 2D Cartesian grid.

| Metric | Baseline (single GPU) | PR #42 (1D FR only) | This design (2D, N=8) |
|--------|----------------------|---------------------|----------------------|
| Persistent memory per rank | 5 × 1.2 GB × Nsites | ~750 MB × Nsites | ~750 MB × Nsites |
| Peak transient (in-map) | ~6 GB | ~5 GB | **~1.3 GB** |
| Communication per VUMPS step | 0 GB | ~2 GB | **~36 GB** |
| Max χ at fixed peak budget | ~540 (Kagome D=10 OOM data) | ~540 | **~950** |

**Key insight**: The OOM mode (Sofia 1003757, χ=540 forward) is **in-iteration peak memory**, not persistent. PR #42's distribute-then-gather-per-step pattern doesn't help peak. Full 2D distribution does.

---

## Motivation

Production evidence (`docs/data-inventory.md` 1.25 Kagome):

- **Sofia 1003757**: D=10 Kagome reached χ=532 then OOM at χ=540 forward direction on H200 (140 GB maxed). Working memory dominates over persistent.
- **Sofia 1005525**: D=12 Kagome worked around 1003757 by setting `total_splits=64` (= `forloop_iter × nprocs`), reducing per-iteration working set. This confirms the OOM is **inside a single map call's working set**, not between steps.

PR #42 stored FR distributed but allgathered to full per `vumps_step` — peak memory unchanged. Same problem would persist if we extended to all 5 tensors with in-step gather. **Only fully-distributed compute (in-map) addresses the OOM mode.**

---

## Goals

- **G1**: Every 4-leg tensor stored as `1/N` of full per rank, persistently.
- **G2**: Map computations (FLmap, FRmap, ACmap, Cmap) **never hold full `χ²·D²` tensor on any rank**. Peak slab size ≤ `χ²·D²/min(N1,N2)`.
- **G3**: Max χ scales as `N^{1/4}` for fixed per-rank peak (vs `N^0` for 1D in-map allgather).
- **G4**: Gradient parity with serial / PR #42 to ~`1e-8` (matching prior PR verification standard).

## Non-goals (defer to v2)

- Splitting D legs (cost-benefit analysis below).
- Distributed SVD/QR. V1 uses allgather + local QR for `ACCtoAL` / `ACCtoAR`.
- KrylovKit `eigsolve` integration. V1 uses `simple_eig` (power iteration) only, with distributed `dot`/`norm`.
- CPU / NVMe offload integration. Orthogonal — can stack with 2D in a later PR.
- Compute–communication overlap via async streams (SUMMA pattern). Performance optimization for v2.

---

## 1. Grid & Partition Convention

### 1.1 MPI 2D Cartesian Grid

```
N = N1 × N2 total ranks
Each rank: coordinates (r1, r2) with r1 ∈ [0, N1), r2 ∈ [0, N2)

3 communicators (built via MPI.Cart_create + MPI.Cart_sub):
  comm_world: all N ranks
  row_comm[r1]: N2 ranks with same r1 (varies r2)
  col_comm[r2]: N1 ranks with same r2 (varies r1)
```

(N1, N2) configured via `VUMPS` alg fields. Default: `N1 = floor(√N), N2 = N ÷ N1`. For non-perfect-square N (e.g., N=8), rectangular grid (2×4) is supported but more complex; for v1 prefer N1=N2 when possible.

### 1.2 Tensor Partition

Every 4-leg tensor `T[α, β, γ, δ]` with shape `χ × D × D × χ` stored as:

- **First χ leg (α) partitioned on N1** → rank `(r1, *)` holds `α_slice_r1`
- **Last χ leg (δ) partitioned on N2** → rank `(*, r2)` holds `δ_slice_r2`

Per-rank local shape: `(χ/N1, D, D, χ/N2)`. **Memory: `χ²·D²/N` per rank per tensor.**

| Tensor | Label form | first χ (N1) | last χ (N2) |
|--------|-----------|--------------|--------------|
| `AL[a, b, c, d]` | left χ, D, D, right χ | a | d |
| `AR[a, b, c, d]` | same | a | d |
| `AC[a, b, c, d]` | same | a | d |
| `FL[a, e, f, i]` | up-left χ, D, D, down-left χ | a | i |
| `FR[d, g, h, l]` | up-right χ, D, D, down-right χ | d | l |

When `AL` is used as `ALd = conj(AL)` (relabeled `[i, j, k, l]`): `ALd.i = AL.a` (on N1), `ALd.l = AL.d` (on N2). Same partition convention applies. **`ALu` and `ALd` may be physically different tensors** (e.g., `vumps_step_power` uses ALu=AL, ALd=conj(ALp); ObsEnv uses ALu/ALd from different runtimes); each follows the convention independently.

### 1.3 Special tensors

- **C (χ × χ)**: full on every rank. Small (~5 MB at χ=768), simplifies `ALCtoAC` and downstream consumers.
- **M (D^k)**: full on every rank. Small (D⁵ ≈ 1 MB at D=16).
- **Pattern matrices in StructArrays**: unit cell structure, orthogonal to rank grid. Unchanged.

### 1.4 Data structure

No new tensor wrapper type. `VUMPSRuntime`'s `StructArray.data::Vector{Array{T,4}}` elements just have smaller shape `(χ/N1, D, D, χ/N2)`. Grid metadata (N1, N2, sub-comms) lives in `alg` struct, threaded through function calls.

### 1.5 Memory accounting (D=16, χ=768, leg5, F64)

Each `χ²·D² ≈ 1.2 GB`. Per-rank persistent for 5 tensors:

| N | (N1, N2) | per tensor | 5-tensor total | C + M overhead | Total |
|---|---------|-----------|---------------|---------------|-------|
| 8 | (2, 4) | 150 MB | 750 MB | ~10 MB | **~760 MB** |
| 16 | (4, 4) | 75 MB | 375 MB | ~10 MB | **~385 MB** |
| 64 | (8, 8) | 19 MB | 94 MB | ~10 MB | **~104 MB** |

### 1.6 Peak memory in maps

Max transient slab during a map call: `≤ χ²·D² / min(N1, N2)`. For ACmap at N=8 N1=2 N2=4:

- Persistent: 760 MB
- `AC_gathered` slab along N2: ~110 MB
- AllReduce intermediate: ~150 MB
- `result_local` (output): ~150 MB
- Peak: **~1.3 GB per rank** (vs ~5 GB baseline for 1D in-map allgather).

### 1.7 N1:N2 tuning

This knob controls **communication pattern shape**, not compute volume (FLOPs are invariant). Trade-offs:

- `(1, N)` = pure 1D last χ (degenerates to PR #42 style)
- `(N, 1)` = pure 1D first χ
- `(√N, √N)` = balanced 2D (default)
- Rectangular `(2, 4)` etc.: imbalanced slab sizes between row/col comm but still valid

---

## 2. Per-map Dataflow

All 4 maps share the same 4-step pattern: **AllGather (cross-axis dim) → local einsum → AllReduce (co-distributed dim) → AllToAll (cross-axis free leg transpose)**.

### 2.1 FLmap (leg5)

```
result[d, g, h, l] := FL[a, e, f, i] * ALd[i, j, k, l] * M1 * M2 * ALu[a, b, c, d]
```

| Leg | Role | Distribution | Status |
|-----|------|--------------|--------|
| `a` | contract (FL ↔ ALu) | FL.a=N1, ALu.a=N1 | **co-distributed on N1** |
| `i` | contract (FL ↔ ALd) | FL.i=N2, ALd.i=N1 | **cross-axis** |
| `d` | free (ALu → result) | ALu.d=N2, result.d=N1 | **cross-axis** |
| `l` | free (ALd → result) | ALd.l=N2, result.l=N2 | aligned |

**Forward**:
1. `ALd_gathered = AllGather(ALd, dim=i, col_comm)` (~75 MB/rank)
2. `partial = einsum(FL, ALd_gathered, M, ALu)` — local, output `[d_slice_r2, g, h, l_slice_r2]`
3. `partial_full_a = AllReduce(partial, col_comm)` — sum a contributions (~150 MB)
4. `result = AllToAll(partial_full_a, d:N2→N1, row_comm)` (~75 MB)

**Total forward comm: ~300 MB/rank.**

**Backward** = composition of each step's rrule (Section 3).

### 2.2 FRmap (leg5)

```
result[a, e, f, i] := ARd[i, j, k, l] * FR[d, g, h, l] * M1 * M2 * ARu[a, b, c, d]
```

| Leg | Role | Distribution | Status |
|-----|------|--------------|--------|
| `l` | contract (FR ↔ ARd) | FR.l=N2, ARd.l=N2 | **co-distributed on N2** |
| `d` | contract (FR ↔ ARu) | FR.d=N1, ARu.d=N2 | **cross-axis** |
| `a` | free (ARu → result) | ARu.a=N1, result.a=N1 | aligned |
| `i` | free (ARd → result) | ARd.i=N1, result.i=N2 | **cross-axis** |

**Forward** (mirror of FLmap):
1. `FR_gathered = AllGather(FR, dim=d, col_comm)` (~75 MB)
2. `partial = einsum(ARd, FR_gathered, M, ARu)` — output `[a_slice_r1, e, f, i_slice_r1]`
3. `partial_full = AllReduce(partial, row_comm)` — sum l + d (~150 MB)
4. `result = AllToAll(partial_full, i:N1→N2, col_comm)` (~75 MB)

**Total: ~300 MB/rank.**

### 2.3 ACmap (leg5)

```
result[i, j, k, l] := AC[a, b, c, d] * FR[d, g, h, l] * M * FL[a, e, f, i]
```

| Leg | Role | Distribution | Status |
|-----|------|--------------|--------|
| `a` | contract (FL ↔ AC) | FL.a=N1, AC.a=N1 | **co-distributed on N1** |
| `d` | contract (AC ↔ FR) | AC.d=N2, FR.d=N1 | **cross-axis** |
| `i` | free (FL → result) | FL.i=N2, result.i=N1 | **cross-axis** |
| `l` | free (FR → result) | FR.l=N2, result.l=N2 | aligned |

**Forward**:
1. `AC_gathered = AllGather(AC, dim=d, row_comm)` (~110 MB)
2. `partial = einsum(AC_gathered, FR, M, FL)` — output `[i_slice_r2, j, k, l_slice_r2]`
3. `partial_full = AllReduce(partial, col_comm)` — sum a + d (~150 MB)
4. `result = AllToAll(partial_full, i:N2→N1, row_comm)` (~75 MB)

**Total: ~335 MB/rank.**

### 2.4 Cmap (leg4 example)

```
result[e, f] := FL[a, c, d, e] * C[a, b] * FR[b, c, d, f]
```

C is **full on each rank** (per Section 1.3). FL, FR are 2D.

**Forward**:
1. `FL_gathered = AllGather(FL, dim=e, row_comm)` (~110 MB)
2. `FR_gathered = AllGather(FR, dim=f, row_comm)` (~110 MB)
3. `partial = einsum(FL_gathered, C, FR_gathered)` — output full `[e, f]` (partial in a, b)
4. `result = AllReduce(partial, COMM_WORLD)` — small (~10 MB, since C is χ² = 5 MB)

**Total: ~230 MB/rank.**

### 2.5 Per-step communication summary

| Map | Comm per call | Peak slab |
|-----|---------------|-----------|
| FLmap | ~300 MB | ~300 MB |
| FRmap | ~300 MB | ~300 MB |
| ACmap | ~335 MB | ~450 MB |
| Cmap | ~230 MB | ~250 MB |

**Per vumps_step** (assume ~30 simple_eig power iter per env, 4 envs):  
- Total communication: ~36 GB / step
- Peak per rank: ~1.3 GB (persistent + max slab)

**vs PR #42**: ~180 GB/step communication, ~6 GB peak. **5× comm improvement, ~4× peak improvement.**

NCCL @ 50 GB/s on H200 ⇒ ~0.7 s/step pure communication (5% of typical 15 s vumps_step).

---

## 3. Communication Primitives + AD rrules

### 3.1 Four primitives

| Name | Op | MPI backend |
|------|-----|-----|
| `allgather_dim(t, dim, comm)` | Gather distributed dim to full | `MPI.Allgatherv!` (existing `allgatherv_p2p!`) |
| `reduce_scatter_dim(t, dim, comm)` | Inverse: sum + scatter | `MPI.Reduce_scatter!` |
| `allreduce_dim(t, op, comm)` | Sum across ranks (shape preserved) | `MPI.Allreduce!` (existing `allreduce_p2p!`) |
| `alltoall_dim_swap(t, src_dim, src_comm, dst_dim, dst_comm)` | Redistribute dim from src_comm axis to dst_comm axis | `MPI.Alltoallv!` |

### 3.2 AD rrules (all standard)

| Forward | Backward |
|---------|----------|
| `allgather_dim` | `reduce_scatter_dim` (same dim, same comm) |
| `reduce_scatter_dim` | `allgather_dim` |
| `allreduce_dim(+)` | `allreduce_dim(+)` (self-adjoint) |
| `alltoall_dim_swap(src→dst)` | `alltoall_dim_swap(dst→src)` |

```julia
function ChainRulesCore.rrule(::typeof(allgather_dim), tensor_local, dim, comm)
    result = allgather_dim(tensor_local, dim, comm)
    function back(d_result)
        d_local = reduce_scatter_dim(unthunk(d_result), dim, comm)
        return NoTangent(), d_local, NoTangent(), NoTangent()
    end
    return result, back
end
# ... similar for other 3 ...
```

### 3.3 `alltoall_dim_swap` — the hard one

This is the only non-trivial primitive. Implementation outline:

```julia
function alltoall_dim_swap(tensor, src_dim, src_comm, dst_dim, dst_comm)
    # Precondition: src_comm and dst_comm are orthogonal axes of 2D Cart grid
    N_src = MPI.Comm_size(src_comm)
    N_dst = MPI.Comm_size(dst_comm)
    
    chi_src_local = size(tensor, src_dim)       # χ/N_src
    chi_full = N_src * chi_src_local            # χ
    chi_dst_local = chi_full ÷ N_dst            # χ/N_dst
    
    # Split tensor along src_dim into N_dst chunks (by destination dst_dim slice)
    # Use MPI.Alltoallv! on src_comm with chunked sendbuf
    # Receive N_src chunks, concatenate along src_dim into output of size (..., chi_dst_local, ...)
    ...
end
```

**Edge cases requiring care**:
1. **χ not divisible by N1 or N2** — use existing `split_ranges` helper for uneven slices; pass per-rank counts to `Alltoallv`.
2. **N1 ≠ N2 (rectangular grid)** — `Alltoallv` with non-uniform counts and displacements.

Estimated ~1.5 weeks for `alltoall_dim_swap` alone, including edge cases.

### 3.4 Struct-level wrappers

```julia
prescatter_2d_struct(sa::StructArray, grid) -> StructArray  # full → 2D
allgather_2d_struct(sa::StructArray, grid) -> StructArray   # 2D → full
```

Both apply to each unique site in the unit cell pattern.

### 3.5 Distributed linear algebra (for `simple_eig`)

```julia
distributed_dot(x::StructArray, y::StructArray, comm) -> scalar  # local dot + AllReduce
distributed_norm(x::StructArray, comm) -> real scalar
```

`simple_eig` gains kwargs `inner_product` and `norm_fn`, defaulting to non-distributed. The `_simple_eig_*map` wrappers pass the distributed versions when in 2D mode.

### 3.6 Code volume estimate

- Primitives: ~400 lines (4 functions, mostly straightforward)
- `alltoall_dim_swap`: ~150 lines (alone)
- rrules: ~100 lines
- Struct wrappers: ~100 lines
- distributed_dot / distributed_norm + simple_eig adaption: ~80 lines
- Grid setup utilities: ~70 lines
- **Total Phase 1 code: ~900 lines**

---

## 4. Integration with Existing Code

### 4.1 `VUMPSRuntime` struct

Unchanged. Inner `Array` shapes shrink from `(χ, D, D, χ)` to `(χ/N1, D, D, χ/N2)`. `pattern` matrix unchanged.

### 4.2 `init_VUMPSRuntime`

```julia
function init_VUMPSRuntime(M, χ, alg)
    # Rank 0 generates initial A (small, χ_init = D²), bcast
    A = initial_A(M, χ)
    AL_full, L, _ = left_canonical(A)
    R, AR_full, _ = right_canonical(AL_full)
    C_full = LRtoC(L, R)
    
    AL_full = MPI.bcast(AL_full, 0, COMM_WORLD)
    AR_full = MPI.bcast(AR_full, 0, COMM_WORLD)
    C_full = MPI.bcast(C_full, 0, COMM_WORLD)
    
    AL = prescatter_2d_struct(AL_full, alg.grid)
    AR = prescatter_2d_struct(AR_full, alg.grid)
    # C stays full
    
    _, FL = leftenv(AL, conj(AL), M; alg)   # produces 2D FL
    _, FR = rightenv(AR, conj(AR), M; alg)  # produces 2D FR
    
    return VUMPSRuntime(AL, AR, C_full, FL, FR)
end
```

### 4.3 `vumps_step` — cleaner than PR #42

```julia
function vumps_step(rt, M, alg)
    @unpack AL, C, AR, FL, FR = rt
    
    AC = ALCtoAC(AL, C)                          # 2D
    _, FL = leftenv(AL, conj(AL), M, FL; alg)    # 2D in / 2D out
    _, FR = rightenv(AR, conj(AR), M, FR; alg)   # 2D in / 2D out
    _, C  = Cenv(C, FL, FR; alg)                 # C stays full
    _, AC = ACenv(AC, FL, M, FR; alg)            # 2D in / 2D out
    
    AL, AR, errL, errR = ACCtoALAR(AC, C)        # internal allgather AC for QR
    err = errL + errR
    
    return VUMPSRuntime(AL, AR, C, FL, FR), err
end
```

No allgather/prescatter cycles per step (vs PR #42). AD tape simpler.

### 4.4 `ACCtoALAR` (QR via local — v1)

QR is `qrpos(_to_front(AC[i]))` in current `ACCtoAL`. For v1: allgather AC, run existing QR, prescatter result.

```julia
function ACCtoAL_2d(AC, C, grid)
    AC_full = allgather_2d_struct(AC, grid)        # ~1.2 GB transient per site
    AL_full, errL = ACCtoAL(AC_full, C)             # reuse existing serial QR
    AL = prescatter_2d_struct(AL_full, grid)
    return AL, errL
end
```

Per `vumps_step`: 2 × allgather (≈3 GB total comm) + 2 × cuSOLVER QR (~10 ms each) + 2 × prescatter. **One transient peak of ~1.2 GB during QR call.** Acceptable for v1.

**v2 path**: Tall-Skinny QR (TSQR) distributed across N1 axis, since matrix view of AC is `(χ·D²/N1, χ)` — extremely tall-skinny, perfect for TSQR. Defer until D ≥ 20 forces it.

### 4.5 `ObsEnv` — fully 2D

```julia
function ObsEnv(rt, M, alg, Fo=[rt.FL, rt.FR])
    @unpack AL, AR, C, FL, FR = rt
    # No allgather; FL, FR stay 2D same as vumps_step
    AC = ALCtoAC(AL, C)
    _, FLo = leftenv(AL, AL, M, Fo[1]; ifobs=true, alg)
    _, FRo = rightenv(AR, AR, M, Fo[2]; ifobs=true, alg)
    return VUMPSEnv(AC, AR, AC, AR, FL, FR, FLo, FRo)
end
```

Downstream observable kernels handle 2D allgather on-demand (or defer their own redesign).

### 4.6 `simple_eig` modification

```julia
function simple_eig(f, v; power_iter, 
                    inner_product = LinearAlgebra.dot,
                    norm_fn = LinearAlgebra.norm,
                    kwargs...)
    # ... existing logic, but replace `dot(v1, v2)` → `inner_product(v1, v2)`
    #     and `norm(v)` → `norm_fn(v)`
end
```

In 2D mode, `_simple_eig_*map` wrappers pass `inner_product=distributed_dot(grid)`, `norm_fn=distributed_norm(grid)`.

### 4.7 `general.jl` change summary

| Function | Delta | Note |
|----------|-------|------|
| `vumps_step` | -10 lines | Remove PR #42 allgather/prescatter cycle |
| `vumps_step_power` | -10 lines | Same |
| `init_VUMPSRuntime` | +20 lines | bcast + prescatter_2d |
| `leftenv` / `rightenv` | ±0 lines | Inner map call replaced by 2D version |
| `Cenv` | ±0 lines | Same |
| `ACenv` | -30 lines | Remove `fr_distributed` branch |
| `ALCtoAC` | +30 lines | 2D version (partial+allreduce on row_comm) |
| `ACCtoAL` / `ACCtoAR` | +30 lines | v1 wrapper (allgather → QR → prescatter) |
| `ObsEnv` | +5 lines | Mostly unchanged |
| `simple_eig` | +20 lines | Add `inner_product` / `norm_fn` kwargs |
| `_simple_eig_*` (4 fns) | +5 lines each | Thread grid-aware dot/norm |
| `distributed_dot` / `distributed_norm` | +60 lines (new file) | — |

**Total Phase 4 code: ~+200 lines, ~50 modified.**

---

## 5. Implementation Phases

| Phase | Name | Effort | Key deliverable | Milestone signal |
|-------|------|--------|-----------------|------------------|
| 0 | Sandbox sanity | 0.5 wk | Standalone primitive test on 4 GPU | Design risks surfaced early |
| 1 | Primitives + rrules | 3 wk | 4 primitives, gradcheck passing | AD layer correct |
| 2 | FLmap end-to-end | 1.5 wk | FLmap_2D + leftenv + gradcheck | **"Distributed converges"** |
| 3 | Remaining 3 maps | 2 wk | FRmap, ACmap, Cmap | All maps tested |
| 4 | vumps_step integration | 2 wk | Full VUMPS run | **"VUMPS correct"** |
| 5 | LBFGS gradient parity | 2 wk | ∂E/∂M parity 1e-8 | **"AD chain through"** |
| 6 | Sofia production benchmark | 1 wk | D=10 χ=1000+, perf report | Ready to push χ |
| 7 | v2 polish (optional) | 4 wk | TSQR / SUMMA overlap / etc. | Triggered only if bottleneck found |

**Total v1: ~12 weeks single-developer (compress significantly if collaborator joins).**

### Parallelization opportunity

With 2 developers:
- Dev A: Phase 1 primitives + AD (independent infrastructure)
- Dev B: Phase 4 high-level VUMPS rewrite using mock primitives (can start in parallel after Phase 0)

Merge at Phase 2/3. **Compressed timeline: ~8 weeks.**

### Bail-out / Reroute decision points

| Trigger | Action |
|---------|--------|
| Phase 1: `alltoall_dim_swap` rectangular grid implementation gets stuck | Restrict to N1=N2 perfect square for v1 |
| Phase 1: gradcheck fails on a primitive | Revisit rrule math before continuing |
| Phase 5: gradient parity only reaches 1e-4 | Pause production; debug AD subtleties (most likely simple_eig + distributed_dot interaction) |
| Phase 6: 2D doesn't reduce peak memory | Investigate Krylov queue / AD tape (may be the actual culprit, not tensor distribution) |
| Phase 6: 2D too slow | Phase 7 SUMMA overlap |

---

## 6. Testing Strategy

7-level test pyramid:

| Level | Test | Tooling | Phase |
|-------|------|---------|-------|
| 1 | Primitive forward correctness | Unit test, 4-16 GPU | 1 |
| 2 | Primitive gradcheck | Zygote vs finite diff | 1 |
| 3 | Map-level forward + backward parity | vs serial, 1e-8 | 2-3 |
| 4 | Env-level (leftenv etc.) convergence | Eigenvalue + eigenvector match | 3-4 |
| 5 | `vumps_step` convergence trajectory | Fix-point match | 4 |
| 6 | End-to-end LBFGS gradient parity | ∂E/∂M, 1e-8 | 5 |
| 7 | Production benchmark + edge cases | Sofia 8/16 GPU | 6 |

### Phase 5 risk

The Level 6 gradient parity test is the highest-risk validation. Order of debugging if it fails:

1. `simple_eig` rrule under distributed dot/norm
2. Primitive rrule `unthunk` timing
3. ACCtoAL allgather + QR + prescatter rrule chain
4. KrylovKit path accidentally taken (verify `ifsimple_eig=true` honored)
5. F32/F64 precision drift in `inner_etype` cast

Reserve **1.5 weeks** specifically for this debug.

### Edge case tests

- N1=1 (degenerate 1D last χ): must match PR #42 behavior
- N=1: must match pure serial
- χ not divisible by N1 or N2: split_ranges handles uneven
- Multi-site unit cells (Plaquette 4, Kagome 3)
- Different leg structures (leg3, leg4, leg5, leg8)

### CI configuration

```yaml
# .github/workflows/distributed_2d.yml
phases_1_2: Sofia 4 GPU, ~10 min
phases_3_4: Sofia 4-8 GPU, ~1 hr
phase_5: Sofia 8 GPU, ~2 hr (gradient parity)
phase_6: Sofia 8 / 16 GPU, sbatch (production timing)
```

Regression detection: if new commit slows baseline `D=10 χ=540 N=8` benchmark by >20% or grows peak >10%, alert.

---

## 7. Future Work (v2 / v3)

| Optimization | Trigger | Effort |
|-------------|---------|--------|
| Distributed TSQR for ACCtoAL/AR | D ≥ 20 | ~3 weeks |
| SUMMA / Slice2D's compute-comm overlap | Per-step time map-dominated | ~2 weeks |
| Block-cyclic 2D partition | Load imbalance > 20% | ~1 week |
| F32/F16 in alltoall messages | Message size bottleneck | ~1 week |
| D leg slicing | D ≥ 24 (4-D partition grid) | ~6 weeks (research) |
| KrylovKit eigsolve via custom inner | Quasi-Newton fails | ~2 weeks |
| 2.5D / 3D Solomonik partition | N ≥ 64, comm-bound | ~research, deprioritized |

### Stacking with NVMe offload

NVMe offload (`docs/2026-MM-DD-nvme-offload-design.md`, TBD) is **orthogonal** to 2D. After 2D port:
- 2D shrinks persistent to ~750 MB / rank
- NVMe offload can move that to NVMe between `vumps_step` calls
- Combined: near-zero GPU persistent, all memory budget for transient working set
- Combined max χ scaling: ~2× over 2D alone

### Stacking with simpler eigensolvers

KrylovKit default keeps ~30 Krylov vectors. Switching to LOBPCG (3 blocks) or just `simple_eig` (power, 2 vectors) shrinks queue 10×. **Simple_eig is v1's default anyway**; LOBPCG could be Phase 7 addition for accuracy gains.

---

## Appendix A: Math notation conventions

### Tensor leg labels (leg5 case, double-layer iPEPS)

```
FL[a, e, f, i]:   a, i are χ; e, f are D
FR[d, g, h, l]:   d, l are χ; g, h are D
AL[a, b, c, d]:   a, d are χ; b, c are D
AR[a, b, c, d]:   a, d are χ; b, c are D
AC[a, b, c, d]:   a, d are χ; b, c are D
M1[e, j, g, b, p], M2[f, k, h, c, p]:   all D
```

The "first χ" of each tensor is always the leftmost χ leg in the label; "last χ" is rightmost. Within FLmap and friends, the actual contraction structure determines which dimensions match.

### Communication direction conventions

- **row_comm[r1]**: contains ranks `(r1, r2)` for all `r2` ∈ [0, N2)
- **col_comm[r2]**: contains ranks `(r1, r2)` for all `r1` ∈ [0, N1)

Mnemonic: "row" = horizontal slice of the grid; "col" = vertical slice.

---

## Appendix B: Bibliography

- Hong & Kung 1981 — I/O complexity lower bound for matrix multiplication
- Loomis & Whitney 1949 — surface-volume inequality (foundation for comm lower bounds)
- Solomonik & Demmel 2011 — Communication-Optimal Parallel 2.5D Matrix Multiplication
- Slice2D 1969 — A cellular computer to implement the Kalman Filter Algorithm
- van de Geijn & Watts 1997 — SUMMA: scalable universal matrix multiplication
- Demmel et al. 2012 — Communication-avoiding QR (TSQR / CAQR)
- Kossaifi et al. 2019 — Tensor decompositions and applications (review)

---

## Appendix C: Predecessor PR #42 — what we kept and what we changed

**Kept from PR #42**:
- The `prescatter_for_parallel` / `allgather_for_parallel` AD rrule pattern (generalized to 4 primitives in this design).
- `_allgather_FR` / `_prescatter_FR` StructArray-level helpers (generalized to `prescatter_2d_struct`).
- Design doc structure and verification approach (gradient parity 1e-8).

**Changed from PR #42**:
- All 5 tensors distributed, not just FR.
- 2D grid, not 1D.
- No allgather/prescatter cycles per vumps_step (in-map distributed compute, not in-step gather).
- `ACCtoAL` / `ACCtoAR` allgather AC for QR (PR #42 didn't touch this).
- C tensor kept full (PR #42 kept it full too, consistent).

**Why PR #42 was closed**: Distributing only FR provided ~10-15% memory savings persistent, but didn't address peak memory (the actual OOM cause on Sofia 1003757). This design addresses peak directly.
