# Cannon-Style 2D Distributed FLmap — Design

**Date**: 2026-06-10
**Status**: Design approved, awaiting implementation plan
**Scope**: Map level only (FLmap forward + backward + Sofia 4-GPU validation)
**Relation to prior work**: Narrows the per-map dataflow of
`docs/2026-05-11-2d-distributed-vumps-runtime-design.md` (AllGather+AllReduce+AllToAll)
to a Cannon-style two-stage ring. Partition convention and the rrule adjoint table
carry over from that document; its grid/struct-level sections remain valid for the
later leftenv/vumps integration rounds.

---

## Motivation

The current `parallel()` slice scheme (`src/contraction/forloop_parallel_MPI.jl`)
replicates every input tensor (FL, ALu, ALd, M) on every GPU and only splits the
*compute* along the last χ leg, reassembling the full result with `allgatherv_p2p!`.
It saves wall time, not memory: each card stores all environments.

Cannon-style distribution stores the environment in N1×N2 blocks across ranks.
No rank ever materializes the full FL or the full serial contraction intermediate.
Target deployment is 8×8 (64 GPUs) up to 12×12 (144 GPUs) across nodes; the 2×2
Sofia 4-GPU configuration in this round is a fast-iteration vehicle, so the design
is judged on its large-P communication scaling, not on 2×2 numbers.

### Why two stages (and not a one-shot block kernel)

FLmap is a sandwich contraction — **both** χ legs (a, i) of FL are contracted:

```
result[d,g,h,l] = FL[a,e,f,i] · ALd[i,j,k,l] · M1[e,j,g,b,p] · M2[f,k,h,c,p] · ALu[a,b,c,d]
```

A one-shot scheme that circulates FL blocks and calls the existing FLmap kernel
per block pair recomputes the FL·ALd intermediate on every rank of a grid row:
(N1+N2)/2× FLOP overhead — 2× at 2×2 but **8× at 8×8**. The two-stage split below
keeps total FLOPs exactly serial/P.

Comparison of candidate inner-loop primitives (decided in brainstorming):

| Approach | Verdict |
|----------|---------|
| **A. Two-stage Cannon ring (chosen)** | Point-to-point shifts, zero new deps on existing MPI Isend/Irecv infra, NCCL p2p addable later |
| B. SUMMA broadcast variant | Same distribution/staging; swap ring shifts for row broadcasts. Revisit if profiling demands comm/compute overlap; ~95% shared infrastructure |
| C. One-shot block-kernel accumulate | Rejected: FLOP overhead scales with grid size |

---

## §1 Grid, partition convention, forward dataflow

### Grid

`CannonGrid`: N1×N2 ranks, P = N1·N2. Rank → coordinates `(r1, r2)` with
`r1 = rank ÷ N2`, `r2 = rank mod N2`. Two cached sub-communicators:

- `row_comm` — fixed r1, varies r2 (stage-1 ring)
- `col_comm` — fixed r2, varies r1 (stage-2 reduce-scatter)

`@assert nprocs == N1*N2` at construction. Uneven χ handled by the existing
`split_ranges` helper.

### Partition (leg5)

Convention (inherited from the 2026-05-11 design): **first χ leg split N1-ways,
last χ leg split N2-ways.** Rank (r1, r2) holds:

- input block `FL_blk = FL[a_range(r1), :, :, i_range(r2)]` — χ²D²/P each
- output block `result[d_range(r1), :, :, l_range(r2)]`

**Output distribution = input distribution** (d sits on a's axis, l on i's axis),
so power iteration feeds the output straight back in; the environment never
materializes in full. ALu, ALd, M stay fully replicated this round (AL
distribution belongs to the leftenv integration round).

### Forward — stage 1 (contract i; FL blocks rotate along the row; M folded ONCE after the ring)

```
H[a,e,f,j,k,l] = Σ_t  FL[a,e,f, i∈block t] · ALd[i∈block t, j,k, l∈block r2]
G[a,b,c,g,h,l] = H[a,e,f,j,k,l] · M1[e,j,g,b,p] · M2[f,k,h,c,p]      # fold, once
```

At step k = 0..N2-1, rank (r1,r2) holds FL block `t_k = mod(r2+k, N2)`, contracts
it with the local ALd slice, accumulates into the resident **pre-fold** block H
(size (χ/N1)·D⁴·(χ/N2)), then ring-shifts the FL block along `row_comm` with
double buffering. After a full cycle the FL blocks are back home (the backward
replay relies on this invariant; the caller's input block is never mutated —
shifts operate on internal copies).

The M1/M2 fold happens **once, after the ring** — its cost is independent of the
i-block extent, so folding inside the loop would redo it N2 times (overhead
growing with grid size: ~1.33× total FLOPs at 2×2, ~3.3× at 8×8 — the same
disease that disqualified the one-shot approach C). Fold-after-ring keeps total
FLOPs exactly serial/P.

### Forward — stage 2 (contract a; column reduce-scatter)

```
partial[d,g,h, l∈block r2] = G[a,b,c,g,h,l] · ALu[a∈block r1, b,c,d]   # d full-length
```

Ring reduce-scatter over the N1 ranks of `col_comm`: sum partials and split along
`d_range`, leaving rank (r1,r2) with `result[d_range(r1), :, :, l_range(r2)]`.
d-blocks are strided in column-major memory → stage through a contiguous
pre-allocated buffer (`_comm_recvbuf` pattern) before each Isend.

### Cost model (per rank, per map call)

| Quantity | Value |
|----------|-------|
| Communication | ≈ χ²D²(1/N1 + 1/N2) · sizeof(T) — at 2×2 comparable to the slice scheme's allgather; **1/4 at 8×8, 1/6 at 12×12** |
| FLOPs | exactly serial / P (fold-after-ring; no redundancy) |
| Peak transient memory | ≈ (2+d)·χ²D⁴/P during the M fold (`@tensor`'s pairwise temporaries: H + H·M1 intermediate + G) — i.e. the serial transient peak / P. Ring phase itself is ~2·χ²D⁴/P (H + per-step FL·ALd temp); FL double buffer 2χ²D²/P and stage-2 partial χ²D²/N2 are lower-order. |

At χ=400, D=10, d=2, ComplexF64, 2×2: per-rank transient peak ≈ 25.6 GB vs
~102 GB serial — both dominated by the pre-fold D⁴ intermediates; the
distributed value is the serial value / P. The existing `forloop_iter`-style
sub-slicing (subdividing the local l range inside stage 1) is orthogonal and
composable.

---

## §2 Backward rrule and API

### AD strategy

One hand-written `ChainRulesCore.rrule` for the whole map, following the
codebase precedent of the `parallel` rrule (`src/autodiff/rules.jl:241`):
`Zygote.pullback` on the local @tensor kernels, communication adjoints
hard-coded. Zygote never sees MPI or in-place mutation.

### Communication adjoints

| Forward | Backward |
|---------|----------|
| column reduce-scatter | column allgather |
| row ring shift (FL block rotation) | replay rotation; dFL accumulator block travels paired with its FL block |
| replicated inputs (ALu/ALd/M) | gradient allreduce(+) over COMM_WORLD |

### Backward flow (rank (r1,r2) receives `dresult[d_range(r1), :, :, l_range(r2)]`)

1. **Column allgather**: `dpartial[d full, g, h, l∈r2] = allgather(dresult, col_comm)`
   — adjoint of reduce-scatter.
2. **Post-ring pullback** (local, once): one composite
   `pullback((H, ALu_s, M1, M2) -> stage2(fold(H, M1, M2), ALu_s))` applied to
   `dpartial` yields `dH`, the a_r1 slice contribution to `dALu`, and **dM1/dM2
   in one shot** (fold-after-ring means the M gradients are not per-step sums).
   The H block is captured in the rrule closure (χ²D⁴/P; a recompute option can
   trade memory for compute later — design hook only).
3. **Stage-1 reverse ring**: replay the FL rotation with each FL block's **dFL
   accumulator travelling alongside it**. Each step: the cheap two-tensor
   `pullback(stage1_kernel)(dH)` adds the dFL contribution into the paired
   accumulator and accumulates the local `dALd[i∈t_k, :, :, l∈r2]` contribution;
   then (FL block, dFL block) shift together. After a full cycle every dFL block
   lands on its home rank — **dFL stays distributed**, matching "distributed
   input → distributed gradient". Backward message volume ≈ 2× forward.
4. **Replicated-arg gradient reduction**: dALu (only the a_r1 slice nonzero per
   rank), dALd (only the l_r2 slice nonzero), dM1/dM2 — all via
   `allreduce_p2p!(+, COMM_WORLD)`, which sums overlapping slices and stitches
   disjoint ones in one call, and picks up the existing NCCL fast path for free.

### API and file layout

```julia
# src/contraction/cannon_2d.jl  (new file, included after forloop_parallel_MPI.jl)
cannon_grid(N1, N2; comm=MPI.COMM_WORLD) -> CannonGrid   # cached; row/col sub-comms
FLmap_cannon(FL_blk, ALu, ALd, M, grid; inner_etype=nothing) -> result_blk
cannon_scatter(FL_full, grid) -> FL_blk   # boundary shim: slice local block (adjoint: embed + allreduce)
cannon_gather(blk, grid) -> full          # boundary shim: 2D allgather (adjoint: slice)
```

- Stage-1/2 `@tensor` kernels: **leg5 only** this round (single M routed through
  `(M, conj(M))` into the (M1, M2) tuple path — covers Kagome/honeycomb
  production). leg4/leg8 follow the same pattern later.
- `inner_etype` boundary cast reuses `_boundary_cast`: downcast once at entry,
  upcast once at exit; rotation messages travel in the lower precision (2× bandwidth).
- Transport: MPI `Isend/Irecv` (CUDA-aware, zero new dependencies).
  `ncclSend/ncclRecv` fast path is a later increment (~40 lines in
  `nccl_wrapper.jl`, behind the existing `TENET_USE_NCCL` switch).
- `cannon_scatter/gather` get rrules so map-level tests run end-to-end under
  Zygote; they are also the seam for the later leftenv integration.
- Guards: grid size assertion; block-shape mismatch errors immediately.

---

## §3 Testing and Sofia validation

### Level 1 — local CPU correctness (`test/test_cannon.jl`, `mpiexec -n 4`)

Pattern follows `test/test_mpi.jl`. Small random tensors (χ=16, D=4, ComplexF64, leg5).

- Forward parity: `cannon_gather(FLmap_cannon(cannon_scatter(FL), …))` vs serial
  `FLmap`, rel. err ≤ 1e-12
- Gradient parity: full Zygote chain (scatter → cannon → gather → scalar loss) vs
  serial pullback; **all five gradients compared individually** (dFL, dALu, dALd,
  dM1, dM2) ≤ 1e-10
- Grid variants: 2×2, 1×4, 4×1 (degenerate 1×N paths must be correct);
  non-divisible blocking (χ=18 over 4); single-M and (M1,M2)-tuple entry points

### Level 2 — Sofia 4×H200 GPU validation

- Submit script `examples/MPI_parallel/Sofia/submit_test_cannon.sh` using the
  canonical v17 environment (`LD_PRELOAD=/usr/lib64/libcuda.so.1`,
  `CUDA_LAUNCH_BLOCKING=1`, CUDA stripped from Julia's LD_LIBRARY_PATH)
- Production-like sizes D=10, χ=400, leg5 tuple, 2×2 grid:
  - forward/gradient parity vs the existing `FLmap_parallel` slice path
    (gradient ≤ 1e-8, the repo's PR verification standard)
  - **peak memory per rank** (CUDA pool stats): Cannon vs slice, validating the
    ≈1/4 transient-intermediate prediction
  - timing: at 2×2 comm volume matches slice, expect ≤ ~1.5× slice wall time per
    map (rotation sync overhead); speed is not optimized this round — the comm
    advantage materializes at 8×8
- Driver reuses the existing `test/sofia_mpi_test_driver.jl` framework

### Pass criteria

Level 1 all green + Level 2 gradient parity 1e-8 + peak memory clearly below
slice. Debug order on failure: backward rotation step alignment first (most
error-prone), then reduce-scatter strided staging, then `inner_etype` precision.

### Explicitly out of scope this round

leftenv/simple_eig integration, AL tensor distribution, NCCL p2p fast path,
leg4/leg8 kernels, 8×8 cross-node stress testing — all listed as follow-up
routes, not v1 deliverables.
