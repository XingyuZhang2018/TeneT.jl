# Distributed QR/contraction seams — χ768 optimization (M7)

Date 2026-06-16. Branch `claude/ecstatic-golick-e3f6f0`. Process: research → design → adversarial
review → implement → gate, parity-gated.

## Motivation (from the χ768 Sofia smokes)

- Distributed init ✓ (M7-pre, done), distributed FORWARD boundary CONVERGES at χ768 D16 on 16 GPU
  (job 1287330: PlaqVUMPS 10 steps, err 2.67e-4). The replicated model OOMs at χ624.
- The **AD (gradient) pass OOMs** (`_slice2d_fold2`, baseline near 143 GB). forloop_iter 16→64 shrank
  the failing alloc 1.125 GiB→288 MiB but still OOM at the same step ⇒ **baseline-dominated, not
  chunk-dominated**. The baseline is the **non-distributed full-χ QR/contraction seams**:
  - `ALCtoAC_slice2d` (slice2d.jl:349): **gather full χ768 AL** (4 cells × 2.42 GB ≈ 9.6 GB) → serial
    `ALCtoAC` → scatter. Every vumps step.
  - `ACCtoAL_slice2d` / `ACCtoALAR_slice2d` (slice2d.jl:556/370): **gather full χ768 AC** (≈9.6 GB) →
    serial `qrpos` (QAC = (χDD,χ) ≈ 9.6 GB) → scatter. Every step, and in the AD the adjoint
    materializes them again.
  - `oc_22` (observable.jl): gathers a full-χ Q for the J2-bond qrpos (energy path; smaller).
  These appear in BOTH the forward and the backward of the AD → ~2× in the gradient pass → OOM.

Goal: distribute the seams so χ768 (and larger) OPTIMIZATION fits on 16 GPU.

## The two seam kinds

**1. `ALCtoAC` = a CONTRACTION (not a QR).** `AC[a,i,j,d] = Σ_b AL[a,i,j,b]·C[b,d]` (general.jl:196).
AL_blk's last leg `b` is r2-blocked (row_comm); C is replicated full (b,d). The full sum over `b`
needs all r2-blocks ⇒ **reduce-scatter over row_comm**:
- local `partial[a,i,j,d] = Σ_{b∈my r2-block} AL_blk[a,i,j,b]·C[myblock_b, d]` (d FULL),
- `AC_blk = reduce_scatter_row_last(partial)` — sums the partials over r2 + scatters the output `d`
  to r2-blocks. NO gather. Reuses `_slice2d_row_reduce_scatter_last` (slice2d.jl:402, NCCL path +
  rrule). C replicated → its cotangent allreduced like a replicated operand.
Memory: partial is (r1-block · DD · full-χ) ≈ 600 MB/cell vs 9.6 GB full gather.

**2. `ACCtoAL`/`ACCtoAR` = a tall-skinny QR (TSQR).** `AL = qrpos(_to_front(AC))·qrpos(C)'`
(general.jl:685). The cost is `qrpos` of the (χ·D·D, χ) matrix — tall-skinny (χDD ≫ χ). The matrix
is 2D-block-distributed: **rows = χ(r1-blocked)·D·D, cols = χ(r2-blocked)**. `qrpos(C)` is (χ×χ),
tiny → replicate. Distributed plan:
- **col-gather** the r2 cols → each rank holds (its r1-row-block · DD) × FULL-χ cols (≈600 MB/cell,
  allgather over row_comm),
- **TSQR over col_comm (r1 partition)**: local `qrpos` of the row-block → (Q_local, R_local); combine
  R_locals over col_comm (stack → QR → final R + per-rank combining factors); Q = Q_local·(combiner),
- `AL = Q·QC'` (row-block × χ) → reshape (r1-block, D, D, full-χ) → slice last leg to r2-block ⇒ AL_blk.
This replaces the 9.6 GB gather + 9.6 GB QAC with a 600 MB/cell row-block. `errL = ‖R − RC‖` is a
replicated scalar (R is the global TSQR R, identical per rank; never allreduced — like the M5 seam).

**AD is the hard part of #2.** The qrpos rrule (misc.jl) is for a single full matrix. A distributed
TSQR needs a distributed-QR adjoint (back-propagate through the local QRs + the R-combine + the
col-gather). Known but error-prone; this is the research-level piece.

## Phased plan (de-risk: do the moderate win first, measure, then the hard one)

- **Phase 1 — distribute `ALCtoAC`** (reduce-scatter). Moderate, reuses existing reduce-scatter +
  rrule. Removes one 9.6 GB/step gather (forward AND its adjoint). Then **re-run the χ768 AD smoke**
  to measure the relief: if the AD now fits → done (the ACCtoAL gather alone fit in the forward, and
  removing ALCtoAC may be enough headroom for the AD). If still OOM → Phase 2.
- **Phase 2 — distribute `ACCtoAL`/`ACCtoALAR`** (col-gather + TSQR + AD). The hard one. Only if
  Phase 1 is insufficient.
- **Phase 3 — `oc_22` qrpos seam** (energy path). Smaller; gather Q → its qrpos is (χDD,χ) too. Defer
  until the boundary seams are done (the smoke OOMs in the boundary AD, before energy_value).

## Parity gates (each phase, 4-rank CPU, bit-exact vs serial — the M5-0 standard)

- Phase 1: `ALCtoAC_slice2d(AL_blk, C)` (reduce-scatter) vs `scatter(ALCtoAC(gather(AL_blk), C))` —
  forward bit-parity (≤1e-12; CPU reassoc), and the reduce-scatter gradient vs serial (R-1: the
  reduce-scatter adjoint = allgather, must not over/under-count; dC replicated).
- Phase 2: `ACCtoAL_slice2d` (TSQR) vs serial `ACCtoAL(gather(AC_blk), C)` — AL up-to-gauge (QR gauge:
  compare AL†AL=I + the seam's bit-exactness on a fixed input, like M5-0/M5-2) + errL; TSQR gradient
  vs serial qrpos rrule (the load-bearing AD check).
- Integration: the existing M5 step gates (M5-0/1/2, M5p-0/1/2) must stay green (the seams feed
  vumps_step); the local real-physics χ23 must still reproduce -0.470392432614.
- GPU: re-run the χ768 smoke; expect the AD to fit (no `_slice2d_fold2` OOM) and `SMOKE forward
  energy_χ768 ≈ -0.4967` + `gradient computed`.

## Risks / adversarial-review targets

1. Phase 1 reduce-scatter direction: confirm it's row_comm (r2 sum, scatter output `d` to r2) — the
   contracted leg is AL's last leg (r2). Pick the correct primitive (`_row_reduce_scatter_last` vs
   `_col_reduce_scatter`) and verify against a bit-parity gate (a wrong axis is silently wrong).
2. Phase 1 may NOT suffice (the ACCtoAL gather + QAC + tape may still OOM the AD). Measure before
   committing to Phase 2. Honest possibility: only Phase 2 (TSQR) truly unlocks χ768 AD.
3. Phase 2 TSQR AD correctness — the distributed-QR adjoint is the highest-risk piece. Gate it
   against the serial qrpos rrule with COMPLEX cotangents (real-only hides conj bugs).
4. Gauge: ACCtoAL's AL is QR-gauge; don't compare AL directly across FP paths when AC is near-
   singular (the M5 conditioning lesson) — gate via AL†AL + bit-exact seam on a fixed input.
5. C replicated throughout (both seams) — its cotangent handling (allreduce vs not) must match the
   slice2d_dot/slice2d_norm asymmetry rule.

## Adversarial review R1 (opus) — DIAGNOSIS WAS WRONG; pivot to checkpointing

R1 found the motivation MISATTRIBUTED the χ768 AD OOM. The real cause is NOT the full-χ seam
gathers — it's the **un-checkpointed AD-loop tape**: the smoke driver sets no `ifcheckpoint`, so
`step_checkpoint=Plain()` (interface.jl:83), and the Plaquette `leading_boundary` differentiates
straight through the `maxiter_ad=4` loop (plaquette.jl:250-277) keeping all 4 `vumps_step` tapes
live. The forward fit only because its no-AD loop is `ignore_derivatives` (per-step gather freed);
the AD keeps everything ×4. forloop_iter 16→64 shrinking the alloc 4× but OOMing at the same point
is consistent with a tape-dominated baseline (the seam gathers don't depend on forloop either).

**Fix = zero new code: `step_checkpoint=Recompute()`** — already wired (`checkpoint(alg.step_checkpoint,
vumps_step, …)`, plaquette.jl:268), already gated, UNBLOCKED on the slice2d path (`vumps_step_slice2d`
only asserts `inner_checkpoint isa Plain`; step/subop are free). Collapses the 4-step tape to ~1.
CPU-validated (test/local_slice2d_realphys.jl, step_checkpoint=Recompute): Phase-1 slice2d==serial==
-0.470392432614 (Δ2.2e-16) AND Phase-2 descent BIT-IDENTICAL to Plain (recompute is exact; the
slice2d recompute-in-backward re-runs the MPI collectives rank-uniformly, fine).

⇒ **Distributing the seams (esp. the weeks-long TSQR + distributed-QR AD) is PREMATURE and was a
misdiagnosis** — it saves ~9.6 GB against a 4×-step tape and doesn't touch the step multiplier.
NEW PLAN: (1) `step_checkpoint=Recompute` + re-run χ768 smoke (job 1287338); escalate to
`OffloadRecompute` if still tight. (2) `ALCtoAC` reduce-scatter only later, for FORWARD headroom at
even larger χ. (3) TSQR last, if ever. The reduce-scatter direction in §1 (row_comm,
`_slice2d_row_reduce_scatter_last`) was confirmed correct by R1 — keep for (2).
