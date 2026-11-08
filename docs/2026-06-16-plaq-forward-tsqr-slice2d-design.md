# Plaquette Slice2D Forward TSQR Seam Design

Date: 2026-06-16. Scope: unblock Plaquette observable runs at large chi where the
current Slice2D path still performs replicated full-chi QR.

## Goal

Make the `VUMPS{<:Plaquette}` Slice2D path use distributed TSQR seams by
default. The path is intended to avoid replicated full-chi QR during
initialization and VUMPS steps.

## Targeted Failure

Sofia job 1287380 failed at chi=2048 during Plaquette Slice2D initialization:
`init_VUMPSRuntime_slice2d -> left_canonical -> qrpos!`. The same full-chi QR
exists in every Plaquette `ACCtoAL_slice2d` step.

## Approach

1. Keep gather-based QR as reference-only test helpers.
2. Use `ACCtoAL_tsqr_slice2d(AC_blk, C, grid)` as the production seam.
   - Row-gather the last chi leg so each row rank owns
     `(local first chi block) x D x D x full chi`.
   - Run local QR on `_to_front` of that row block.
   - Gather local R factors over `col_comm`, QR the stacked R, and multiply the
     local Q by the matching combiner block.
   - Multiply by `qrpos(C)'`, reshape to a row block, then slice the last chi
     leg back to this rank's `r2` block.
3. Use the Plaquette Slice2D distributed init path by default.
   - Construct random block tensors directly instead of allocating full random
     `A` on each rank.
   - Use the same TSQR helper for a one-step left-canonical starting point.
   - Build `C` from the TSQR R factors and initialize `FL` as block random data
     before solving `leftenv_slice2d`.
4. Route Plaquette `vumps_step_slice2d` to TSQR unconditionally on Slice2D.

## Non-Goals

- No distributed QR adjoint in this change.
- No General-mode `ACCtoALAR` TSQR.
- No distributed replacement for the full iterative `left_canonical` fixed-point
  solve. The TSQR init is a valid canonical starting point and the VUMPS boundary
  iteration refines it.

## Gates

- 4-rank CPU MPI parity: `ACCtoAL_tsqr_slice2d` vs serial `ACCtoAL` at small chi.
- 4-rank CPU MPI init sanity: default Slice2D TSQR init is rank-uniform and
  produces left-isometric gathered `AL`.
- Existing Plaquette M5 tests remain green with the default Slice2D TSQR path.
- Sofia smoke before resubmitting the chi=2048 observable.
