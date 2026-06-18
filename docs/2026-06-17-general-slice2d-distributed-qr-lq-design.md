# General Slice2D distributed QR/LQ seams

Date: 2026-06-17. Scope: replace the full-gather QR/contraction seams in
`VUMPS{General}` Slice2D steps. This is the follow-up to the Plaquette TSQR
path and makes the General runtime obey the Slice2D rule: production code must
not materialize full `AL`, `AR`, `AC`, `FL`, or `FR` on every rank.

## Goal

Make `vumps_step_slice2d(rt::VUMPSRuntime, M, grid, alg::VUMPS{General})`
block-distributed end to end for the persistent boundary tensors:

- `AL`, `AR`, `AC`, `FL`, and `FR` stay in Slice2D blocks
  `(first chi leg split by grid.r1, last chi leg split by grid.r2)`.
- `C` remains replicated full chi-by-chi because it is small compared with the
  rank-local boundary tensors and is already a replicated operand throughout
  the Slice2D maps.
- The old full-gather seams remain available only as explicit reference/debug
  helpers used by tests.
- Complex-valued AD must match the serial `ACCtoAL`/`ACCtoAR` pullbacks block
  by block.
- `step_checkpoint=Recompute()` must be safe: every rank re-enters the same
  MPI collective sequence in forward and backward.

## Current problem

The current General Slice2D step still contains two full-gather seams:

```julia
AC = ALCtoAC_slice2d(AL, C, grid)
AL, AR, errL, errR = checkpoint(sub, (ac, c) -> ACCtoALAR_slice2d(ac, c, grid), AC, C)
```

`ALCtoAC_slice2d` gathers full `AL`, calls serial `ALCtoAC`, and scatters full
`AC`. `ACCtoALAR_slice2d` gathers full `AC`, calls serial `ACCtoALAR`, and
scatters full `AL`/`AR`. This breaks the Slice2D memory model. The Plaquette
branch added `ACCtoAL_tsqr_slice2d`, but General still lacks the matching
distributed LQ path for `ACCtoAR`.

There is also an `ALCtoAC_slice2d_dist` helper used by observable code. It is a
temporary side path rather than the canonical General seam. This design replaces
it with one production `ALCtoAC_slice2d` implementation and renames the old
gather implementation for reference use.

## Non-goals

- Do not distribute General initialization in the first phase. The existing
  `left_canonical`/`right_canonical` fixed-point initialization remains a
  separate phase because it is not a single QR/LQ seam.
- Do not change the Slice2D FL/FR/AC/C map algorithms except where their
  existing gather primitives are reused by the seam AD.
- Do not change serial `ACCtoAL`, `ACCtoAR`, `ACCtoALAR`, `ALCtoAC`, `qrpos`, or
  `lqpos`.
- Do not support rectangular grids for the new QR/LQ seams in the first phase.
  The existing General Slice2D path already assumes square-grid behavior in the
  cross-axis maps.

## Design overview

Add three distributed seam primitives and route General `vumps_step_slice2d`
through them:

```julia
AC = ALCtoAC_slice2d(AL, C, grid)                  # distributed contraction
AL, AR, errL, errR = ACCtoALAR_dist_slice2d(AC, C, grid)
```

The production meanings become:

- `ALCtoAC_slice2d`: distributed contraction, no full `AL` or `AC`.
- `ACCtoAL_tsqr_slice2d`: distributed left QR, already present for Plaquette and
  promoted to the shared General/Plaquette implementation.
- `ACCtoAR_tslq_slice2d`: new distributed right LQ.
- `ACCtoALAR_dist_slice2d`: calls both distributed exits and preserves the
  serial residual accumulation rules.

The old gather wrappers are renamed:

- `ALCtoAC_slice2d_gather_ref`
- `ACCtoAL_slice2d_gather_ref`
- `ACCtoALAR_slice2d_gather_ref`

Tests use reference helpers for parity. Production `vumps_step_slice2d` does not.

## Distributed `ALCtoAC`

Serial:

```julia
AC[a,i,j,d] = sum_b AL[a,i,j,b] * C[b,d]
```

Slice2D storage:

- `AL_blk` owns `a in r1`, `b in r2`.
- `C` is replicated full `(b,d)`.
- `AC_blk` must own `a in r1`, `d in r2`.

Forward:

1. Build a local partial with full output `d`:

   ```julia
   partial[a_local,i,j,d_full] =
       sum_{b in my r2 block} AL_blk[a_local,i,j,b] * C[b,d_full]
   ```

2. Sum over `row_comm` and keep this rank's `d` block:

   ```julia
   AC_blk = _slice2d_row_reduce_scatter_last(partial, grid, d_ranges)
   ```

This removes the row allgather in `ALCtoAC_slice2d_dist` and gives the operation
the correct contraction shape: the contracted `b` leg is distributed over
`row_comm`, while the output `d` leg is scattered over the same communicator.

AD:

- The reduce-scatter adjoint is row allgather.
- `dAL_blk` is local contraction against the corresponding `C[b_local,d_full]`.
- `dC` is replicated. Each rank computes its local contribution over
  `a_local` and `b_local`, then allreduces over `grid.comm`.
- The allreduce for `dC` is required because `C` is a replicated input; this is
  the same replicated-operand rule used elsewhere in Slice2D AD.

## Distributed left QR: `ACCtoAL_tsqr_slice2d`

This is the Plaquette TSQR seam, shared with General.

For each StructArray data block:

1. Row-gather the last chi leg:

   ```julia
   AC_row = slice2d_gather_row(AC_blk, grid, last_ranges)
   ```

   Shape: `(chi_r1, D, D, chi_full)`.

2. Reshape to the serial `_to_front` matrix:

   ```julia
   A_mat = reshape(AC_row, chi_r1 * D * D, chi_full)
   ```

3. Local QR:

   ```julia
   Qloc, Rloc = qrpos(A_mat)
   ```

4. Gather the small `Rloc` factors over `col_comm`, stack them, and QR again:

   ```julia
   Rstack = slice2d_gather_col(Rloc, grid, r_ranges)
   Q2stack, RAC = qrpos(Rstack)
   QAC = Qloc * Q2stack[my_r1_rows, :]
   ```

5. QR the replicated `C` through `qrpos_colrep(C, grid)`.

6. Produce and slice `AL`:

   ```julia
   AL_row = reshape(QAC * QC', size(AC_row))
   AL_blk = slice_last_leg(AL_row, my_r2_range)
   errL = norm(RAC - RC)
   ```

AD:

- `qrpos` handles local QR and the small R-stack QR.
- `slice2d_gather_row` and `slice2d_gather_col` provide the matching
  reduce-scatter adjoints.
- `qrpos_colrep` allreduces the `C` cotangent contribution from `QC`; the `RC`
  residual cotangent remains local to the replicated `errL` path as in the
  current Plaquette implementation.

## Distributed right LQ: `ACCtoAR_tslq_slice2d`

Serial `ACCtoAR` does:

```julia
LAC, QAC = lqpos(_to_tail(AC[i,j]))
LC,  QC  = lqpos(C[i,jr])
AR[i,j] = reshape(QC' * QAC, size(AC[i,j]))
```

Use the identity `lqpos(A)` equals sign-fixed QR of `A'`, with outputs
transposed back. The distributed LQ is therefore a TSQR on the adjoint/tail
matrix.

For a block `AC_blk[a_local,i,j,d_local]`, `_to_tail(AC)` has matrix shape:

```text
rows = chi_full                    # first chi leg a
cols = D * D * chi_full            # i,j,d
```

The local block owns a slice of rows (`a_local`) and a slice of the last part of
the columns (`d_local`). For LQ, transpose the matrix and run a QR on:

```text
_to_tail(AC)' : (D * D * chi_full) x chi_full
```

The row dimension of this transposed matrix is distributed by the original
`d`/last-leg partition, so the communication pattern mirrors left TSQR with
axes swapped:

1. Column-gather the first chi leg of `AC_blk` over `col_comm` so the local
   rank has `AC_col[a_full,i,j,d_local]`.
2. Reshape/permutedim to the transposed tail matrix row block:

   ```julia
   AtailT_row = reshape(permutedims(AC_col, tail_order), D * D * chi_r2, chi_full)
   ```

   The exact reshape helper must be paired with a round-trip test against
   `_to_tail(AC)'`.

3. Run the same `_tsqr_front_rowblock` pattern over `row_comm` instead of
   `col_comm`. This requires a generalized TSQR helper:

   ```julia
   _tsqr_front_axis(A_row, comm_axis, axis_rank, axis_size)
   ```

4. Convert QR outputs back to LQ outputs:

   ```julia
   Q_tail = Q_from_qr'      # QAC
   L_tail = R_from_qr'      # LAC
   ```

5. Factor replicated `C` through `lqpos_colrep(C[i,jr], grid)`.

6. Produce `AR_col = reshape(QC' * QAC, full-a, D, D, local-d)`, then slice the
   first chi leg back to this rank's `r1` block:

   ```julia
   AR_blk = slice_first_leg(AR_col, my_r1_range)
   errR = norm(LAC - LC)
   ```

AD:

- Prefer implementing right LQ in terms of `qrpos` on the adjoint matrix so it
  reuses the existing `qrpos` rrule. A thin wrapper may expose LQ-shaped outputs,
  but its pullback should be the composed QR pullback plus reshape/adjoint
  operations, not a new dense LQ formula.
- `lqpos_colrep(C, grid)` mirrors `qrpos_colrep`: allreduce the replicated `C`
  gradient contribution from `QC`, and include the residual branch from `LC`.
- The gather adjoints are axis-swapped from left TSQR:
  `slice2d_gather_col` backpropagates through `_slice2d_col_reduce_scatter`.

## Residual accumulation and repeated patterns

Serial `ACCtoAL` and `ACCtoAR` do not iterate over the same index set:

- `ACCtoAL` loops over `1:length(AC)`, i.e. all StructArray positions.
- `ACCtoAR` loops over `1:length(AC.data)`, finds the first pattern position,
  and uses `jr = mod1(j - 1, Nj)`.

`ACCtoALAR_dist_slice2d` must preserve this asymmetry. The implementation should
not blindly map both exits over `eachindex(AC.data)` unless it also matches the
serial residual and cell-selection semantics.

Required structure:

- `ACCtoAL_tsqr_slice2d` returns the same `AL` data and `errL` as serial
  `ACCtoAL`, including repeated-pattern multiplicities.
- `ACCtoAR_tslq_slice2d` returns the same `AR` data and `errR` as serial
  `ACCtoAR`, using the first occurrence and the serial `jr` rule.
- `ACCtoALAR_dist_slice2d` simply combines those two outputs.

## Routing

Phase 1 changes General only:

```julia
function vumps_step_slice2d(rt::VUMPSRuntime, M::StructArray, grid::Slice2DGrid, alg::VUMPS{General})
    @unpack AL, C, AR, FL, FR = rt
    sub = alg.subop_checkpoint
    AC = ALCtoAC_slice2d(AL, C, grid)
    _, FL = checkpoint(sub, (a, b, m, fl) -> leftenv_slice2d(a, b, m, fl, grid; alg), AL, conj(AL), M, FL)
    _, FR = checkpoint(sub, (a, b, m, fr) -> rightenv_slice2d(a, b, m, fr, grid; alg), AR, conj(AR), M, FR)
    _, AC = checkpoint(sub, (ac, fl, m, fr) -> ACenv_slice2d(ac, fl, m, fr, grid; alg), AC, FL, M, FR)
    _, C  = Cenv_slice2d(C, FL, FR, grid; alg)
    AL, AR, errL, errR = checkpoint(sub, (ac, c) -> ACCtoALAR_dist_slice2d(ac, c, grid), AC, C)
    return VUMPSRuntime(AL, AR, for_gc(C), FL, FR), errL + errR
end
```

Plaquette may continue to use `ACCtoAL_tsqr_slice2d`. After the General seam is
stable, the `distributed_qr` flag should be reinterpreted or removed so the
production Slice2D path has one meaning instead of "gather by default, TSQR only
when opted in." During migration, keep the flag only as a compatibility switch.

Observable code should call `ALCtoAC_slice2d`, not `ALCtoAC_slice2d_dist`.

## Test plan

Add a 4-rank CPU MPI test file, e.g. `test/test_slice2d_general_dist_qr_lq.jl`.

Forward seam tests:

- `ALCtoAC_slice2d` vs `scatter(ALCtoAC(gather(AL), C))`.
- `ACCtoAL_tsqr_slice2d` vs serial `ACCtoAL`.
- `ACCtoAR_tslq_slice2d` vs serial `ACCtoAR`.
- `ACCtoALAR_dist_slice2d` vs serial `ACCtoALAR`.
- Include both all-distinct and repeated patterns such as `[1 2; 2 1]`.
- Use complex fixtures whose QR/LQ diagonals are bounded away from zero to
  avoid accidental sign-gauge flips.

Gradient seam tests:

- For each seam, compare `Zygote.gradient` against the serial full reference.
- Compare `dAC_blk`, `dAL_blk`, and `dAR_blk` blockwise, not just by global
  norm.
- Compare replicated `dC` directly on every rank and assert cross-rank
  uniformity.
- Use complex cotangents. Real-only tests can miss conjugation bugs.

Integration tests:

- One General `vumps_step_slice2d` parity test against serial at small chi.
- One `leading_boundary`/AD smoke with `step_checkpoint=Recompute()` to verify
  the backward replay collective order.
- Existing M4/M5/Plaquette tests must remain green.

Negative controls:

- A deliberately wrong LQ axis choice should fail forward parity.
- A deliberately missing allreduce in `lqpos_colrep` or `qrpos_colrep` should
  fail `dC` parity.

## Migration steps

1. Rename existing gather seams to `*_gather_ref` and update tests to call those
   names explicitly.
2. Implement `ALCtoAC_slice2d` as the row reduce-scatter contraction.
3. Generalize the TSQR helper so left QR can use `col_comm` and right LQ can use
   `row_comm`.
4. Keep `ACCtoAL_tsqr_slice2d` as the shared left QR seam and fix residual
   accumulation if repeated-pattern tests expose a mismatch.
5. Implement `ACCtoAR_tslq_slice2d`, `lqpos_colrep`, and `ACCtoALAR_dist_slice2d`.
6. Route General `vumps_step_slice2d` to the distributed seams.
7. Replace observable calls to `ALCtoAC_slice2d_dist` with `ALCtoAC_slice2d`.
8. Remove `ALCtoAC_slice2d_dist` once all callers are migrated.
9. Leave General init unchanged in this phase, but document it as the next
   remaining full-chi Slice2D gap.

## Risks

- Right LQ axis mistakes are easy: `_to_tail(AC)'` must be matched exactly by
  reshape/permutation round-trip tests.
- Repeated-pattern residuals can be silently overcounted if `ACCtoAL` and
  `ACCtoAR` are mapped over the same data index set.
- Replicated `C` gradients need allreduce treatment. Missing it may pass local
  forward tests and fail only AD parity.
- `Recompute()` replay can deadlock if any rank-dependent branch enters a
  different gather/reduce-scatter order. All seam loops must be rank-uniform.
- The first phase still has full tensors in initialization. That is acceptable
  only because the target of this phase is the AD VUMPS step memory model, not
  startup memory.

## Acceptance criteria

- General Slice2D `vumps_step_slice2d` contains no production call that gathers a
  full `AL`, `AR`, `AC`, `FL`, or `FR`.
- `ALCtoAC_slice2d_dist` has no production callers.
- Forward and gradient parity tests pass on 4 CPU ranks for complex inputs.
- `step_checkpoint=Recompute()` AD smoke passes without deadlock.
- Plaquette `distributed_qr` behavior is not regressed.

## Implementation notes

- The original gather/scatter seams were renamed to `*_gather_ref` and kept as
  reference helpers for forward/gradient parity tests.
- `ALCtoAC_slice2d` is now the production row reduce-scatter contraction and
  carries a custom rrule for the distributed seam.
- `ACCtoAL_tsqr_slice2d` is the shared left-TSQR seam for both Plaquette and
  General left QR.
- `ACCtoAR_tslq_slice2d` implements the right-LQ seam by running QR on
  `_to_tail(AC)'`; for complex inputs the row-block path must conjugate the
  block input because Julia `'` is a conjugate transpose.
- `lqpos_colrep` handles replicated-`C` AD by allreducing the `Q`
  contribution before returning the cotangent.
- General `vumps_step_slice2d` now routes to `ACCtoALAR_dist_slice2d`.
- Observable callers use production `ALCtoAC_slice2d`; the old
  `ALCtoAC_slice2d_dist` helper has been removed.
- The remaining full-chi gap in this phase is General initialization, which
  still uses full canonicalization. Plaquette `distributed_qr` now has a
  distributed init path, while the matching General init work remains future
  work.
