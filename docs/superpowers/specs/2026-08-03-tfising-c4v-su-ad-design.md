# TFIsing Real C4v Simple-Update AD Parameterization Design

## Goal

Add a fixed-bond-dimension, single-site, real-valued C4v Simple-Update (SU)
parameterization for square-lattice TFIsing optimization.  The optimizer must
differentiate through the complete map

```text
raw tensor -> C4v restriction -> SU(tau) -> boundary environment -> energy
```

so that every tensor passed to the environment, energy, and observable code is
the materialized post-SU C4v tensor.  The initial research value is
`tau = 0.01`; `tau = 0` is the exact direct-C4v baseline.

The purpose is not merely to lower a small-chi energy.  It is to test whether a
short SU layer regularizes the AD optimization trajectory enough to produce a
cleaner energy-versus-correlation-length extrapolation than direct C4v
optimization.

## Motivation and Existing Behavior

The current generic `SU_parameterization` is not a valid implementation of this
map for a one-site C4v cell:

- a `[1;;]` cell aliases both ends of a bond, while the generic update produces
  distinct left and right SVD factors and writes both back to the same tensor;
- the update mutates arrays, which Zygote rejects on the requested path;
- the generic TFIsing gate falls through to Heisenberg bond terms and omits the
  TFIsing transverse-field evolution;
- the generic result has no C4v postcondition; and
- the legacy `build_A(A, params, rt)` references inactive `one_bond_SU` and
  `hv_SU_update` functions.

The earlier single-site SU growth design deliberately kept C4v projection
caller-controlled.  This design adds a separate mode with a stronger contract;
it does not reinterpret the generic multi-cell SU implementation.

The forward self-bond construction follows the one-factor Takagi idea in the
commented TeneT_demo implementation and the active real C4v D-resize
implementation in TeneT.c.  Unlike those paths, this layer keeps D fixed and is
part of the differentiated objective.

## Scope

In scope:

- real `Float32`/`Float64` square-lattice TFIsing tensors;
- one unique site with pattern `[1;;]` and uniform virtual dimension D;
- full C4v input and output symmetry;
- a TFIsing-specific real imaginary-time gate;
- a pure, fixed-D, four-direction self-bond Takagi sweep;
- AD through the SU layer, boundary environment, and energy;
- unambiguous raw and post-SU checkpoints;
- a CPU validation example at `D = 2`, `chi = 8`, `tau = 0.01`,
  `J = 1`, and `h = 3.04438`; and
- a paired direct-C4v versus SU-C4v smoke comparison.

Out of scope for the first implementation:

- complex iPEPS tensors;
- C4-only tensors without reflections;
- multi-site unit cells;
- models other than square-lattice TFIsing;
- changing D inside the differentiable layer;
- GPU, MPI, or cluster production validation;
- a production claim that the extrapolation has improved; and
- a tau scan beyond the exact `tau = 0` baseline and initial `tau = 0.01`
  experiment.

## Public Configuration and Compatibility

Keep the existing `ifSU` and `SUτ` parameters and
add an explicit SU mode parameter:

```julia
SUmode::Symbol = :generic
```

The supported meanings are:

- `ifSU == false`: no SU layer, independent of `SUmode`;
- `ifSU == true && SUmode == :generic`: preserve current behavior; and
- `ifSU == true && SUmode == :c4v_real`: use the new implementation.

The TFIsing C4v example selects `:c4v_real` explicitly.  There is no automatic
model-based switch, because silently changing existing SU runs would make old
checkpoints and results ambiguous.

The new path rejects non-real tensors, non-square lattices, non-TFIsing models,
non-single-site patterns, nonuniform virtual dimensions, or non-C4v input with
an `ArgumentError` that names the violated contract.  The pre-SU C4v assertion
also catches an example that forgot to pass `C4v_restriction` to
`optimise_ipeps`.

## Materialized-State Contract

Let `P_C4v` be `C4v_restriction` and `N(A) = A / norm(A)`.  The state seen by
all physics code is

```text
A0    = N(P_C4v(Araw))
Aphys = c4v_real_su(A0, model, tau)
```

`c4v_real_su` includes its final C4v projection and normalization.  Its public
postconditions are:

- the input is unchanged;
- the output has the same shape, real element type, and virtual D;
- the output is normalized and C4v symmetric to numerical precision; and
- `tau == 0` returns `A0` without entering a factorization.

The four internal directional tensors are implementation intermediates, not
states that may be passed to an environment or saved as physical checkpoints.

The existing optimizer already applies `restriction_ipeps` before `energy`, and
`energy` calls `build_A`.  `build_A` remains the single switch point for the SU
mode.  Initialization, optimization, and observables must all use the same
helper beneath `build_A`, so the SU layer is applied exactly once on every path.
The inactive runtime overload must be removed or routed through the same helper;
it may not retain a second SU implementation.

## TFIsing Imaginary-Time Layer

Split the model Hamiltonian using the same local operator convention as the
existing `TFIsing{Square}` energy implementation:

```text
H = Hzz + Hx
```

Use a real second-order (Strang) layer:

```text
Gx(tau/2)
four rotated self-bond Gzz(tau) updates
Gx(tau/2)
```

where `Gx(dt) = exp(-dt Hx)` acts on the physical leg and
`Gzz(dt) = exp(-dt Hzz)` acts on a nearest-neighbor self-bond.  For the usual
negative-coupling TFIsing convention these exponentials contain the
corresponding positive `J` and `h` coefficients; deriving them from the model
operators avoids duplicating a sign or spin-versus-Pauli convention.

`tau` is defined as the step used by each directional self-bond update.  The
four-turn convention updates each virtual leg once and follows the existing
C4v self-bond sweep; it is a parameterization hyperparameter, not a claim that
the layer equals a particular total physical evolution time.

The gate builder is TFIsing-specific and may not call the generic Heisenberg
fallback.  Gate tests compare its dense matrix against direct exponentiation of
the analytic TFIsing local operators.

## Real C4v Self-Bond Update

For the current direction, contract two copies of the same real site tensor
with `Gzz` into the symmetric pair matrix M using the established index order.
Then:

1. restore exact numerical symmetry with `M = (M + transpose(M)) / 2`;
2. compute the thin SVD of M;
3. for each retained right singular vector `v_i`, compute
   `lambda_i = dot(v_i, M * v_i)`;
4. reject `lambda_i < -tol_psd * scale`;
5. clamp only roundoff-sized negative values to zero; and
6. form `X = V[:, 1:D] * Diagonal(sqrt.(lambda[1:D]))`.

For a real symmetric positive-semidefinite pair matrix this gives the truncated
one-factor representation `M_D = X * transpose(X)`.  One X is essential: an
ordinary `U*sqrt(S)` / `sqrt(S)*V'` split assigns two incompatible tensors to
the same C4v site.

Reshape X into the updated site tensor, rotate virtual legs by 90 degrees, and
repeat four times.  After the fourth turn restore the original orientation,
apply the second onsite half-step, project with `C4v_restriction`, and normalize.

Do not replace a negative Takagi value by its absolute value.  A significant
negative value means that a real `X*X^T` representation is invalid and complex
support would be required; the first implementation reports this rather than
changing the state silently.

## Gauge Continuity and AD

The real factor has an orthogonal virtual gauge.  Align X to the input tensor's
unfolding B0 by an orthogonal Procrustes factor O and use `X * O`, with
`O * transpose(O) = I`.  This leaves `X * transpose(X)` unchanged while reducing
sign flips and rotations of nearly degenerate retained channels.

The Procrustes construction and Takagi construction use SVD and reuse TeneT's
existing regularized SVD reverse rule.  The implementation is functional:
comprehensions, reshapes, permutations, and newly allocated arrays replace
indexed mutation and `Zygote.Buffer` writes.  Model operator construction and
diagnostic-only branch choices may be marked non-differentiable, but no
physical gate, tensor contraction, singular value, retained factor, C4v
projection, normalization, boundary solve, or energy contribution is detached
from AD.

Gradient validation uses real directional derivatives.  A normalized random
direction dA is tested with centered differences over a decreasing epsilon
sequence; acceptance requires a stable error plateau, not agreement at one
hand-picked epsilon.

## Checkpoints and Observables

The current `No.<iter>.jld2` file stores the raw optimizer variable under
`bcipeps`.  Preserve that key for optimizer restart compatibility and add:

- `physical_bcipeps`: the post-SU, normalized C4v tensor in the conventional
  six-dimensional one-site layout;
- `SUmode` and `SUτ`;
- model parameters needed to reconstruct the gates;
- iteration and chi; and
- a state-semantics/version marker.

`bcipeps` is never used directly for energy, correlation length, transfer
spectrum, or magnetization in the new workflow.  New measurement tools prefer
`physical_bcipeps`.  When reading an older checkpoint without that key they
retain the old `bcipeps` semantics instead of guessing that SU was present.

Optimizer callbacks receive the same materialization closure used by the
objective, ensuring that the saved physical tensor is byte-consistent with the
state whose energy was evaluated.  Observable parameter copies must preserve
`ifSU`, `SUmode`, and `SUτ`; loading an already materialized tensor for a
standalone measurement disables SU to avoid applying the layer twice.

## Error Handling and Diagnostics

The new path fails with context rather than falling back to generic SU.  Error
messages include the direction/turn, tau, retained spectrum, and relevant
residual for:

- a non-real or non-C4v input;
- a nonsymmetric pair matrix;
- a negative retained Takagi value beyond tolerance;
- a nonfinite factor, output, energy, or gradient;
- a Takagi reconstruction residual beyond tolerance; or
- a post-SU C4v residual beyond tolerance.

Diagnostics are ordinary values or ignored by AD and are off by default in
production.  Enabling diagnostics must not change the returned tensor.

## Testing

Focused unit tests cover:

- TFIsing onsite and bond gates against direct dense exponentiation;
- pair-matrix index order and symmetry against a direct contraction;
- real Takagi reconstruction for full rank and fixed-D truncation;
- rejection of a deliberately indefinite symmetric matrix;
- no mutation or storage aliasing;
- exact `tau == 0` behavior;
- output shape, real element type, normalization, and full C4v residual;
- rotation bookkeeping, proving every virtual leg is updated once;
- generic `SUmode == :generic` behavior remains unchanged; and
- checkpoint raw/physical keys and no-double-SU observable behavior.

AD tests cover:

- an SU-only scalar loss versus centered directional finite differences; and
- the complete `raw -> C4v -> SU -> VUMPS -> energy` graph at `D = 2`,
  `chi = 8`, and `tau = 0.01`.

The full-graph test runs on CPU with deterministic real `Float64` input.  It
requires finite energy and gradient, a stable finite-difference comparison, and
completion of a few short LBFGS steps with a lower best energy than the initial
evaluation.  It is an integration test, not a precision benchmark for the
critical energy.

## Paired Research Smoke

Add a repository-native TFIsing C4v example mode that runs two short jobs from
the same real raw tensor and random seed:

1. direct C4v with `ifSU = false` (equivalent to the `tau = 0` materializer);
2. C4v SU with `ifSU = true`, `SUmode = :c4v_real`, and `SUτ = 0.01`.

Both use `D = 2`, `chi = 8`, identical optimizer limits, boundary tolerances,
and measurement code.  Record energy, gradient norm, correlation length, and
the leading transfer spectrum.  The smoke answers whether the new graph is
correct, stable, and measurably distinct.  It does not establish improved
extrapolation.

Only after these gates pass should the experiment advance to D=3 and larger
chi.  The later comparison must use fresh converged environments and matched
seeds/settings, and judge improvement from correlation-length range and scaling
stability rather than iteration number or fit R-squared alone.

## Success Criteria

The implementation is ready for larger research runs when:

- all existing tests and the new focused tests pass;
- the full `D = 2`, `chi = 8` forward/backward is finite and agrees with a
  directional finite difference;
- the post-SU tensor is real, normalized, and C4v symmetric;
- raw and physical checkpoints round-trip with unambiguous semantics;
- the paired smoke completes and produces internally consistent observables;
  and
- no existing generic SU workflow changes unless it opts into `:c4v_real`.
