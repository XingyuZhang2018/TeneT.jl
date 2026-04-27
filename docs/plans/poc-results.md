# PoC Results — Fixed-Point iPEPS + MCF

## E1 (baseline: env=A, H=a, decomp=X, MCF=on) — **FAILED on iter 2**

### Setup
- D=2, χ=8, 1×1 unit cell, rotated AFM Heisenberg (Jx=Jy=-1, Jz=1).
- Initial state: LBFGS-converged jld2 (E_LBFGS = -0.660231, gnorm = 9.55e-8).
- cfg: `iPEPSFixedPointConfig(env_mode=:A, H_eff_mode=:a, decompose_method=:X, mcf_ifignore_gauge=false, outer_maxiter=50)`.
- Run: `julia --project=. examples/Heisenberg/Heisenberg_Square_FixedPoint.jl E1`.

### Trace

**iter 1:**
- λ = -0.7499999999999908   ← suspicious: equals -3/4 J = isolated 2-site singlet energy
- E = -0.5                  ← expected ≈ -0.66 from LBFGS
- dλ = Inf                  ← first iter
- dA = 3.80                 ← huge departure from LBFGS-converged A (which had norm 1)
- **trunc_err = 194.84**    ← catastrophic truncation
- internal `local_min_norm` LBFGS warnings (didn't fully converge gauge optimisation)

**iter 2:**
- crash in V sweep: `KrylovKit.geneigsolve: ArgumentError: initial vector should not have norm zero`
- after H sweep + :X decomp + MCF, A[1,1] has near-zero norm; `build_phi(A, A, :V)` produces zero.

### Root cause analysis

1. **All low-level operators verified correct** (N_op, H_op, build_phi, decompose_phi)
   via strong sanity checks (`<φ|N_op|φ> ≈ contract_n_12`, `e_bond ≈ energy_value`).
2. **The failure is the conceptual gap warned about in the design doc**:
   > "8-leg φ_new is generally NOT in the variational manifold {φ : φ = A·A};
   > the projection back is lossy and method-dependent."
3. `geneigsolve` searches the **unconstrained** 8-leg vector space and finds the
   true 2-site lowest eigenstate of H_eff vs N_eff. For Heisenberg with
   converged env, this is essentially the spin-singlet (energy = -3/4 J),
   which has bond rank far higher than D=2.
4. `decompose_phi(:X, D_max=2)` truncates the 8-leg φ_new back to a rank-D
   PEPS tensor, discarding ~194 of the eigenvalue² mass. The retained A_new
   bears no resemblance to the LBFGS-converged starting tensor.
5. `local_min_norm` then operates on a near-zero tensor and produces a
   degenerate gauge; combined with the rank-truncation loss, A is effectively
   lost.

### Why MCF cannot rescue this
MCF was hypothesised to "stabilise" the iteration by fixing the gauge
ambiguity. But here the failure is not gauge ambiguity — it is **manifold
escape**: geneigsolve leaves the PEPS variational manifold by O(1) factor
in a single step, before MCF even runs. MCF on a half-destroyed A cannot
recover.

### Implication for criterion #4 (key science Q)
Cannot answer "is MCF essential?" because the iteration fails for reasons
unrelated to gauge ambiguity. **E2 (MCF off) would also fail by the same
mechanism**, possibly faster, but the comparison would be uninformative.

### What would be needed to make this work
The PoC's hypothesis that "geneigsolve + lossy projection" iterates toward
a PEPS ground state is wrong in this form. To make it work would require
constraining the search to the PEPS manifold, e.g.:

- **Steepest-descent on the manifold**: parameterise φ = A·A directly as a
  function of A, and solve a 5-leg generalised eigenvalue problem on A
  (rather than 8-leg on φ). This is essentially a 2-site full-update (FU)
  step but with eigenvalue rather than gradient updates. There's existing
  TeneT machinery (`FUOptimize`) that does similar work.
- **Trust-region step / damped update**: A_new = (1 - α) A_old + α A_after_decomp,
  with α ∈ (0, 1) chosen so trunc_err stays small.
- **Initialise φ_old = build_phi(A_old, A_old)` as the geneigsolve starting
  vector** (this we do!) and **bias geneigsolve toward staying near it**,
  e.g. by a small Krylov subspace or by extrapolation. Plan already uses
  `krylovdim=20`, which is generous; reducing it might help but probably
  not change the fundamental mechanism.

### Status
- Per plan instructions, STOPPED. Did not run E2-E5.
- All implementation and 51 unit/integration tests pass.
- The failure is algorithmic, not a code bug.

### Recommendation
- Do NOT pursue this approach further as written.
- Document findings (Task 21) and close out.
- A follow-up PoC could try the manifold-constrained variant (parameterise
  on A directly), which is closer to existing FUOptimize.
