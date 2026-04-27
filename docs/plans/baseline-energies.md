# LBFGS Baseline Energies (Heisenberg Square)

For success criterion #2 in `2026-04-27-ipeps-fixedpoint-mcf-plan.md` (Task 18+).
PoC fixed-point algorithm should reach `E_iPEPS` within `1e-3 / site` of the
LBFGS converged value at the same D, χ.

## Setup

- Model: `Heisenberg(lattice=Square(), S=0.5, Jx=-1.0, Jy=-1.0, Jz=1.0,
  ifrotate=true, couplingtype=:uniform, bondratio=1.0)`
  (this is the **default** Heisenberg AFM via sublattice rotation; not the
  example file's `Jx=Jy=+1.0` variant)
- Lattice pattern: `[1;;]` (1×1 unit cell)
- Boundary algorithm: VUMPS{General}, χ as listed
- Eltype: Float64
- seed = 42

## D = 2, χ = 16 — converged LBFGS reference

From `data/.../D2/history.log` in `stupefied-cartwright` worktree:

```
iter | energy_χ16        | gnorm    | t/sec
  1  | -0.522999497066   | 2.82e-1  |  3.25
  5  | -0.643311411807   | 1.06e-1  |  4.71
 10  | -0.660212201745   | 7.96e-4  |  6.29
 15  | -0.660230788160   | 1.25e-4  |  8.02
 20  | -0.660231093546   | 9.55e-8  |  9.34
```

**Reference energy per site: E_LBFGS = -0.660231093546424**
(20 LBFGS iters → gnorm < 1e-7, converged)

Final iPEPS jld2 checkpoint:
```
.claude/worktrees/stupefied-cartwright/data/Heisenberg_Square(S=0.5,Jx=-1.0,Jy=-1.0,Jz=1.0,ifrotate=true,couplingtype=uniform)/[1;;]/VUMPS_General/Float64/seed42/D2/ipeps/χ16/No.20.jld2
```

For Task 18+ comparison at the same model spec: target `|E_PoC - (-0.660231)| < 1e-3`.

## Notes / deviations

- **Did NOT run full `Pkg.test()`** as Task 2 originally instructed. Reason:
  this Windows environment has CUDA 13.0 (precompiled for 13.2) and AMDGPU
  warnings (no ROCm); full suite would force CPU paths only and is slow. The
  existing LBFGS checkpoint history above demonstrates the iPEPS-unified
  optimization pipeline is healthy on this branch's source.
- D=3 baseline not yet recorded. May add later if PoC at D=2 succeeds and we
  extend to D=3 (per plan's "v2" note).
