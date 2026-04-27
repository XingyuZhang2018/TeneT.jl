# iPEPS Fixed-Point Eigenvalue Iteration with MCF — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a proof-of-concept iPEPS optimizer that replaces gradient descent with a 2-site generalized eigenvalue iteration `H_eff φ = λ N_eff φ` gauge-fixed by `local_min_norm` (MCF), and evaluate whether it converges and whether MCF is essential.

**Architecture:** New file `src/ipeps_optimize/optimize_fixedpoint.jl` exposing `optimize_ipeps_fixedpoint(A, χ, model, params, config)`. Two bond sweeps (H + V) per outer step, each: build 8-leg `φ` from `A·A`, build linear operators `H_op`/`N_op` from boundary VUMPS env, call `KrylovKit.geneigsolve`, decompose `φ_new` back via SVD with reflection-symmetrization, MCF gauge-fix.

**Tech Stack:** Julia 1.9+, TeneT.jl on `iPEPS-unified` branch, KrylovKit.geneigsolve, TensorOperations `@tensor`, existing `local_min_norm`, `leading_boundary`.

**Design doc:** [docs/plans/2026-04-27-ipeps-fixedpoint-mcf-design.md](2026-04-27-ipeps-fixedpoint-mcf-design.md)

**Branch / worktree:** This plan was written from worktree `claude/mystifying-tesla-242a2c` (which is on an old VUMPS-only branch). Implementation requires a fresh worktree based on `iPEPS-unified`. Task 1 handles this.

**Leg conventions (TeneT.jl):** A is 5-leg `(l, d, r, u, p)`:
- 1 = left, 2 = down, 3 = right, 4 = up, 5 = physical
- (6 = unit cell index when stored as `StructArray`)

---

## Phase 0: Setup

### Task 1: Create implementation worktree

**Files:** none modified; creates new worktree directory

**Step 1: Verify base branch has the expected files**

Run:
```bash
git -C "D:/1 - research/1.26 - iPEPS_opt/TeneT.jl" branch -a | grep iPEPS-unified
ls "D:/1 - research/1.26 - iPEPS_opt/TeneT.jl/src/ipeps_optimize/"
```
Expected: `iPEPS-unified` listed; directory contains `optimize.jl`, `restriction.jl`, `precondition.jl`, `init.jl`, etc.

**Step 2: Create new worktree off `iPEPS-unified`**

Run:
```bash
git -C "D:/1 - research/1.26 - iPEPS_opt/TeneT.jl" worktree add \
  ".claude/worktrees/ipeps-fixedpoint-mcf" \
  -b claude/ipeps-fixedpoint-mcf iPEPS-unified
```
Expected: new directory at `.claude/worktrees/ipeps-fixedpoint-mcf`, branch `claude/ipeps-fixedpoint-mcf` created off `iPEPS-unified`.

**Step 3: Copy design doc + this plan into the new worktree**

```bash
mkdir -p "D:/1 - research/1.26 - iPEPS_opt/TeneT.jl/.claude/worktrees/ipeps-fixedpoint-mcf/docs/plans"
cp "D:/1 - research/1.26 - iPEPS_opt/TeneT.jl/.claude/worktrees/mystifying-tesla-242a2c/docs/plans/2026-04-27-ipeps-fixedpoint-mcf-design.md" \
   "D:/1 - research/1.26 - iPEPS_opt/TeneT.jl/.claude/worktrees/ipeps-fixedpoint-mcf/docs/plans/"
cp "D:/1 - research/1.26 - iPEPS_opt/TeneT.jl/.claude/worktrees/mystifying-tesla-242a2c/docs/plans/2026-04-27-ipeps-fixedpoint-mcf-plan.md" \
   "D:/1 - research/1.26 - iPEPS_opt/TeneT.jl/.claude/worktrees/ipeps-fixedpoint-mcf/docs/plans/"
```

**Step 4: Commit design doc + plan in new worktree**

```bash
cd "D:/1 - research/1.26 - iPEPS_opt/TeneT.jl/.claude/worktrees/ipeps-fixedpoint-mcf"
git add docs/plans/
git commit -m "docs(plans): import iPEPS fixed-point + MCF design and plan"
```

**From this task on, the working directory is the new worktree.**

---

### Task 2: Verify baseline test suite is healthy

**Files:** none modified

**Step 1: Run a fast subset of tests as a baseline smoke check**

Run:
```bash
cd "D:/1 - research/1.26 - iPEPS_opt/TeneT.jl/.claude/worktrees/ipeps-fixedpoint-mcf"
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e 'using Pkg; Pkg.test()'  # full test suite
```
Expected: all tests pass. If something is broken on `iPEPS-unified`, fix or skip BEFORE proceeding (this work depends on a healthy baseline).

**Step 2: Note baseline LBFGS energy for Heisenberg D=2-3**

This is needed for success criterion #2. Either run an existing example or read it from the project's data files. Record the value(s) in a note for later.

Run an existing Heisenberg LBFGS example for D=2, χ=8 with a small max iter (e.g. 50) and capture final energy. Document in `docs/plans/baseline-energies.md`.

**No commit required** for this task (read-only).

---

## Phase 1: Config struct

### Task 3: Add `iPEPSFixedPointConfig` struct

**Files:**
- Create: `src/ipeps_optimize/optimize_fixedpoint.jl`
- Modify: `src/TeneT.jl` (add `include("ipeps_optimize/optimize_fixedpoint.jl")` after the existing optimize.jl include)
- Test: `test/test_optimize_fixedpoint.jl` (new)
- Modify: `test/runtests.jl` (add `include("test_optimize_fixedpoint.jl")`)

**Step 1: Write failing test**

Create `test/test_optimize_fixedpoint.jl`:

```julia
@testset "iPEPSFixedPointConfig" begin
    cfg = iPEPSFixedPointConfig()
    @test cfg.env_mode == :A
    @test cfg.H_eff_mode == :a
    @test cfg.decompose_method == :X
    @test cfg.mcf_ifignore_gauge == false
    @test cfg.outer_maxiter == 200

    cfg2 = iPEPSFixedPointConfig(env_mode=:C, mcf_ifignore_gauge=true)
    @test cfg2.env_mode == :C
    @test cfg2.mcf_ifignore_gauge == true
end
```

Add `include("test_optimize_fixedpoint.jl")` at the end of `test/runtests.jl`.

**Step 2: Run, verify it fails**

```bash
julia --project=. -e 'using Pkg; Pkg.test(test_args=["iPEPSFixedPointConfig"])'
```
Expected: FAIL with "UndefVarError: iPEPSFixedPointConfig not defined".

**Step 3: Implement**

Create `src/ipeps_optimize/optimize_fixedpoint.jl`:

```julia
# iPEPS fixed-point eigenvalue iteration with MCF gauge fix.
# PoC for the alternative to gradient-based optimization.
# See docs/plans/2026-04-27-ipeps-fixedpoint-mcf-design.md.

using Parameters
using KrylovKit
using TensorOperations
using LinearAlgebra

export iPEPSFixedPointConfig, optimize_ipeps_fixedpoint

@with_kw struct iPEPSFixedPointConfig
    env_mode::Symbol           = :A          # :A full / :B warm / :C 1-step
    env_warm_steps::Int        = 5
    H_eff_mode::Symbol         = :a          # :a central bond / :b all 7
    decompose_method::Symbol   = :X          # :X symmetrize / :Y avg / :Z lazy
    geneig_krylovdim::Int      = 20
    geneig_tol::Float64        = 1e-10
    geneig_maxiter::Int        = 100
    mcf_ifignore_gauge::Bool   = false
    outer_maxiter::Int         = 200
    outer_tol_λ::Float64       = 1e-8
    outer_tol_A::Float64       = 1e-7
    log_every::Int             = 1
    save_every::Int            = 10
end

# build_phi, make_N_op, make_H_op, decompose_phi, sweep_bond!,
# optimize_ipeps_fixedpoint will be added in subsequent tasks.
```

Add to `src/TeneT.jl` after the `include("ipeps_optimize/optimize.jl")` line:
```julia
include("ipeps_optimize/optimize_fixedpoint.jl")
```

**Step 4: Run, verify it passes**

```bash
julia --project=. -e 'using Pkg; Pkg.test(test_args=["iPEPSFixedPointConfig"])'
```
Expected: PASS.

**Step 5: Commit**

```bash
git add src/TeneT.jl src/ipeps_optimize/optimize_fixedpoint.jl test/test_optimize_fixedpoint.jl test/runtests.jl
git commit -m "feat(ipeps_optimize): add iPEPSFixedPointConfig struct"
```

---

## Phase 2: `build_phi` (form 2-site bond tensor)

### Task 4: Test for `build_phi` — horizontal direction

**Files:**
- Test: `test/test_optimize_fixedpoint.jl` (extend)

**Step 1: Add failing test**

Append to `test/test_optimize_fixedpoint.jl`:

```julia
@testset "build_phi horizontal" begin
    D, d = 2, 2
    A = randn(D, D, D, D, d)  # legs (l, d, r, u, p)
    φ = TeneT.build_phi(A, A, Val(:H))
    # φ has 8 legs: (l_left, d_left, p_left, u_left, d_right, r_right, u_right, p_right)
    # We define ordering as: φ[l, d_l, u_l, p_l, d_r, u_r, r, p_r]
    @test ndims(φ) == 8
    @test size(φ) == (D, D, D, d, D, D, D, d)

    # Sanity: contraction value should equal explicit @tensor
    φ_ref = similar(φ)
    @tensor φ_ref[l, dl, ul, pl, dr, ur, r, pr] := A[l, dl, c, ul, pl] * A[c, dr, r, ur, pr]
    @test φ ≈ φ_ref
end
```

**Step 2: Run, verify it fails**

```bash
julia --project=. -e 'using Pkg; Pkg.test(test_args=["build_phi horizontal"])'
```
Expected: FAIL with `build_phi` undefined.

**Step 3: Implement**

Append to `src/ipeps_optimize/optimize_fixedpoint.jl`:

```julia
"""
    build_phi(A_l, A_r, ::Val{:H})

Form a 2-site horizontal-bond tensor `φ` by contracting the right leg of `A_l`
with the left leg of `A_r`.

Output legs: `(l, d_l, u_l, p_l, d_r, u_r, r, p_r)`.
"""
function build_phi(A_l, A_r, ::Val{:H})
    @tensor φ[l, dl, ul, pl, dr, ur, r, pr] :=
        A_l[l, dl, c, ul, pl] * A_r[c, dr, r, ur, pr]
    return φ
end
```

**Step 4: Run, verify it passes**

```bash
julia --project=. -e 'using Pkg; Pkg.test(test_args=["build_phi horizontal"])'
```
Expected: PASS.

**Step 5: Commit**

```bash
git add src/ipeps_optimize/optimize_fixedpoint.jl test/test_optimize_fixedpoint.jl
git commit -m "feat(ipeps_optimize): add build_phi for horizontal bond"
```

---

### Task 5: Test + implement `build_phi` — vertical direction

**Files:**
- Test: `test/test_optimize_fixedpoint.jl` (extend)
- Modify: `src/ipeps_optimize/optimize_fixedpoint.jl`

**Step 1: Add failing test**

Append:
```julia
@testset "build_phi vertical" begin
    D, d = 2, 2
    A = randn(D, D, D, D, d)
    φ = TeneT.build_phi(A, A, Val(:V))
    # Vertical bond: A_top's down (2) contracts with A_bot's up (4).
    # φ legs: (l_t, d_t≡bond, ... wait, let's be explicit)
    # We define: top tensor's leg 2 (down) joins bottom tensor's leg 4 (up).
    # Output legs: (l_t, p_t, r_t, l_b, d_b, r_b, u_b, p_b)? Need to fix order.
    # Convention chosen: (l_top, u_top, r_top, p_top, l_bot, d_bot, r_bot, p_bot)
    @test ndims(φ) == 8
    @test size(φ) == (D, D, D, d, D, D, D, d)

    φ_ref = similar(φ)
    @tensor φ_ref[lt, ut, rt, pt, lb, db, rb, pb] :=
        A[lt, c, rt, ut, pt] * A[lb, db, rb, c, pb]
    @test φ ≈ φ_ref
end
```

**Step 2: Run, verify it fails**

```bash
julia --project=. -e 'using Pkg; Pkg.test(test_args=["build_phi vertical"])'
```
Expected: FAIL with method error (no `Val{:V}` method).

**Step 3: Implement**

Append:
```julia
"""
    build_phi(A_t, A_b, ::Val{:V})

Form a 2-site vertical-bond tensor `φ` by contracting the down leg of `A_t`
with the up leg of `A_b`.

Output legs: `(l_t, u_t, r_t, p_t, l_b, d_b, r_b, p_b)`.
"""
function build_phi(A_t, A_b, ::Val{:V})
    @tensor φ[lt, ut, rt, pt, lb, db, rb, pb] :=
        A_t[lt, c, rt, ut, pt] * A_b[lb, db, rb, c, pb]
    return φ
end
```

**Step 4: Run, verify it passes**

**Step 5: Commit**

```bash
git commit -am "feat(ipeps_optimize): add build_phi for vertical bond"
```

---

## Phase 3: `make_N_op` (norm operator on 2-site φ)

### Task 6: Test for `make_N_op` — horizontal

**Files:**
- Test: `test/test_optimize_fixedpoint.jl` (extend)
- Modify: `src/ipeps_optimize/optimize_fixedpoint.jl`

**Step 1: Add failing test**

```julia
@testset "make_N_op horizontal — sanity" begin
    using TeneT: leading_boundary, init_VUMPSRuntime, build_A
    D, d, χ = 2, 2, 8
    Ni, Nj = 1, 1
    pattern = ones(Int, Ni, Nj)
    # Build a random iPEPS, converge VUMPS env
    A_raw = randn(D, D, D, D, d, 1)
    A_raw /= norm(A_raw)
    params = make_default_params(D=D, χ=χ)  # helper, see Task 7
    A = build_A(A_raw, params)
    rt = init_VUMPSRuntime(A, χ, params.boundary_alg)
    rt, _ = leading_boundary(rt, A, params.boundary_alg)

    # Build φ from A·A and N_op acting on φ
    φ = TeneT.build_phi(A[1,1], A[1,1], Val(:H))
    N_op = TeneT.make_N_op(rt, A, Val(:H), params)

    Nφ = N_op(φ)
    # The inner product <φ|N_op|φ> should equal <ψ|ψ>_per-bond up to env normalization
    val = sum(conj(φ) .* Nφ)
    @test isfinite(val)
    @test real(val) > 0   # PSD norm operator
end
```

(Note: `make_default_params` helper to be added — see Task 7.)

**Step 2: Run, verify it fails**

Expected: FAIL — `make_N_op` not defined.

**Step 3: Implement**

Append to `src/ipeps_optimize/optimize_fixedpoint.jl`:

```julia
"""
    make_default_params(; D, χ)

Helper to build a minimal `iPEPSOptimize` params object for testing.
"""
function make_default_params(; D::Int, χ::Int)
    # NB: defer to existing iPEPSOptimize / boundary_alg defaults
    boundary_alg = VUMPS(χ=χ, maxiter=20, tol=1e-10, ifsave_env=false, verbosity=0,
                        ifparallel=false, forloop_iter=1)
    return iPEPSOptimize(; D=D, χ=χ, boundary_alg=boundary_alg,
                          model=Heisenberg{Square}(1.0),  # placeholder
                          pattern=ones(Int, 1, 1),
                          verbosity=0)
end

"""
    make_N_op(rt, A, ::Val{:H}, params) -> Function

Return a function `N_op(φ) -> Nφ` that applies the 2-site norm operator on a
horizontal bond. The operator is the boundary VUMPS environment around the
2-site region, contracted with `φ` and outer A's.

For 1×1 unit cell, the env around 2-site horizontal region is built from
ACu, ACd, FLo, FRo, ARu, ARd of the converged VUMPS runtime.
"""
function make_N_op(rt::VUMPSRuntime, A, ::Val{:H}, params)
    # IMPLEMENTATION:
    # In 1×1 unit cell, the 2-site horizontal region needs:
    #   left environment FLo[1,1]
    #   right environment FRo[1,1]  (or shifted; for 1×1 same)
    #   top: ACu[1,1] and ACu[1,1] adjacent — for 2 sites stacked horizontally,
    #        the top boundary covers both columns
    #   bottom: ACd similarly
    # Define N_op(φ) by contracting double-layer:
    #     <ψ_φ | ψ_φ> with the slot for our 2-site φ open.
    # This is a contraction returning a tensor of same shape as φ.
    env = ObsEnv(rt, A, params.boundary_alg)
    @unpack ACu, ARu, ACd, ARd, FLo, FRo = env

    # Capture the relevant tensors. For 1×1: i = j = 1.
    FL = FLo[1,1]
    FR = FRo[1,1]
    Au_l, Au_r = ACu[1,1], ARu[1,1]
    Ad_l, Ad_r = ACd[1,1], ARd[1,1]

    function N_op(φ)
        # φ has legs (l, dl, ul, pl, dr, ur, r, pr).
        # Define double-layer contraction with conj(A)·A on the 4 outer sites
        # plus φ slot. Specifics: TODO — full @tensor expression to be written
        # by following the pattern in `precondition.jl`'s `contract_n_*` helpers
        # but adapted for 2-site span.
        error("make_N_op: not yet implemented (Task 7 fills this in)")
    end
    return N_op
end

# Vertical version analogous; defined in Task 8.
```

**Step 4 + 5:** This task is split — the test is added now (failing); the contraction is implemented in **Task 7**. Commit only the helper + skeleton:

```bash
git commit -am "feat(ipeps_optimize): make_N_op skeleton + default params helper"
```

---

### Task 7: Implement `make_N_op` horizontal contraction

**Files:**
- Modify: `src/ipeps_optimize/optimize_fixedpoint.jl`

**Step 1: Re-read the test from Task 6** (already failing).

**Step 2: Implement the actual contraction**

Replace the body of `N_op(φ)` inside `make_N_op` with the real `@tensor` expression. This requires writing out the double-layer contraction for 2-site φ. Reference: `precondition.jl`'s 1-site `Mumap_parallel`, but with the φ slot covering 2 sites.

Pseudocode (precise indices to be derived during implementation):
```
N_op(φ) =
    contract(FL, ACu_l, ACu_r, conj(A_above), conj(A_above), ACd_l, ACd_r,
             conj(A_below), conj(A_below), FR, conj(φ_slot))
where the inner-row pattern (left to right):
    FL — [Au_l (on top)]   [Au_r (on top)]   — FR
         [φ_left]           [φ_right]
    FL — [Ad_l (on bottom)] [Ad_r (on bottom)] — FR
```

Concretely, the kernel for the 1×1 case:
```julia
function N_op(φ)
    # Build double-layer tensor for the 2-site span by contracting φ with conj(φ_slot)
    # See examples/Heisenberg/*.jl `contract_n_12` for related patterns
    Nφ = ...  # @tensor expression here, leaving φ open
    return Nφ
end
```

The implementer should:
1. Look at `contract_n_11` and `contract_n_12` in `src/contraction/unit_contraction/basic.jl`
2. Adapt the 12-leg version to 2-site span by stacking the appropriate pieces
3. Verify the leg ordering matches `build_phi` output

**Step 3: Run test from Task 6, verify it passes**

**Step 4: Commit**

```bash
git commit -am "feat(ipeps_optimize): implement make_N_op for horizontal bond"
```

---

### Task 8: `make_N_op` vertical version

**Files:** Modify `src/ipeps_optimize/optimize_fixedpoint.jl`, add test in `test/test_optimize_fixedpoint.jl`.

**Step 1: Add failing test** (analogous to Task 6 but `Val(:V)` and check for vertical).

**Step 2: Run, verify failure.**

**Step 3: Implement vertical contraction** (90° rotation of horizontal logic — different env tensors active).

**Step 4: Run, verify pass.**

**Step 5: Commit.**

---

## Phase 4: `decompose_phi` (φ → A_new)

### Task 9: Test + implement `decompose_phi` method `:Z` (lazy SVD, simplest)

**Files:** Modify `optimize_fixedpoint.jl`, extend test file.

**Step 1: Add failing test**

```julia
@testset "decompose_phi :Z horizontal" begin
    D, d = 2, 2
    A = randn(D, D, D, D, d)
    φ = TeneT.build_phi(A, A, Val(:H))
    A_new, trunc_err = TeneT.decompose_phi(φ, Val(:H); method=:Z, D_max=D)
    @test size(A_new) == size(A)
    @test trunc_err >= 0
    # Round-trip: rebuilding φ should be close to original (since input was already rank-D bond)
    φ_back = TeneT.build_phi(A_new, A_new, Val(:H))
    # φ_back ≈ φ up to gauge — at least norms match
    @test abs(norm(φ_back) - norm(φ)) / norm(φ) < 0.5  # generous; not gauge-invariant
end
```

**Step 2: Run, verify failure.**

**Step 3: Implement**

```julia
"""
    decompose_phi(φ, ::Val{:H}; method=:X, D_max::Int) -> (A_new, trunc_err)

Decompose a 2-site bond tensor `φ` into an updated 1-site iPEPS tensor `A_new`,
truncating the central bond to dimension `D_max`. Returns truncation error
(sum of squared discarded singular values).

Methods:
- `:Z` lazy: SVD φ across central bipartition, take left factor as A_new.
- `:Y` avg: SVD then average left and right factors.
- `:X` symmetrize-first: enforce reflection symmetry on φ, then eigendecompose.
"""
function decompose_phi(φ, dir::Val{:H}; method::Symbol=:X, D_max::Int)
    if method == :Z
        return _decompose_phi_lazy(φ, dir, D_max)
    elseif method == :Y
        return _decompose_phi_avg(φ, dir, D_max)
    elseif method == :X
        return _decompose_phi_symmetrize(φ, dir, D_max)
    else
        error("Unknown decompose method: $method")
    end
end

function _decompose_phi_lazy(φ, ::Val{:H}, D_max::Int)
    # φ: (l, dl, ul, pl, dr, ur, r, pr)
    # Bipartition: (l, dl, ul, pl) | (dr, ur, r, pr) → SVD across central bond
    Dl, Ddl, Dul, dl_sz, Ddr, Dur, Dr, dr_sz = size(φ)
    M = reshape(φ, Dl * Ddl * Dul * dl_sz, Ddr * Dur * Dr * dr_sz)
    U, S, V = svd(M)
    # Truncate to D_max
    keep = min(D_max, length(S))
    trunc_err = sum(abs2, S[keep+1:end])
    Ut = U[:, 1:keep] .* sqrt.(S[1:keep]')   # U_l · √S
    # Reshape back to A_new shape (l, d, r, u, p) where r is the new central bond
    A_new_left = reshape(Ut, Dl, Ddl, Dul, dl_sz, keep)
    # Permute to (l, d, r, u, p) convention: (Dl, Ddl, keep, Dul, dl_sz)
    A_new = permutedims(A_new_left, (1, 2, 5, 3, 4))  # (l, d, r, u, p)
    return A_new, trunc_err
end
```

**Step 4: Run, verify pass.**

**Step 5: Commit.**

---

### Task 10: `decompose_phi` method `:Y` (avg)

**Files:** Modify `optimize_fixedpoint.jl`, extend test file.

**Step 1: Add failing test** (analogous to Task 9 but with `method=:Y`).

**Step 2: Run, verify failure.**

**Step 3: Implement**

```julia
function _decompose_phi_avg(φ, ::Val{:H}, D_max::Int)
    Dl, Ddl, Dul, dl_sz, Ddr, Dur, Dr, dr_sz = size(φ)
    M = reshape(φ, Dl * Ddl * Dul * dl_sz, Ddr * Dur * Dr * dr_sz)
    U, S, V = svd(M)
    keep = min(D_max, length(S))
    trunc_err = sum(abs2, S[keep+1:end])
    Ut = U[:, 1:keep] .* sqrt.(S[1:keep]')          # left factor
    Vt = sqrt.(S[1:keep]) .* V[:, 1:keep]'           # right factor (transposed)
    A_left  = reshape(Ut, Dl, Ddl, Dul, dl_sz, keep)
    A_right = reshape(Vt', keep, Ddr, Dur, Dr, dr_sz)  # legs (central, d, u, r, p)
    # Permute both to (l, d, r, u, p) convention:
    A_left  = permutedims(A_left,  (1, 2, 5, 3, 4))    # (l, d, central, u, p)
    A_right = permutedims(A_right, (1, 2, 4, 3, 5))    # (central, d, r, u, p)
    # For 1×1 unit cell, A_left and A_right both stand in for A.
    # Rename axes consistently: A_left's leg-3 = central; A_right's leg-1 = central.
    # For "averaging" we need both to share the same (l,d,r,u,p) shape.
    # Make A_right's leg-1 (central) align with A_left's leg-3:
    # treat A_right as a ((central, d, r, u, p) → (l, d, r, u, p)) by relabeling.
    # The shape is the same; the values may differ.
    @assert size(A_left) == size(A_right)
    A_new = (A_left .+ A_right) ./ 2
    return A_new, trunc_err
end
```

**Step 4: Run, verify pass.**

**Step 5: Commit.**

---

### Task 11: `decompose_phi` method `:X` (symmetrize-first)

**Files:** Modify `optimize_fixedpoint.jl`, extend test file.

**Step 1: Add failing test**

```julia
@testset "decompose_phi :X horizontal" begin
    D, d = 2, 2
    A = randn(D, D, D, D, d)
    φ = TeneT.build_phi(A, A, Val(:H))
    A_new, trunc_err = TeneT.decompose_phi(φ, Val(:H); method=:X, D_max=D)
    @test size(A_new) == size(A)
    @test trunc_err >= 0

    # Symmetrize-first: rebuilding φ should approximately match a symmetrized version of input
    φ_back = TeneT.build_phi(A_new, A_new, Val(:H))
    # Reflection of horizontal φ swaps the two sites:
    # original legs:    (l, dl, ul, pl, dr, ur, r, pr)
    # reflected legs:   (r, dr, ur, pr, dl, ul, l, pl)  (permutation 7,5,6,8,2,3,1,4)
    φ_refl = permutedims(φ, (7, 5, 6, 8, 2, 3, 1, 4))
    φ_sym  = (φ + φ_refl) / 2
    # φ_back should approximate φ_sym after truncation
    @test norm(φ_back - φ_sym) / norm(φ_sym) < 0.5  # generous
end
```

**Step 2: Run, verify failure.**

**Step 3: Implement**

```julia
function _decompose_phi_symmetrize(φ, ::Val{:H}, D_max::Int)
    # Reflection on horizontal bond: swap the two sites.
    # Original legs: (l, dl, ul, pl, dr, ur, r, pr) [indices 1..8]
    # After reflection: (r, dr, ur, pr, dl, ul, l, pl) → permutation (7,5,6,8,2,3,1,4)
    φ_refl = permutedims(φ, (7, 5, 6, 8, 2, 3, 1, 4))
    φ_sym  = (φ .+ φ_refl) ./ 2

    Dl, Ddl, Dul, dl_sz, Ddr, Dur, Dr, dr_sz = size(φ_sym)
    @assert Dl == Dr "horizontal symmetrize requires Dl == Dr"
    @assert Ddl == Ddr
    @assert Dul == Dur
    M = reshape(φ_sym, Dl * Ddl * Dul * dl_sz, Ddr * Dur * Dr * dr_sz)
    # Symmetric M = M' (after symmetrization), so use eigendecomposition for stability.
    Msym = (M + M') / 2
    F = eigen(Hermitian(Msym))
    # Sort by descending |eigenvalue|
    perm = sortperm(abs.(F.values), rev=true)
    vals = F.values[perm]
    vecs = F.vectors[:, perm]
    keep = min(D_max, length(vals))
    trunc_err = sum(abs2, vals[keep+1:end])
    # A_new = vecs · √vals (only keep top ones), reshape to A
    Σ_kept = sqrt.(complex.(vals[1:keep]))   # complex in case of negative eigenvalues
    Ut = vecs[:, 1:keep] .* Σ_kept'
    A_new_left = reshape(Ut, Dl, Ddl, Dul, dl_sz, keep)
    A_new = permutedims(A_new_left, (1, 2, 5, 3, 4))  # (l, d, r, u, p)
    # Discard imaginary part if input was real
    if eltype(φ) <: Real
        A_new = real(A_new)
    end
    return A_new, real(trunc_err)
end
```

**Step 4: Run, verify pass.**

**Step 5: Commit.**

---

### Task 12: `decompose_phi` vertical versions for `:Z`, `:Y`, `:X`

**Files:** Modify `optimize_fixedpoint.jl`, extend test file with vertical-direction analogs.

Each method needs a `Val{:V}` dispatch with the appropriate leg permutations. The reflection for vertical bond swaps top ↔ bottom.

**Step 1: Add failing tests for `:Z`, `:Y`, `:X` with `Val{:V}`.**

**Step 2: Run, verify all 3 fail.**

**Step 3: Implement `_decompose_phi_lazy`, `_decompose_phi_avg`, `_decompose_phi_symmetrize` with `Val{:V}` dispatch.**

**Step 4: Run, verify all pass.**

**Step 5: Commit.**

---

## Phase 5: `make_H_op` mode `:a` (h on central bond only)

### Task 13: Test for `make_H_op` mode `:a` — horizontal

**Files:** Extend test file, modify `optimize_fixedpoint.jl`.

**Step 1: Add failing test**

```julia
@testset "make_H_op :a horizontal — sanity" begin
    D, d, χ = 2, 2, 8
    A_raw = randn(D, D, D, D, d, 1)
    A_raw /= norm(A_raw)
    params = TeneT.make_default_params(D=D, χ=χ)
    A = build_A(A_raw, params)
    rt = init_VUMPSRuntime(A, χ, params.boundary_alg)
    rt, _ = leading_boundary(rt, A, params.boundary_alg)

    φ = TeneT.build_phi(A[1,1], A[1,1], Val(:H))
    H_op = TeneT.make_H_op(rt, A, Val(:H), params; mode=:a)
    N_op = TeneT.make_N_op(rt, A, Val(:H), params)

    Hφ = H_op(φ)
    Nφ = N_op(φ)
    # Sanity: <φ|H|φ> / <φ|N|φ> should approximately equal a single bond's expectation value
    val_H = real(sum(conj(φ) .* Hφ))
    val_N = real(sum(conj(φ) .* Nφ))
    e_bond = val_H / val_N
    @test isfinite(e_bond)
    # Heisenberg single-bond ⟨h⟩ for a random state is small (bounded ~|J|)
    @test abs(e_bond) < 10.0
end
```

**Step 2: Run, verify failure (`make_H_op` undefined).**

**Step 3: Implement**

```julia
"""
    make_H_op(rt, A, ::Val{:H}, params; mode::Symbol) -> Function

Return `H_op(φ) -> Hφ` applying the effective Hamiltonian on a 2-site
horizontal bond. Mode `:a` includes only the bond term acting on the central
bond between the two sites of φ. Mode `:b` (v2) sums all 7 bonds.
"""
function make_H_op(rt::VUMPSRuntime, A, dir::Val{:H}, params; mode::Symbol=:a)
    if mode == :a
        return _make_H_op_central(rt, A, dir, params)
    elseif mode == :b
        error("make_H_op mode :b not yet implemented (v2)")
    else
        error("Unknown H_eff mode: $mode")
    end
end

function _make_H_op_central(rt::VUMPSRuntime, A, ::Val{:H}, params)
    # Get N_op (same env contraction structure)
    N_op = make_N_op(rt, A, Val(:H), params)
    # Get the bond Hamiltonian
    h, _ = hamiltonian(params.model)   # h: 4-leg (p1, p2, p1', p2')
    h = _arraytype(A[1,1])(h)
    function H_op(φ)
        # Apply h to the two physical legs of φ:
        # φ has legs (l, dl, ul, pl, dr, ur, r, pr) with physical legs at positions 4 and 8
        @tensor φh[l, dl, ul, pl_new, dr, ur, r, pr_new] :=
            φ[l, dl, ul, pl, dr, ur, r, pr] * h[pl_new, pr_new, pl, pr]
        # Then contract through env exactly like N_op
        return N_op(φh)
    end
    return H_op
end
```

**Step 4: Run, verify pass.**

**Step 5: Commit.**

---

### Task 14: `make_H_op` mode `:a` — vertical

**Step 1: Add failing test for `Val{:V}`.**

**Step 2: Run, verify failure.**

**Step 3: Implement** by adding `_make_H_op_central(rt, A, ::Val{:V}, params)`.

**Step 4: Run, verify pass.**

**Step 5: Commit.**

---

## Phase 6: `sweep_bond!`

### Task 15: Smoke test for `sweep_bond!` — one direction

**Files:** Extend test file, modify `optimize_fixedpoint.jl`.

**Step 1: Add failing test**

```julia
@testset "sweep_bond! horizontal smoke" begin
    D, d, χ = 2, 2, 8
    A_raw = randn(D, D, D, D, d, 1) / sqrt(D^4*d)
    params = TeneT.make_default_params(D=D, χ=χ)
    A = build_A(A_raw, params)
    rt = init_VUMPSRuntime(A, χ, params.boundary_alg)
    rt, _ = leading_boundary(rt, A, params.boundary_alg)

    cfg = iPEPSFixedPointConfig(decompose_method=:X)
    A_before = deepcopy(A)
    λ, A_new, trunc_err = TeneT.sweep_bond(rt, A, Val(:H), params, cfg)
    @test isfinite(λ)
    @test size(A_new) == size(A_before)
    @test trunc_err >= 0
    @test !(A_new ≈ A_before)  # something changed
end
```

**Step 2: Run, verify failure.**

**Step 3: Implement**

```julia
"""
    sweep_bond(rt, A, dir, params, cfg) -> (λ, A_new, trunc_err)

One bond sweep step: build φ, geneigsolve, decompose back to A.
Does NOT update env or apply MCF (those are outer-loop concerns).
"""
function sweep_bond(rt, A, dir, params, cfg::iPEPSFixedPointConfig)
    A_central = A[1, 1]   # 1×1 unit cell assumption
    φ_old = build_phi(A_central, A_central, dir)
    H_op  = make_H_op(rt, A, dir, params; mode=cfg.H_eff_mode)
    N_op  = make_N_op(rt, A, dir, params)

    λs, φs, info = geneigsolve(
        x -> (H_op(x), N_op(x)), φ_old, 1, :SR;
        krylovdim = cfg.geneig_krylovdim,
        tol       = cfg.geneig_tol,
        maxiter   = cfg.geneig_maxiter,
        ishermitian = true,
        isposdef    = true,
    )
    if info.converged < 1
        @warn "geneigsolve did not converge" info=info dir=dir
    end
    λ_new, φ_new = real(λs[1]), φs[1]

    D = size(A_central, 1)
    A_new_central, trunc_err = decompose_phi(φ_new, dir;
                                             method = cfg.decompose_method,
                                             D_max  = D)
    A_new = deepcopy(A)
    A_new[1, 1] = A_new_central
    return λ_new, A_new, trunc_err
end
```

**Step 4: Run, verify pass.**

**Step 5: Commit.**

---

## Phase 7: Top-level driver

### Task 16: Smoke test for `optimize_ipeps_fixedpoint` (1 outer step)

**Files:** Extend test file, modify `optimize_fixedpoint.jl`.

**Step 1: Add failing test**

```julia
@testset "optimize_ipeps_fixedpoint 1-step smoke" begin
    D, d, χ = 2, 2, 8
    A_raw = randn(D, D, D, D, d, 1) / sqrt(D^4*d)
    params = TeneT.make_default_params(D=D, χ=χ)
    cfg = iPEPSFixedPointConfig(outer_maxiter=1, log_every=1)

    history = TeneT.optimize_ipeps_fixedpoint(A_raw, χ, params.model, params, cfg)
    @test length(history) == 1
    @test haskey(history[1], :λ)
    @test haskey(history[1], :E)
    @test haskey(history[1], :dλ)
    @test isfinite(history[1].λ)
end
```

**Step 2: Run, verify failure.**

**Step 3: Implement**

```julia
"""
    optimize_ipeps_fixedpoint(A_raw, χ, model, params, cfg) -> history

Top-level driver. Returns per-step history (Vector of NamedTuples with metrics).
"""
function optimize_ipeps_fixedpoint(A_raw, χ::Int, model, params, cfg::iPEPSFixedPointConfig)
    A = build_A(A_raw, params)
    rt = init_VUMPSRuntime(A, χ, params.boundary_alg)
    rt, _ = leading_boundary(rt, A, params.boundary_alg)

    history = []
    λ_prev = NaN
    A_prev = deepcopy(A)

    for k in 1:cfg.outer_maxiter
        t0 = time()
        # Env update by mode
        t_env0 = time()
        if cfg.env_mode == :A
            rt, _ = leading_boundary(rt, A, params.boundary_alg)
        elseif cfg.env_mode == :B
            warm_alg = deepcopy(params.boundary_alg)
            warm_alg.maxiter = cfg.env_warm_steps
            rt, _ = leading_boundary(rt, A, warm_alg)
        elseif cfg.env_mode == :C
            warm_alg = deepcopy(params.boundary_alg)
            warm_alg.maxiter = 1
            rt, _ = leading_boundary(rt, A, warm_alg)
        end
        t_env = time() - t_env0

        # Sweep H + V
        t_eig0 = time()
        trunc_errs = Float64[]
        λs = Float64[]
        for dir in (Val(:H), Val(:V))
            λ_dir, A, te = sweep_bond(rt, A, dir, params, cfg)
            push!(λs, λ_dir)
            push!(trunc_errs, te)
        end
        t_eig = time() - t_eig0

        # MCF
        t_mcf0 = time()
        A_central_new = local_min_norm(A[1,1], params; ifignore_gauge=cfg.mcf_ifignore_gauge)
        A[1, 1] = A_central_new
        t_mcf = time() - t_mcf0

        # Energy diagnostic (full E from current env, useful for cross-check)
        E = real(energy_value(model, A, ObsEnv(rt, A, params.boundary_alg), params)[1])
        λ_now = mean(λs)
        dλ = isnan(λ_prev) ? Inf : abs(λ_now - λ_prev)
        dA = norm(A[1,1] .- A_prev[1,1])

        rec = (
            iter=k, λ=λ_now, E=E, dλ=dλ, dA=dA,
            trunc_err = maximum(trunc_errs),
            t_total = time() - t0, t_env=t_env, t_eig=t_eig, t_mcf=t_mcf,
        )
        push!(history, rec)

        if k % cfg.log_every == 0
            @info "outer step $k" λ=λ_now E=E dλ=dλ dA=dA trunc_err=rec.trunc_err
        end

        if dλ < cfg.outer_tol_λ && dA < cfg.outer_tol_A
            @info "converged at iter $k"
            break
        end
        λ_prev = λ_now
        A_prev = deepcopy(A)
    end
    return history
end
```

**Step 4: Run, verify pass.**

**Step 5: Commit.**

---

## Phase 8: Example + experiments

### Task 17: Write Heisenberg example script

**Files:** Create `examples/Heisenberg/Heisenberg_Square_FixedPoint.jl`.

**Step 1: No test (this is an entry-point script).**

**Step 2: Write script**

```julia
# Heisenberg PoC: fixed-point eigenvalue iteration with MCF.
# Usage: julia --project=. examples/Heisenberg/Heisenberg_Square_FixedPoint.jl <experiment>
# experiment ∈ {E1, E2, E3, E4, E5}

using TeneT
using JLD2
using LinearAlgebra
using Random
Random.seed!(42)

const D = 2
const d = 2
const χ = 8
const model = Heisenberg{Square}(1.0)  # with sublattice rotation
const params = ...  # build per existing pattern

# Initial state: load LBFGS-optimized A if available, else random init
A_init = if isfile("data/lbfgs_D$(D)_chi$(χ).jld2")
    load("data/lbfgs_D$(D)_chi$(χ).jld2", "A")
else
    @warn "No LBFGS baseline found; using random init"
    randn(D, D, D, D, d, 1) / sqrt(D^4 * d)
end

experiments = Dict(
    "E1" => iPEPSFixedPointConfig(env_mode=:A, H_eff_mode=:a, decompose_method=:X, mcf_ifignore_gauge=false),
    "E2" => iPEPSFixedPointConfig(env_mode=:A, H_eff_mode=:a, decompose_method=:X, mcf_ifignore_gauge=true),
    "E3" => iPEPSFixedPointConfig(env_mode=:C, H_eff_mode=:a, decompose_method=:X, mcf_ifignore_gauge=false),
    "E4" => iPEPSFixedPointConfig(env_mode=:A, H_eff_mode=:a, decompose_method=:Z, mcf_ifignore_gauge=false),
    "E5" => iPEPSFixedPointConfig(env_mode=:A, H_eff_mode=:a, decompose_method=:Y, mcf_ifignore_gauge=false),
)

exp_name = get(ARGS, 1, "E1")
cfg = experiments[exp_name]

println("Running experiment $exp_name with config:")
println(cfg)

history = optimize_ipeps_fixedpoint(A_init, χ, model, params, cfg)

# Save log
mkpath("data/fixedpoint_logs")
save("data/fixedpoint_logs/$(exp_name)_D$(D)_chi$(χ).jld2", "history", history, "cfg", cfg)
println("Saved to data/fixedpoint_logs/$(exp_name)_D$(D)_chi$(χ).jld2")
println("Final: E = $(history[end].E), λ = $(history[end].λ), dλ = $(history[end].dλ)")
```

**Step 3: Verify it runs without error (1 outer step):**

```bash
julia --project=. -e 'using TeneT; cfg=iPEPSFixedPointConfig(outer_maxiter=1); ...'  # quick smoke
```

**Step 4: Commit**

```bash
git add examples/Heisenberg/Heisenberg_Square_FixedPoint.jl
git commit -m "feat(examples): Heisenberg fixed-point PoC entry point"
```

---

### Task 18: Run E1 (baseline experiment)

**Files:** generates `data/fixedpoint_logs/E1_D2_chi8.jld2`.

**Step 1: Run**

```bash
julia --project=. examples/Heisenberg/Heisenberg_Square_FixedPoint.jl E1 2>&1 | tee data/fixedpoint_logs/E1_run.txt
```

**Step 2: Inspect log**

Read tail of `E1_run.txt`. Look for:
- Did `dλ` decrease over iterations?
- Did the run reach `dλ < 1e-6` or maxiter?
- Final E within `1e-3 / site` of LBFGS baseline?

**Step 3: Document outcome**

Append to `docs/plans/poc-results.md` (create if not exists):

```markdown
## E1 (baseline: env=A, H=a, decomp=X, MCF=on)

- Final iter: ___
- Final E: ___
- LBFGS baseline E: ___
- Gap: ___ / site
- dλ at end: ___
- Status: CONVERGED / DIVERGED / OSCILLATING / SLOW
- Notes: ___
```

**Step 4: Commit**

```bash
git add docs/plans/poc-results.md data/fixedpoint_logs/E1_*.jld2 data/fixedpoint_logs/E1_run.txt
git commit -m "experiment(E1): baseline fixed-point + MCF — <one-line outcome>"
```

**If E1 fails (criterion #1: doesn't converge), STOP** and investigate before running E2-E5. Likely root causes: leg-ordering bug in `make_H_op` / `make_N_op`, sign issues with the Hamiltonian after sublattice rotation, or genuinely the algorithm doesn't work.

---

### Task 19: Run E2 (control: MCF off — the science question)

**Files:** generates `data/fixedpoint_logs/E2_D2_chi8.jld2`.

```bash
julia --project=. examples/Heisenberg/Heisenberg_Square_FixedPoint.jl E2 2>&1 | tee data/fixedpoint_logs/E2_run.txt
```

Document outcome in `docs/plans/poc-results.md`. Compare E1 vs E2:
- If E1 converges and E2 oscillates / fails → MCF is essential ✓
- If both converge similarly → MCF is not load-bearing here
- If E2 converges faster → MCF is harmful (unexpected, worth investigating)

Commit with one-line outcome.

---

### Task 20: Run E3, E4, E5 (sensitivity sweep)

```bash
for exp in E3 E4 E5; do
    julia --project=. examples/Heisenberg/Heisenberg_Square_FixedPoint.jl $exp 2>&1 | tee data/fixedpoint_logs/${exp}_run.txt
done
```

Document outcomes in `docs/plans/poc-results.md`.

Commit per experiment.

---

### Task 21: Compile findings summary

**Files:** `docs/plans/poc-findings.md` (new)

Summarize across all 5 experiments:

```markdown
# PoC Findings — iPEPS Fixed-Point + MCF

## Was the algorithm correct?
- Convergence (criterion 1): ___
- Physical accuracy vs LBFGS (criterion 2): ___
- Stability (criterion 3): ___

## Was MCF essential? (criterion 4 — key science question)
- E1 vs E2 comparison: ___
- Conclusion: ___

## Sensitivity to design choices
- Env mode (A vs C, E1 vs E3): ___
- Decompose method (X vs Y vs Z, E1 vs E4 vs E5): ___

## Recommendations
- Further work on this approach? Yes/No, why
- v2 priorities: ___
```

Commit:

```bash
git add docs/plans/poc-findings.md
git commit -m "docs(plans): PoC findings — fixed-point iPEPS + MCF"
```

---

## Total task count: 21

## Estimated wall-clock

- Tasks 1-3 (setup): 30 min
- Tasks 4-5 (build_phi): 20 min
- Tasks 6-8 (make_N_op): 60 min — contraction details require care
- Tasks 9-12 (decompose_phi): 90 min
- Tasks 13-14 (make_H_op): 40 min
- Task 15 (sweep_bond!): 30 min
- Task 16 (top-level driver): 60 min
- Tasks 17-18 (example + E1): 30 min, plus runtime variable
- Tasks 19-21 (E2-E5 + findings): 60 min, plus runtime

**Total active: ~7 hours; plus experiment compute time (~hours each, depending on D, χ, outer_maxiter).**

## When NOT to continue

- After Task 18 if E1 (baseline) fails to converge: STOP, revisit design — likely a bug or conceptual gap.
- After Task 19 if MCF on/off comparison is inconclusive: still useful data, but reconsider whether to continue with E3-E5.
- After Task 21 (findings): if conclusion is "approach doesn't work", document and close out — do NOT proceed to v2.
