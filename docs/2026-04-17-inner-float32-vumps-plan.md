# Inner Float32 in FLmap/FRmap/ACmap — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a `VUMPS.inner_etype::Union{Nothing,Type}=nothing` option that, when set
(e.g. to `Float32`), causes `FLmap`, `FRmap`, and `ACmap` to perform their internal
`@tensor` contractions in the reduced precision (`Float32` or `ComplexF32`, derived from
input eltype) while everything else in VUMPS stays in the original precision. Then run a
CPU-side L1 (single-call) and L4 (full iPEPS optimization) benchmark on
`Heisenberg_Square_VUMPS_C4v.jl` to decide whether the precision drop preserves final
energy to within `gradtol=1e-7`.

**Architecture:** Precision boundary is inside each `FLmap/FRmap/ACmap` method body
(not at the `_parallel` entry and not at VUMPS step level). The `_parallel` wrappers
take a new `inner_etype` kwarg and forward it to the map via a closure; `forloop` and
`parallel` wrappers plus their rrules stay unchanged. VUMPS struct gains a single field
threaded through `leftenv_c4v` / `ACenv_c4v`. See
`docs/2026-04-17-inner-float32-vumps-design.md` for the full design.

**Tech Stack:** Julia 1.x, TensorOperations (`@tensor`), Zygote (automatic rrule for
`convert.`), KrylovKit (`eigsolve`), OptimKit (`LBFGS`), `@kwdef` structs, existing
TeneT.jl test harness (`test/runtests.jl` with `ATYPES` looping).

**Base branch:** `origin/iPEPS-unified` (the worktree has already been reset to it).

**Success gates:** See §5.2 in the design doc. The plan's implementation tasks are
complete when Task 16 (L4 D=2 run) records results; then a human gate decides whether
Task 17 (L4 D=3) proceeds.

---

## Ground rules for the implementing agent

- **TDD:** For every source change, first add a failing test in `test/test_contraction.jl`
  (or `test/test_boundary.jl` where noted), confirm it fails with the expected message,
  then implement, then confirm pass, then commit. Do not skip the failing-run step — it
  is the only way to know the test actually exercises the new code.
- **Keep `inner_etype=nothing` the default everywhere.** Existing callers must be
  byte-for-byte unaffected. The first test in each map-family task verifies this via a
  regression check (`result_default == result_with_explicit_nothing`).
- **Complex-vs-real:** `inner_etype=Float32` must cast a `ComplexF64` input to `ComplexF32`,
  not error. Implement once in a small helper `_downcast_eltype(T, A)` so every map body
  uses the same logic.
- **One test, one commit.** Commit after each task. Commit messages follow the repo's
  existing style (`feat:`, `fix:`, `test:`, `docs:` prefix; see `git log --oneline -10`).
- **Do not touch** `Cmap`, `Lmap`, `Rmap`, `Mmap`, `Mumap`, `Mdmap`, `simple_eig`,
  `eigsolve`, `qrpos`, `qr_for_ad`, `norm`, `tol` checks, `init_ipeps`, `optimise_ipeps`,
  or any observable path. The design explicitly scopes them out.
- **Do not hand-write rrules.** Zygote auto-differentiates through `convert.(T, x)`.
- **Before every command**, confirm working directory is the worktree root
  (`D:\1 - research\1.26 - iPEPS_opt\TeneT.jl\.claude\worktrees\thirsty-heisenberg-6de4de`).

---

## Task 0: Baseline — verify test suite passes on the reset worktree

**Files:** none (pre-flight only)

**Step 1: Confirm clean status**

Run: `git status --short && git log --oneline -1`
Expected: clean working tree; HEAD at `1a79b5c docs: design for inner Float32 ...`.

**Step 2: Run the existing contraction tests to establish a green baseline**

Run (Windows bash):
```bash
julia --project=. -e 'using Pkg; Pkg.test("TeneT"; test_args=["contraction"])'
```
If the `test_args` filter is not wired, run the full suite:
```bash
julia --project=. test/runtests.jl
```
Expected: all tests pass. If CUDA is not available, `ATYPES = [Array]` and CUDA-only
tests are skipped — this is fine for the CPU-first phase.

**Step 3: Note baseline timing**

Record the contraction testset wall time (grep `Test Summary` output). This is the
regression baseline — if later tasks slow the suite by >2x we have an issue.

**Step 4: No commit.** This task is pre-flight verification only.

---

## Task 1: Add `_downcast_eltype` helper with a unit test

**Files:**
- Create: (new utility section inside) `src/contraction/basic.jl`
- Test: `test/test_contraction.jl`

**Step 1: Add failing test**

Add at the top of `test/test_contraction.jl` (inside the outer `@testset "Contraction"`):

```julia
@testset "_downcast_eltype helper" begin
    using TeneT: _downcast_eltype

    # Real: Float64 → Float32
    A64 = randn(Float64, 3, 4)
    A32 = _downcast_eltype(Float32, A64)
    @test eltype(A32) == Float32
    @test size(A32) == size(A64)
    @test maximum(abs, Float64.(A32) .- A64) < 1e-5

    # Complex: ComplexF64 → ComplexF32
    Z64 = randn(ComplexF64, 3, 4)
    Z32 = _downcast_eltype(Float32, Z64)
    @test eltype(Z32) == ComplexF32
    @test size(Z32) == size(Z64)
    @test maximum(abs, ComplexF64.(Z32) .- Z64) < 1e-5

    # Identity: target equals current eltype — return input unchanged (no copy)
    B64 = randn(Float64, 2, 2)
    @test _downcast_eltype(Float64, B64) === B64

    # Nothing: pass-through
    C64 = randn(ComplexF64, 2, 2)
    @test _downcast_eltype(nothing, C64) === C64
end
```

**Step 2: Add to `TeneT` import list** in `test/runtests.jl`:

Add `_downcast_eltype` to the `using TeneT:` list (after `CTCtoT` on line ~20).

**Step 3: Run the failing test**

Run: `julia --project=. test/runtests.jl 2>&1 | grep -A2 "_downcast"`
Expected: LoadError or UndefVarError on `_downcast_eltype`.

**Step 4: Implement `_downcast_eltype` in `src/contraction/basic.jl`**

Add at the very top of `src/contraction/basic.jl` (before the first `ALCtoAC_map`):

```julia
"""
    _downcast_eltype(T, A) -> A'

Cast `A`'s element type to the smallest type compatible with real scalar type `T`
while preserving complex-ness. Used to lower precision inside FLmap/FRmap/ACmap
when `VUMPS.inner_etype` is set.

Rules:
- `T === nothing`               → return `A` unchanged (identity)
- `T === real(eltype(A))`       → return `A` unchanged (identity; no copy)
- `eltype(A) <: Complex`        → cast to `Complex{T}`
- otherwise                     → cast to `T`
"""
_downcast_eltype(::Nothing, A) = A
function _downcast_eltype(T::Type, A)
    Ta = eltype(A)
    if Ta <: Complex
        T === real(Ta) && return A
        return Complex{T}.(A)
    else
        T === Ta && return A
        return T.(A)
    end
end
```

**Step 5: Re-run test — expect pass**

Run: `julia --project=. test/runtests.jl 2>&1 | grep "_downcast\|Test Summary" | head`
Expected: `_downcast_eltype helper: Pass`.

**Step 6: Commit**

```bash
git add src/contraction/basic.jl test/test_contraction.jl test/runtests.jl
git commit -m "feat: add _downcast_eltype helper for internal precision reduction

Introduces _downcast_eltype(T, A) that casts A's eltype to T-compatible precision
while preserving complex-vs-real. Used by upcoming FLmap/FRmap/ACmap inner_etype
kwarg. Identity cases (nothing, same eltype) return input without allocation."
```

---

## Task 2: Add `inner_etype` kwarg to `FLmap(::leg5)` + `FLmap(M::leg5, M2::leg5)` + Tuple dispatch

**Files:**
- Modify: `src/contraction/basic.jl:23-33` (leg5 methods and Tuple dispatch)
- Test: `test/test_contraction.jl`

**Step 1: Add failing test for `FLmap(FL, ALu, ALd, M::leg5, M2::leg5)` with inner_etype**

Add a new `@testset` inside the existing `@testset "basic.jl — atype=$atype"` block, after the existing "FLmap leg5 (bilayer)" block:

```julia
@testset "FLmap leg5 — inner_etype=Float32 (bilayer)" begin
    T = ComplexF64
    FL  = atype(randn(T, χ, D, D, χ))
    ALu = atype(randn(T, χ, D, D, χ))
    ALd = atype(randn(T, χ, D, D, χ))
    M   = atype(randn(T, D, D, D, D, D))

    # Default path (nothing) must byte-match current behavior
    r_default = FLmap(FL, ALu, ALd, M)
    r_explicit_nothing = FLmap(FL, ALu, ALd, M; inner_etype=nothing)
    @test Array(r_default) == Array(r_explicit_nothing)

    # Float32 path: eltype of result must match FL (ComplexF64), but values differ
    # by at most ~1e-6 relative
    r_f32 = FLmap(FL, ALu, ALd, M; inner_etype=Float32)
    @test eltype(r_f32) == ComplexF64
    @test size(r_f32) == size(r_default)
    rel_err = maximum(abs, Array(r_f32) .- Array(r_default)) /
              maximum(abs, Array(r_default))
    @test rel_err < 1e-5    # generous; typical Float32 @tensor error ~1e-7..1e-6

    # Real-input sanity: Float64 in → Float32 inner → Float64 out
    FLr = atype(randn(Float64, χ, D, D, χ))
    ALur = atype(randn(Float64, χ, D, D, χ))
    ALdr = atype(randn(Float64, χ, D, D, χ))
    Mr  = atype(randn(Float64, D, D, D, D, D))
    r_r = FLmap(FLr, ALur, ALdr, Mr; inner_etype=Float32)
    @test eltype(r_r) == Float64
end
```

**Step 2: Run — expect failure**

Run: `julia --project=. test/runtests.jl 2>&1 | grep -A2 "inner_etype=Float32 (bilayer)"`
Expected: `MethodError: no method matching FLmap(...; inner_etype=...)`.

**Step 3: Implement — modify `src/contraction/basic.jl`**

Replace the existing `FLmap(FL, ALu, ALd, M1::leg5, M2::leg5)`, `FLmap(FL, ALu, ALd, M::leg5)`, and `FLmap(FL, ALu, ALd, M::Tuple{leg5,leg5})` definitions with:

```julia
function FLmap(FL, ALu, ALd, M1::leg5, M2::leg5; inner_etype=nothing)
    if inner_etype === nothing || inner_etype == real(eltype(FL))
        @tensor result[d,g,h,l] := FL[a,e,f,i] * ALd[i,j,k,l] * M1[e,j,g,b,p] * M2[f,k,h,c,p] * ALu[a,b,c,d]
        return result
    else
        T_out = eltype(FL)
        FL_t  = _downcast_eltype(inner_etype, FL)
        ALu_t = _downcast_eltype(inner_etype, ALu)
        ALd_t = _downcast_eltype(inner_etype, ALd)
        M1_t  = _downcast_eltype(inner_etype, M1)
        M2_t  = _downcast_eltype(inner_etype, M2)
        @tensor result_t[d,g,h,l] := FL_t[a,e,f,i] * ALd_t[i,j,k,l] * M1_t[e,j,g,b,p] * M2_t[f,k,h,c,p] * ALu_t[a,b,c,d]
        return T_out.(result_t)
    end
end

FLmap(FL, ALu, ALd, M::leg5; inner_etype=nothing) =
    FLmap(FL, ALu, ALd, M, conj(M); inner_etype)

FLmap(FL, ALu, ALd, M::Tuple{leg5,leg5}; inner_etype=nothing) =
    FLmap(FL, ALu, ALd, M[1], M[2]; inner_etype)
```

**Step 4: Run — expect pass**

Run: `julia --project=. test/runtests.jl 2>&1 | grep "FLmap leg5 — inner_etype"`
Expected: `Pass`.

**Step 5: Commit**

```bash
git add src/contraction/basic.jl test/test_contraction.jl
git commit -m "feat: add inner_etype kwarg to FLmap leg5 methods

When inner_etype is set (e.g. Float32), the @tensor contraction inside FLmap runs
in downcasted precision while inputs and result are preserved in their original
eltype. Default (nothing) path is byte-identical to previous behavior.
Covers leg5 bilayer M, M::leg5 (M,conjM) convenience, and Tuple{leg5,leg5}."
```

---

## Task 3: Extend `inner_etype` to `FLmap(::leg4)` and `FLmap(::leg8)`

**Files:**
- Modify: `src/contraction/basic.jl:19-21` (leg4) and `:27-30` (leg8)
- Test: `test/test_contraction.jl`

**Step 1: Add failing tests**

Add after the leg5 inner_etype testset:

```julia
@testset "FLmap leg4 — inner_etype=Float32" begin
    T = ComplexF64
    FL  = atype(randn(T, χ, D, χ))
    ALu = atype(randn(T, χ, D, χ))
    ALd = atype(randn(T, χ, D, χ))
    M   = atype(randn(T, D, D, D, D))
    r0 = FLmap(FL, ALu, ALd, M)
    r32 = FLmap(FL, ALu, ALd, M; inner_etype=Float32)
    @test eltype(r32) == ComplexF64
    @test size(r32) == size(r0)
    @test maximum(abs, Array(r32) .- Array(r0)) / maximum(abs, Array(r0)) < 1e-5
end

@testset "FLmap leg8 — inner_etype=Float32" begin
    T = ComplexF64
    FL  = atype(randn(T, χ, D, D, χ))
    ALu = atype(randn(T, χ, D, D, χ))
    ALd = atype(randn(T, χ, D, D, χ))
    M   = atype(randn(T, D, D, D, D, D, D, D, D))
    r0 = FLmap(FL, ALu, ALd, M)
    r32 = FLmap(FL, ALu, ALd, M; inner_etype=Float32)
    @test eltype(r32) == ComplexF64
    @test maximum(abs, Array(r32) .- Array(r0)) / maximum(abs, Array(r0)) < 1e-5
end
```

**Step 2: Verify failures**

Run: `julia --project=. test/runtests.jl 2>&1 | grep -E "leg4 — inner|leg8 — inner"`
Expected: MethodError for both.

**Step 3: Implement**

Replace the leg4 and leg8 `FLmap` methods:

```julia
function FLmap(FL, ALu, ALd, M::leg4; inner_etype=nothing)
    if inner_etype === nothing || inner_etype == real(eltype(FL))
        @tensor result[c,e,h] := FL[a,d,f] * ALd[f,g,h] * M[d,g,e,b] * ALu[a,b,c]
        return result
    else
        T_out = eltype(FL)
        FL_t  = _downcast_eltype(inner_etype, FL)
        ALu_t = _downcast_eltype(inner_etype, ALu)
        ALd_t = _downcast_eltype(inner_etype, ALd)
        M_t   = _downcast_eltype(inner_etype, M)
        @tensor result_t[c,e,h] := FL_t[a,d,f] * ALd_t[f,g,h] * M_t[d,g,e,b] * ALu_t[a,b,c]
        return T_out.(result_t)
    end
end

function FLmap(FL, ALu, ALd, M::leg8; inner_etype=nothing)
    if inner_etype === nothing || inner_etype == real(eltype(FL))
        @tensor result[d,g,h,l] := FL[a,e,f,i] * ALd[i,j,k,l] * M[e,f,j,k,g,h,b,c] * ALu[a,b,c,d]
        return result
    else
        T_out = eltype(FL)
        FL_t  = _downcast_eltype(inner_etype, FL)
        ALu_t = _downcast_eltype(inner_etype, ALu)
        ALd_t = _downcast_eltype(inner_etype, ALd)
        M_t   = _downcast_eltype(inner_etype, M)
        @tensor result_t[d,g,h,l] := FL_t[a,e,f,i] * ALd_t[i,j,k,l] * M_t[e,f,j,k,g,h,b,c] * ALu_t[a,b,c,d]
        return T_out.(result_t)
    end
end
```

**Step 4: Verify pass**

Run: `julia --project=. test/runtests.jl 2>&1 | grep -E "FLmap leg4 — inner|FLmap leg8 — inner"`
Expected: Pass both.

**Step 5: Commit**

```bash
git add src/contraction/basic.jl test/test_contraction.jl
git commit -m "feat: extend inner_etype kwarg to FLmap leg4 and leg8"
```

---

## Task 4: Add `inner_etype` kwarg to all `FRmap` methods

**Files:**
- Modify: `src/contraction/basic.jl` (FRmap leg4, leg5, leg8, leg5 convenience, Tuple dispatch)
- Test: `test/test_contraction.jl`

**Step 1: Failing tests**

Add after the FLmap inner_etype testsets:

```julia
@testset "FRmap leg4 — inner_etype=Float32" begin
    T = ComplexF64
    FR  = atype(randn(T, χ, D, χ))
    ARu = atype(randn(T, χ, D, χ))
    ARd = atype(randn(T, χ, D, χ))
    M   = atype(randn(T, D, D, D, D))
    r0 = FRmap(FR, ARu, ARd, M)
    r32 = FRmap(FR, ARu, ARd, M; inner_etype=Float32)
    @test eltype(r32) == ComplexF64
    @test maximum(abs, Array(r32) .- Array(r0)) / maximum(abs, Array(r0)) < 1e-5
end

@testset "FRmap leg5 — inner_etype=Float32" begin
    T = ComplexF64
    FR  = atype(randn(T, χ, D, D, χ))
    ARu = atype(randn(T, χ, D, D, χ))
    ARd = atype(randn(T, χ, D, D, χ))
    M   = atype(randn(T, D, D, D, D, D))
    r0 = FRmap(FR, ARu, ARd, M)
    r32 = FRmap(FR, ARu, ARd, M; inner_etype=Float32)
    @test maximum(abs, Array(r32) .- Array(r0)) / maximum(abs, Array(r0)) < 1e-5
end

@testset "FRmap leg8 — inner_etype=Float32" begin
    T = ComplexF64
    FR  = atype(randn(T, χ, D, D, χ))
    ARu = atype(randn(T, χ, D, D, χ))
    ARd = atype(randn(T, χ, D, D, χ))
    M   = atype(randn(T, D, D, D, D, D, D, D, D))
    r0 = FRmap(FR, ARu, ARd, M)
    r32 = FRmap(FR, ARu, ARd, M; inner_etype=Float32)
    @test maximum(abs, Array(r32) .- Array(r0)) / maximum(abs, Array(r0)) < 1e-5
end
```

**Step 2–3: Verify failure → implement** in `src/contraction/basic.jl` — the 5 FRmap definitions (three methods + two forwarders). Pattern identical to FLmap task 2+3 (copy the structure, swap FL↔FR / AL↔AR, match `@tensor` indices from the original method bodies exactly — do not "clean up" the index letters).

**Step 4–5: Verify pass + commit**

```bash
git commit -m "feat: add inner_etype kwarg to all FRmap methods"
```

---

## Task 5: Add `inner_etype` kwarg to `ACmap` and `ACdmap` methods

**Files:**
- Modify: `src/contraction/basic.jl` (ACmap leg3/leg4/leg5/leg8 and Tuple dispatch; same for ACdmap)
- Test: `test/test_contraction.jl`

**Step 1: Failing tests**

Before writing these, first `Grep` the `ACmap` and `ACdmap` signatures in
`src/contraction/basic.jl` and confirm which leg-types exist. Write one inner_etype
regression test per leg-type you find, following the FRmap pattern exactly. Both
ACmap and ACdmap must be covered.

**Step 2: Implement** with the same `if inner_etype === nothing ... else _downcast ... T_out.(result_t) end` pattern applied to every ACmap and ACdmap method.

**Step 3: Verify pass + commit**

```bash
git commit -m "feat: add inner_etype kwarg to ACmap and ACdmap methods"
```

---

## Task 6: Thread `inner_etype` through `FLmap_parallel` / `FRmap_parallel` / `ACmap_parallel` / `ACdmap_parallel`

**Files:**
- Modify: `src/contraction/forloop_parallel_MPI.jl:281-365` (the four `*_parallel` functions)
- Test: `test/test_contraction.jl` (inside the existing `@testset "parallel forloop — atype=$atype"` block)

**Step 1: Failing tests**

Add four test sets mirroring the existing `forloop_iter=1 matches FLmap` pattern, but
with `inner_etype=Float32`:

```julia
@testset "FLmap_parallel inner_etype=Float32 forloop_iter=1 matches FLmap(inner_etype)" begin
    T = ComplexF64
    FL  = atype(randn(T, χ, D, χ))
    ALu = atype(randn(T, χ, D, χ))
    ALd = atype(randn(T, χ, D, χ))
    M   = atype(randn(T, D, D, D, D))
    direct  = FLmap(FL, ALu, ALd, M; inner_etype=Float32)
    par_res = FLmap_parallel(FL, ALu, ALd, M; ifparallel=false, forloop_iter=1, inner_etype=Float32)
    @test Array(direct) ≈ Array(par_res)
end

@testset "FLmap_parallel inner_etype=Float32 forloop_iter=2 matches FLmap(inner_etype) within tol" begin
    T = ComplexF64
    FL  = atype(randn(T, χ, D, χ))
    ALu = atype(randn(T, χ, D, χ))
    ALd = atype(randn(T, χ, D, χ))
    M   = atype(randn(T, D, D, D, D))
    direct  = FLmap(FL, ALu, ALd, M; inner_etype=Float32)
    par_res = FLmap_parallel(FL, ALu, ALd, M; ifparallel=false, forloop_iter=2, inner_etype=Float32)
    # forloop=2 + Float32 accumulate slightly differently — use tolerance not ≈
    @test maximum(abs, Array(direct) .- Array(par_res)) /
          maximum(abs, Array(direct)) < 1e-5
end
```

Repeat for FRmap_parallel, ACmap_parallel, ACdmap_parallel — use the existing test
patterns in the same file as templates.

**Step 2: Verify failures**

Run: `julia --project=. test/runtests.jl 2>&1 | grep "inner_etype=Float32 forloop_iter"`
Expected: `MethodError` — `inner_etype` not a known kwarg.

**Step 3: Implement — edit each of the 4 `*_parallel` functions**

Pattern (for `FLmap_parallel`; apply analogously to the other three):

```julia
function FLmap_parallel(FL, ALu, ALd, M; ifparallel, forloop_iter, inner_etype=nothing)
    # N_in / N_out / size_out computation unchanged — copy from current source
    # ...
    f = inner_etype === nothing ? FLmap :
        (args...) -> FLmap(args...; inner_etype)
    if ifparallel
        return parallel(f, FL, ALu, ALd, M; forloop_iter, N_in, N_out, size_out)
    else
        return forloop(f, FL, ALu, ALd, M; forloop_iter, N_in, N_out, size_out)
    end
end
```

**Important:** Do not modify `forloop`, `parallel`, `forloop_sum`, `parallel_sum`, or
their rrules. They pass `f` as a value and call `f(split_args...)` — our closure handles
the kwarg internally.

**Step 4: Verify pass + commit**

```bash
git add src/contraction/forloop_parallel_MPI.jl test/test_contraction.jl
git commit -m "feat: thread inner_etype through FLmap/FRmap/ACmap/ACdmap _parallel wrappers"
```

---

## Task 7: Add `inner_etype` field to `VUMPS` struct

**Files:**
- Modify: `src/boundary_algorithm/interface.jl:10-28` (the `VUMPS{F}` `@kwdef mutable struct`)
- Test: `test/test_boundary.jl` (or `test/test_types.jl`, whichever already covers VUMPS struct construction — run `grep -l "VUMPS(" test/` to decide)

**Step 1: Failing test**

Add inside the existing boundary/types testset:

```julia
@testset "VUMPS struct — inner_etype field" begin
    # default
    alg = VUMPS{C4v}()
    @test alg.inner_etype === nothing

    # explicit nothing
    alg2 = VUMPS{C4v}(; inner_etype=nothing)
    @test alg2.inner_etype === nothing

    # Float32
    alg3 = VUMPS{C4v}(; inner_etype=Float32)
    @test alg3.inner_etype === Float32

    # Pattern matches General and Plaquette — struct is generic in F
    alg4 = VUMPS{General}(; inner_etype=Float32)
    @test alg4.inner_etype === Float32
end
```

**Step 2: Verify failure** — `type VUMPS has no field inner_etype`.

**Step 3: Implement**

In `src/boundary_algorithm/interface.jl`, add inside the `VUMPS{F}` `@kwdef mutable struct`, after `ifcheckpoint::Bool = false`:

```julia
    inner_etype::Union{Nothing, Type} = nothing
```

**Step 4: Verify pass + commit**

```bash
git add src/boundary_algorithm/interface.jl test/test_boundary.jl
git commit -m "feat: add inner_etype field to VUMPS struct (default nothing)"
```

---

## Task 8: Thread `inner_etype` through `leftenv_c4v` and `ACenv_c4v`

**Files:**
- Modify: `src/boundary_algorithm/vumps/c4v.jl:1-26` (the two env functions; do **not** touch `Cenv_c4v`)
- Test: `test/test_boundary.jl`

**Step 1: Failing integration test**

Add:

```julia
@testset "VUMPS{C4v} step with inner_etype=Float32 runs without error" begin
    atype = Array
    T = Float64
    D, χ = 2, 8
    pattern = [1;;]

    M = atype(randn(T, D, D, D, D, D))
    alg = VUMPS{C4v}(; forloop_iter=1, power_iter=1, maxiter=1, miniter=0,
                    tol=1e-10, verbosity=0, inner_etype=Float32)
    # Pack into StructArray to match init_env signature
    Ms = StructArray([reshape(M, D, D, D, D, D, 1)])
    rt = TeneT.init_env(Ms, χ, alg)

    # Run one vumps_step — should not throw; environments stay Float64
    rt_new, err = vumps_step(rt, M, alg)
    @test eltype(rt_new.AL) == T
    @test eltype(rt_new.C)  == T
    @test eltype(rt_new.FL) == T
    @test isfinite(err)
end
```

(If `init_env` / `vumps_step` / `StructArray` wrapping requires slightly different
signatures, adapt from the patterns already used in `test/test_boundary.jl`. The key
assertions are: no error, Float64 output eltypes, finite err.)

**Step 2: Verify failure** — probably `UndefRefError` or key mismatch from `@unpack`.

**Step 3: Implement**

In `src/boundary_algorithm/vumps/c4v.jl`:

```julia
function leftenv_c4v(ALu, ALd, M, FL; alg, kwargs...)
    @unpack power_iter, ifparallel, forloop_iter, ifcheckpoint, inner_etype = alg
    f(FL) = ifcheckpoint ?
        checkpoint(FLmap_parallel, FL, ALu, ALd, M; ifparallel, forloop_iter, inner_etype) :
        FLmap_parallel(FL, ALu, ALd, M; ifparallel, forloop_iter, inner_etype)
    # remainder unchanged
    if alg.ifsimple_eig
        λFLs, FLs = simple_eig(f, FL; power_iter)
    else
        λFLs, FLs, info = eigsolve(f, FL, 1, :LM; alg_rrule=GMRES(verbosity=-1), maxiter=100, shermitian=false, kwargs...)
        alg.verbosity >= 1 && info.converged == 0 && @warn "FLenv_c4v not converged"
    end
    return λFLs[1], FLs[1]
end

function ACenv_c4v(AC, FL, M; alg, kwargs...)
    @unpack power_iter, ifparallel, forloop_iter, ifcheckpoint, inner_etype = alg
    f(AC) = ifcheckpoint ?
        checkpoint(ACmap_parallel, AC, FL, FL, M; ifparallel, forloop_iter, inner_etype) :
        ACmap_parallel(AC, FL, FL, M; ifparallel, forloop_iter, inner_etype)
    # remainder unchanged
    if alg.ifsimple_eig
        λACs, ACs = simple_eig(f, AC; power_iter)
    else
        λACs, ACs, info = eigsolve(f, AC, 1, :LM; alg_rrule=GMRES(verbosity=-1), maxiter=100, shermitian=false, kwargs...)
        alg.verbosity >= 1 && info.converged == 0 && @warn "ACenv_c4v not converged"
    end
    return λACs[1], ACs[1]
end
```

`Cenv_c4v` is **not** modified.

**Step 4: Verify pass + commit**

```bash
git add src/boundary_algorithm/vumps/c4v.jl test/test_boundary.jl
git commit -m "feat: thread inner_etype from VUMPS alg through leftenv_c4v and ACenv_c4v"
```

---

## Task 9: End-to-end smoke — run the `Heisenberg_Square_VUMPS_C4v.jl` example with `inner_etype=Float32`, D=2, χ=8, 2 LBFGS steps

**Files:**
- Create: `examples/Heisenberg/Heisenberg_Square_VUMPS_C4v_f32_smoke.jl`

**Step 1: Write the smoke script**

Copy `examples/Heisenberg/Heisenberg_Square_VUMPS_C4v.jl` and modify:
- χ = 8 (not 16)
- `LBFGS(200; maxiter=2, ...)` (only 2 steps)
- `boundary_alg = VUMPS{C4v}(...; inner_etype=Float32, ...)` (add the field)
- `folder` renamed to avoid clobbering real-run data
- `seed = 42`, D = 2
- Use `atype = Array` (not CUDA)

**Step 2: Run it**

```bash
julia --project=. examples/Heisenberg/Heisenberg_Square_VUMPS_C4v_f32_smoke.jl
```
Expected: no errors, prints 2 LBFGS iterations, ends with a finite energy. Wall time
< 30s on CPU.

**Step 3: If any error is thrown**, stop and debug. Likely issues:
- Missing `inner_etype` kwarg on some `*_parallel` wrapper (Task 6 regression)
- `@unpack` doesn't know the field (Task 7 regression on non-C4v mode variant)
- Zygote cannot differentiate through `_downcast_eltype` — if this happens, add a
  `ChainRulesCore.rrule` for `_downcast_eltype` that does the reverse cast. Record
  as an addendum.

**Step 4: Commit**

```bash
git add examples/Heisenberg/Heisenberg_Square_VUMPS_C4v_f32_smoke.jl
git commit -m "test: end-to-end smoke for inner_etype=Float32 on Heisenberg C4v example

Runs 2 LBFGS steps at D=2 χ=8 with VUMPS inner contractions in Float32,
confirming the full optimization path (forward + Zygote backward) works."
```

---

## Task 10: Write `examples/benchmark_inner_Float32_L1.jl` — L1 single-call diagnostic

**Files:**
- Create: `examples/benchmark_inner_Float32_L1.jl`
- Create (path only): `docs/benchmarks/` directory for output

**Step 1: Write the script**

Structure (full file, ~80 lines):

```julia
# examples/benchmark_inner_Float32_L1.jl
#
# L1 diagnostic for the inner_etype=Float32 experiment.
# Compares single FLmap_parallel calls in three modes and reports relative errors.
# Design: docs/2026-04-17-inner-float32-vumps-design.md §4.1

using Random, Statistics, Printf, LinearAlgebra, Dates, TeneT

const RESULTS_PATH = joinpath(@__DIR__, "..", "docs", "benchmarks",
                              "CPU_inner_Float32.md")

function run_L1(D::Int, χ::Int, seed::Int; d::Int=D)
    Random.seed!(seed)
    FL  = randn(Float64, χ, D, D, χ)
    ALu = randn(Float64, χ, D, D, χ)
    ALd = randn(Float64, χ, D, D, χ)
    M   = randn(Float64, D, D, D, D, d)

    R0  = TeneT.FLmap_parallel(FL, ALu, ALd, M; ifparallel=false, forloop_iter=1,  inner_etype=nothing)
    R16 = TeneT.FLmap_parallel(FL, ALu, ALd, M; ifparallel=false, forloop_iter=16, inner_etype=nothing)
    R32 = TeneT.FLmap_parallel(FL, ALu, ALd, M; ifparallel=false, forloop_iter=1,  inner_etype=Float32)

    nrm = norm(R0)
    err_forloop = norm(R16 .- R0) / nrm
    err_float32 = norm(R32 .- R0) / nrm
    ratio = err_float32 / max(err_forloop, eps(Float64))
    return (D=D, χ=χ, seed=seed, err_forloop=err_forloop,
            err_float32=err_float32, ratio=ratio)
end

function main()
    configs = [(2, 16), (2, 32), (3, 16), (3, 32), (3, 64)]
    seeds = [42, 43, 44]

    rows = []
    for (D, χ) in configs, s in seeds
        push!(rows, run_L1(D, χ, s))
    end

    println("=" ^ 80)
    println("L1 results (single FLmap_parallel call)")
    println("=" ^ 80)
    @printf("%-3s %-4s %-5s %-14s %-14s %-10s\n",
            "D", "χ", "seed", "err_forloop", "err_float32", "ratio")
    for r in rows
        @printf("%-3d %-4d %-5d %-14.3e %-14.3e %-10.3f\n",
                r.D, r.χ, r.seed, r.err_forloop, r.err_float32, r.ratio)
    end

    # Append to Markdown report (create if missing, overwrite section if present)
    mkpath(dirname(RESULTS_PATH))
    open(RESULTS_PATH, "a") do io
        println(io, "\n## L1 — single FLmap_parallel call (", today(), ", CPU)\n")
        println(io, "| D | χ | seed | err_forloop | err_float32 | ratio |")
        println(io, "|---|---|------|-------------|-------------|-------|")
        for r in rows
            @printf(io, "| %d | %d | %d | %.3e | %.3e | %.3f |\n",
                    r.D, r.χ, r.seed, r.err_forloop, r.err_float32, r.ratio)
        end
    end
    @printf("\nReport appended to %s\n", RESULTS_PATH)

    # Gate decision
    max_ratio = maximum(r.ratio for r in rows)
    println("\nmax ratio = ", max_ratio)
    if max_ratio < 10
        println("GATE: PASS — proceed to L4.")
    elseif max_ratio < 100
        println("GATE: PASS WITH CAUTION — L4 may fail.")
    else
        println("GATE: STOP — reconsider precision strategy (see design §5.2).")
    end
end

main()
```

**Step 2: Run it**

```bash
julia --project=. examples/benchmark_inner_Float32_L1.jl
```
Expected: runtime < 1 minute; markdown report created/updated at
`docs/benchmarks/CPU_inner_Float32.md`; GATE line prints PASS / PASS WITH CAUTION / STOP.

**Step 3: Inspect the report**

Open `docs/benchmarks/CPU_inner_Float32.md`. Verify the table is well-formed and the
numbers pass a sanity check (err_float32 should be in the 1e-7 .. 1e-5 range;
err_forloop should be similar order of magnitude or smaller).

**Step 4: Commit both script and report**

```bash
git add examples/benchmark_inner_Float32_L1.jl docs/benchmarks/CPU_inner_Float32.md
git commit -m "feat(bench): add L1 single-FLmap inner_etype=Float32 diagnostic script + first results"
```

**Step 5 (Gate):** If GATE printed STOP, halt here and report back to the human — the plan
does not continue to L4 if L1 says the ratio is ≥100. Open a discussion task on whether
to explore Float32+Float64 accumulation (boundary (c) from the design).

---

## Task 11: Write `examples/benchmark_inner_Float32_L4.jl` — L4 full iPEPS comparison (D=2 stage)

**Files:**
- Create: `examples/benchmark_inner_Float32_L4.jl`

**Step 1: Write the script**

The script must:
- Take D, χ, seed, precision as parameters
- Use exactly the same model / optimizer settings as
  `examples/Heisenberg/Heisenberg_Square_VUMPS_C4v.jl` except:
  - `inner_etype` set per experiment arm
  - Unique `folder` per experiment arm so runs do not clobber each other
- For D=2 stage, sweep `configs = [(2, 16), (2, 32)]`, `seeds = [42, 43, 44]`,
  `precisions = [nothing, Float32]` — 12 runs total
- Record for each run: `E_final`, `n_lbfgs_steps`, `wall_clock`, `peak_rss`
  (via `Sys.maxrss()` before/after), and optionally `grad_norm_trajectory`
- Append a table to `docs/benchmarks/CPU_inner_Float32.md` under a new
  `## L4 D=2 ...` heading

Structure sketch (~150 lines; fill in full logic from the example file):

```julia
using TeneT, OptimKit, LinearAlgebra, Random, Zygote, Printf, Dates, Statistics

const RESULTS_PATH = joinpath(@__DIR__, "..", "docs", "benchmarks",
                              "CPU_inner_Float32.md")

function run_L4(D, χ, seed, inner_etype)
    Random.seed!(seed)
    atype = Array
    etype = Float64
    pattern = [1;;]
    model = Heisenberg(lattice=Square(), S=0.5, Jx=1.0, Jy=1.0, Jz=1.0,
                       ifrotate=true, couplingtype=:uniform, bondratio=1.0)
    tag = inner_etype === nothing ? "f64" : string(inner_etype)
    folder = joinpath(pkgdir(TeneT), "data/bench_L4/$tag/D$(D)_chi$(χ)_s$(seed)/")

    boundary_alg = VUMPS{C4v}(; ifsimple_eig=true, ifparallel=false, ifcheckpoint=true,
                              forloop_iter=1, maxiter=3, miniter=0,
                              maxiter_ad=4, miniter_ad=4, power_iter=1,
                              power_iter_ad=5, power_iter_obs=40,
                              show_every=10, tol=1e-10, verbosity=0,
                              inner_etype=inner_etype)
    params = GradientOptimize(; model=model, pattern=pattern,
                              boundary_alg=boundary_alg,
                              optimizer=LBFGS(200; maxiter=200, verbosity=0,
                                              gradtol=1e-7,
                                              linesearch=HagerZhangLineSearch(maxfg=5)),
                              ifcheckpoint=false, forloop_iter=1,
                              maxiter_restart=1, verbosity=0, folder=folder,
                              ifSU=false, SUτ=0, ifprecondition=true,
                              iter_precond=0, reuse_env=true,
                              ifsave_env=true, ifload_env=false,
                              ifsave_lbfgs=true, ifload_lbfgs=false)

    A = init_ipeps(; atype, etype, No=0, D=D, χ=χ, params)

    restriction_ipeps(A) = C4v_restriction(A)

    rss_before = Sys.maxrss()
    t_start = time()
    result = optimise_ipeps(A, χ, 0, params; restriction_ipeps)
    wall = time() - t_start
    rss_after = Sys.maxrss()

    # Extract final energy, step count from `result` — see optimise_ipeps return
    # shape and adapt. Below is a placeholder.
    E_final = result.fmin       # or result.E, or result[1] — confirm with println first
    n_steps = result.numiter    # likewise

    return (D=D, χ=χ, seed=seed,
            precision=(inner_etype === nothing ? "Float64" : string(inner_etype)),
            E=E_final, n_steps=n_steps, wall=wall,
            rss_bytes=(rss_after - rss_before))
end

function main()
    # LOCK BLAS threads to eliminate variance
    BLAS.set_num_threads(4)

    configs = [(2, 16), (2, 32)]
    seeds = [42, 43, 44]
    precisions = [nothing, Float32]

    rows = []
    for (D, χ) in configs, s in seeds, p in precisions
        @printf("Running D=%d χ=%d seed=%d precision=%s ...\n",
                D, χ, s, p === nothing ? "Float64" : string(p))
        r = run_L4(D, χ, s, p)
        push!(rows, r)
        @printf("  → E=%.12f  n_steps=%d  wall=%.1fs\n", r.E, r.n_steps, r.wall)
    end

    # Append table
    mkpath(dirname(RESULTS_PATH))
    open(RESULTS_PATH, "a") do io
        println(io, "\n## L4 D=2 — full iPEPS optimization (", today(), ", CPU)\n")
        println(io, "BLAS threads: ", BLAS.get_num_threads())
        println(io, "\n| D | χ | seed | precision | E_final | n_steps | wall (s) | ΔRSS (MB) |")
        println(io, "|---|---|------|-----------|---------|---------|----------|-----------|")
        for r in rows
            @printf(io, "| %d | %d | %d | %s | %.12f | %d | %.1f | %.0f |\n",
                    r.D, r.χ, r.seed, r.precision, r.E, r.n_steps, r.wall,
                    r.rss_bytes / 1e6)
        end
    end

    # Pass/fail summary
    println("\n=== D=2 energy comparison (median over seeds) ===")
    for (D, χ) in configs
        E_f64 = [r.E for r in rows if r.D==D && r.χ==χ && r.precision=="Float64"]
        E_f32 = [r.E for r in rows if r.D==D && r.χ==χ && r.precision=="Float32"]
        dE = abs(median(E_f64) - median(E_f32))
        pass = dE < 1e-7
        @printf("  (D=%d, χ=%d): |ΔE| = %.3e  →  %s\n", D, χ, dE, pass ? "PASS" : "FAIL")
    end
end

main()
```

**Step 2: Dry-run with 1 seed first to validate script plumbing**

Comment out `seeds = [42, 43, 44]` → `seeds = [42]` temporarily, run:
```bash
julia --project=. examples/benchmark_inner_Float32_L4.jl
```
Expected: 4 runs complete (2 configs × 1 seed × 2 precisions), no errors, table
appended. If `result.fmin` / `result.numiter` naming is wrong, `println(result)` to
inspect and fix.

Restore seeds to 3 values.

**Step 3: Full D=2 run**

```bash
julia --project=. examples/benchmark_inner_Float32_L4.jl | tee /tmp/L4_D2.log
```
Expected wall time: ~2–4 minutes total.

**Step 4: Inspect results**

Open `docs/benchmarks/CPU_inner_Float32.md`. Check the pass/fail summary.

**Step 5: Commit**

```bash
git add examples/benchmark_inner_Float32_L4.jl docs/benchmarks/CPU_inner_Float32.md
git commit -m "feat(bench): L4 full-iPEPS comparison D=2 + results

Compares final energy of Heisenberg C4v iPEPS optimization under all-Float64
vs inner_etype=Float32. D=2, chi ∈ {16,32}, 3 seeds each."
```

---

## Task 12: Human gate on L4 D=2 results

**This is a checkpoint, not an automated step.**

Review `docs/benchmarks/CPU_inner_Float32.md`. Criteria from design §5.2:

- `median_seeds |E_F32 − E_F64| < 1e-7` for both (D=2, χ=16) and (D=2, χ=32)
- Float32 `n_lbfgs_steps` ≤ 2× Float64 baseline
- Energy trajectories (if recorded) do not diverge

**If PASS:** proceed to Task 13 (D=3).

**If FAIL:** pause implementation; write a results-analysis note into
`docs/benchmarks/CPU_inner_Float32.md` (under a "Phase-1 D=2 conclusion" section)
describing which configs failed and by how much, and flagging for follow-up. Do **not**
run Task 13. Consult the human about next steps (options: run L2/L3 diagnostics,
switch to boundary (c) Float32×Float64 accumulation, declare Float32 unsuitable for
this model).

---

## Task 13 (conditional — only if Task 12 PASS): Extend L4 to D=3 stage

**Files:**
- Modify: `examples/benchmark_inner_Float32_L4.jl` (widen configs, or add a second
  invocation mode)

**Step 1: Modify the script**

Add a `configs_D3 = [(3, 16), (3, 32)]` list and run it as a second pass under a
`## L4 D=3 ...` heading in the report. Keep D=2 results intact. Optionally add a
CLI argument (`ARGS[1] == "D2" / "D3"`) so the runner can choose the stage.

**Step 2: Run**

```bash
julia --project=. examples/benchmark_inner_Float32_L4.jl D3 | tee /tmp/L4_D3.log
```
Expected wall time: ~2 hours total (12 runs × 10min).

**Step 3: Append results + human gate** — same pass/fail criterion as D=2.

**Step 4: Commit**

```bash
git add examples/benchmark_inner_Float32_L4.jl docs/benchmarks/CPU_inner_Float32.md
git commit -m "feat(bench): extend L4 to D=3 + results"
```

---

## Task 14: Write Phase-1 conclusion section in the report

**Files:**
- Modify: `docs/benchmarks/CPU_inner_Float32.md` (append final section)

**Step 1: Append conclusion**

Under a `## Phase-1 conclusion (CPU, 2026-04-17)` heading, write 3–6 sentences
summarizing:
- Did D=2 pass? D=3?
- What was the typical ΔE magnitude relative to gradtol?
- Any surprises (LBFGS step count difference? RSS difference? Unexpected divergence?)
- Recommendation for Phase-2: proceed to GPU (same boundary) / expand to boundary (ii) /
  abandon

**Step 2: Commit**

```bash
git add docs/benchmarks/CPU_inner_Float32.md
git commit -m "docs: Phase-1 CPU inner_etype=Float32 conclusion"
```

---

## Exit criteria

Plan is complete when:
- Tasks 0–11 all committed
- Task 12 gate decision recorded
- Task 13 either completed (if D=2 passed) or skipped (with reason documented)
- Task 14 conclusion committed

Any task that introduces a regression in existing tests (`julia test/runtests.jl`)
must be fixed before committing. No task may merge failing tests into the branch.
