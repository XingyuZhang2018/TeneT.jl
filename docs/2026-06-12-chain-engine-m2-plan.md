# Chain Engine M2 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (or superpowers:executing-plans) to implement this plan task-by-task, with a
> double review (spec compliance + code quality) after every task.

**Goal:** All 21 contraction-map methods of `src/contraction/basic.jl` declared
as engine `Chain`s with @tensor-pinned intermediate layouts; the
`forloop`/`parallel` rrules rerouted from per-slice `Zygote.pullback` to the
engine's generic backward; per-map parity gates 1e-12 (forward) / 1e-10
(gradient) against the original `@tensor` kernels; single-M `conj` support
moved INTO the engine (per-op conj flags, no `conj(M)` materialization).

**Architecture:** The M1 engine (`src/contraction/chain_engine.jl`) gains
per-op conj flags and a layout-derivation helper that reproduces `@tensor`'s
temporary layouts. A new `src/contraction/chain_maps.jl` holds the chain
declarations, the map→chain glue (`_chain_map`), the `engine_backward`
registry, and the global toggle. Each map method in `basic.jl` gets a one-line
guard that routes to the chain when the toggle is on and operands are dense;
the original `@tensor` bodies remain verbatim as the fallback and parity
reference. The `forloop`/`parallel` rrules in `src/autodiff/rules.jl` try
`engine_backward` first and keep `Zygote.pullback` as the fallback for
unregistered `f`.

**Tech Stack:** Julia, TensorOperations 5.5.1 runtime API (verified installed),
Zygote (reference only — never inside engine/map backward paths), CUDA.jl
(`_free!`), MPI.jl (4-rank tests), existing kernels as parity references.

**Branch:** `claude/ecstatic-golick-e3f6f0` (this worktree; M1 commits live
here — the memory note saying `claude/sad-saha-3ec6bf` is stale).

---

## Context for implementers

**Key files.** Engine: `src/contraction/chain_engine.jl` (M1, 247 lines —
read it first; every M2 engine change extends it). Map kernels:
`src/contraction/basic.jl`. Slicing + wrappers:
`src/contraction/forloop_parallel_MPI.jl` (forloop :431, parallel :456,
forloop_sum :488, parallel_sum :514, the seven `*_parallel` wrappers
:543-680). AD rules: `src/autodiff/rules.jl` (rrule(forloop) :156,
rrule(parallel) :241). Leg aliases: `src/utils/misc.jl:3-6`. Include order:
`src/TeneT.jl` (basic :61, forloop :62, cannon :63, chain_engine :64 —
chain_maps.jl is inserted right after :64, before observable).

**Run commands (Windows; ALWAYS use the Bash tool — PowerShell mangles
quotes):**

```bash
cd "/d/1 - research/1.26 - iPEPS_opt/TeneT.jl/.claude/worktrees/ecstatic-golick-e3f6f0"
julia --project=. test/test_chain_engine.jl     # M1 engine suite (standalone)
julia --project=. test/test_chain_maps.jl       # NEW — M2 map-chain suite
julia --project=. test/run_test_cannon.jl       # 4-rank MPI cannon suite
julia --project=. test/run_test_parallel_engine.jl  # NEW (Task 7) — 4-rank MPI
julia --project=. test/runtests.jl              # full serial suite (Task 11)
```

`Manifest.toml` exists in this worktree (copied from the main checkout during
research — do not delete it). Tests are CPU; `CUDA.functional()` is false here.

**Verified facts (research round, 2026-06-12 — do NOT re-derive, but the
@macroexpand probes in Tasks 2/8 must be actually run):**

1. **@tensor lowers to left-assoc pairwise contractions in written-operand
   order** (`((A*B)*C)*D`; explicit parens override, as in Mumap/Mdmap), each
   link a `tensorcontract!` with temps `tensorfree!`d after last use.
2. **@tensor's temporary layouts are NOT the `(openA..., openB...)` concat.**
   The macro lays out each temp as a STABLE PARTITION of its natural
   `(openA..., openB...)` order: labels NOT contracted by the next operand
   first, labels contracted by the next operand last — BOTH blocks keeping
   their relative order in the natural concat (NOT the next operand's label
   order; this was probed three independent times on FLmap leg4, ACmap leg5,
   and Mumap/Mdmap — review round resolved the ambiguity). FLmap leg4 temps:
   `(a,h,d,g)` and `(h,e,a,b)`. ACmap leg5 temps: `(a,c,h,l,b,g)`,
   `(a,l,e,j,c,h,p)`, `(l,j,k,a,e,f)`. This is the M1 Part-7 lesson in
   mechanical form: matching these layouts is what makes cuTENSOR see
   identical permutation problems (M1 gate closed only after pinning;
   `benchmarks/Sofia_VUB_H200.md` Part 7).
3. **Runtime conj flags are bitwise exact**: `tensorcontract(..., A, IA, false,
   B, IB, true) == A ⋆ conj(B)` with norm diff 0.0. Integer labels work.
   `tensorcontract!(C, A, pA, conjA, B, pB, conjB, pAB, α, β)` exists
   (interface.jl:148).
4. **AD reality census**: the ONLY functions ever passed to
   `forloop`/`parallel` are FLmap/FRmap/ACmap/ACdmap (4-arg form: `(X, Y, Z,
   M)` with M dense leg4/leg5/leg8 or `Tuple{leg5,leg5}`). `forloop_sum`/
   `parallel_sum` (used by Mmap/Mumap/Mdmap) have **no rrule** — those maps
   are forward-only today (Mumap: preconditioner linsolve closure;
   Mmap/Mdmap: dead in src). LDmap/DRmap/RUmap/LUmap's only caller
   (`oc_Q_22_getQ_CBE`) is commented out. Cmap and FLmap_C3v are
   Zygote-differentiated directly (Cenv loop / qrctmrg_step). leftenv etc.
   have no rrules — fixed-point adjoints come from KrylovKit's eigsolve rrule
   (which Zygote-pullbacks the map closure → hits rrule(forloop)) or from
   unrolled `simple_eig`.
5. **leg aliases are Unions** (`AbstractArray{T,N}` ∪ `Vector{<:…}` ∪
   `StructArray{…}`, src/utils/misc.jl:3-6), but kernels only ever receive
   dense arrays or SubArray views at runtime. StructArray is NOT an
   AbstractArray, has `eltype Any`, and supports no TensorOperations methods —
   the chainability guard must exclude it and Vector-of-arrays.
6. **Toggle idiom**: module-level `const X = Ref(...)` (cf. `GC_RECLAIM_POOL`,
   src/utils/gpu.jl:74).
7. `pullback` in rules.jl is Zygote's. `_downcast_eltype` (basic.jl:14),
   `_boundary_cast` (forloop_parallel_MPI.jl:414), `_free!`
   (cannon_2d.jl:71-72) already exist and are reused.
8. **`Chain.out` is currently typed `NTuple{O, Symbol}`** (chain_engine.jl:19
   and the inner ctor :28) — the integer-label chains (FLmap_C3v, corner
   maps) CANNOT be constructed until Task 1 widens it. Task 1 owns this
   change; later tasks assume it.

**Design invariants (every reviewer checks every task against these):**

- **No Zygote inside any engine or map backward path.** Zygote remains only as
  (a) the parity reference in tests, (b) the fallback branch for unregistered
  `f` in the forloop/parallel rrules, (c) the outer driver that *calls* our
  rrules.
- **Old `@tensor` bodies stay verbatim** behind the guard until M2 is done;
  the toggle default stays `false` until Task 11.
- **Parity gates: forward rtol 1e-12, gradient rtol 1e-10**, complex tensors,
  non-trivial random dOut, every leg variant and every M form of every map.
- **Chain ops order = the map's @tensor written operand order** (NOT the map's
  positional-arg order). Intermediate layouts pinned via
  `tensor_pinned_inters` (Task 2) so cuTENSOR sees the same permutation
  problems as today. `FLMAP_LEG5_CHAIN` keeps its proven hand-kernel pins —
  do not touch it.
- **Engine never frees caller arrays**; every owned array freed after its last
  consumer; rrule closures capture inputs only (recompute-style backward).
- Existing suites stay green after every task: `test_chain_engine.jl` +
  `run_test_cannon.jl` (and from Task 7 on, `run_test_parallel_engine.jl`).

---

## The per-op conj rule (Task 1 math — reviewers verify against this)

`Chain` gains `conjs::NTuple{N,Bool}`. Forward link k (k in 2:N) contracts
`acc ⋆ tensors[k]` with `conjA = (k == 2 ? conjs[1] : false)` (intermediates
are never conj-flagged) and `conjB = conjs[k]`.

Backward of a link `C = A° ⋆ B°` where `X° = cX ? conj(X) : X`, with cotangent
dC, in tensorcontract flag form (derivation: standard adjoints
`dA° = dC ⋆ conj(B°)`, `dB° = conj(A°) ⋆ dC`, then `dX = cX ? conj(dX°) : dX°`,
and `conj(tensorcontract(..fA.., ..fB..)) = tensorcontract(..!fA.., ..!fB..)`):

```
d(tensors[k]) = tensorcontract(ops[k],  Iprev, lIprev, !(cA ⊻ cB), dC, lC, cB)
dIprev        = tensorcontract(lIprev,  dC,    lC,     cA,  tensors[k], ops[k], !(cA ⊻ cB))
```

with `cA = (k == 2 ? conjs[1] : false)`, `cB = conjs[k]`. For
`cA = cB = false` these reduce exactly to the existing M1 lines
(chain_engine.jl:194,196) — that is the regression anchor. The same flag rule
applies in `chain_link1_back` (cA = conjs[1], cB = conjs[2]) and in the
`*_from1` variants (k ≥ 3, so cA = false always). NOTE: the FORWARD flag rule
also applies inside the forward-recompute loops of `chain_backward`
(chain_engine.jl:181-186) and `chain_backward_from1` (:225-229) — a
backward-only reading recomputes wrong intermediates.

Single-M maps then pass the SAME tensor twice and the glue sums the two
returned gradient slots: `dM = g_slot1 .+ g_slot2` (both already w.r.t. raw M
because the engine handled the conj). This matches Zygote's `conj` pullback
(`dM += conj(d_conjM)`) — proven by the Task-1 tests.

---

## Chain declarations (the data — Task 2 helper pins all `inters`)

All ops/out tuples below are transcribed from the `@tensor` bodies in
`basic.jl` (line refs given); the implementer of each task re-checks them
against the source before committing. `tensor_chain(ops, out) =
Chain(ops, out, tensor_pinned_inters(ops, out))`. `conj_variant(ch, slots...)`
returns the same chain with `conjs[slot] = true`.

| # | Constant | ops (@tensor order) | out | map args → chain tensors |
|---|----------|---------------------|-----|--------------------------|
| 1 | `FLMAP_LEG4_CHAIN` (basic.jl:56) | FL(a,d,f), ALd(f,g,h), M(d,g,e,b), ALu(a,b,c) | (c,e,h) | (FL,ALu,ALd,M) → (FL,ALd,M,ALu) |
| 2 | `FLMAP_LEG5_CHAIN` (exists, :70) | FL, ALd, M1, M2, ALu | (d,g,h,l) | (FL,ALu,ALd,M1,M2) → (FL,ALd,M1,M2,ALu); 1M: `conj_variant(…, 4)`, tensors (FL,ALd,M,M,ALu) |
| 3 | `FLMAP_LEG8_CHAIN` (:85) | FL(a,e,f,i), ALd(i,j,k,l), M(e,f,j,k,g,h,b,c), ALu(a,b,c,d) | (d,g,h,l) | (FL,ALu,ALd,M) → (FL,ALd,M,ALu) |
| 4 | `FLMAP_C3V_CHAIN` (:105) | FL(1,4,44,5), ALu(1,2,22,3), M1(4,7,2,10), M2(44,77,22,10), M3(6,8,7,11), M4(66,88,77,11), ALd(5,6,66,9) | (3,8,88,9) | (FL,ALu,ALd,M1..M4) → (FL,ALu,M1,M2,M3,M4,ALd); 1M/2M: `conj_variant(…, 4, 6)` with tensors (FL,ALu,M,M,M,M,ALd) / (FL,ALu,M1,M1,M2,M2,ALd) |
| 5 | `FRMAP_LEG4_CHAIN` (:137) | ARd(f,g,h), FR(c,e,h), M(d,g,e,b), ARu(a,b,c) | (a,d,f) | (FR,ARu,ARd,M) → (ARd,FR,M,ARu) |
| 6 | `FRMAP_LEG5_CHAIN` (:151) | ARd(i,j,k,l), FR(d,g,h,l), M1(e,j,g,b,p), M2(f,k,h,c,p), ARu(a,b,c,d) | (a,e,f,i) | → (ARd,FR,M1,M2,ARu); 1M: `conj_variant(…, 4)` |
| 7 | `FRMAP_LEG8_CHAIN` (:166) | ARd(i,j,k,l), FR(d,g,h,l), M(e,f,j,k,g,h,b,c), ARu(a,b,c,d) | (a,e,f,i) | → (ARd,FR,M,ARu) |
| 8 | `ACMAP_LEG4_CHAIN` (:241) | AC(a,b,c), FR(c,e,h), M(d,g,e,b), FL(a,d,f) | (f,g,h) | (AC,FL,FR,M) → (AC,FR,M,FL) |
| 9 | `ACMAP_LEG5_CHAIN` (:255) | AC(a,b,c,d), FR(d,g,h,l), M1(e,j,g,b,p), M2(f,k,h,c,p), FL(a,e,f,i) | (i,j,k,l) | → (AC,FR,M1,M2,FL); 1M: `conj_variant(…, 4)` |
| 10 | `ACMAP_LEG8_CHAIN` (:270) | AC(a,b,c,d), FR(d,g,h,l), M(e,f,j,k,g,h,b,c), FL(a,e,f,i) | (i,j,k,l) | → (AC,FR,M,FL) |
| 11 | `ACDMAP_LEG4_CHAIN` (:317) | ACd(f,g,h), FR(c,e,h), M(d,g,e,b), FL(a,d,f) | (a,b,c) | (ACd,FL,FR,M) → (ACd,FR,M,FL) |
| 12 | `ACDMAP_LEG5_CHAIN` (:331) | ACd(i,j,k,l), FR(d,g,h,l), M1(e,j,g,b,p), M2(f,k,h,c,p), FL(a,e,f,i) | (a,b,c,d) | → (ACd,FR,M1,M2,FL); 1M: `conj_variant(…, 4)` |
| 13 | `CMAP_LEG3_CHAIN` (:300) | FL(a,c,d), C(a,b), FR(b,c,e) | (d,e) | (C,FL,FR) → (FL,C,FR) |
| 14 | `CMAP_LEG4_CHAIN` (:304) | FL(a,c,d,e), C(a,b), FR(b,c,d,f) | (e,f) | (C,FL,FR) → (FL,C,FR) |
| 15 | `MMAP_CHAIN` (:351) | AC(a,b,c), FR(c,e,h), FL(a,d,f), ACd(f,g,h) | (d,g,e,b) | (AC,ACd,FL,FR) → (AC,FR,FL,ACd) |
| 16 | `MUMAP_INNER_CHAIN` + `MUMAP_OUTER_CHAIN` (:355, tree!) | inner: FL(a,e,f,i), ACd(i,j,k,l), Mu(e,j,g,b,p) → Y; outer: AC(a,b,c,d), FR(d,g,h,l), Y(labels = inner out) | inner out **(a,b,g,l,f,k,p)**, inner inter (a,f,k,l,e,j), outer inter (c,h,a,b,g,l) — all probe-verified (re-confirm in Task 8); outer out (f,k,h,c,p) | (AC,ACd,FL,FR,Mu) — composed glue `_chain_Mumap` |
| 17 | `MDMAP_INNER_CHAIN` + `MDMAP_OUTER_CHAIN` (:359, tree!) | inner: FL(a,e,f,i), ACd(i,j,k,l), Md(f,k,h,c,p) → Y; outer: AC(a,b,c,d), FR(d,g,h,l), Y | inner out **(a,c,h,l,e,j,p)**, inner inter (a,e,j,l,f,k), outer inter (b,g,a,c,h,l) — probe-verified (re-confirm in Task 8); outer out (e,j,g,b,p) | (AC,ACd,FL,FR,Md) — `_chain_Mdmap` |
| 18 | `LDMAP_CHAIN` (:363) | L(1,5,6,9), D(9,10,11,12), M1(5,10,7,2,13), M2(6,11,8,3,13) | (1,2,3,7,8,12) | (L,D,M1,M2) as-is; 1M: `conj_variant(…, 4)` |
| 19 | `DRMAP_CHAIN` (:364) | D(9,10,11,12), R(4,7,8,12), M1(5,10,7,2,13), M2(6,11,8,3,13) | (9,5,6,2,3,4) | (D,R,M1,M2) as-is; 1M: `conj_variant(…, 4)` |
| 20 | `RUMAP_CHAIN` (:365) | U(1,2,3,4), R(4,7,8,12), M1(5,10,7,2,13), M2(6,11,8,3,13) | (12,10,11,5,6,1) | (R,U,M1,M2) → (U,R,M1,M2); 1M: `conj_variant(…, 4)` |
| 21 | `LUMAP_CHAIN` (:366) | L(1,5,6,9), U(1,2,3,4), M1(5,10,7,2,13), M2(6,11,8,3,13) | (9,10,11,7,8,4) | (L,U,M1,M2) as-is; 1M: `conj_variant(…, 4)` |

Integer labels (rows 4, 18-21) work in runtime `tensorcontract` (verified) —
but the `Chain` STRUCT only admits them after Task 1 widens `out` (context
fact #8). Mumap/Mdmap are the only trees; everything else is a pure
left-assoc chain (each link contracts ≥1 label with the accumulated tensor —
re-verify per task).

---

### Task 1: Per-op conj flags in the engine core

**Files:** Modify `src/contraction/chain_engine.jl`; extend
`test/test_chain_engine.jl` (additive testsets only — existing ones must not
change).

**Step 1 — failing tests.** Append to `test/test_chain_engine.jl`:

```julia
@testset "per-op conj flags vs Zygote (8 combos, 3-op chain)" begin
    Random.seed!(21)
    A = rand(ComplexF64, 4, 5, 3)        # (:x,:y,:s)
    B = rand(ComplexF64, 3, 6, 2)        # (:s,:t,:u)
    C = rand(ComplexF64, 2, 4, 7)        # (:u,:x,:v)
    ops, out = ((:x,:y,:s), (:s,:t,:u), (:u,:x,:v)), (:y,:t,:v)
    mc(x, c) = c ? conj(x) : x
    function ref(a, b, c, cA, cB, cC)
        I1 = tensorcontract((:x,:y,:t,:u), mc(a,cA), ops[1], false, mc(b,cB), ops[2], false)
        return tensorcontract(out, I1, (:x,:y,:t,:u), false, mc(c,cC), ops[3], false)
    end
    for cA in (false,true), cB in (false,true), cC in (false,true)
        ch = Chain(ops, out, nothing, (cA,cB,cC))
        r  = ref(A, B, C, cA, cB, cC)
        @test TeneT.chain_apply(ch, (A,B,C)) ≈ r rtol = 1e-12
        dOut = rand(ComplexF64, size(r))
        _, back = Zygote.pullback((a,b,c) -> ref(a,b,c,cA,cB,cC), A, B, C)
        dz = back(dOut)
        ge = TeneT.chain_backward(ch, (A,B,C), dOut)
        for i in 1:3
            @test ge[i] ≈ dz[i] rtol = 1e-10
        end
        # link1 helpers with flags — value-level vs the materialized reference
        I1ref = tensorcontract((:x,:y,:t,:u), mc(A,cA), ops[1], false, mc(B,cB), ops[2], false)
        Hacc = zero(I1ref)
        TeneT.chain_link1_add!(Hacc, ch, A, B)
        @test Hacc ≈ I1ref rtol = 1e-12
        dH = rand(ComplexF64, size(I1ref))
        _, lb = Zygote.pullback((a, b) -> tensorcontract((:x,:y,:t,:u),
                    mc(a,cA), ops[1], false, mc(b,cB), ops[2], false), A, B)
        dA_z, dB_z = lb(dH)
        dA1, dB1 = TeneT.chain_link1_back(ch, dH, A, B)
        @test dA1 ≈ dA_z rtol = 1e-10
        @test dB1 ≈ dB_z rtol = 1e-10
    end
end

@testset "integer-label chains construct and run" begin
    Random.seed!(24)
    A = rand(ComplexF64, 3, 4); B = rand(ComplexF64, 4, 5); C = rand(ComplexF64, 5, 2)
    ch = Chain(((1,2), (2,3), (3,4)), (1,4))      # Int labels end-to-end
    @test TeneT.chain_apply(ch, (A,B,C)) ≈ A*B*C rtol = 1e-12
end

@testset "conj_variant + from1 variants with flags (4-op chain)" begin
    Random.seed!(22)
    A = rand(ComplexF64, 4, 3); B = rand(ComplexF64, 3, 5)
    M = rand(ComplexF64, 5, 6); D = rand(ComplexF64, 6, 2)
    ops, out = ((:a,:b), (:b,:c), (:c,:d), (:d,:e)), (:a,:e)
    base = Chain(ops, out)
    ch = TeneT.conj_variant(base, 3)
    @test ch.conjs == (false, false, true, false)
    ref = A * B * conj(M) * D
    @test TeneT.chain_apply(ch, (A,B,M,D)) ≈ ref rtol = 1e-12
    dOut = rand(ComplexF64, size(ref))
    _, back = Zygote.pullback((a,b,m,d) -> a*b*conj(m)*d, A, B, M, D)
    dz = back(dOut)
    ge = TeneT.chain_backward(ch, (A,B,M,D), dOut)
    for i in 1:4; @test ge[i] ≈ dz[i] rtol = 1e-10; end
    # from1: start from H = A*B (links 3..4 carry the conj flag on op 3)
    H = A * B
    @test TeneT.chain_apply_from1(ch, H, (M, D)) ≈ ref rtol = 1e-12
    dH, dM, dD = TeneT.chain_backward_from1(ch, H, (M, D), dOut)
    @test dM ≈ dz[3] rtol = 1e-10
    @test dD ≈ dz[4] rtol = 1e-10
    @test dH ≈ dOut * (conj(M) * D)' rtol = 1e-10  # dH = dOut ⋆ conj(conj(M)*D)
end

@testset "single-M FLmap leg5 via conj_variant (engine-level)" begin
    Random.seed!(23)
    χ, D, d = 12, 3, 2
    FL  = rand(ComplexF64, χ, D, D, χ); ALu = rand(ComplexF64, χ, D, D, χ)
    ALd = rand(ComplexF64, χ, D, D, χ); M   = rand(ComplexF64, D, D, D, D, d)
    ch1m = TeneT.conj_variant(FLMAP_LEG5_CHAIN, 4)
    ref  = TeneT.FLmap(FL, ALu, ALd, M)          # dispatches to (M, conj(M))
    @test TeneT.chain_apply(ch1m, (FL, ALd, M, M, ALu)) ≈ ref rtol = 1e-12
    dOut = rand(ComplexF64, size(ref))
    _, back = Zygote.pullback((fl,alu,ald,m) -> TeneT.FLmap(fl,alu,ald,m), FL, ALu, ALd, M)
    dFL_z, dALu_z, dALd_z, dM_z = back(dOut)
    g = TeneT.chain_backward(ch1m, (FL, ALd, M, M, ALu), dOut)
    @test g[1] ≈ dFL_z rtol = 1e-10
    @test g[2] ≈ dALd_z rtol = 1e-10
    @test g[3] .+ g[4] ≈ dM_z rtol = 1e-10       # slot-sum convention
    @test g[5] ≈ dALu_z rtol = 1e-10
end
```

(Matrix chains in the second testset keep the math auditable; `tensorcontract`
handles 2-leg tensors fine.)

**Step 2 — run, expect FAIL** (`Chain` has no 4-arg ctor / no `conjs` /
`conj_variant` undefined):
`julia --project=. test/test_chain_engine.jl`

**Step 3 — implement** in `chain_engine.jl`:
- **Widen `out`**: field `out::NTuple{O, Symbol}` → `out::NTuple{O, Any}` and
  drop the `Symbol` constraint from the inner-ctor signature — the
  integer-label chains (FLmap_C3v, corner maps) need this; all label logic
  (`_link_labels`, the `Set` assertion, `tensorcontract`) is already
  type-agnostic.
- Add field `conjs::NTuple{N, Bool}` to `Chain`; inner constructor takes
  `(ops, out, inters, conjs)` and keeps the inters assertion; outer ctors
  `Chain(ops, out) = Chain(ops, out, nothing)`,
  `Chain(ops, out, inters) = Chain(ops, out, inters, ntuple(_ -> false, length(ops)))`.
  `FLMAP_LEG5_CHAIN`'s 3-arg construction keeps working unchanged.
- `conj_variant(ch::Chain{N}, slots::Int...)` — same ops/out/inters, conjs
  with the given slots set true (on top of ch.conjs).
- Thread flags exactly per the formula box above through: `chain_apply`
  (conjA at k==2, conjB everywhere), `chain_apply_from1` (k ≥ 3: conjB only),
  `chain_link1_add!` (pass `ch.conjs[1]`/`ch.conjs[2]` to `tensorcontract!`),
  `chain_link1_back`, `chain_backward`, `chain_backward_from1`. Keep the
  free-ordering EXACTLY as is (flags do not change ownership).

**Step 4 — run, expect PASS, including all pre-existing testsets.** Also run
`julia --project=. test/run_test_cannon.jl` (must stay green).

**Step 5 — commit:** `feat: per-op conj flags in the chain engine`

---

### Task 2: Layout-pinning helper, toggle, chainability guard, chain_apply rrule

**Files:** Modify `src/contraction/chain_engine.jl` (helper), create
`src/contraction/chain_maps.jl`, modify `src/TeneT.jl` (include after
chain_engine.jl), modify `src/autodiff/rules.jl` (rrule), create
`test/test_chain_maps.jl`.

**Step 1 — @macroexpand probe (CONFIRMATION — the rule is already resolved).**
The review round ran this probe twice independently; the macro orders the
to-be-contracted block by appearance in the TEMP'S NATURAL `(openA…, openB…)`
order, NOT in the next operand (ACmap-leg5 I₁ = `(a,c,h,l,b,g)`,
I₂ = `(a,l,e,j,c,h,p)` — both discriminate). Re-run the probe once as
confirmation (delete the script after):

```julia
using TensorOperations
FL = rand(5,3,4); ALd = rand(4,3,6); M = rand(3,3,3,3); ALu = rand(5,3,7);
println(@macroexpand @tensor result[c,e,h] := FL[a,d,f]*ALd[f,g,h]*M[d,g,e,b]*ALu[a,b,c])
AC = rand(5,3,3,5); FR = rand(5,3,3,5); M1 = rand(3,3,3,3,2); M2 = rand(3,3,3,3,2); FL5 = rand(5,3,3,5);
println(@macroexpand @tensor r[i,j,k,l] := AC[a,b,c,d]*FR[d,g,h,l]*M1[e,j,g,b,p]*M2[f,k,h,c,p]*FL5[a,e,f,i])
```

Decode each temp's layout from its producing `pAB` over the natural
`(openA…, openB…)` order (flattened `pAB` tuple = permutation natural→temp;
validate your decoding on the final link, whose output must equal the declared
LHS). Expected: FLmap-leg4 temps `(a,h,d,g)`, `(h,e,a,b)`; ACmap-leg5 temps
`(a,c,h,l,b,g)`, `(a,l,e,j,c,h,p)`, `(l,j,k,a,e,f)`. If the probe disagrees
with these, STOP and re-open the rule question — do not reconcile silently.

**Step 2 — failing tests.** `test/test_chain_maps.jl` (new, standalone, same
header style as test_chain_engine.jl):

```julia
# M2 map-chain tests. Run: julia --project=. test/test_chain_maps.jl
using Test, LinearAlgebra, Random, Zygote
using TeneT
using TeneT: Chain, chain_interlabels, tensor_chain, tensor_pinned_inters,
             conj_variant, chain_apply, use_chain_engine, CHAIN_ENGINE
using TensorOperations: tensorcontract

@testset "tensor_pinned_inters matches @tensor temp layouts" begin
    # FLmap leg4 — transcribed from @macroexpand (research probe + Step 1):
    ops = ((:a,:d,:f), (:f,:g,:h), (:d,:g,:e,:b), (:a,:b,:c))
    @test tensor_pinned_inters(ops, (:c,:e,:h)) == ((:a,:h,:d,:g), (:h,:e,:a,:b))
    # ACmap leg5 — probe-verified layouts (review round, two independent runs):
    ops5 = ((:a,:b,:c,:d), (:d,:g,:h,:l), (:e,:j,:g,:b,:p), (:f,:k,:h,:c,:p), (:a,:e,:f,:i))
    @test tensor_pinned_inters(ops5, (:i,:j,:k,:l)) ==
          ((:a,:c,:h,:l,:b,:g), (:a,:l,:e,:j,:c,:h,:p), (:l,:j,:k,:a,:e,:f))
    # N=3 edge case (one intermediate), N=2 → nothing
    @test tensor_pinned_inters(((:a,:b), (:b,:c), (:c,:d)), (:a,:d)) isa NTuple{1,Tuple}
    @test tensor_pinned_inters(((:a,:b), (:b,:c)), (:a,:c)) === nothing
end

@testset "toggle + chainability guard" begin
    @test CHAIN_ENGINE[] == false                       # M2 default until Task 11
    A = rand(ComplexF64, 2, 2); V = [A, A]
    @test TeneT._chainable(A)
    @test TeneT._chainable(view(A, :, 1:1))
    @test !TeneT._chainable(V)                          # Vector-of-arrays excluded
    @test TeneT._chainable((A, A)) && !TeneT._chainable((A, V))
    CHAIN_ENGINE[] = true
    @test use_chain_engine(A, (A, A))
    @test !use_chain_engine(A, V)
    CHAIN_ENGINE[] = false
    @test !use_chain_engine(A)
end

@testset "rrule(chain_apply) under Zygote == Zygote over @tensor" begin
    Random.seed!(31)
    χ, D, d = 8, 3, 2
    FL  = rand(ComplexF64, χ, D, D, χ); ALu = rand(ComplexF64, χ, D, D, χ)
    ALd = rand(ComplexF64, χ, D, D, χ)
    M1  = rand(ComplexF64, D, D, D, D, d); M2 = rand(ComplexF64, D, D, D, D, d)
    loss_chain(t...) = sum(abs2, chain_apply(TeneT.FLMAP_LEG5_CHAIN, t))
    loss_tensor(fl, ald, m1, m2, alu) = sum(abs2, TeneT.FLmap(fl, alu, ald, m1, m2))
    gc = Zygote.gradient(loss_chain, FL, ALd, M1, M2, ALu)
    gt = Zygote.gradient(loss_tensor, FL, ALd, M1, M2, ALu)
    for i in 1:5; @test gc[i] ≈ gt[i] rtol = 1e-10; end
end

@testset "_chain_map inner_etype cast path" begin
    Random.seed!(32)
    χ, D = 8, 3
    # FLmap leg4 geometry (basic.jl:56 / test_contraction.jl:68-74):
    FL  = rand(ComplexF64, χ, D, χ); ALd = rand(ComplexF64, χ, D, χ)
    M   = rand(ComplexF64, D, D, D, D); ALu = rand(ComplexF64, χ, D, χ)
    ch  = tensor_chain(((:a,:d,:f), (:f,:g,:h), (:d,:g,:e,:b), (:a,:b,:c)), (:c,:e,:h))
    r64 = TeneT._chain_map(ch, (FL, ALd, M, ALu), nothing)
    r32 = TeneT._chain_map(ch, (FL, ALd, M, ALu), Float32)
    @test eltype(r32) == ComplexF64                     # upcast at exit
    @test r32 ≈ r64 rtol = 1e-5                         # F32 accuracy
end

println("test_chain_maps done")
```

**Step 3 — run, expect FAIL** (missing helper/file/etc.).

**Step 4 — implement.**

In `chain_engine.jl` (below `_link_labels`):

```julia
# @tensor's temp-layout rule (probe-verified three times, see plan): each
# intermediate is a STABLE PARTITION of its natural (openA..., openB...)
# order — labels not contracted by the NEXT operand first, labels contracted
# by the next operand last, both blocks in natural relative order. Pinning
# new chains to these layouts gives cuTENSOR the identical permutation
# problems as the original @tensor kernels (M1 Part-7 lesson).
function tensor_pinned_inters(ops::NTuple{N, Tuple}, out::Tuple) where {N}
    N < 3 && return nothing
    inters = Vector{Tuple}(undef, N - 2)
    layout = ops[1]
    for j in 1:(N - 2)
        natural = _link_labels(layout, ops[j + 1])
        nextop  = ops[j + 2]
        keep = filter(l -> !(l in nextop), natural)
        cont = filter(l -> l in nextop, natural)   # temp-natural order, NOT nextop order
        inters[j] = (keep..., cont...)
        layout = inters[j]
    end
    @assert Set(out) == Set(_link_labels(layout, ops[N])) "tensor_pinned_inters: out must be a permutation of the final link's labels"
    return Tuple(inters)
end

tensor_chain(ops, out) = Chain(ops, out, tensor_pinned_inters(ops, out))
```

New `src/contraction/chain_maps.jl` (include in TeneT.jl after
chain_engine.jl):

```julia
# M2: map→chain glue. Chain declarations for the 21 basic.jl map methods,
# the engine toggle, and the engine_backward registry used by the
# forloop/parallel rrule reroute. Plan: docs/2026-06-12-chain-engine-m2-plan.md

# Global switch. Default false until M2 parity + the Sofia gate pass (Task 11).
const CHAIN_ENGINE = Ref(false)
set_chain_engine!(b::Bool) = (CHAIN_ENGINE[] = b; b)

# Dense numeric arrays (incl. SubArray views) only: leg aliases also admit
# Vector-of-arrays and StructArray, which the runtime API cannot contract.
_chainable(x) = x isa AbstractArray && eltype(x) <: Number
_chainable(t::Tuple) = all(_chainable, t)
_chainable(x, rest...) = _chainable(x) && _chainable(rest...)
use_chain_engine(args...) = CHAIN_ENGINE[] && _chainable(args...)

# Route a map through the engine, mirroring the kernels' inner_etype
# structure: downcast at entry, upcast at exit, identity when unset.
function _chain_map(ch::Chain, tensors::Tuple, inner_etype)
    if inner_etype === nothing || inner_etype == real(eltype(tensors[1]))
        return chain_apply(ch, tensors)
    end
    T_out = eltype(tensors[1])
    return T_out.(chain_apply(ch, map(t -> _downcast_eltype(inner_etype, t), tensors)))
end

# engine_backward(f, args, dOut) -> gradients in f's positional-arg order
# (Tuple-M arg ⇒ tuple gradient), or `nothing` when this (f, args) form has
# no chain — the rrule caller then falls back to Zygote. NEVER uses Zygote.
engine_backward(f, args, dOut) = nothing
```

rrule in `src/autodiff/rules.jl` (near the forloop rrule):

```julia
function ChainRulesCore.rrule(::typeof(chain_apply), ch::Chain{N}, tensors::NTuple{N, Any}) where {N}
    out = chain_apply(ch, tensors)
    # Recompute-style: the closure captures only caller-owned inputs; no
    # intermediate ever outlives the call (the cannon/Part-6 OOM lesson).
    function chain_apply_pullback(dOut)
        return NoTangent(), NoTangent(), chain_backward(ch, tensors, unthunk(dOut))
    end
    return out, chain_apply_pullback
end
```

**Step 5 — run both test files + cannon suite; expect PASS. Commit:**
`feat: layout-pinning helper, engine toggle, chain_apply rrule (M2 scaffold)`

---

### Task 3: FLmap family (leg4, leg8, leg5 single-M) — the per-map template

Every later map task follows this task's shape. **Files:** modify
`src/contraction/chain_maps.jl` (chains + engine_backward), modify
`src/contraction/basic.jl` (guard lines only), extend
`test/test_chain_maps.jl`.

**Step 1 — failing tests** (append; this block is the TEMPLATE — later tasks
instantiate it per map):

```julia
@testset "FLmap chains: parity over all variants" begin
    Random.seed!(41)
    χ, D, d = 8, 3, 2
    FL4  = rand(ComplexF64, χ, D, χ);    ALu4 = rand(ComplexF64, χ, D, χ)
    ALd4 = rand(ComplexF64, χ, D, χ);    M4   = rand(ComplexF64, D, D, D, D)
    FL5  = rand(ComplexF64, χ, D, D, χ); ALu5 = rand(ComplexF64, χ, D, D, χ)
    ALd5 = rand(ComplexF64, χ, D, D, χ); M5   = rand(ComplexF64, D, D, D, D, d)
    M8   = rand(ComplexF64, D, D, D, D, D, D, D, D)
    cases = [
        ((FL4, ALu4, ALd4, M4),        "leg4"),
        ((FL5, ALu5, ALd5, M5),        "leg5 single-M"),
        ((FL5, ALu5, ALd5, (M5, conj(M5))), "leg5 tuple"),
        ((FL5, ALu5, ALd5, M8),        "leg8"),
    ]
    for (args, name) in cases
        TeneT.set_chain_engine!(false)
        ref = TeneT.FLmap(args...)
        gref = Zygote.gradient((a...) -> sum(abs2, TeneT.FLmap(a...)), args...)
        geng = try                       # toggle hygiene: never leak ON state
            TeneT.set_chain_engine!(true)
            @test TeneT.FLmap(args...) ≈ ref rtol = 1e-12
            Zygote.gradient((a...) -> sum(abs2, TeneT.FLmap(a...)), args...)
        finally
            TeneT.set_chain_engine!(false)
        end
        for i in 1:4
            if gref[i] isa Tuple
                for j in 1:2; @test geng[i][j] ≈ gref[i][j] rtol = 1e-10; end
            else
                @test geng[i] ≈ gref[i] rtol = 1e-10
            end
        end
        # engine_backward registry (the forloop-reroute entry point):
        dOut = rand(ComplexF64, size(ref))
        _, bk = Zygote.pullback((a...) -> TeneT.FLmap(a...), args...)
        gz = bk(dOut)
        ge = TeneT.engine_backward(TeneT.FLmap, args, dOut)
        @test ge !== nothing
        for i in 1:4
            if gz[i] isa Tuple
                for j in 1:2; @test ge[i][j] ≈ gz[i][j] rtol = 1e-10; end
            else
                @test ge[i] ≈ gz[i] rtol = 1e-10
            end
        end
    end
    # inner_etype path survives the reroute (mirrors test_contraction.jl):
    r32 = try
        TeneT.set_chain_engine!(true)
        TeneT.FLmap(FL5, ALu5, ALd5, M5; inner_etype=Float32)
    finally
        TeneT.set_chain_engine!(false)
    end
    @test r32 ≈ TeneT.FLmap(FL5, ALu5, ALd5, M5; inner_etype=Float32) rtol = 1e-5
end
```

**Toggle hygiene rule (applies to EVERY toggled region in every M2 test
file):** wrap engine-ON sections in `try ... finally set_chain_engine!(false)`
(after Task 11: restore `true`) — a mid-region exception must never leak the
toggle state into later testsets.

**Step 2 — run, expect FAIL. Step 3 — implement.**

`chain_maps.jl` additions:

```julia
const FLMAP_LEG4_CHAIN = tensor_chain(((:a,:d,:f), (:f,:g,:h), (:d,:g,:e,:b), (:a,:b,:c)), (:c,:e,:h))
const FLMAP_LEG8_CHAIN = tensor_chain(((:a,:e,:f,:i), (:i,:j,:k,:l), (:e,:f,:j,:k,:g,:h,:b,:c), (:a,:b,:c,:d)), (:d,:g,:h,:l))
const FLMAP_LEG5_CHAIN_1M = conj_variant(FLMAP_LEG5_CHAIN, 4)

function engine_backward(::typeof(FLmap), args::NTuple{4, Any}, dOut)
    FL, ALu, ALd, M = args
    if M isa Tuple && length(M) == 2 && _chainable(M)
        g = chain_backward(FLMAP_LEG5_CHAIN, (FL, ALd, M[1], M[2], ALu), dOut)
        return (g[1], g[5], g[2], (g[3], g[4]))
    elseif M isa AbstractArray && ndims(M) == 5
        g = chain_backward(FLMAP_LEG5_CHAIN_1M, (FL, ALd, M, M, ALu), dOut)
        dM = g[3]; dM .+= g[4]; _free!(g[4])           # slot-sum, in place
        return (g[1], g[5], g[2], dM)
    elseif M isa AbstractArray && ndims(M) == 4
        g = chain_backward(FLMAP_LEG4_CHAIN, (FL, ALd, M, ALu), dOut)
        return (g[1], g[4], g[2], g[3])
    elseif M isa AbstractArray && ndims(M) == 8
        g = chain_backward(FLMAP_LEG8_CHAIN, (FL, ALd, M, ALu), dOut)
        return (g[1], g[4], g[2], g[3])
    end
    return nothing
end
```

`basic.jl` guard lines (top of each method, original bodies untouched):

```julia
function FLmap(FL, ALu, ALd, M::leg4; inner_etype=nothing)
    use_chain_engine(FL, ALu, ALd, M) &&
        return _chain_map(FLMAP_LEG4_CHAIN, (FL, ALd, M, ALu), inner_etype)
    ...existing body...
```

same for the leg8 method (`FLMAP_LEG8_CHAIN`), the (M1,M2)-pair method
(`FLMAP_LEG5_CHAIN`, tensors `(FL, ALd, M1, M2, ALu)`), and the single-M
dispatch becomes:

```julia
function FLmap(FL, ALu, ALd, M::leg5; inner_etype=nothing)
    use_chain_engine(FL, ALu, ALd, M) &&
        return _chain_map(FLMAP_LEG5_CHAIN_1M, (FL, ALd, M, M, ALu), inner_etype)
    return FLmap(FL, ALu, ALd, M, conj(M); inner_etype)
end
```

(NOTE: forward refs from basic.jl to chain_maps.jl names are fine — basic.jl
is included first but the guards only EXECUTE after the module is fully
loaded. The Tuple{leg5,leg5} forwarder at basic.jl:100 needs no guard — it
forwards to the pair method which has one.)

**Step 4 — run all suites** (`test_chain_maps.jl`, `test_chain_engine.jl`,
`run_test_cannon.jl`) — PASS. **Step 5 — commit:**
`feat: FLmap family as chains (leg4/leg8/single-M) + engine_backward entry`

---

### Task 4: FRmap family (leg4, leg5 pair + 1M, leg8)

Instantiate the Task-3 template for FRmap. Chains #5-7 from the table
(`tensors (ARd, FR, M…, ARu)`; careful: map args are `(FR, ARu, ARd, M)` so
`engine_backward` returns `(g[2], g[5], g[1], dM-form)` for leg5 — derive the
permutation per variant from the table and SAY IT in a comment). Guards on
basic.jl:135/149/164 methods + single-M dispatch :179. Tests: the same
4-case × {fwd, Zygote-on/off, engine_backward, inner_etype} block with FRmap
geometry (`FR5 = rand(ComplexF64, χ,D,D,χ)` etc. — copy shapes from
`test_contraction.jl:275-292`).
**Run** test_chain_maps.jl + test_chain_engine.jl + run_test_cannon.jl — PASS.
**Commit:** `feat: FRmap family as chains`

### Task 5: ACmap family (leg4, leg5 pair + 1M, leg8)

Chains #8-10 (`tensors (AC, FR, M…, FL)`; map args `(AC, FL, FR, M)` ⇒ leg5
grad permutation `(g[1], g[5], g[2], M-form)`). Guards on basic.jl:239/253/268
+ :283. Same test block, ACmap geometry.
**Run** test_chain_maps.jl + test_chain_engine.jl + run_test_cannon.jl — PASS.
**Commit:** `feat: ACmap family as chains`

### Task 6: ACdmap family (leg4, leg5 pair + 1M) + Cmap (leg3/leg4)

Chains #11-14. ACdmap: map args `(ACd, FL, FR, M)`, tensors
`(ACd, FR, M…, FL)`. Cmap has no inner_etype and no M — guards are
`use_chain_engine(C, FL, FR) && return chain_apply(CMAP_LEG3_CHAIN, (FL, C, FR))`
(use `chain_apply` directly; `_chain_map` is for inner_etype maps). Cmap needs
NO engine_backward entry (never goes through forloop/parallel — census), but
DOES need the Zygote-on/off gradient parity test (it is differentiated
directly in Cenv loops). ACdmap needs engine_backward (4-arg, leg4 + leg5
forms; it has no leg8 method — do not invent one).
**Run** test_chain_maps.jl + test_chain_engine.jl + run_test_cannon.jl — PASS.
**Commit:** `feat: ACdmap family + Cmap as chains`

---

### Task 7: forloop/parallel rrule reroute + MPI test

**Files:** modify `src/autodiff/rules.jl` (both rrules), extend
`test/test_chain_maps.jl` (serial forloop reroute tests), create
`test/test_parallel_engine.jl` + `test/run_test_parallel_engine.jl` (4-rank,
clone the run_test_cannon.jl launcher pattern).

**Step 1 — failing serial tests** (append to test_chain_maps.jl):

```julia
@testset "forloop rrule reroute: engine == Zygote path" begin
    Random.seed!(51)
    χ, D, d = 8, 3, 2
    FL  = rand(ComplexF64, χ, D, D, χ); ALu = rand(ComplexF64, χ, D, D, χ)
    ALd = rand(ComplexF64, χ, D, D, χ); M5  = rand(ComplexF64, D, D, D, D, d)
    for Mform in (M5, (M5, conj(M5))), n in (1, 3)
        loss(fl, alu, ald, m) = sum(abs2,
            TeneT.FLmap_parallel(fl, alu, ald, m; ifparallel=false, forloop_iter=n))
        TeneT.set_chain_engine!(false)
        gref = Zygote.gradient(loss, FL, ALu, ALd, Mform)
        TeneT.set_chain_engine!(true)
        geng = Zygote.gradient(loss, FL, ALu, ALd, Mform)
        TeneT.set_chain_engine!(false)
        for i in 1:4
            if gref[i] isa Tuple
                for j in 1:2; @test geng[i][j] ≈ gref[i][j] rtol = 1e-10; end
            else
                @test geng[i] ≈ gref[i] rtol = 1e-10
            end
        end
    end
    # Same for FRmap_parallel (dim-1 split), ACmap_parallel, ACdmap_parallel,
    # n ∈ (1, 3), single-M form only (tuple form covered above via FLmap).
    # ... (instantiate analogously)
    # inner_etype (do_cast) branch — real production traffic
    # (leftenv/rightenv/ACenv thread it under Zygote); cast tolerance:
    loss32(fl, alu, ald, m) = sum(abs2, TeneT.FLmap_parallel(fl, alu, ald, m;
        ifparallel=false, forloop_iter=3, inner_etype=Float32))
    TeneT.set_chain_engine!(false)
    g32ref = Zygote.gradient(loss32, FL, ALu, ALd, M5)
    g32eng = try
        TeneT.set_chain_engine!(true)
        Zygote.gradient(loss32, FL, ALu, ALd, M5)
    finally
        TeneT.set_chain_engine!(false)
    end
    for i in 1:4; @test g32eng[i] ≈ g32ref[i] rtol = 1e-5; end
end
```

**Step 2 — implement the reroute.** In BOTH rrules, replace each
`_, bp = pullback(f, split_args...); dargs_range = bp(dview)` site (and the
forloop_iter==1 `pullback(f, args_c...)` site) with:

```julia
dargs_range = use_chain_engine(split_args...) ?
              engine_backward(f, split_args, dview) : nothing
if dargs_range === nothing
    _, bp = pullback(f, split_args...)       # fallback: unregistered f / form
    dargs_range = bp(dview)
end
```

For the `forloop_iter == 1` branch, decide **at forward time** on
`use_chain_engine(args_c...)`:

- toggle/guard FALSE → keep the existing eager
  `result_c, back = pullback(f, args_c...)` **byte-identical** (no behavior
  change while the toggle is off — this protects the Task-12 TENSOR baseline
  and the Tasks-7→11 window from a double-forward regression);
- TRUE → `result_c = f(args_c...)` (tape-free; f routes through the chain),
  and `realback` calls `engine_backward(f, args_c, dresult_c)` with a
  `pullback(f, args_c...)`-recompute as the residual inner fallback (reached
  only if the registry rejects the concrete arg form).

The gradient-assembly loops (`.=` for the split arg, `.+=`/tuple-wise for the
rest) stay byte-identical — engine_backward's return shape was designed to
slot in.

dview notes: BOTH rrules hand the engine views (`view(_dresult_c, ...)` in
forloop; parallel's `_dresult_c[out_idx_r...]` sits under `@views` at
rules.jl:284, so it is a view too, NOT a copy) — the engine accepts views and
never frees `dOut`; do not change the indexing in this task.

**Step 3 — MPI test.** `test/test_parallel_engine.jl`: 4 ranks; FLmap
(last-dim split → has_split_gather=true) and FRmap (dim-1 split → allreduce
branch); single-M + tuple-M; `Zygote.gradient` over
`*_parallel(...; ifparallel=true, forloop_iter=2)` with toggle on vs toggle
off; rank-0 asserts 1e-10 agreement of ALL gradients. Launcher
`run_test_parallel_engine.jl` mirrors run_test_cannon.jl (`mpiexec -n 4`).
IMPORTANT: the launcher is launcher-only — the TEST file must clone
test_cannon.jl's header (`MPI.Init()`, comm/rank consts, 4-rank assert; no
explicit Finalize — MPI.jl's atexit handles it), because `parallel()` touches
`MPI.COMM_WORLD` immediately.

**Step 4 — run:** test_chain_maps.jl, run_test_parallel_engine.jl,
run_test_cannon.jl — PASS.
**Step 5 — commit:** `feat: forloop/parallel rrules reroute to the chain engine`

---

### Task 8: M-maps (Mmap chain; Mumap/Mdmap composed trees)

**Step 1 — @macroexpand probe** of Mumap and Mdmap (their bodies are
parenthesized TREES; decode pAB exactly as in Task 2 Step 1). Probe-verified
expectations (review round): Mumap Y = `(a,b,g,l,f,k,p)`, X = `(c,h,a,b,g,l)`;
Mdmap Y = `(a,c,h,l,e,j,p)`, X = `(b,g,a,c,h,l)` — these two Y's look similar;
do NOT cross-contaminate. Y is a B-side temp, so its layout follows a
DIFFERENT rule than tensor_pinned_inters derives — `MUMAP_INNER_CHAIN.out` /
`MDMAP_INNER_CHAIN.out` must be declared as the probe values, never
"corrected" by running the helper. The inner chains' single inter IS
helper-derivable (`(a,f,k,l,e,j)` / `(a,e,j,l,f,k)`), and the outer chains
come out right via `tensor_chain` once ops[3] = the pinned Y labels (their
single inter = X above — assert it). The outer chain's ops[3] label tuple
MUST equal the inner chain's `out` (add a `@test` for this consistency).

**Step 2 — failing tests:** fwd parity 1e-12 for Mmap/Mumap/Mdmap vs toggle
off (shapes: `test_contraction.jl:362` for Mmap; Mumap/Mdmap take
AC/ACd/FL/FR (χ,D,D,χ) + M (D,D,D,D,d) — read the `Mumap_parallel` call sites
in precondition.jl for the real geometry). Gradient parity 1e-10 via
Zygote-on/off (`sum(abs2, Mumap(...))`) — this works because the composed
glue gets an rrule (below) even though production use is forward-only today.

**Step 3 — implement.** `MMAP_CHAIN` = tensor_chain (#15), guard in basic.jl.
Composed maps:

```julia
function _chain_Mumap(AC, ACd, FL, FR, Mu)
    Y = chain_apply(MUMAP_INNER_CHAIN, (FL, ACd, Mu))
    out = chain_apply(MUMAP_OUTER_CHAIN, (AC, FR, Y))
    _free!(Y)
    return out
end
function ChainRulesCore.rrule(::typeof(_chain_Mumap), AC, ACd, FL, FR, Mu)
    Y = chain_apply(MUMAP_INNER_CHAIN, (FL, ACd, Mu))
    out = chain_apply(MUMAP_OUTER_CHAIN, (AC, FR, Y))
    _free!(Y)                                  # recomputed in the backward
    function back(dOut)
        Y2 = chain_apply(MUMAP_INNER_CHAIN, (FL, ACd, Mu))
        dAC, dFR, dY = chain_backward(MUMAP_OUTER_CHAIN, (AC, FR, Y2), unthunk(dOut))
        dFL, dACd, dMu = chain_backward(MUMAP_INNER_CHAIN, (FL, ACd, Mu), dY)
        _free!(Y2); _free!(dY)
        return NoTangent(), dAC, dACd, dFL, dFR, dMu
    end
    return out, back
end
```

COPY-PASTE HAZARD: `_chain_Mdmap` and its rrule are the same shape — when
writing them, replace EVERY `MUMAP_*` with `MDMAP_*` (the gradient parity
tests for BOTH maps will fail if one chain leaks into the other; that is the
detection mechanism — make sure both maps' grad tests exist). Guards in
basic.jl's Mumap/Mdmap route to `_chain_Mumap`/`_chain_Mdmap`. NO
engine_backward entries (forloop_sum/parallel_sum have no rrule — out of
scope; record this in the commit message).

**Step 4 — run** test_chain_maps.jl + test_chain_engine.jl +
run_test_cannon.jl + run_test_parallel_engine.jl — PASS. **Step 5 — commit:**
`feat: Mmap/Mumap/Mdmap as chains (tree maps composed, rrule'd glue)`

---

### Task 9: Corner maps LDmap/DRmap/RUmap/LUmap

Chains #18-21 (integer labels; RUmap's chain tensors order is `(U, R, M1, M2)`
while its args are `(R, U, ...)` — table). Guards on all three dispatch
levels (pair core, single-M via `conj_variant(…, 4)` with tensors
`(X, Y, M, M)`; the Tuple forwarders at basic.jl:373-376 need none — so two
guarded levels per map). Tests: fwd 1e-12 + Zygote-on/off grad 1e-10 for all
4 maps × {pair, single-M, tuple}. Geometry: the INPUTS L/D/R/U are all 4-leg
`(χ,D,D,χ)`-style (basic.jl:363-366 index lists); only the OUTPUTS are 6-leg
— read `oc_Q_22_getQ_CBE` observable.jl:108-136 for the concrete shapes; M is
(D,D,D,D,d). These maps are near-dead in src — the tests ARE the spec.
**Run** test_chain_maps.jl + test_chain_engine.jl + run_test_cannon.jl +
run_test_parallel_engine.jl — PASS.
**Commit:** `feat: corner maps LD/DR/RU/LU as chains`

### Task 10: FLmap_C3v (7-operand chain) + cast-branch bugfix

**Pre-fix (separate commit FIRST):** basic.jl:114 — the `inner_etype` branch
contracts raw `M3`/`M4` instead of downcast `M3_t`/`M4_t` (and never computes
them). Fix to downcast all four M's; run test_contraction.jl's C3v-adjacent
sets (none exist — add a 5-line Float32-consistency test). Commit:
`fix: FLmap_C3v inner_etype branch downcasts M3/M4`.

Then the template: `FLMAP_C3V_CHAIN` (#4, 7 ops, integer labels),
`FLMAP_C3V_CHAIN_CONJ46 = conj_variant(FLMAP_C3V_CHAIN, 4, 6)`; guards on the
7-arg core (tensors `(FL, ALu, M1, M2, M3, M4, ALd)`), the single-M dispatch
(tensors `(FL, ALu, M, M, M, M, ALd)` — Zygote-on/off grad parity here also
validates 4-fold slot accumulation through the chain_apply rrule), and the
2M dispatch (tensors `(FL, ALu, M1, M1, M2, M2, ALd)`). Geometry from
qrctmrg.jl:70/93 (T is 4-leg `(χ,D,D,χ)` per qrctmrg.jl:23 — transcribe
actual leg counts before writing tests). No engine_backward entry (not a
forloop map).
**Run** test_chain_maps.jl + test_chain_engine.jl + run_test_cannon.jl +
run_test_parallel_engine.jl — PASS.
**Commit:** `feat: FLmap_C3v as a 7-operand chain`

---

### Task 11: Flip the default ON + full suites

1. `CHAIN_ENGINE = Ref(true)` (chain_maps.jl) — one-line diff, trivially
   revertable.
2. Remove the `@test CHAIN_ENGINE[] == false` line in test_chain_maps.jl
   (assert `true` now).
3. Run, in order, ALL of: `test_chain_engine.jl`, `test_chain_maps.jl`,
   `run_test_cannon.jl`, `run_test_parallel_engine.jl`, and the FULL
   `test/runtests.jl` (CPU; the Float32/inner_etype and *_parallel testsets in
   test_contraction.jl now exercise the chain path — that is the point).
   Budget ≥30 min for the full suite; capture and report any failure verbatim
   (failure ⇒ fix or revert the flip; do NOT weaken a test).
4. Update `docs/2026-06-12-deterministic-contraction-engine-design.md`: M2
   row — add "engine default ON (serial gates green; Sofia gate pending
   Task 12)".

**Commit:** `feat: chain engine ON by default (M2 serial gates green)`

---

### Task 12: Sofia perf gate (Part 8) — production-path A/B

**Files:** create `examples/MPI_parallel/bench_chain_gate_m2_sofia.jl` +
`examples/MPI_parallel/Sofia/submit_bench_chain_gate_m2.sh` (clone the Part-7
pair: `bench_chain_gate_sofia.jl` / `submit_bench_chain_gate.sh`; 1 GPU; bump
`--time` from the template's 00:30:00 to 00:45:00 — six maps now).

Driver design (the toggle IS the A/B switch — both paths run the literal
production code):

- Maps: FLmap, FRmap, ACmap, ACdmap — leg5 single-M production form, via
  `*_parallel(...; ifparallel=false, forloop_iter=n)`. PLUS the two cheap
  unguarded production paths the toggle also flips: **Cmap** leg4 (direct
  call, fwd + Zygote-pullback bwd, n/a chunking) and **Mumap** fwd-only via
  `Mumap_parallel(...; ifparallel=false, forloop_iter=n)` (preconditioner
  path). FLmap_C3v and the corner maps stay unbenched — justified in
  out-of-scope.
- Cells: (D,χ) ∈ {(10,512), (12,1024), (16,1024)}; per-path chunk counts: the
  Part-6 nA formula for the engine path, nB for the Zygote path (read them
  from `bench_kernel_ab_sofia.jl`).
- fwd: time `*_parallel` with `set_chain_engine!(true)` vs `false`.
  bwd: time `Zygote.pullback` + pullback-call over the same expression, toggle
  on (engine-rerouted rrule) vs off (Zygote per-slice). Same timeit harness +
  mem probe as Part 7 (per-rep GC outside the window).
- Parity columns: fwd 1e-12 / grad 1e-10 per cell (toggle on vs off).
- Gate, printed by the driver per map × cell × direction:
  `CHAIN/TENSOR ≤ 1.05` time and `≤ 1.10` mem — and flag any cell where
  CHAIN is SLOWER than TENSOR (per the design doc: "any map slower than its
  @tensor original is a bug"). FLmap-leg5 expectation from Part 7: CHAIN
  clearly faster on bwd at production cells.

Submit via the established Sofia flow (sync clone → sbatch → monitor at the
30-min RUNNING cadence). Record the table + PASS/FAIL as **Part 8** in
`examples/MPI_parallel/benchmarks/Sofia_VUB_H200.md`; update the design doc M2
row to "done" on PASS. On FAIL: record, leave the toggle ON only if the
failing cells are noise-level (Part-7 precedent: small-cell bwd 1.074 was
accepted as variance); a real regression ⇒ revert Task 11's flip in a new
commit and open the per-map layout-pinning diagnosis as the follow-up — do
NOT improvise alternative layouts inline.

**Commits:** `feat: M2 chain-gate driver (toggle A/B, 4 maps)` then
`docs: Part 8 — M2 chain engine perf gate` after the run.

---

## Done criteria for M2

- All 21 map methods have chain declarations with @tensor-pinned layouts;
  every one parity-gated 1e-12/1e-10 in `test/test_chain_maps.jl`.
- `rrule(forloop)`/`rrule(parallel)` use `engine_backward` for
  FLmap/FRmap/ACmap/ACdmap (all M forms); Zygote remains only as the
  unregistered-`f` fallback.
- Single-M maps run conj-free (engine flags) in forward and backward.
- `CHAIN_ENGINE[] == true` by default; full serial suite + both 4-rank MPI
  suites green; Sofia Part 8 recorded.
- Old `@tensor` bodies intact behind the guard (retirement is NOT M2).

## Out of scope (explicit)

- rrules for `forloop_sum`/`parallel_sum` (none exist today; Mmap/Mumap/Mdmap
  remain forward-only in production).
- Cannon distributed wrappers for FRmap/ACmap/ACdmap/Cmap and the ACmap
  cross-axis dataflow design — M3.
- Chains for ALCtoAC_map/CTtoT/CTCtoT/Lmap/Rmap (not in the 21; trivial
  1-2-link contractions).
- Sofia perf rows for FLmap_C3v and the corner maps. Justification: the
  corner maps' only caller is commented out (dead code — no production
  traffic to regress); FLmap_C3v runs only in the qrctmrg path, which has no
  Sofia production benchmark today — its layouts are pinned by the same
  mechanism the gate validates on six other maps, and a dedicated qrctmrg
  campaign is the right place to measure it. This consciously waives the
  design doc's per-map perf guard for those two; flag it in the Part-8 notes.
- Per-map intermediate-layout TUNING beyond @tensor pinning (only if Part 8
  flags a map).
- Eager-freeing the downcast copies in the `_chain_map` cast path (GC'd, as
  today).
