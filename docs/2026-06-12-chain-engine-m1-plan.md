# Chain Engine M1 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (or
> subagent-driven-development) to implement this plan task-by-task.

**Goal:** The contraction-chain engine core (declarative `Chain`, runtime
forward executor with eager frees, generic pairwise adjoints, recompute-style
backward walk) plus the FLmap-leg5 chain instance, parity-gated against the
proven hand kernels and perf-gated on Sofia.

**Architecture:** A `Chain` declares a left-assoc pairwise contraction
sequence as index-label tuples; the executor calls TensorOperations' runtime
`tensorcontract` (labels form, verified present in v5.5.1: `tensorcontract(IC,
A, IA, conjA, B, IB, conjB)`) so cuTENSOR sees the same contractions as
`@tensor`; every intermediate is owned and `TeneT._free!`d after its last
consumer. The backward recomputes intermediates and walks generic link
adjoints (`dA = dC·conj(B)`, `dB = conj(A)·dC`) in the cannon-v3-proven
ordering. M1 scope cuts: no per-op conj flags (tuple-M only; single-M stays a
map-level concern), no production rerouting (hand kernels stay; switch-over is
M2 after the perf gate).

**Tech Stack:** Julia, TensorOperations 5.5.1 runtime API, CUDA.jl
(`unsafe_free!` via existing `TeneT._free!`), existing cannon kernels as the
parity reference. Branch: `claude/sad-saha-3ec6bf`.

**Context for implementers:** Hand-kernel reference in
`src/contraction/cannon_2d.jl`: `_cannon_stage1` (H=FL·ALd), `_cannon_fold1`
(T=H·M1), `_cannon_fold2` (G=T·M2), `_cannon_stage2` (P=G·ALu),
`_cannon_stage1_add!`, and 8 adjoints `_cannon_stage1_dFL/_dALd`,
`_cannon_fold1_dH/_dM1`, `_cannon_fold2_dT/_dM2`, `_cannon_stage2_dG/_dALu`.
Serial reference `FLmap(FL, ALu, ALd, M1, M2)` in `src/contraction/basic.jl:68`.
Tests for the engine are LOCAL (no MPI): plain
`julia --project=. test/test_chain_engine.jl`. Use the Bash tool for julia
commands (PowerShell mangles quotes). New testfile is standalone (not wired
into runtests.jl), mirroring the test_cannon.jl convention.

---

### Task 1: Chain type + label analysis

**Files:** Create `src/contraction/chain_engine.jl`; modify `src/TeneT.jl`
(include after `contraction/cannon_2d.jl`); create `test/test_chain_engine.jl`.

Test first:

```julia
# test/test_chain_engine.jl — run: julia --project=. test/test_chain_engine.jl
using Test, LinearAlgebra, Random, Zygote
using TeneT
using TeneT: Chain, chain_interlabels, FLMAP_LEG5_CHAIN

@testset "chain label analysis" begin
    ch = Chain(((:a,:e,:f,:i), (:i,:j,:k,:l), (:e,:j,:g,:b,:p), (:f,:k,:h,:c,:p), (:a,:b,:c,:d)),
               (:d,:g,:h,:l))
    ils = chain_interlabels(ch)
    @test ils[1] == (:a,:e,:f,:j,:k,:l)        # H
    @test ils[2] == (:a,:f,:k,:l,:g,:b,:p)     # T  (I_{k-1}-minus-shared, then op-minus-shared)
    @test ils[3] == (:a,:l,:g,:b,:h,:c)        # G
    @test FLMAP_LEG5_CHAIN.out == (:d,:g,:h,:l)
end
println("test_chain_engine done")
```

Implementation:

```julia
# src/contraction/chain_engine.jl
# Deterministic-memory contraction-chain engine (M1).
# Design: docs/2026-06-12-deterministic-contraction-engine-design.md
#
# A Chain declares a left-assoc pairwise sequence as index-label tuples; the
# executor runs TensorOperations' runtime tensorcontract (same cuTENSOR calls
# as @tensor) with every intermediate owned and _free!'d after its last use.
# M1 scope: no per-op conj flags (tuple-M maps only).

struct Chain{N, O}
    ops::NTuple{N, Tuple}    # index labels per operand; ops[1] is the carried tensor
    out::NTuple{O, Symbol}   # output labels of the final intermediate
end

# Labels of intermediate I_k = I_{k-1} ⋆ ops[k+1]: open labels of I_{k-1}
# (not shared with the op) followed by open labels of the op.
function _link_labels(labs::Tuple, op::Tuple)
    keep = filter(l -> !(l in op), labs)
    new  = filter(l -> !(l in labs), op)
    return (keep..., new...)
end

function chain_interlabels(ch::Chain{N}) where {N}
    ils = Vector{Tuple}(undef, N - 1)
    labs = ch.ops[1]
    for k in 2:N
        labs = k == N ? ch.out : _link_labels(labs, ch.ops[k])
        ils[k - 1] = labs
    end
    return Tuple(ils)
end

# FLmap leg5 (tuple-M), operand order (FL, ALd, M1, M2, ALu) — the current
# left-assoc order of the serial kernel and the cannon stage pipeline.
const FLMAP_LEG5_CHAIN = Chain(
    ((:a,:e,:f,:i), (:i,:j,:k,:l), (:e,:j,:g,:b,:p), (:f,:k,:h,:c,:p), (:a,:b,:c,:d)),
    (:d,:g,:h,:l))
```

NOTE for the implementer: verify `_link_labels`'s derived orders against the
HAND KERNEL index orders (H[a,e,f,j,k,l] ✓, G mismatch alert: the hand
`_cannon_fold2` emits G[a,b,c,g,h,l] but left-assoc concat gives
(:a,:l,:g,:b,:h,:c)) — that is EXPECTED and fine: the engine's intermediate
layouts may differ from the hand kernels'; only the FINAL output (`out`) and
the numerical values must match. The Task-2 parity test therefore compares
values up to the engine's own layout (final output exact; intermediates via
permutedims in the test only).

Commit: `feat: Chain declaration + label analysis (engine M1)`

### Task 2: forward executor + parity

Append test:

```julia
@testset "chain_apply == FLmap == hand pipeline" begin
    Random.seed!(11)
    χ, D, d = 12, 3, 2
    FL  = rand(ComplexF64, χ, D, D, χ)
    ALu = rand(ComplexF64, χ, D, D, χ)
    ALd = rand(ComplexF64, χ, D, D, χ)
    M1  = rand(ComplexF64, D, D, D, D, d)
    M2  = rand(ComplexF64, D, D, D, D, d)
    ref = TeneT.FLmap(FL, ALu, ALd, M1, M2)
    out = TeneT.chain_apply(FLMAP_LEG5_CHAIN, (FL, ALd, M1, M2, ALu))
    @test out ≈ ref rtol = 1e-12
end
```

Implementation (append):

```julia
"""
    chain_apply(ch, tensors) -> result

Run the chain left-assoc via runtime `tensorcontract`; owned intermediates
are `_free!`d as soon as consumed. `tensors` order matches `ch.ops`.
"""
function chain_apply(ch::Chain{N}, tensors::NTuple{N, Any}) where {N}
    acc  = tensors[1]
    labs = ch.ops[1]
    for k in 2:N
        IC  = k == N ? ch.out : _link_labels(labs, ch.ops[k])
        nxt = tensorcontract(IC, acc, labs, false, tensors[k], ch.ops[k], false)
        k > 2 && _free!(acc)     # acc owned from link k-1; inputs never freed
        acc, labs = nxt, IC
    end
    return acc
end
```

(`tensorcontract` is exported by TensorOperations, already a TeneT dep; if
name resolution fails inside the module add
`using TensorOperations: tensorcontract` near the top of the file.)

Commit: `feat: chain forward executor with eager frees`

### Task 3: accumulating first link (ring building block)

Test: zero-init + two i-block adds == full first link, mirroring the
`_cannon_stage1_add!` testset (reuse the same slicing pattern with the chain:
`chain_link1_add!(H, FLMAP_LEG5_CHAIN, FL[:, :, :, 1:6], ALd[1:6, :, :, :])`
twice over complementary i-ranges ≈ `TeneT._cannon_stage1(FL, ALd)` rtol
1e-12 — note the engine's H layout equals the hand kernel's here:
(:a,:e,:f,:j,:k,:l) ✓ direct comparison valid).

Implementation: probe `methods(TensorOperations.tensorcontract!)` and use the
labels-or-Index2Tuple mutating form to add α=1, β=1 into a caller buffer:

```julia
# H labels = _link_labels(ops[1], ops[2]); β=1 accumulation for the cannon ring.
function chain_link1_add!(H, ch::Chain, A, B)
    IH = _link_labels(ch.ops[1], ch.ops[2])
    # Index2Tuple form (verified in TensorOperations 5.5.1):
    #   tensorcontract!(C, A, pA, conjA, B, pB, conjB, α, β)
    # pA = (open-positions, contracted-positions) in A; pB = (contracted, open);
    # pAB = trivial permutation of (openA..., openB...) onto IH.
    ...derive integer tuples from labels (helper _index2tuples(labsA, labsB, IC))...
    return H
end
```

The implementer derives `_index2tuples` (pure label bookkeeping, ~15 lines)
and MUST validate it against three hand cases in the test (link1 of FLmap; a
transposed toy case; a case with non-contiguous contracted positions). If the
Index2Tuple signature differs in the installed version, adapt from
`methods(...)` output — the test is the contract.

Commit: `feat: chain_link1_add! accumulating first link`

### Task 4: generic backward walk + gradient parity

Test (CPU): compare `chain_backward(FLMAP_LEG5_CHAIN, tensors, dOut)` against
(a) Zygote pullback of `FLmap` w.r.t. all five operands (map operand order!),
(b) the assembled hand-adjoint chain from cannon (`_cannon_stage2_dG` → … →
`_cannon_stage1_dFL/_dALd`), both rtol 1e-10, complex tensors, including a
non-trivial dOut (`rand(ComplexF64, size(out))`).

Implementation (append):

```julia
"""
    chain_backward(ch, tensors, dOut) -> NTuple{N} of gradients

Recompute-style: rebuilds the intermediates, then walks the reversed chain
with the generic pairwise adjoints (dA = dC·conj(B), dB = conj(A)·dC),
freeing every owned array after its last consumer.
"""
function chain_backward(ch::Chain{N}, tensors::NTuple{N, Any}, dOut) where {N}
    # 1. forward recompute, KEEPING intermediates (they are adjoint operands)
    inters = Vector{Any}(undef, N - 1)      # inters[k] = I_k; I_1 = tensors[1]⋆tensors[2]
    labs   = Vector{Tuple}(undef, N - 1)
    acc, l = tensors[1], ch.ops[1]
    for k in 2:N
        IC = k == N ? ch.out : _link_labels(l, ch.ops[k])
        acc = tensorcontract(IC, acc, l, false, tensors[k], ch.ops[k], false)
        inters[k - 1], labs[k - 1] = acc, IC
        l = IC
    end
    _free!(inters[N - 1])                    # final output not needed in backward
    # 2. reversed walk
    grads = Vector{Any}(undef, N)
    dC, lC = dOut, ch.out
    for k in N:-1:2
        Iprev  = k == 2 ? tensors[1]  : inters[k - 2]
        lIprev = k == 2 ? ch.ops[1]   : labs[k - 2]
        # d(op_k) = conj(Iprev) ⋆ dC
        grads[k] = tensorcontract(ch.ops[k], Iprev, lIprev, true, dC, lC, false)
        # dIprev = dC ⋆ conj(op_k)
        dprev = tensorcontract(lIprev, dC, lC, false, tensors[k], ch.ops[k], true)
        k < N && _free!(dC)                  # owned cotangent from the previous step
        k > 2 && _free!(inters[k - 2] === Iprev ? Iprev : nothing)  # see note
        dC, lC = dprev, lIprev
    end
    grads[1] = dC
    return Tuple(grads)
end
```

FREE-ORDERING NOTE (the implementer must get this exactly right and the test
pins it only indirectly — review carefully): `Iprev = inters[k-2]` has its
last use in iteration k (both adjoint contractions); free it at the END of
iteration k, never `tensors[1]` (caller-owned) and never before `dprev` is
computed. The sketch above is intentionally not final — restructure the loop
body so the frees are unambiguous, e.g. compute both contractions, then
`k > 2 && _free!(Iprev)`, `k < N && _free!(dC_old)`.

Commit: `feat: generic chain backward (recompute + eager frees)`

### Task 5: cannon-local-pipeline equivalence (chunked)

Test: replicate the v3 local pipeline with the engine — for the 2×2-rank
local workload (`FL_row[χ/2,D,D,χ]`, `ALd_col[χ,D,D,χ/2]`, `ALu_row`,
chunks n ∈ {1,3} over the local l range):

- forward: per chunk `chain_link1_add!` over i-blocks + tail links == the
  hand staged pipeline output (use `staged`-style reference assembled from
  cannon kernels), rtol 1e-12;
- backward: chain_backward per chunk, gradients accumulated chunk-wise ==
  hand-adjoint chunk loop, rtol 1e-10.

No new src code expected — this testset proves the engine API is sufficient
for the M2 rerouting of `_cannon_forward_sliced` and the cannon rrule. If an
API gap appears (e.g. backward needs to start from a PROVIDED H instead of
recomputing link 1 per chunk), extend the engine minimally
(`chain_backward(...; from1 = H)`) and note it in the report.

Commit: `test: chain engine reproduces the cannon local pipeline`

### Task 6: perf gate on Sofia

**Files:** Create `examples/MPI_parallel/bench_chain_gate_sofia.jl` +
`examples/MPI_parallel/Sofia/submit_bench_chain_gate.sh` (clone the
`submit_bench_kernel_ab.sh` env, 1 GPU, 30 min).

Driver: the kernel-A/B workload at cells (10,512), (12,1024), (16,1024) with
the per-path n of Part 6 (nA formula); three paths × fwd/bwd, same timeit
(per-rep GC outside window) and mem probe as Part 6:
- HAND: the cannon staged/hand-adjoint functions (Part 6's path A — copy
  those helper functions from `bench_kernel_ab_sofia.jl`)
- CHAIN: `chain_apply`/`chain_link1_add!`-chunked forward and
  `chain_backward`-chunked backward (Task 5's pipeline)
- TENSOR: monolithic `@tensor` FLmap via forloop (Part 6's path B) for
  continuity.

Gate, printed by the driver: `CHAIN/HAND ratio` per cell per direction; PASS
iff every ratio ≤ 1.05 and mem within 10%. Parity columns as usual.

Submit via the established Sofia flow (sync clone → sbatch → 30-min cron
watch). Record the table + PASS/FAIL in
`benchmarks/Sofia_VUB_H200.md` Part 7. FAIL ⇒ stop after recording; the
`@generated`-fallback decision returns to the design round (do NOT improvise
it inline).

Commit: `feat: chain-engine perf gate driver` then
`docs: Part 7 — chain engine perf gate` after the run.

---

## Done criteria for M1

All five test groups green locally; Sofia gate recorded; no production code
rerouted (hand kernels and all existing paths untouched and green —
`julia --project=. test/run_test_cannon.jl` must still pass at the end).
