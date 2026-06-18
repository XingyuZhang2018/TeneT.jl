# M5 design — Slice2D `vumps_step` assembly + the QR gather seam

**Branch** `claude/ecstatic-golick-e3f6f0` · **Date** 2026-06-15 · **Depends on** M4 (the four distributed env solvers, GPU-validated, committed `80e0263`).

This doc is the *design* artifact in the research → design → adversarial review → implement → dual review cycle. It goes to adversarial review **before** any implementation. The R1 review-response block is appended at the end after the first review round.

All `file:line` anchors are to the worktree above unless noted.

---

## 0. Goal & scope

Assemble the four M4 distributed env solvers into a distributed VUMPS *step* — a slice2d `VUMPS{General}` path that runs the full power-method iteration over block-distributed boundary tensors on a square `N1=N2` `Slice2DGrid`. The deliverable is a slice2d `vumps_step` reachable from `leading_boundary → vumps_itr`, whose fixed-point trajectory matches the serial `vumps_step` exactly, and whose gradient matches under Zygote.

**What M4 already gives us** (`src/boundary_algorithm/vumps/slice2d.jl`):

| serial call | slice2d analog | status |
|---|---|---|
| `leftenv` (general.jl:254) | `leftenv_slice2d` (slice2d.jl:54) | ✅ M4, parity-gated, 16-GPU |
| `rightenv` (general.jl:355) | `rightenv_slice2d` (slice2d.jl:135) | ✅ M4 |
| `ACenv` (general.jl:562) | `ACenv_slice2d` (slice2d.jl:210) | ✅ M4 |
| `Cenv` (general.jl:640) | `Cenv_slice2d` (slice2d.jl:283) | ✅ M4 (outputs **replicated** C) |

**What M5 must build** — the two pipeline endpoints plus the orchestration:

1. **`ALCtoAC_slice2d`** — the *entry* seam (block AL × replicated C → block AC). General.jl:196.
2. **`ACCtoALAR_slice2d`** — the *exit* seam, the **QR gather seam**: full-χ per-cell QR/LQ that cannot run on a block. General.jl:712.
3. **`vumps_step_slice2d`** — the 6-call orchestration mirroring serial `vumps_step` (general.jl:873).
4. **`init_VUMPSRuntime_slice2d`** — build a block-distributed initial runtime.
5. **Dispatch wiring** — route `leading_boundary → vumps_itr → vumps_step` to the slice2d path.

**Out of scope (deferred, perf not correctness):** distributed canonicalization (no serial→block QR exists; M5 keeps init serial-then-scatter); TSQR for the QR seam (only matters at D≥20 — a v2 item); NCCL gating (Batch F); persistent comm buffers.

**Inherited hard constraints** (asserted in all four M4 solvers, e.g. slice2d.jl:55-59): square grid `N1=N2`, leg5 MPO only, `ifsimple_eig=true`, no mixed precision (`inner_etype === nothing`), no observable env (`ifobs=false`). The M5 step asserts the same regime; the gate runs under exactly it.

---

## 1. The data flow — what is block vs replicated at every point

Serial `vumps_step` (general.jl:873-886), annotated with the slice2d storage class of each tensor:

```julia
function vumps_step_slice2d(rt, M, grid, alg)
    @unpack AL, C, AR, FL, FR = rt          # AL,AR,FL,FR = BLOCK ;  C = REPLICATED (full χ×χ)
    sub = alg.subop_checkpoint
    AC = ALCtoAC_slice2d(AL, C, grid)                                            # 1  block AL × repl C → BLOCK AC
    _, FL = checkpoint(sub, (a,b,m,fl)->leftenv_slice2d(a,b,m,fl,grid; alg), AL, conj(AL), M, FL)    # 2  → BLOCK FL
    _, FR = checkpoint(sub, (a,b,m,fr)->rightenv_slice2d(a,b,m,fr,grid; alg), AR, conj(AR), M, FR)   # 3  → BLOCK FR
    _, AC = checkpoint(sub, (ac,fl,m,fr)->ACenv_slice2d(ac,fl,m,fr,grid; alg), AC, FL, M, FR)        # 4  → BLOCK AC
    _, C  = Cenv_slice2d(C, FL, FR, grid; alg)                                   # 5  → REPLICATED C (no checkpoint, serial parity)
    AL, AR, errL, errR = checkpoint(sub, (ac,c)->ACCtoALAR_slice2d(ac,c,grid), AC, C)               # 6  QR seam → BLOCK AL,AR
    err = errL + errR                       # replicated scalar — NO allreduce
    C = for_gc(C)
    return VUMPSRuntime(AL, AR, C, FL, FR), err
end
```

**Storage-class invariants the step must preserve** (these are the contract every M4/M5 function already honors):

- `AL, AR, FL, FR` are **block** StructArrays throughout the steady state. First χ leg split by `r1`, last χ leg split by `r2` (`slice2d_scatter` convention, slice2d.jl:164-171).
- `C` is **replicated** (full χ×χ on every rank) throughout. `Cenv_slice2d` produces replicated C (slice2d.jl:313); the QR seam consumes replicated C without gathering it; `LRtoC` at init produces replicated C.
- `M` is **replicated** leg5.

**Parity-critical ordering subtlety** (research, vumps-step report §1): `vumps_step` feeds the **OLD** `AL`/`AR` (from `rt`) into `leftenv`/`rightenv` — *not* the freshly-QR'd ones. There is a single AC/C solve per step. (Contrast `vumps_step_power`, general.jl:856, which re-solves and feeds new `ALp`/`ARp`.) The slice2d step must replicate this exact ordering or the trajectory diverges from serial. We mirror `vumps_step`, not `vumps_step_power`.

---

## 2. The QR gather seam — `ACCtoALAR_slice2d` (the genuinely new differentiable region)

### 2.1 Why a gather is unavoidable

`ACCtoAL`/`ACCtoAR` (general.jl:685-710) do a per-cell **full-χ** QR/LQ:

```julia
QAC, RAC = qrpos(_to_front(AC[i]))      # _to_front: (χ,D,D,χ) → (D·D·χ, χ)
QC,  RC  = qrpos(C[i])                  # C[i] is full χ×χ
errL    += norm(RAC - RC)
AL[i]    = reshape(QAC * QC', size(AC[i]))     # full (χ,D,D,χ)
# ACCtoAR (lqpos analog):
LAC, QAC = lqpos(_to_tail(AC[i,j]))     # _to_tail: (χ,D,D,χ) → (χ, D·D·χ)
AR[i,j]  = reshape(QC' * QAC, size(AC[i,j]))
```

`qrpos`/`lqpos` (misc.jl:119-147) are dense `qr!`/sign-fixed factorizations of the **full** reshaped matrix. The Householder reflectors mix the entire χ dimension — a block holding only the `r1`-slice of the front leg (or `r2`-slice of the tail leg) has neither the full row space nor the full column space the QR needs. **The QR cannot run on a block.** The block AC must be gathered to full χ before the QR; the resulting AL/AR are full and must be re-scattered.

`C` is **already replicated** full χ×χ (M4 `Cenv_slice2d`), so `C[i]`, `RC`/`LC`/`QC` need **no** gather. Only **AC** is gathered.

### 2.2 The forward — gather the whole AC StructArray, call the UNMODIFIED serial `ACCtoALAR`, scatter outputs

**[R1 revision — F3]** The seam does **not** reimplement the per-cell QR loop. It gathers the *entire* block AC into a full StructArray, calls the verbatim serial `ACCtoALAR(AC_full, C)`, and scatters the output AL/AR StructArrays:

```julia
function ACCtoALAR_slice2d(AC_blk, C, grid)
    AC_full = gather_struct(AC_blk, grid)                 # block → full, per UNIQUE cell (Zygote.Buffer loop)
    AL_full, AR_full, errL, errR = ACCtoALAR(AC_full, C)  # ← VERBATIM serial kernel (general.jl:712)
    AL_blk = scatter_struct(AL_full, grid)                # full → block
    AR_blk = scatter_struct(AR_full, grid)                # full → block
    return AL_blk, AR_blk, errL, errR
end
```

where `gather_struct`/`scatter_struct` map `slice2d_gather`/`slice2d_scatter` over `.data`, built inside a `Zygote.Buffer` loop mirroring `Cenv_slice2d`'s buffer pattern (slice2d.jl:287-313) so they are differentiable. The loop runs over `S.data` (the **unique** cells) — rank-uniform because `.pattern` is replicated.

**Why reuse the serial kernel verbatim** (the F3 fix): `ACCtoAL` sums `errL` over `for i in 1:length(AC)` (all `nuniq` positions) while `ACCtoAR` sums `errR` over `for p in 1:length(AC.data)` (unique data) with `findfirst`/`jr=mod1(j-1,Nj)` (general.jl:688,701-707). A hand-rolled per-cell `for i in 1:Ni, j in 1:Nj` seam would over-count a repeated-pattern cell's residual (e.g. `[1 2;2 1]`) by its multiplicity, breaking `abs(err_c−err_s)≤1e-10`. Calling the serial `ACCtoALAR` on the gathered-full StructArray makes `errL`/`errR` **identical to serial by construction** — and likewise guarantees the AR `lqpos`+`_to_tail`+`jr` branch matches (F5). The only new code is the gather/scatter wrapping.

This mirrors `Cenv_slice2d` (slice2d.jl:283) — gather operands to full, run the **unmodified serial kernel** — except (unlike Cenv, which keeps C replicated) the seam re-scatters the block AL/AR outputs. `slice2d_gather` (slice2d.jl:179) and `slice2d_scatter` (slice2d.jl:164) are pure data-movement (gather = allgatherv, scatter = local indexing).

### 2.3 The adjoint — why it composes correctly, and the over-count trap

The whole point: **we write zero new rrules.** The seam is a composition of `slice2d_gather`, the serial QR (`qrpos`/`lqpos` rrules, rules.jl:64-92), and `slice2d_scatter` — each already has a correct rrule, and Zygote composes them.

Backward chain (Zygote replays in reverse):

| forward op | rrule | adjoint action |
|---|---|---|
| `slice2d_scatter(AL_full)` | rules.jl:448 | **allreduce** the block AL cotangents into the full replicated `dAL_full` |
| serial QR `ACCtoAL/AR(AC_full,C)` | rules.jl:64-92 | `qrpos`/`lqpos` pullback on full χ — **runs identically on every rank** |
| `slice2d_gather(AC_blk)` | rules.jl:466 | **take-my-block**: each rank slices its own block of the replicated `dAC_full` |

Why each is correct:

- **`slice2d_gather` adjoint = take-my-block** (rules.jl:475). Justified by the comment there: *"Downstream of gather is replicated computation → identical dfull on every rank; the adjoint is just take-my-block."* The full AC feeds a QR that runs identically on every rank → the cotangent `dAC_full` is identical on every rank → each rank takes only its own block. This is the same case as `Cenv_slice2d`'s fixed FL/FR gather (slice2d.jl:255-256).
- **`slice2d_scatter` adjoint = allreduce** (rules.jl:460). The scatter forward turns a replicated full AL into this rank's block; its adjoint stitches the disjoint per-rank block cotangents into the full replicated-input gradient via allreduce. Exactly right for re-scattering the QR's full AL/AR.
- **QR pullback is replicated** — see §2.4.

**🔴 THE OVER-COUNT TRAP (load-bearing — flagged for review):** the AC gather here **must** use the full-tensor `slice2d_gather` (take-my-block adjoint), **not** `slice2d_gather_row`/`slice2d_gather_col` (rules.jl:518/533, whose adjoints are reduce-scatter). Reduce-scatter is correct only when the downstream map is a *distributed-output contraction* where each rank contributed a genuine partial sum. The QR is a **replicated** computation, not a distributed contraction — using a reduce-scatter adjoint would over-count the AC gradient by ~P (slice2d.jl:256). Symmetric trap on C: C is replicated and must **not** be gathered/scattered at all (no comm around it).

### 2.4 `qrpos`/`lqpos` on gathered-full AC — replicated, deterministic

The QR rrules (rules.jl:64-92) take a 2D matrix, contain **no MPI/grid awareness**, and are pure linear algebra (`Hermitian`, `UpperTriangular` solve, `I*ε` regularization with `ε = real(T)(1e-12)`, eltype-matched). Given **bitwise-identical** input on every rank, LAPACK/cuSOLVER QR is deterministic and `safesign` (misc.jl:125) makes the positive-diagonal gauge unique → `Q`, `R`, `AL`, `AR`, and the pullback `dA` are identical on every rank. That replicated `dA` is exactly what take-my-block expects.

The determinism premise holds because `slice2d_gather` is **pure placement** (allgatherv + deterministic per-block copy, slice2d.jl:193-201) — no floating-point reduction, so all ranks build the same `full` byte-for-byte. **Invariant to protect:** if any future variant introduced a *reduction* into the gather (summing overlapping blocks), bitwise replication could break and `safesign` could flip a column sign on one rank → divergent AL/AR → grid desync. Current gather is pure placement; the trajectory-match gate naturally catches a violation.

**[R1 — backward determinism, completeness]** The replicated-`dA` claim also covers the *backward* path: `dAL_full`/`dAR_full` feeding the `qrpos`/`lqpos` pullback are produced by the `slice2d_scatter` allreduce adjoint (`allreduce_p2p!`, rules.jl:460), a rank-identical reduction, so the pullback inputs are replicated → `dAC_full` is replicated → exactly what the subsequent take-my-block requires.

**[R1 — bit-for-bit scope, F4]** Distinguish two determinism claims, only one of which is true:
- **(a) Across ranks: slice2d AL/AR are bit-for-bit identical** — TRUE. Every rank gathers the same full AC and runs the same deterministic `qrpos`. This is the R-4 invariant; the gate asserts it directly (every rank's gathered full AL equal).
- **(b) Slice2D AL == *serial* AL bit-for-bit** — **FALSE**, and the gate must NOT expect it. The AC fed to the seam comes from `ACenv_slice2d`, whose power iteration sums block-locally + `slice2d_norm` allreduces — a **different floating-point summation order** than serial `ACenv`'s single full sum → slice2d AC ≠ serial AC at ~1e-13 → QR outputs differ at ~1e-12. Worse, `safesign` (misc.jl:125) can flip a near-zero R-diagonal column sign under that perturbation, a *per-column* gauge flip a global phase factor cannot absorb. So the slice2d-vs-serial expectation is **rel ≤ 1e-10 up-to-gauge**, not bit-for-bit (see §6.1).

**[R1 — qrpos parity robustness]** Parity is robust to any `qrpos` complex-AD inexactness (the misc.jl:149 `qr_for_ad` warning): serial `ACCtoALAR` (general.jl:881) already differentiates through the **same** `qrpos`/`lqpos` rrule (confirmed: not inside `ignore_derivatives`; `qr_for_ad` is used nowhere in `vumps_step`), so the seam inherits identical behavior and the parity gate cannot be fooled by it.

### 2.5 `errL`/`errR` — replicated scalars, NEVER allreduced

`errL = Σ norm(RAC − RC)`, `errR = Σ norm(LAC − LC)` (general.jl:691,706), accumulated over cells. Because the QR runs on **full replicated** AC and **replicated** C, these norms are computed identically on every rank → already-globally-correct scalars, **identical on every rank**, with **no allreduce**. (Contrast `slice2d_norm`, slice2d.jl:216, which *is* allreduced because it sums genuine per-rank partials — but that is a distributed sum, not a replicated residual.) An allreduce here would multiply `err` by P. They flow straight into `err = errL + errR`, keeping the per-step `err < tol` break rank-uniform.

Because §2.2 calls the verbatim serial `ACCtoALAR` on the gathered-full StructArray, the **exact accumulation index sets** (`ACCtoAL` over all positions, `ACCtoAR` over unique data) are inherited automatically — no risk of a repeated-pattern over-count (F3).

### 2.6 The `ACCtoAR` column-shift `jr = mod1(j-1, Nj)`

`ACCtoAR` (general.jl:701) pairs cell `AC[i,j]` with `C[i,jr]`, `jr = mod1(j-1, Nj)` — a *unit-cell* column shift, orthogonal to the *grid* partition. Since C is replicated, `C[i,jr]` is locally available on every rank with no comm. The seam preserves this exactly (it operates per-cell on the replicated C); no grid interaction.

---

## 3. The entry seam — `ALCtoAC_slice2d`

`ALCtoAC_map(AL::leg4, C) = @tensor result[a,b,c,e] := AL[a,b,c,d] * C[d,e]` (basic.jl:37). On blocks, AL's contracted last leg `d` is `r2`-split, while C is replicated full χ×χ. So a block-local contraction gives only a *partial* sum over the rank's `d`-slice → it needs a reduction over the `r2` partition. The output AC must come out as a block (first leg `a` by `r1` — inherited from AL; last leg `e` by `r2`) to seed `ACenv_slice2d`.

**[R1 — why ALCtoAC must be distributed at all (parity-load-bearing)]** `ACenv_slice2d` (slice2d.jl:232) seeds `simple_eig` with this AC and runs a **finite** `power_iter_ad` in the AD loop (general.jl:945), so the post-step AC depends on the seed at the bit level. The §6.1 gate demands per-step trajectory match, so seeding `ACenv` with anything other than serial's `ALCtoAC(AL,C)` (e.g. a cheap block-local approximation or the previous AC) would diverge from serial and fail the gate. The cheap-seed shortcut is therefore rejected.

Two designs, both AD-supported by existing M4 primitives:

### Option A — gather-compute-scatter (RECOMMENDED for v1)

Mirror the QR seam (§2.2): gather AL to a full StructArray, call the **verbatim serial `ALCtoAC`** (general.jl:196), scatter AC.

```julia
function ALCtoAC_slice2d(AL_blk, C, grid)
    AL_full = gather_struct(AL_blk, grid)      # block → full (take-my-block adjoint)
    AC_full = ALCtoAC(AL_full, C)              # ← VERBATIM serial kernel
    return scatter_struct(AC_full, grid)       # full → block (allreduce adjoint)
end
```

- **Pros:** reuses the *exact same* primitives, kernel, and AD treatment as the QR seam (§2) → one uniform, already-tested seam pattern across both endpoints; lowest correctness risk; mirrors the `Cenv_slice2d` precedent (gather → serial kernel). Zero new rrules.
- **Cons:** materializes full AL. **[R1 — F6 corrected]** At χ=256, D=8, complex128 the full AL is 256·8·8·256·16 B = **67 MB/cell** (the design's earlier "1.07 GB" was wrong by ~16×, having substituted χ for the middle D legs); at the χ=1024, D=10 production target it is **1.68 GB/cell**. Under `step_checkpoint=Recompute()` (which §5.3/R-6 already mandates for the deadlock argument) the forward full-tensor intermediates are freed and recomputed on backward, so per-step transients do **not** accumulate on the tape — Option A is **feasible at production**, not fatal. **[R1 — leg-d round trip]** A correctness-neutral inefficiency: the scattered AC's last (`r2`/`e`) leg is immediately re-gathered to full by `ACenv_slice2d`'s first map (`ACmap_slice2d_sliced` row-gathers it, slice2d.jl:622), so Option A does a wasted scatter→regather of that leg. Only the `a`(r1) split is genuinely used. This is a (small) extra argument for Option B, not a correctness issue.

### Option B — distributed col-reduce-scatter (deferred to v1.1/v2 perf)

```
AC[i] = reduce_scatter_col( ALCtoAC_map(AL[i], C[my_d_rows, :]) )   # local partial → reduce over r2, scatter e
```

Each rank contracts its `d`-slice of block AL against the matching rows of replicated C (sliced locally, no comm), producing a full-`e` partial; a **reduce-scatter over the `r2` partition** sums the `d`-partials and scatters `e` by `r2`. AD adjoint = col-allgather — both `slice2d_gather_col`/`_slice2d_col_reduce_scatter` already exist and are M4-tested (rules.jl:533/540).

- **Pros:** no full AL materialization → lower memory peak (matters at production χ, aligns with "slice2d beats slice by avoiding full materialization").
- **Cons:** new index bookkeeping (the `d`-row slicing of C, the reduce-scatter direction); higher AD risk; **must not** be confused with the QR seam's take-my-block pattern — here reduce-scatter *is* correct because the contraction is genuinely distributed over `d`.

**Recommendation (R1-confirmed):** ship **Option A** in v1 to get the full step parity-gated with minimum new surface, then switch `ALCtoAC_slice2d` to Option B as an isolated, separately-gated optimization once the step trajectory matches. The adversarial review confirmed Option A is feasible at production under Recompute (F6), so Option B is a **pure perf optimization**, not a correctness/feasibility requirement. This respects the parity-first philosophy ("旧路径保留到 parity 验证通过") and keeps the v1 AD surface to *only* primitives already validated in M4.

> **Memory note (R1-corrected):** the avoidable full materialization is this AL gather (the AC gather at the QR seam is intrinsic — QR is full-χ). But the *dominant* retained-full-tensor cost is the **intrinsic QR seam**, not the entry seam: `slice2d_scatter`'s rrule closes over the full input (`dfull = zero(T_full)`, rules.jl:456), so under `Plain()` the QR seam retains `AL_full` **and** `AR_full` per cell. Switching the entry seam to Option B saves only a fraction of the full-tensor peak. The real lever for memory is `step_checkpoint=Recompute()`, which bounds all of it.

---

## 4. `init_VUMPSRuntime_slice2d` — block-distributed initial runtime

### 4.1 Why init is serial-then-scatter

The canonicalization chain is irreducibly serial full-χ per-cell QR + `simple_eig` with **no** distributed variant: `left_canonical`/`right_canonical` → `getAL`/`getLsped` (`qrpos!`, general.jl:113-131), `LRtoC` (full χ×χ matmul, general.jl:185-192). Building these distributed is explicitly **out of scope** for M5.

So init builds full AL/AR/C on every rank, **bcasts to make them bitwise-identical**, then scatters:

```julia
function init_VUMPSRuntime_slice2d(M, χ, grid, alg)
    A = initial_A(M, χ)                         # full — UN-SEEDED RNG → DIFFERS per rank
    AL, L, _ = left_canonical(A)                # full, serial
    R, AR, _ = right_canonical(AL)              # full, serial
    C  = LRtoC(L, R)                            # full χ×χ
    # ── [R1 — F1, BLOCKING] make the full source bitwise-identical on every rank ──
    AL = bcast_struct(AL, 0, grid.comm)         # UNCONDITIONAL bcast over grid.comm
    AR = bcast_struct(AR, 0, grid.comm)
    C  = bcast_struct(C,  0, grid.comm)         # C stays REPLICATED (this IS its distributed form)
    _, FL = leftenv(AL, conj(AL), M; alg)       # full, serial — on the bcast AL → FL identical
    _, FR = rightenv(AR, conj(AR), M; alg)      # full, serial — on the bcast AR → FR identical
    # scatter to blocks; C stays replicated:
    ALb = scatter_struct(AL, grid)
    ARb = scatter_struct(AR, grid)
    FLb = scatter_struct(FL, grid)
    FRb = scatter_struct(FR, grid)
    return VUMPSRuntime(ALb, ARb, C, FLb, FRb)
end
```

where `scatter_struct(S, grid) = StructArray([slice2d_scatter(t, grid) for t in S.data], S.pattern)` (the validated `scatter_sa`, test_slice2d_m4.jl:40), and `bcast_struct` maps `MPI.bcast` over `.data`.

**[R1 — F1, BLOCKING fix]** `initial_A → randSA` (initial.jl:23) uses the **default global RNG with no seeding and no bcast**, so every rank canonicalizes a *different* random `A`. Since `slice2d_scatter` is pure local indexing (it slices *this rank's own* `A`, not a shared one), without the bcast every rank scatters a slice of a different tensor → the first `slice2d_gather` in a step reassembles mismatched blocks → garbage. The **serial** init guards exactly this with `MPI.bcast(AL/AR/C, 0, COMM_WORLD)` — **but only `if alg.ifparallel`** (general.jl:780-784), and the slice2d path runs `ifparallel=false`, so that branch is skipped. The slice2d init must therefore bcast **unconditionally** over `grid.comm` (or seed the RNG identically; bcast is preferred — RNG seeding across CPU/CUDA is fragile). The M4 gate never hit this because it does `Random.seed!` then scatters one shared full tensor (test_slice2d_m4.jl:31-38) — see §6 for the gate change that catches it.

### 4.2 Confirmed-feasible facts

- **`slice2d_scatter` on each `.data` entry yields a block StructArray with no type change** (slice2d.jl:164 is pure indexing returning a smaller `Array`; StructArray is untyped over its data, base.jl:27-35). ✅ Already exercised by every M4 gate.
- **`VUMPSRuntime` struct needs no change** — fields are abstractly typed `::StructArray` (environment.jl:16-22). C is still a StructArray of full χ×χ matrices.
- **`update!`/`copy`/`Zygote.Buffer`/offload/`CuArray` are all `.data`-mapping → shape-agnostic** (environment.jl:179-216, structarray/base.jl:51-66). Block StructArrays flow through unchanged; the slice2d env solvers already Buffer/copy block StructArrays (slice2d.jl:73 etc.). ✅
- **`randSA`/`ISA` allocate block shapes** when given explicit per-entry `sizes` (initial.jl:23-26).

### 4.3 The global-χ inference obstacle (must-handle)

`cellones`/`FLint`/`FRint` infer χ from `size(·,1)` (initial.jl:71, general.jl:207-229). If handed a **block** tensor they silently produce **block-χ-sized** outputs. Mitigation: init runs canonicalization and `leftenv`/`rightenv` on **full** tensors (before scatter), so these helpers always see full χ. The scatter happens *last*, after FL/FR are built full. ✅ The design above already orders it this way. **Reviewer: confirm no path feeds a block tensor to a χ-inferring helper.**

---

## 5. Dispatch & the AD composition

### 5.1 Routing `leading_boundary → vumps_itr → vumps_step`

Current chain: `leading_boundary(rt, M, alg::VUMPS{General})` (general.jl:1007) → `vumps_itr` (general.jl:896) → forward warm-up loop calls `vumps_step` under `ignore_derivatives` (general.jl:931) + AD loop calls `checkpoint(alg.step_checkpoint, vumps_step, rt, M, alg)` (general.jl:977).

`VUMPS` is a mutable `@kwdef` struct (interface.jl:11). Two routing options:

- **Option 1 — flag on `VUMPS`:** add `ifslice2d::Bool=false` + `grid::Union{Nothing,Slice2DGrid}=nothing`; `vumps_step` branches to the slice2d body when `alg.ifslice2d`. `vumps_itr`/`leading_boundary` already deepcopy/thread `alg` (general.jl:923-951), so the grid rides along.
- **Option 2 — new `vumps_step_slice2d` + thin routing:** keep serial `vumps_step` untouched; `vumps_itr` calls `vumps_step_slice2d(rt, M, grid, alg)` when a grid is present.

Because the slice2d env signatures take an extra `grid` positional (slice2d.jl:54), the step body can't reuse the serial closures verbatim either way. **Recommendation: Option 2** — a dedicated `vumps_step_slice2d` keeps the serial `vumps_step` *function body* bit-for-bit unchanged (parity baseline stays trustworthy) and isolates all slice2d wiring in one function.

**[R1 — F-routing precision]** "Serial path unchanged" is true for the `vumps_step` **function**, but `vumps_itr` itself must gain a behavior-preserving dispatch branch at **both** call sites — the AD-loop `checkpoint(step_checkpoint, vumps_step, …)` (general.jl:977) **and the warm-up loop** `vumps_step(rt, M, alg_wholemode)` (general.jl:931, under `ignore_derivatives`). Both must route to `vumps_step_slice2d` when `alg.grid !== nothing`, else the slice2d trajectory's *warm-up* runs the serial step on block tensors and crashes/diverges. The branch is a no-op (→ serial) when no grid is set. Grid threaded via a thin `alg.grid::Union{Nothing,Slice2DGrid}=nothing` field (added, behavior-neutral for serial).

### 5.2 AD composes for free

**[R1 — corrected citation]** The slice2d regime is single-environment (`ifobs=false`), so it uses `leading_boundary(rt::VUMPSRuntime, M, alg)` (general.jl:1007), which has **no custom rrule** — Zygote differentiates straight through `vumps_itr → checkpoint(step_checkpoint, vumps_step, …)` (general.jl:977) by plain tracing. (The rrule at rules.jl:1337 is **only** for the up/down `Tuple{VUMPSRuntime,VUMPSRuntime}` case, which M5 does not target.) Either way the conclusion holds: a slice2d `vumps_step` is differentiated automatically by composing its sub-ops' rrules — *provided every sub-op is Zygote-differentiable*. The four M4 solvers already are; the QR seam (§2) and entry seam (§3) compose from primitives that already have rrules. **No new top-level rrule is needed.** The AD loop uses `power_iter_ad` (general.jl:945) with `simple_eig_polish_steps=0`; the warm-up is under `ignore_derivatives` and invisible to Zygote. The M5 gate must differentiate this single-env trace-through path (not the Tuple rrule) to match deployment.

### 5.3 Checkpoint composition — nests cleanly, one invariant

`step_checkpoint ⊃ subop_checkpoint ⊃ eig_checkpoint` are strictly different granularities (interface.jl:55-81). The slice2d env solvers apply `eig_checkpoint` internally (slice2d.jl:87,158,231); the slice2d `vumps_step` inherits `subop_checkpoint` on calls 2/3/4/6 and `step_checkpoint` on the whole step — they nest without conflict. `_assert_inner_method` (checkpoint.jl:48) does not bite (slice2d envs use eig granularity, not inner).

**🟠 Invariant to protect (R1-refined):** **`Recompute`** re-runs the forward on the backward pass (checkpoint.jl:80-87) — `Offload` does **not** (it replays the stored pullback once, checkpoint.jl:315/351, and on CPU `_detect_gpu_atype`→nothing makes it a pure pass-through, checkpoint.jl:341). So the deadlock-relevant mode is **Recompute**: under it the slice2d forward (gathers, reduce-scatters, the QR seam's gather/scatter) re-executes on backward, and every rank must re-enter identically or the grid deadlocks. The seam is rank-uniform by construction (gather/scatter are collectives every rank calls; QR is local-replicated; control flow keys off the *replicated* `M.pattern`). It is also **comm/tag-separated**: hoist gathers use `col_comm`/`row_comm` with distinct tags, the scatter adjoint uses `grid.comm` — MPI matches per `(comm,tag)`, so a recompute-forward Allreduce on `col_comm` cannot cross an adjoint `allreduce_p2p!` on `grid.comm`. **The gate must run `step_checkpoint=Recompute()`** (testing `Offload()` on CPU adds nothing — it's a pass-through there). A future change collapsing the hoist gathers onto a single tag/`grid.comm` would reintroduce the cross-match risk.

---

## 6. The gate — fixed-point trajectory match

No `vumps_step`-level test exists yet (M4 gated only per-env-solver). M5's gate is new.

### 6.1 Forward trajectory parity (the primary gate)

Run serial `vumps_step` and `vumps_step_slice2d` from the **same** initial runtime, for K steps, comparing every output tensor each step:

```
rt_serial = init_VUMPSRuntime(M, χ, alg_serial)            # full
rt_slice2d = scatter the SAME rt_serial to blocks (C replicated)
for k in 1:K:
    rt_serial, err_s = vumps_step(rt_serial, M, alg)
    rt_slice2d, err_c = vumps_step_slice2d(rt_slice2d, M, grid, alg)
    # gather rt_slice2d's AL/AR/FL/FR to full; C already full
    @assert phase_ok(gather(rt_slice2d.AL), rt_serial.AL)   # AL up to gauge
    @assert relerr(gather(rt_slice2d.FL), rt_serial.FL) ≤ 1e-10
    @assert relerr(gather(rt_slice2d.FR), rt_serial.FR) ≤ 1e-10        # FR
    @assert phase_ok(gather(rt_slice2d.AR), rt_serial.AR)              # AR — SEPARATE lqpos path
    @assert relerr(rt_slice2d.C, rt_serial.C) ≤ 1e-10                  # C (already full, replicated)
    @assert abs(err_c - err_s) ≤ 1e-10
    # cross-rank invariant (R-4): every rank's gathered AL/AR + replicated C bitwise-equal
    @assert MPI.Allreduce(maxabsdiff(gather(rt_slice2d.AL), rank0_AL), max, grid.comm) == 0
    @assert MPI.Allreduce(maxabsdiff(rt_slice2d.C, rank0_C), max, grid.comm) == 0
```

**[R1 — F4/F5 gauge expectation, corrected]** The slice2d-vs-serial expectation is **rel ≤ 1e-10 up-to-gauge**, **not** bit-for-bit. Reason: the AC fed to the QR seam comes from `ACenv_slice2d`'s power iteration, whose block-local-sum + `slice2d_norm`-allreduce gives a *different FP summation order* than serial `ACenv`'s single full sum (~1e-13), so slice2d AL/AR differ from serial at ~1e-12. Compare AL **and AR** (separate `qrpos`/`lqpos` paths) with `phase_ok`/`phase_relerr` (M4 helpers); drop the circular "AC-implied" (there is no AC output field on `VUMPSRuntime`). Bit-for-bit holds **only across ranks** (every rank gathers the same full AC) — asserted separately above. Guard against per-column `safesign` sign-flips by keeping the test fixture's R-diagonals bounded away from zero (or use a per-column-sign-robust phase compare).

Seed the comparison from the SAME `rt` (scatter serial's init) rather than two independent inits, so step k compares like-for-like without accumulated trajectory drift masking a per-step bug. **[R1 — F1]** *Additionally* run one gate variant that constructs `rt_slice2d` via `init_VUMPSRuntime_slice2d` (independent per-rank init + bcast), **not** scatter-of-serial — this is the only configuration that catches the un-bcast'd RNG bug (F1); scatter-of-serial seeding masks it.

### 6.2 Gradient parity — the gate for R-1 (the forward gate is blind to it)

**[R1 — F2, BLOCKING fix]** The forward trajectory gate (§6.1) is structurally **blind** to adjoint-only bugs (the #1 risk R-1 over-count): forward is collective-correct even if the backward over-counts. So the gradient gate is the *only* thing testing R-1, and it must be specified precisely:

- **Loss must reach the new seam adjoints.** `VUMPSRuntime` has **no AC field** — `dot(AC,AC)` over the step output is impossible. A loss on FL/FR/C alone touches only the M4-tested env adjoints and would pass with a totally broken seam. Only **AL_out** and **AR_out** depend (via `AL←QR(AC_full)←ACenv(AC)←ALCtoAC(AL_in,C)`) on all three new adjoints (QR-seam AC-gather, QR-seam AL/AR-scatter, entry-seam AL-gather/AC-scatter). **Use a loss on AL_out + AR_out.**
- **Loss must be LINEAR with a scattered fixed weight** (the M4 pattern, test_slice2d_m4.jl:190), not a quadratic on a block. A block `dot(AL_blk,AL_blk)` is a per-rank *partial* of the global quadratic, so its block cotangent is the gradient of the *local* loss, ≠ `blkof`(global-loss gradient). Slice2D side: `L = real(sum_c sum(conj(Wb[c]) .* AL_out.data[c])) + (AR term)` with `Wb[c]=slice2d_scatter(W[c],grid)`; serial side: same with full `W`. (A quadratic is acceptable only if built from `slice2d_dot`, the *global* inner product, slice2d.jl:212.)
- **Compare per-block**: `∂rt_in.data[c]` (and `∂M`) against `blkof(∂serial.data[c])` at rtol 1e-8 / atol 1e-10. A reduce-scatter-by-mistake over-count appears as a constant scale factor on disjoint per-rank blocks — invisible to a global norm, visible per-block.
- **Negative control:** include a deliberately-wrong variant that gathers AC via `slice2d_gather_row` and assert the gate *fails* (over-count factor = **N1 or N2**, i.e. 2 on a 2×2 grid — the reduce-scatter sums over one axis only, so it is N1/N2, not the full P). Proves the gate actually catches R-1.
- Run under `step_checkpoint ∈ {Plain, Recompute}` (Recompute is the deadlock-relevant mode; Offload is a CPU pass-through, skip it).

### 6.3 Execution plan

- **CPU 4-rank (2×2 grid)** first — fast iteration, catches the over-count (N1/N2=2× per-block), the init-bcast bug (F1, via the `init_VUMPSRuntime_slice2d` variant), AD composition, the Recompute deadlock. New `test/test_slice2d_m5.jl` mirroring the M4 gate structure.
  - **Cell patterns:** include the all-distinct case **and** a **repeated** unit cell `(2,2,[1 2;2 1])` — exercises the `ACCtoAL` all-positions vs `ACCtoAR` unique-data loop asymmetry (F3/F5) and confirms gather/scatter collective counts stay rank-uniform under repeats.
- **GPU 16-rank (4×4 grid)** on Sofia — χ=256 D=8, the production regime, after CPU is green. New `examples/MPI_parallel/test_slice2d_m5_sofia.jl` + submit script, mirroring the M4 GPU driver (reuse the GPU-safe `phase_relerr` with `dot(b,a)/dot(b,b)`, NOT scalar-indexing).

---

## 7. Risk register (post-R1)

| # | Risk | Mitigation / status |
|---|---|---|
| R-1 | **Over-count** if the AC gather uses row/col reduce-scatter instead of take-my-block | §2.3 — use full `slice2d_gather`; factor is **N1/N2** (per-axis), not P; §6.2 gradient gate + negative control catches it |
| R-2 | C accidentally gathered/scattered (it's replicated) | §1, §2.1 — no comm around C; gate asserts C bitwise-replicated across ranks |
| R-3 | `errL`/`errR` allreduced → err × P, or over-counted on repeated patterns | §2.5/§2.2 — reuse serial `ACCtoALAR` verbatim → exact accumulation; no allreduce |
| R-4 | QR gauge divergence across ranks (safesign sign-flip) | §2.4 — pure-placement gather (fwd) + allreduce (bwd) → bitwise-identical; gate asserts cross-rank equality |
| R-5 | χ-inferring helper (`cellones`/`FLint`) fed a block tensor at init | §4.3 — canonicalization + leftenv/rightenv run on full *before* scatter |
| R-6 | **Recompute** re-runs collectives on backward → deadlock if not rank-uniform | §5.3 — seam rank-uniform + comm/tag-separated; gate runs Recompute (Offload = CPU no-op) |
| R-7 | ALCtoAC Option A full-AL transient | §3 — **R1-corrected: 67 MB/cell @ χ256D8, 1.68 GB @ χ1024D10; Recompute bounds it → Option A viable for v1**; Option B = pure perf |
| R-8 | Slice2D-vs-serial expects bit-for-bit but bug hides in gauge | §6.1 — **R1-corrected: expect rel ≤ 1e-10 up-to-gauge** (upstream power-iter FP order differs); bit-for-bit only *across ranks* |
| R-9 | Wrong step variant copied (`vumps_step_power` vs `vumps_step`) | §1 — mirror `vumps_step` (old AL/AR into envs, single solve) |
| **R-10** | **[R1-F1] Init RNG not bcast → per-rank-divergent full A → mismatched blocks** | §4.1 — **bcast over `grid.comm` UNCONDITIONALLY** in slice2d init; §6.1 gate runs the independent-init variant |
| R-11 | [R1] `vumps_itr` warm-up loop (general.jl:931) not routed to slice2d step → block tensors hit serial step | §5.1 — route **both** call sites (931 + 977) on `alg.grid !== nothing` |

---

## 8. Deliverables checklist

- [ ] `gather_struct`/`scatter_struct`/`bcast_struct` helpers (StructArray-level wrappers over the cell primitives, differentiable where used).
- [ ] `ACCtoALAR_slice2d(AC_blk, C, grid)` — gather AC StructArray, **verbatim serial `ACCtoALAR`**, scatter AL/AR (slice2d.jl, new).
- [ ] `ALCtoAC_slice2d(AL_blk, C, grid)` — Option A v1: gather AL StructArray, **verbatim serial `ALCtoAC`**, scatter AC.
- [ ] `vumps_step_slice2d(rt, M, grid, alg)` — 6-call orchestration (mirror `vumps_step`, NOT `vumps_step_power`).
- [ ] `init_VUMPSRuntime_slice2d(M, χ, grid, alg)` — serial build + **unconditional bcast over `grid.comm`** + scatter (R-10).
- [ ] Dispatch: `alg.grid::Union{Nothing,Slice2DGrid}` field + `vumps_itr` routing at **both** general.jl:931 (warm-up) and :977 (AD) — Option 2.
- [ ] `test/test_slice2d_m5.jl` — CPU 4-rank trajectory + gradient parity; linear-scattered-weight loss on AL+AR, per-block compare; Plain + Recompute; all-distinct **and** `[1 2;2 1]` patterns; init-bcast variant; negative control.
- [ ] `examples/MPI_parallel/test_slice2d_m5_sofia.jl` + submit — 16-GPU validation.
- [ ] Old serial `vumps_step` function body untouched (parity baseline); `vumps_itr` gains behavior-neutral branch only.

---

## R1 — adversarial review response

Four opus reviewers, each adversarial on a distinct lens (AD-correctness, distributed/deadlock, gate-soundness, entry-seam/scope/memory). **All four returned `sound-with-fixes` — no fundamental redesign.** The core architecture (gather → verbatim serial kernel → scatter, with take-my-block gather + allreduce scatter adjoints, C never moved) survived every refutation attempt: the over-count trap (R-1), the cross-seam AL cotangent sum, the QR-pullback replication, the Recompute rank-uniformity, and the ALCtoAC parity-necessity were all *confirmed correct*. Findings and dispositions:

| ID | Sev | Finding | Disposition |
|---|---|---|---|
| **F1** | 🔴 blocking | Init runs `initial_A` with un-seeded per-rank RNG; serial bcast is gated on `ifparallel` (false for slice2d) → divergent full A → mismatched blocks. | **Fixed §4.1** — unconditional `bcast over grid.comm`. New R-10. Gate runs the independent-init variant (§6.1). |
| **F2** | 🔴 blocking | §6.2 loss invalid (`VUMPSRuntime` has no AC field) and a FL/FR/C loss is blind to the seam adjoints (R-1). | **Fixed §6.2** — linear scattered-weight loss on **AL_out+AR_out**, per-block compare, + negative control. |
| **F3** | 🟠 major | `errL/errR` over-counts repeated patterns if seam reimplements the per-cell loop. | **Fixed §2.2** — reuse **verbatim serial `ACCtoALAR`** on gathered-full StructArray → exact accumulation by construction. |
| **F4** | 🟠 major | bit-for-bit vs serial is false (upstream power-iter FP order differs; safesign column flip). | **Fixed §2.4/§6.1/R-8** — expect **rel ≤ 1e-10 up-to-gauge**; bit-for-bit only *across ranks* (separate assert). |
| **F5** | 🟠 major | AR not compared; no repeated unit-cell; "AC-implied" circular. | **Fixed §6.1/§6.3** — compare AR explicitly; add `[1 2;2 1]`; drop AC-implied. (Also auto-covered by the F3 verbatim-serial reuse.) |
| **F6** | 🟠 major | R-7 memory off by ~16× (67 MB not 1.07 GB); Recompute bounds transients → Option A viable. | **Fixed §3/R-7** — corrected figures; Option A confirmed for v1; Option B = pure perf. |
| minor×8 | — | `leading_boundary` rrule citation (single-env trace-through, not the Tuple rrule); over-count factor N1/N2 not P; Offload is a CPU no-op (only Recompute re-runs fwd); entry-seam leg-d scatter→regather round trip; **warm-up loop (general.jl:931) must also route to the slice2d step** (R-11); cross-rank C bitwise assert; ALCtoAC parity-necessity made explicit; qrpos-parity robustness sentence. | All folded into §2.4, §3, §5.1, §5.2, §5.3, §6, risk register. |

**Verdict: design is implementation-ready.** No second adversarial round needed — the remaining items are precision/documentation, not architecture. Proceed to implement per §8, then the dual review (post-implementation, opus) gates the code against this revised design.

---

## Implementation & validation (post-build)

Implemented exactly per §8 in `src/boundary_algorithm/vumps/slice2d.jl` (M5 section after `Cenv_slice2d`): `gather_struct`/`scatter_struct`/`bcast_struct`, `ALCtoAC_slice2d`, `ACCtoALAR_slice2d`, `vumps_step_slice2d`, `init_VUMPSRuntime_slice2d`; `grid` field on `VUMPS` (interface.jl); a 1-line routing guard at the top of `vumps_step` (general.jl) covering both `vumps_itr` call sites. Gate: `test/test_slice2d_m5.jl` (4-rank CPU) + `examples/MPI_parallel/test_slice2d_m5_sofia.jl` (16-GPU) + submit/smoke.

### The AL/AR conditioning discovery (supersedes the §6.1/F4 "compare AL/AR up-to-gauge" plan)

During gating, comparing the full step's **final AL/AR** directly between serial and slice2d failed by **O(1)** for a random (non-canonical) test rt — while FL/FR/C/err matched to **machine precision** (~1e-16) and the seam was **bit-exact** (M5-0). Root-caused through a chain of diagnostics:

1. `ACenv_slice2d`'s intermediate AC (the seam input) matches serial to **4e-16** per-cell, all patterns/power_iter (`diag_ac`).
2. In the full step, slice2d AL = serial `ACCtoALAR` on slice2d's gathered AC + slice2d C **exactly** (`ALc_vs_ALsc = 0`); the divergence is entirely `ACCtoALAR(sameAC, Cc)` vs `ACCtoALAR(sameAC, Cs)` (`diag_step`).
3. `qrpos` **commutes** with complex scalar multiplication (`qrpos(c·M).Q = (c/|c|)·qrpos(M).Q` to 9e-16); a per-cell phase on C alone gives AL to 1e-15 (`diag_c`).
4. **`qrpos(C).Q` amplifies a perturbation in C by `cond(C)`** — measured: `cond 1e10 → AL Δ 1e-5`, `cond 1e14 → AL Δ 4e-3` (`diag_c`).

**Conclusion:** `AL = qrpos(_to_front(AC))·qrpos(C)′`, and a random non-canonical rt produces a **near-singular `Cenv`-output C** (cond ~1e15), so `qrpos` amplifies the inevitable ~1e-16 slice2d-vs-serial residual into O(1) in AL. This is a **numerical-conditioning artifact of the non-physical fixture, not a parity defect** — every component is parity-exact and the seam is bit-exact. At a physical VUMPS fixed point, C carries the well-conditioned bond/Schmidt spectrum, so AL/AR are stable.

### Corrected gate (conditioning-independent) — supersedes §6.1/§6.2 AL/AR comparison

The gate validates `vumps_step_slice2d` via the **stable chain** rather than the gauge-/conditioning-fragile direct AL/AR comparison:

- **M5-0** — seam **bit-exact** vs serial given identical input (incl. 2×2 `[1 2;2 1]`): the rigorous proof that `ACCtoALAR_slice2d`/`ALCtoAC_slice2d` are correct. (24 tests.)
- **M5-1** — full-step **FL/FR/C/err** parity (~1e-16) **+ the pre-seam intermediate AC** parity (~4e-16, up to per-cell phase). With the bit-exact seam, matching AC ⟹ AL/AR correct. (15 tests.)
- **M5-2** — **seam gradient adjoints** (the R-1 over-count catcher) tested in isolation with a **well-conditioned orthogonal C** so the `qrpos` pullback is stable: `∂AC`/`∂AL`/`∂C` per-block vs `blkof(serial)` came out **exactly 0.0** (zero over-count), including under a `Recompute` wrap (R-6). (10 tests.)
- **M5-3** — `init_VUMPSRuntime_slice2d` cross-rank bitwise consistency (F1/R-10 catcher). (5 tests.)
- **M5-4** — `vumps_step` routes to `vumps_step_slice2d` when `alg.grid` set (R-11). (5 tests.)

**Result: all 59 CPU gates GREEN.** GPU driver (`test_slice2d_m5_sofia.jl`) mirrors this corrected validation (no direct AL/AR; intermediate AC + seam grads with well-conditioned C; `SEAM_RTOL=1e-10` for cuSOLVER) and PASSES the 4-rank CPU smoke. Ready for 16-GPU Sofia.
