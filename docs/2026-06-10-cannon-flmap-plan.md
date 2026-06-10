# Cannon-Style 2D Distributed FLmap — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement `FLmap_cannon` — a two-stage Cannon-ring distributed FLmap where the environment tensor lives in N1×N2 blocks across GPUs — with forward, hand-written rrule backward, CPU mpiexec tests, and Sofia 4×H200 validation.

**Architecture:** Rank (r1,r2) on an N1×N2 grid holds FL block `FL[a_range(r1), :, :, i_range(r2)]`; ALu/ALd/M stay replicated this round. Stage 1 rotates FL blocks along the row ring while accumulating a stationary intermediate G; stage 2 contracts with ALu and reduce-scatters along the column. Output distribution = input distribution, so the map is iterable. Backward replays the ring with paired dFL accumulators. Design: `docs/2026-06-10-cannon-flmap-design.md`.

**Tech Stack:** Julia, MPI.jl (CUDA-aware Isend/Irecv), TensorOperations `@tensor`, Zygote + ChainRulesCore (hand-written rrule), CUDA.jl, Slurm on Sofia VUB.

**Branch:** `claude/sad-saha-3ec6bf` (this worktree). Commit after every task.

**Conventions the executor must know:**

- Rank ↔ grid: `rank = r1*N2 + r2`. `row_comm` rank == r2 (size N2), `col_comm` rank == r1 (size N1).
- Uneven χ handled by the existing `split_ranges(χ, N)` helper (`src/contraction/forloop_parallel_MPI.jl:19`).
- All MPI tests rely on **identical random tensors on every rank**: `Random.seed!` with a fixed value immediately before each `rand` group. Never branch on `rank` before generating test tensors.
- Local tests run on **CPU Arrays** (CUDA-aware MPI does not exist on Windows MS-MPI). GPU coverage comes from the Sofia driver.
- `synchronize(x)` (TeneT's, `src/utils/gpu.jl:108`) is a no-op for Array, stream-sync for CuArray. Call it before every `MPI.Isend`/after compute, exactly as the existing collectives do.
- MPI tags: this feature uses `_TAG_BASE + 700/710/720/730` (existing code stops at 600).

---

## Task 1: Worktree environment sanity

**Files:** none (environment only)

**Step 1: Copy Manifest.toml from the main checkout** (git worktrees skip gitignored files)

```powershell
Copy-Item "D:\1 - research\1.26 - iPEPS_opt\TeneT.jl\Manifest.toml" "."
```

**Step 2: Verify the package loads**

Run: `julia --project=. -e "using TeneT; println(\"OK\")"`
Expected: `OK` (first run may precompile for a few minutes).

**Step 3: Verify mpiexec works (4 CPU ranks)**

Run: `julia --project=. -e "using MPI; run(``$(MPI.mpiexec()) -n 4 hostname``)"`
Expected: 4 hostname lines. If MPI.mpiexec errors, run `julia --project=. -e "using Pkg; Pkg.build(\"MPI\")"` first.

No commit (no repo changes).

---

## Task 2: CannonGrid + test launcher

**Files:**
- Create: `src/contraction/cannon_2d.jl`
- Modify: `src/TeneT.jl` (add include after line 62)
- Create: `test/test_cannon.jl`
- Create: `test/run_test_cannon.jl`

**Step 1: Write the launcher** `test/run_test_cannon.jl`:

```julia
# Launch test_cannon.jl under 4 MPI ranks. Usage: julia --project=. test/run_test_cannon.jl
using MPI
const julia_exe = first(Base.julia_cmd().exec)
const proj = dirname(Base.active_project())
run(`$(MPI.mpiexec()) -n 4 $julia_exe --project=$proj $(joinpath(@__DIR__, "test_cannon.jl"))`)
```

**Step 2: Write the failing test** `test/test_cannon.jl`:

```julia
# Cannon-style 2D distributed FLmap tests. Run via: julia --project=. test/run_test_cannon.jl
# CPU Arrays only (no CUDA-aware MPI on local machines); GPU is covered by
# examples/MPI_parallel/test_cannon_sofia.jl.
using Test
using MPI
using LinearAlgebra
using Random
using Zygote
using TeneT
using TeneT: cannon_grid, CannonGrid, cannon_scatter, cannon_gather, FLmap_cannon,
             FLmap, split_ranges, _cannon_stage1, _cannon_stage2

MPI.Init()
const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
@assert MPI.Comm_size(comm) == 4 "test_cannon.jl expects exactly 4 ranks"

# Identical tensors on every rank: fixed seed immediately before each rand group.
function make_leg5(χ, D; d=2, seed=42)
    Random.seed!(seed)
    FL  = rand(ComplexF64, χ, D, D, χ)
    ALu = rand(ComplexF64, χ, D, D, χ)
    ALd = rand(ComplexF64, χ, D, D, χ)
    M1  = rand(ComplexF64, D, D, D, D, d)
    M2  = rand(ComplexF64, D, D, D, D, d)
    W   = rand(ComplexF64, χ, D, D, χ)     # fixed weight for scalar losses
    return FL, ALu, ALd, M1, M2, W
end

@testset "cannon_grid" begin
    for (N1, N2) in ((2, 2), (1, 4), (4, 1))
        g = cannon_grid(N1, N2)
        @test g isa CannonGrid
        @test g.r1 * N2 + g.r2 == rank
        @test MPI.Comm_size(g.row_comm) == N2
        @test MPI.Comm_size(g.col_comm) == N1
        @test MPI.Comm_rank(g.row_comm) == g.r2
        @test MPI.Comm_rank(g.col_comm) == g.r1
    end
end

println("rank $rank: test_cannon.jl done")
```

**Step 3: Run to verify it fails**

Run: `julia --project=. test/run_test_cannon.jl`
Expected: FAIL — `UndefVarError` (`cannon_grid` not defined / not importable).

**Step 4: Implement** — create `src/contraction/cannon_2d.jl`:

```julia
# Cannon-style 2D distributed FLmap (map level).
# Design: docs/2026-06-10-cannon-flmap-design.md
#
# Rank (r1, r2) on an N1×N2 grid holds
#   FL block:      FL[a_range(r1), :, :, i_range(r2)]
#   result block:  result[d_range(r1), :, :, l_range(r2)]
# ALu / ALd / M are fully replicated (AL distribution comes with the later
# leftenv integration round). Stage 1 rotates FL blocks along the row ring
# while accumulating the stationary intermediate G; stage 2 contracts with
# ALu and reduce-scatters along the column. Output distribution = input
# distribution, so the map iterates without redistribution.

struct CannonGrid
    N1::Int
    N2::Int
    r1::Int
    r2::Int
    rank::Int
    comm::MPI.Comm
    row_comm::MPI.Comm   # fixed r1; rank within == r2 (size N2)
    col_comm::MPI.Comm   # fixed r2; rank within == r1 (size N1)
end

const _cannon_grid_cache = Ref{Union{Nothing, CannonGrid}}(nothing)

function cannon_grid(N1::Integer, N2::Integer; comm = MPI.COMM_WORLD)
    g = _cannon_grid_cache[]
    if g !== nothing && g.N1 == N1 && g.N2 == N2 && g.comm == comm
        return g
    end
    nprocs = MPI.Comm_size(comm)
    @assert nprocs == N1 * N2 "cannon_grid: nprocs=$nprocs ≠ N1*N2=$(N1 * N2)"
    rank = MPI.Comm_rank(comm)
    r1, r2 = divrem(rank, N2)
    row_comm = MPI.Comm_split(comm, r1, r2)
    col_comm = MPI.Comm_split(comm, r2, r1)
    g = CannonGrid(N1, N2, r1, r2, rank, comm, row_comm, col_comm)
    _cannon_grid_cache[] = g
    return g
end
```

**Step 5: Wire the include** — in `src/TeneT.jl`, after `include("contraction/forloop_parallel_MPI.jl")` (line 62), add:

```julia
include("contraction/cannon_2d.jl")
```

**Step 6: Run test to verify it passes** (the later imports `cannon_scatter` etc. will still fail — temporarily comment the not-yet-existing names out of the `using TeneT:` line, or define the testset import minimally; simplest is to import only what exists so far and extend the import line per task)

Run: `julia --project=. test/run_test_cannon.jl`
Expected: `Test Summary: cannon_grid | Pass 24` (6 tests × 4 ranks print per-rank summaries) and 4× `rank N: test_cannon.jl done`.

**Step 7: Commit**

```powershell
git add src/contraction/cannon_2d.jl src/TeneT.jl test/test_cannon.jl test/run_test_cannon.jl
git commit -m "feat: add CannonGrid 2D process grid for distributed FLmap"
```

---

## Task 3: Stage kernels (serial-equivalence tested)

**Files:**
- Modify: `src/contraction/cannon_2d.jl`
- Modify: `test/test_cannon.jl`

**Step 1: Append the failing test** to `test/test_cannon.jl` (and add `_cannon_stage1`, `_cannon_stage2` to the import line):

```julia
@testset "stage kernels == FLmap (local, no MPI)" begin
    χ, D = 8, 3
    FL, ALu, ALd, M1, M2, _ = make_leg5(χ, D; seed=101)
    G = _cannon_stage1(FL, ALd, M1, M2)
    P = _cannon_stage2(G, ALu)
    ref = FLmap(FL, ALu, ALd, M1, M2)
    @test P ≈ ref rtol = 1e-12
end
```

**Step 2: Run to verify it fails**

Run: `julia --project=. test/run_test_cannon.jl`
Expected: FAIL — `_cannon_stage1` not defined.

**Step 3: Implement** — append to `src/contraction/cannon_2d.jl`:

```julia
# ─── Stage kernels (leg5) ─────────────────────────────────────────────────
#
# FLmap splits into two stages so distributed FLOPs stay exactly serial/P:
#   stage 1 contracts the i leg and folds in M (G is the only big transient,
#   sized (χ/N1)·D⁴·(χ/N2) — the serial intermediate / P);
#   stage 2 contracts a/b/c with ALu, leaving a full-length d leg for the
#   column reduce-scatter.

function _cannon_stage1(FL, ALd, M1, M2)
    @tensor G[a, b, c, g, h, l] := FL[a, e, f, i] * ALd[i, j, k, l] *
                                   M1[e, j, g, b, p] * M2[f, k, h, c, p]
    return G
end

# In-place accumulating variant for the forward ring (avoids a second G-sized
# temporary). Backward uses the non-mutating version through Zygote.pullback.
function _cannon_stage1_add!(G, FL, ALd, M1, M2)
    @tensor G[a, b, c, g, h, l] += FL[a, e, f, i] * ALd[i, j, k, l] *
                                   M1[e, j, g, b, p] * M2[f, k, h, c, p]
    return G
end

function _cannon_stage2(G, ALu)
    @tensor P[d, g, h, l] := G[a, b, c, g, h, l] * ALu[a, b, c, d]
    return P
end
```

**Step 4: Run test to verify it passes**

Run: `julia --project=. test/run_test_cannon.jl`
Expected: both testsets pass on all 4 ranks.

**Step 5: Commit**

```powershell
git add src/contraction/cannon_2d.jl test/test_cannon.jl
git commit -m "feat: add two-stage Cannon contraction kernels for leg5 FLmap"
```

---

## Task 4: cannon_scatter / cannon_gather (forward)

**Files:**
- Modify: `src/contraction/cannon_2d.jl`
- Modify: `test/test_cannon.jl`

**Step 1: Append the failing test** (add `cannon_scatter`, `cannon_gather` to imports):

```julia
@testset "scatter/gather roundtrip" begin
    for (N1, N2) in ((2, 2), (1, 4), (4, 1)), χ in (16, 18)   # 18: uneven blocks
        Random.seed!(500 + χ + 10N1)
        g = cannon_grid(N1, N2)
        FL = rand(ComplexF64, χ, 4, 4, χ)
        blk = cannon_scatter(FL, g)
        a_rs = split_ranges(χ, N1); i_rs = split_ranges(χ, N2)
        @test size(blk) == (length(a_rs[g.r1 + 1]), 4, 4, length(i_rs[g.r2 + 1]))
        @test blk == FL[a_rs[g.r1 + 1], :, :, i_rs[g.r2 + 1]]
        FL2 = cannon_gather(blk, g)
        @test FL2 ≈ FL
    end
end
```

**Step 2: Run to verify it fails** — `cannon_scatter` not defined.

**Step 3: Implement** — append to `src/contraction/cannon_2d.jl`:

```julia
# ─── Boundary shims: full ↔ distributed blocks ────────────────────────────

"""
    cannon_scatter(T_full, grid) -> block

Local block of a replicated tensor: first leg split N1-ways (by r1), last leg
split N2-ways (by r2). Pure indexing — no communication.
"""
function cannon_scatter(T_full::AbstractArray, grid::CannonGrid)
    n = ndims(T_full)
    a_rs = split_ranges(size(T_full, 1), grid.N1)
    i_rs = split_ranges(size(T_full, n), grid.N2)
    inds = ntuple(j -> j == 1 ? a_rs[grid.r1 + 1] :
                       (j == n ? i_rs[grid.r2 + 1] : Colon()), n)
    return T_full[inds...]
end

"""
    cannon_gather(blk, grid) -> full

Reassemble the full tensor from all ranks' blocks (allgatherv on flattened
blocks, then per-block placement). Test/integration shim — not on the hot path.
"""
function cannon_gather(blk::AbstractArray, grid::CannonGrid)
    n = ndims(blk)
    χ1 = MPI.Allreduce(size(blk, 1), +, grid.col_comm)
    χ2 = MPI.Allreduce(size(blk, n), +, grid.row_comm)
    a_rs = split_ranges(χ1, grid.N1)
    i_rs = split_ranges(χ2, grid.N2)
    mid = size(blk)[2:n-1]
    P = grid.N1 * grid.N2
    counts = Cint[length(a_rs[divrem(r, grid.N2)[1] + 1]) * prod(mid) *
                  length(i_rs[divrem(r, grid.N2)[2] + 1]) for r in 0:P-1]
    displs = cumsum([0; counts[1:end-1]])
    buf = similar(blk, sum(counts))
    copyto!(view(buf, displs[grid.rank + 1] + 1 : displs[grid.rank + 1] + counts[grid.rank + 1]),
            vec(blk))
    allgatherv_p2p!(buf, counts, grid.comm)
    full = similar(blk, χ1, mid..., χ2)
    for r in 0:P-1
        s, t = divrem(r, grid.N2)
        seg = reshape(view(buf, displs[r + 1] + 1 : displs[r + 1] + counts[r + 1]),
                      length(a_rs[s + 1]), mid..., length(i_rs[t + 1]))
        inds = ntuple(j -> j == 1 ? a_rs[s + 1] : (j == n ? i_rs[t + 1] : Colon()), n)
        view(full, inds...) .= seg
    end
    return full
end
```

**Step 4: Run test to verify it passes.**

**Step 5: Commit**

```powershell
git add src/contraction/cannon_2d.jl test/test_cannon.jl
git commit -m "feat: add cannon_scatter/cannon_gather boundary shims"
```

---

## Task 5: FLmap_cannon forward

**Files:**
- Modify: `src/contraction/cannon_2d.jl`
- Modify: `test/test_cannon.jl`

**Step 1: Append the failing test** (add `FLmap_cannon` to imports):

```julia
@testset "FLmap_cannon forward" begin
    for (N1, N2) in ((2, 2), (1, 4), (4, 1)), χ in (16, 18)
        D = 3
        FL, ALu, ALd, M1, M2, _ = make_leg5(χ, D; seed=600 + χ + 10N1)
        g = cannon_grid(N1, N2)
        ref = FLmap(FL, ALu, ALd, M1, M2)
        out_blk = FLmap_cannon(cannon_scatter(FL, g), ALu, ALd, (M1, M2), g)
        a_rs = split_ranges(χ, N1); i_rs = split_ranges(χ, N2)
        @test size(out_blk) == (length(a_rs[g.r1 + 1]), D, D, length(i_rs[g.r2 + 1]))
        @test cannon_gather(out_blk, g) ≈ ref rtol = 1e-12
        # single-M entry point (M2 = conj(M1) internally)
        out1 = cannon_gather(FLmap_cannon(cannon_scatter(FL, g), ALu, ALd, M1, g), g)
        @test out1 ≈ FLmap(FL, ALu, ALd, M1) rtol = 1e-12
        # iterability: feed the output block straight back in
        out2_blk = FLmap_cannon(out_blk, ALu, ALd, (M1, M2), g)
        ref2 = FLmap(ref, ALu, ALd, M1, M2)
        @test cannon_gather(out2_blk, g) ≈ ref2 rtol = 1e-11
    end
end
```

**Step 2: Run to verify it fails** — `FLmap_cannon` not defined.

**Step 3: Implement** — append to `src/contraction/cannon_2d.jl`:

```julia
# ─── Ring / column communication ──────────────────────────────────────────

# Send `cur` to the left row neighbor (r2-1) and receive the next block from
# the right (r2+1). Fresh exact-size buffer per step (uneven χ blocks differ
# in size; CUDA pool makes the allocation cheap). Never mutates `cur`.
function _cannon_row_shift(cur, grid::CannonGrid, recv_size; tag = _TAG_BASE + 700)
    dest = mod(grid.r2 - 1, grid.N2)
    src  = mod(grid.r2 + 1, grid.N2)
    recv = similar(cur, recv_size)
    synchronize(cur)
    req_r = MPI.Irecv!(recv, grid.row_comm; source = src, tag = tag)
    req_s = MPI.Isend(cur, grid.row_comm; dest = dest, tag = tag)
    MPI.Waitall([req_s, req_r])
    return recv
end

# Sum `partial` (full d leg, local l block) over the column and keep the local
# d block. Direct algorithm: each rank sends every other rank its chunk and
# accumulates the N1-1 contributions for its own chunk. Chunk extraction via
# getindex (allocating) keeps MPI buffers contiguous.
function _cannon_col_reduce_scatter(partial, grid::CannonGrid, d_rs)
    N1, r1 = grid.N1, grid.r1
    acc = partial[d_rs[r1 + 1], :, :, :]
    N1 == 1 && return acc
    synchronize(partial)
    reqs = MPI.Request[]
    recvbufs = Vector{typeof(acc)}(undef, N1)
    for j in 0:N1-1
        j == r1 && continue
        rb = similar(acc)
        recvbufs[j + 1] = rb
        push!(reqs, MPI.Irecv!(rb, grid.col_comm; source = j, tag = _TAG_BASE + 710))
    end
    sendbufs = Vector{Any}(undef, N1)   # keep alive until Waitall
    for j in 0:N1-1
        j == r1 && continue
        sb = partial[d_rs[j + 1], :, :, :]
        synchronize(sb)
        sendbufs[j + 1] = sb
        push!(reqs, MPI.Isend(sb, grid.col_comm; dest = j, tag = _TAG_BASE + 710))
    end
    MPI.Waitall(reqs)
    for j in 0:N1-1
        j == r1 && continue
        acc .+= recvbufs[j + 1]
    end
    return acc
end

# ─── Forward ──────────────────────────────────────────────────────────────

# Shared by FLmap_cannon and its rrule. Returns (result_blk, G); the rrule
# captures G for the stage-2 pullback.
function _cannon_forward(FL_blk, ALu, ALd, M1, M2, grid::CannonGrid)
    N1, N2, r1, r2 = grid.N1, grid.N2, grid.r1, grid.r2
    χ = size(ALu, 1)
    a_rs = split_ranges(χ, N1)
    i_rs = split_ranges(χ, N2)
    l_rng = i_rs[r2 + 1]
    @assert size(FL_blk, 1) == length(a_rs[r1 + 1]) && size(FL_blk, 4) == length(l_rng) "FLmap_cannon: block shape $(size(FL_blk)) inconsistent with grid ($(N1)×$(N2)) and χ=$χ"

    # Stage 1: rotate FL blocks along the row ring, accumulate stationary G.
    Dg, Dh = size(M1, 3), size(M2, 3)
    Db, Dc = size(M1, 4), size(M2, 4)
    G = similar(FL_blk, length(a_rs[r1 + 1]), Db, Dc, Dg, Dh, length(l_rng))
    G .= 0
    cur = FL_blk
    for k in 0:N2-1
        t = mod(r2 + k, N2)
        ALd_slice = view(ALd, i_rs[t + 1], :, :, l_rng)
        _cannon_stage1_add!(G, cur, ALd_slice, M1, M2)
        if k < N2 - 1
            t_next = mod(r2 + k + 1, N2)
            cur = _cannon_row_shift(cur, grid,
                (length(a_rs[r1 + 1]), size(FL_blk, 2), size(FL_blk, 3), length(i_rs[t_next + 1])))
        end
    end

    # Stage 2: contract a/b/c with the local row slice of replicated ALu,
    # then reduce-scatter the full-d partial along the column.
    ALu_slice = view(ALu, a_rs[r1 + 1], :, :, :)
    partial = _cannon_stage2(G, ALu_slice)
    result = _cannon_col_reduce_scatter(partial, grid, a_rs)
    return result, G
end

"""
    FLmap_cannon(FL_blk, ALu, ALd, M, grid; inner_etype=nothing) -> result_blk

Distributed FLmap on an N1×N2 Cannon grid. `FL_blk` and the returned block
follow the convention: first χ leg split N1-ways by r1, last χ leg split
N2-ways by r2. `M` is a leg5 tensor or an `(M1, M2)` tuple; ALu/ALd/M are
replicated on every rank. See docs/2026-06-10-cannon-flmap-design.md.
"""
function FLmap_cannon(FL_blk, ALu, ALd, M, grid::CannonGrid; inner_etype = nothing)
    M1, M2 = M isa Tuple ? M : (M, conj(M))
    T_orig = eltype(FL_blk)
    do_cast = inner_etype !== nothing && inner_etype != real(T_orig)
    if do_cast
        FL_blk = _downcast_eltype(inner_etype, FL_blk)
        ALu = _downcast_eltype(inner_etype, ALu)
        ALd = _downcast_eltype(inner_etype, ALd)
        M1 = _downcast_eltype(inner_etype, M1)
        M2 = _downcast_eltype(inner_etype, M2)
    end
    result, _ = _cannon_forward(FL_blk, ALu, ALd, M1, M2, grid)
    return do_cast ? T_orig.(result) : result
end
```

**Step 4: Run test to verify it passes**

Run: `julia --project=. test/run_test_cannon.jl`
Expected: all testsets pass on 4 ranks. If the forward parity fails only for (2,2) but passes (1,4): bug in the column reduce-scatter; only (4,1) failing: bug in the row ring.

**Step 5: Commit**

```powershell
git add src/contraction/cannon_2d.jl test/test_cannon.jl
git commit -m "feat: FLmap_cannon forward — two-stage Cannon ring"
```

---

## Task 6: rrules for cannon_scatter / cannon_gather

**Files:**
- Modify: `src/autodiff/rules.jl` (append after the `parallel` rrule, line ~342)
- Modify: `test/test_cannon.jl`

**Step 1: Append the failing test:**

```julia
@testset "scatter/gather rrules" begin
    for (N1, N2) in ((2, 2), (1, 4))
        Random.seed!(700 + 10N1)
        g = cannon_grid(N1, N2)
        χ = 12
        FL = rand(ComplexF64, χ, 3, 3, χ)
        W  = rand(ComplexF64, χ, 3, 3, χ)
        # identity chain: gather(scatter(x)) == x, so dFL must equal conj-free W pullback
        loss(x) = real(sum(W .* cannon_gather(cannon_scatter(x, g), g)))
        l, back = Zygote.pullback(loss, FL)
        @test l ≈ real(sum(W .* FL))
        dFL = back(1.0)[1]
        l_ref, back_ref = Zygote.pullback(x -> real(sum(W .* x)), FL)
        @test dFL ≈ back_ref(1.0)[1]
    end
end
```

**Step 2: Run to verify it fails** — Zygote errors trying to differentiate through MPI calls / mutation (`Mutating arrays is not supported` or similar).

**Step 3: Implement** — append to `src/autodiff/rules.jl`:

```julia
# ─── AD rules for Cannon 2D distributed FLmap ─────────────────────────────
# Design: docs/2026-06-10-cannon-flmap-design.md §2. Communication adjoints:
# reduce-scatter ↔ allgather, ring shift ↔ reverse-replayed ring shift,
# replicated input ↔ allreduce(+) of per-rank gradient slices.

function ChainRulesCore.rrule(::typeof(cannon_scatter), T_full::AbstractArray, grid::CannonGrid)
    blk = cannon_scatter(T_full, grid)
    n = ndims(T_full)
    a_rs = split_ranges(size(T_full, 1), grid.N1)
    i_rs = split_ranges(size(T_full, n), grid.N2)
    inds = ntuple(j -> j == 1 ? a_rs[grid.r1 + 1] :
                       (j == n ? i_rs[grid.r2 + 1] : Colon()), n)
    function scatter_back(dblk)
        dfull = zero(T_full)
        view(dfull, inds...) .= unthunk(dblk)
        # Blocks are disjoint across ranks: allreduce stitches them into the
        # full replicated-input gradient on every rank.
        allreduce_p2p!(dfull, +, grid.comm)
        return NoTangent(), dfull, NoTangent()
    end
    return blk, scatter_back
end

function ChainRulesCore.rrule(::typeof(cannon_gather), blk::AbstractArray, grid::CannonGrid)
    full = cannon_gather(blk, grid)
    n = ndims(blk)
    a_rs = split_ranges(size(full, 1), grid.N1)
    i_rs = split_ranges(size(full, n), grid.N2)
    inds = ntuple(j -> j == 1 ? a_rs[grid.r1 + 1] :
                       (j == n ? i_rs[grid.r2 + 1] : Colon()), n)
    # Downstream of gather is replicated computation → identical dfull on every
    # rank; the adjoint is just "take my block".
    gather_back(dfull) = (NoTangent(), unthunk(dfull)[inds...], NoTangent())
    return full, gather_back
end
```

**Step 4: Run test to verify it passes.**

**Step 5: Commit**

```powershell
git add src/autodiff/rules.jl test/test_cannon.jl
git commit -m "feat: rrules for cannon_scatter/cannon_gather"
```

---

## Task 7: rrule for FLmap_cannon (the core backward)

**Files:**
- Modify: `src/contraction/cannon_2d.jl` (add `_cannon_col_allgather`)
- Modify: `src/autodiff/rules.jl`
- Modify: `test/test_cannon.jl`

**Step 1: Append the failing test:**

```julia
@testset "FLmap_cannon gradient parity" begin
    for (N1, N2) in ((2, 2), (1, 4), (4, 1)), χ in (16, 18)
        D = 3
        FL, ALu, ALd, M1, M2, W = make_leg5(χ, D; seed=800 + χ + 10N1)
        g = cannon_grid(N1, N2)

        loss_ref(FL, ALu, ALd, M1, M2) =
            real(sum(W .* FLmap(FL, ALu, ALd, M1, M2)))
        loss_can(FL, ALu, ALd, M1, M2) =
            real(sum(W .* cannon_gather(
                FLmap_cannon(cannon_scatter(FL, g), ALu, ALd, (M1, M2), g), g)))

        l_ref, back_ref = Zygote.pullback(loss_ref, FL, ALu, ALd, M1, M2)
        l_can, back_can = Zygote.pullback(loss_can, FL, ALu, ALd, M1, M2)
        @test l_can ≈ l_ref rtol = 1e-12
        g_ref = back_ref(1.0)
        g_can = back_can(1.0)
        for (i, name) in enumerate(("dFL", "dALu", "dALd", "dM1", "dM2"))
            @test isapprox(g_can[i], g_ref[i]; rtol = 1e-10)
        end

        # single-M entry: checks the dM = dM1 + conj(dM2) composition
        loss1_ref(FL, M) = real(sum(W .* FLmap(FL, ALu, ALd, M)))
        loss1_can(FL, M) = real(sum(W .* cannon_gather(
            FLmap_cannon(cannon_scatter(FL, g), ALu, ALd, M, g), g)))
        _, b1r = Zygote.pullback(loss1_ref, FL, M1)
        _, b1c = Zygote.pullback(loss1_can, FL, M1)
        gr, gc = b1r(1.0), b1c(1.0)
        @test isapprox(gc[1], gr[1]; rtol = 1e-10)   # dFL
        @test isapprox(gc[2], gr[2]; rtol = 1e-10)   # dM
    end
end
```

**Step 2: Run to verify it fails** — Zygote mutation error inside `_cannon_forward`.

**Step 3: Implement the column allgather** — append to `src/contraction/cannon_2d.jl`:

```julia
# Adjoint of _cannon_col_reduce_scatter: gather all column ranks' d-blocks
# into a full-d tensor (identical on every column rank).
function _cannon_col_allgather(dblk, grid::CannonGrid, d_rs)
    N1, r1 = grid.N1, grid.r1
    nmid = ndims(dblk) - 1
    χ = sum(length, d_rs)
    full = similar(dblk, χ, size(dblk)[2:end]...)
    view(full, d_rs[r1 + 1], ntuple(_ -> Colon(), nmid)...) .= dblk
    N1 == 1 && return full
    synchronize(dblk)
    reqs = MPI.Request[]
    recvbufs = Vector{Any}(undef, N1)
    for j in 0:N1-1
        j == r1 && continue
        rb = similar(dblk, length(d_rs[j + 1]), size(dblk)[2:end]...)
        recvbufs[j + 1] = rb
        push!(reqs, MPI.Irecv!(rb, grid.col_comm; source = j, tag = _TAG_BASE + 730))
    end
    for j in 0:N1-1
        j == r1 && continue
        push!(reqs, MPI.Isend(dblk, grid.col_comm; dest = j, tag = _TAG_BASE + 730))
    end
    MPI.Waitall(reqs)
    for j in 0:N1-1
        j == r1 && continue
        view(full, d_rs[j + 1], ntuple(_ -> Colon(), nmid)...) .= recvbufs[j + 1]
    end
    return full
end
```

**Step 4: Implement the rrule** — append to `src/autodiff/rules.jl`:

```julia
function ChainRulesCore.rrule(::typeof(FLmap_cannon), FL_blk, ALu, ALd, M, grid::CannonGrid;
                              inner_etype = nothing)
    is_tuple = M isa Tuple
    M1, M2 = is_tuple ? M : (M, conj(M))
    T_orig = eltype(FL_blk)
    do_cast = inner_etype !== nothing && inner_etype != real(T_orig)
    FL_c  = do_cast ? _downcast_eltype(inner_etype, FL_blk) : FL_blk
    ALu_c = do_cast ? _downcast_eltype(inner_etype, ALu) : ALu
    ALd_c = do_cast ? _downcast_eltype(inner_etype, ALd) : ALd
    M1_c  = do_cast ? _downcast_eltype(inner_etype, M1) : M1
    M2_c  = do_cast ? _downcast_eltype(inner_etype, M2) : M2

    result_c, G = _cannon_forward(FL_c, ALu_c, ALd_c, M1_c, M2_c, grid)
    result = do_cast ? T_orig.(result_c) : result_c

    function cannon_back(dresult)
        N1, N2, r1, r2 = grid.N1, grid.N2, grid.r1, grid.r2
        χ = size(ALu_c, 1)
        a_rs = split_ranges(χ, N1)
        i_rs = split_ranges(χ, N2)
        l_rng = i_rs[r2 + 1]

        d_c = unthunk(dresult)
        d_c = do_cast ? _boundary_cast(inner_etype, d_c) : d_c

        # 1. Adjoint of the column reduce-scatter: allgather dresult blocks.
        dpartial = _cannon_col_allgather(d_c, grid, a_rs)

        # 2. Stage-2 pullback (local; G captured from forward).
        ALu_slice = view(ALu_c, a_rs[r1 + 1], :, :, :)
        _, bp2 = pullback(_cannon_stage2, G, ALu_slice)
        dG, dALu_slice = bp2(dpartial)
        dALu = zero(ALu_c)
        view(dALu, a_rs[r1 + 1], :, :, :) .= dALu_slice

        # 3. Stage-1 reverse: replay the FL rotation with each block's dFL
        #    accumulator travelling alongside it. The accumulator for block t
        #    starts (zeros) at its home rank, collects one contribution per
        #    rank in the row, and the final shift lands it home — so dFL
        #    stays distributed, matching the input convention.
        dALd = zero(ALd_c)
        dM1 = zero(M1_c)
        dM2 = zero(M2_c)
        cur = FL_c
        dacc = zero(FL_c)
        for k in 0:N2-1
            t = mod(r2 + k, N2)
            ALd_slice = view(ALd_c, i_rs[t + 1], :, :, l_rng)
            _, bp1 = pullback(_cannon_stage1, cur, ALd_slice, M1_c, M2_c)
            dcur_k, dALd_k, dM1_k, dM2_k = bp1(dG)
            dacc .+= dcur_k
            view(dALd, i_rs[t + 1], :, :, l_rng) .+= dALd_k
            dM1 .+= dM1_k
            dM2 .+= dM2_k
            if N2 > 1
                t_next = mod(r2 + k + 1, N2)
                sz = (length(a_rs[r1 + 1]), size(FL_c, 2), size(FL_c, 3),
                      length(i_rs[t_next + 1]))
                if k < N2 - 1   # last replay shift of cur is unnecessary
                    cur = _cannon_row_shift(cur, grid, sz; tag = _TAG_BASE + 700)
                end
                dacc = _cannon_row_shift(dacc, grid, sz; tag = _TAG_BASE + 720)
            end
        end

        # 4. Replicated-input gradients: per-rank slices summed/stitched by a
        #    single allreduce each (picks up the NCCL fast path when enabled).
        allreduce_p2p!(dALu, +, grid.comm)
        allreduce_p2p!(dALd, +, grid.comm)
        allreduce_p2p!(dM1, +, grid.comm)
        allreduce_p2p!(dM2, +, grid.comm)

        dM = is_tuple ? (dM1, dM2) : dM1 .+ conj(dM2)
        dFL = dacc
        if do_cast
            dFL = T_orig.(dFL); dALu = T_orig.(dALu); dALd = T_orig.(dALd)
            dM = is_tuple ? (T_orig.(dM[1]), T_orig.(dM[2])) : T_orig.(dM)
        end
        return NoTangent(), dFL, dALu, dALd, dM, NoTangent()
    end
    return result, cannon_back
end
```

**Step 5: Run test to verify it passes**

Run: `julia --project=. test/run_test_cannon.jl`
Expected: all testsets pass. Debug order if gradient parity fails (design doc §3): (a) dacc shift-count alignment — verify with N2=2 by hand: dacc must shift on **every** k including the last, `cur` must not shift on the last; (b) `dacc` shift buffer size `sz` — it is sized for block `t_next`, same as `cur`'s; (c) allreduce double counting — each rank must write its dALu/dALd contribution into a **zero** full-size array, slices only.

**Step 6: Commit**

```powershell
git add src/contraction/cannon_2d.jl src/autodiff/rules.jl test/test_cannon.jl
git commit -m "feat: hand-written rrule for FLmap_cannon (reverse-replay ring)"
```

---

## Task 8: inner_etype boundary cast test

**Files:**
- Modify: `test/test_cannon.jl`

**Step 1: Append the test** (implementation already landed in Tasks 5/7):

```julia
@testset "inner_etype Float32 boundary cast" begin
    χ, D = 16, 3
    FL, ALu, ALd, M1, M2, W = make_leg5(χ, D; seed=900)
    g = cannon_grid(2, 2)
    ref = FLmap(FL, ALu, ALd, M1, M2)
    out = cannon_gather(
        FLmap_cannon(cannon_scatter(FL, g), ALu, ALd, (M1, M2), g; inner_etype = Float32), g)
    @test eltype(out) == ComplexF64          # upcast at exit
    @test out ≈ ref rtol = 1e-4              # F32 accuracy

    loss(FL) = real(sum(W .* cannon_gather(
        FLmap_cannon(cannon_scatter(FL, g), ALu, ALd, (M1, M2), g; inner_etype = Float32), g)))
    loss_ref(FL) = real(sum(W .* FLmap(FL, ALu, ALd, M1, M2)))
    dFL = Zygote.pullback(loss, FL)[2](1.0)[1]
    dFL_ref = Zygote.pullback(loss_ref, FL)[2](1.0)[1]
    @test eltype(dFL) == ComplexF64
    @test dFL ≈ dFL_ref rtol = 1e-3
end
```

**Step 2: Run** — expected PASS directly (cast support was built in). If it fails on eltype: the upcast at an exit path was missed.

**Step 3: Commit**

```powershell
git add test/test_cannon.jl
git commit -m "test: inner_etype Float32 boundary cast for FLmap_cannon"
```

---

## Task 9: Sofia GPU driver + submit script

**Files:**
- Create: `examples/MPI_parallel/test_cannon_sofia.jl`
- Create: `examples/MPI_parallel/Sofia/submit_test_cannon.sh`

**Step 1: Write the driver** `examples/MPI_parallel/test_cannon_sofia.jl`:

```julia
# Sofia 4×H200 Cannon FLmap validation: parity vs the slice path, per-rank
# memory accounting, timing. Launched by Sofia/submit_test_cannon.sh
# (mpirun -np 4, CUDA_VISIBLE_DEVICES pins one GPU per rank).
using MPI, CUDA, Zygote, LinearAlgebra, Random, Printf
using TeneT
using TeneT: cannon_grid, cannon_scatter, cannon_gather, FLmap_cannon,
             FLmap, FLmap_parallel, split_ranges, synchronize

MPI.Init()
const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
@assert MPI.Comm_size(comm) == 4

const χ = parse(Int, get(ENV, "TENET_CANNON_CHI", "400"))
const D = parse(Int, get(ENV, "TENET_CANNON_D", "10"))
report(s...) = rank == 0 && println(s...)
mem_used_gb() = (CUDA.total_memory() - CUDA.available_memory()) / 2^30
function mem_line(label)
    m = mem_used_gb()
    mmax = MPI.Allreduce(m, MPI.MAX, comm)
    rank == 0 && @printf("[mem]  %-32s rank0 %6.2f GB   max %6.2f GB\n", label, m, mmax)
end
function timed(f, label, n)
    f(); synchronize(CUDA.zeros(1)); MPI.Barrier(comm)        # warm
    t = MPI.Wtime()
    for _ in 1:n; f(); end
    synchronize(CUDA.zeros(1)); MPI.Barrier(comm)
    dt = (MPI.Wtime() - t) / n
    dtmax = MPI.Allreduce(dt, MPI.MAX, comm)
    rank == 0 && @printf("[time] %-32s %8.3f s/call (max over ranks)\n", label, dtmax)
end

report("=== Cannon FLmap Sofia test: χ=$χ D=$D, 2×2 grid ===")
Random.seed!(42)   # identical tensors on every rank
FL  = CuArray(rand(ComplexF64, χ, D, D, χ))
ALu = CuArray(rand(ComplexF64, χ, D, D, χ))
ALd = CuArray(rand(ComplexF64, χ, D, D, χ))
M1  = CuArray(rand(ComplexF64, D, D, D, D, 2))
M2  = CuArray(rand(ComplexF64, D, D, D, D, 2))
W   = CuArray(rand(ComplexF64, χ, D, D, χ))
g = cannon_grid(2, 2)
mem_line("tensors allocated")

# ── slice baseline ──
loss_slice(FL, ALu, ALd, M1, M2) = real(sum(W .* FLmap_parallel(
    FL, ALu, ALd, (M1, M2); ifparallel = true, forloop_iter = 1)))
r_slice = FLmap_parallel(FL, ALu, ALd, (M1, M2); ifparallel = true, forloop_iter = 1)
l_s, back_s = Zygote.pullback(loss_slice, FL, ALu, ALd, M1, M2)
g_s = back_s(1.0)
mem_line("slice fwd+bwd (peak retained)")
timed(() -> FLmap_parallel(FL, ALu, ALd, (M1, M2); ifparallel = true, forloop_iter = 1),
      "slice forward", 3)
timed(() -> Zygote.pullback(loss_slice, FL, ALu, ALd, M1, M2)[2](1.0),
      "slice fwd+bwd", 2)
GC.gc(); CUDA.reclaim(); mem_line("after reclaim")

# ── cannon ──
blk = cannon_scatter(FL, g)
loss_can(FL_b, ALu, ALd, M1, M2) = real(sum(W .* cannon_gather(
    FLmap_cannon(FL_b, ALu, ALd, (M1, M2), g), g)))
out_blk = FLmap_cannon(blk, ALu, ALd, (M1, M2), g)
r_can = cannon_gather(out_blk, g)
l_c, back_c = Zygote.pullback(loss_can, blk, ALu, ALd, M1, M2)
g_c = back_c(1.0)
mem_line("cannon fwd+bwd (peak retained)")
timed(() -> FLmap_cannon(blk, ALu, ALd, (M1, M2), g), "cannon forward", 3)
timed(() -> Zygote.pullback(loss_can, blk, ALu, ALd, M1, M2)[2](1.0),
      "cannon fwd+bwd", 2)

# ── parity ──
err_f = norm(r_can - r_slice) / norm(r_slice)
report(@sprintf("[parity] forward  rel err = %.2e  (PASS ≤ 1e-10: %s)",
                err_f, err_f <= 1e-10))
@assert l_c ≈ l_s rtol = 1e-10
# dFL: cannon returns the block gradient; compare against the slice of slice-path dFL
a_rs = split_ranges(χ, 2); i_rs = split_ranges(χ, 2)
dFL_slice_blk = g_s[1][a_rs[g.r1 + 1], :, :, i_rs[g.r2 + 1]]
errs = Float64[
    norm(g_c[1] - dFL_slice_blk) / norm(dFL_slice_blk),
    norm(g_c[2] - g_s[2]) / norm(g_s[2]),
    norm(g_c[3] - g_s[3]) / norm(g_s[3]),
    norm(g_c[4] - g_s[4]) / norm(g_s[4]),
    norm(g_c[5] - g_s[5]) / norm(g_s[5]),
]
err_max = MPI.Allreduce(maximum(errs), MPI.MAX, comm)
report(@sprintf("[parity] gradient max rel err = %.2e  (PASS ≤ 1e-8: %s)",
                err_max, err_max <= 1e-8))
report(err_f <= 1e-10 && err_max <= 1e-8 ? "=== RESULT: PASS ===" : "=== RESULT: FAIL ===")
```

**Step 2: Write the submit script** `examples/MPI_parallel/Sofia/submit_test_cannon.sh` (clone of `submit_test_nccl_compare.sh` reduced to 1 node / 4 GPUs; canonical v17 environment):

```bash
#!/bin/bash -l
#SBATCH --job-name=Sofia_cannon_test
#SBATCH --output=%x_%j.out
#SBATCH --partition=zen4_h200
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:nvidia_h200:4
#SBATCH --account=pilot_2026_0002

# Sofia 4-GPU Cannon FLmap validation: parity vs slice path + memory + timing.
# Driver: ../test_cannon_sofia.jl  Design: docs/2026-06-10-cannon-flmap-design.md

source /etc/profile
source /etc/profile.d/modules.sh
module load GDRCopy/2.4.4-GCCcore-14.2.0 \
            UCX-CUDA/1.18.0-GCCcore-14.2.0-CUDA-12.8.0 \
            OpenMPI/5.0.7-GCC-14.2.0 \
            CUDA/12.8.0 \
            NCCL/2.27.7-GCCcore-14.2.0-CUDA-12.8.0

WD=/sofia/scratch/pilot/pilot_2026_0002/xz
export JULIA_DEPOT_PATH=$WD/.julia
export HOME=$WD
JULIA=$WD/julia-1.11.3/bin/julia
CLEAN_LD=$(echo $LD_LIBRARY_PATH | tr ':' '\n' | grep -v 'CUDA/12.8.0' | tr '\n' ':')

BASE_ENVS="export CUDA_VISIBLE_DEVICES=\$OMPI_COMM_WORLD_LOCAL_RANK; \
export UCX_TLS=rc_x,self,sm,cuda_copy,cuda_ipc; \
export UCX_MEMTYPE_CACHE=n; \
export UCX_WARN_UNUSED_ENV_VARS=n; \
export CUDA_LAUNCH_BLOCKING=1; \
export LD_PRELOAD=/usr/lib64/libcuda.so.1"

echo "=== Sofia 4-GPU Cannon FLmap test ==="
echo "Start: $(date)"
mpirun -np 4 -x UCX_MODULE_DIR -x LD_LIBRARY_PATH=$CLEAN_LD -x PATH -x HOME -x JULIA_DEPOT_PATH \
    bash -c "$BASE_ENVS; exec $JULIA --project=../../.. ../test_cannon_sofia.jl"
echo "=== Done: $(date) ==="
```

**Step 3: Local smoke check of the driver logic** (CPU, replace CuArray→Array mentally — do NOT run the driver locally; just verify it parses):

Run: `julia --project=. -e "include_string(Main, replace(read(\"examples/MPI_parallel/test_cannon_sofia.jl\", String), r\"^using\"=>\"#using\"; count=1)); println(\"parse ok\")"` — simpler alternative: `julia --project=. -e "Meta.parseall(read(\"examples/MPI_parallel/test_cannon_sofia.jl\", String)); println(\"parse ok\")"`
Expected: `parse ok`

**Step 4: Commit**

```powershell
git add examples/MPI_parallel/test_cannon_sofia.jl examples/MPI_parallel/Sofia/submit_test_cannon.sh
git commit -m "feat: Sofia 4-GPU Cannon FLmap validation driver + submit script"
```

---

## Task 10: Run on Sofia and record results

**Files:**
- Modify (results): `docs/2026-06-10-cannon-flmap-design.md` (append a Results section)

**Step 1: Push the branch**

```powershell
git push origin claude/sad-saha-3ec6bf
```

**Step 2: Sync to Sofia and submit.** Sofia access: `wsl -d Ubuntu -u xingzhan -e ssh sofia '<cmd>'`. Locate the repo checkout under `/sofia/scratch/pilot/pilot_2026_0002/xz` (e.g. `ls` it first), then:

```bash
cd <repo-on-sofia>
git fetch origin claude/sad-saha-3ec6bf && git checkout claude/sad-saha-3ec6bf
cd examples/MPI_parallel/Sofia
sbatch submit_test_cannon.sh
```

Alternatively drive this step with the `/hpc` skill (SUBMIT mode handles sync+submit+ledger).

**Step 3: Monitor** — per established cadence: first check 5 min after submit, then 15 min while PENDING, 30 min while RUNNING. Job should finish well inside 30 min walltime.

**Step 4: Verify output** — `Sofia_cannon_test_<jobid>.out` must contain:

```
[parity] forward  rel err = ...  (PASS ≤ 1e-10: true)
[parity] gradient max rel err = ...  (PASS ≤ 1e-8: true)
=== RESULT: PASS ===
```

plus `[mem]` and `[time]` lines for both paths. Expected memory picture at map level (be honest in the report): transients are serial/P for **both** paths; Cannon's map-level saving is the full-vs-block FL input + result (≈ 384 MB/rank at χ=400 D=10); the big wins (eigsolver copies, AD tape) arrive with the leftenv integration round. Timing expectation: cannon ≤ ~1.5× slice per call at 2×2.

**Step 5: Optional larger size** — resubmit with `TENET_CANNON_CHI=800` exported in the script's `BASE_ENVS` to make the full-vs-block difference more visible (FL full = 1.6 GB at χ=800... still H200-comfortable).

**Step 6: Record results** — append a `## Results (2026-06-XX, Sofia 4×H200)` section to `docs/2026-06-10-cannon-flmap-design.md` with the parity numbers, memory lines, and timing table. Commit:

```powershell
git add docs/2026-06-10-cannon-flmap-design.md
git commit -m "docs: Sofia 4-GPU Cannon FLmap validation results"
```

---

## Out of scope (do not implement)

leftenv/simple_eig integration, AL tensor distribution, NCCL send/recv fast path, leg4/leg8 kernels, 8×8 cross-node runs. If a task seems to require one of these, stop and surface it instead.
