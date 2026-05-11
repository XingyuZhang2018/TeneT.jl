# 2D Distributed VUMPS Runtime Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Full 2D block distribution of all 4-leg tensors (FL/FR/AL/AR/AC) on an N1×N2 MPI Cartesian grid, with each rank holding only `(χ/N1, D, D, χ/N2)` slabs at all times, including during map computations.

**Architecture:** 4 new MPI communication primitives (`allgather_dim`, `reduce_scatter_dim`, `allreduce_dim`, `alltoall_dim_swap`) compose into per-map dataflows (FLmap/FRmap/ACmap/Cmap). Each primitive has a paired AD rrule. Higher-level functions (leftenv/rightenv/Cenv/ACenv/vumps_step) are modified to thread 2D tensors throughout; QR (ACCtoAL/AR) and observables (ObsEnv) use selective allgather wrappers in v1.

**Tech Stack:** Julia 1.10+, MPI.jl, CUDA.jl, ChainRulesCore.jl, Zygote.jl, KrylovKit.jl (bypassed in favor of `simple_eig` for v1). Hardware: H200 / GH200 GPU clusters (Sofia / JSC).

**Reference:** `docs/2026-05-11-2d-distributed-vumps-runtime-design.md`

---

## Pre-flight

### Task 0: Worktree setup

**Goal:** Isolated worktree to avoid conflicting with `iPEPS-unified` development.

**Step 1:** Create worktree

```bash
cd "D:/1 - research/1.26 - iPEPS_opt/TeneT.jl"
git worktree add ../TeneT.jl-2d-distributed -b feat/2d-distributed-runtime iPEPS-unified
cd ../TeneT.jl-2d-distributed
```

**Step 2:** Copy Manifest.toml (gitignored per memory note `feedback_julia_worktree_manifest.md`)

```bash
cp "../TeneT.jl/Manifest.toml" .
```

**Step 3:** Verify Julia env

```bash
julia --project=. -e 'using Pkg; Pkg.status()'
```

Expected: No errors, packages match main branch.

**Step 4:** Mark worktree

```bash
echo "feat/2d-distributed-runtime" > .worktree-tag
git add .worktree-tag && git commit -m "chore: tag worktree for 2D distributed VUMPS work"
```

---

## Phase 0: Sandbox Sanity (0.5 week, 4 tasks)

**Goal:** Catch design risks BEFORE touching production code. Standalone scripts, no source modifications.

### Task 0.1: Bare 4-rank MPI Cart grid sanity

**Files:**
- Create: `sandbox/2d_grid_sanity.jl`

**Step 1: Write the script**

```julia
# sandbox/2d_grid_sanity.jl
using MPI

MPI.Init()
const N = MPI.Comm_size(MPI.COMM_WORLD)
const rank = MPI.Comm_rank(MPI.COMM_WORLD)

N1, N2 = 2, 2  # require N=4 for this sanity
@assert N == 4 "Run with mpirun -n 4"

cart_comm = MPI.Cart_create(MPI.COMM_WORLD, [N1, N2]; periodic=[false, false])
coords = MPI.Cart_coords(cart_comm)
r1, r2 = coords[1], coords[2]

# Row comm: vary r2, fix r1
row_comm = MPI.Cart_sub(cart_comm, [false, true])
# Col comm: vary r1, fix r2
col_comm = MPI.Cart_sub(cart_comm, [true, false])

@info "rank=$rank (r1=$r1, r2=$r2) row_size=$(MPI.Comm_size(row_comm)) col_size=$(MPI.Comm_size(col_comm))"
MPI.Barrier(MPI.COMM_WORLD)
MPI.Finalize()
```

**Step 2: Run on Sofia (or any 4-GPU node)**

```bash
mpirun -n 4 julia --project=. sandbox/2d_grid_sanity.jl
```

Expected output (order may vary):
```
rank=0 (r1=0, r2=0) row_size=2 col_size=2
rank=1 (r1=0, r2=1) row_size=2 col_size=2
rank=2 (r1=1, r2=0) row_size=2 col_size=2
rank=3 (r1=1, r2=1) row_size=2 col_size=2
```

**Step 3: Commit**

```bash
git add sandbox/2d_grid_sanity.jl
git commit -m "sandbox: 2D MPI Cart grid sanity"
```

### Task 0.2: AllGather along sub-comm forward correctness

**Files:**
- Create: `sandbox/2d_allgather_sanity.jl`

**Step 1: Write script**

```julia
using MPI, Test
include("2d_grid_sanity.jl")  # for grid setup

# Each rank has a (2, 2) block; full = (4, 4) along last dim of col_comm
χ_local = 2
data = fill(Float64(rank), χ_local, χ_local)

# AllGather along col_comm (varies r1, gathers first dim)
counts = Cint[χ_local * χ_local for _ in 1:N1]
buf = similar(data, χ_local * N1, χ_local)
buf[r1*χ_local+1 : (r1+1)*χ_local, :] = data
MPI.Allgatherv!(buf, counts, col_comm)

# Verify: at rank (r1, r2), buf should contain data for r1=0..N1-1 stacked
expected = zeros(χ_local * N1, χ_local)
for r1_other in 0:N1-1
    expected[r1_other*χ_local+1 : (r1_other+1)*χ_local, :] .= Float64(rank_at(r1_other, r2))
end
@test buf ≈ expected
@info "rank=$rank: allgather along col_comm OK"
```

**Step 2:** `mpirun -n 4 julia --project=. sandbox/2d_allgather_sanity.jl`

Expected: 4 lines of "allgather along col_comm OK", no test failures.

**Step 3: Commit**

```bash
git add sandbox/2d_allgather_sanity.jl
git commit -m "sandbox: verify allgather along sub-comm"
```

### Task 0.3: All-to-all dim swap forward (the risky one)

**Files:**
- Create: `sandbox/2d_alltoall_sanity.jl`

**Step 1: Write script** — implement the cross-axis dim transpose manually with `MPI.Alltoallv!` for the N1=N2=2 case.

```julia
# Each rank (r1, r2) has tensor T of shape (4, 4) with chi=8 distributed.
# Initial: first dim on N1 (r1 owns rows [r1*2:r1*2+1])
#          second dim on N2 (r2 owns cols [r2*2:r2*2+1])
# Want to swap: after, first dim on N2, second dim on N1

χ_full = 8
χ_per_rank = 4  # χ_full / N1 (or N2, since N1=N2)

# Initial data: T[i, j] = i * 100 + j (global indices)
i_offset = r1 * χ_per_rank
j_offset = r2 * χ_per_rank
T = [Float64((i_offset + i_local) * 100 + (j_offset + j_local))
     for i_local in 0:χ_per_rank-1, j_local in 0:χ_per_rank-1]

# Now do alltoall on row_comm (varies r2): each rank sends its r2-slice to be received by other r2 ranks
# After swap: dst rank (r1, r2) wants T[i for i_slice_r2, j for j_slice_r1]
# ...

# Verify final T[i_local, j_local] = ((r2*χ_per_rank + i_local) * 100 + (r1*χ_per_rank + j_local))
@test ...
```

(Detailed Alltoallv implementation deferred to Task 1.9 in real code; this sandbox just proves the pattern works for the simplest case.)

**Step 2:** Run, verify, debug as needed. This is the highest-risk primitive — surfacing edge cases early.

**Step 3:** Commit

### Task 0.4: Gradient through one primitive (Zygote sanity)

**Files:**
- Create: `sandbox/2d_grad_sanity.jl`

**Step 1:** Write a toy that runs Zygote.gradient through `allgather` followed by `sum(abs2)`. Verify gradient flows back to the local slice. No MPI needed (mock with 1-rank case).

```julia
using Zygote, ChainRulesCore

# Mock allgather (1-rank trivial)
my_allgather(x) = x  # identity for 1 rank

function ChainRulesCore.rrule(::typeof(my_allgather), x)
    return x, d -> (NoTangent(), d)
end

x = rand(4)
g = Zygote.gradient(x -> sum(abs2, my_allgather(x)), x)[1]
@test g ≈ 2x
```

**Step 2-3:** Verify and commit.

**Phase 0 deliverable:** All 4 sandbox scripts run cleanly on Sofia 4-GPU node. **Bail-out signal:** if `alltoall_sanity.jl` is too hard to write for rectangular grids (N1 ≠ N2), restrict v1 to N1 = N2 perfect square.

---

## Phase 1: Primitives + AD rrules (3 weeks, 18 tasks)

### Task 1.1: `Cart2DGrid` struct + alg field

**Files:**
- Create: `src/contraction/cart2d_grid.jl`
- Modify: `src/boundary_algorithm/interface.jl` (VUMPS alg struct, add `N1::Int, N2::Int`)
- Modify: `src/TeneT.jl` to include new file

**Step 1: Write the struct**

```julia
# src/contraction/cart2d_grid.jl
struct Cart2DGrid
    N1::Int
    N2::Int
    world::MPI.Comm
    cart::MPI.Comm
    row_comm::MPI.Comm  # varies r2 (size N2)
    col_comm::MPI.Comm  # varies r1 (size N1)
    r1::Int
    r2::Int
end

function Cart2DGrid(N1::Int, N2::Int, world=MPI.COMM_WORLD)
    @assert N1 * N2 == MPI.Comm_size(world) "N1*N2 must equal world size"
    cart = MPI.Cart_create(world, [N1, N2]; periodic=[false, false])
    coords = MPI.Cart_coords(cart)
    r1, r2 = coords[1], coords[2]
    row_comm = MPI.Cart_sub(cart, [false, true])
    col_comm = MPI.Cart_sub(cart, [true, false])
    return Cart2DGrid(N1, N2, world, cart, row_comm, col_comm, r1, r2)
end

# Singleton: when N1=N2=1 (serial mode)
function Cart2DGrid()
    Cart2DGrid(1, 1)
end
```

**Step 2: Write test**

```julia
# test/cart2d_grid_test.jl
using Test, MPI, TeneT

@testset "Cart2DGrid setup" begin
    MPI.Initialized() || MPI.Init()
    grid = Cart2DGrid(2, 2)
    @test grid.N1 == 2 && grid.N2 == 2
    @test MPI.Comm_size(grid.row_comm) == 2
    @test MPI.Comm_size(grid.col_comm) == 2
end
```

**Step 3: Run test**

```bash
mpirun -n 4 julia --project=. test/cart2d_grid_test.jl
```

Expected: all tests pass.

**Step 4: Commit**

```bash
git add src/contraction/cart2d_grid.jl src/TeneT.jl test/cart2d_grid_test.jl
git commit -m "feat(2d): Cart2DGrid struct for N1xN2 MPI grid"
```

### Task 1.2: VUMPS alg field extension

**Files:**
- Modify: `src/boundary_algorithm/interface.jl`

**Step 1: Add fields to VUMPS struct**

Locate `struct VUMPS` definition. Add:
```julia
    N1::Int = 1
    N2::Int = 1
    grid::Union{Cart2DGrid, Nothing} = nothing
```

**Step 2: Default grid construction**

Add in `VUMPS` constructor: if `ifparallel && grid === nothing`, infer from MPI size:
```julia
if alg.ifparallel && alg.N1 * alg.N2 != 1
    grid = Cart2DGrid(alg.N1, alg.N2)
end
```

**Step 3: Test backward compat**

```bash
julia --project=. -e 'using TeneT; alg = VUMPS(...); @assert alg.N1 == 1'
```

Existing single-GPU code paths unchanged when N1=N2=1.

**Step 4: Commit**

```bash
git commit -am "feat(2d): VUMPS alg gains N1/N2/grid fields"
```

### Task 1.3: `allgather_dim` primitive forward

**Files:**
- Modify: `src/contraction/forloop_parallel_MPI.jl`

**Step 1: Write the failing test first** (TDD)

```julia
# test/distributed_primitives_test.jl
@testset "allgather_dim forward" begin
    grid = Cart2DGrid(2, 2)
    χ_local = 4
    # Each rank's local data depends on r1
    local_data = fill(Float64(grid.r1 + 1), χ_local, 8)
    
    # Gather along col_comm (size N1=2): output should have full first dim
    full = allgather_dim(local_data, 1, grid.col_comm)
    @test size(full) == (8, 8)
    # First 4 rows should be 1.0 (from r1=0), last 4 rows 2.0 (from r1=1)
    @test all(full[1:4, :] .== 1.0)
    @test all(full[5:8, :] .== 2.0)
end
```

**Step 2: Run, verify FAIL**

```bash
mpirun -n 4 julia --project=. test/distributed_primitives_test.jl
```

Expected: ERROR: `allgather_dim` not defined.

**Step 3: Implement**

```julia
# Append to src/contraction/forloop_parallel_MPI.jl

"""
    allgather_dim(tensor_local, dim, comm) -> tensor_full

Collect `tensor_local`'s `dim`-th dimension across all ranks in `comm`, producing
a tensor with the full extent on that dimension. Other dimensions unchanged.

For equal-size partitions only (use split_ranges variant for uneven).
"""
function allgather_dim(tensor_local::AbstractArray{T,N}, dim::Int, comm) where {T,N}
    M = MPI.Comm_size(comm)
    M == 1 && return copy(tensor_local)
    
    rank = MPI.Comm_rank(comm)
    χ_local = size(tensor_local, dim)
    χ_full = M * χ_local
    full_shape = ntuple(d -> d == dim ? χ_full : size(tensor_local, d), N)
    
    result = similar(tensor_local, full_shape)
    local_range = (rank * χ_local + 1):((rank + 1) * χ_local)
    idx = ntuple(d -> d == dim ? local_range : (:), N)
    result[idx...] = tensor_local
    synchronize(tensor_local)  # GPU sync
    
    elem_count = prod(size(tensor_local))
    counts = Cint[elem_count for _ in 1:M]
    allgatherv_p2p!(result, counts, comm)
    return result
end
```

**Step 4: Run test, verify PASS**

```bash
mpirun -n 4 julia --project=. test/distributed_primitives_test.jl
```

Expected: 1 test passed.

**Step 5: Commit**

```bash
git add src/contraction/forloop_parallel_MPI.jl test/distributed_primitives_test.jl
git commit -m "feat(2d): allgather_dim primitive"
```

### Task 1.4: `allgather_dim` rrule

**Step 1: Write failing test**

```julia
# Append to test/distributed_primitives_test.jl
@testset "allgather_dim gradcheck" begin
    grid = Cart2DGrid(2, 2)
    χ_local = 4
    x = rand(Float64, χ_local, 8)
    
    # Loss: sum of squares of allgathered result
    loss(x) = sum(abs2, allgather_dim(x, 1, grid.col_comm))
    
    g_zygote = Zygote.gradient(loss, x)[1]
    
    # Finite difference
    g_fd = similar(x)
    eps = 1e-6
    for i in eachindex(x)
        x_plus = copy(x); x_plus[i] += eps
        x_minus = copy(x); x_minus[i] -= eps
        g_fd[i] = (loss(x_plus) - loss(x_minus)) / (2eps)
    end
    
    @test maximum(abs.(g_zygote .- g_fd) ./ (abs.(g_fd) .+ 1e-10)) < 1e-5
end
```

**Step 2: Run, verify FAIL** (no rrule defined → may error or give wrong gradient).

**Step 3: Implement rrule in `src/autodiff/rules.jl`**

```julia
function ChainRulesCore.rrule(::typeof(allgather_dim), tensor_local, dim::Int, comm)
    result = allgather_dim(tensor_local, dim, comm)
    function back(d_result)
        d_local = reduce_scatter_dim(unthunk(d_result), dim, comm)
        return NoTangent(), d_local, NoTangent(), NoTangent()
    end
    return result, back
end
```

(Depends on `reduce_scatter_dim` from Task 1.5 — implement that first.)

**Step 4: Run test, verify PASS** (after Task 1.5 done).

**Step 5: Commit**

```bash
git commit -am "feat(2d): allgather_dim rrule via reduce_scatter_dim"
```

### Task 1.5: `reduce_scatter_dim` primitive + rrule

**Files:** Same as 1.3.

**Step 1: Write failing test**

```julia
@testset "reduce_scatter_dim forward" begin
    grid = Cart2DGrid(2, 2)
    χ_full = 8
    # All ranks start with same full tensor
    full = ones(Float64, χ_full, 8)
    
    # Reduce-scatter along col_comm: each rank sums + keeps own slice
    local_data = reduce_scatter_dim(full, 1, grid.col_comm)
    @test size(local_data) == (4, 8)
    # Sum over 2 ranks of 1s = 2.0
    @test all(local_data .== 2.0)
end
```

**Step 2: Run, verify FAIL.**

**Step 3: Implement**

```julia
function reduce_scatter_dim(tensor_full::AbstractArray{T,N}, dim::Int, comm) where {T,N}
    M = MPI.Comm_size(comm)
    M == 1 && return copy(tensor_full)
    
    # Simple v1 implementation: allreduce then slice
    # TODO v2: use MPI.Reduce_scatter! for true O(X/N) comm
    allreduce_p2p!(tensor_full, +, comm)
    
    rank = MPI.Comm_rank(comm)
    χ_full = size(tensor_full, dim)
    χ_local = χ_full ÷ M
    local_range = (rank * χ_local + 1):((rank + 1) * χ_local)
    idx = ntuple(d -> d == dim ? local_range : (:), N)
    return tensor_full[idx...]
end

function ChainRulesCore.rrule(::typeof(reduce_scatter_dim), tensor_full, dim, comm)
    result = reduce_scatter_dim(tensor_full, dim, comm)
    function back(d_result)
        d_full = allgather_dim(unthunk(d_result), dim, comm)
        return NoTangent(), d_full, NoTangent(), NoTangent()
    end
    return result, back
end
```

**Step 4: Run forward + gradcheck tests, verify PASS.**

**Step 5: Commit**

```bash
git commit -am "feat(2d): reduce_scatter_dim primitive + rrule"
```

### Task 1.6: `allreduce_dim` primitive + rrule

**Files:** Same as 1.3.

**Step 1: Write failing test**

```julia
@testset "allreduce_dim sum is self-adjoint" begin
    grid = Cart2DGrid(2, 2)
    x = fill(Float64(grid.r1 + 1), 4, 8)
    
    summed = allreduce_dim(x, +, grid.col_comm)
    @test all(summed .== 3.0)  # 1.0 + 2.0
    
    # Gradcheck: ∂(sum(abs2, allreduce(x)))/∂x at each rank = 2 * allreduce(x)
    # Since allreduce(+)) is self-adjoint, backward of allreduce(d_y) = allreduce(d_y)
    loss(x) = sum(abs2, allreduce_dim(x, +, grid.col_comm))
    g = Zygote.gradient(loss, x)[1]
    @test g ≈ 2 * summed  # each rank sees gradient 2*summed
end
```

**Step 2-3:** Implement

```julia
function allreduce_dim(tensor, op, comm)
    MPI.Comm_size(comm) == 1 && return copy(tensor)
    result = copy(tensor)
    allreduce_p2p!(result, op, comm)
    return result
end

function ChainRulesCore.rrule(::typeof(allreduce_dim), tensor, op::typeof(+), comm)
    result = allreduce_dim(tensor, op, comm)
    function back(d_result)
        d_tensor = allreduce_dim(unthunk(d_result), +, comm)
        return NoTangent(), d_tensor, NoTangent(), NoTangent()
    end
    return result, back
end
```

**Step 4-5:** Verify, commit.

### Task 1.7: `alltoall_dim_swap` — uniform partition special case

**Files:** Same as 1.3.

**Step 1: Write failing test** for N1 = N2 case (e.g., 2x2 grid):

```julia
@testset "alltoall_dim_swap N1=N2 case" begin
    grid = Cart2DGrid(2, 2)
    χ_per_rank = 4
    # Initial: tensor with first dim on row_comm (varies r2)
    T = [Float64((grid.r2 * χ_per_rank + i) * 100 + (grid.r1 * χ_per_rank + j))
         for i in 0:χ_per_rank-1, j in 0:χ_per_rank-1]
    
    # Swap: first dim goes from row_comm to col_comm
    T_swapped = alltoall_dim_swap(T, 1, grid.row_comm, 1, grid.col_comm)
    
    # After swap, first dim indexed by r1 instead of r2
    expected = [Float64((grid.r1 * χ_per_rank + i) * 100 + (grid.r2 * χ_per_rank + j))
                for i in 0:χ_per_rank-1, j in 0:χ_per_rank-1]
    @test T_swapped ≈ expected
end
```

**Step 2: Run, verify FAIL.**

**Step 3: Implement (uniform partition, equal-size chunks)**

```julia
function alltoall_dim_swap(tensor::AbstractArray{T,N}, 
                          src_dim::Int, src_comm,
                          dst_dim::Int, dst_comm) where {T,N}
    # v1: assume N_src == N_dst (square grid) and uniform chunk size
    N_src = MPI.Comm_size(src_comm)
    N_dst = MPI.Comm_size(dst_comm)
    
    if N_src == 1 && N_dst == 1
        return copy(tensor)
    end
    
    @assert N_src == N_dst "v1 requires square grid; got src_comm size $N_src != dst_comm size $N_dst"
    
    χ_src_local = size(tensor, src_dim)
    χ_dst_local = χ_src_local  # for square grid
    
    # Reshape tensor so src_dim is split into N_dst chunks
    # Each chunk → send to corresponding rank in src_comm
    # Receive N_src chunks → concatenate along src_dim (or dst_dim, depending on convention)
    
    # Build sendbuf: chunks contiguous in memory
    chunk_size = χ_src_local ÷ N_dst
    @assert chunk_size * N_dst == χ_src_local "uneven partition not yet supported (Task 1.8)"
    
    other_dims = prod(size(tensor)) ÷ χ_src_local
    sendbuf = similar(tensor, prod(size(tensor)))
    # Reorder so dst-rank-r chunk is contiguous
    # ... reshape + permutedims as needed ...
    
    recvbuf = similar(sendbuf)
    counts = Cint[chunk_size * other_dims for _ in 1:N_src]
    MPI.Alltoallv!(sendbuf, counts, recvbuf, counts, src_comm)
    
    # Reshape recvbuf back to (..., χ_dst_local, ...)
    output_shape = ntuple(d -> d == src_dim ? χ_dst_local : size(tensor, d), N)
    result = reshape(recvbuf, output_shape)
    return result
end
```

**Note:** This skeleton glosses over the reshape/permutedims dance. Full implementation in actual coding: ~80-120 lines. **Allocate full week of dev time to this task.**

**Step 4: Run test, debug, verify PASS.**

**Step 5: Commit**

```bash
git commit -am "feat(2d): alltoall_dim_swap for square grid (N1=N2)"
```

### Task 1.8: `alltoall_dim_swap` — rectangular grid (N1 ≠ N2)

**Files:** Same as 1.3.

**Step 1: Write failing test for 2x4 grid** (N1=2, N2=4, χ=8):

```julia
@testset "alltoall_dim_swap N1=2 N2=4" begin
    @assert MPI.Comm_size(MPI.COMM_WORLD) == 8 "Run with mpirun -n 8"
    grid = Cart2DGrid(2, 4)
    χ_per_N1 = 4  # χ/N1 = 8/2
    χ_per_N2 = 2  # χ/N2 = 8/4
    # ... similar test pattern, with rectangular slices ...
end
```

**Step 2-4: Implement with `Alltoallv!` using non-uniform counts/displacements** via `split_ranges(χ_full, max(N1, N2))` to get unified atomic chunks.

**Step 5: Commit**

### Task 1.9: `alltoall_dim_swap` rrule

**Step 1: Write gradcheck test.**

**Step 2-4:** Implement rrule (self-inverse: backward swaps dst→src).

```julia
function ChainRulesCore.rrule(::typeof(alltoall_dim_swap),
                              tensor, src_dim, src_comm, dst_dim, dst_comm)
    result = alltoall_dim_swap(tensor, src_dim, src_comm, dst_dim, dst_comm)
    function back(d_result)
        d_tensor = alltoall_dim_swap(unthunk(d_result),
                                     dst_dim, dst_comm,
                                     src_dim, src_comm)
        return NoTangent(), d_tensor, NoTangent(), NoTangent(), NoTangent()
    end
    return result, back
end
```

**Step 5:** Commit.

### Task 1.10: `prescatter_2d` / `allgather_2d` for raw arrays

**Files:** Same.

**Step 1: Failing test** — roundtrip `allgather_2d(prescatter_2d(full)) ≈ full`.

**Step 2-4: Implement**

```julia
"""
    prescatter_2d(tensor_full, grid) -> tensor_local

Take a full tensor (χ, D, D, χ) and return the local block (χ/N1, D, D, χ/N2)
for this rank.
"""
function prescatter_2d(tensor_full::AbstractArray, grid::Cart2DGrid)
    χ1, _, _, χ2 = size(tensor_full)
    # Use existing split_ranges for non-uniform handling
    r1_ranges = split_ranges(χ1, grid.N1)
    r2_ranges = split_ranges(χ2, grid.N2)
    return tensor_full[r1_ranges[grid.r1+1], :, :, r2_ranges[grid.r2+1]]
end

function allgather_2d(tensor_local::AbstractArray, grid::Cart2DGrid)
    # First gather along row_comm (last χ), then col_comm (first χ)
    after_row = allgather_dim(tensor_local, ndims(tensor_local), grid.row_comm)
    return allgather_dim(after_row, 1, grid.col_comm)
end
```

**Step 5:** Commit.

### Task 1.11: StructArray-level wrappers `prescatter_2d_struct` / `allgather_2d_struct`

Wraps prior task to operate on each unique data element of a StructArray.

**Step 1-5:** Similar pattern, test with multi-site unit cells (Plaquette 4-site).

### Task 1.12: `distributed_dot` and `distributed_norm`

**Files:**
- Create: `src/utils/distributed_linalg.jl`
- Modify: `src/TeneT.jl`

**Step 1: Failing test**

```julia
@testset "distributed_dot" begin
    grid = Cart2DGrid(2, 2)
    χ_local = 4
    # Two identical distributed tensors
    x_local = fill(2.0, χ_local, 8, 8, χ_local)
    y_local = fill(3.0, χ_local, 8, 8, χ_local)
    
    # Wrap in StructArray (1-site)
    x = StructArray([x_local], [1])
    y = StructArray([y_local], [1])
    
    d = distributed_dot(x, y, grid.world)
    # Global: sum over all elements of x.*y = 2*3 * χ_full² * D² = 6 * 64 * 64
    expected = 6.0 * 8 * 8 * 8 * 8
    @test d ≈ expected
end
```

**Step 2-4: Implement**

```julia
# src/utils/distributed_linalg.jl
function distributed_dot(x::StructArray, y::StructArray, comm=MPI.COMM_WORLD)
    @assert x.pattern == y.pattern "patterns must match for distributed_dot"
    
    seen = Set{Int}()
    T = eltype(x.data[1])
    local_acc = zero(T)
    for idx in eachindex(x.pattern)
        p = x.pattern[idx]
        p in seen && continue
        push!(seen, p)
        local_acc += dot(x.data[p], y.data[p])
    end
    return MPI.Allreduce(local_acc, +, comm)
end

distributed_norm(x::StructArray, comm=MPI.COMM_WORLD) = 
    sqrt(real(distributed_dot(x, x, comm)))
```

**Step 5:** Commit.

### Task 1.13: rrule for `distributed_dot`

**Step 1-5:** Standard `dot` rrule, but the AllReduce is part of forward. Backward returns `d_y * d_dot_result` and `d_x * d_dot_result` on each rank (no further reduction since allreduce already done in forward).

### Task 1.14: `simple_eig` extension for distributed inner products

**Files:**
- Modify: `src/utils/misc.jl`

**Step 1: Failing test** — verify `simple_eig` reaches same eigenvalue with `inner_product=distributed_dot, norm_fn=distributed_norm` vs serial dot/norm.

**Step 2-4: Add kwargs to `simple_eig`**

Locate `function simple_eig(f, v; power_iter, ...)` in `src/utils/misc.jl`. Add kwargs:

```julia
function simple_eig(f, v; power_iter, ...,
                    inner_product = LinearAlgebra.dot,
                    norm_fn = LinearAlgebra.norm)
    # ... replace all `norm(v)` with `norm_fn(v)`
    # ... replace all `dot(v1, v2)` with `inner_product(v1, v2)`
end
```

**Step 5:** Commit.

### Task 1.15: `_simple_eig_FLmap` (and 3 siblings) thread the distributed inner products

**Files:**
- Modify: `src/boundary_algorithm/vumps/general.jl`
- Modify: `src/boundary_algorithm/vumps/c4v.jl`
- Modify: `src/boundary_algorithm/vumps/plaquette.jl`

In each `_simple_eig_*map_*` function, detect 2D mode via `alg.grid !== nothing` and pass:

```julia
if alg.grid !== nothing
    return simple_eig(f, x; power_iter, segment_checkpoint,
                      inner_product = (a, b) -> distributed_dot(a, b, alg.grid.world),
                      norm_fn = x -> distributed_norm(x, alg.grid.world))
else
    return simple_eig(f, x; power_iter, segment_checkpoint)
end
```

**Step 1-5:** TDD pattern; commit.

### Task 1.16-1.18: Phase 1 integration tests

- 1.16: Run all primitive tests on (2,2) grid → all PASS
- 1.17: Run on (2,4) and (4,2) rectangular → all PASS  
- 1.18: Run on (4,4) and (1,4) edge cases → all PASS

**Each test commits after PASS.**

**Phase 1 deliverable:** ~900 lines of primitive code + rrules, all gradchecked to 1e-5. CI runs on 4 GPUs.

---

## Phase 2: FLmap End-to-End (1.5 weeks, 8 tasks)

### Task 2.1: `FLmap_parallel_2D` forward (leg5 first, most common case)

**Files:**
- Modify: `src/contraction/forloop_parallel_MPI.jl`

**Step 1: Write failing test** — small problem, compare 2D output to serial FLmap.

```julia
@testset "FLmap_parallel_2D vs serial (leg5)" begin
    grid = Cart2DGrid(2, 2)
    χ_full = 8  # divisible by both N1, N2
    D = 2
    
    # Random full tensors
    FL_full = rand(ComplexF64, χ_full, D, D, χ_full)
    ALu_full = rand(ComplexF64, χ_full, D, D, χ_full)
    ALd_full = rand(ComplexF64, χ_full, D, D, χ_full)
    M1 = rand(ComplexF64, D, D, D, D, D)
    M2 = rand(ComplexF64, D, D, D, D, D)
    
    # Serial reference
    result_serial = FLmap(FL_full, ALu_full, ALd_full, M1, M2)
    
    # 2D path
    FL_local = prescatter_2d(FL_full, grid)
    ALu_local = prescatter_2d(ALu_full, grid)
    ALd_local = prescatter_2d(ALd_full, grid)
    
    result_local = FLmap_parallel_2D(FL_local, ALu_local, ALd_local, M1, M2; grid)
    result_full = allgather_2d(result_local, grid)
    
    @test result_full ≈ result_serial rtol=1e-12
end
```

**Step 2: Run, verify FAIL** (FLmap_parallel_2D not defined).

**Step 3: Implement (per design Section 2.1)**

```julia
function FLmap_parallel_2D(FL, ALu, ALd, M1, M2; grid::Cart2DGrid)
    # Step 1: AllGather ALd along col_comm (i = first χ, N1 axis)
    ALd_full_i = allgather_dim(ALd, 1, grid.col_comm)
    
    # Step 2: Local partial einsum
    # FL[a_N1, e, f, i_N2] * ALd_full_i[i_full, j, k, l_N2] * M1 * M2 * ALu[a_N1, b, c, d_N2]
    # → partial[d_N2, g, h, l_N2] (sums over a_slice, i, b, c, e, f, j, k, p)
    @tensor partial[d, g, h, l] := FL[a, e, f, i] * ALd_full_i[i, j, k, l] * 
                                    M1[e, j, g, b, p] * M2[f, k, h, c, p] * 
                                    ALu[a, b, c, d]
    
    # Step 3: AllReduce along col_comm (sum over a_slice contributions)
    partial_full_a = allreduce_dim(partial, +, grid.col_comm)
    
    # Step 4: AllToAll transpose d from N2 (row_comm) → N1 (col_comm)
    result = alltoall_dim_swap(partial_full_a, 1, grid.row_comm, 1, grid.col_comm)
    
    return result  # shape (d_slice_r1, g, h, l_slice_r2)
end
```

**Step 4: Run test, verify PASS.**

**Step 5: Commit**

### Task 2.2: FLmap leg4 and leg8 variants

Same pattern as 2.1, different `@tensor` contraction expressions matching `src/contraction/basic.jl::FLmap(...)` overloads.

### Task 2.3: FLmap gradient parity test

```julia
@testset "FLmap_parallel_2D gradient parity" begin
    # ... same setup as 2.1 ...
    
    loss_serial(FL) = sum(abs2, FLmap(FL, ALu_full, ALd_full, M1, M2))
    g_serial = Zygote.gradient(loss_serial, FL_full)[1]
    
    function loss_2d(FL_local)
        result = FLmap_parallel_2D(FL_local, ALu_local, ALd_local, M1, M2; grid)
        return real(distributed_dot(...))  # need to wrap for distributed sum
    end
    g_2d_local = Zygote.gradient(loss_2d, FL_local)[1]
    g_2d_full = allgather_2d(g_2d_local, grid)
    
    @test g_2d_full ≈ g_serial rtol=1e-8
end
```

**Phase 2 milestone: this test passing = AD chain through map primitives is correct.**

### Task 2.4: Wire FLmap_parallel_2D into `leftenv`

**Files:**
- Modify: `src/boundary_algorithm/vumps/general.jl`

In `leftenv`, dispatch:

```julia
f(FL1j) = checkpoint(inner_checkpoint, 
                     alg.grid !== nothing ? FLmap_parallel_2D : FLmap_parallel,
                     ...)
```

### Task 2.5: leftenv end-to-end test

```julia
@testset "leftenv 2D vs serial" begin
    # Random small problem
    # Run leftenv with simple_eig in both serial and 2D modes
    # Verify dominant eigenvalue matches to 1e-10
    # Verify eigenvector (allgathered) matches to 1e-8
end
```

### Task 2.6: leftenv gradient parity

```julia
@testset "leftenv 2D gradient parity" begin
    # Backward through full leftenv
    # ∂FL_out / ∂(ALu, M) compared between 2D and serial
end
```

**Tasks 2.7-2.8:** Edge cases (Nsites=2, N1≠N2, non-divisible χ).

**Phase 2 deliverable: "Distributed converges" — one full env solver works end-to-end with correct gradients.**

---

## Phase 3: FRmap, ACmap, Cmap (2 weeks)

Each map: 4 tasks following the FLmap pattern from Phase 2.

### Tasks 3.1-3.4: FRmap_parallel_2D
- 3.1: leg5 forward + test
- 3.2: leg4 + leg8 variants
- 3.3: rightenv integration
- 3.4: rightenv gradient parity

### Tasks 3.5-3.8: ACmap_parallel_2D
Same pattern, ACenv integration.

### Tasks 3.9-3.12: Cmap_parallel_2D (C full, only FL/FR distributed)
- Cenv integration
- Note: only Cmap returns full tensor (since C is full); other maps return 2D

**Phase 3 deliverable: all 4 maps + 4 env solvers work in 2D.**

---

## Phase 4: vumps_step Integration (2 weeks, 8 tasks)

### Task 4.1: `prescatter_2d_struct` / `allgather_2d_struct` polish (if not done in 1.11)

### Task 4.2: `init_VUMPSRuntime` 2D path

Modify `init_VUMPSRuntime` in `general.jl`:
- Generate full A on rank 0
- Broadcast
- `prescatter_2d_struct` to 2D
- Call `leftenv` / `rightenv` (now 2D)
- Return VUMPSRuntime with 2D inner tensors

### Task 4.3: `ALCtoAC_2d`

```julia
function ALCtoAC_2d(AL::StructArray, C::StructArray, grid)
    # AL is 2D (a on N1, d on N2), C is full
    # AC[a, b, c, d_new] = sum over r of AL[a, b, c, r] * C[r, d_new]
    
    # Each rank computes partial over r in r_slice_r2
    # AllReduce along row_comm gives full r
    # ...
end
```

### Task 4.4: `ACCtoAL_2d` / `ACCtoAR_2d` (allgather + local QR)

```julia
function ACCtoAL_2d(AC, C, grid)
    AC_full = allgather_2d_struct(AC, grid)
    AL_full, errL = ACCtoAL(AC_full, C)  # reuse existing serial QR
    AL_local = prescatter_2d_struct(AL_full, grid)
    return AL_local, errL
end
```

### Task 4.5: `vumps_step` 2D — clean version

Remove all `_allgather_FR` / `_prescatter_FR` cycles from PR #42 (since we don't have PR #42 merged — this branch is from before that PR's pattern was applied). Pure 2D throughout:

```julia
function vumps_step(rt, M, alg)
    @unpack AL, C, AR, FL, FR = rt
    AC = ALCtoAC(AL, C)
    _, FL = leftenv(AL, conj(AL), M, FL; alg)
    _, FR = rightenv(AR, conj(AR), M, FR; alg)
    _, C  = Cenv(C, FL, FR; alg)
    _, AC = ACenv(AC, FL, M, FR; alg)
    AL, AR, errL, errR = ACCtoALAR(AC, C)
    err = errL + errR
    return VUMPSRuntime(AL, AR, C, FL, FR), err
end
```

### Task 4.6: `vumps_step_power` 2D

### Task 4.7: `ObsEnv` 2D (FL/FR stay 2D throughout)

### Task 4.8: End-to-end vumps_step test

```julia
@testset "vumps_step convergence" begin
    M, χ = setup_small_problem(D=2, χ=4, Nsites=1)
    rt_serial = init_VUMPSRuntime(M, χ, alg_serial)
    rt_2d = init_VUMPSRuntime(M, χ, alg_2d)
    
    for step in 1:50
        rt_serial, _ = vumps_step(rt_serial, M, alg_serial)
        rt_2d, _ = vumps_step(rt_2d, M, alg_2d)
    end
    
    rt_2d_full = allgather_runtime(rt_2d)
    @test rt_2d_full.FL ≈ rt_serial.FL rtol=1e-8
    @test rt_2d_full.AC ≈ rt_serial.AC rtol=1e-8
end
```

**Phase 4 deliverable: "VUMPS correct" — full VUMPS solve converges to same state as serial.**

---

## Phase 5: LBFGS Gradient Parity (2 weeks)

### Task 5.1: Single-step gradient parity infrastructure

```julia
function compute_energy_2d(M, alg)
    rt = init_VUMPSRuntime(M, χ, alg)
    for _ in 1:N_vumps_steps
        rt, _ = vumps_step(rt, M, alg)
    end
    env = ObsEnv(rt, M, alg)
    return compute_energy(env, M)
end
```

### Task 5.2: 1e-8 gradient parity test

```julia
@testset "End-to-end gradient parity" begin
    M = random_iPEPS_tensor(D=2, Nsites=1)
    
    g_serial = Zygote.gradient(M -> compute_energy_2d(M, alg_serial), M)[1]
    g_2d = Zygote.gradient(M -> compute_energy_2d(M, alg_2d), M)[1]
    
    @test relative_error(g_2d, g_serial) < 1e-8
end
```

### Tasks 5.3-5.5: Debug ladder

Order of investigation if test fails:
1. simple_eig with distributed dot/norm — verify `inner_product` callback gets correct grid
2. Each primitive's rrule — re-run gradchecks individually
3. ACCtoAL allgather+QR rrule chain
4. ifsimple_eig=true is actually honored (not falling through to KrylovKit)
5. F32/F64 precision drift in inner_etype cast

**Allocate 1.5 weeks for debug. This is the highest-risk task.**

---

## Phase 6: Sofia Production Benchmark (1 week)

### Task 6.1: sbatch template for D=10 χ=540 Kagome (baseline reproduce)

**Files:**
- Create: `hpc/2d_distributed/submit_kagome_D10_chi540.sh`

```bash
#!/bin/bash
#SBATCH --job-name=2d_kagome_D10_chi540
#SBATCH --partition=zen4_h200
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --gpus-per-node=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
module load OpenMPI

# Sofia LD path: strip CUDA, see memory note reference_sofia_ld_path.md
export LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | tr ':' '\n' | grep -v cuda | tr '\n' ':' | sed 's/:$//')

# Sofia UCX/NCCL fix: see memory note project_sofia_hpc_ucx_fix.md
export LD_PRELOAD=/usr/lib64/libcuda.so.1
export OMPI_MCA_btl=^openib
export OMPI_MCA_pml=ucx

# NCCL fast path
export TENET_USE_NCCL=1

# CUDA visibility
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 2D grid configuration
export TENET_N1=2
export TENET_N2=4

cd /sofia/scratch/pilot/pilot_2026_0002/xz/TeneT.jl-2d-distributed

mpirun -n 8 \
  julia --project=. \
        examples/Heisenberg/Heisenberg_Kagome_merge_VUMPS_General.jl \
        --D 10 --chi 540 --N1 2 --N2 4 \
        --max_iter 100 \
        --ifload_env=false --ifload_lbfgs=false \
        --source=BSC_seed42_D8_No5_chi224 \
        --output=data_2d/D10_chi540/
```

**Step 2:** scp to Sofia, sbatch submit.

**Step 3:** Monitor — per memory note `feedback_monitoring_interval.md`: post-submit 5 min, PENDING 15 min, RUNNING 30 min.

**Step 4:** Verify NOT OOM (this is the 1003757 OOM case — 2D should pass).

### Task 6.2: sbatch for D=10 χ=768 (PR #42 OOM regime)

Same template, change `--chi 768`. Expect: success, since 2D peak ~1/√8 of baseline.

### Task 6.3: sbatch for D=10 χ=1000 (push 1.5x)

### Task 6.4: sbatch for D=16 χ=512 (D push, JSC GH200 reference)

### Task 6.5: Performance analysis script

```julia
# scripts/analyze_2d_benchmark.jl
function analyze_run(logdir)
    # Parse vumps_step wall-clock, NCCL profiling, nvidia-smi peak memory
    # Compare to baseline (1.25 archive: 1003757 / 1005525)
    # Output: data-inventory entry + plot
end
```

### Task 6.6: Update `data-inventory.md`

Use hpc-archive skill to record results into `D:\1 - research\1.26 - iPEPS_opt\docs\data-inventory.md`.

**Phase 6 deliverable: production benchmark report demonstrating 2D enables D=10 χ=1000+ on Sofia 8 GPU.**

---

## Phase 7: v2 Optimizations (deferred, 4 weeks if triggered)

Only execute these if Phase 6 reveals a specific bottleneck.

### Task 7.1: Distributed TSQR for ACCtoAL/AR (trigger: D≥20)

### Task 7.2: SUMMA-style compute-comm overlap (trigger: comm >20% step time)

### Task 7.3: Block-cyclic 2D partition (trigger: rank load imbalance >20%)

### Task 7.4: KrylovKit eigsolve with custom inner product (trigger: simple_eig accuracy insufficient)

(Each task gets full task breakdown if triggered.)

---

## Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| `alltoall_dim_swap` rectangular grid bugs | High | Phase 0 sandbox; allocate full week; restrict to N1=N2 if blocked |
| Phase 5 gradient parity test fails | Medium-high | 1.5 week debug buffer; ladder of likely causes documented |
| Krylov queue memory dominates over tensor peak | Medium | Phase 6 measures separately; LOBPCG in v2 if needed |
| ChainRulesCore version compatibility | Low | Existing PR #38 rrule patterns proven on this codebase |
| Sofia local NVMe path unknown | Low (deferred) | NVMe is v3, not blocking |

---

## Definition of Done

- [ ] All Phase 1-6 tasks committed to `feat/2d-distributed-runtime` branch
- [ ] CI green on Sofia 4-GPU (Phase 1-2) and 8-GPU (Phase 3+)
- [ ] Phase 5 gradient parity test passes at 1e-8
- [ ] Phase 6 benchmark report committed to `docs/2026-MM-DD-2d-benchmark-phase6-report.md`
- [ ] PR opened against `iPEPS-unified`, ready for code review
- [ ] data-inventory.md entry added via hpc-archive skill

---

## Execution Handoff

Plan complete and saved to `docs/2026-05-11-2d-distributed-vumps-runtime-plan.md`. Two execution options:

**1. Subagent-Driven (this session)** — I dispatch a fresh subagent per task, review between tasks, fast iteration. Best for Phase 0 sandbox and Phase 1 primitives where each task is well-scoped.

**2. Parallel Session (separate)** — Open a new session in the worktree (`D:/1 - research/1.26 - iPEPS_opt/TeneT.jl-2d-distributed/`) with `superpowers:executing-plans` skill, batch execute with checkpoints. Better for the long phases (3-5) where you want to step away.

**Which approach?**
