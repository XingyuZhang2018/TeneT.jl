# sandbox/2d_allgather_sanity.jl
#
# Phase 0 Task 0.2 — Allgatherv along a Cartesian sub-communicator.
#
# Purpose:
#   Verify that `MPI.Allgatherv!` along the `col_comm` sub-communicator from
#   Task 0.1 produces the data layout the 2D distributed VUMPS runtime will
#   rely on: when each rank contributes its own local `(χ_local, χ_local)`
#   block, the gathered buffer on every rank in column `r2` should contain
#   the blocks of all ranks (r1', r2) for r1' = 0..N1-1, ordered by r1'.
#
#   This sandbox exercises the bare collective in isolation — no production
#   code, no GPU buffers, no custom p2p path. It is the simplest possible
#   pre-flight check for the gather we will need to materialise the "full"
#   FL/FR boundary on a column before mixed canonicalisation.
#
# When to run:
#   Linux/macOS node with `mpirun -n 4` (e.g. Sofia HPC). Cannot run on
#   Windows. Invocations with N != N1*N2 trip the `@assert` and exit.
#
# How to run:
#   mpirun -n 4 julia --project=. sandbox/2d_allgather_sanity.jl
#
# Expected output (rank 0 prints one sorted summary; no interleaving):
#   [2D allgather sanity] N=4, dims=(N1=2, N2=2), χ_local=2
#     rank=0  (r1=0, r2=0)  col_rank=0  PASS
#     rank=1  (r1=0, r2=1)  col_rank=0  PASS
#     rank=2  (r1=1, r2=0)  col_rank=1  PASS
#     rank=3  (r1=1, r2=1)  col_rank=1  PASS
#   [2D allgather sanity] PASS
#
# Deviations from the original plan script (with rationale):
#   * Grid setup is inlined rather than `include`-ing 2d_grid_sanity.jl, since
#     that script calls `MPI.Init`/`MPI.Finalize` as a top-level driver and
#     re-including it would either re-init MPI or hang on the barrier. If
#     Phase 0 grows past ~3 sandbox scripts, refactor the grid setup into a
#     small module (TODO).
#   * The recv buffer is 3D — (χ_local, χ_local, N1) — instead of the plan's
#     2D (χ_local * N1, χ_local). Julia is column-major, and `Allgatherv!`
#     concatenates contiguous chunks; the plan's row-stacked 2D layout would
#     interleave bytes from different ranks in memory. A 3D recv buffer keeps
#     each rank's contribution contiguous and the verification trivial.
#   * `rank_at(r1', r2)` is implemented via `MPI.Cart_rank(cart_comm, …)`
#     rather than a hand-rolled formula, so it stays correct under any future
#     reordering of the cartesian comm.
#   * Uses `MPI.VBuffer(recvbuf, counts)` — the MPI.jl 0.20 signature is
#     `Allgatherv!(sendbuf, recvbuf::VBuffer, comm)`, NOT
#     `Allgatherv!(buf, counts, comm)` as written in the plan.

using MPI, Test

MPI.Init()

const comm = MPI.COMM_WORLD
const N    = MPI.Comm_size(comm)
const rank = MPI.Comm_rank(comm)

# Hard-coded 2x2 grid (matches Task 0.1).
const N1, N2 = 2, 2
@assert N == N1 * N2 "Run with `mpirun -n 4` (got N=$N; this sandbox requires N1*N2 = $(N1*N2))"

cart_comm = MPI.Cart_create(comm, [N1, N2]; periodic=[false, false])
coords    = MPI.Cart_coords(cart_comm)   # 0-based [r1, r2]
r1, r2    = coords[1], coords[2]

# Col comm: vary r1, fix r2  (the dimension we gather along — collects FL/FR
# slabs distributed over rows within a single column of the iPEPS grid).
col_comm = MPI.Cart_sub(cart_comm, [true, false])
col_rank = MPI.Comm_rank(col_comm)
col_size = MPI.Comm_size(col_comm)
@assert col_size == N1 "col_comm size $col_size != N1=$N1 on rank $rank"

# `Cart_sub` preserves the order of the remaining coordinates, so the col_comm
# rank of cartesian (r1', r2) is exactly r1'. We assert this invariant rather
# than rely on it implicitly — if it ever changed, the test below would still
# need to map col_rank → r1'.
@assert col_rank == r1 "col_rank=$col_rank disagrees with r1=$r1 on rank $rank"

# Resolve the global rank of an arbitrary cartesian coordinate (r1', r2).
rank_at(r1_other, r2_other) = MPI.Cart_rank(cart_comm, [r1_other, r2_other])

# ─── The gather ────────────────────────────────────────────────────────────
#
# Each rank owns one (χ_local, χ_local) block. We tag the block with the
# rank value so the receiver can verify which rank produced each chunk.
const χ_local = 2
sendbuf = fill(Float64(rank), χ_local, χ_local)

# 3D recv buffer: (χ_local, χ_local, N1). Slice [:, :, k+1] receives the
# block from col_comm rank k, which is cartesian (k, r2).
recvbuf = zeros(Float64, χ_local, χ_local, N1)

# Each rank sends exactly χ_local^2 Float64s. With a uniform recv layout
# `counts[k] = χ_local^2` for k = 1..N1.
counts = Cint[χ_local * χ_local for _ in 1:N1]
MPI.Allgatherv!(sendbuf, MPI.VBuffer(recvbuf, counts), col_comm)

# ─── Verify ────────────────────────────────────────────────────────────────
expected = zeros(Float64, χ_local, χ_local, N1)
for r1_other in 0:N1-1
    expected[:, :, r1_other + 1] .= Float64(rank_at(r1_other, r2))
end

local_pass = recvbuf == expected   # exact equality: all entries are integer-valued

# Gather one record per rank to rank 0 for a single ordered summary.
record = Int[rank, r1, r2, col_rank, local_pass ? 1 : 0]
all_records = MPI.Gather(record, comm; root=0)

if rank == 0
    println("[2D allgather sanity] N=$N, dims=(N1=$N1, N2=$N2), χ_local=$χ_local")
    @assert length(all_records) == 5 * N
    all_ok = true
    for r in 0:N-1
        base = 5 * r
        rk, rr1, rr2, crk, ok = all_records[base+1], all_records[base+2],
                                all_records[base+3], all_records[base+4],
                                all_records[base+5]
        status = ok == 1 ? "PASS" : "FAIL"
        println("  rank=$rk  (r1=$rr1, r2=$rr2)  col_rank=$crk  $status")
        all_ok &= (ok == 1)
    end
    println(all_ok ? "[2D allgather sanity] PASS" : "[2D allgather sanity] FAIL")
end

# Also trip a local @test on every rank so any failure is loud in the per-rank
# log even before rank 0's summary lands.
@test local_pass

MPI.Barrier(comm)
MPI.Finalize()
