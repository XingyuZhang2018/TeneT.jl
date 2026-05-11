# sandbox/2d_grid_sanity.jl
#
# Phase 0 Task 0.1 — Bare 4-rank MPI Cartesian grid sanity check.
#
# Purpose:
#   Surface design risks BEFORE touching production code by exercising the
#   building blocks of a 2D distributed VUMPS runtime in isolation:
#     - 2x2 Cartesian process grid via `MPI.Cart_create`
#     - per-rank (r1, r2) coordinates via `MPI.Cart_coords`
#     - row / column sub-communicators via `MPI.Cart_sub`
#   We just confirm that for N=4 ranks the cartesian layout maps to the
#   expected (r1, r2) coordinates and that each row/col sub-communicator
#   has size 2.
#
# When to run:
#   On any node that supports `mpirun -n 4` (e.g. Sofia HPC). Cannot be
#   executed on Windows since OpenMPI's process launcher is Linux/macOS only.
#
#   Invocations with N != N1*N2 (e.g. `julia sandbox/2d_grid_sanity.jl`)
#   will trip the `@assert N == N1*N2` and exit cleanly — this is by design.
#
# How to run:
#   mpirun -n 4 julia --project=. sandbox/2d_grid_sanity.jl
#
# Expected output (rank 0 prints a single sorted summary; no interleaving):
#   [2D grid sanity] N=4, dims=(N1=2, N2=2)
#     rank=0  (r1=0, r2=0)  row_size=2  col_size=2
#     rank=1  (r1=0, r2=1)  row_size=2  col_size=2
#     rank=2  (r1=1, r2=0)  row_size=2  col_size=2
#     rank=3  (r1=1, r2=1)  row_size=2  col_size=2
#   [2D grid sanity] PASS

using MPI

MPI.Init()

const comm = MPI.COMM_WORLD
const N    = MPI.Comm_size(comm)
const rank = MPI.Comm_rank(comm)

# Hard-coded 2x2 grid: Task 0.1 only exercises the bare cartesian layout, so
# we deliberately do NOT call MPI.Dims_create — the grid shape is part of the
# sanity check, not something to autotune away.
const N1, N2 = 2, 2
@assert N == N1 * N2 "Run with `mpirun -n 4` (got N=$N; this sandbox requires N1*N2 = $(N1*N2))"

# `periodic=[false, false]` matches the open-boundary 2D VUMPS layout we plan
# to use. `reorder=false` (the MPI.jl default) preserves the COMM_WORLD rank
# numbering so the (r1, r2) ↔ rank mapping is reproducible across runs.
cart_comm = MPI.Cart_create(comm, [N1, N2]; periodic=[false, false])
coords    = MPI.Cart_coords(cart_comm)   # 0-based [r1, r2]
r1, r2    = coords[1], coords[2]

# Row comm: vary r2, fix r1  (used for AC -> AL/AR sweeps along a row)
row_comm = MPI.Cart_sub(cart_comm, [false, true])
# Col comm: vary r1, fix r2  (used for FL/FR transfer along a column)
col_comm = MPI.Cart_sub(cart_comm, [true, false])

row_size = MPI.Comm_size(row_comm)
col_size = MPI.Comm_size(col_comm)

# Sanity invariants we want to hold on every rank.
@assert row_size == N2 "row_comm size $row_size != N2=$N2 on rank $rank"
@assert col_size == N1 "col_comm size $col_size != N1=$N1 on rank $rank"

# To avoid interleaved stdout from concurrent ranks, gather one 5-element
# Int record per rank onto rank 0 and let rank 0 print a single sorted block.
record = Int[rank, r1, r2, row_size, col_size]
all_records = MPI.Gather(record, comm; root=0)

if rank == 0
    println("[2D grid sanity] N=$N, dims=(N1=$N1, N2=$N2)")
    # `Gather` flattens per-rank vectors into one Vector{Int} of length 5N.
    @assert length(all_records) == 5 * N
    for r in 0:N-1
        base = 5 * r
        rk, rr1, rr2, rs, cs = all_records[base+1], all_records[base+2],
                               all_records[base+3], all_records[base+4],
                               all_records[base+5]
        println("  rank=$rk  (r1=$rr1, r2=$rr2)  row_size=$rs  col_size=$cs")
    end
    println("[2D grid sanity] PASS")
end

MPI.Barrier(comm)
MPI.Finalize()
