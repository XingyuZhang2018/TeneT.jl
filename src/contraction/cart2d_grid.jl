"""
    Cart2DGrid

A thin wrapper around a 2D MPI Cartesian process topology used by the
distributed VUMPS / iPEPS contraction kernels.

Fields
------
* `N1`, `N2`     — grid dimensions along axis 1 and axis 2 respectively;
                   `N1 * N2 == MPI.Comm_size(world)` is enforced.
* `world`        — the input parent communicator (usually
                   `MPI.COMM_WORLD`); kept for sanity checks / lifecycle.
* `cart`         — the Cartesian communicator returned by
                   `MPI.Cart_create(world, [N1, N2]; periodic=[false,false])`.
                   `reorder=false` (MPI.jl default) means the (r1, r2)
                   coordinate ↔ world-rank mapping is reproducible.
* `row_comm`     — sub-communicator obtained from `Cart_sub(cart, [false, true])`.
                   Varies `r2`, fixes `r1`; size `N2`. Used for "row-wise"
                   collectives along axis 2 — e.g. the AC → AL/AR sweep
                   across one row of the unit cell in the co-distributed
                   contraction.
* `col_comm`     — sub-communicator obtained from `Cart_sub(cart, [true, false])`.
                   Varies `r1`, fixes `r2`; size `N1`. Used for "column-wise"
                   collectives along axis 1 — e.g. FL / FR transfer matrix
                   sweeps down one column.
* `r1`, `r2`     — this rank's 0-indexed coordinates in the grid; satisfy
                   `0 ≤ r1 < N1` and `0 ≤ r2 < N2`.

See `docs/2026-05-11-2d-distributed-vumps-runtime-design.md` (Section 1)
for the broader runtime design that this grid plugs into.

Lifecycle
---------
`MPI.Init` must already have been called by the caller before constructing
a `Cart2DGrid`; we do **not** initialize MPI for you, so that the host
program controls the MPI lifecycle (single `Init` / `Finalize` per
process).

Serial mode
-----------
When run without `mpirun` (single Julia process), `MPI.Comm_size(world)`
is 1 and `Cart2DGrid()` returns a degenerate 1×1 grid whose `row_comm`
and `col_comm` both have size 1. This lets the same boundary-algorithm
code path work uniformly in serial and distributed mode.
"""
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

"""
    Cart2DGrid(N1::Int, N2::Int, world::MPI.Comm = MPI.COMM_WORLD)

Construct a 2D `N1 × N2` Cartesian process grid on top of `world`.
Requires `MPI.Init` to have been called and `N1 * N2 == MPI.Comm_size(world)`.
"""
function Cart2DGrid(N1::Int, N2::Int, world::MPI.Comm = MPI.COMM_WORLD)
    @assert N1 * N2 == MPI.Comm_size(world) (
        "Cart2DGrid: N1 * N2 = $(N1 * N2) must equal MPI world size " *
        "$(MPI.Comm_size(world))"
    )
    cart   = MPI.Cart_create(world, [N1, N2]; periodic=[false, false])
    coords = MPI.Cart_coords(cart)            # 0-based [r1, r2]
    r1, r2 = coords[1], coords[2]
    # Row comm: vary r2 (size N2). Col comm: vary r1 (size N1).
    row_comm = MPI.Cart_sub(cart, [false, true])
    col_comm = MPI.Cart_sub(cart, [true,  false])
    return Cart2DGrid(N1, N2, world, cart, row_comm, col_comm, r1, r2)
end

"""
    Cart2DGrid()

Degenerate 1×1 serial-mode grid; convenient for single-process runs that
share code with the distributed path. Requires `MPI.Init` to have been
called (MPI.jl initialises lazily, but explicit `MPI.Init()` is what the
distributed entry points already do, so we keep the contract uniform).
"""
Cart2DGrid() = Cart2DGrid(1, 1)
