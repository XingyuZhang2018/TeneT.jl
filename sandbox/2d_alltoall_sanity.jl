# sandbox/2d_alltoall_sanity.jl
#
# Phase 0 Task 0.3 — Cross-axis dim swap (the all-to-all that production
# `alltoall_dim_swap` will implement, in its simplest possible form).
#
# Purpose:
#   The riskiest primitive of the 2D distributed VUMPS runtime is the
#   "dim swap" — taking a tensor whose i-axis is distributed across one
#   grid axis (say N2, the row_comm) and re-distributing the same i-axis
#   across the OTHER grid axis (N1, the col_comm), with the other axes
#   left untouched. Inside FLmap / FRmap this is what turns a per-row
#   slab into a per-column slab so the next contraction's parallelism
#   lines up. Get the index permutation wrong here and the entire 2D
#   VUMPS sweep silently produces garbage.
#
#   This sandbox isolates the swap from any tensor / Zygote / autodiff
#   machinery and verifies it on the smallest non-trivial grid (N1=N2=2),
#   so that any indexing or layout bug surfaces before we hide the call
#   behind a TensorMap wrapper in Phase 1.
#
# Formulation (small enough to reason about by hand):
#   Global matrix M[i, j] of shape (χ_full = 8, ncol = 4). The i-axis is
#   the "swap" dim, the j-axis is held fully on every rank (no
#   distribution along j). χ_per_rank = χ_full / N1 = 4.
#
#   Initial layout — i-axis distributed on N2 / row_comm:
#     rank (r1, r2)  holds  M[r2*4 + (0..3), :]
#       => (0,0)→rows 0..3,  (0,1)→rows 4..7,
#          (1,0)→rows 0..3,  (1,1)→rows 4..7
#
#   After dim_swap — i-axis now distributed on N1 / col_comm:
#     rank (r1, r2)  holds  M[r1*4 + (0..3), :]
#       => (0,0)→rows 0..3,  (0,1)→rows 0..3,
#          (1,0)→rows 4..7,  (1,1)→rows 4..7
#
#   Data motion for the 2x2 case:
#     diagonal ranks (r1 == r2)        : nothing to do (already correct)
#     off-diagonal ranks (r1 != r2)    : swap with the rank at (r2, r1)
#
#   So a single `MPI.Sendrecv!` between rank (r1, r2) and rank (r2, r1)
#   suffices. The implementation is therefore deliberately minimal:
#     - For N1=N2=2 each rank has exactly ONE partner.
#     - `Sendrecv!` makes the pairing explicit (and trivially correct).
#     - Production `alltoall_dim_swap` (Phase 1 Task 1.9) will use
#       `MPI.Alltoallv!` for the general (N1 != N2, larger N) case where
#       each rank pairs with many partners and the displacements matter.
#
# Cell entries:
#   M[i, j] = 1000 * i_global + j   (1-based, easy to read in a dump)
#   so a wrong index permutation produces an obviously-shifted value.
#
# When to run:
#   Linux/macOS node with `mpirun -n 4` (e.g. Sofia HPC). Cannot run on
#   Windows. Invocations with N != N1*N2 trip the `@assert` and exit.
#
# How to run:
#   mpirun -n 4 julia --project=. sandbox/2d_alltoall_sanity.jl
#
# Expected output (rank 0 prints one sorted summary; no interleaving):
#   [2D alltoall sanity] N=4, dims=(N1=2, N2=2), χ_full=8, χ_per_rank=4, ncol=4
#     rank=0  (r1=0, r2=0)  partner=0  PASS
#     rank=1  (r1=0, r2=1)  partner=2  PASS
#     rank=2  (r1=1, r2=0)  partner=1  PASS
#     rank=3  (r1=1, r2=1)  partner=3  PASS
#   [2D alltoall sanity] PASS
#
# Intentional simplifications (do NOT generalise here — that's Phase 1):
#   * Hard-coded N1=N2=2: rectangular grids are out of scope for the
#     sandbox. With (N1, N2) general, the "swap" no longer reduces to a
#     single sendrecv per rank.
#   * `Sendrecv!` instead of `Alltoallv!`: for the 2x2 case the latter
#     would still work but obscures the rank-pair structure. Phase 1's
#     production swap will use `Alltoallv!` with explicit counts/displs.
#   * j-axis is fully replicated (not distributed). The production swap
#     will also leave non-swap axes alone, so this matches the real
#     access pattern — we just don't bother distributing j here.

using MPI, Test

MPI.Init()

const comm = MPI.COMM_WORLD
const N    = MPI.Comm_size(comm)
const rank = MPI.Comm_rank(comm)

# Hard-coded 2x2 grid (matches Task 0.1 / 0.2).
const N1, N2 = 2, 2
@assert N == N1 * N2 "Run with `mpirun -n 4` (got N=$N; this sandbox requires N1*N2 = $(N1*N2))"

cart_comm = MPI.Cart_create(comm, [N1, N2]; periodic=[false, false])
coords    = MPI.Cart_coords(cart_comm)   # 0-based [r1, r2]
r1, r2    = coords[1], coords[2]

# Resolve the global rank of an arbitrary cartesian coordinate (r1', r2').
rank_at(r1_other, r2_other) = MPI.Cart_rank(cart_comm, [r1_other, r2_other])

# ─── Problem size ──────────────────────────────────────────────────────────
const χ_full     = 8
const χ_per_rank = χ_full ÷ N1   # == 4; equals χ_full ÷ N2 since N1 == N2 here
const ncol       = 4
@assert χ_per_rank * N1 == χ_full

# ─── Build initial local block (i distributed on N2 / row_comm) ────────────
#
# Global cell value: cell(i_global, j) = 1000 * i_global + j  (1-based).
# Wrong permutations show up as predictable, off-by-row arithmetic errors.
cell(i_global, j) = 1000.0 * i_global + j

i_offset_init = r2 * χ_per_rank   # initial: rows belong to r2
T_local = [cell(i_offset_init + i_local, j)
           for i_local in 1:χ_per_rank, j in 1:ncol]
@assert size(T_local) == (χ_per_rank, ncol)

# ─── The dim swap: rank (r1, r2) <-> rank (r2, r1) ─────────────────────────
#
# Partner is the rank whose cartesian coords are (r2, r1). For diagonal
# ranks (r1 == r2) the partner is itself and Sendrecv! degenerates to a
# memcpy — which is fine, and we explicitly cover this case so the code
# path is uniform.
const partner = rank_at(r2, r1)

# Recv buffer same shape as local block; we keep the j-axis untouched.
T_swapped = similar(T_local)

# `Sendrecv!` is symmetric: every rank sends T_local and receives the
# partner's T_local into T_swapped. Both endpoints of every pair post the
# matching call, so MPI matches them by (src, dst, tag) automatically.
MPI.Sendrecv!(T_local, T_swapped, comm; dest=partner, source=partner)

# ─── Verify against expected post-swap layout ──────────────────────────────
#
# After the swap, rank (r1, r2) should hold rows r1*4 + (1..4) of the
# global matrix M (i-axis is now distributed on N1 / col_comm).
i_offset_final = r1 * χ_per_rank
expected = [cell(i_offset_final + i_local, j)
            for i_local in 1:χ_per_rank, j in 1:ncol]

local_pass = T_swapped == expected   # exact equality: all entries are integer-valued floats

# ─── Gather pass/fail for a single ordered summary ─────────────────────────
record = Int[rank, r1, r2, partner, local_pass ? 1 : 0]
all_records = MPI.Gather(record, comm; root=0)

if rank == 0
    println("[2D alltoall sanity] N=$N, dims=(N1=$N1, N2=$N2), " *
            "χ_full=$χ_full, χ_per_rank=$χ_per_rank, ncol=$ncol")
    @assert length(all_records) == 5 * N
    all_ok = true
    for r in 0:N-1
        base = 5 * r
        rk, rr1, rr2, prt, ok = all_records[base+1], all_records[base+2],
                                all_records[base+3], all_records[base+4],
                                all_records[base+5]
        status = ok == 1 ? "PASS" : "FAIL"
        println("  rank=$rk  (r1=$rr1, r2=$rr2)  partner=$prt  $status")
        all_ok &= (ok == 1)
    end
    println(all_ok ? "[2D alltoall sanity] PASS" : "[2D alltoall sanity] FAIL")
end

# Also trip a local @test on every rank so any failure is loud in the
# per-rank log even before rank 0's summary lands.
@test local_pass

MPI.Barrier(comm)
MPI.Finalize()
