# test/distributed_primitives_test.jl
#
# Phase 1 Task 1.3 — `allgather_dim` forward primitive.
#
# Verifies that `allgather_dim` (the first 2D distributed VUMPS communication
# primitive) materialises the full extent of an equally-partitioned dimension
# of a local tensor across a Cartesian sub-communicator, producing identical
# results on every participating rank. Both code paths are exercised:
#
#   * `dim == ndims(t)`: contiguous last-dim slab handed directly to
#     `allgatherv_p2p!`. Cheapest path; matches the column-major linear-buffer
#     view that `allgatherv_p2p!` assumes (see sandbox/2d_allgather_sanity.jl
#     for the 3D-recvbuf rationale).
#   * `dim != ndims(t)`: internal `permutedims(dim ↔ ndims) → allgather →
#     permutedims back` dance. Required by Section 2 of the design doc
#     (FLmap/FRmap need `dim == 1` gathers).
#
# Designed to run under either:
#
#   mpirun -n 4 julia --project=. test/distributed_primitives_test.jl
#       → exercises a 2x2 Cart2DGrid with non-trivial col_comm / row_comm
#         sub-communicators of size 2 each (the production-relevant case)
#
#   julia --project=. test/distributed_primitives_test.jl
#       → exercises the 1x1 / single-rank short-circuit path
#
# Cannot be invoked under `mpirun` on Windows since OpenMPI's process
# launcher is Linux/macOS only — multi-rank validation happens on Sofia HPC.
# Single-rank runs work everywhere and cover the M=1 copy-not-alias path.
#
# This file is NOT included from `test/runtests.jl` (which is serial); like
# `test/cart2d_grid_test.jl` it is invoked by a separate MPI test driver.

using Test
using MPI
using TeneT

MPI.Initialized() || MPI.Init()

const world = MPI.COMM_WORLD
const N     = MPI.Comm_size(world)
const rank  = MPI.Comm_rank(world)

@testset "allgather_dim primitive" begin
    if N == 4
        grid = Cart2DGrid(2, 2)

        @testset "Gather along col_comm — last dim (direct path)" begin
            # 2D tensor, last-dim distributed on col_comm (size N1=2).
            # Each rank's data is tagged with `grid.r1 + 1` so the receiver
            # can verify which rank produced each gathered slab.
            chi_local = 4
            data = fill(Float64(grid.r1 + 1), 8, chi_local)
            full = allgather_dim(data, 2, grid.col_comm)

            @test size(full) == (8, 2 * chi_local)
            @test eltype(full) === Float64
            # col_comm rank 0 == r1=0 (Cart_sub preserves coord order), so
            # cols 1:chi_local hold 1.0, cols chi_local+1:2*chi_local hold 2.0.
            @test all(full[:, 1:chi_local]                  .== 1.0)
            @test all(full[:, chi_local+1:2*chi_local]       .== 2.0)
        end

        @testset "Gather along row_comm — last dim (direct path)" begin
            # Symmetry check: same shape, different sub-comm, tag on `r2`.
            chi_local = 4
            data = fill(Float64(grid.r2 + 1), 8, chi_local)
            full = allgather_dim(data, 2, grid.row_comm)

            @test size(full) == (8, 2 * chi_local)
            @test all(full[:, 1:chi_local]                  .== 1.0)
            @test all(full[:, chi_local+1:2*chi_local]       .== 2.0)
        end

        @testset "Gather along col_comm — first dim (permutedims dance)" begin
            # The non-last-dim path. Each rank places an identifiable block at
            # dim=1; gather must place rows 1:chi_local from col_rank 0 and
            # rows chi_local+1:2*chi_local from col_rank 1.
            chi_local = 4
            data = fill(Float64(grid.r1 + 1), chi_local, 6)
            full = allgather_dim(data, 1, grid.col_comm)

            @test size(full) == (2 * chi_local, 6)
            @test all(full[1:chi_local, :]                  .== 1.0)
            @test all(full[chi_local+1:2*chi_local, :]       .== 2.0)
        end

        @testset "Gather 4D tensor — middle dim exercises dance" begin
            # 4D production-shape rehearsal. dim=2 (middle, not last) so the
            # permutedims dance is exercised on a non-trivial rank. Tag scheme
            # uses col_comm rank to make per-rank slabs distinct.
            chi_local = 3
            data = fill(Float64(grid.r1 + 1), 2, chi_local, 4, 5)
            full = allgather_dim(data, 2, grid.col_comm)

            @test size(full) == (2, 2 * chi_local, 4, 5)
            @test all(full[:, 1:chi_local, :, :]                  .== 1.0)
            @test all(full[:, chi_local+1:2*chi_local, :, :]       .== 2.0)
        end

        @testset "Distinct per-rank payload — element-level correctness" begin
            # Beyond the constant-fill tests above, populate the local slab
            # with a rank-tagged linear ramp so the gather has to reconstruct
            # every individual element (not just a block-uniform value). This
            # catches stride bugs in the permutedims dance that a fill-test
            # would miss.
            chi_local = 4
            payload   = Float64.(reshape(1:(chi_local*6), chi_local, 6))
            data      = payload .+ 100.0 * (grid.r1 + 1)
            full      = allgather_dim(data, 1, grid.col_comm)

            @test size(full) == (2 * chi_local, 6)
            @test full[1:chi_local, :]            == payload .+ 100.0
            @test full[chi_local+1:2*chi_local, :] == payload .+ 200.0
        end

        @testset "M=1 short-circuit returns a fresh copy (not an alias)" begin
            # Each rank gets its own size-1 sub-comm by splitting world on
            # `color = rank`. `allgather_dim` on a 1-rank comm must return a
            # value-equal copy that does NOT alias the input — the rrule
            # (Task 1.4) relies on this for mutation safety.
            single = MPI.Comm_split(world, rank, 0)
            try
                @test MPI.Comm_size(single) == 1
                data = rand(Float64, 4, 6)
                out = allgather_dim(data, 2, single)
                @test out == data
                @test out !== data        # not the same object
                @test pointer(out) != pointer(data)
                # Mutating the output must not affect the input.
                out[1, 1] = 999.0
                @test data[1, 1] != 999.0
            finally
                MPI.free(single)
            end
        end
    elseif N == 1
        # Local-machine sanity: covers only the M=1 short-circuit. Multi-rank
        # paths are validated under `mpirun -n 4` on Sofia.
        @testset "M=1 short-circuit (single-rank driver)" begin
            data = rand(Float64, 4, 6)
            out = allgather_dim(data, 2, MPI.COMM_WORLD)
            @test out == data
            @test out !== data
            @test pointer(out) != pointer(data)
        end
    else
        # Be loud rather than silently skip — see Phase 0 lesson: anything
        # other than 1 or 4 ranks indicates a misconfigured driver.
        if rank == 0
            @warn "distributed_primitives_test expects 1 or 4 MPI ranks; got N=$N. Skipping."
        end
    end
end

MPI.Barrier(world)
# Deliberately do NOT call MPI.Finalize() — this file may be `include`d from
# a larger MPI test driver (test/sofia_mpi_test_driver.jl or similar).
