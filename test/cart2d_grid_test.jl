# test/cart2d_grid_test.jl
#
# Phase 1 Task 1.1 — `Cart2DGrid` constructor sanity check.
#
# Verifies that the 2D Cartesian MPI grid wrapper exposes consistent
# (r1, r2) coordinates and row/col sub-communicators. Designed to run
# under either:
#
#   mpirun -n 4 julia --project=. test/cart2d_grid_test.jl
#       → exercises the 2x2 Cartesian path
#
#   julia --project=. test/cart2d_grid_test.jl
#       → exercises the 1x1 serial-mode default constructor
#
# Cannot be invoked on Windows under `mpirun` since OpenMPI's process
# launcher is Linux/macOS only — multi-rank execution is deferred to
# Sofia HPC. Single-rank (no `mpirun`) runs work everywhere.

using Test
using MPI
using TeneT

MPI.Initialized() || MPI.Init()

const world = MPI.COMM_WORLD
const N     = MPI.Comm_size(world)
const rank  = MPI.Comm_rank(world)

@testset "Cart2DGrid" begin
    if N == 4
        @testset "2x2 grid (4 ranks)" begin
            grid = Cart2DGrid(2, 2)

            # Shape
            @test grid.N1 == 2
            @test grid.N2 == 2

            # Sub-communicator sizes:
            #   row_comm varies r2 → length N2
            #   col_comm varies r1 → length N1
            @test MPI.Comm_size(grid.row_comm) == 2
            @test MPI.Comm_size(grid.col_comm) == 2

            # Per-rank (r1, r2) must lie in valid ranges.
            @test 0 <= grid.r1 < grid.N1
            @test 0 <= grid.r2 < grid.N2

            # (r1, r2) → cart rank consistency: the cart-rank derived
            # from this rank's coords must equal its world rank, since
            # we called Cart_create with the default reorder=false.
            @test MPI.Cart_rank(grid.cart, [grid.r1, grid.r2]) ==
                  MPI.Comm_rank(grid.world)

            # Cross-rank gather: confirm every (r1, r2) appears exactly
            # once across the grid. Wrap the accumulator in `let` per
            # Phase 0's top-level-soft-scope lesson.
            let
                record  = Int[grid.r1, grid.r2]
                gathered = MPI.Gather(record, world; root=0)
                if rank == 0
                    @test length(gathered) == 2 * N
                    coords_set = Set{Tuple{Int,Int}}()
                    for k in 0:N-1
                        push!(coords_set,
                              (gathered[2k+1], gathered[2k+2]))
                    end
                    expected = Set((i, j) for i in 0:1, j in 0:1)
                    @test coords_set == expected
                end
            end
        end
    elseif N == 1
        @testset "1x1 serial-mode default" begin
            grid = Cart2DGrid()
            @test grid.N1 == 1
            @test grid.N2 == 1
            @test grid.r1 == 0
            @test grid.r2 == 0
            @test MPI.Comm_size(grid.row_comm) == 1
            @test MPI.Comm_size(grid.col_comm) == 1
            @test MPI.Cart_rank(grid.cart, [grid.r1, grid.r2]) ==
                  MPI.Comm_rank(grid.world)
        end
    else
        # Be loud rather than silently skip — anything other than 1 or
        # 4 ranks indicates a misconfigured driver.
        if rank == 0
            @warn "Cart2DGrid test expects 1 or 4 MPI ranks; got N=$N. Skipping."
        end
    end
end

MPI.Barrier(world)
# Deliberately do NOT call MPI.Finalize() here so the test file can be
# `include`d from a larger driver if that ever becomes useful.
