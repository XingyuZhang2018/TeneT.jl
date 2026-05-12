# test/distributed_primitives_test.jl
#
# Phase 1 Tasks 1.3–1.5 — `allgather_dim` / `reduce_scatter_dim` primitives
# plus their mutual rrules.
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
using Zygote

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

# ─── Task 1.5: reduce_scatter_dim forward primitive ──────────────────────
@testset "reduce_scatter_dim primitive" begin
    if N == 4
        grid = Cart2DGrid(2, 2)
        M_col = MPI.Comm_size(grid.col_comm)
        @assert M_col == 2 "Cart2DGrid(2,2).col_comm should have size 2; got $M_col"

        @testset "Forward: identical 1s reduced over col_comm (size 2) → 2.0 in each rank's slice" begin
            # All ranks contribute the same `1.0`-filled tensor. allreduce
            # sums M_col copies → 2.0. Slicing returns this rank's piece.
            full = ones(Float64, 8, 8)
            local_data = reduce_scatter_dim(full, 1, grid.col_comm)
            @test size(local_data) == (4, 8)
            @test eltype(local_data) === Float64
            @test all(local_data .== Float64(M_col))   # = 2.0
        end

        @testset "Forward: distinct per-rank ramp exercises sum semantics" begin
            # Each rank holds a DIFFERENT full tensor: rank r1 has (r1+1) * ones.
            # Sum over col_comm = 1 + 2 = 3. Every rank's slice has the same
            # post-sum value (sum is rank-replicated), differing only by which
            # slab of rows we keep.
            data = fill(Float64(grid.r1 + 1), 8, 8)
            local_data = reduce_scatter_dim(data, 1, grid.col_comm)
            @test size(local_data) == (4, 8)
            @test all(local_data .== 3.0)
        end

        @testset "Forward: last-dim slice (no layout dance needed)" begin
            # Unlike allgather_dim, reduce_scatter_dim does not need a permutedims
            # dance for dim != ndims because allreduce_p2p! is layout-agnostic.
            # We still verify last-dim slicing works to lock the invariant.
            data = fill(Float64(grid.r1 + 1), 8, 8)
            local_data = reduce_scatter_dim(data, 2, grid.col_comm)
            @test size(local_data) == (8, 4)
            @test all(local_data .== 3.0)
        end

        @testset "Forward: 4D middle-dim slice" begin
            # Production-shape rehearsal: 4D tensor, slice along dim=2 (middle).
            # Verifies the general N-D slicing logic.
            data = fill(Float64(grid.r1 + 1), 2, 6, 4, 5)
            local_data = reduce_scatter_dim(data, 2, grid.col_comm)
            @test size(local_data) == (2, 3, 4, 5)
            @test all(local_data .== 3.0)
        end

        @testset "Round-trip: reduce_scatter ∘ allgather is M·identity" begin
            # Each rank starts with its own local tensor x_r. allgather_dim
            # makes every rank see the SAME concatenated tensor g of M·χ_local
            # rows. reduce_scatter_dim then allreduces M identical copies of g
            # (→ M·g) and slices, so rank r gets M · g[r-slice] = M · x_r.
            #
            # Hence the round-trip is **M·identity**, not identity. This is
            # the expected behavior for the rrule composition: if `allgather`
            # appears in a forward pass, its rrule via `reduce_scatter_dim`
            # introduces the M factor that correctly accounts for the M
            # rank-replicated copies of the downstream scalar loss.
            local_orig = fill(Float64(grid.r1 + 1), 4, 8)
            full = allgather_dim(local_orig, 1, grid.col_comm)
            recovered = reduce_scatter_dim(full, 1, grid.col_comm)
            @test size(recovered) == size(local_orig)
            @test recovered ≈ M_col .* local_orig
        end

        @testset "M=1 short-circuit returns a fresh copy (not an alias)" begin
            # Mirror of the allgather_dim M=1 test: rrule mutation safety.
            single = MPI.Comm_split(world, rank, 0)
            try
                @test MPI.Comm_size(single) == 1
                data = rand(Float64, 4, 6)
                out = reduce_scatter_dim(data, 2, single)
                @test out == data
                @test out !== data
                @test pointer(out) != pointer(data)
                out[1, 1] = 999.0
                @test data[1, 1] != 999.0
            finally
                MPI.free(single)
            end
        end
    elseif N == 1
        @testset "M=1 short-circuit (single-rank driver)" begin
            data = rand(Float64, 4, 6)
            out = reduce_scatter_dim(data, 2, MPI.COMM_WORLD)
            @test out == data
            @test out !== data
            @test pointer(out) != pointer(data)
        end
    else
        if rank == 0
            @warn "reduce_scatter_dim tests skipped: expects N=1 or N=4 ranks; got $N."
        end
    end
end

# ─── Task 1.4 + 1.5: rrule gradient flow via Zygote ──────────────────────
#
# These tests verify the rrule fires and produces correctly-shaped tangents
# under the PR #42 "single loss" gradient convention adopted for the
# 2D-distributed VUMPS rewrite. The `allgather_dim` rrule is a pure SLICE
# (no allreduce); see `src/autodiff/rules.jl` for the rationale. Full
# finite-difference gradcheck across MPI ranks is left for end-to-end
# parity tests (Phase 5).

@testset "allgather_dim + reduce_scatter_dim rrules (gradient flow via Zygote)" begin
    if N == 4
        grid = Cart2DGrid(2, 2)
        M_col = MPI.Comm_size(grid.col_comm)
        chi_local = 4

        @testset "allgather_dim rrule: ∇ loss(x) = sum(abs2, allgather(x))" begin
            # Forward: y = allgather_dim(x, 1, col_comm) is the SAME shared
            # tensor on every rank (concat of M local slices). The loss
            # L = sum(abs2, y) is therefore identical across ranks.
            #
            # Adjoint (PR #42 "single loss" convention): d_y = 2 · y on every
            # rank. The rrule extracts the local rank's slice via a pure SLICE
            # (no allreduce), so rank r recovers d_y[r-slice] = 2 · y[r-slice]
            # = 2 · x_r. No M-factor.
            #
            # Rationale: Zygote computes ∂loss/∂x_local for the loss as
            # evaluated on rank r. Since allgather replicates the output,
            # every rank computes the SAME scalar loss; the local x_local
            # only contributes to its r-slice of the gathered tensor, so
            # ∂loss/∂x_r = 2 · x_r with no M-factor. The mathematical
            # linear-adjoint (reduce_scatter) would represent the "M
            # independent losses" interpretation, which we do not want.
            x = rand(Float64, chi_local, 8)
            loss(z) = sum(abs2, allgather_dim(z, 1, grid.col_comm))
            g = Zygote.gradient(loss, x)[1]

            @test g !== nothing
            @test size(g) == size(x)
            @test g ≈ 2 * x        # PR #42 convention: NO M-factor.
        end

        @testset "reduce_scatter_dim rrule: ∇ loss(x) = sum(abs2, reduce_scatter(x))" begin
            # Forward: y_r = reduce_scatter_dim(full_bcast, 1, col_comm) =
            #               (M · full_bcast)[r-slice] = M · full_bcast[r-slice].
            # L_r = sum(abs2, y_r) — DIFFERENT on each rank (each rank slices
            # a different part of the reduced full).
            #
            # Adjoint: d_y_r = 2 · y_r = 2M · full_bcast[r-slice]. The rrule
            # pushes back via allgather, which concatenates each rank's 2M·slice
            # into a single full tensor on every rank → 2M · full_bcast.
            # This rrule is unchanged from the linear-adjoint convention —
            # reduce_scatter forward is "M-to-M" (slices differ per rank), not
            # replicating, so no M-factor issue arises.
            full = rand(Float64, 8, 8)
            # Broadcast so every rank starts with the same input (the usual
            # rrule-adjoint invariant for reduce_scatter).
            full_bcast = MPI.bcast(full, 0, grid.world)
            loss(z) = sum(abs2, reduce_scatter_dim(z, 1, grid.col_comm))
            g = Zygote.gradient(loss, full_bcast)[1]

            @test g !== nothing
            @test size(g) == size(full_bcast)
            @test g ≈ 2 * M_col * full_bcast
        end

        @testset "rrule round-trip: ∇ loss(x) = sum(abs2, reduce_scatter(allgather(x)))" begin
            # This test characterizes what the *rrule chain* computes — which
            # under the PR #42 "single loss" convention differs from the
            # naive per-rank true gradient. Read both derivations:
            #
            # ── What the rrule chain computes (this is what Zygote returns) ─
            #   forward:        y_full = allgather(x)              # M·χ_local rows, identical on each rank
            #                   z_r    = reduce_scatter(y_full)     # = M · y_full[r-slice] = M · x_r
            #   L_r           = sum(abs2, z_r) = M² · sum(abs2, x_r)
            #   d_z_r         = 2 · z_r = 2M · x_r
            #   reduce_scatter rrule (allgather): d_y_full = allgather(d_z) = 2M · concat_r(x_r)
            #   allgather rrule (SLICE, PR #42):  d_x      = d_y_full[r-slice] = 2M · x_r
            #   ────────────────────────────────────────────────────────────────
            #   Chain output: g = 2 · M_col · x.
            #
            # ── Naive per-rank true ∂L_r/∂x_r ───────────────────────────────
            #   L_r = M² · |x_r|² ⇒ ∂L_r/∂x_r = 2M² · x_r.
            #
            # The chain (2M·x) differs from the per-rank true gradient (2M²·x)
            # by an M-factor: this is the cost of the "single loss" convention
            # when both allgather AND reduce_scatter appear in a chain — the
            # M-factor partially (not fully) cancels. This is NOT a bug; it
            # is the Zygote+MPI convention we adopt to match PR #42's serial
            # parity. The round-trip is documented here as a chain-composition
            # check, not a "matches the math" check.
            x = rand(Float64, chi_local, 8)
            loss(z) = sum(abs2, reduce_scatter_dim(
                                    allgather_dim(z, 1, grid.col_comm),
                                    1, grid.col_comm))
            g = Zygote.gradient(loss, x)[1]

            @test g !== nothing
            @test size(g) == size(x)
            @test g ≈ 2 * M_col * x        # NOT 2 * M² * x — see comment above.
        end
    elseif N == 1
        # M=1: rrules degenerate to identity. `2 · 1 · x = 2x`.
        # (Same result under both PR #42 and linear-adjoint conventions.)
        @testset "rrules degenerate to identity at M=1" begin
            x = rand(Float64, 4, 6)
            loss_g(z) = sum(abs2, allgather_dim(z, 2, MPI.COMM_WORLD))
            loss_r(z) = sum(abs2, reduce_scatter_dim(z, 2, MPI.COMM_WORLD))
            g_g = Zygote.gradient(loss_g, x)[1]
            g_r = Zygote.gradient(loss_r, x)[1]
            @test g_g ≈ 2 * x
            @test g_r ≈ 2 * x
        end
    else
        if rank == 0
            @warn "rrule gradient tests skipped: expects N=1 or N=4 ranks; got $N."
        end
    end
end

# ─── Task 1.6: allreduce_dim primitive + identity rrule ─────────────────
#
# Forward: sums tensor across all ranks in comm, every rank receives the
# same reduced value. Shape preserved.
#
# rrule: IDENTITY — passes d_result through unchanged (no allreduce in
# backward). This matches the PR #42 "per-rank Zygote semantics"
# convention: ∂(allreduce(x_r))/∂x_r = 1 on rank r since other ranks'
# x_{r'} are treated as constants. See rules.jl comment for the
# rationale on why the mathematical "self-adjoint = allreduce" form
# would introduce a spurious M-factor.

@testset "allreduce_dim primitive" begin
    if N == 4
        grid = Cart2DGrid(2, 2)

        @testset "Forward: sums rank-specific inputs" begin
            # rank-r1 has constant tensor (r1+1); sum over col_comm = 1 + 2 = 3
            x = fill(Float64(grid.r1 + 1), 4, 8)
            summed = allreduce_dim(x, +, grid.col_comm)
            @test all(summed .== 3.0)
            @test size(summed) == size(x)  # shape preserved
            @test summed !== x  # not aliased
        end

        @testset "Forward: M=1 short-circuit returns copy" begin
            single = MPI.Comm_split(world, rank, 0)
            try
                x = rand(Float64, 4, 8)
                y = allreduce_dim(x, +, single)
                @test y == x
                @test y !== x  # copy, not alias
                @test pointer(y) != pointer(x)
            finally
                MPI.free(single)
            end
        end

        @testset "rrule: identity backward (no M factor)" begin
            x = fill(Float64(grid.r1 + 1), 4, 8)
            # loss = sum(abs2, allreduce(x)) ≡ sum(abs2, summed) where summed = 3.0
            loss(x) = sum(abs2, allreduce_dim(x, +, grid.col_comm))
            g = Zygote.gradient(loss, x)[1]
            # Per Zygote per-rank semantics: ∂loss/∂x_r = 2*summed * ∂summed/∂x_r = 2*summed*1 = 2*summed
            # Identity rrule passes d_summed = 2*summed through unchanged, so g = 2*summed = 6.
            # The mathematical "self-adjoint = allreduce" would multiply by M and give 12 — wrong.
            @test g ≈ 2 .* fill(3.0, 4, 8)  # 2 * summed (which is 3.0 everywhere)
            # Cross-check: g should NOT be 2*M*summed (12 each), which the wrong rrule would yield
            @test !all(g .== 12.0)
        end

        @testset "rrule: gradient flow with different per-rank d_y" begin
            # When d_y is different across ranks (production case), identity rrule
            # gives different d_x per rank — this is the "per-rank contribution"
            # form expected by VUMPS's boundary allreduce pattern.
            x = fill(Float64(grid.r1 + 1), 4, 8)
            # Loss weights y differently per rank: rank-0 multiplies by 1.0, rank-1 by 2.0
            weight = Float64(grid.r1 + 1)
            loss(x) = sum(weight .* allreduce_dim(x, +, grid.col_comm))
            g = Zygote.gradient(loss, x)[1]
            # On rank r: y = summed = 3.0 (same). loss_r = weight_r * sum(summed) = weight_r * 96
            # ∂loss_r/∂x_r = weight_r * ∂(sum(summed))/∂x_r = weight_r * (4*8) = weight_r * 32
            # Identity rrule: d_x = d_y = weight_r * ones (shape (4,8))
            # No M-factor; gradient is rank-specific (per rank's local view)
            @test g ≈ fill(weight, size(x))
        end
    else
        @warn "Skipping allreduce_dim tests: requires N=4 (got $N)"
    end
end

MPI.Barrier(world)
# Deliberately do NOT call MPI.Finalize() — this file may be `include`d from
# a larger MPI test driver (test/sofia_mpi_test_driver.jl or similar).
