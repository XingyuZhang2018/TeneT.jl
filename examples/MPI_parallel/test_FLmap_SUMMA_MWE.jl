# SUMMA-FLmap MWE — minimal working example, validates the algorithm from
# docs/2026-05-12-summa-flmap-design.md against 1D FLmap_parallel.
#
# Run with the same submit pattern as test_MPI_config.jl (4 GPU square grid,
# CUDA+NCCL env). See timing/submit_*.sh.
#
# What this tests:
#   1. Correctness: SUMMA-FLmap output equals 1D FLmap_parallel output to ~1e-10
#   2. Timing: per-iteration forward time vs 1D FLmap_parallel
#
# Setup: deterministic seeded random tensors so both 1D and 2D paths see the
# same numerical data. Compare results to validate the SUMMA algorithm.

using CUDA, MPI, LinearAlgebra, Printf, Random, TeneT
using TensorOperations

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nprocs = MPI.Comm_size(comm)
CUDA.device!(0)

# ─── Cart2DGrid setup (square grid only for v1) ────────────────────────────
N1 = Int(floor(sqrt(nprocs)))
while nprocs % N1 != 0 || N1 * N1 != nprocs
    N1 -= 1
    N1 == 0 && error("SUMMA MWE requires square grid (N=$nprocs is not a square)")
end
N2 = N1
M  = N1  # alias: M = N1 = N2

grid = Cart2DGrid(N1, N2)
const r1 = grid.r1
const r2 = grid.r2

rank == 0 && println("=" ^ 70)
rank == 0 && println("SUMMA-FLmap MWE — N=$nprocs (M=$M square grid)  device=$(CUDA.device())")
rank == 0 && println("hostname=$(gethostname())")
rank == 0 && println("=" ^ 70)
flush(stdout)

# ─── Per-rank slice indices ────────────────────────────────────────────────
slice_of(r, M_, χ) = (r * (χ ÷ M_) + 1):((r + 1) * (χ ÷ M_))

# ─── SUMMA-FLmap implementation ─────────────────────────────────────────────
"""
    FLmap_parallel_2D_SUMMA(FL_local, ALu_local, ALd_local, M1, M2; grid)

Compute `result[d=slice_r1, g, h, l=slice_r2]` = full FLmap with inputs
in 2D-distributed form (per-rank slabs of shape (χ/M, D, D, χ/M)).

Algorithm (see docs/2026-05-12-summa-flmap-design.md):
  Phase 0a: AllGather ALu along col_comm (a → FULL on dim 1)
  Phase 0b: Bcast within row_comm to bring d=slice_r1 to all row members
  Phase B:  AllGather FL along col_comm (a → FULL on dim 1)
  Phase A:  SUMMA over i — M iterations of paired broadcasts + einsum
"""
function FLmap_parallel_2D_SUMMA(FL_local, ALu_local, ALd_local, M1, M2; grid)
    @assert grid.N1 == grid.N2 "SUMMA v1 requires square grid"
    M = grid.N1
    chi_per_M = size(FL_local, 1)
    chi_full = M * chi_per_M
    D = size(FL_local, 2)
    @assert size(ALu_local, 1) == chi_per_M
    @assert size(ALd_local, 1) == chi_per_M

    # ─── Phase 0a: AllGather ALu.a along col_comm[r2] ────────────────────
    # Before: ALu_local[a=slice_r1, b, c, d=slice_r2] — shape (χ/M, D, D, χ/M)
    # After:  ALu_a_full[a=ALL,     b, c, d=slice_r2] — shape (χ,   D, D, χ/M)
    ALu_a_full = allgather_dim_direct(ALu_local, 1, grid.col_comm)

    # ─── Phase 0b: Bcast within row_comm[r1] from rank with r2 == r1 ─────
    # Source rank in row_comm[r1] is the col-coord r2 = r1 (the "diagonal" rank).
    # Before bcast: source rank has ALu_a_full[a=ALL, d=slice_r1], others have d=slice_r2.
    # After bcast:  ALL row_comm[r1] members have ALu_d_my[a=ALL, d=slice_r1].
    ALu_d_my = (grid.r2 == grid.r1) ? copy(ALu_a_full) :
                                       similar(ALu_a_full)
    MPI.Bcast!(ALu_d_my, grid.r1, grid.row_comm)

    # ─── Phase B: AllGather FL.a along col_comm[r2] ──────────────────────
    # Before: FL_local[a=slice_r1, e, f, i=slice_r2] — shape (χ/M, D, D, χ/M)
    # After:  FL_a_full[a=ALL,     e, f, i=slice_r2] — shape (χ,   D, D, χ/M)
    FL_a_full = allgather_dim_direct(FL_local, 1, grid.col_comm)

    # ─── Phase A: SUMMA over i (M iterations) ────────────────────────────
    # partial[d=slice_r1, g, h, l=slice_r2] = sum_t (per-iter contribution)
    partial = CUDA.zeros(eltype(FL_local), chi_per_M, D, D, chi_per_M)

    for t in 0:M-1
        # ─── FL_t = FL[a=ALL, e, f, i=slice_t] via Bcast in row_comm[r1] ──
        # Source for i=slice_t in row_comm[r1] (varies r2) is the col-coord r2 = t.
        FL_t = (grid.r2 == t) ? copy(FL_a_full) : similar(FL_a_full)
        MPI.Bcast!(FL_t, t, grid.row_comm)

        # ─── ALd_t = ALd[i=slice_t, j, k, l=slice_r2] via Bcast in col_comm[r2] ──
        # Source for i=slice_t in col_comm[r2] (varies r1) is the row-coord r1 = t.
        ALd_t = (grid.r1 == t) ? copy(ALd_local) : similar(ALd_local)
        MPI.Bcast!(ALd_t, t, grid.col_comm)

        # ─── Local einsum, accumulate into partial ─────────────────────────
        # FL_t[a, e, f, i_local] × ALd_t[i_local, j, k, l_local]
        #   × M1[e, j, g, b, p] × M2[f, k, h, c, p] × ALu_d_my[a, b, c, d_local]
        # Output: partial[d_local, g, h, l_local]
        @tensor partial[d, g, h, l] += FL_t[a, e, f, i] *
                                        ALd_t[i, j, k, l] *
                                        M1[e, j, g, b, p] *
                                        M2[f, k, h, c, p] *
                                        ALu_d_my[a, b, c, d]
    end

    return partial
end

# ─── Build deterministic test tensors ──────────────────────────────────────
# Use FIXED seed on EACH rank — gives identical full tensors across ranks.
# Then each rank can extract its own slice for 2D, or use full for 1D.

function build_deterministic_full(D, χ, seed)
    Random.seed!(seed)
    p = 2
    FL  = CuArray(randn(Float64, χ, D, D, χ))
    ALu = CuArray(randn(Float64, χ, D, D, χ))
    ALd = CuArray(randn(Float64, χ, D, D, χ))
    M1  = CuArray(randn(Float64, D, D, D, D, p))
    M2  = CuArray(randn(Float64, D, D, D, D, p))
    return FL, ALu, ALd, M1, M2
end

# ─── Main: run correctness + timing comparison ─────────────────────────────
rank == 0 && @printf("%-30s %15s %15s %15s %10s\n",
    "Config", "1D fwd (ms)", "2D-SUMMA (ms)", "ratio 2D/1D", "rel err")
rank == 0 && println("─" ^ 90)

for (D, χ) in [(10, 64), (10, 128), (10, 256)]
    total_splits = 128
    forloop_iter = total_splits ÷ nprocs

    # Deterministic full tensors on each rank (same seed → same data everywhere)
    FL_full, ALu_full, ALd_full, M1_full, M2_full =
        build_deterministic_full(D, χ, 42 + D + 100*χ)
    M_full_combined = M1_full  # for 1D FLmap_parallel which expects single M

    # Per-rank 2D slices
    chi_per_M = χ ÷ M
    sa = slice_of(grid.r1, M, χ)  # a slice for this rank's r1
    sl = slice_of(grid.r2, M, χ)  # l slice for this rank's r2
    FL_local  = FL_full[sa, :, :, sl] |> CuArray
    ALu_local = ALu_full[sa, :, :, sl] |> CuArray
    ALd_local = ALd_full[sa, :, :, sl] |> CuArray

    # ─── Warmup ───
    _ = TeneT.FLmap_parallel(FL_full, ALu_full, ALd_full,
                              (M1_full, M2_full); ifparallel=true, forloop_iter)
    CUDA.synchronize()
    _ = FLmap_parallel_2D_SUMMA(FL_local, ALu_local, ALd_local,
                                 M1_full, M2_full; grid)
    CUDA.synchronize()
    MPI.Barrier(comm)

    # ─── 1D timing ───
    nrep = 3
    t_1d = @elapsed for _ in 1:nrep
        TeneT.FLmap_parallel(FL_full, ALu_full, ALd_full,
                              (M1_full, M2_full); ifparallel=true, forloop_iter)
        CUDA.synchronize()
    end
    t_1d /= nrep

    GC.gc(); CUDA.reclaim()
    MPI.Barrier(comm)

    # ─── 2D-SUMMA timing ───
    t_2d = @elapsed for _ in 1:nrep
        FLmap_parallel_2D_SUMMA(FL_local, ALu_local, ALd_local,
                                 M1_full, M2_full; grid)
        CUDA.synchronize()
    end
    t_2d /= nrep

    # ─── Correctness check ───
    # 1D output: result_1d[d, g, h, l] FULL on each rank (shape (χ, D, D, χ))
    result_1d = TeneT.FLmap_parallel(FL_full, ALu_full, ALd_full,
                                      (M1_full, M2_full); ifparallel=true, forloop_iter)
    CUDA.synchronize()
    # 2D output: result_2d[d=slice_r1, g, h, l=slice_r2] — shape (χ/M, D, D, χ/M)
    result_2d = FLmap_parallel_2D_SUMMA(FL_local, ALu_local, ALd_local,
                                         M1_full, M2_full; grid)
    CUDA.synchronize()

    # Compare 2D output to the corresponding slice of 1D output
    sd = slice_of(grid.r1, M, χ)  # d slice for this rank's r1
    sl_out = slice_of(grid.r2, M, χ)  # l slice
    result_1d_slice = result_1d[sd, :, :, sl_out]

    abs_diff = Array(result_1d_slice .- result_2d)
    rel_err = norm(abs_diff) / norm(Array(result_1d_slice))

    # Allreduce rel_err to max across ranks
    rel_err_max = MPI.Allreduce(rel_err, max, comm)

    rank == 0 && @printf("D=%-2d χ=%-4d                  %15.2f %15.2f %15.3fx %10.2e\n",
        D, χ, t_1d*1000, t_2d*1000, t_2d/t_1d, rel_err_max)
    flush(stdout)

    GC.gc(); CUDA.reclaim()
end

rank == 0 && println()
rank == 0 && println("=" ^ 70)
rank == 0 && println("Done")
rank == 0 && println("=" ^ 70)

GC.gc(); CUDA.reclaim()
MPI.Barrier(comm)
MPI.Finalize()
