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
function _square_M(nprocs)
    m = Int(floor(sqrt(nprocs)))
    while m * m != nprocs
        m -= 1
        m == 0 && error("SUMMA MWE requires square grid (N=$nprocs is not a square)")
    end
    return m
end
const N1 = _square_M(nprocs)
const N2 = N1
const M  = N1  # alias: M = N1 = N2

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
function FLmap_parallel_2D_SUMMA(FL_local, ALu_local, ALd_local, M1, M2; grid, verbose=false)
    @assert grid.N1 == grid.N2 "SUMMA v1 requires square grid"
    M = grid.N1
    chi_per_M = size(FL_local, 1)
    chi_full = M * chi_per_M
    D = size(FL_local, 2)
    @assert size(ALu_local, 1) == chi_per_M
    @assert size(ALd_local, 1) == chi_per_M

    # Timer helper: synchronize device + barrier on world comm before reading
    # the clock so the measured interval reflects all-rank progress. Without
    # the barrier, a fast rank reports nonsense for collective phases.
    rank_world = MPI.Comm_rank(grid.world)
    function tick!(label, t0)
        verbose || return time()
        CUDA.synchronize()
        MPI.Barrier(grid.world)
        rank_world == 0 && @printf("    %-32s %8.3f ms\n", label, (time()-t0)*1000)
        return time()
    end

    verbose && rank_world == 0 && println("  --- SUMMA-FLmap timing breakdown (χ=$chi_full, M=$M) ---")
    CUDA.synchronize(); MPI.Barrier(grid.world)
    t0 = time()

    # ─── Phase 0a: AllGather ALu.a along col_comm[r2] ────────────────────
    ALu_a_full = allgather_dim_direct(ALu_local, 1, grid.col_comm)
    t0 = tick!("Phase 0a (col_comm AG ALu)", t0)

    # ─── Phase 0b: Bcast within row_comm[r1] from rank with r2 == r1 ─────
    ALu_d_my = (grid.r2 == grid.r1) ? copy(ALu_a_full) :
                                       similar(ALu_a_full)
    MPI.Bcast!(ALu_d_my, grid.r1, grid.row_comm)
    t0 = tick!("Phase 0b (row_comm Bcast ALu)", t0)

    # ─── Phase B: AllGather FL.a along col_comm[r2] ──────────────────────
    FL_a_full = allgather_dim_direct(FL_local, 1, grid.col_comm)
    t0 = tick!("Phase B  (col_comm AG FL)", t0)

    # ─── Phase A: SUMMA over i (M iterations) ────────────────────────────
    partial = CUDA.zeros(eltype(FL_local), chi_per_M, D, D, chi_per_M)
    t0 = tick!("Phase A  alloc partial", t0)

    t_bcast_row = 0.0
    t_bcast_col = 0.0
    t_einsum = 0.0
    for t in 0:M-1
        # Bcast FL_t in row_comm
        ts = time()
        FL_t = (grid.r2 == t) ? copy(FL_a_full) : similar(FL_a_full)
        MPI.Bcast!(FL_t, t, grid.row_comm)
        if verbose
            CUDA.synchronize(); MPI.Barrier(grid.world)
            t_bcast_row += time() - ts
        end

        # Bcast ALd_t in col_comm
        ts = time()
        ALd_t = (grid.r1 == t) ? copy(ALd_local) : similar(ALd_local)
        MPI.Bcast!(ALd_t, t, grid.col_comm)
        if verbose
            CUDA.synchronize(); MPI.Barrier(grid.world)
            t_bcast_col += time() - ts
        end

        # Local 5-tensor einsum, accumulate.
        #
        # Bracketing — force "FL × ALu first" tree.
        #
        # Default @tensor is left-to-right, so unbracketed would contract
        # FL × ALd first (share i), producing intermediate (a, e, f, j, k, l)
        # of size 256×10⁴×64 = 16M for SUMMA per-step (vs 4.1M for 1D
        # because 1D's per-rank l=16 not 64). Peak intermediate after
        # the next × M1 step is 260MB and ~90ms per step at χ=256.
        #
        # "FL × ALu first" contracts a (256): intermediate (e, f, i, b, c, d)
        # = 10⁴ × 64² = 4.1M = 33MB. Then × M1 → 8.2M, × M2 → 4.1M,
        # × ALd → 410K. Total FLOPs ~3.4× lower (2.1e10 vs 7.1e10).
        ts = time()
        @tensor partial[d, g, h, l] += (
            (
                (
                    (FL_t[a, e, f, i] * ALu_d_my[a, b, c, d])
                    * M1[e, j, g, b, p]
                )
                * M2[f, k, h, c, p]
            )
            * ALd_t[i, j, k, l]
        )
        if verbose
            CUDA.synchronize(); MPI.Barrier(grid.world)
            t_einsum += time() - ts
        end
    end
    if verbose && rank_world == 0
        @printf("    %-32s %8.3f ms  (per-step %.2f)\n",
            "Phase A  inner: bcast row_comm", t_bcast_row*1000, t_bcast_row*1000/M)
        @printf("    %-32s %8.3f ms  (per-step %.2f)\n",
            "Phase A  inner: bcast col_comm", t_bcast_col*1000, t_bcast_col*1000/M)
        @printf("    %-32s %8.3f ms  (per-step %.2f)\n",
            "Phase A  inner: einsum (5-tens)", t_einsum*1000, t_einsum*1000/M)
    end

    return partial
end

# ─── Alt 1: AllGather both axes + single einsum ────────────────────────────
"""
    FLmap_parallel_2D_AG(FL_local, ALu_local, ALd_local, M1, M2; grid, verbose)

Variant that AllGathers FL on **both** χ axes and ALd on its i axis,
turning the 4-step SUMMA inner loop into a single einsum identical in
shape to 1D's. Trades transient memory peak for kernel-launch reduction.

Persistent memory per rank is the same as pure SUMMA (1/M² of 1D).
Transient peak rises to ≈ full FL/ALd (matching 1D), but only during one
FLmap call; releasable between calls.

Phases:
  0a/0b: same as SUMMA — col_comm AG ALu then row_comm Bcast → ALu_d_my
  B:     col_comm AG FL on a-axis    → FL_a_full
  B2:    row_comm AG FL_a_full on i-axis → FL_full
  C:     col_comm AG ALd on i-axis   → ALd_full
  A:     single einsum (FL × ALu first tree) → partial[d=slice_r1, g, h, l=slice_r2]
"""
function FLmap_parallel_2D_AG(FL_local, ALu_local, ALd_local, M1, M2; grid, verbose=false)
    @assert grid.N1 == grid.N2 "AG v1 requires square grid"
    M = grid.N1
    chi_per_M = size(FL_local, 1)
    D = size(FL_local, 2)
    @assert size(ALu_local, 1) == chi_per_M
    @assert size(ALd_local, 1) == chi_per_M

    rank_world = MPI.Comm_rank(grid.world)
    function tick!(label, t0)
        verbose || return time()
        CUDA.synchronize(); MPI.Barrier(grid.world)
        rank_world == 0 && @printf("    %-32s %8.3f ms\n", label, (time()-t0)*1000)
        return time()
    end

    verbose && rank_world == 0 && println("  --- 2D-AG-FLmap timing breakdown (χ=$(M*chi_per_M), M=$M) ---")
    CUDA.synchronize(); MPI.Barrier(grid.world)
    t0 = time()

    # 0a: col_comm AG ALu on a-axis → (a=ALL, b, c, d=slice_r2)
    ALu_a_full = allgather_dim_direct(ALu_local, 1, grid.col_comm)
    t0 = tick!("Phase 0a (col_comm AG ALu)", t0)

    # 0b: row_comm Bcast from diagonal → (a=ALL, b, c, d=slice_r1)
    ALu_d_my = (grid.r2 == grid.r1) ? copy(ALu_a_full) : similar(ALu_a_full)
    MPI.Bcast!(ALu_d_my, grid.r1, grid.row_comm)
    t0 = tick!("Phase 0b (row_comm Bcast ALu)", t0)

    # B: col_comm AG FL on a-axis → (a=ALL, e, f, i=slice_r2)
    FL_a_full = allgather_dim_direct(FL_local, 1, grid.col_comm)
    t0 = tick!("Phase B  (col_comm AG FL a)", t0)

    # B2: row_comm AG FL_a_full on i-axis → (a=ALL, e, f, i=ALL)
    FL_full = allgather_dim_direct(FL_a_full, 4, grid.row_comm)
    t0 = tick!("Phase B2 (row_comm AG FL i)", t0)

    # C: col_comm AG ALd on i-axis → (i=ALL, j, k, l=slice_r2)
    ALd_full = allgather_dim_direct(ALd_local, 1, grid.col_comm)
    t0 = tick!("Phase C  (col_comm AG ALd)", t0)

    # A: single einsum with FL × ALu first tree
    @tensor partial[d, g, h, l] := (
        (
            (
                (FL_full[a, e, f, i] * ALu_d_my[a, b, c, d])
                * M1[e, j, g, b, p]
            )
            * M2[f, k, h, c, p]
        )
        * ALd_full[i, j, k, l]
    )
    t0 = tick!("Phase A  single einsum", t0)

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
rank == 0 && @printf("%-22s %12s %12s %12s %10s %10s %10s\n",
    "Config", "1D (ms)", "SUMMA (ms)", "AG (ms)", "ratio S/1D", "ratio AG/1D", "rel err")
rank == 0 && println("─" ^ 100)

for (D, χ) in [(10, 256), (10, 512), (10, 768)]
    # forloop_iter=1: each rank takes one χ/nprocs slice. Avoids cuTENSOR
    # complaining when total_splits exceeds χ at small χ (the MWE point is
    # algorithm validation, not 1D forloop tuning).
    forloop_iter = 1

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

    # ─── Warmup all three paths ───
    _ = TeneT.FLmap_parallel(FL_full, ALu_full, ALd_full,
                              (M1_full, M2_full); ifparallel=true, forloop_iter)
    CUDA.synchronize()
    _ = FLmap_parallel_2D_SUMMA(FL_local, ALu_local, ALd_local,
                                 M1_full, M2_full; grid)
    CUDA.synchronize()
    _ = FLmap_parallel_2D_AG(FL_local, ALu_local, ALd_local,
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
    t_2d_summa = @elapsed for _ in 1:nrep
        FLmap_parallel_2D_SUMMA(FL_local, ALu_local, ALd_local,
                                 M1_full, M2_full; grid)
        CUDA.synchronize()
    end
    t_2d_summa /= nrep

    GC.gc(); CUDA.reclaim()
    MPI.Barrier(comm)

    # ─── 2D-AG timing ───
    t_2d_ag = @elapsed for _ in 1:nrep
        FLmap_parallel_2D_AG(FL_local, ALu_local, ALd_local,
                              M1_full, M2_full; grid)
        CUDA.synchronize()
    end
    t_2d_ag /= nrep

    # ─── Correctness check — compare both variants to 1D ───
    result_1d = TeneT.FLmap_parallel(FL_full, ALu_full, ALd_full,
                                      (M1_full, M2_full); ifparallel=true, forloop_iter)
    CUDA.synchronize()
    result_summa = FLmap_parallel_2D_SUMMA(FL_local, ALu_local, ALd_local,
                                            M1_full, M2_full; grid)
    result_ag = FLmap_parallel_2D_AG(FL_local, ALu_local, ALd_local,
                                      M1_full, M2_full; grid)
    CUDA.synchronize()

    sd = slice_of(grid.r1, M, χ)
    sl_out = slice_of(grid.r2, M, χ)
    result_1d_slice = result_1d[sd, :, :, sl_out]

    rel_err_summa = norm(Array(result_1d_slice .- result_summa)) / norm(Array(result_1d_slice))
    rel_err_ag    = norm(Array(result_1d_slice .- result_ag))    / norm(Array(result_1d_slice))
    rel_err_max = max(MPI.Allreduce(rel_err_summa, max, comm),
                      MPI.Allreduce(rel_err_ag,    max, comm))

    rank == 0 && @printf("D=%-2d χ=%-4d           %12.2f %12.2f %12.2f %10.3fx %10.3fx %10.2e\n",
        D, χ, t_1d*1000, t_2d_summa*1000, t_2d_ag*1000,
        t_2d_summa/t_1d, t_2d_ag/t_1d, rel_err_max)
    flush(stdout)

    # ─── verbose breakdown for both variants ───
    GC.gc(); CUDA.reclaim(); MPI.Barrier(comm)
    _ = FLmap_parallel_2D_SUMMA(FL_local, ALu_local, ALd_local,
                                 M1_full, M2_full; grid, verbose=true)
    CUDA.synchronize(); flush(stdout)

    GC.gc(); CUDA.reclaim(); MPI.Barrier(comm)
    _ = FLmap_parallel_2D_AG(FL_local, ALu_local, ALd_local,
                              M1_full, M2_full; grid, verbose=true)
    CUDA.synchronize(); flush(stdout)

    GC.gc(); CUDA.reclaim()
end

rank == 0 && println()
rank == 0 && println("=" ^ 70)
rank == 0 && println("Done")
rank == 0 && println("=" ^ 70)

GC.gc(); CUDA.reclaim()
MPI.Barrier(comm)
MPI.Finalize()
