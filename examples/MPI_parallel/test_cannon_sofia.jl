# Sofia 4×H200 Cannon FLmap validation: parity vs the slice path, per-rank
# memory accounting, timing. Launched by Sofia/submit_test_cannon.sh
# (mpirun -np 4, CUDA_VISIBLE_DEVICES pins one GPU per rank).
using MPI, CUDA, Zygote, LinearAlgebra, Random, Printf
using TeneT
using TeneT: cannon_grid, cannon_scatter, cannon_gather, FLmap_cannon,
             FLmap, FLmap_parallel, split_ranges

MPI.Init()
const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
@assert MPI.Comm_size(comm) == 4

const χ = parse(Int, get(ENV, "TENET_CANNON_CHI", "400"))
const D = parse(Int, get(ENV, "TENET_CANNON_D", "10"))
report(s...) = rank == 0 && println(s...)
mem_used_gb() = (CUDA.total_memory() - CUDA.available_memory()) / 2^30
function mem_line(label)
    m = mem_used_gb()
    mmax = MPI.Allreduce(m, MPI.MAX, comm)
    rank == 0 && @printf("[mem]  %-32s rank0 %6.2f GB   max %6.2f GB\n", label, m, mmax)
end
function timed(f, label, n)
    f(); CUDA.synchronize(); MPI.Barrier(comm)        # warm
    t = MPI.Wtime()
    for _ in 1:n; f(); end
    CUDA.synchronize(); MPI.Barrier(comm)
    dt = (MPI.Wtime() - t) / n
    dtmax = MPI.Allreduce(dt, MPI.MAX, comm)
    rank == 0 && @printf("[time] %-32s %8.3f s/call (max over ranks)\n", label, dtmax)
end

report("=== Cannon FLmap Sofia test: χ=$χ D=$D, 2×2 grid ===")
Random.seed!(42)   # identical tensors on every rank
FL  = CuArray(rand(ComplexF64, χ, D, D, χ))
ALu = CuArray(rand(ComplexF64, χ, D, D, χ))
ALd = CuArray(rand(ComplexF64, χ, D, D, χ))
M1  = CuArray(rand(ComplexF64, D, D, D, D, 2))
M2  = CuArray(rand(ComplexF64, D, D, D, D, 2))
W   = CuArray(rand(ComplexF64, χ, D, D, χ))
g = cannon_grid(2, 2)
mem_line("tensors allocated")

# ── slice baseline ──
loss_slice(FL, ALu, ALd, M1, M2) = real(sum(W .* FLmap_parallel(
    FL, ALu, ALd, (M1, M2); ifparallel = true, forloop_iter = 1)))
r_slice = FLmap_parallel(FL, ALu, ALd, (M1, M2); ifparallel = true, forloop_iter = 1)
l_s, back_s = Zygote.pullback(loss_slice, FL, ALu, ALd, M1, M2)
g_s = back_s(1.0)
mem_line("slice fwd+bwd (peak retained)")
timed(() -> FLmap_parallel(FL, ALu, ALd, (M1, M2); ifparallel = true, forloop_iter = 1),
      "slice forward", 3)
timed(() -> Zygote.pullback(loss_slice, FL, ALu, ALd, M1, M2)[2](1.0),
      "slice fwd+bwd", 2)
GC.gc(); CUDA.reclaim(); mem_line("after reclaim")

# ── cannon ──
blk = cannon_scatter(FL, g)
loss_can(FL_b, ALu, ALd, M1, M2) = real(sum(W .* cannon_gather(
    FLmap_cannon(FL_b, ALu, ALd, (M1, M2), g), g)))
out_blk = FLmap_cannon(blk, ALu, ALd, (M1, M2), g)
r_can = cannon_gather(out_blk, g)
l_c, back_c = Zygote.pullback(loss_can, blk, ALu, ALd, M1, M2)
g_c = back_c(1.0)
mem_line("cannon fwd+bwd (peak retained)")
timed(() -> FLmap_cannon(blk, ALu, ALd, (M1, M2), g), "cannon forward", 3)
timed(() -> Zygote.pullback(loss_can, blk, ALu, ALd, M1, M2)[2](1.0),
      "cannon fwd+bwd", 2)

# ── parity ──
err_f = norm(r_can - r_slice) / norm(r_slice)
report(@sprintf("[parity] forward  rel err = %.2e  (PASS ≤ 1e-10: %s)",
                err_f, err_f <= 1e-10))
@assert isapprox(l_c, l_s; rtol = 1e-10)
# dFL: cannon returns the block gradient; compare against the slice of slice-path dFL
a_rs = split_ranges(χ, 2); i_rs = split_ranges(χ, 2)
dFL_slice_blk = g_s[1][a_rs[g.r1 + 1], :, :, i_rs[g.r2 + 1]]
errs = Float64[
    norm(g_c[1] - dFL_slice_blk) / norm(dFL_slice_blk),
    norm(g_c[2] - g_s[2]) / norm(g_s[2]),
    norm(g_c[3] - g_s[3]) / norm(g_s[3]),
    norm(g_c[4] - g_s[4]) / norm(g_s[4]),
    norm(g_c[5] - g_s[5]) / norm(g_s[5]),
]
err_max = MPI.Allreduce(maximum(errs), MPI.MAX, comm)
report(@sprintf("[parity] gradient max rel err = %.2e  (PASS ≤ 1e-8: %s)",
                err_max, err_max <= 1e-8))
report(err_f <= 1e-10 && err_max <= 1e-8 ? "=== RESULT: PASS ===" : "=== RESULT: FAIL ===")
