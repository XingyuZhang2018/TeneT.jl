# 16-GPU cross-node Allreduce diagnostic
# Compares: TeneT.allreduce_p2p! vs system MPI.Allreduce!
# Isolates Phase 1/2/3 timings to find where 128MB cross-node takes 44ms.

using CUDA, MPI, TeneT, Printf

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nprocs = MPI.Comm_size(comm)
CUDA.device!(0)

local_comm, _ = TeneT._get_local_comm(comm)
local_rank = MPI.Comm_rank(local_comm)
local_size = MPI.Comm_size(local_comm)
n_nodes    = nprocs ÷ local_size

if rank == 0
    println("nprocs=$nprocs  local_size=$local_size  n_nodes=$n_nodes")
    println("host of rank 0: $(gethostname())")
end

function timed_avg(f, nrep=20)
    f(); CUDA.synchronize(); MPI.Barrier(comm)   # warmup
    t = @elapsed for _ in 1:nrep
        f(); CUDA.synchronize()
    end
    MPI.Barrier(comm)
    return t / nrep
end

for (label, N) in [("8MB", 1_000_000), ("128MB", 16_000_000)]
    rank == 0 && println("\n=== $label (N=$N Float64) ===")

    # TeneT 3-phase ring allreduce
    buf_t = CUDA.ones(Float64, N)
    t_tenet = timed_avg(() -> TeneT.allreduce_p2p!(buf_t, +, comm))
    rank == 0 && @printf("  TeneT  allreduce_p2p!    : %.3f ms\n", t_tenet*1e3)

    # System MPI.Allreduce on CuArray crashes on Sofia (OpenMPI coll_cuda bug).
    # Use host array as baseline instead — still gives the cross-node IB cost.
    send_h = ones(Float64, N)
    recv_h = zeros(Float64, N)
    t_mpi = timed_avg(() -> MPI.Allreduce!(send_h, recv_h, +, comm))
    rank == 0 && @printf("  system MPI.Allreduce (host): %.3f ms\n", t_mpi*1e3)

    # Manual breakdown of TeneT phases
    buf_b = CUDA.ones(Float64, N)
    slice_counts = TeneT.split_count(N, local_size)
    slice_displs = cumsum([0; slice_counts[1:end-1]])
    prev_l, next_l = TeneT._ring_neighbors(local_rank, local_size)

    # Phase 1 timing: intra-node ring reduce-scatter
    TeneT._ensure_buf!(TeneT._comm_recvbuf, buf_b, maximum(slice_counts))
    function phase1()
        for step in 1:local_size-1
            send_idx = mod(local_rank - step + 1, local_size) + 1
            recv_idx = mod(local_rank - step,     local_size) + 1
            send_sub = view(buf_b, (slice_displs[send_idx]+1):(slice_displs[send_idx]+slice_counts[send_idx]))
            recv_sub = view(TeneT._comm_recvbuf[], 1:slice_counts[recv_idx])
            CUDA.synchronize()
            rs = MPI.Isend(send_sub,  local_comm; dest=next_l,   tag=1000+step)
            rr = MPI.Irecv!(recv_sub, local_comm; source=prev_l, tag=1000+step)
            MPI.Waitall([rs, rr])
            view(buf_b, (slice_displs[recv_idx]+1):(slice_displs[recv_idx]+slice_counts[recv_idx])) .+= recv_sub
        end
    end
    t_p1 = timed_avg(phase1)
    rank == 0 && @printf("  Phase 1 intra-node RS    : %.3f ms\n", t_p1*1e3)

    # Phase 2 only: per-local-rank sibling ring allreduce on "my slice"
    my_slice = mod(local_rank + 1, local_size) + 1
    slice_view = view(buf_b, (slice_displs[my_slice]+1):(slice_displs[my_slice]+slice_counts[my_slice]))
    sib_comm = TeneT._per_local_rank_comm(comm, local_comm)
    sib_size = MPI.Comm_size(sib_comm)
    sib_rank = MPI.Comm_rank(sib_comm)
    rank == 0 && @printf("  (sib_size=%d sib slice=%.1f MB)\n", sib_size, length(slice_view)*8/1024^2)
    if sib_size > 1
        function phase2()
            TeneT._allreduce_ring_on_slice!(slice_view, sib_comm, sib_rank, sib_size)
        end
        t_p2 = timed_avg(phase2)
        rank == 0 && @printf("  Phase 2 cross-node ring  : %.3f ms\n", t_p2*1e3)

        # Baseline: system MPI.Allreduce on the slice (host memory, cross-node IB only)
        send_sh = ones(Float64, length(slice_view))
        recv_sh = zeros(Float64, length(slice_view))
        t_sib_sys = timed_avg(() -> MPI.Allreduce!(send_sh, recv_sh, +, sib_comm))
        rank == 0 && @printf("  Phase 2 via system MPI (host): %.3f ms\n", t_sib_sys*1e3)
    end
end

MPI.Finalize()
