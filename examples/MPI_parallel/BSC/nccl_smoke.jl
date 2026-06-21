using MPI

if !MPI.Initialized()
    MPI.Init()
end

try
    using CUDA

    wrapper = get(
        ENV,
        "TENET_NCCL_WRAPPER",
        normpath(joinpath(@__DIR__, "..", "..", "..", "src", "contraction", "parallel", "nccl_wrapper.jl")),
    )
    include(wrapper)

    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    @assert nprocs == 2 "nccl_smoke.jl expects exactly 2 ranks"
    @assert get(ENV, "TENET_USE_NCCL", "0") == "1"
    @assert CUDA.functional()

    CUDA.device!(0)
    println(
        "rank=", rank,
        " size=", nprocs,
        " visible=", get(ENV, "CUDA_VISIBLE_DEVICES", ""),
        " device=", CUDA.name(CUDA.device()),
    )

    MPI.Barrier(comm)

    buf = CUDA.fill(Float32(rank + 1), 16)
    _nccl_allreduce!(buf, comm)
    CUDA.synchronize()
    @assert all(Array(buf) .== Float32(3))

    send = CUDA.fill(Float32(10 + rank), 4)
    gathered = CUDA.zeros(Float32, 8)
    _nccl_allgather_equal!(send, gathered, comm)
    CUDA.synchronize()
    @assert Array(gathered) == Float32[10, 10, 10, 10, 11, 11, 11, 11]

    scatter_send = CuArray(Float32[
        rank + 1, rank + 1, rank + 1, rank + 1,
        10 * (rank + 1), 10 * (rank + 1), 10 * (rank + 1), 10 * (rank + 1),
    ])
    scattered = CUDA.zeros(Float32, 4)
    _nccl_reduce_scatter_equal!(scatter_send, scattered, comm)
    CUDA.synchronize()
    expected_scatter = rank == 0 ? fill(Float32(3), 4) : fill(Float32(30), 4)
    @assert Array(scattered) == expected_scatter

    MPI.Barrier(comm)
    rank == 0 && println("BSC 2-GPU NCCL smoke passed")
finally
    if MPI.Initialized() && !MPI.Finalized()
        MPI.Finalize()
    end
end
