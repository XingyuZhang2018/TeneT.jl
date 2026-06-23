using Test
using JLD2
using MPI
using TeneT
using TeneT: jld2_mpi_read_slab, jld2_mpi_write_slab!

if !MPI.Initialized()
    MPI.Init()
end

const COMM = MPI.COMM_WORLD
const RANK = MPI.Comm_rank(COMM)
const NPROCS = MPI.Comm_size(COMM)

@testset "JLD2 MPI slab payload write" begin
    @test NPROCS == 2

    path = joinpath(tempdir(), "tenet_jld2_mpi_slab_$(getpid()).jld2")
    path_buf = Vector{UInt8}(undef, 512)
    if RANK == 0
        path_bytes = codeunits(path)
        fill!(path_buf, 0)
        path_buf[1:length(path_bytes)] .= path_bytes
    end
    MPI.Bcast!(path_buf, 0, COMM)
    nul = findfirst(==(0x00), path_buf)
    shared_path = String(path_buf[1:nul-1])

    dims = (6,)
    data_address = Ref{Int64}(0)
    if RANK == 0
        isfile(shared_path) && rm(shared_path)
        jldopen(shared_path, "w"; iotype=IOStream) do f
            f["A"] = zeros(Int32, dims)
        end
        data_address[] = jldopen(shared_path, "r"; iotype=IOStream) do f
            dset = JLD2.get_dataset(f, "A")
            ad = JLD2.ArrayDataset(dset)
            Int64(getfield(ad, :data_address))
        end
    end
    MPI.Bcast!(data_address, 0, COMM)
    MPI.Barrier(COMM)

    ranges = RANK == 0 ? (1:3,) : (4:6,)
    values = Int32[RANK + 1, RANK + 11, RANK + 21]
    jld2_mpi_write_slab!(shared_path, data_address[], dims, ranges, values; comm=COMM)
    MPI.Barrier(COMM)

    @test jld2_mpi_read_slab(shared_path, data_address[], Int32, dims, ranges; comm=COMM) == values
    MPI.Barrier(COMM)

    if RANK == 0
        @test load(shared_path, "A"; iotype=IOStream) == Int32[1, 11, 21, 2, 12, 22]
        rm(shared_path; force=true)
    end
end
