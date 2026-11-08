using Test
using JLD2
using MPI
using TeneT
using TeneT: StructArray, VUMPSRuntime, scatter_struct,
             slice2d_grid, save_rt_slice2d, load_rt_slice2d

if !MPI.Initialized()
    MPI.Init()
end

const COMM = MPI.COMM_WORLD
const RANK = MPI.Comm_rank(COMM)
const NPROCS = MPI.Comm_size(COMM)

function _seq_tensor(::Type{T}, dims, offset) where {T}
    return reshape(collect(T, offset .+ (1:prod(dims))), dims...)
end

function _seq_structarray(::Type{T}, dims, offset) where {T}
    pattern = [1 2; 2 1]
    return StructArray([_seq_tensor(T, dims, offset),
                        _seq_tensor(T, dims, offset + 10_000)], pattern)
end

function _full_test_runtime()
    χ, D = 6, 2
    return VUMPSRuntime(
        _seq_structarray(Float64, (χ, D, D, χ), 1),
        _seq_structarray(Float64, (χ, D, D, χ), 2),
        _seq_structarray(Float64, (χ, χ), 3),
        _seq_structarray(Float64, (χ, D, D, χ), 4),
        _seq_structarray(Float64, (χ, D, D, χ), 5),
    )
end

_scatter_rt(rt::VUMPSRuntime, grid) =
    VUMPSRuntime(scatter_struct(rt.AL, grid),
                 scatter_struct(rt.AR, grid),
                 rt.C,
                 scatter_struct(rt.FL, grid),
                 scatter_struct(rt.FR, grid))

function _test_same_rt(got::VUMPSRuntime, expected::VUMPSRuntime)
    for field in (:AL, :AR, :C, :FL, :FR)
        got_sa = getfield(got, field)
        expected_sa = getfield(expected, field)
        @test got_sa.pattern == expected_sa.pattern
        for k in eachindex(got_sa.data)
            @test got_sa.data[k] == expected_sa.data[k]
        end
    end
end

@testset "Slice2D single-file env IO over MPI" begin
    @test NPROCS == 4

    path_buf = Vector{UInt8}(undef, 512)
    if RANK == 0
        path = joinpath(tempdir(), "tenet_slice2d_env_single_$(getpid()).jld2")
        isfile(path) && rm(path)
        bytes = codeunits(path)
        fill!(path_buf, 0)
        path_buf[1:length(bytes)] .= bytes
    end
    MPI.Bcast!(path_buf, 0, COMM)
    nul = findfirst(==(0x00), path_buf)
    checkpoint_path = String(path_buf[1:nul-1])
    folder, file = splitdir(checkpoint_path)

    full_rt = _full_test_runtime()
    write_grid = slice2d_grid(2, 2; comm=COMM)
    save_rt_slice2d(folder, _scatter_rt(full_rt, write_grid), write_grid; file)
    MPI.Barrier(COMM)

    if RANK == 0
        @test isfile(checkpoint_path)
        jldopen(checkpoint_path, "r"; iotype=IOStream) do f
            @test f["format"] == "TeneT.Slice2DEnvSingle.v1"
            @test size(f["AL/data/1"]) == size(full_rt.AL.data[1])
            @test f["C/data/1"] == full_rt.C.data[1]
        end
    end

    loaded_same = load_rt_slice2d(folder, Array, write_grid; file)
    _test_same_rt(loaded_same, _scatter_rt(full_rt, write_grid))

    read_grid = slice2d_grid(1, 4; comm=COMM)
    loaded_reshard = load_rt_slice2d(folder, Array, read_grid; file)
    _test_same_rt(loaded_reshard, _scatter_rt(full_rt, read_grid))

    MPI.Barrier(COMM)
    RANK == 0 && rm(checkpoint_path; force=true)
end
