using Test
using JLD2
using MPI
using TeneT
using TeneT: Slice2DGrid, StructArray, VUMPSRuntime,
             scatter_struct, save_rt_slice2d, load_rt_slice2d

if !MPI.Initialized()
    MPI.Init()
end

_fake_slice2d_grid(N1, N2, r1, r2) =
    Slice2DGrid(N1, N2, r1, r2, r1 * N2 + r2, MPI.COMM_SELF, MPI.COMM_SELF, MPI.COMM_SELF)

function _seq_tensor(::Type{T}, dims, offset) where {T}
    return reshape(collect(T, offset .+ (1:prod(dims))), dims...)
end

function _seq_structarray(::Type{T}, dims, offset) where {T}
    pattern = [1 2; 2 1]
    data = [_seq_tensor(T, dims, offset), _seq_tensor(T, dims, offset + 10_000)]
    return StructArray(data, pattern)
end

function _full_test_runtime(offset=0)
    χ, D = 6, 2
    return VUMPSRuntime(
        _seq_structarray(Float64, (χ, D, D, χ), offset + 1),
        _seq_structarray(Float64, (χ, D, D, χ), offset + 2),
        _seq_structarray(Float64, (χ, χ), offset + 3),
        _seq_structarray(Float64, (χ, D, D, χ), offset + 4),
        _seq_structarray(Float64, (χ, D, D, χ), offset + 5),
    )
end

_scatter_rt(rt::VUMPSRuntime, grid::Slice2DGrid) =
    VUMPSRuntime(scatter_struct(rt.AL, grid),
                 scatter_struct(rt.AR, grid),
                 rt.C,
                 scatter_struct(rt.FL, grid),
                 scatter_struct(rt.FR, grid))

mutable struct _Slice2DFinalizeParams
    folder::String
    reuse_env::Bool
    ifsave_env::Bool
    save_every::Int
    show_every::Int
    verbosity::Int
    boundary_alg
    imag_tol::Float64
    last_stop_reason::Symbol
    last_stop_χ::Int
    last_stop_eimag::Float64
end

mutable struct _Slice2DLoadParams <: TeneT.iPEPSOptimize
    folder::String
    ifload_env::Bool
    boundary_alg
    verbosity::Int
end

function _test_same_rt(got::VUMPSRuntime, expected::VUMPSRuntime)
    for field in (:AL, :AR, :C, :FL, :FR)
        got_sa = getfield(got, field)
        expected_sa = getfield(expected, field)
        @test got_sa.pattern == expected_sa.pattern
        @test length(got_sa.data) == length(expected_sa.data)
        for k in eachindex(got_sa.data)
            @test got_sa.data[k] == expected_sa.data[k]
        end
    end
end

@testset "Slice2D environment shard IO" begin
    full_rt = _full_test_runtime()
    folder = mktempdir()
    file = "χ6.slice2d.jld2"

    for r1 in 0:1, r2 in 0:1
        old_grid = _fake_slice2d_grid(2, 2, r1, r2)
        save_rt_slice2d(folder, _scatter_rt(full_rt, old_grid), old_grid; file)
    end

    checkpoint_path = joinpath(folder, file)
    @test isfile(checkpoint_path)
    @test !isdir(checkpoint_path)
    jldopen(checkpoint_path, "r"; iotype=IOStream) do f
        @test f["format"] == "TeneT.Slice2DEnvSingle.v1"
        @test haskey(f, "AL/data/1")
        @test haskey(f, "C/data/1")
        @test size(f["AL/data/1"]) == size(full_rt.AL.data[1])
        @test f["C/data/1"] == full_rt.C.data[1]
    end

    loaded_full = load_rt_slice2d(folder, Array, _fake_slice2d_grid(1, 1, 0, 0); file)
    _test_same_rt(loaded_full, full_rt)

    for r2 in 0:1
        new_grid = _fake_slice2d_grid(1, 2, 0, r2)
        loaded_blk = load_rt_slice2d(folder, Array, new_grid; file)
        _test_same_rt(loaded_blk, _scatter_rt(full_rt, new_grid))
    end

    rt_tuple = (full_rt, _full_test_runtime(100_000))
    tuple_file = "χ6_tuple.slice2d.jld2"
    for r1 in 0:1, r2 in 0:1
        old_grid = _fake_slice2d_grid(2, 2, r1, r2)
        save_rt_slice2d(folder, (_scatter_rt(rt_tuple[1], old_grid),
                                 _scatter_rt(rt_tuple[2], old_grid)), old_grid; file=tuple_file)
    end
    loaded_tuple = load_rt_slice2d(folder, Array, _fake_slice2d_grid(1, 1, 0, 0); file=tuple_file)
    @test loaded_tuple isa Tuple{VUMPSRuntime,VUMPSRuntime}
    _test_same_rt(loaded_tuple[1], rt_tuple[1])
    _test_same_rt(loaded_tuple[2], rt_tuple[2])
end

@testset "Slice2D finalize saves environment shards" begin
    folder = mktempdir()
    grid = _fake_slice2d_grid(1, 1, 0, 0)
    rt = _full_test_runtime()
    params = _Slice2DFinalizeParams(
        folder,
        false,
        true,
        0,
        1,
        0,
        VUMPS(General(); grid),
        Inf,
        :not_started,
        0,
        0.0,
    )

    x = zeros(Float64, 2)
    g = ones(Float64, 2)
    TeneT._finalize!(x, 0.5, g, 1, rt, deepcopy(rt), 2, 6, params, time(), [1.0, 1.0, 0.0, 0.0])

    env_dir = joinpath(folder, "D2", "environment")
    @test isfile(joinpath(env_dir, "χ6.slice2d.jld2"))
    loaded = load_rt_slice2d(env_dir, Array, grid; file="χ6.slice2d.jld2")
    _test_same_rt(loaded, rt)
end

@testset "initialize_env loads Slice2D shard checkpoints" begin
    folder = mktempdir()
    grid = _fake_slice2d_grid(1, 1, 0, 0)
    rt = _full_test_runtime()
    env_dir = joinpath(folder, "D2", "environment")
    save_rt_slice2d(env_dir, rt, grid; file="χ6.slice2d.jld2")

    params = _Slice2DLoadParams(folder, true, VUMPS(General(); grid), 0)
    loaded = TeneT.initialize_env(zeros(Float64, 1), 2, 6, params)
    _test_same_rt(loaded, rt)
end
