const _SLICE2D_ENV_FORMAT = "TeneT.Slice2DEnv.v1"
const _SLICE2D_ENV_SINGLE_FORMAT = "TeneT.Slice2DEnvSingle.v1"

_slice2d_env_dir(folder::AbstractString, file::AbstractString) = joinpath(folder, file)
_slice2d_env_file(folder::AbstractString, file::AbstractString) = joinpath(folder, file)
_slice2d_manifest_path(dir::AbstractString) = joinpath(dir, "manifest.jld2")
_slice2d_shard_path(dir::AbstractString, rank::Integer) = joinpath(dir, @sprintf("rank_%06d.jld2", rank))

_slice2d_kind(::VUMPSRuntime) = "VUMPSRuntime"
_slice2d_kind(::PlaquetteVUMPSRuntime) = "PlaquetteVUMPSRuntime"
_slice2d_kind(::Tuple{VUMPSRuntime,VUMPSRuntime}) = "Tuple"

_slice2d_fields(::Val{:VUMPSRuntime}) = (:AL, :AR, :C, :FL, :FR)
_slice2d_fields(::Val{:PlaquetteVUMPSRuntime}) = (:AL, :C, :FL)
_slice2d_split_field(field::Symbol) = field === :C ? false : true
_slice2d_key(prefix::AbstractString, key::AbstractString) = isempty(prefix) ? key : "$prefix/$key"

function _slice2d_kind_val(kind::AbstractString)
    kind == "VUMPSRuntime" && return Val(:VUMPSRuntime)
    kind == "PlaquetteVUMPSRuntime" && return Val(:PlaquetteVUMPSRuntime)
    throw(ArgumentError("unknown Slice2D runtime kind $kind"))
end

function _slice2d_runtime_parts(rt)
    if rt isa Tuple{VUMPSRuntime,VUMPSRuntime}
        return (("part_1", rt[1]), ("part_2", rt[2]))
    end
    return (("", rt),)
end

function _slice2d_infer_global_dims(local_dims::NTuple{N, Int}, grid::Slice2DGrid) where {N}
    comm_size = MPI.Comm_size(grid.comm)
    if comm_size == grid.N1 * grid.N2
        gathered = MPI.Allgather(collect(Int64, local_dims), grid.comm)
        first_sizes = zeros(Int, grid.N1)
        last_sizes = zeros(Int, grid.N2)
        middle = local_dims[2:end-1]
        for rank in 0:(comm_size - 1)
            r1, r2 = divrem(rank, grid.N2)
            dims = Tuple(Int.(gathered[(rank * N + 1):((rank + 1) * N)]))
            first_sizes[r1 + 1] = dims[1]
            last_sizes[r2 + 1] = dims[end]
            middle == dims[2:end-1] ||
                throw(DimensionMismatch("Slice2D shard middle dimensions differ across ranks"))
        end
        return (sum(first_sizes), middle..., sum(last_sizes))
    end

    # Unit-test fallback for fake grids backed by COMM_SELF. Production grids
    # always satisfy comm_size == N1*N2, so uneven partitions are handled above.
    return (local_dims[1] * grid.N1, local_dims[2:end-1]..., local_dims[end] * grid.N2)
end

function _slice2d_global_dims(A::AbstractArray, grid::Slice2DGrid, split::Bool)
    dims = Tuple(Int.(size(A)))
    return split ? _slice2d_infer_global_dims(dims, grid) : dims
end

function _slice2d_local_slab(global_dims::NTuple{N, Int}, grid::Slice2DGrid) where {N}
    first_range = split_ranges(global_dims[1], grid.N1)[grid.r1 + 1]
    last_range = split_ranges(global_dims[end], grid.N2)[grid.r2 + 1]
    return _slice2d_slab_ranges(Val(N), first_range, global_dims[2:end-1], last_range)
end

function _slice2d_create_empty_dataset!(f, path::AbstractString, ::Type{T}, dims::NTuple{N, Int}) where {T, N}
    isbitstype(T) || throw(ArgumentError("single-file Slice2D IO requires isbits element types; got $T"))
    datatype = JLD2.h5type(f, T, zero(T))
    dataspace = JLD2.WriteDataspace(JLD2.DS_SIMPLE, Tuple(UInt64.(reverse(dims))), ())
    dset = JLD2.create_dataset(f, path, datatype, dataspace)
    if isdefined(JLD2, :allocate_early)
        getfield(JLD2, :allocate_early)(dset, T)
    else
        nbytes = prod(Int128.(dims)) * sizeof(T)
        limit = parse(Int128, get(ENV, "TENET_SLICE2D_WINDOWS_ALLOC_LIMIT", string(1 << 30)))
        nbytes <= limit ||
            throw(ArgumentError("JLD2.allocate_early is unavailable on this platform; refusing to allocate $(nbytes) bytes for $path"))
        JLD2.write_dataset(dset, zeros(T, dims))
    end
    return nothing
end

function _slice2d_collect_single_specs(rt, grid::Slice2DGrid)
    specs = Dict{String, Vector{Int}}()
    for (prefix, part) in _slice2d_runtime_parts(rt)
        for field in _slice2d_fields(_slice2d_kind_val(_slice2d_kind(part)))
            S = getfield(part, field)
            split = _slice2d_split_field(field)
            for k in eachindex(S.data)
                data_path = _slice2d_key(prefix, "$field/data/$k")
                specs[data_path] = collect(Int, _slice2d_global_dims(Array(S.data[k]), grid, split))
            end
        end
    end
    return specs
end

function _slice2d_write_single_manifest!(f, prefix::AbstractString, rt, grid::Slice2DGrid, specs)
    isempty(prefix) || (f[_slice2d_key(prefix, "kind")] = _slice2d_kind(rt))
    for field in _slice2d_fields(_slice2d_kind_val(_slice2d_kind(rt)))
        S = getfield(rt, field)
        split = _slice2d_split_field(field)
        f[_slice2d_key(prefix, "$field/pattern")] = S.pattern
        f[_slice2d_key(prefix, "$field/length")] = length(S.data)
        f[_slice2d_key(prefix, "$field/split")] = split
        for k in eachindex(S.data)
            A = Array(S.data[k])
            data_path = _slice2d_key(prefix, "$field/data/$k")
            global_dims = Tuple(specs[data_path])
            f[_slice2d_key(prefix, "$field/global_dims/$k")] = collect(Int, global_dims)
            if split
                _slice2d_create_empty_dataset!(f, data_path, eltype(A), global_dims)
            else
                f[data_path] = A
            end
        end
    end
    return nothing
end

function _slice2d_create_single_file(path::AbstractString, rt, grid::Slice2DGrid, specs)
    parent = dirname(path)
    isdir(parent) || mkpath(parent)
    isdir(path) && throw(ArgumentError("Slice2D single-file checkpoint path is a directory: $path"))
    jldopen(path, "w"; iotype=IOStream) do f
        f["format"] = _SLICE2D_ENV_SINGLE_FORMAT
        f["kind"] = _slice2d_kind(rt)
        f["N1"] = grid.N1
        f["N2"] = grid.N2
        f["nprocs"] = grid.N1 * grid.N2
        if rt isa Tuple{VUMPSRuntime,VUMPSRuntime}
            f["length"] = 2
        end
        for (prefix, part) in _slice2d_runtime_parts(rt)
            _slice2d_write_single_manifest!(f, prefix, part, grid, specs)
        end
    end
    return nothing
end

function _slice2d_dataset_address_and_dims(path::AbstractString, dset_path::AbstractString)
    return jldopen(path, "r"; iotype=IOStream) do f
        dset = JLD2.get_dataset(f, dset_path)
        ad = JLD2.ArrayDataset(dset)
        (Int64(getfield(ad, :data_address)), Tuple(Int.(size(ad))))
    end
end

function _slice2d_write_single_payloads(path::AbstractString, prefix::AbstractString, rt, grid::Slice2DGrid)
    for field in _slice2d_fields(_slice2d_kind_val(_slice2d_kind(rt)))
        _slice2d_split_field(field) || continue
        S = getfield(rt, field)
        for k in eachindex(S.data)
            dset_path = _slice2d_key(prefix, "$field/data/$k")
            data_address, global_dims = _slice2d_dataset_address_and_dims(path, dset_path)
            ranges = _slice2d_local_slab(global_dims, grid)
            values = Array(S.data[k])
            size(values) == length.(ranges) ||
                throw(DimensionMismatch("local $dset_path has size $(size(values)); expected $(length.(ranges)) for grid rank $(grid.rank)"))
            jld2_mpi_write_slab!(path, data_address, global_dims, ranges, values; comm=grid.comm)
        end
    end
    return nothing
end

function _slice2d_write_single_payloads(path::AbstractString, rt, grid::Slice2DGrid)
    for (prefix, part) in _slice2d_runtime_parts(rt)
        _slice2d_write_single_payloads(path, prefix, part, grid)
    end
    return nothing
end

function _slice2d_write_manifest(dir::AbstractString, rt, grid::Slice2DGrid)
    jldopen(_slice2d_manifest_path(dir), "w"; iotype=IOStream) do f
        f["format"] = _SLICE2D_ENV_FORMAT
        f["kind"] = _slice2d_kind(rt)
        f["N1"] = grid.N1
        f["N2"] = grid.N2
        f["nprocs"] = grid.N1 * grid.N2
        for field in _slice2d_fields(_slice2d_kind_val(_slice2d_kind(rt)))
            S = getfield(rt, field)
            f["$field/pattern"] = S.pattern
            f["$field/length"] = length(S.data)
            f["$field/split"] = _slice2d_split_field(field)
        end
    end
end

function _slice2d_write_shard(dir::AbstractString, rt, grid::Slice2DGrid)
    jldopen(_slice2d_shard_path(dir, grid.rank), "w"; iotype=IOStream) do f
        f["format"] = _SLICE2D_ENV_FORMAT
        f["kind"] = _slice2d_kind(rt)
        f["N1"] = grid.N1
        f["N2"] = grid.N2
        f["r1"] = grid.r1
        f["r2"] = grid.r2
        f["rank"] = grid.rank
        for field in _slice2d_fields(_slice2d_kind_val(_slice2d_kind(rt)))
            split = _slice2d_split_field(field)
            (!split && grid.rank != 0) && continue
            S = getfield(rt, field)
            f["$field/pattern"] = S.pattern
            f["$field/length"] = length(S.data)
            for k in eachindex(S.data)
                f["$field/local_dims/$k"] = collect(Int, size(S.data[k]))
                f["$field/data/$k"] = Array(S.data[k])
            end
        end
    end
end

"""
    save_rt_slice2d(folder, rt, grid; file="rt.slice2d.jld2")

Save a Slice2D runtime into one JLD2 file. Rank 0 creates the metadata and
replicated fields; all ranks then write their local `AL`/`AR`/`FL`/`FR` slabs
into the shared payload with MPI-IO.
"""
function save_rt_slice2d(folder::AbstractString, rt, grid::Slice2DGrid; file::String="rt.slice2d.jld2")
    path = _slice2d_env_file(folder, file)
    specs = _slice2d_collect_single_specs(rt, grid)
    grid.rank == 0 && _slice2d_create_single_file(path, rt, grid, specs)
    MPI.Barrier(grid.comm)
    _slice2d_write_single_payloads(path, rt, grid)
    MPI.Barrier(grid.comm)
    return rt
end

function _slice2d_read_manifest(dir::AbstractString)
    path = _slice2d_manifest_path(dir)
    isfile(path) || throw(ArgumentError("Slice2D environment manifest not found: $path"))
    return jldopen(path, "r"; iotype=IOStream) do f
        format = f["format"]
        format == _SLICE2D_ENV_FORMAT ||
            throw(ArgumentError("unsupported Slice2D environment format $format"))
        (; kind = f["kind"],
           N1 = Int(f["N1"]),
           N2 = Int(f["N2"]),
           nprocs = Int(f["nprocs"]))
    end
end

function _slice2d_field_pattern(dir::AbstractString, field::Symbol)
    jldopen(_slice2d_manifest_path(dir), "r"; iotype=IOStream) do f
        return f["$field/pattern"]
    end
end

function _slice2d_field_length(dir::AbstractString, field::Symbol)
    jldopen(_slice2d_manifest_path(dir), "r"; iotype=IOStream) do f
        return Int(f["$field/length"])
    end
end

function _slice2d_field_is_split(dir::AbstractString, field::Symbol)
    jldopen(_slice2d_manifest_path(dir), "r"; iotype=IOStream) do f
        return Bool(f["$field/split"])
    end
end

function _slice2d_shard_grid(dir::AbstractString, rank::Integer)
    jldopen(_slice2d_shard_path(dir, rank), "r"; iotype=IOStream) do f
        return (; r1 = Int(f["r1"]), r2 = Int(f["r2"]))
    end
end

function _slice2d_local_dims(dir::AbstractString, rank::Integer, field::Symbol, k::Integer)
    jldopen(_slice2d_shard_path(dir, rank), "r"; iotype=IOStream) do f
        return Tuple(Int.(f["$field/local_dims/$k"]))
    end
end

function _slice2d_dataset_eltype(dir::AbstractString, rank::Integer, field::Symbol, k::Integer)
    jldopen(_slice2d_shard_path(dir, rank), "r"; iotype=IOStream) do f
        dset = JLD2.get_dataset(f, "$field/data/$k")
        return eltype(JLD2.ArrayDataset(dset))
    end
end

function _slice2d_old_layout(dir::AbstractString, meta, field::Symbol, k::Integer)
    first_sizes = zeros(Int, meta.N1)
    last_sizes = zeros(Int, meta.N2)
    middle = nothing
    ranks = Vector{NamedTuple}(undef, meta.nprocs)
    for rank in 0:(meta.nprocs - 1)
        shard_grid = _slice2d_shard_grid(dir, rank)
        dims = _slice2d_local_dims(dir, rank, field, k)
        first_sizes[shard_grid.r1 + 1] = dims[1]
        last_sizes[shard_grid.r2 + 1] = dims[end]
        middle === nothing && (middle = dims[2:end-1])
        ranks[rank + 1] = (; rank, shard_grid.r1, shard_grid.r2, dims)
    end
    return (; first_ranges = split_ranges(first_sizes),
              last_ranges = split_ranges(last_sizes),
              middle = middle === nothing ? () : middle,
              ranks)
end

function _slice2d_intersection(a::UnitRange{Int}, b::UnitRange{Int})
    lo = max(first(a), first(b))
    hi = min(last(a), last(b))
    return lo <= hi ? (lo:hi) : nothing
end

_slice2d_localize(r::UnitRange{Int}, owner::UnitRange{Int}) =
    (first(r) - first(owner) + 1):(last(r) - first(owner) + 1)

function _slice2d_slab_ranges(::Val{N}, first_range, middle, last_range) where {N}
    return ntuple(dim -> dim == 1 ? first_range :
                         (dim == N ? last_range : (1:middle[dim - 1])), N)
end

function _slice2d_load_split_tensor(dir::AbstractString, meta, field::Symbol, k::Integer, grid::Slice2DGrid)
    layout = _slice2d_old_layout(dir, meta, field, k)
    global_dims = (sum(length, layout.first_ranges), layout.middle..., sum(length, layout.last_ranges))
    N = length(global_dims)
    new_first = split_ranges(global_dims[1], grid.N1)[grid.r1 + 1]
    new_last = split_ranges(global_dims[end], grid.N2)[grid.r2 + 1]
    local_dims = (length(new_first), global_dims[2:end-1]..., length(new_last))
    T = _slice2d_dataset_eltype(dir, 0, field, k)
    out = Array{T}(undef, local_dims)
    covered = falses(local_dims)

    for shard in layout.ranks
        old_first = layout.first_ranges[shard.r1 + 1]
        old_last = layout.last_ranges[shard.r2 + 1]
        first_overlap = _slice2d_intersection(old_first, new_first)
        last_overlap = _slice2d_intersection(old_last, new_last)
        (first_overlap === nothing || last_overlap === nothing) && continue

        old_first_local = _slice2d_localize(first_overlap, old_first)
        old_last_local = _slice2d_localize(last_overlap, old_last)
        new_first_local = _slice2d_localize(first_overlap, new_first)
        new_last_local = _slice2d_localize(last_overlap, new_last)
        old_slab = _slice2d_slab_ranges(Val(N), old_first_local, layout.middle, old_last_local)
        new_slab = _slice2d_slab_ranges(Val(N), new_first_local, layout.middle, new_last_local)

        jldopen(_slice2d_shard_path(dir, shard.rank), "r"; iotype=IOStream) do f
            dset = JLD2.get_dataset(f, "$field/data/$k")
            view(out, new_slab...) .= jld2_read_slab(dset, T, old_slab)
        end
        covered[new_slab...] .= true
    end

    all(covered) || error("Slice2D environment shard set does not cover $field[$k] for rank $(grid.rank)")
    return out
end

function _slice2d_load_replicated_tensor(dir::AbstractString, field::Symbol, k::Integer)
    return jldopen(_slice2d_shard_path(dir, 0), "r"; iotype=IOStream) do f
        f["$field/data/$k"]
    end
end

function _slice2d_load_structarray(dir::AbstractString, meta, field::Symbol, grid::Slice2DGrid)
    pattern = _slice2d_field_pattern(dir, field)
    len = _slice2d_field_length(dir, field)
    split = _slice2d_field_is_split(dir, field)
    data = Vector{Array}(undef, len)
    for k in 1:len
        data[k] = split ?
            _slice2d_load_split_tensor(dir, meta, field, k, grid) :
            _slice2d_load_replicated_tensor(dir, field, k)
    end
    return StructArray(data, pattern)
end

function _slice2d_load_runtime(dir::AbstractString, meta, ::Val{:VUMPSRuntime}, grid::Slice2DGrid)
    return VUMPSRuntime(
        _slice2d_load_structarray(dir, meta, :AL, grid),
        _slice2d_load_structarray(dir, meta, :AR, grid),
        _slice2d_load_structarray(dir, meta, :C, grid),
        _slice2d_load_structarray(dir, meta, :FL, grid),
        _slice2d_load_structarray(dir, meta, :FR, grid),
    )
end

function _slice2d_load_runtime(dir::AbstractString, meta, ::Val{:PlaquetteVUMPSRuntime}, grid::Slice2DGrid)
    return PlaquetteVUMPSRuntime(
        _slice2d_load_structarray(dir, meta, :AL, grid),
        _slice2d_load_structarray(dir, meta, :C, grid),
        _slice2d_load_structarray(dir, meta, :FL, grid),
    )
end

function _slice2d_read_single_meta(path::AbstractString)
    isfile(path) || throw(ArgumentError("Slice2D single-file checkpoint not found: $path"))
    return jldopen(path, "r"; iotype=IOStream) do f
        format = f["format"]
        format == _SLICE2D_ENV_SINGLE_FORMAT ||
            throw(ArgumentError("unsupported Slice2D single-file environment format $format"))
        (; kind = f["kind"],
           N1 = Int(f["N1"]),
           N2 = Int(f["N2"]),
           nprocs = Int(f["nprocs"]))
    end
end

function _slice2d_single_field_pattern(path::AbstractString, prefix::AbstractString, field::Symbol)
    jldopen(path, "r"; iotype=IOStream) do f
        return f[_slice2d_key(prefix, "$field/pattern")]
    end
end

function _slice2d_single_field_length(path::AbstractString, prefix::AbstractString, field::Symbol)
    jldopen(path, "r"; iotype=IOStream) do f
        return Int(f[_slice2d_key(prefix, "$field/length")])
    end
end

function _slice2d_single_field_is_split(path::AbstractString, prefix::AbstractString, field::Symbol)
    jldopen(path, "r"; iotype=IOStream) do f
        return Bool(f[_slice2d_key(prefix, "$field/split")])
    end
end

function _slice2d_single_kind(path::AbstractString, prefix::AbstractString)
    jldopen(path, "r"; iotype=IOStream) do f
        return f[_slice2d_key(prefix, "kind")]
    end
end

function _slice2d_load_single_split_tensor(path::AbstractString, prefix::AbstractString, field::Symbol, k::Integer, grid::Slice2DGrid)
    dset_path = _slice2d_key(prefix, "$field/data/$k")
    return jldopen(path, "r"; iotype=IOStream) do f
        dset = JLD2.get_dataset(f, dset_path)
        ad = JLD2.ArrayDataset(dset)
        T = eltype(ad)
        global_dims = Tuple(Int.(size(ad)))
        ranges = _slice2d_local_slab(global_dims, grid)
        jld2_read_slab(dset, T, ranges)
    end
end

function _slice2d_load_single_replicated_tensor(path::AbstractString, prefix::AbstractString, field::Symbol, k::Integer)
    return jldopen(path, "r"; iotype=IOStream) do f
        f[_slice2d_key(prefix, "$field/data/$k")]
    end
end

function _slice2d_load_single_structarray(path::AbstractString, prefix::AbstractString, field::Symbol, grid::Slice2DGrid)
    pattern = _slice2d_single_field_pattern(path, prefix, field)
    len = _slice2d_single_field_length(path, prefix, field)
    split = _slice2d_single_field_is_split(path, prefix, field)
    data = Vector{Array}(undef, len)
    for k in 1:len
        data[k] = split ?
            _slice2d_load_single_split_tensor(path, prefix, field, k, grid) :
            _slice2d_load_single_replicated_tensor(path, prefix, field, k)
    end
    return StructArray(data, pattern)
end

function _slice2d_load_single_runtime(path::AbstractString, prefix::AbstractString, ::Val{:VUMPSRuntime}, grid::Slice2DGrid)
    return VUMPSRuntime(
        _slice2d_load_single_structarray(path, prefix, :AL, grid),
        _slice2d_load_single_structarray(path, prefix, :AR, grid),
        _slice2d_load_single_structarray(path, prefix, :C, grid),
        _slice2d_load_single_structarray(path, prefix, :FL, grid),
        _slice2d_load_single_structarray(path, prefix, :FR, grid),
    )
end

function _slice2d_load_single_runtime(path::AbstractString, prefix::AbstractString, ::Val{:PlaquetteVUMPSRuntime}, grid::Slice2DGrid)
    return PlaquetteVUMPSRuntime(
        _slice2d_load_single_structarray(path, prefix, :AL, grid),
        _slice2d_load_single_structarray(path, prefix, :C, grid),
        _slice2d_load_single_structarray(path, prefix, :FL, grid),
    )
end

function _slice2d_load_single(path::AbstractString, atype, grid::Slice2DGrid)
    meta = _slice2d_read_single_meta(path)
    if meta.kind == "Tuple"
        rt = (_slice2d_load_single_runtime(path, "part_1", _slice2d_kind_val(_slice2d_single_kind(path, "part_1")), grid),
              _slice2d_load_single_runtime(path, "part_2", _slice2d_kind_val(_slice2d_single_kind(path, "part_2")), grid))
        return atype.(rt)
    end
    rt = _slice2d_load_single_runtime(path, "", _slice2d_kind_val(meta.kind), grid)
    return atype(rt)
end

"""
    load_rt_slice2d(folder, atype, grid; file="rt.slice2d.jld2")

Load the Slice2D checkpoint for the current rank. Single-file checkpoints read
the current rank's slab from full datasets; legacy directory checkpoints are
still accepted as a fallback.
"""
function load_rt_slice2d(folder::AbstractString, atype, grid::Slice2DGrid; file::String="rt.slice2d.jld2")
    path = _slice2d_env_file(folder, file)
    isfile(path) && return _slice2d_load_single(path, atype, grid)

    dir = _slice2d_env_dir(folder, file)
    meta = _slice2d_read_manifest(dir)
    if meta.kind == "Tuple"
        rt = (load_rt_slice2d(dir, Array, grid; file="part_1"),
              load_rt_slice2d(dir, Array, grid; file="part_2"))
        return atype.(rt)
    end
    rt = _slice2d_load_runtime(dir, meta, _slice2d_kind_val(meta.kind), grid)
    return atype(rt)
end
