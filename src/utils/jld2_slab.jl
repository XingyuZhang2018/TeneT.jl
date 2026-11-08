function _jld2_check_ranges(dims::NTuple{N, Int}, ranges::NTuple{N, UnitRange{Int}}) where {N}
    for dim in 1:N
        r = ranges[dim]
        isempty(r) && throw(ArgumentError("empty slab ranges are not supported"))
        (first(r) >= 1 && last(r) <= dims[dim]) ||
            throw(ArgumentError("slab range $r is outside dimension $dim with extent $(dims[dim])"))
    end
    return ntuple(dim -> length(ranges[dim]), N)
end

function _jld2_check_values(values::AbstractArray{T, N}, ranges::NTuple{N, UnitRange{Int}}) where {T, N}
    expected = ntuple(dim -> length(ranges[dim]), N)
    size(values) == expected ||
        throw(DimensionMismatch("slab values have size $(size(values)); expected $expected"))
    return expected
end

function _jld2_linear_strides(dims::NTuple{N, Int}) where {N}
    return ntuple(dim -> dim == 1 ? 1 : prod(dims[i] for i in 1:(dim - 1)), N)
end

function _jld2_tail_indices(::Val{1}, ranges::NTuple{1, UnitRange{Int}})
    return (CartesianIndex(),)
end

function _jld2_tail_indices(::Val{N}, ranges::NTuple{N, UnitRange{Int}}) where {N}
    return CartesianIndices(ntuple(dim -> 1:length(ranges[dim + 1]), N - 1))
end

function _jld2_segment_offset(
    data_address::Integer,
    elem_size::Integer,
    dims::NTuple{N, Int},
    ranges::NTuple{N, UnitRange{Int}},
    tail::NTuple{M, Int},
) where {N, M}
    strides = _jld2_linear_strides(dims)
    linear_index = first(ranges[1])
    for dim in 2:N
        linear_index += (ranges[dim][tail[dim - 1]] - 1) * strides[dim]
    end
    return Int64(data_address) + Int64(linear_index - 1) * Int64(elem_size)
end

function _jld2_same_repr_dataset(dset::JLD2.Dataset, ::Type{T}) where {T}
    ad = JLD2.ArrayDataset(dset)
    rr = getfield(ad, :rr)
    rr isa JLD2.SameRepr{T} ||
        throw(ArgumentError("direct slab IO supports only JLD2 SameRepr{$T} datasets; got $(typeof(rr))"))
    return ad, Int64(getfield(ad, :data_address)), Int(JLD2.odr_sizeof(rr)), size(ad)
end

"""
    jld2_write_slab!(dset, ranges, values)

Write `values` into `dset[ranges...]` by writing contiguous first-dimension
segments directly into the JLD2 payload. This is intended for plain contiguous
JLD2 array datasets whose on-disk representation is `SameRepr{T}`.
"""
function jld2_write_slab!(
    dset::JLD2.Dataset,
    ranges::NTuple{N, UnitRange{Int}},
    values::AbstractArray{T, N},
) where {T, N}
    ad, data_address, elem_size, dims = _jld2_same_repr_dataset(dset, T)
    getfield(ad, :writable) ||
        throw(ArgumentError("dataset is not writable; open the JLD2 file in r+ or w mode"))
    _jld2_check_ranges(dims, ranges)
    _jld2_check_values(values, ranges)

    io = getfield(getfield(ad, :f), :io)
    buf = Vector{T}(undef, length(ranges[1]))
    for tail_index in _jld2_tail_indices(Val(N), ranges)
        tail = Tuple(tail_index)
        copyto!(buf, view(values, :, tail...))
        seek(io, _jld2_segment_offset(data_address, elem_size, dims, ranges, tail))
        write(io, buf)
    end
    flush(io)
    return values
end

"""
    jld2_read_slab(dset, T, ranges)

Read `dset[ranges...]` from a plain contiguous JLD2 array dataset using the
same first-dimension segment layout as `jld2_write_slab!`.
"""
function jld2_read_slab(
    dset::JLD2.Dataset,
    ::Type{T},
    ranges::NTuple{N, UnitRange{Int}},
) where {T, N}
    out = Array{T}(undef, ntuple(dim -> length(ranges[dim]), N))
    return jld2_read_slab!(out, dset, ranges)
end

function jld2_read_slab!(
    out::AbstractArray{T, N},
    dset::JLD2.Dataset,
    ranges::NTuple{N, UnitRange{Int}},
) where {T, N}
    ad, data_address, elem_size, dims = _jld2_same_repr_dataset(dset, T)
    expected = _jld2_check_ranges(dims, ranges)
    size(out) == expected ||
        throw(DimensionMismatch("output has size $(size(out)); expected $expected"))

    io = getfield(getfield(ad, :f), :io)
    buf = Vector{T}(undef, length(ranges[1]))
    for tail_index in _jld2_tail_indices(Val(N), ranges)
        tail = Tuple(tail_index)
        seek(io, _jld2_segment_offset(data_address, elem_size, dims, ranges, tail))
        read!(io, buf)
        copyto!(view(out, :, tail...), buf)
    end
    return out
end

"""
    jld2_mpi_write_slab!(path, data_address, dims, ranges, values; comm=MPI.COMM_WORLD)

Write `values` into the payload of a pre-existing contiguous JLD2 dataset using
MPI-IO. `data_address` must be the file offset obtained from `JLD2.ArrayDataset`
for that dataset, and the dataset must use the same in-memory and on-disk
representation for element type `T`.
"""
function jld2_mpi_write_slab!(
    path::AbstractString,
    data_address::Integer,
    dims::NTuple{N, Int},
    ranges::NTuple{N, UnitRange{Int}},
    values::AbstractArray{T, N};
    comm = MPI.COMM_WORLD,
) where {T, N}
    isbitstype(T) ||
        throw(ArgumentError("MPI slab writes require an isbits element type; got $T"))
    _jld2_check_ranges(dims, ranges)
    _jld2_check_values(values, ranges)

    elem_size = sizeof(T)
    file = MPI.File.open(comm, String(path), MPI.API.MPI_MODE_RDWR[], MPI.Info())
    buf = Vector{T}(undef, length(ranges[1]))
    try
        for tail_index in _jld2_tail_indices(Val(N), ranges)
            tail = Tuple(tail_index)
            copyto!(buf, view(values, :, tail...))
            offset = _jld2_segment_offset(data_address, elem_size, dims, ranges, tail)
            MPI.File.write_at(file, offset, buf)
        end
    finally
        close(file)
    end
    return values
end

"""
    jld2_mpi_read_slab(path, data_address, T, dims, ranges; comm=MPI.COMM_WORLD)

Read `ranges` from the payload of a pre-existing contiguous JLD2 dataset using
MPI-IO. This is the read-side counterpart of `jld2_mpi_write_slab!`.
"""
function jld2_mpi_read_slab(
    path::AbstractString,
    data_address::Integer,
    ::Type{T},
    dims::NTuple{N, Int},
    ranges::NTuple{N, UnitRange{Int}};
    comm = MPI.COMM_WORLD,
) where {T, N}
    out = Array{T}(undef, ntuple(dim -> length(ranges[dim]), N))
    return jld2_mpi_read_slab!(out, path, data_address, dims, ranges; comm)
end

function jld2_mpi_read_slab!(
    out::AbstractArray{T, N},
    path::AbstractString,
    data_address::Integer,
    dims::NTuple{N, Int},
    ranges::NTuple{N, UnitRange{Int}};
    comm = MPI.COMM_WORLD,
) where {T, N}
    isbitstype(T) ||
        throw(ArgumentError("MPI slab reads require an isbits element type; got $T"))
    expected = _jld2_check_ranges(dims, ranges)
    size(out) == expected ||
        throw(DimensionMismatch("output has size $(size(out)); expected $expected"))

    elem_size = sizeof(T)
    file = MPI.File.open(comm, String(path), MPI.API.MPI_MODE_RDONLY[], MPI.Info())
    buf = Vector{T}(undef, length(ranges[1]))
    try
        for tail_index in _jld2_tail_indices(Val(N), ranges)
            tail = Tuple(tail_index)
            offset = _jld2_segment_offset(data_address, elem_size, dims, ranges, tail)
            MPI.File.read_at!(file, offset, buf)
            copyto!(view(out, :, tail...), buf)
        end
    finally
        close(file)
    end
    return out
end
