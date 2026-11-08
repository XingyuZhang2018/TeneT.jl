using Test
using JLD2
using TeneT
using TeneT: jld2_read_slab, jld2_write_slab!

@testset "JLD2 slab IO helpers" begin
    path = tempname() * ".jld2"
    dims = (7, 5, 4, 6)
    ranges = (2:5, 1:5, 2:3, 4:6)
    values = reshape(collect(Float64, 1:prod(length.(ranges))), length.(ranges)...)

    jldopen(path, "w"; iotype=IOStream) do f
        f["A"] = zeros(Float64, dims)
    end

    jldopen(path, "r+"; iotype=IOStream) do f
        dset = JLD2.get_dataset(f, "A")
        jld2_write_slab!(dset, ranges, values)
    end

    jldopen(path, "r"; iotype=IOStream) do f
        dset = JLD2.get_dataset(f, "A")
        @test jld2_read_slab(dset, Float64, ranges) == values
    end

    full = load(path, "A"; iotype=IOStream)
    @test full[ranges...] == values
    mask = trues(size(full))
    mask[ranges...] .= false
    @test all(full[mask] .== 0.0)
end
