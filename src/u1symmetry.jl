import Base: ==, +, -, *, /, ≈, size, reshape, permutedims, transpose, conj, show, similar, adjoint, copy, sqrt, getindex, setindex!, Array, broadcasted, vec, map, ndims, indexin, sum, zero
using BitBasis
using CUDA
import CUDA: CuArray
import LinearAlgebra: tr, norm, dot, rmul!, axpy!, mul!, diag, Diagonal, lmul!, axpby!
import OMEinsum: _compactify!, subindex, einsum, Tr, Repeat, tensorpermute
import Zygote: accum
export U1Array, U1reshape, U1reshapeinfo
export randU1, asU1Array, asArray, getqrange, getbulkdims
export dtr

"""
    U1Array{T, N}

a struct to hold the N-order U1 tensors
- `qn`(`quantum number`): `N`-length Array
- `dir`(`out or in`): +1 or -1
- `tensor`: continuous storage block tensor ordered by qn
- `size`  : original none-symetric array size
- `dims` size of `tensor`
- `division`: division location for reshape
"""
struct U1Array{T, N} <: AbstractSymmetricArray{T,N}
    qn::Vector{Vector{Int}}
    dir::Vector{Int}
    tensor::AbstractArray{T}
    size::Tuple{Vararg{Int, N}}
    dims::Vector{Vector{Int}}
    division::Int
    function U1Array(qn::Vector{Vector{Int}}, dir::Vector{Int}, tensor::AbstractArray{T}, size::Tuple{Vararg{Int, N}}, dims::Vector{Vector{Int}}, division::Int) where {T,N}
        new{T, N}(qn, dir, tensor, size, dims, division)
    end
end

size(A::U1Array) = A.size
size(A::U1Array, a) = size(A)[a]
getdir(A::U1Array) = A.dir
conj(A::U1Array) = U1Array(A.qn, -A.dir, conj(A.tensor), A.size, A.dims, A.division)
map(conj, A::U1Array) = conj(A)
norm(A::U1Array) = norm(A.tensor)

*(A::U1Array, B::Number) = U1Array(A.qn, A.dir, A.tensor * B, A.size, A.dims, A.division)
*(B::Number, A::U1Array) = A * B
/(A::U1Array, B::Number) = U1Array(A.qn, A.dir, A.tensor / B, A.size, A.dims, A.division)
broadcasted(*, A::U1Array, B::Number) = U1Array(A.qn, A.dir, A.tensor .* B, A.size, A.dims, A.division)
broadcasted(*, B::Number, A::U1Array) = U1Array(A.qn, A.dir, A.tensor .* B, A.size, A.dims, A.division)
broadcasted(/, A::U1Array, B::Number) = A / B

function +(A::U1Array, B::U1Array)
    @assert A.qn == B.qn
    U1Array(A.qn, A.dir, A.tensor + B.tensor, A.size, A.dims, A.division)
    # if B.qn == A.qn
    #     U1Array(B.qn, B.dir, A.tensor + B.tensor, B.size, B.dims, B.division)
    # else
    #     qn = intersect(A.qn, B.qn)
    #     tensor = A.tensor[indexin(qn, A.qn)] + B.tensor[indexin(qn, B.qn)]
    #     extraqn = setdiff(A.qn, B.qn)            # setdiff result is dependent on order
    #     if length(extraqn) !== 0
    #         push!(qn, extraqn...)
    #         push!(tensor, A.tensor[indexin(extraqn, A.qn)]...)
    #     end
    #     extraqn = setdiff(B.qn, A.qn)
    #     if length(extraqn) !== 0
    #         push!(qn, extraqn...)
    #         push!(tensor, B.tensor[indexin(extraqn, B.qn)]...)
    #     end

    #     dims = map(x -> collect(size(x)), tensor)
    #     U1Array(qn, A.dir, tensor, A.size, dims, A.division)
    # end
end

function -(A::U1Array, B::U1Array)
    @assert A.qn == B.qn
    U1Array(A.qn, A.dir, A.tensor - B.tensor, A.size, A.dims, A.division)
    # if B.qn == A.qn
    #     U1Array(B.qn, B.dir, A.tensor - B.tensor, B.size, B.dims, B.division)
    # else
    #     atype = typeof(A.tensor[1])
    #     qn = intersect(A.qn, B.qn)
    #     tensor = Vector{atype}(A.tensor[indexin(qn, A.qn)] - B.tensor[indexin(qn, B.qn)])
    #     extraqn = setdiff(A.qn, B.qn)            # setdiff result is related to order
    #     if length(extraqn) !== 0
    #         push!(qn, extraqn...)
    #         push!(tensor, A.tensor[indexin(extraqn, A.qn)]...)
    #     end
    #     extraqn = setdiff(B.qn, A.qn)
    #     if length(extraqn) !== 0
    #         push!(qn, extraqn...)
    #         push!(tensor, -B.tensor[indexin(extraqn, B.qn)]...)
    #     end

    #     dims = map(x -> collect(size(x)), tensor)
    #     U1Array(qn, A.dir, tensor, A.size, dims, A.division)
    #     # exchangeind = indexin(B.qn, A.qn)
    #     # U1Array(B.qn, B.dir, A.tensor[exchangeind] - B.tensor, B.size, B.dims, B.division)
    # end
end

-(A::U1Array) = U1Array(A.qn, A.dir, -A.tensor, A.size, A.dims, A.division)

CuArray(A::U1Array) = U1Array(A.qn, A.dir, CuArray(A.tensor), A.size, A.dims, A.division)
Array(A::U1Array) = U1Array(A.qn, A.dir, Array(A.tensor), A.size, A.dims, A.division)

function dot(A::U1Array, B::U1Array) 
    @assert A.qn == B.qn
    dot(A.tensor, B.tensor)
end

function ≈(A::U1Array{TA,NA}, B::U1Array{TB,NB}) where {TA,NA,TB,NB}
    NA != NB && throw(Base.error("$A and $B have different dimensions"))
    A.dir != B.dir && throw(Base.error("$A and $B have different directions"))
    @assert A.qn == B.qn
    A.tensor ≈ B.tensor
end

function ==(A::U1Array{TA,NA}, B::U1Array{TB,NB}) where {TA,NA,TB,NB}
    NA != NB && throw(Base.error("$A and $B have different dimensions"))
    A.dir != B.dir && throw(Base.error("$A and $B have different directions"))
    @assert A.qn == B.qn
    A.tensor == B.tensor
end

"""
    blockdiv(dims) 

get divsions of block U1 array from U1 contious storge block dims 
"""
blockdiv(dims::Vector{Vector{Int}}) = [sum(prod.(dims[1 : i - 1])) + 1 : sum(prod.(dims[1 : i])) for i in 1:length(dims)]

function show(::IOBuffer, A::U1Array)
    println("particle number: \n", A.qn)
    println("direction: \n", A.dir)
    println("dims: \n", A.dims)
    # div = blockdiv(A.dims)
    # tensor = [reshape(A.tensor[div[i]], A.dims[i]...) for i in 1:length(A.dims)]
    println("tensor: \n", A.tensor)
end

"""
    maxq(D::Int)

give the maximum `q` value for a given dimensions `D`
"""
maxq(D::Int) = floor(Int, log2(D))


"""
    minuseven!(A)

return the negative of the array with even indices
"""
minuseven(A) = (A = Array{Int}(A); A[[i % 2 == 0 for i in 1:length(A)]] .*= -1; A)

getq(s::Int...) = map(s -> [sum(bitarray(i - 1, maxq(s) + 1)) for i = 1:s], s)
getqrange(s::Tuple{Vararg{Int}}) = getqrange(s...)
getqrange(s::Int...) = (q = getq(s...); [map(q -> sort(unique(q)), q)...])

"""
    bkdims = getbulkdims(size::Int...)

distribute dims of different part dims of U1 tensor bulk by bits division
"""
getbulkdims(s::Tuple{Vararg{Int}}) = getbulkdims(s...)
function getbulkdims(s::Int...)
    q = getq(s...)
    [map(q -> [sum(q .== i) for i in sort(unique(q))], q)...]
end

function randU1(atype, dtype, s...; dir::Vector{Int}, indqn::Vector{Vector{Int}} = getqrange(s), indims::Vector{Vector{Int}} = getbulkdims(s), q::Vector{Int}=[0])
    s != Tuple(map(sum, indims)) && throw(Base.error("$s is not valid"))
    L = length(dir)
    qn = Vector{Vector{Int}}()
    dims = Vector{Vector{Int}}()
    @inbounds for i in CartesianIndices(Tuple(length.(indqn)))
        qni = [indqn[j][i.I[j]] for j in 1:L]
        if sum(qni .* dir) in q
            bulkdims = [indims[j][i.I[j]] for j in 1:L]
            push!(qn, qni)
            push!(dims, bulkdims)
        end
    end
    qns = sort(qn)
    dims = dims[indexin(qns, qn)]
    tensor = atype(rand(dtype, sum(prod.(dims))))
    U1Array(qns, dir, tensor, s, dims, 1)
end

function zerosU1(atype, dtype, s...; dir::Vector{Int}, indqn::Vector{Vector{Int}} = getqrange(s), indims::Vector{Vector{Int}} = getbulkdims(s), q::Vector{Int}=[0])
    s != Tuple(map(sum, indims)) && throw(Base.error("$s is not valid"))
    L = length(dir)
    qn = Vector{Vector{Int}}()
    dims = Vector{Vector{Int}}()
    @inbounds for i in CartesianIndices(Tuple(length.(indqn)))
        qni = [indqn[j][i.I[j]] for j in 1:L]
        if sum(qni .* dir) in q
            bulkdims = [indims[j][i.I[j]] for j in 1:L]
            push!(qn, qni)
            push!(dims, bulkdims)
        end
    end
    qns = sort(qn)
    dims = dims[indexin(qns, qn)]
    tensor = atype(zeros(dtype, sum(prod.(dims))))
    U1Array(qns, dir, tensor, s, dims, 1)
end

zero(A::U1Array) = U1Array(A.qn, A.dir, zero(A.tensor), A.size, A.dims, A.division)

function IU1(atype, dtype, D; dir::Vector{Int}, indqn::Vector{Vector{Int}} = getqrange(D, D), indims::Vector{Vector{Int}} = getbulkdims(D, D), q::Vector{Int}=[0])
    (D, D) != Tuple(map(sum, indims))
    L = length(dir)
    qn = Vector{Vector{Int}}()
    tensor = Vector{atype{dtype}}()
    dims = Vector{Vector{Int}}()
    @inbounds for i in CartesianIndices(Tuple(length.(indqn)))
        qni = [indqn[j][i.I[j]] for j in 1:L]
        if sum(qni .* dir) in q
            bulkdims = [indims[j][i.I[j]] for j in 1:L]
            push!(qn, qni)
            push!(dims, bulkdims)
            push!(tensor, atype{dtype}(I, bulkdims...))
        end
    end
    qns = sort(qn)
    perm = indexin(qns, qn)
    dims = dims[perm]
    tensor = vcat(map(vec, tensor[perm])...)
    U1Array(qns, dir, tensor, (D, D), dims, 1)
end

# getindex(A::U1Array, index::CartesianIndex) = getindex(A, index.I...)
# function getindex(A::U1Array{T,N}, index::Int...) where {T,N}
#     bits = map(x -> ceil(Int, log2(x)), size(A))
#     qn = collect(map((index, bits) -> sum(minuseven(bitarray(index - 1, bits))), index, bits))
#     # sum(qn.*Adir) != 0 && return 0.0
#     ind = findfirst(x->x in [qn], A.qn)
#     ind === nothing && return 0.0
#     qlist = [U1selection(size(A, i)) for i = 1:N]
#     qrange = getqrange(size(A)...)
#     shift = getshift(qrange)
#     position = [sum(qlist[i][qn[i] + shift[i]][1:index[i]]) for i in 1:N]
#     CUDA.@allowscalar A.tensor[ind][position...]
# end

# setindex!(A::U1Array, x::Number, index::CartesianIndex) = setindex!(A, x, index.I...)
# function setindex!(A::U1Array{T,N}, x::Number, index::Int...) where {T,N}
#     bits = map(x -> ceil(Int, log2(x)), size(A))
#     qn = collect(map((index, bits) -> sum(minuseven(bitarray(index - 1, bits))), index, bits))
#     ind = findfirst(x->x in [qn], A.qn)
#     qlist = [U1selection(size(A, i)) for i = 1:N]
#     qrange = getqrange(size(A)...)
#     shift = getshift(qrange)
#     position = [sum(qlist[i][qn[i] + shift[i]][1:index[i]]) for i in 1:N]
#     CUDA.@allowscalar A.tensor[ind][position...] = x
# end

function U1selection(indqn::Vector{Int}, indims::Vector{Int})
    maxs = sum(indims)
    mq = maxq(maxs)
    q = [sum(bitarray(i - 1, mq + 1)) for i = 1:maxs]
    [q .== i for i in sort(unique(q))]
end

function asArray(A::U1Array{T,N}; indqn::Vector{Vector{Int}} = getqrange(size(A)), indims::Vector{Vector{Int}} = getbulkdims(size(A))) where {T <: Number, N}
    atype = _arraytype(A.tensor)
    tensor = zeros(T, size(A))
    Aqn = A.qn
    Atensor = A.tensor
    qlist = [U1selection(indqn[i], indims[i]) for i = 1:N]
    div = blockdiv(A.dims)
    for i in 1:length(Aqn)
        tensor[[qlist[j][indexin([Aqn[i][j]], indqn[j])...] for j = 1:N]...] = Array(Atensor[div[i]])
    end
    atype(tensor)
end

function asArray(A::U1Array{T,N}; indqn::Vector{Vector{Int}} = getqrange(size(A)), indims::Vector{Vector{Int}} = getbulkdims(size(A))) where {T <: AbstractArray, N}
    atype = _arraytype(A.tensor[1])
    etype = eltype(A.tensor[1])
    tensor = zeros(etype, size(A))
    Aqn = A.qn
    Atensor = A.tensor
    qlist = [U1selection(indqn[i], indims[i]) for i = 1:N]
    for i in 1:length(Aqn)
        tensor[[qlist[j][indexin([Aqn[i][j]], indqn[j])...] for j = 1:N]...] = Array(Atensor[i])
    end
    atype(tensor)
end

"""
    qn = getqn(dir::Vector{Int}, indqn; q::Vector{Int}=[0])

give the qn of length L
"""
function getqn(dir::Vector{Int}, indqn::Vector{Vector{Int}}; q::Vector{Int}=[0])
    L = length(dir)
    qn = Vector{Vector{Int}}()
    @inbounds for i in CartesianIndices(Tuple(length.(indqn)))
        qni = [indqn[j][i.I[j]] for j in 1:L]
        if sum(qni .* dir) in q
            push!(qn, qni)
        end
    end
    sort!(qn)
end

# function deletezerobulk(A::U1Array)
#     nozeroind = norm.(A.tensor) .!== 0
#     U1Array(A.qn[nozeroind], A.dir, A.tensor[nozeroind], A.size, A.dims[nozeroind], A.division)
# end

# have Bugs with CUDA@v3.5.0, rely on https://github.com/JuliaGPU/CUDA.jl/issues/1304
# which is fixed in new vervion, but its allocation is abnormal
function asU1Array(A::AbstractArray{T,N}; dir::Vector{Int}, indqn::Vector{Vector{Int}} = getqrange(size(A)), indims::Vector{Vector{Int}} = getbulkdims(size(A)), q::Vector{Int}=[0]) where {T, N}
    atype = _arraytype(A)
    Aarray = Array(A)
    qlist = [U1selection(indqn[i], indims[i]) for i = 1:N]
    Aqn = getqn(dir, indqn; q = q)
    tensor = [atype(Aarray[[qlist[j][indexin([Aqn[i][j]], indqn[j])...] for j = 1:N]...]) for i in 1:length(Aqn)]
    dims = map(x -> collect(size(x)), tensor)
    tensor = vcat(map(vec, tensor)...)
    U1Array(Aqn, dir, tensor, size(A), dims, 1)
end

# # only for OMEinsum binary permutedims before reshape
permutedims(A::U1Array, perm) = tensorpermute(A, perm)
function tensorpermute(A::U1Array{T, N}, perm) where {T <: Number, N}
    length(perm) == 0 && return copy(A)
    dims = A.dims
    qn = map(x -> x[collect(perm)], A.qn)
    p = sortperm(qn)
    div = blockdiv(dims)
    tensor = A.tensor
    tensor = vcat([vec(permutedims(reshape(@view(tensor[div[i]]), dims[i]...), perm)) for i in 1:length(div)][p]...)
    dims = map(x -> x[collect(perm)], dims)
    U1Array(qn[p], A.dir[collect(perm)], tensor, A.size[collect(perm)], dims[p], A.division)
end

reshape(A::U1Array, s::Tuple{Vararg{Int}}) = reshape(A, s...)
function reshape(A::U1Array{T,N}, s::Int...) where {T <: Number,N}
    div = 1
    if length(s) < N
        sizeA = size(A)
        p = sizeA[1]
        while p != s[1]
            div += 1
            p *= sizeA[div]
        end
        return U1Array(A.qn, A.dir, A.tensor, A.size, A.dims, div)
    else
        return A
    end
end

"""
    *(A::U1Array{TA,NA}, B::U1Array{TB,NB}) where {TA,TB,NA,NB}

core code for U1Array product
"""
function *(A::U1Array{T, NA}, B::U1Array{T, NB}) where {T, NA, NB}
    atype = _arraytype(A.tensor)
    qn, dims, blockidims, blockjdims, blockkdims = [Vector{Vector{Int}}() for _ in 1:7]
    Aindexs, Bindexs =  [Vector() for _ in 1:2]
    Aqn, Bqn = A.qn, B.qn
    Adims, Bdims = A.dims, B.dims
    Abdiv = blockdiv(Adims)
    Bbdiv = blockdiv(Bdims)
    Adiv, Bdiv = A.division, B.division
    Atensor = [reshape(@view(A.tensor[Abdiv[i]]), prod(Adims[i][1:Adiv]), prod(Adims[i][Adiv+1:end])) for i in 1:length(Abdiv)]
    Btensor = [reshape(@view(B.tensor[Bbdiv[i]]), prod(Bdims[i][1:Bdiv]), prod(Bdims[i][Bdiv+1:end])) for i in 1:length(Bbdiv)]
    
    timesAdir = getdir(A)[Adiv+1:end]
    timesBdir = getdir(B)[1:Bdiv]
    sum(timesAdir .+ timesBdir) !== 0 && throw(Base.error("U1Array product: out and in direction not match, expect: $(-timesAdir), got: $(timesBdir)"))

    midqrange = unique!(map(qn -> sum(qn[Adiv+1:end] .* A.dir[Adiv+1:end]), Aqn))
    for q in midqrange
        timesinfo!(qn, dims, Aindexs, Bindexs, blockidims, blockjdims, blockkdims,
                        Aqn, Adiv, A.dir, size.(Atensor), Adims, Bqn, Bdiv, size.(Btensor), Bdims, q)
    end
    tensorlen = sum([sum(blockidims[i]) * sum(blockkdims[i]) for i in 1:length(blockidims)])
    tensor = atype <: Array ? zeros(T, tensorlen) : CUDA.zeros(T, tensorlen)

    p = sortperm(qn)
    pp = indexin(qn, qn[p])
    bdiv = blockdiv(dims[p])[pp]
    divs = [size(Aindexs[i], 1) * size(Bindexs[i], 2) for i in 1:length(Aindexs)]
    bdivind = [sum(divs[1:i-1]) + 1 : sum(divs[1:i]) for i in 1:length(Aindexs)]
    for i in 1:length(Aindexs)
        u1bulktimes!(tensor, Aindexs[i], Bindexs[i], blockidims[i], blockjdims[i], blockkdims[i], bdiv[bdivind[i]], Atensor, Btensor)
    end
    qn == [[]] && return Array(tensor)[]
    U1Array(qn[p], [A.dir[1:Adiv]..., B.dir[Bdiv+1:end]...], tensor, (size(A)[1:Adiv]..., size(B)[Bdiv+1:end]...), dims[p], Adiv)
end

function no_nothing_col(index)
    indexcol = Int[]
    for i in 1:size(index,1)
        for j in 1:size(index,2)
            if index[i, j] !== nothing
                push!(indexcol, index[i, j])
                break
            end
        end
    end
    indexcol
end

function no_nothing_row(index)
    indexrow = Int[]
    for j in 1:size(index,2)
        for i in 1:size(index,1)
            if index[i, j] !== nothing
                push!(indexrow, index[i, j])
                break
            end
        end
    end
    indexrow
end

"""
    u1bulktimes!(qn, tensor, A, B, q)

fill into different quantum number,  then dispatch to result tensor after product
"""
function timesinfo!(qn, dims, Aindexs, Bindexs, blockidims, blockjdims, blockkdims,
                    Aqn, Adiv, Adir, Atensorsize, Adims, Bqn, Bdiv, Btensorsize, Bdims, q)

    ind_A = [sum(Aqn[Adiv+1:end] .* Adir[Adiv+1:end]) == q for Aqn in Aqn]
    matrix_j = intersect!(map(x->x[Adiv+1:end], Aqn[ind_A]), map(x->x[1:Bdiv], Bqn))
    ind_A = [Aqn[Adiv+1:end] in matrix_j for Aqn in Aqn]
    matrix_i = unique!(map(x->x[1:Adiv], Aqn[ind_A]))
    ind_B = [Bqn[1:Bdiv] in matrix_j for Bqn in Bqn]
    sum(ind_B) == 0 && return
    matrix_k = unique!(map(x->x[Bdiv+1:end], Bqn[ind_B]))

    Aindex = indexin([[i; j] for i in matrix_i, j in matrix_j], Aqn)
    Bindex = indexin([[j; k] for j in matrix_j, k in matrix_k], Bqn)
    push!(Aindexs, Aindex)
    push!(Bindexs, Bindex)

    if nothing in Aindex
        indexcol = no_nothing_col(Aindex)
        indexrow = no_nothing_row(Aindex)
    else
        indexcol = @view Aindex[:, 1]
        indexrow = @view Aindex[1, :]
    end

    oriblockidims = map(ind -> Adims[ind][1:Adiv], indexcol)
    push!(blockidims, map(ind -> Atensorsize[ind][1], indexcol))
    push!(blockjdims, map(ind -> Atensorsize[ind][2], indexrow))

    indexrow = nothing in Bindex ? no_nothing_row(Bindex) : (@view Bindex[1, :])
    oriblockkdims = map(ind -> Bdims[ind][Bdiv+1:end], indexrow)
    push!(blockkdims, map(ind -> Btensorsize[ind][2], indexrow))

    for i in 1:length(matrix_i), k in 1:length(matrix_k)
        push!(qn, [matrix_i[i]; matrix_k[k]])
        push!(dims, [oriblockidims[i]; oriblockkdims[k]])
    end
end

function u1bulktimes!(tensor, Aindex, Bindex, blockidims, blockjdims, blockkdims, bdiv, Atensor, Btensor)
    atype = _arraytype(tensor)
    etype = eltype(Atensor[1])
    Amatrix = atype <: Array ? zeros(etype, sum(blockidims), sum(blockjdims)) : CUDA.zeros(etype, sum(blockidims), sum(blockjdims))
    for i in 1:size(Aindex, 1), j in 1:size(Aindex, 2)
        Aindex[i, j] !== nothing && (Amatrix[sum(blockidims[1:i-1])+1:sum(blockidims[1:i]), sum(blockjdims[1:j-1])+1:sum(blockjdims[1:j])] .= Atensor[Aindex[i, j]])
    end 

    Bmatrix = atype <: Array ? zeros(etype, sum(blockjdims), sum(blockkdims)) : CUDA.zeros(etype, sum(blockjdims), sum(blockkdims))
    for j in 1:size(Bindex, 1), k in 1:size(Bindex, 2)
        Bindex[j, k] !== nothing && (Bmatrix[sum(blockjdims[1:j-1])+1:sum(blockjdims[1:j]), sum(blockkdims[1:k-1])+1:sum(blockkdims[1:k])] .= Btensor[Bindex[j, k]])
    end

    Cmatrix = Amatrix * Bmatrix

    for i in 1:size(Aindex, 1), k in 1:size(Bindex, 2)
        idim, kdim = sum(blockidims[1:i-1])+1:sum(blockidims[1:i]), sum(blockkdims[1:k-1])+1:sum(blockkdims[1:k])
        tensor[bdiv[(i-1) * size(Bindex, 2) + k]] .= vec(@view(Cmatrix[idim, kdim]))
    end
end

# # for OMEinsum contract to get number
# # vec(A::U1Array) = A

transpose(A::U1Array) = U1Array(A.qn, A.dir, transpose(A.tensor), A.size, A.dims, 0)

function tr(A::U1Array{T,2}) where {T}
    qn = A.qn
    tensor = A.tensor
    dims = A.dims
    s = 0.0
    div = blockdiv(dims)
    tensor = [reshape(tensor[div[i]], dims[i]...) for i in 1:length(div)]
    @inbounds @simd for i in 1:length(qn)
        qn[i][1] == qn[i][2] && (s += tr(tensor[i]))
    end
    s
end

function _compactify!(y, x::U1Array, indexer)
    x = asArray(Array(x))
    @inbounds @simd for ci in CartesianIndices(y)
        y[ci] = x[subindex(indexer, ci.I)]
    end
    return y
end

broadcasted(*, A::U1Array, B::Base.RefValue) = U1Array(A.qn, A.dir, A.tensor .* B, A.size, A.dims, A.division)
broadcasted(*, B::Base.RefValue, A::U1Array) = U1Array(A.qn, A.dir, A.tensor .* B, A.size, A.dims, A.division)

# for ein"abab ->"(A)[]
function dtr(A::U1Array{T,4}) where {T}
    qn = A.qn
    tensor = A.tensor
    dims = A.dims
    s = 0.0
    div = blockdiv(dims)
    tensor = [reshape(tensor[div[i]], dims[i]...) for i in 1:length(div)]
    @inbounds @simd for i in 1:length(qn)
        qn[i][1] == qn[i][3] && qn[i][2] == qn[i][4] && (s += Array(ein"abab ->"(tensor[i]))[])
    end
    s
end


# for Zygote compatibility
accum(A::U1Array, B::U1Array...) = +(A, B...)

# for KrylovKit compatibility
rmul!(A::U1Array, B::Number) = (rmul!(A.tensor, B); A)

similar(A::U1Array) = U1Array(map(copy, A.qn), copy(A.dir), similar(A.tensor), A.size, A.dims, A.division)
similar(A::U1Array, atype) = U1Array(map(copy, A.qn), copy(A.dir), atype(similar(A.tensor)), A.size, A.dims, A.division)
function diag(A::U1Array)
    tensor = A.tensor
    dims = A.dims
    bdiv = blockdiv(dims) 
    tensor = [reshape(tensor[bdiv[i]], dims[i]...) for i in 1:length(bdiv)]
    CUDA.@allowscalar collect(Iterators.flatten(diag.(tensor)))
end
copy(A::U1Array) = U1Array(map(copy, A.qn), copy(A.dir), copy(A.tensor), A.size, A.dims, A.division)

function mul!(Y::U1Array, A::U1Array, B::Number)
    @assert Y.qn == A.qn
    mul!(Y.tensor, A.tensor, B)
    Y
end

function axpy!(α::Number, X::U1Array, Y::U1Array)
    @assert Y.qn == X.qn
    axpy!(α, X.tensor, Y.tensor)
    Y
end

function axpby!(α::Number, x::U1Array, β::Number, y::U1Array)
    @assert x.qn == y.qn
    axpby!(α, x.tensor, β, y.tensor)
    y
end

# # for leftorth and rightorth compatibility
Diagonal(A::U1Array) = U1Array(A.qn, A.dir, map(Diagonal, A.tensor), A.size, A.dims, A.division)
sqrt(A::U1Array) = U1Array(A.qn, A.dir, map(x->sqrt.(x), A.tensor), A.size, A.dims, A.division)
broadcasted(sqrt, A::U1Array) = sqrt(A)
function lmul!(A::U1Array, B::U1Array)
    C = A*B
    B.tensor[:] = C.tensor
    for i = 1:length(B.qn)
        B.qn[i] = C.qn[i]
        B.dims[i] = C.dims[i]
    end
    B
end

# only for order-three tensor's qr and lq
function qrpos!(A::U1Array{T,N}) where {T,N}
    Qqn = Vector{Vector{Int}}()
    Rqn = Vector{Vector{Int}}()
    atype = _arraytype(A.tensor[1])
    Qtensor = Vector{atype{T}}()
    Rtensor = Vector{atype{T}}()

    for p in unique!(map(x->sum(x[A.division+1:end] .* A.dir[A.division+1:end]), A.qn))
        # @show p
        u1bulkQR!(Qqn, Qtensor, Rqn, Rtensor, A, p)
    end
    Asize = A.size
    Adims = A.dims
    exchangeind = indexin(Qqn, A.qn)
    U1Array(Qqn, A.dir, Qtensor, Asize, Adims[exchangeind], A.division), U1Array(Rqn, [-A.dir[end], A.dir[end]], Rtensor, (Asize[end], Asize[end]), map(x -> [size(x)...], Rtensor), 1)
end

function u1bulkQR!(Qqn, Qtensor, Rqn, Rtensor, A, p)
    Atensor = A.tensor
    Aqn = A.qn
    Adiv = A.division

    # ind_A = findall(x->sum(x[Adiv+1:end] .* A.dir[Adiv+1:end]) == p, Aqn)
    ind_A = [sum(Aqn[Adiv+1:end] .* A.dir[Adiv+1:end]) == p for Aqn in Aqn]
    matrix_j = unique!(map(x->x[Adiv+1:end], Aqn[ind_A]))
    matrix_i = unique!(map(x->x[1:Adiv], Aqn[ind_A]))

    ind = indexin([[i; matrix_j[1]] for i in matrix_i], Aqn)
    Amatrix = vcat(Atensor[ind]...)
    bulkidims = [size(Atensor[i], 1) for i in ind]
    bulkjdims = [size(Amatrix, 2)]

    Q, R = qrpos!(Amatrix)
    for i in 1:length(matrix_i)
        push!(Qqn, [matrix_i[i]; matrix_j[1]])
        idim, jdim = sum(bulkidims[1:i-1])+1:sum(bulkidims[1:i]), 1:sum(bulkjdims[1])
        push!(Qtensor, Q[idim, jdim])
    end
    
    push!(Rqn, [matrix_j[1]; matrix_j[1]])
    idim, jdim = 1:sum(bulkjdims[1]), 1:sum(bulkjdims[1])
    push!(Rtensor, R[idim, jdim])
end

function lqpos!(A::U1Array{T,N}) where {T,N}
    Lqn = Vector{Vector{Int}}()
    Qqn = Vector{Vector{Int}}()
    atype = _arraytype(A.tensor[1])
    Ltensor = Vector{atype{T}}()
    Qtensor = Vector{atype{T}}()

    for p in unique!(map(x->x[1] * A.dir[1], A.qn))
        u1bulkLQ!(Lqn, Ltensor, Qqn, Qtensor, A, p)
    end
    Asize = A.size
    Adims = A.dims
    exchangeind = indexin(Qqn, A.qn)
    U1Array(Lqn, [A.dir[1], -A.dir[1]], Ltensor, (Asize[1], Asize[1]), map(x -> [size(x)...], Ltensor), 1), U1Array(Qqn, A.dir, Qtensor, Asize, Adims[exchangeind], A.division)
end

function u1bulkLQ!(Lqn, Ltensor, Qqn, Qtensor, A, p)
    Atensor = A.tensor
    Aqn = A.qn
    Adiv = A.division

    ind_A = [sum(Aqn[1] .* A.dir[1]) == p for Aqn in Aqn]
    matrix_j = unique!(map(x->x[Adiv+1:end], Aqn[ind_A]))
    matrix_i = unique!(map(x->x[1], Aqn[ind_A]))

    ind = indexin([[matrix_i[1]; j] for j in matrix_j], Aqn)
    Amatrix = hcat(Atensor[ind]...)
    bulkidims = [size(Amatrix, 1)]
    bulkjdims = [size(Atensor[i], 2) for i in ind]
    
    L, Q = lqpos!(Amatrix)

    push!(Lqn, [matrix_i[1]; matrix_i[1]])
    idim, jdim = 1:sum(bulkidims[1]), 1:sum(bulkidims[1])
    push!(Ltensor, L[idim, jdim])
    for j in 1:length(matrix_j)
        push!(Qqn, [matrix_i[1]; matrix_j[j]])
        idim, jdim = 1:sum(bulkidims[1]), sum(bulkjdims[1:j-1])+1:sum(bulkjdims[1:j])
        push!(Qtensor, Q[idim, jdim])
    end
end

# # for ' in ACCtoALAR of VUMPS
function adjoint(A::U1Array{T,N}) where {T,N}
    div = A.division 
    qn = map(x->x[[div+1:end;1:div]], A.qn)
    tensor = map(adjoint, A.tensor)
    dims = map(x -> x[[div+1:end;1:div]], A.dims)
    U1Array(qn, -A.dir[[div+1:end;1:div]], tensor, A.size[[div+1:end;1:div]], dims, N - div)
end

# only for U1 Matrix
function sysvd!(A::U1Array{T,N}) where {T,N}
    # Atensor = asArray(A)
    # Utensor, Stensor, Vtensor = sysvd!(Atensor)
    # dir = getdir(A)
    # U = asU1Array(Utensor; dir = dir, q=collect(-2:2))
    # S = asU1Array(Diagonal(Stensor); dir = dir, q=[0])
    # V = asU1Array(Vtensor; dir = dir, q=collect(-2:2))
    # return U, S, V
    tensor = A.tensor
    qn = A.qn
    div = A.division
    atype = _arraytype(tensor[1])
    Utensor = Vector{atype{T}}()
    Stensor = Vector{atype{T}}()
    Vtensor = Vector{atype{T}}()
    @inbounds @simd for t in tensor
        U, S, V = sysvd!(t)
        push!(Utensor, U)
        push!(Stensor, S)
        push!(Vtensor, V)
    end
    Nm = map(x->min(x...), A.dims)
    N1 = map((x, y) -> [x[1], y], A.dims, Nm)
    N2 = map((x, y) -> [y, x[2]], A.dims, Nm)
    Asize = A.size
    sm = min(Asize...)
    U1Array(qn, A.dir, Utensor, (Asize[1], sm), N1, div), U1Array(qn, A.dir, Stensor, (sm, sm), [[Nm[i], Nm[i]] for i in 1:length(qn)], div), U1Array(qn, A.dir, Vtensor, (sm, Asize[2]), N2, div)
end

"""
    U1reshape(A::U1Array{T, N}, a::Int...) where {T, N}

U1reshape only for shape `(D,D,D,D,D,D,D,D) <-> (D^2,D^2,D^2,D^2)` and `(χ,D,D,χ) <-> (χ,D^2,χ)`, and the high-oreder U1tensor is from randU1 or zerosU1 function.
"""
U1reshape(A::U1Array, s::Tuple{Vararg{Int}}; kwarg...) = U1reshape(A, s...; kwarg...)
function U1reshape(A::U1Array{T, N}, s::Int...; reinfo) where {T <: Number, N}
    tensor = A.tensor
    dims = A.dims
    bdiv = blockdiv(dims) 
    tensor = [reshape(tensor[bdiv[i]], dims[i]...) for i in 1:length(bdiv)]
    A = U1Array(A.qn, A.dir, tensor, A.size, dims, A.division)
    U1reshape(A, s; reinfo = reinfo)
end

function U1reshape(A::U1Array{T, N}, s::Int...; reinfo) where {T <: AbstractArray, N}
    atype = _arraytype(A.tensor[1])
    etype = eltype(A.tensor[1])
    if N > length(s)
        _, _, _, indqn, indims, _, _ = reinfo
        cA = zerosU1(Array, ComplexF64, size(A)...; dir = A.dir, indqn = indqn, indims = indims)
        qndiff = setdiff(cA.qn, A.qn)
        supind = indexin(qndiff, cA.qn)
        Aqn = [A.qn; cA.qn[supind]]
        cAbdiv = blockdiv(cA.dims)
        Atensor = [A.tensor; [reshape(cA.tensor[cAbdiv[supind[i]]], cA.dims[supind[i]]...) for i in 1:length(supind)]]
        exchangeind = indexin(cA.qn, Aqn)
        Aqn = cA.qn
        Adims = cA.dims
        Atensor = Atensor[exchangeind]
        div = division(s, size(A))
        reqn = [[sum(p[d] .* A.dir[d]) for d in div] for p in Aqn]
        redims = [[prod(dims[d]) for d in div] for dims in Adims]
        retensor = [reshape(t, s...) for (t, s) in zip(map(Array, Atensor), redims)]
        ureqn = unique(reqn)
        retensors = Vector{atype{etype}}()
        choosesilces = [[] for _ in 1:length(ureqn)]
        chooseinds = [[] for _ in 1:length(ureqn)]
        for i in 1:length(ureqn)
            q = ureqn[i]
            bulkind = findall(x->x in [q], reqn)
            oriqn = Aqn[bulkind]
        
            indqnfrom = [unique(map(x->x[div], oriqn)) for div in div]
            rebulkdims = [[prod(map((x,y,z)->x[indexin(y, z)...], indims[div[i]], indqnfrom, indqn[div[i]])) for indqnfrom in indqnfrom[i]] for i in 1:length(indqnfrom)]
            # @show indqnfrom
            # indqnfrom = [[[0, 0], [1, -1]], [[0, 0], [1, -1]], [[0, 0], [1, -1]], [[0, 0], [1, -1]]]
            # rebulkdims = [[1, 4], [1, 4], [1, 4], [1, 4]]
            silce = [[(sum(rebulkdims[1:(i-1)]) + 1) : sum(rebulkdims[1:i]) for i in 1:length(rebulkdims)] for rebulkdims in rebulkdims]
            tensor = atype(zeros(etype, map(sum, rebulkdims)...))
            for j in 1:length(bulkind)
                chooseind = [indexin([oriqn[j][div[i]]], indqnfrom[i]) for i in 1:length(div)]
                choosesilce = map((s,i)->s[i...], silce, chooseind)
                tensor[choosesilce...] = retensor[bulkind[j]]
                push!(choosesilces[i], choosesilce)
                push!(chooseinds[i], bulkind[j])
            end
            push!(retensors, tensor)
        end
        nozeroind = norm.(retensors) .!== 0
        dims = map(x -> collect(size(x)), retensors)[nozeroind]
        dir = [A.dir[d][end] for d in div]     # last dir of reshape
        qn = map(qn->qn .* dir, ureqn)[nozeroind]
        p = sortperm(qn)
        tensor = atype(vcat(map(vec, retensors[nozeroind][p])...))
        U1Array(qn[p], dir, tensor, s, dims[p], 1), (choosesilces, chooseinds, A.dir, indqn, indims, Aqn, Adims)
    else
        choosesilces, chooseinds, redir, indqn, indims, reqn, redims = reinfo
        retensors = Array{Array,1}(undef, sum(length.(chooseinds)))
        div = division(size(A), s)
        ureqn = unique([[sum(p[d] .* redir[d]) for d in div] for p in reqn])
        exchangeind = indexin(ureqn, map(qn->qn .* A.dir, A.qn))
        # @show ureqn A.qn exchangeind
        Atensor = A.tensor[exchangeind]
        for i in 1:length(choosesilces)
            for j in 1:length(choosesilces[i])
                retensors[chooseinds[i][j]] = reshape(Array(Atensor[i][choosesilces[i][j]...]), redims[chooseinds[i][j]]...)
            end
        end
        nozeroind = norm.(retensors) .!== 0
        dims = map(x -> collect(size(x)), retensors)[nozeroind]
        qn = reqn[nozeroind]
        p = sortperm(qn)
        tensor = atype(vcat(map(vec, retensors[nozeroind][p])...))
        U1Array(qn[p], redir, tensor, s, dims[p], 1), (choosesilces, chooseinds, redir, indqn, indims, reqn, redims)
    end
end

function U1reshapeinfo(s, sizeA, dir, indqn, indims)
    length(sizeA) < length(s) && throw(Base.error("$sizeA must be longer than $s"))
    div = division(s, sizeA)
    A = zerosU1(Array, ComplexF64, sizeA...; dir = dir, indqn = indqn, indims = indims)
    reqn = [[sum(p[d] .* dir[d]) for d in div] for p in A.qn]
    ureqn = unique(reqn)
    choosesilces = [[] for _ in 1:length(ureqn)]
    chooseinds = [[] for _ in 1:length(ureqn)]
    Aqn = A.qn
    for i in 1:length(ureqn)
        q = ureqn[i]
        bulkind = findall(x->x in [q], reqn)
        oriqn = Aqn[bulkind]
    
        indqnfrom = [unique(map(x->x[div], oriqn)) for div in div]
        rebulkdims = [[prod(map((x,y,z)->x[indexin(y, z)...], indims[div[i]], indqnfrom, indqn[div[i]])) for indqnfrom in indqnfrom[i]] for i in 1:length(indqnfrom)]
        # @show indqnfrom
        # indqnfrom = [[[0, 0], [1, -1]], [[0, 0], [1, -1]], [[0, 0], [1, -1]], [[0, 0], [1, -1]]]
        # rebulkdims = [[1, 4], [1, 4], [1, 4], [1, 4]]
        silce = [[(sum(rebulkdims[1:(i-1)]) + 1) : sum(rebulkdims[1:i]) for i in 1:length(rebulkdims)] for rebulkdims in rebulkdims]
        for j in 1:length(bulkind)
            chooseind = [indexin([oriqn[j][div[i]]], indqnfrom[i]) for i in 1:length(div)]
            choosesilce = map((s,i)->s[i...], silce, chooseind)
            push!(choosesilces[i], choosesilce)
            push!(chooseinds[i], bulkind[j])
        end
    end
    choosesilces, chooseinds, dir, indqn, indims, Aqn, A.dims
end