import Base: ==, +, -, *, /, ≈, size, reshape, permutedims, transpose, conj, show, similar, adjoint, copy, sqrt,  Array, broadcasted, vec, map, ndims, indexin, sum, zero
using CUDA
import CUDA: CuArray
import LinearAlgebra: tr, norm, dot, rmul!, axpy!, mul!, diag, Diagonal, lmul!
import OMEinsum: _compactify!, subindex, einsum, Tr, Repeat, tensorpermute
import Zygote: accum
export Z2Array, AbstractSymmetricArray
export randZ2, Z2reshape, Z2reshapeinfo
export parityconserving
export asZ2Array, asArray, getZ2blockdims

"""
    Z2Array{T, N}

a struct to hold the N-order Z2 tensors
- `parity`: `2^(N-1)` `N`-length `0/1` Array
- `tensor`: `2^(N-1)` bulk tensor
- `size`  : original none-symetric array size
- `dims` size of `tensor`
- `division`: division location for reshape
"""
struct Z2Array{T, N} <: AbstractSymmetricArray{T,N}
    parity::Vector{Vector{Int}}
    tensor::Vector{AbstractArray{T}}
    size::Tuple{Vararg{Int, N}}
    dims::Vector{Vector{Int}}
    division::Int
    function Z2Array(parity::Vector{Vector{Int}}, tensor::Vector{<:AbstractArray{T}}, size::Tuple{Vararg{Int, N}}, dims::Vector{Vector{Int}}, division::Int) where {T,N}
        new{T, N}(parity, tensor, size, dims, division)
    end
end

size(A::Z2Array) = A.size
size(A::Z2Array, a) = size(A)[a]
conj(A::Z2Array) = Z2Array(A.parity, map(conj, A.tensor), A.size, A.dims, A.division)
map(conj, A::Z2Array) = conj(A)
norm(A::Z2Array) = norm(A.tensor)

*(A::Z2Array, B::Number) = Z2Array(A.parity, A.tensor * B, A.size, A.dims, A.division)
*(B::Number, A::Z2Array{T,N}) where {T,N} = A * B
/(A::Z2Array{T,N}, B::Number) where {T,N} = Z2Array(A.parity, A.tensor / B, A.size, A.dims, A.division)
broadcasted(*, A::Z2Array, B::Number) = Z2Array(A.parity, A.tensor .* B, A.size, A.dims, A.division)
broadcasted(*, B::Number, A::Z2Array) = Z2Array(A.parity, A.tensor .* B, A.size, A.dims, A.division)
broadcasted(/, A::Z2Array, B::Number) = A / B

function +(A::Z2Array, B::Z2Array)
    if B.parity == A.parity
        Z2Array(B.parity, A.tensor + B.tensor, B.size, B.dims, B.division)
    else
        exchangeind = indexin(B.parity, A.parity)
        Z2Array(B.parity, A.tensor[exchangeind] + B.tensor, B.size, B.dims, B.division)
    end
end

function -(A::Z2Array, B::Z2Array)
    if B.parity == A.parity
        Z2Array(B.parity, A.tensor - B.tensor, B.size, B.dims, B.division)
    else
        exchangeind = indexin(B.parity, A.parity)
        Z2Array(B.parity, A.tensor[exchangeind] - B.tensor, B.size, B.dims, B.division)
    end
end

-(A::Z2Array) = Z2Array(A.parity, map(-, A.tensor), A.size, A.dims, A.division)

CuArray(A::Z2Array) = Z2Array(A.parity, map(CuArray, A.tensor), A.size, A.dims, A.division)
Array(A::Z2Array) = Z2Array(A.parity, map(Array, A.tensor), A.size, A.dims, A.division)

function dot(A::Z2Array, B::Z2Array) 
    if A.parity == B.parity 
        dot(A.tensor, B.tensor)
    else
        exchangeind = indexin(A.parity, B.parity)
        dot(A.tensor, B.tensor[exchangeind])
    end
end

function ≈(A::Z2Array{TA,NA}, B::Z2Array{TB,NB}) where {TA,NA,TB,NB}
    NA != NB && throw(Base.error("$A and $B have different dimensions"))
    exchangeind = indexin(A.parity, B.parity)
    A.tensor ≈ B.tensor[exchangeind]
end

function ==(A::Z2Array{TA,NA}, B::Z2Array{TB,NB}) where {TA,NA,TB,NB}
    NA != NB && throw(Base.error("$A and $B have different dimensions"))
    exchangeind = indexin(A.parity, B.parity)
    A.tensor == B.tensor[exchangeind]
end

function show(::IOBuffer, A::Z2Array)
    println("parity: \n", A.parity)
    println("dims: \n", A.dims)
    println("tensor: \n", A.tensor)
end

zero(A::Z2Array) = Z2Array(A.parity, map(zero, A.tensor), A.size, A.dims, A.division)

"""
    parityconserving(T::Array, ::Val{:electron})

Transform an arbitray tensor which has arbitray legs and each leg have index 1 or 2 into parity conserving form
"""
function parityconserving(T::Union{Array,CuArray}, siteinds)
	s = size(T)
	p = zeros(s)
	@inbounds for index in CartesianIndices(T)
		i = Tuple(index)
		sum(index_to_parity.(i, siteinds)) % 2 == 0 && (p[index] = 1)
	end
	p = _arraytype(T)(p)

	return reshape(p.*T,s...)
end

getp(s::Int...; siteinds) = map(s -> [index_to_parity(i, siteinds) for i = 1:s], s)

getZ2blockdims(s::Tuple{Vararg{Int}}; kwargs...) = getZ2blockdims(s...; kwargs...)
function getZ2blockdims(s::Int...; kwargs...)
    parity = getp(s...; kwargs...)
    [map(p -> [sum(p .== i) for i in [0,1]], parity)...]
end

"""
    parity = getqn(dir::Vector{Int}, indqn; q::Vector{Int}=[0])

give the parity of length L
"""
function getparity(N; f::Vector{Int}=[0])
    parity = Vector{Vector{Int}}()
    @inbounds for i in CartesianIndices(Tuple(2 for i=1:N))
        parityi = [[0,1][i.I[j]] for j in 1:N]
        if sum(parityi) % 2 in f
            push!(parity, parityi)
        end
    end
    sort!(parity)
end

function randZ2(atype, dtype, s...; indims::Vector{Vector{Int}}, f::Vector{Int}=[0], dir=nothing, indqn=nothing)
    s != Tuple(map(sum, indims)) && throw(Base.error("$s is not valid"))
    L = length(s)
    parity = Vector{Vector{Int}}()
    tensor = Vector{atype{dtype}}()
    dims = Vector{Vector{Int}}()
    @inbounds for i in CartesianIndices(Tuple(2 for i=1:L))
        parityi = [[0,1][i.I[j]] for j in 1:L]
        if sum(parityi) % 2 in f
            blockdims = [indims[j][i.I[j]] for j in 1:L]
            push!(parity, parityi)
            push!(dims, blockdims)
            push!(tensor, atype(rand(dtype, blockdims...)))
        end
    end
    p = sortperm(parity)
    Z2Array(parity[p], tensor[p], s, dims[p], 1)
end

function zerosZ2(atype, dtype, s...; indims::Vector{Vector{Int}}, f::Vector{Int}=[0])
    s != Tuple(map(sum, indims)) && throw(Base.error("$s is not valid"))
    L = length(s)
    parity = Vector{Vector{Int}}()
    tensor = Vector{atype{dtype}}()
    dims = Vector{Vector{Int}}()
    @inbounds for i in CartesianIndices(Tuple(2 for i=1:L))
        parityi = [[0,1][i.I[j]] for j in 1:L]
        if sum(parityi) % 2 in f
            blockdims = [indims[j][i.I[j]] for j in 1:L]
            push!(parity, parityi)
            push!(dims, blockdims)
            push!(tensor, atype(zeros(dtype, blockdims...)))
        end
    end
    p = sortperm(parity)
    Z2Array(parity[p], tensor[p], s, dims[p], 1)
end


function IZ2(atype, dtype, D; indims::Vector{Vector{Int}}, f::Vector{Int}=[0],dir=nothing, indqn=nothing)
    (D, D) != Tuple(map(sum, indims))
    parity = Vector{Vector{Int}}()
    tensor = Vector{atype{dtype}}()
    dims = Vector{Vector{Int}}()
    @inbounds for i in CartesianIndices(Tuple(2 for i=1:2))
        parityi = [[0,1][i.I[j]] for j in 1:2]
        if sum(parityi) % 2 in f
            bulkdims = [indims[j][i.I[j]] for j in 1:2]
            push!(parity, parityi)
            push!(dims, bulkdims)
            push!(tensor, atype{dtype}(I, bulkdims...))
        end
    end
    p = sortperm(parity)
    Z2Array(parity[p], tensor[p], (D, D), dims[p], 1)
end

function z2selection(maxN::Int; siteinds, dir=nothing)
    q = [index_to_parity(i, siteinds) for i = 1:maxN]
    return [(q .== 0),(q .== 1)]
end

function asArray(A::Z2Array{T,N}; kwargs...) where {T,N}
    atype = _arraytype(A.tensor[1])
    tensor = zeros(T, size(A))
    parity = A.parity
    qlist = [z2selection(size(A)[i]; kwargs...) for i = 1:N]
    for i in 1:length(parity)
        tensor[[qlist[j][parity[i][j]+1] for j = 1:N]...] = Array(A.tensor[i])
    end
    atype(tensor)
end

# have Bugs with CUDA@v3.5.0, rely on https://github.com/JuliaGPU/CUDA.jl/issues/1304
# which is fixed in new vervion, but its allocation is abnormal
function asZ2Array(A::AbstractArray{T,N}; kwargs...) where {T,N}
    atype = _arraytype(A)
    Aarray = Array(A)
    qlist = [z2selection(size(A)[i]; kwargs...) for i = 1:N]
    parity = getparity(N)
    tensor = [atype(Aarray[[qlist[j][parity[i][j]+1] for j = 1:N]...]) for i in 1:length(parity)]
    dims = map(x -> [size(x)...], tensor)
    # nozeroind = norm.(tensor) .!= 0
    Z2Array(parity, tensor, size(A), dims, 1)
end

# only for OMEinsum binary permutedims before reshape
permutedims(A::Z2Array, perm) = tensorpermute(A, perm)
function tensorpermute(A::Z2Array, perm)
    length(perm) == 0 && return copy(A)
    parity = map(x -> x[collect(perm)], A.parity)
    # exchangeind = indexin(A.parity, parity)
    tensor = map(x -> permutedims(x, perm), A.tensor)
    dims = map(x -> x[collect(perm)], A.dims)
    Z2Array(parity, tensor, A.size[collect(perm)], dims, A.division)
end

reshape(A::Z2Array, a::Tuple{Vararg{Int}}) = reshape(A, a...)
function reshape(A::Z2Array{T,N}, a::Int...) where {T,N}
    Atensor = A.tensor
    div = 1
    if length(a) < N
        sizeA = size(A)
        p = sizeA[1]
        while p != a[1]
            div += 1
            p *= sizeA[div]
        end
        s = map(size, Atensor)
        tensor = map((x, y) -> reshape(x, prod(y[1:div]), prod(y[div+1:end])), Atensor, s)
        return Z2Array(A.parity, tensor, A.size, A.dims, div)
    else
        tensor = map((x, y) -> reshape(x, y...), Atensor, A.dims)
        return Z2Array(A.parity, tensor, A.size, A.dims, A.division)
    end
end

"""
    *(A::Z2Array{TA,NA}, B::Z2Array{TB,NB}) where {TA,TB,NA,NB}

core code for Z2Array product
"""
function *(A::Z2Array{TA,NA}, B::Z2Array{TB,NB}) where {TA,TB,NA,NB}
    parity = Vector{Vector{Int}}()
    dims = Vector{Vector{Int}}()
    atype = _arraytype(B.tensor[1])
    T = TA == ComplexF64 || TB == ComplexF64 ? ComplexF64 : Float64
    tensor = Vector{atype{T}}()
    divA, divB = A.division, B.division

    bulktimes!(parity, tensor, dims, A, B, 0)
    !(divA in [0, NA]) && !(divB in [0, NB]) && bulktimes!(parity, tensor, dims, A, B, 1)
    parity == [[]] && return Array(tensor[1])[]
    p = sortperm(parity)
    Z2Array(parity[p], tensor[p], (size(A)[1:divA]..., size(B)[divB+1:end]...), dims[p], divA)
end

"""
    bulktimes!(parity, tensor, A, B, p)

fill into even and odd matrix,  p = 0 for even, p = 1 for odd, then dispatch to result tensor after product
"""
function bulktimes!(parity, tensor, dims, A, B, p)
    Aparity, Atensor = A.parity, A.tensor
    Bparity, Btensor = B.parity, B.tensor
    Adims, Bdims = A.dims, B.dims
    divA, divB = A.division, B.division
    atype = _arraytype(Btensor[1])
    etype = eltype(Btensor[1])

    ind_A = findall(x->sum(x[divA+1:end]) % 2 == p, Aparity)
    matrix_j = unique(map(x->x[divA+1:end], Aparity[ind_A]))
    matrix_i = unique(map(x->x[1:divA], Aparity[ind_A]))
    ind_B = findall(x->x[1:divB] in matrix_j, Bparity)
    matrix_k = unique(map(x->x[divB+1:end], Bparity[ind_B]))

    index = [findfirst(x->x in [[i; j]], Aparity) for i in matrix_i, j in matrix_j]
    oribulkidims = map(ind -> Adims[ind][1:divA], index[:, 1])
    bulkidims = map(ind -> size(Atensor[ind], 1), index[:, 1])
    bulkjdims = map(ind -> size(Atensor[ind], 2), index[1, :])
    # Amatrix = hvcat(ntuple(i->length(bulkjdims), length(bulkidims)), Atensor[index']...)
    Amatrix = atype <: Array ? zeros(etype, sum(bulkidims), sum(bulkjdims)) : CUDA.zeros(etype, sum(bulkidims), sum(bulkjdims))
    for i in 1:length(matrix_i), j in 1:length(matrix_j)
        Amatrix[sum(bulkidims[1:i-1])+1:sum(bulkidims[1:i]), sum(bulkjdims[1:j-1])+1:sum(bulkjdims[1:j])] = Atensor[index[i, j]]
    end

    index = [findfirst(x->x in [[j; k]], Bparity) for j in matrix_j, k in matrix_k]
    oribulkkdims = map(ind -> Bdims[ind][divB+1:end], index[1, :])
    bulkkdims = map(ind -> size(Btensor[ind], 2), index[1, :])
    # Bmatrix = hvcat(ntuple(i->length(bulkkdims), length(bulkjdims)), Btensor[index']...)
    Bmatrix = atype <: Array ? zeros(etype, sum(bulkjdims), sum(bulkkdims)) : CUDA.zeros(etype, sum(bulkjdims), sum(bulkkdims))
    for j in 1:length(matrix_j), k in 1:length(matrix_k)
        Bmatrix[sum(bulkjdims[1:j-1])+1:sum(bulkjdims[1:j]), sum(bulkkdims[1:k-1])+1:sum(bulkkdims[1:k])] = Btensor[index[j, k]]
    end
    
    C = atype(Amatrix) * atype(Bmatrix)

    for i in 1:length(matrix_i), k in 1:length(matrix_k)
        push!(parity, [matrix_i[i]; matrix_k[k]])
        push!(dims, [oribulkidims[i]; oribulkkdims[k]])
        idim, kdim = sum(bulkidims[1:i-1])+1:sum(bulkidims[1:i]), sum(bulkkdims[1:k-1])+1:sum(bulkkdims[1:k])
        push!(tensor, C[idim, kdim])
    end
end

# for OMEinsum contract to get number
vec(A::Z2Array) = A

function transpose(A::Z2Array)
    tensor = map(transpose, A.tensor)
    Z2Array(A.parity, tensor, A.size, A.dims, 0)
end

function tr(A::Z2Array{T,N}) where {T,N}
    parity = A.parity
    tensor = A.tensor
    half = Int(length(parity[1])/2)
    s = 0.0
    @inbounds for i in 1:length(parity)
        parity[i][1:half] == parity[i][half+1:end] && (s += tr(tensor[i]))
    end
    s
end

function _compactify!(y, x::Z2Array, indexer)
    x = asArray(Array(x), Val(siteinds))
    @inbounds for ci in CartesianIndices(y)
        y[ci] = x[subindex(indexer, ci.I)]
    end
    return y
end

broadcasted(*, A::Z2Array, B::Base.RefValue) = Z2Array(A.parity, A.tensor .* B, A.size, A.dims, A.division)
broadcasted(*, B::Base.RefValue, A::Z2Array) = Z2Array(A.parity, A.tensor .* B, A.size, A.dims, A.division)

# for ein"abab ->"(A)[]
function dtr(A::Z2Array{T,N}) where {T,N}
    parity = A.parity
    tensor = A.tensor
    s = 0.0
    @inbounds for i in 1:length(parity)
        parity[i][1] == parity[i][3] && parity[i][2] == parity[i][4] && (s += Array(ein"abab ->"(tensor[i]))[])
    end
    s
end


# for Zygote compatibility
accum(A::Z2Array, B::Z2Array...) = +(A, B...)

# for KrylovKit compatibility
rmul!(A::Z2Array, B::Number) = (map(x -> rmul!(x, B), A.tensor); A)

function lmul!(A::Z2Array{T,N}, B::Z2Array) where {T,N}
    C = A*B
    for i = 1:2^(N-1)
        B.tensor[i] = C.tensor[i]
    end
    return B
end

similar(A::Z2Array) = Z2Array(A.parity, map(similar, A.tensor), A.size, A.dims, A.division)
similar(A::Z2Array, atype) = Z2Array(A.parity, map(x -> atype(similar(x)), A.tensor), A.size, A.dims, A.division)
diag(A::Z2Array{T,N}) where {T,N} = CUDA.@allowscalar collect(Iterators.flatten(diag.(A.tensor)))
copy(A::Z2Array{T,N}) where {T,N} = Z2Array(A.parity, map(copy, A.tensor), A.size, A.dims, A.division)

function mul!(Y::Z2Array, A::Z2Array, B::Number) 
    if Y.parity == A.parity
        map((Y, A) -> mul!(Y, A, B), Y.tensor, A.tensor)
    else
        exchangeind = indexin(Y.parity, A.parity)
        map((Y, A) -> mul!(Y, A, B), Y.tensor, A.tensor[exchangeind])
    end
    Y
end

function axpy!(α::Number, A::Z2Array, B::Z2Array)
    if B.parity == A.parity
        map((x,y) -> axpy!(α, x, y), A.tensor, B.tensor)
    else
        exchangeind = indexin(B.parity, A.parity)
        map((x,y) -> axpy!(α, x, y), A.tensor[exchangeind], B.tensor)
    end
    return B
end

function axpby!(α::Number, x::Z2Array, β::Number, y::Z2Array)
    @assert x.parity == y.parity
    axpby!(α, x.tensor, β, y.tensor)
    y
end

# for leftorth and rightorth compatibility
Diagonal(A::Z2Array) = Z2Array(A.parity, map(Diagonal, A.tensor), A.size, A.dims, A.division)
sqrt(A::Z2Array) = Z2Array(A.parity, map(x->sqrt.(x), A.tensor), A.size, A.dims, A.division)
broadcasted(sqrt, A::Z2Array) = sqrt(A)

# only for order-three tensor's qr and lq
function qrpos!(A::Z2Array{T,N}) where {T,N}
    Qparity = Vector{Vector{Int}}()
    Rparity = Vector{Vector{Int}}()
    atype = _arraytype(A.tensor[1])
    Qtensor = Vector{atype{T}}()
    Rtensor = Vector{atype{T}}()

    Z2bulkQR!(Qparity, Qtensor, Rparity, Rtensor, A, 0)
    Z2bulkQR!(Qparity, Qtensor, Rparity, Rtensor, A, 1)
    Asize = A.size
    Adims = A.dims
    exchangeind = indexin(Qparity, A.parity)
    Z2Array(Qparity, Qtensor, Asize, Adims[exchangeind], A.division), Z2Array(Rparity, Rtensor, (Asize[end], Asize[end]), map(x -> [size(x)...], Rtensor), 1)
end

function Z2bulkQR!(Qparity, Qtensor, Rparity, Rtensor, A, p)
    Atensor = A.tensor
    Aparity = A.parity
    div = A.division

    ind_A = findall(x->sum(x[div+1:end]) % 2 == p, Aparity)
    matrix_j = unique(map(x->x[div+1:end], Aparity[ind_A]))
    matrix_i = unique(map(x->x[1:div], Aparity[ind_A]))

    ind = [findfirst(x->x in [[i; matrix_j[1]]], Aparity) for i in matrix_i]
    Amatrix = vcat(Atensor[ind]...)
    bulkidims = [size(Atensor[i],1) for i in ind]
    bulkjdims = [size(Amatrix, 2)]

    Q, R = qrpos!(Amatrix)
    for i in 1:length(matrix_i), j in 1:length(matrix_j)
        push!(Qparity, [matrix_i[i]; matrix_j[j]])
        idim, jdim = sum(bulkidims[1:i-1])+1:sum(bulkidims[1:i]), sum(bulkjdims[1:j-1])+1:sum(bulkjdims[1:j])
        push!(Qtensor, Q[idim, jdim])
    end
    for i in 1:length(matrix_j), j in 1:length(matrix_j)
        push!(Rparity, [matrix_j[i]; matrix_j[j]])
        idim, jdim = sum(bulkjdims[1:i-1])+1:sum(bulkjdims[1:i]), sum(bulkjdims[1:j-1])+1:sum(bulkjdims[1:j])
        push!(Rtensor, R[idim, jdim])
    end
end

function lqpos!(A::Z2Array{T,N}) where {T,N}
    Lparity = Vector{Vector{Int}}()
    Qparity = Vector{Vector{Int}}()
    atype = _arraytype(A.tensor[1])
    Ltensor = Vector{atype{T}}()
    Qtensor = Vector{atype{T}}()

    Z2bulkLQ!(Lparity, Ltensor, Qparity, Qtensor, A, 0)
    Z2bulkLQ!(Lparity, Ltensor, Qparity, Qtensor, A, 1)
    Asize = A.size
    Adims = A.dims
    exchangeind = indexin(Qparity, A.parity)
    Z2Array(Lparity, Ltensor, (Asize[1], Asize[1]), map(x -> [size(x)...], Ltensor), 1), Z2Array(Qparity, Qtensor, Asize, Adims[exchangeind], A.division)
end

function Z2bulkLQ!(Lparity, Ltensor, Qparity, Qtensor, A, p)
    Atensor = A.tensor
    Aparity = A.parity
    div = A.division

    ind_A = findall(x->sum(x[div+1:end]) % 2 == p, Aparity)
    matrix_j = unique(map(x->x[div+1:end], Aparity[ind_A]))
    matrix_i = unique(map(x->x[1:div], Aparity[ind_A]))

    v, bulkidims, bulkjdims = [] , Int[], Int[]
    for j in matrix_j
        h = []
        for i in matrix_i
            ind = findfirst(x->x in [[i; j]], Aparity)
            push!(h, Atensor[ind])   
        end
        vi = vcat(h...)
        push!(v, vi)
        push!(bulkjdims, size(vi, 2))
    end
    Amatrix = hcat(v...)
    push!(bulkidims, size(Amatrix, 1))
    
    L, Q = lqpos!(Amatrix)
    for i in 1:length(matrix_i), j in 1:length(matrix_i)
        push!(Lparity, [matrix_i[i]; matrix_i[j]])
        idim, jdim = sum(bulkidims[1:i-1])+1:sum(bulkidims[1:i]), sum(bulkidims[1:j-1])+1:sum(bulkidims[1:j])
        push!(Ltensor, L[idim, jdim])
    end
    for i in 1:length(matrix_i), j in 1:length(matrix_j)
        push!(Qparity, [matrix_i[i]; matrix_j[j]])
        idim, jdim = sum(bulkidims[1:i-1])+1:sum(bulkidims[1:i]), sum(bulkjdims[1:j-1])+1:sum(bulkjdims[1:j])
        push!(Qtensor, Q[idim, jdim])
    end
end

# for ' in ACCtoALAR of TeneT
function adjoint(A::Z2Array{T,N}) where {T,N}
    div = A.division 
    parity = map(x->x[[div+1:end;1:div]], A.parity)
    exchangeind = indexin(A.parity, parity)
    tensor = map(adjoint, A.tensor)[exchangeind]
    dims = map(x -> x[[div+1:end;1:div]], A.dims)[exchangeind]
    Z2Array(A.parity, tensor, A.size[[div+1:end;1:div]], dims, N - div)
end

# only for Z2 Matrix
function sysvd!(A::Z2Array{T,N}) where {T,N}
    tensor = A.tensor
    parity = A.parity
    div = A.division
    atype = _arraytype(tensor[1])
    Utensor = Vector{atype{T}}()
    Stensor = Vector{atype{T}}()
    Vtensor = Vector{atype{T}}()
    @inbounds for t in tensor
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
    Z2Array(parity, Utensor, (Asize[1], sm), N1, div), Z2Array(parity, Stensor, (sm, sm), [Nm, Nm], div), Z2Array(parity, Vtensor, (sm, Asize[2]), N2, div)
end

"""
    div = division(a::NTuple{Na, Int}, b::NTuple{Nb, Int}) where {Na, Nb}

give the reshape division of b by a, where b is the original shape and a is the new shape.
"""
function division(a::NTuple{Na, Int}, b::NTuple{Nb, Int}) where {Na, Nb}
    prod(a) != prod(b) && throw(Base.error("$a and $b must have the same product"))
    Na > Nb && throw(Base.error("$a must be shorter than $b"))
    div = Int[zeros(Int, Na)..., Nb]
    for i in 2:Na
        idiv = div[i-1] + 1
        p = b[idiv]
        while p != a[i-1]
            idiv += 1 
            p *= b[idiv]
        end
        div[i] = idiv
    end
    [div[i] + 1 : div[i+1] for i in 1:Na]
end

"""
    Z2reshape(A::U1Array{T, N}, a::Int...) where {T, N}

Z2reshape only for shape `(D,D,D,D,D,D,D,D) <-> (D^2,D^2,D^2,D^2)` and `(χ,D,D,χ) <-> (χ,D^2,χ)`, and the high-oreder U1tensor is from randZ2 or zerosZ2 function.
"""
# for consecutive storage
# Z2reshape(A::U1Array, s::Tuple{Vararg{Int}}; kwarg...) = Z2reshape(A, s...; kwarg...)
# function Z2reshape(A::U1Array{T, N}, s::Int...; reinfo = nothing) where {T <: Number, N}
#     dims = A.dims
#     bdiv = blockdiv(dims) 
#     tensor = [reshape(@view(A.tensor[bdiv[i]]), dims[i]...) for i in 1:length(bdiv)]
#     A = U1Array(A.qn, A.dir, tensor, A.size, dims, A.division)
#     U1reshape(A, s; reinfo = reinfo)
# end

Z2reshape(A::Z2Array, a::Tuple{Vararg{Int}}; kwarg...) = Z2reshape(A, a...; kwarg...)
function Z2reshape(A::Z2Array{T, N}, s::Int...; reinfo, kwarg...) where {T, N}
    atype = typeof(A.tensor[1]) <: CuArray ? CuArray : Array
    etype = eltype(A.tensor[1])
    if N > length(s)
        if reinfo === nothing
            indims = getZ2blockdims(size(A); kwarg...)
        else
            _, _, _, _, indims, _, _ = reinfo
        end
        indparity = [[0,1] for _ in 1:N]
        cA = zerosZ2(Array, ComplexF64, size(A)...; indims = indims)
        paritydiff = setdiff(cA.parity, A.parity)
        supind = indexin(paritydiff, cA.parity)
        Aparity = [A.parity; cA.parity[supind]]
        Atensor = [A.tensor; [cA.tensor[supind[i]] for i in 1:length(supind)]]
        exchangeind = indexin(cA.parity, Aparity)
        Aqn = cA.parity
        Adims = cA.dims
        Atensor = Atensor[exchangeind]
        div = division(s, size(A))
        reqn = [[sum(p[d]) % 2 for d in div] for p in Aqn]
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
            rebulkdims = [[prod(map((x,y,z)->x[indexin(y, z)...], indims[div[i]], indqnfrom, indparity[div[i]])) for indqnfrom in indqnfrom[i]] for i in 1:length(indqnfrom)]
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
        # nozeroind = norm.(retensors) .!= 0
        dims = map(x -> collect(size(x)), retensors)
        qn = ureqn
        p = sortperm(qn)
        Z2Array(qn[p], retensors[p], s, dims[p], 1), (choosesilces, chooseinds, nothing, nothing, indims, Aqn, Adims)
    else
        choosesilces, chooseinds, _, _, indims, reqn, redims = reinfo
        retensors = Array{Array,1}(undef, sum(length.(chooseinds)))
        div = division(size(A), s)
        ureqn = unique([[sum(p[d]) % 2 for d in div] for p in reqn])
        exchangeind = indexin(ureqn, A.parity)
        # @show ureqn A.qn exchangeind
        Atensor = A.tensor[exchangeind]
        for i in 1:length(choosesilces)
            for j in 1:length(choosesilces[i])
                retensors[chooseinds[i][j]] = reshape(Array(Atensor[i][choosesilces[i][j]...]), redims[chooseinds[i][j]]...)
            end
        end
        # nozeroind = norm.(retensors) .!= 0
        dims = map(x -> collect(size(x)), retensors)
        qn = reqn
        p = sortperm(qn)
        Z2Array(qn[p], atype.(retensors[p]), s, dims[p], 1), (choosesilces, chooseinds, nothing, nothing, indims, reqn, redims)
    end
end

function Z2reshapeinfo(s, sizeA, indims)
    length(sizeA) < length(s) && throw(Base.error("$sizeA must be longer than $s"))
    div = division(s, sizeA)
    A = zerosZ2(Array, ComplexF64, sizeA...; indims = indims)
    indparity = [[0,1] for _ in 1:length(sizeA)]
    reqn = [[sum(p[d]) % 2 for d in div] for p in A.parity]
    ureqn = unique(reqn)
    choosesilces = [[] for _ in 1:length(ureqn)]
    chooseinds = [[] for _ in 1:length(ureqn)]
    Aqn = A.parity
    for i in 1:length(ureqn)
        q = ureqn[i]
        bulkind = findall(x->x in [q], reqn)
        oriqn = Aqn[bulkind]
    
        indqnfrom = [unique(map(x->x[div], oriqn)) for div in div]
        rebulkdims = [[prod(map((x,y,z)->x[indexin(y, z)...], indims[div[i]], indqnfrom, indparity[div[i]])) for indqnfrom in indqnfrom[i]] for i in 1:length(indqnfrom)]
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
    choosesilces, chooseinds, nothing, nothing, indims, Aqn, A.dims
end