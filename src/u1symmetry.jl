import Base: ==, +, -, *, /, ≈, size, reshape, permutedims, transpose, conj, show, similar, adjoint, copy, sqrt, getindex, setindex!, Array, broadcasted, vec, map, ndims, indexin, sum, zero
using BitBasis
using CUDA
import CUDA: CuArray
import LinearAlgebra: tr, norm, dot, rmul!, axpy!, mul!, diag, Diagonal, lmul!
import OMEinsum: _compactify!, subindex, einsum, Tr, Repeat, tensorpermute
import Zygote: accum
export AbstractU1Array, U1tensor
export randU1, U1tensor2tensor, tensor2U1tensor
export dtr

# """
#     pn_conserving(T::Array)
# Transform an arbitray tensor which has arbitray legs and each leg have index 1 or 2 into pn conserving form
# ----
# The following is faster but rely on updates of CUDA.jl(available in master branch)
# function pn_conserving!(T::Union{Array,CuArray})
# 	bits = map(x -> Int(ceil(log2(x))), size(T))
#     T[map(x->sum(sum.(bitarray.((Tuple(x).-1) ,bits))) % 2 !== 0 ,CartesianIndices(T))].=0
#     return T
# end
# pn_conserving(T) = pn_conserving!(copy(T))
# """
# function pn_conserving(T::Union{Array,CuArray})
# 	s = size(T)
# 	p = zeros(s)
# 	bits = map(x -> Int(ceil(log2(x))), s)
# 	@inbounds @simd for index in CartesianIndices(T)
# 		i = Tuple(index) .- 1
# 		sum(sum.(bitarray.(i,bits))) % 2 == 0 && (p[index] = 1)
# 	end
# 	p = _arraytype(T)(p)

# 	return reshape(p.*T,s...)
# end

abstract type AbstractU1Array{T,N} <: AbstractArray{T,N} end
"""
    U1tensor{T, N}

a struct to hold the N-order U1 tensors
- `pn`(`particle number`): `N`-length Array
- `oi`(`out and in`): +1 or -1
- `tensor`: bulk tensor
- `size`  : original none-symetric array size
- `dims` size of `tensor`
- `division`: division location for reshape
"""
struct U1tensor{T, N} <: AbstractU1Array{T,N}
    pn::Vector{Vector{Int}}
    tensor::Vector{AbstractArray{T}}
    size::Tuple{Vararg{Int, N}}
    dims::Vector{Vector{Int}}
    division::Int
    function U1tensor(pn::Vector{Vector{Int}}, tensor::Vector{<:AbstractArray{T}}, size::Tuple{Vararg{Int, N}}, dims::Vector{Vector{Int}}, division::Int) where {T,N}
        new{T, N}(pn, tensor, size, dims, division)
    end
end

size(A::AbstractU1Array) = A.size
size(A::AbstractU1Array, a) = size(A)[a]
conj(A::AbstractU1Array) = U1tensor(-A.pn, map(conj, A.tensor), A.size, A.dims, A.division)
map(conj, A::AbstractU1Array) = conj(A)
# norm(A::AbstractU1Array) = norm(A.tensor)

# *(A::AbstractU1Array, B::Number) = U1tensor(A.pn, A.tensor * B, A.size, A.dims, A.division)
# *(B::Number, A::AbstractU1Array{T,N}) where {T,N} = A * B
# /(A::AbstractU1Array{T,N}, B::Number) where {T,N} = U1tensor(A.pn, A.tensor / B, A.size, A.dims, A.division)
# broadcasted(*, A::AbstractU1Array, B::Number) = U1tensor(A.pn, A.tensor .* B, A.size, A.dims, A.division)
# broadcasted(*, B::Number, A::AbstractU1Array) = U1tensor(A.pn, A.tensor .* B, A.size, A.dims, A.division)
# broadcasted(/, A::AbstractU1Array, B::Number) = A / B

# function +(A::AbstractU1Array, B::AbstractU1Array)
#     if B.pn == A.pn
#         U1tensor(B.pn, A.tensor + B.tensor, B.size, B.dims, B.division)
#     else
#         exchangeind = indexin(B.pn, A.pn)
#         U1tensor(B.pn, A.tensor[exchangeind] + B.tensor, B.size, B.dims, B.division)
#     end
# end

# function -(A::AbstractU1Array, B::AbstractU1Array)
#     if B.pn == A.pn
#         U1tensor(B.pn, A.tensor - B.tensor, B.size, B.dims, B.division)
#     else
#         exchangeind = indexin(B.pn, A.pn)
#         U1tensor(B.pn, A.tensor[exchangeind] - B.tensor, B.size, B.dims, B.division)
#     end
# end

# -(A::AbstractU1Array) = U1tensor(A.pn, map(-, A.tensor), A.size, A.dims, A.division)

# CuArray(A::AbstractU1Array) = U1tensor(A.pn, map(CuArray, A.tensor), A.size, A.dims, A.division)
# Array(A::AbstractU1Array) = U1tensor(A.pn, map(Array, A.tensor), A.size, A.dims, A.division)

# function dot(A::AbstractU1Array, B::AbstractU1Array) 
#     if A.pn == B.pn 
#         dot(A.tensor, B.tensor)
#     else
#         exchangeind = indexin(A.pn, B.pn)
#         dot(A.tensor, B.tensor[exchangeind])
#     end
# end

function ≈(A::AbstractU1Array{TA,NA}, B::AbstractU1Array{TB,NB}) where {TA,NA,TB,NB}
    NA != NB && throw(Base.error("$A and $B have different dimensions"))
    exchangeind = indexin(A.pn, B.pn)
    A.tensor ≈ B.tensor[exchangeind]
end

function show(::IOBuffer, A::U1tensor)
    println("particle number: \n", A.pn)
    println("dims: \n", A.dims)
    println("tensor: \n", A.tensor)
end

"""
    bkdims = u1bulkdims(size::Int...)

distribute dims of different part dims of U1 tensor bulk by average and midmax only for odd parts 
"""
function u1bulkdims(size::Int...)
    bits = map(x -> ceil(Int, log2(x)), size)
    pn = map((bits, size) -> [sum(bitarray(i - 1, bits)) for i = 1:size], bits, size)
    map((bits, pn) -> filter(x->x!=0, [sum(pn .== i) for i = 0:bits]), bits, pn)
end

function randU1(atype, dtype, oi, a...)
    L = length(a)
    bkdims = u1bulkdims(a...) # custom initial
    pn = Vector{Vector{Int}}()
    tensor = Vector{atype{dtype}}()
    parts = length.(bkdims)
    shift = 1
    @inbounds for i in CartesianIndices(Tuple(0:parts[j]-1 for j=1:L))
        if sum(i.I .* oi) % max(parts...) == 0
            dims = Tuple(bkdims[j][i.I[j]+shift] for j in 1:L)
            # if !(0 in dims)
                push!(pn, collect(i.I.* oi))
                push!(tensor, atype(rand(dtype, dims)))
            # end
        end
    end
    dims = map(x -> [size(x)...], tensor)
    U1tensor(pn, tensor, a, dims, 1)
end

function zerosU1(atype, dtype, oi, a...)
    L = length(a)
    bkdims = u1bulkdims(a...) # custom initial
    pn = Vector{Vector{Int}}()
    tensor = Vector{atype{dtype}}()
    parts = length.(bkdims)
    shift = 1
    @inbounds for i in CartesianIndices(Tuple(0:(parts[j]-1) for j=1:L))
        if sum(i.I .* oi) % max(parts...) == 0
            dims = Tuple(bkdims[j][i.I[j]+shift] for j in 1:L)
            if !(0 in dims)
                push!(pn, collect(i.I.* oi))
                push!(tensor, atype(zeros(dtype, dims)))
            end
        end
    end
    dims = map(x -> [size(x)...], tensor)
    U1tensor(pn, tensor, a, dims, 1)
end

zero(A::AbstractU1Array) = U1tensor(A.pn, map(zero, A.tensor), A.size, A.dims, A.division)

function IU1(atype, dtype, oi, D)
    bkdims = u1bulkdims(D, D) # custom initial
    pn = Vector{Vector{Int}}()
    tensor = Vector{atype{dtype}}()
    parts = length.(bkdims)
    shift = 1
    @inbounds for i in CartesianIndices(Tuple(0:(parts[j]-1) for j=1:2))
        if sum(i.I .* oi) % max(parts...) == 0
            dims = Tuple(bkdims[j][i.I[j]+shift] for j in 1:2)
            if !(0 in dims)
                push!(pn, collect(i.I .* oi))
                push!(tensor, atype{dtype}(I, dims))
            end
        end
    end
    dims = map(x -> [size(x)...], tensor)
    U1tensor(pn, tensor, (D,D), dims, 1)
end

# getindex(A::AbstractU1Array, index::CartesianIndex) = getindex(A, index.I...)
# function getindex(A::AbstractU1Array, index::Int...)
#     bits = map(x -> ceil(Int, log2(x)), size(A))
#     pn = collect(map((index, bits) -> sum(bitarray(index - 1, bits)) % 2, index, bits))
#     sum(pn) % 2 != 0 && return 0.0
#     ip = findfirst(x->x in [pn], A.pn)
#     CUDA.@allowscalar A.tensor[ip][(Int.(floor.((index .-1 ) ./ 2)) .+ 1)...]
# end

# setindex!(A::AbstractU1Array, x::Number, index::CartesianIndex) = setindex!(A, x, index.I...)
# function setindex!(A::AbstractU1Array, x::Number, index::Int...)
#     bits = map(x -> ceil(Int, log2(x)), size(A))
#     pn = collect(map((index, bits) -> sum(bitarray(index - 1, bits)) % 2, index, bits))
#     ip = findfirst(x->x in [pn], A.pn)
#     CUDA.@allowscalar A.tensor[ip][(Int.(floor.((index .-1 ) ./ 2)) .+ 1)...] = x
# end

function U1selection(maxN::Int)
    bit = ceil(Int, log2(maxN))
    q = [sum(bitarray(i-1, bit)) for i = 1:maxN]
    [q .== i for i in 0:bit]
end

function U1tensor2tensor(A::U1tensor{T,N}) where {T,N}
    atype = _arraytype(A.tensor[1])
    tensor = zeros(T, size(A))
    pn = A.pn
    qlist = [U1selection(size(A)[i]) for i = 1:N]
    shift = 1
    for i in 1:length(pn)
        tensor[[qlist[j][abs(pn[i][j])+shift] for j = 1:N]...] = Array(A.tensor[i])
    end
    atype(tensor)
end

"""
p = getpn(size)

give the pn of length L
"""
function getpn(size, oi::Vector)
    bkdims = u1bulkdims(size...)
    parts = length.(bkdims)
    L = length(size)
    pn = Vector{Vector{Int}}()
    shift = 1
    @inbounds for i in CartesianIndices(Tuple(0:(parts[j]-1) for j=1:L))
        dims = Tuple(bkdims[j][i.I[j]+shift] for j in 1:L)
        !(0 in dims) && push!(pn, collect(i.I .* oi))
    end
    pn
end

# have Bugs with CUDA@v3.5.0, rely on https://github.com/JuliaGPU/CUDA.jl/issues/1304
# which is fixed in new vervion, but its allocation is abnormal
function tensor2U1tensor(A::AbstractArray{T,N}, oi::Vector) where {T,N}
    atype = _arraytype(A)
    Aarray = Array(A)
    qlist = [U1selection(size(A)[i]) for i = 1:N]
    pn = getpn(size(A), oi)
    shift = 1
    tensor = [atype(Aarray[[qlist[j][abs(pn[i][j])+shift] for j = 1:N]...]) for i in 1:length(pn)]
    dims = map(x -> [size(x)...], tensor)
    U1tensor(pn, tensor, size(A), dims, 1)
end

# # only for OMEinsum binary permutedims before reshape
permutedims(A::AbstractU1Array, perm) = tensorpermute(A, perm)
function tensorpermute(A::AbstractU1Array, perm)
    length(perm) == 0 && return copy(A)
    pn = map(x -> x[collect(perm)], A.pn)
    tensor = map(x -> permutedims(x, perm), A.tensor)
    dims = map(x -> x[collect(perm)], A.dims)
    U1tensor(pn, tensor, A.size[collect(perm)], dims, A.division)
end

reshape(A::AbstractU1Array, a::Tuple{Vararg{Int}}) = reshape(A, a...)
function reshape(A::AbstractU1Array{T,N}, a::Int...) where {T,N}
    Atensor = A.tensor
    div = 1
    if length(a) < N
        sizeA = size(A)
        p = sizeA[1]
        while p != a[1]
            div += 1
            p *= sizeA[div]
        end
        tensor = map((x, y) -> reshape(x, prod(y[1:div]), prod(y[div+1:end])), Atensor, A.dims)
        return U1tensor(A.pn, tensor, A.size, A.dims, div)
    else
        tensor = map((x, y) -> reshape(x, y...), Atensor, A.dims)
        return U1tensor(A.pn, tensor, A.size, A.dims, A.division)
    end
end

"""
    *(A::AbstractU1Array{TA,NA}, B::AbstractU1Array{TB,NB}) where {TA,TB,NA,NB}

core code for U1tensor product
"""
function *(A::AbstractU1Array{TA,NA}, B::AbstractU1Array{TB,NB}; parts = 3) where {TA,TB,NA,NB}
    pn = Vector{Vector{Int}}()
    dims = Vector{Vector{Int}}()
    atype = _arraytype(B.tensor[1])
    T = TA == ComplexF64 || TB == ComplexF64 ? ComplexF64 : Float64
    tensor = Vector{atype{T}}()
    divA, divB = A.division, B.division

    Aoi = sign.(sum(A.pn))[divA+1:end]
    Boi = sign.(sum(B.pn)[1:divB])
    sum(Aoi .+ Boi) !== 0 && throw(Base.error("U1tensor product: out and in direction not match, expect: $(-Aoi), got: $(Boi)"))
    if !(divA in [0, NA]) && !(divB in [0, NB]) 
        for matrix_i in unique(map(x->x[1:divA], A.pn))
            # @show i
            u1bulktimes!(pn, tensor, dims, A, B, [matrix_i])
        end
    else
        u1bulktimes!(pn, tensor, dims, A, B, [[0 for _ in 1:divA]])
    end
    pn == [[]] && return Array(tensor[1])[]
    U1tensor(pn, tensor, (size(A)[1:divA]..., size(B)[divB+1:end]...), dims, divA)
end

"""
    u1bulktimes!(pn, tensor, A, B, p)

fill into even and odd matrix,  p = 0 for even, p = 1 for odd, then dispatch to result tensor after product
"""
function u1bulktimes!(pn, tensor, dims, A, B, matrix_i; parts = 3)
    Apn, Atensor = A.pn, A.tensor
    Bpn, Btensor = B.pn, B.tensor
    Adims, Bdims = A.dims, B.dims
    divA, divB = A.division, B.division
    atype = _arraytype(Btensor[1])
    etype = eltype(Btensor[1])

    ind_A = findall(x->x[1:divA] in matrix_i, Apn)
    matrix_j = unique(map(x->x[divA+1:end], Apn[ind_A]))
    ind_B = findall(x->x[1:divB] in -matrix_j, Bpn)
    ind_B == [] && return
    matrix_k = unique(map(x->x[divB+1:end], Bpn[ind_B]))

    # @show Apn Bpn matrix_i matrix_j ind_B matrix_k
    index = [findfirst(x->x in [[i; j]], Apn) for i in matrix_i, j in matrix_j]
    oribulkidims = map(ind -> Adims[ind][1:divA], index[:, 1])
    bulkidims = map(ind -> size(Atensor[ind], 1), index[:, 1])
    bulkjdims = map(ind -> size(Atensor[ind], 2), index[1, :])
    # Amatrix = hvcat(ntuple(i->length(bulkjdims), length(bulkidims)), Atensor[index']...)
    Amatrix = atype == Array ? zeros(etype, sum(bulkidims), sum(bulkjdims)) : CUDA.zeros(etype, sum(bulkidims), sum(bulkjdims))
    for i in 1:length(matrix_i), j in 1:length(matrix_j)
        Amatrix[sum(bulkidims[1:i-1])+1:sum(bulkidims[1:i]), sum(bulkjdims[1:j-1])+1:sum(bulkjdims[1:j])] .= Atensor[index[i, j]]
    end

    index = [findfirst(x->x in [[-j; k]], Bpn) for j in matrix_j, k in matrix_k]
    oribulkkdims = map(ind -> Bdims[ind][divB+1:end], index[1, :])
    bulkkdims = map(ind -> size(Btensor[ind], 2), index[1, :])
    # Bmatrix = hvcat(ntuple(i->length(bulkkdims), length(bulkjdims)), Btensor[index']...)
    Bmatrix = atype == Array ? zeros(etype, sum(bulkjdims), sum(bulkkdims)) : CUDA.zeros(etype, sum(bulkjdims), sum(bulkkdims))
    for j in 1:length(matrix_j), k in 1:length(matrix_k)
        Bmatrix[sum(bulkjdims[1:j-1])+1:sum(bulkjdims[1:j]), sum(bulkkdims[1:k-1])+1:sum(bulkkdims[1:k])] .= Btensor[index[j, k]]
    end
    
    C = Amatrix * Bmatrix

    for i in 1:length(matrix_i), k in 1:length(matrix_k)
        push!(pn, [matrix_i[i]; matrix_k[k]])
        push!(dims, [oribulkidims[i]; oribulkkdims[k]])
        idim, kdim = sum(bulkidims[1:i-1])+1:sum(bulkidims[1:i]), sum(bulkkdims[1:k-1])+1:sum(bulkkdims[1:k])
        push!(tensor, C[idim, kdim])
    end
end


# # for OMEinsum contract to get number
# # vec(A::AbstractU1Array) = A

function transpose(A::AbstractU1Array)
    tensor = map(transpose, A.tensor)
    U1tensor(A.pn, tensor, A.size, A.dims, 0)
end

function tr(A::AbstractU1Array{T,N}) where {T,N}
    pn = A.pn
    tensor = A.tensor
    half = Int(length(pn[1])/2)
    s = 0.0
    @inbounds @simd for i in 1:length(pn)
        pn[i][1:half] == -pn[i][half+1:end] && (s += tr(tensor[i]))
    end
    s
end

# function _compactify!(y, x::AbstractU1Array, indexer)
#     x = U1tensor2tensor(Array(x))
#     @inbounds @simd for ci in CartesianIndices(y)
#         y[ci] = x[subindex(indexer, ci.I)]
#     end
#     return y
# end

# broadcasted(*, A::AbstractU1Array, B::Base.RefValue) = U1tensor(A.pn, A.tensor .* B, A.size, A.dims, A.division)
# broadcasted(*, B::Base.RefValue, A::AbstractU1Array) = U1tensor(A.pn, A.tensor .* B, A.size, A.dims, A.division)

# for ein"abab ->"(A)[]
function dtr(A::AbstractU1Array{T,N}) where {T,N}
    pn = A.pn
    tensor = A.tensor
    s = 0.0
    @inbounds @simd for i in 1:length(pn)
        pn[i][1] == -pn[i][3] && pn[i][2] == -pn[i][4] && (s += Array(ein"abab ->"(tensor[i]))[])
    end
    s
end


# for Zygote compatibility
accum(A::AbstractU1Array, B::AbstractU1Array...) = +(A, B...)

# for KrylovKit compatibility
rmul!(A::AbstractU1Array, B::Number) = (map(x -> rmul!(x, B), A.tensor); A)

function lmul!(A::AbstractU1Array{T,N}, B::AbstractU1Array) where {T,N}
    C = A*B
    for i = 1:length(B.pn)
        B.tensor[i] = C.tensor[i]
    end
    return B
end

similar(A::AbstractU1Array) = U1tensor(A.pn, map(similar, A.tensor), A.size, A.dims, A.division)
similar(A::AbstractU1Array, atype) = U1tensor(A.pn, map(x -> atype(similar(x)), A.tensor), A.size, A.dims, A.division)
diag(A::AbstractU1Array{T,N}) where {T,N} = CUDA.@allowscalar collect(Iterators.flatten(diag.(A.tensor)))
copy(A::AbstractU1Array{T,N}) where {T,N} = U1tensor(A.pn, map(copy, A.tensor), A.size, A.dims, A.division)

mul!(Y::AbstractU1Array, A::AbstractU1Array, B::Number) = (map((Y, A) -> mul!(Y, A, B), Y.tensor, A.tensor); Y)

function axpy!(α::Number, A::AbstractU1Array, B::AbstractU1Array)
    if B.pn == A.pn
        map((x,y) -> axpy!(α, x, y), A.tensor, B.tensor)
    else
        exchangeind = indexin(B.pn, A.pn)
        map((x,y) -> axpy!(α, x, y), A.tensor[exchangeind], B.tensor)
    end
    return B
end

# # for leftorth and rightorth compatibility
# Diagonal(A::AbstractU1Array) = U1tensor(A.pn, map(Diagonal, A.tensor), A.size, A.dims, A.division)
# sqrt(A::AbstractU1Array) = U1tensor(A.pn, map(x->sqrt.(x), A.tensor), A.size, A.dims, A.division)
# broadcasted(sqrt, A::AbstractU1Array) = sqrt(A)

# # only for order-three tensor's qr and lq
# function qrpos!(A::AbstractU1Array{T,N}) where {T,N}
#     Qpn = Vector{Vector{Int}}()
#     Rpn = Vector{Vector{Int}}()
#     atype = _arraytype(A.tensor[1])
#     Qtensor = Vector{atype{T}}()
#     Rtensor = Vector{atype{T}}()

#     bulkQR!(Qpn, Qtensor, Rpn, Rtensor, A, 0)
#     bulkQR!(Qpn, Qtensor, Rpn, Rtensor, A, 1)
#     Asize = A.size
#     Adims = A.dims
#     exchangeind = indexin(Qpn, A.pn)
#     U1tensor(Qpn, Qtensor, Asize, Adims[exchangeind], A.division), U1tensor(Rpn, Rtensor, (Asize[end], Asize[end]), map(x -> [size(x)...], Rtensor), 1)
# end

# function bulkQR!(Qpn, Qtensor, Rpn, Rtensor, A, p)
#     Atensor = A.tensor
#     Apn = A.pn
#     div = A.division

#     ind_A = findall(x->sum(x[div+1:end]) % 2 == p, Apn)
#     matrix_j = unique(map(x->x[div+1:end], Apn[ind_A]))
#     matrix_i = unique(map(x->x[1:div], Apn[ind_A]))

#     ind = [findfirst(x->x in [[i; matrix_j[1]]], Apn) for i in matrix_i]
#     Amatrix = vcat(Atensor[ind]...)
#     bulkidims = [size(Atensor[i],1) for i in ind]
#     bulkjdims = [size(Amatrix, 2)]

#     Q, R = qrpos!(Amatrix)
#     for i in 1:length(matrix_i), j in 1:length(matrix_j)
#         push!(Qpn, [matrix_i[i]; matrix_j[j]])
#         idim, jdim = sum(bulkidims[1:i-1])+1:sum(bulkidims[1:i]), sum(bulkjdims[1:j-1])+1:sum(bulkjdims[1:j])
#         push!(Qtensor, Q[idim, jdim])
#     end
#     for i in 1:length(matrix_j), j in 1:length(matrix_j)
#         push!(Rpn, [matrix_j[i]; matrix_j[j]])
#         idim, jdim = sum(bulkjdims[1:i-1])+1:sum(bulkjdims[1:i]), sum(bulkjdims[1:j-1])+1:sum(bulkjdims[1:j])
#         push!(Rtensor, R[idim, jdim])
#     end
# end

# function lqpos!(A::AbstractU1Array{T,N}) where {T,N}
#     Lpn = Vector{Vector{Int}}()
#     Qpn = Vector{Vector{Int}}()
#     atype = _arraytype(A.tensor[1])
#     Ltensor = Vector{atype{T}}()
#     Qtensor = Vector{atype{T}}()

#     bulkLQ!(Lpn, Ltensor, Qpn, Qtensor, A, 0)
#     bulkLQ!(Lpn, Ltensor, Qpn, Qtensor, A, 1)
#     Asize = A.size
#     Adims = A.dims
#     exchangeind = indexin(Qpn, A.pn)
#     U1tensor(Lpn, Ltensor, (Asize[1], Asize[1]), map(x -> [size(x)...], Ltensor), 1), U1tensor(Qpn, Qtensor, Asize, Adims[exchangeind], A.division)
# end

# function bulkLQ!(Lpn, Ltensor, Qpn, Qtensor, A, p)
#     Atensor = A.tensor
#     Apn = A.pn
#     div = A.division

#     ind_A = findall(x->sum(x[div+1:end]) % 2 == p, Apn)
#     matrix_j = unique(map(x->x[div+1:end], Apn[ind_A]))
#     matrix_i = unique(map(x->x[1:div], Apn[ind_A]))

#     v, bulkidims, bulkjdims = [] , Int[], Int[]
#     for j in matrix_j
#         h = []
#         for i in matrix_i
#             ind = findfirst(x->x in [[i; j]], Apn)
#             push!(h, Atensor[ind])   
#         end
#         vi = vcat(h...)
#         push!(v, vi)
#         push!(bulkjdims, size(vi, 2))
#     end
#     Amatrix = hcat(v...)
#     push!(bulkidims, size(Amatrix, 1))
    
#     L, Q = lqpos!(Amatrix)
#     for i in 1:length(matrix_i), j in 1:length(matrix_i)
#         push!(Lpn, [matrix_i[i]; matrix_i[j]])
#         idim, jdim = sum(bulkidims[1:i-1])+1:sum(bulkidims[1:i]), sum(bulkidims[1:j-1])+1:sum(bulkidims[1:j])
#         push!(Ltensor, L[idim, jdim])
#     end
#     for i in 1:length(matrix_i), j in 1:length(matrix_j)
#         push!(Qpn, [matrix_i[i]; matrix_j[j]])
#         idim, jdim = sum(bulkidims[1:i-1])+1:sum(bulkidims[1:i]), sum(bulkjdims[1:j-1])+1:sum(bulkjdims[1:j])
#         push!(Qtensor, Q[idim, jdim])
#     end
# end

# # for ' in ACCtoALAR of VUMPS
# function adjoint(A::AbstractU1Array{T,N}) where {T,N}
#     div = A.division 
#     pn = map(x->x[[div+1:end;1:div]], A.pn)
#     exchangeind = indexin(A.pn, pn)
#     tensor = map(adjoint, A.tensor)[exchangeind]
#     dims = map(x -> x[[div+1:end;1:div]], A.dims)[exchangeind]
#     U1tensor(A.pn, tensor, A.size[[div+1:end;1:div]], dims, N - div)
# end

# # only for U1 Matrix
# function sysvd!(A::AbstractU1Array{T,N}) where {T,N}
#     tensor = A.tensor
#     pn = A.pn
#     div = A.division
#     atype = _arraytype(tensor[1])
#     Utensor = Vector{atype{T}}()
#     Stensor = Vector{atype{T}}()
#     Vtensor = Vector{atype{T}}()
#     @inbounds @simd for t in tensor
#         U, S, V = sysvd!(t)
#         push!(Utensor, U)
#         push!(Stensor, S)
#         push!(Vtensor, V)
#     end
#     Nm = map(x->min(x...), A.dims)
#     N1 = map((x, y) -> [x[1], y], A.dims, Nm)
#     N2 = map((x, y) -> [y, x[2]], A.dims, Nm)
#     Asize = A.size
#     sm = min(Asize...)
#     U1tensor(pn, Utensor, (Asize[1], sm), N1, div), U1tensor(pn, Stensor, (sm, sm), [Nm, Nm], div), U1tensor(pn, Vtensor, (sm, Asize[2]), N2, div)
# end

# """
#     div = division(a::NTuple{Na, Int}, b::NTuple{Nb, Int}) where {Na, Nb}

# give the reshape division of b by a, where b is the original shape and a is the new shape.
# """
# function division(a::NTuple{Na, Int}, b::NTuple{Nb, Int}) where {Na, Nb}
#     prod(a) != prod(b) && throw(Base.error("$a and $b must have the same product"))
#     Na > Nb && throw(Base.error("$a must be shorter than $b"))
#     div = Int[zeros(Int, Na)..., Nb]
#     for i in 2:Na
#         idiv = div[i-1] + 1
#         p = b[idiv]
#         while p != a[i-1]
#             idiv += 1 
#             p *= b[idiv]
#         end
#         div[i] = idiv
#     end
#     [div[i] + 1 : div[i+1] for i in 1:Na]
# end

# """
#     U1reshape(A::AbstractU1Array{T, N}, a::Int...) where {T, N}

# U1reshape only for shape `(D,D,D,D,D,D,D,D) <-> (D^2,D^2,D^2,D^2)` and `(χ,D,D,χ) <-> (χ,D^2,χ)`
# """
# U1reshape(A::AbstractU1Array, a::Tuple{Vararg{Int}}) = U1reshape(A, a...)
# function U1reshape(A::AbstractU1Array{T, N}, a::Int...) where {T, N}
#     atype = _arraytype(A.tensor[1])
#     orderedpn = getpn(N)
#     if orderedpn == A.pn
#         Atensor = A.tensor
#         Adims = A.dims
#     else
#         exchangeind = indexin(orderedpn, A.pn)
#         Atensor = A.tensor[exchangeind]
#         Adims = A.dims[exchangeind]
#     end
#     if N > length(a)
#         div = division(a, size(A))
#         repn = [[sum(p[d]) % 2 for d in div] for p in orderedpn]
#         redims = [[prod(dims[d]) for d in div] for dims in Adims]
#         retensor = [reshape(t, s...) for (t, s) in zip(map(Array, Atensor), redims)]
#         urepn = getpn(length(a))
#         retensors = Vector{atype{T}}()
#         for i in 1:length(urepn)
#             p = urepn[i]
#             bulkind = findall(x->x in [p], repn)
#             rebulkdims = Int.(.+(redims[bulkind]...) ./ (length(bulkind) ./ length.(div)))
#             rebulkdims1 = redims[bulkind[1]]
#             silce = [[1:rebulkdims1[i], (rebulkdims1[i] == rebulkdims[i] ? 1 : 1+rebulkdims1[i]):rebulkdims[i]] for i in 1:length(rebulkdims)]
#             tensor = atype(zeros(T, rebulkdims...))
#             bits = Int(log2(length(bulkind)))
#             for j in 1:length(bulkind)
#                 choose = bitarray(j - 1, bits) .+ 1
#                 length(choose) == 1 && (choose = [choose[], choose[], choose[]])
#                 choosesilce = [silce[i][choose[i]] for i in 1:length(silce)]
#                 tensor[choosesilce...] = retensor[bulkind[j]]
#             end
#             push!(retensors, tensor)
#         end
#         dims = map(x -> [size(x)...], retensors)
#         U1tensor(urepn, atype.(retensors), a, dims, 1)
#     else
#         div = division(size(A), a)
#         repn = getpn(length(a))
#         pn = [[sum(p[d]) % 2 for d in div] for p in repn]
#         rebulkdims = bulkdims(a...)
#         redims = [[rebulkdims[p[i] + 1][i] for i in 1:length(a)] for p in repn]
#         dims = [[prod(dims[d]) for d in div] for dims in redims]
#         retensors = Array{Array,1}(undef, length(repn))
#         for i in 1:length(orderedpn)
#             p = orderedpn[i]
#             bulkind = findall(x->x in [p], pn)
#             bulkdims = Int.(.+(dims[bulkind]...) ./ (length(bulkind) ./ length.(div)))
#             bulkdims1 = dims[bulkind[1]]
#             silce = [[1:bulkdims1[i], (bulkdims1[i] == bulkdims[i] ? 1 : 1+bulkdims1[i]):bulkdims[i]] for i in 1:length(bulkdims)]
#             bits = Int(log2(length(bulkind)))
#             for j in 1:length(bulkind)
#                 choose = bitarray(j - 1, bits) .+ 1
#                 length(choose) == 1 && (choose = [choose[], choose[], choose[]])
#                 choosesilce = [silce[i][choose[i]] for i in 1:length(silce)]
#                 retensors[bulkind[j]] = reshape(Array(Atensor[i])[choosesilce...], redims[bulkind[j]]...)
#             end
#         end
#         dims = map(x -> [size(x)...], retensors)
#         U1tensor(repn, atype.(retensors), a, dims, 1)
#     end
# end
