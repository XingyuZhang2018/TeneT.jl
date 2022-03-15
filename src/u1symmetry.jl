import Base: ==, +, -, *, /, ≈, size, reshape, permutedims, transpose, conj, show, similar, adjoint, copy, sqrt, getindex, setindex!, Array, broadcasted, vec, map, ndims, indexin, sum, zero
using BitBasis
using CUDA
import CUDA: CuArray
import LinearAlgebra: tr, norm, dot, rmul!, axpy!, mul!, diag, Diagonal, lmul!
import OMEinsum: _compactify!, subindex, einsum, Tr, Repeat, tensorpermute
import Zygote: accum
export U1Array
export randU1, asU1Array, asArray
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

"""
    U1Array{T, N}

a struct to hold the N-order U1 tensors
- `pn`(`particle number`): `N`-length Array
- `dir`(`out and in`): +1 or -1
- `tensor`: bulk tensor
- `size`  : original none-symetric array size
- `dims` size of `tensor`
- `division`: division location for reshape
"""
struct U1Array{T, N} <: AbstractSymmetricArray{T,N}
    pn::Vector{Vector{Int}}
    tensor::Vector{AbstractArray{T}}
    size::Tuple{Vararg{Int, N}}
    dims::Vector{Vector{Int}}
    division::Int
    function U1Array(pn::Vector{Vector{Int}}, tensor::Vector{<:AbstractArray{T}}, size::Tuple{Vararg{Int, N}}, dims::Vector{Vector{Int}}, division::Int) where {T,N}
        new{T, N}(pn, tensor, size, dims, division)
    end
end

size(A::U1Array) = A.size
size(A::U1Array, a) = size(A)[a]
conj(A::U1Array) = U1Array(-A.pn, map(conj, A.tensor), A.size, A.dims, A.division)
map(conj, A::U1Array) = conj(A)
norm(A::U1Array) = norm(A.tensor)

*(A::U1Array, B::Number) = U1Array(A.pn, A.tensor * B, A.size, A.dims, A.division)
*(B::Number, A::U1Array{T,N}) where {T,N} = A * B
/(A::U1Array{T,N}, B::Number) where {T,N} = U1Array(A.pn, A.tensor / B, A.size, A.dims, A.division)
broadcasted(*, A::U1Array, B::Number) = U1Array(A.pn, A.tensor .* B, A.size, A.dims, A.division)
broadcasted(*, B::Number, A::U1Array) = U1Array(A.pn, A.tensor .* B, A.size, A.dims, A.division)
broadcasted(/, A::U1Array, B::Number) = A / B

function +(A::U1Array, B::U1Array)
    if B.pn == A.pn
        U1Array(B.pn, A.tensor + B.tensor, B.size, B.dims, B.division)
    else
        exchangeind = indexin(B.pn, A.pn)
        U1Array(B.pn, A.tensor[exchangeind] + B.tensor, B.size, B.dims, B.division)
    end
end

function -(A::U1Array, B::U1Array)
    if B.pn == A.pn
        U1Array(B.pn, A.tensor - B.tensor, B.size, B.dims, B.division)
    else
        exchangeind = indexin(B.pn, A.pn)
        U1Array(B.pn, A.tensor[exchangeind] - B.tensor, B.size, B.dims, B.division)
    end
end

-(A::U1Array) = U1Array(A.pn, map(-, A.tensor), A.size, A.dims, A.division)

CuArray(A::U1Array) = U1Array(A.pn, map(CuArray, A.tensor), A.size, A.dims, A.division)
Array(A::U1Array) = U1Array(A.pn, map(Array, A.tensor), A.size, A.dims, A.division)

function dot(A::U1Array, B::U1Array) 
    if A.pn == B.pn 
        dot(A.tensor, B.tensor)
    else
        exchangeind = indexin(B.pn, A.pn)
        dot(A.tensor[exchangeind], B.tensor)
    end
end

function ≈(A::U1Array{TA,NA}, B::U1Array{TB,NB}) where {TA,NA,TB,NB}
    NA != NB && throw(Base.error("$A and $B have different dimensions"))
    exchangeind = indexin(A.pn, B.pn)
    A.tensor ≈ B.tensor[exchangeind]
end

function ==(A::U1Array{TA,NA}, B::U1Array{TB,NB}) where {TA,NA,TB,NB}
    NA != NB && throw(Base.error("$A and $B have different dimensions"))
    exchangeind = indexin(A.pn, B.pn)
    A.tensor == B.tensor[exchangeind]
end

function show(::IOBuffer, A::U1Array)
    println("particle number: \n", A.pn)
    println("dims: \n", A.dims)
    println("tensor: \n", A.tensor)
end

"""
    bkdims = u1bulkdims(size::Int...)

distribute dims of different part dims of U1 tensor bulk by average and midmax only for odd parts 
"""
function u1bulkdims(size::Int...;parts = 3)
    pn = map(size -> [sum(bitarray(i - 1, parts)) % parts for i = 1:size], size)
    map(pn -> [sum(pn .== i) for i = 0:parts-1], pn)
end

function randU1(atype, dtype, a...; dir, parts = 3)
    L = length(a)
    bkdims = u1bulkdims(a...) # custom initial
    pn = Vector{Vector{Int}}()
    tensor = Vector{atype{dtype}}()
    # parts = length.(bkdims)
    shift = 1
    @inbounds for i in CartesianIndices(Tuple(0:parts-1 for j=1:L))
        if sum(i.I .* dir) % parts == 0
            dims = Tuple(bkdims[j][i.I[j]+shift] for j in 1:L)
            if !(0 in dims)
                push!(pn, collect(i.I.* dir))
                push!(tensor, atype(rand(dtype, dims)))
            end
        end
    end
    dims = map(x -> [size(x)...], tensor)
    U1Array(pn, tensor, a, dims, 1)
end

function zerosU1(atype, dtype, a...; dir, parts = 3)
    L = length(a)
    bkdims = u1bulkdims(a...) # custom initial
    pn = Vector{Vector{Int}}()
    tensor = Vector{atype{dtype}}()
    # parts = length.(bkdims)
    shift = 1
    @inbounds for i in CartesianIndices(Tuple(0:(parts-1) for j=1:L))
        if sum(i.I .* dir) % parts == 0
            dims = Tuple(bkdims[j][i.I[j]+shift] for j in 1:L)
            push!(pn, collect(i.I.* dir))
            push!(tensor, atype(zeros(dtype, dims)))
        end
    end
    dims = map(x -> [size(x)...], tensor)
    U1Array(pn, tensor, a, dims, 1)
end

zero(A::U1Array) = U1Array(A.pn, map(zero, A.tensor), A.size, A.dims, A.division)

function IU1(atype, dtype, D; dir, parts = 3)
    bkdims = u1bulkdims(D, D) # custom initial
    pn = Vector{Vector{Int}}()
    tensor = Vector{atype{dtype}}()
    # parts = length.(bkdims)
    shift = 1
    @inbounds for i in CartesianIndices(Tuple(0:(parts-1) for j=1:2))
        if sum(i.I .* dir) % parts == 0
            dims = Tuple(bkdims[j][i.I[j]+shift] for j in 1:2)
            push!(pn, collect(i.I .* dir))
            push!(tensor, atype{dtype}(I, dims))
        end
    end
    dims = map(x -> [size(x)...], tensor)
    U1Array(pn, tensor, (D,D), dims, 1)
end

# getindex(A::U1Array, index::CartesianIndex) = getindex(A, index.I...)
# function getindex(A::U1Array, index::Int...)
#     bits = map(x -> ceil(Int, log2(x)), size(A))
#     pn = collect(map((index, bits) -> sum(bitarray(index - 1, bits)) % 2, index, bits))
#     sum(pn) % 2 != 0 && return 0.0
#     ip = findfirst(x->x in [pn], A.pn)
#     CUDA.@allowscalar A.tensor[ip][(Int.(floor.((index .-1 ) ./ 2)) .+ 1)...]
# end

# setindex!(A::U1Array, x::Number, index::CartesianIndex) = setindex!(A, x, index.I...)
# function setindex!(A::U1Array, x::Number, index::Int...)
#     bits = map(x -> ceil(Int, log2(x)), size(A))
#     pn = collect(map((index, bits) -> sum(bitarray(index - 1, bits)) % 2, index, bits))
#     ip = findfirst(x->x in [pn], A.pn)
#     CUDA.@allowscalar A.tensor[ip][(Int.(floor.((index .-1 ) ./ 2)) .+ 1)...] = x
# end

function U1selection(maxN::Int; parts = 3)
    # bit = ceil(Int, log2(maxN))
    q = [sum(bitarray(i-1, parts)) % parts for i = 1:maxN]
    [q .== i for i in 0:parts-1]
end

function asArray(A::U1Array{T,N}) where {T,N}
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
function getpn(size, dir::Vector; parts = 3)
    bkdims = u1bulkdims(size...)
    # parts = length.(bkdims)
    L = length(size)
    pn = Vector{Vector{Int}}()
    shift = 1
    @inbounds for i in CartesianIndices(Tuple(0:(parts-1) for j=1:L))
        if sum(i.I .* dir) % parts == 0
            dims = Tuple(bkdims[j][i.I[j]+shift] for j in 1:L)
            !(0 in dims) && push!(pn, collect(i.I .* dir))
        end
    end
    pn
end

# have Bugs with CUDA@v3.5.0, rely on https://github.com/JuliaGPU/CUDA.jl/issues/1304
# which is fixed in new vervion, but its allocation is abnormal
function asU1Array(A::AbstractArray{T,N}; dir::Vector) where {T,N}
    atype = _arraytype(A)
    Aarray = Array(A)
    qlist = [U1selection(size(A)[i]) for i = 1:N]
    pn = getpn(size(A), dir)
    shift = 1
    tensor = [atype(Aarray[[qlist[j][abs(pn[i][j])+shift] for j = 1:N]...]) for i in 1:length(pn)]
    dims = map(x -> [size(x)...], tensor)
    U1Array(pn, tensor, size(A), dims, 1)
end

# # only for OMEinsum binary permutedims before reshape
permutedims(A::U1Array, perm) = tensorpermute(A, perm)
function tensorpermute(A::U1Array, perm)
    length(perm) == 0 && return copy(A)
    pn = map(x -> x[collect(perm)], A.pn)
    tensor = map(x -> permutedims(x, perm), A.tensor)
    dims = map(x -> x[collect(perm)], A.dims)
    U1Array(pn, tensor, A.size[collect(perm)], dims, A.division)
end

reshape(A::U1Array, a::Tuple{Vararg{Int}}) = reshape(A, a...)
function reshape(A::U1Array{T,N}, a::Int...) where {T,N}
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
        return U1Array(A.pn, tensor, A.size, A.dims, div)
    else
        tensor = map((x, y) -> reshape(x, y...), Atensor, A.dims)
        return U1Array(A.pn, tensor, A.size, A.dims, A.division)
    end
end

"""
    *(A::U1Array{TA,NA}, B::U1Array{TB,NB}) where {TA,TB,NA,NB}

core code for U1Array product
"""
function *(A::U1Array{TA,NA}, B::U1Array{TB,NB}; parts = 3) where {TA,TB,NA,NB}
    pn = Vector{Vector{Int}}()
    dims = Vector{Vector{Int}}()
    atype = _arraytype(B.tensor[1])
    T = TA == ComplexF64 || TB == ComplexF64 ? ComplexF64 : Float64
    tensor = Vector{atype{T}}()
    divA, divB = A.division, B.division

    Adir = sign.(sum(A.pn))[divA+1:end]
    Bdir = sign.(sum(B.pn)[1:divB])
    sum(Adir .+ Bdir) !== 0 && throw(Base.error("U1Array product: out and in direction not match, expect: $(-Adir), got: $(Bdir)"))
    if !(divA in [0, NA]) && !(divB in [0, NB]) 
        for p in unique(map(x->mod(sum(x[divA+1:end]), parts), A.pn))
            # @show p
            u1bulktimes!(pn, tensor, dims, A, B, p; parts = parts)
        end
    else
        u1bulktimes!(pn, tensor, dims, A, B, 0; parts = parts)
    end
    pn == [[]] && return Array(tensor[1])[]
    U1Array(pn, tensor, (size(A)[1:divA]..., size(B)[divB+1:end]...), dims, divA)
end

"""
    u1bulktimes!(pn, tensor, A, B, p)

fill into even and odd matrix,  p = 0 for even, p = 1 for odd, then dispatch to result tensor after product
"""
function u1bulktimes!(pn, tensor, dims, A, B, p; parts = 3)
    Apn, Atensor = A.pn, A.tensor
    Bpn, Btensor = B.pn, B.tensor
    Adims, Bdims = A.dims, B.dims
    divA, divB = A.division, B.division
    atype = _arraytype(Btensor[1])
    etype = eltype(Btensor[1])

    ind_A = findall(x->mod(sum(x[divA+1:end]), parts) == p, Apn)
    matrix_j = intersect(map(x->x[divA+1:end], Apn[ind_A]), map(x->-x[1:divB], Bpn))
    ind_A = findall(x->x[divA+1:end] in matrix_j, Apn)
    matrix_i = unique(map(x->x[1:divA], Apn[ind_A]))
    ind_B = findall(x->x[1:divB] in -matrix_j, Bpn)
    ind_B == [] && return
    matrix_k = unique(map(x->x[divB+1:end], Bpn[ind_B]))

    # @show Apn Bpn matrix_i matrix_j ind_B matrix_k
    index = [findfirst(x->x in [[i; j]], Apn) for i in matrix_i, j in matrix_j]
    # @show index
    oribulkidims = map(ind -> Adims[ind][1:divA], index[:, 1])
    bulkidims = map(ind -> size(Atensor[ind], 1), index[:, 1])
    bulkjdims = map(ind -> size(Atensor[ind], 2), index[1, :])
    # Amatrix = hvcat(ntuple(i->length(bulkjdims), length(bulkidims)), Atensor[index']...)
    Amatrix = atype <: Array ? zeros(etype, sum(bulkidims), sum(bulkjdims)) : CUDA.zeros(etype, sum(bulkidims), sum(bulkjdims))
    # @show size(Amatrix)
    for i in 1:length(matrix_i), j in 1:length(matrix_j)
        # println(sum(bulkidims[1:i-1])+1:sum(bulkidims[1:i]), ", ", sum(bulkjdims[1:j-1])+1:sum(bulkjdims[1:j]), " ", index[i, j])
        Amatrix[sum(bulkidims[1:i-1])+1:sum(bulkidims[1:i]), sum(bulkjdims[1:j-1])+1:sum(bulkjdims[1:j])] .= Atensor[index[i, j]]
    end

    index = [findfirst(x->x in [[-j; k]], Bpn) for j in matrix_j, k in matrix_k]
    oribulkkdims = []
    bulkkdims = Vector{Int}()
    # @show index
    # if nothing in index[1, :]
    #     for j in 1:length(matrix_k)
    #         for i in 1:length(matrix_j)
    #             if index[i, j] !== nothing
    #                 oribulkkdims = push!(oribulkkdims, Bdims[index[i, j]][divB+1:end])
    #                 bulkkdims = push!(bulkkdims, size(Btensor[index[i, j]], 2))
    #                 break
    #             end 
    #         end
    #     end
    # else
        oribulkkdims = map(ind -> Bdims[ind][divB+1:end], index[1, :])
        bulkkdims = map(ind -> size(Btensor[ind], 2), index[1, :])
    # end
    # Bmatrix = hvcat(ntuple(i->length(bulkkdims), length(bulkjdims)), Btensor[index']...)
    Bmatrix = atype <: Array ? zeros(etype, sum(bulkjdims), sum(bulkkdims)) : CUDA.zeros(etype, sum(bulkjdims), sum(bulkkdims))
    # @show size(Bmatrix)
    for j in 1:length(matrix_j), k in 1:length(matrix_k)
        # println(sum(bulkjdims[1:j-1])+1:sum(bulkjdims[1:j]), ", ", sum(bulkkdims[1:k-1])+1:sum(bulkkdims[1:k]), " ", index[j, k])
        Bmatrix[sum(bulkjdims[1:j-1])+1:sum(bulkjdims[1:j]), sum(bulkkdims[1:k-1])+1:sum(bulkkdims[1:k])] .= Btensor[index[j, k]]
    end
    
    C = Amatrix * Bmatrix

    for i in 1:length(matrix_i), k in 1:length(matrix_k)
        push!(pn, [matrix_i[i]; matrix_k[k]])
        push!(dims, [oribulkidims[i]; oribulkkdims[k]])
        idim, kdim = sum(bulkidims[1:i-1])+1:sum(bulkidims[1:i]), sum(bulkkdims[1:k-1])+1:sum(bulkkdims[1:k])
        # println(idim, ", ", kdim)
        push!(tensor, C[idim, kdim])
    end
end


# # for OMEinsum contract to get number
# # vec(A::U1Array) = A

function transpose(A::U1Array)
    tensor = map(transpose, A.tensor)
    U1Array(A.pn, tensor, A.size, A.dims, 0)
end

function tr(A::U1Array{T,N}) where {T,N}
    pn = A.pn
    tensor = A.tensor
    half = Int(length(pn[1])/2)
    s = 0.0
    @inbounds @simd for i in 1:length(pn)
        pn[i][1:half] == -pn[i][half+1:end] && (s += tr(tensor[i]))
    end
    s
end

# function _compactify!(y, x::U1Array, indexer)
#     x = U1Array2tensor(Array(x))
#     @inbounds @simd for ci in CartesianIndices(y)
#         y[ci] = x[subindex(indexer, ci.I)]
#     end
#     return y
# end

# broadcasted(*, A::U1Array, B::Base.RefValue) = U1Array(A.pn, A.tensor .* B, A.size, A.dims, A.division)
# broadcasted(*, B::Base.RefValue, A::U1Array) = U1Array(A.pn, A.tensor .* B, A.size, A.dims, A.division)

# for ein"abab ->"(A)[]
function dtr(A::U1Array{T,N}) where {T,N}
    pn = A.pn
    tensor = A.tensor
    s = 0.0
    @inbounds @simd for i in 1:length(pn)
        pn[i][1] == -pn[i][3] && pn[i][2] == -pn[i][4] && (s += Array(ein"abab ->"(tensor[i]))[])
    end
    s
end


# for Zygote compatibility
accum(A::U1Array, B::U1Array...) = +(A, B...)

# for KrylovKit compatibility
rmul!(A::U1Array, B::Number) = (map(x -> rmul!(x, B), A.tensor); A)

function lmul!(A::U1Array{T,N}, B::U1Array) where {T,N}
    C = A*B
    for i = 1:length(B.pn)
        B.tensor[i] = C.tensor[i]
    end
    return B
end

similar(A::U1Array) = U1Array(A.pn, map(similar, A.tensor), A.size, A.dims, A.division)
similar(A::U1Array, atype) = U1Array(A.pn, map(x -> atype(similar(x)), A.tensor), A.size, A.dims, A.division)
diag(A::U1Array{T,N}) where {T,N} = CUDA.@allowscalar collect(Iterators.flatten(diag.(A.tensor)))
copy(A::U1Array{T,N}) where {T,N} = U1Array(A.pn, map(copy, A.tensor), A.size, A.dims, A.division)

function mul!(Y::U1Array, A::U1Array, B::Number)
    exchangeind = indexin(A.pn, Y.pn)
    map((Y, A) -> mul!(Y, A, B), Y.tensor[exchangeind], A.tensor)
    Y
end

function axpy!(α::Number, A::U1Array, B::U1Array)
    if B.pn == A.pn
        map((x,y) -> axpy!(α, x, y), A.tensor, B.tensor)
    else
        if length(A.pn) > length(B.pn)
            exchangeind = indexin(B.pn, A.pn)
            map((x,y) -> axpy!(α, x, y), A.tensor[exchangeind], B.tensor)
        else
            exchangeind = indexin(A.pn, B.pn)
            map((x,y) -> axpy!(α, x, y), A.tensor, B.tensor[exchangeind])
        end
    end
    return B
end

# # for leftorth and rightorth compatibility
Diagonal(A::U1Array) = U1Array(A.pn, map(Diagonal, A.tensor), A.size, A.dims, A.division)
sqrt(A::U1Array) = U1Array(A.pn, map(x->sqrt.(x), A.tensor), A.size, A.dims, A.division)
broadcasted(sqrt, A::U1Array) = sqrt(A)

# only for order-three tensor's qr and lq
function qrpos!(A::U1Array{T,N}; parts = 3) where {T,N}
    Qpn = Vector{Vector{Int}}()
    Rpn = Vector{Vector{Int}}()
    atype = _arraytype(A.tensor[1])
    Qtensor = Vector{atype{T}}()
    Rtensor = Vector{atype{T}}()

    for p in unique(map(x->mod(sum(x[A.division+1:end]), parts), A.pn))
        # @show p
        u1bulkQR!(Qpn, Qtensor, Rpn, Rtensor, A, p; parts = parts)
    end
    Asize = A.size
    Adims = A.dims
    exchangeind = indexin(Qpn, A.pn)
    U1Array(Qpn, Qtensor, Asize, Adims[exchangeind], A.division), U1Array(Rpn, Rtensor, (Asize[end], Asize[end]), map(x -> [size(x)...], Rtensor), 1)
end

function u1bulkQR!(Qpn, Qtensor, Rpn, Rtensor, A, p; parts = 3)
    Atensor = A.tensor
    Apn = A.pn
    Adiv = A.division

    ind_A = findall(x->mod(sum(x[Adiv+1:end]), parts) == p, Apn)
    matrix_j = unique(map(x->x[Adiv+1:end], Apn[ind_A]))
    matrix_i = unique(map(x->x[1:Adiv], Apn[ind_A]))

    ind = [findfirst(x->x in [[i; matrix_j[1]]], Apn) for i in matrix_i]
    Amatrix = vcat(Atensor[ind]...)
    bulkidims = [size(Atensor[i],1) for i in ind]
    bulkjdims = [size(Amatrix, 2)]

    Q, R = qrpos!(Amatrix)
    for i in 1:length(matrix_i)
        push!(Qpn, [matrix_i[i]; matrix_j[1]])
        idim, jdim = sum(bulkidims[1:i-1])+1:sum(bulkidims[1:i]), 1:sum(bulkjdims[1])
        push!(Qtensor, Q[idim, jdim])
    end
    
    push!(Rpn, [-matrix_j[1]; matrix_j[1]])
    idim, jdim = 1:sum(bulkjdims[1]), 1:sum(bulkjdims[1])
    push!(Rtensor, R[idim, jdim])
end

function lqpos!(A::U1Array{T,N}; parts = 3) where {T,N}
    Lpn = Vector{Vector{Int}}()
    Qpn = Vector{Vector{Int}}()
    atype = _arraytype(A.tensor[1])
    Ltensor = Vector{atype{T}}()
    Qtensor = Vector{atype{T}}()

    for p in unique(map(x->mod(x[1], parts), A.pn))
        u1bulkLQ!(Lpn, Ltensor, Qpn, Qtensor, A, p; parts = parts)
    end
    Asize = A.size
    Adims = A.dims
    exchangeind = indexin(Qpn, A.pn)
    U1Array(Lpn, Ltensor, (Asize[1], Asize[1]), map(x -> [size(x)...], Ltensor), 1), U1Array(Qpn, Qtensor, Asize, Adims[exchangeind], A.division)
end

function u1bulkLQ!(Lpn, Ltensor, Qpn, Qtensor, A, p; parts = 3)
    Atensor = A.tensor
    Apn = A.pn
    Adiv = A.division

    ind_A = findall(x->mod(x[1], parts) == p, Apn)
    matrix_j = unique(map(x->x[Adiv+1:end], Apn[ind_A]))
    matrix_i = unique(map(x->x[1], Apn[ind_A]))

    ind = [findfirst(x->x in [[matrix_i[1]; j]], Apn) for j in matrix_j]
    Amatrix = hcat(Atensor[ind]...)
    bulkidims = [size(Amatrix, 1)]
    bulkjdims = [size(Atensor[i], 2) for i in ind]
    
    L, Q = lqpos!(Amatrix)

    push!(Lpn, [matrix_i[1]; -matrix_i[1]])
    idim, jdim = 1:sum(bulkidims[1]), 1:sum(bulkidims[1])
    push!(Ltensor, L[idim, jdim])
    for j in 1:length(matrix_j)
        push!(Qpn, [matrix_i[1]; matrix_j[j]])
        idim, jdim = 1:sum(bulkidims[1]), sum(bulkjdims[1:j-1])+1:sum(bulkjdims[1:j])
        push!(Qtensor, Q[idim, jdim])
    end
end

# # for ' in ACCtoALAR of VUMPS
function adjoint(A::U1Array{T,N}) where {T,N}
    div = A.division 
    pn = map(x->-x[[div+1:end;1:div]], A.pn)
    tensor = map(adjoint, A.tensor)
    dims = map(x -> x[[div+1:end;1:div]], A.dims)
    U1Array(pn, tensor, A.size[[div+1:end;1:div]], dims, N - div)
end

# only for U1 Matrix
function sysvd!(A::U1Array{T,N}) where {T,N}
    tensor = A.tensor
    pn = A.pn
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
    U1Array(pn, Utensor, (Asize[1], sm), N1, div), U1Array(pn, Stensor, (sm, sm), [Nm for _ in 1:length(pn)], div), U1Array(pn, Vtensor, (sm, Asize[2]), N2, div)
end

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
#     U1reshape(A::U1Array{T, N}, a::Int...) where {T, N}

# U1reshape only for shape `(D,D,D,D,D,D,D,D) <-> (D^2,D^2,D^2,D^2)` and `(χ,D,D,χ) <-> (χ,D^2,χ)`
# """
# U1reshape(A::U1Array, a::Tuple{Vararg{Int}}) = U1reshape(A, a...)
# function U1reshape(A::U1Array{T, N}, a::Int...) where {T, N}
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
#         U1Array(urepn, atype.(retensors), a, dims, 1)
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
#         U1Array(repn, atype.(retensors), a, dims, 1)
#     end
# end
