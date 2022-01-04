import Base: ==, +, -, *, /, ≈, size, reshape, permutedims, transpose, conj, show, similar, adjoint, copy, sqrt, getindex, setindex!, Array, broadcasted, vec, map, ndims, _to_subscript_indices
using BitBasis
using CUDA
import CUDA: CuArray
import LinearAlgebra: tr, norm, dot, rmul!, axpy!, mul!, diag, Diagonal, lmul!
import OMEinsum: _compactify!, subindex, einsum, Tr, Repeat
import Zygote: accum
export AbstractZ2Array, Z2tensor

"""
    parity_conserving(T::Array)

Transform an arbitray tensor which has arbitray legs and each leg have index 1 or 2 into parity conserving form

"""
function parity_conserving(T::Union{Array,CuArray})
	s = size(T)
	p = zeros(s)
	bits = map(x -> Int(ceil(log2(x))), s)
	@inbounds for index in CartesianIndices(T)
		i = Tuple(index) .- 1
		sum(sum.(bitarray.(i,bits))) % 2 == 0 && (p[index] = 1)
	end
	p = _arraytype(T)(p)

	return reshape(p.*T,s...)
end

abstract type AbstractZ2Array{T,N} <: AbstractArray{T,N} end
struct Z2tensor{T, N} <: AbstractZ2Array{T,N}
    parity::Vector{Vector{Int}}
    tensor::Vector{AbstractArray{T}}
    division::Int
    function Z2tensor(parity::Vector{<:Vector{Int}}, tensor::Vector{<:AbstractArray{T}}, N::Tuple{Vararg}, division::Int) where T
        new{T, N}(parity, tensor, division)
    end
end

ndims(::AbstractZ2Array{T,N}) where {T,N} = length(N)
size(::AbstractZ2Array{T,N}) where {T,N} = N
size(::AbstractZ2Array{T,N}, a) where {T,N} = N[a]
conj(A::AbstractZ2Array{T,N}) where {T,N} = Z2tensor(A.parity, map(conj, A.tensor), N, A.division)
map(conj, A::AbstractZ2Array) = conj(A)
norm(A::AbstractZ2Array{T,N}) where {T,N} = norm(A.tensor)

*(A::AbstractZ2Array{T,N}, B::Number) where {T,N} = Z2tensor(A.parity, A.tensor * B, N, A.division)
*(B::Number, A::AbstractZ2Array{T,N}) where {T,N} = A * B
/(A::AbstractZ2Array{T,N}, B::Number) where {T,N} = Z2tensor(A.parity, A.tensor / B, N, A.division)
broadcasted(*, A::AbstractZ2Array, B::Number) = A * B
broadcasted(*, B::Number, A::AbstractZ2Array) = A * B
broadcasted(/, A::AbstractZ2Array, B::Number) = A / B

# for Zygote compatibility
accum(A::AbstractZ2Array, B::AbstractZ2Array...) = +(A, B...)

# for KrylovKit compatibility
rmul!(A::AbstractZ2Array{T,N}, B::Number) where {T,N} = Z2tensor(A.parity, rmul!.(A.tensor, B), N, A.division)
lmul!(A::AbstractZ2Array, B::AbstractZ2Array) = A * B
similar(A::AbstractZ2Array{T,N}) where {T,N} = Z2tensor(A.parity, similar(A.tensor), N, A.division)
similar(A::AbstractZ2Array{T,N}, atype) where {T,N} = Z2tensor(A.parity, similar(A.tensor), N, A.division)
diag(A::AbstractZ2Array{T,N}) where {T,N} = CUDA.@allowscalar collect(Iterators.flatten(diag.(A.tensor)))
copy(A::AbstractZ2Array{T,N}) where {T,N} = Z2tensor(A.parity, map(copy, A.tensor), N, A.division)
function mul!(Y::AbstractZ2Array{T,N}, A::AbstractZ2Array{T,N}, B::Number) where {T,N}
    Y = A*B
    Z2tensor(Y.parity, Y.tensor, N, Y.division)
end

function axpy!(α::Number, A::AbstractZ2Array, B::AbstractZ2Array{T,N}) where {T,N}
    exchangeind = indexin(B.parity, A.parity)
    Z2tensor(B.parity, map((x,y)->axpy!(α, x, y), A.tensor[exchangeind], B.tensor), N, B.division)
end

# for leftorth and rightorth compatibility
Diagonal(A::AbstractZ2Array{T,N}) where {T,N} = Z2tensor(A.parity, map(Diagonal, A.tensor), N, A.division)
sqrt(A::AbstractZ2Array{T,N}) where {T,N} = Z2tensor(A.parity, map(x->sqrt.(x), A.tensor), N, A.division)
broadcasted(sqrt, A::AbstractZ2Array) = sqrt(A)

CuArray(A::AbstractZ2Array{T,N}) where {T, N} = Z2tensor(A.parity, CuArray.(A.tensor), N, A.division)
Array(A::AbstractZ2Array{T,N}) where {T, N} = Z2tensor(A.parity, Array.(A.tensor), N, A.division)

function adjoint(A::AbstractZ2Array{T,N}) where {T,N}
    div = A.division 
    parity = map(x->x[[div+1:end;1:div]], A.parity)
    exchangeind = indexin(A.parity, parity)
    tensor = adjoint.(A.tensor)
    Z2tensor(A.parity, tensor[exchangeind], (N[div+1:end]..., N[1:div]...), length(N) - div)
end

function +(A::AbstractZ2Array{T,N}, B::AbstractZ2Array{T,N}) where {T,N}
    exchangeind = indexin(B.parity, A.parity)
    Z2tensor(B.parity, A.tensor[exchangeind] + B.tensor, N, B.division)
end

function -(A::AbstractZ2Array{T,N}, B::AbstractZ2Array{T,N}) where {T,N}
    exchangeind = indexin(B.parity, A.parity)
    Z2tensor(B.parity, A.tensor[exchangeind] - B.tensor, N, B.division)
end

-(A::AbstractZ2Array{T,N}) where {T, N} = Z2tensor(A.parity, -A.tensor, N, A.division)

function dot(A::AbstractZ2Array{T,N}, B::AbstractZ2Array{T,N}) where {T,N}
    exchangeind = indexin(A.parity, B.parity)
    dot(A.tensor, B.tensor[exchangeind])
end

function ≈(A::AbstractZ2Array, B::AbstractZ2Array)
    exchangeind = indexin(A.parity, B.parity)
    length(exchangeind) == length(A.parity) && A.tensor ≈ B.tensor[exchangeind]
end

function show(io::IOBuffer, A::Z2tensor)
    println("parity: \n", A.parity)
    println("tensor: \n", A.tensor)
end

function getindex(A::AbstractZ2Array{T,N}, index) where {T,N}
    typeof(index) <: CartesianIndex && (index = index.I)
    bits = map(x -> Int(ceil(log2(x))), N)
    parity = collect(sum.(bitarray.(index .- 1, bits))) .% 2
    sum(parity) % 2 != 0 && return 0.0
    ip = findfirst(x->x in [parity], A.parity)
    CUDA.@allowscalar A.tensor[ip][(Int.(floor.((index .-1 ) ./ 2)) .+ 1)...]
end

# # for index []
# getindex(A::AbstractZ2Array{T,()}) where {T} = A.tensor[1]
# _to_subscript_indices(::AbstractZ2Array{T,()}) where {T} = ()

function setindex!(A::AbstractZ2Array{T,N}, x::Number, index) where {T,N}
    typeof(index) <: CartesianIndex && (index = index.I)
    bits = map(x -> Int(ceil(log2(x))), N)
    parity = collect(sum.(bitarray.(index .- 1, bits))) .% 2
    ip = findfirst(x->x in [parity], A.parity)
    CUDA.@allowscalar A.tensor[ip][(Int.(floor.((index .-1 ) ./ 2)) .+ 1)...] = x
end

"""
    deven, dodd = bulkdims(N::Int...)

find dims of bulk
"""
function bulkdims(N::Int...)
    bits = map(x -> Int(ceil(log2(x))), N)
    dodd = map((bits,N) -> sum([sum(bitarray(i - 1, bits)) % 2 for i = 1:N]), bits, N)
    deven = N .- dodd
    deven, dodd
end

function randZ2(atype, dtype, a...)
    L = length(a)
    deven, dodd = bulkdims(a...)
    parity = Vector{Vector{Int}}()
    tensor = Vector{atype{dtype}}()
    @inbounds for i in CartesianIndices(Tuple(0:1 for i=1:L))
        if sum(i.I) % 2 == 0
            push!(parity,collect(i.I))
            dims = Tuple(i.I[j] == 0 ? deven[j] : dodd[j] for j in 1:L)
            push!(tensor,atype(rand(dtype, dims)))
        end
    end
    Z2tensor(parity, tensor, a, 1)
end

function zerosZ2(atype, dtype, a...)
    L = length(a)
    deven, dodd = bulkdims(a...)
    parity = Vector{Vector{Int}}()
    tensor = Vector{atype{dtype}}()
    @inbounds for i in CartesianIndices(Tuple(0:1 for i=1:L))
        if sum(i.I) % 2 == 0
            push!(parity,collect(i.I))
            dims = Tuple(i.I[j] == 0 ? deven[j] : dodd[j] for j in 1:L)
            push!(tensor,atype(zeros(dtype, dims)))
        end
    end
    Z2tensor(parity, tensor, a, 1)
end

function IZ2(atype, dtype, D)
    deven, dodd = bulkdims(D, D)
    parity = Vector{Vector{Int}}()
    tensor = Vector{atype{dtype}}()
    push!(parity,[0, 0])
    push!(tensor,atype{dtype}(I, deven...))
    push!(parity,[1, 1])
    push!(tensor,atype{dtype}(I, dodd...))
    Z2tensor(parity, tensor, (D, D), 1)
end

# only for OMEinsum permutedims before reshape
function permutedims(A::AbstractZ2Array{T,N}, perm) where {T,N}
    parity = map(x->x[collect(perm)], A.parity)
    exchangeind = indexin(A.parity, parity)
    tensor = map(x->permutedims(x, perm), A.tensor)
    Z2tensor(A.parity, tensor[exchangeind], N[collect(perm)], 1)
end

reshape(A::AbstractZ2Array, a::Tuple{Vararg{Int}}) = reshape(A, a...)
function reshape(A::AbstractZ2Array{T,N}, a::Int...) where {T,N}
    tensor = copy(A.tensor)
    div = 1
    if length(a) < length(N)
        p = N[1]
        while p != a[1]
            div += 1
            p *= N[div]
        end
        @inbounds for i in 1:length(tensor)
            s = size(tensor[i])
            tensor[i] = reshape(tensor[i], prod(s[1:div]), prod(s[div+1:end]))
        end
        return Z2tensor(A.parity, tensor, N, div)
    else
        deven, dodd = bulkdims(N...)
        parity = A.parity
        @inbounds for i in 1:length(tensor)
            dims = Tuple(parity[i][j] == 0 ? deven[j] : dodd[j] for j in 1:length(N))
            tensor[i] = reshape(tensor[i], dims)
        end
        return Z2tensor(parity, tensor, N, div)
    end
end

"""
    *(A::AbstractZ2Array{TA,NA}, B::AbstractZ2Array{TB,NB}) where {TA,TB,NA,NB}

core code for Z2tensor product
"""
function *(A::AbstractZ2Array{TA,NA}, B::AbstractZ2Array{TB,NB}) where {TA,TB,NA,NB}
    parity = Vector{Vector{Int}}()
    atype = _arraytype(B.tensor[1])
    T = TA == ComplexF64 || TB == ComplexF64 ? ComplexF64 : Float64
    tensor = Vector{atype{T}}()
    divA, divB = A.division, B.division

    bulktimes!(parity, tensor, A, B, 0)
    !(divA in [0, length(NA)]) && !(divB in [0, length(NB)]) && bulktimes!(parity, tensor, A, B, 1)
    parity == [Int64[]] && return Array(tensor[1])[]
    Z2tensor(parity, tensor, (NA[1:divA]..., NB[divB+1:end]...), divA)
end

"""
    bulktimes!(parity, tensor, A, B, p)

fill into even and odd matrix,  p = 0 for even, p = 1 for odd, then dispatch to result tensor after product
"""
function bulktimes!(parity, tensor, A, B, p)
    Aparity, Atensor = A.parity, A.tensor
    Bparity, Btensor = B.parity, B.tensor
    divA, divB = A.division, B.division

    ind_A = findall(x->sum(x[divA+1:end]) % 2 == p, Aparity)
    matrix_j = unique(map(x->x[divA+1:end], Aparity[ind_A]))
    matrix_i = unique(map(x->x[1:divA], Aparity[ind_A]))
    ind_B = findall(x->x[1:divB] in matrix_j, Bparity)
    matrix_k = unique(map(x->x[divB+1:end], Bparity[ind_B]))

    #opt push!
    h, bulkidims = [] , Int[]
    for i in matrix_i
        v = []
        for j in matrix_j
            ind = findfirst(x->x in [[i; j]], Aparity)
            push!(v, Atensor[ind])        
        end
        hi = hcat(v...)
        push!(h, hi)
        push!(bulkidims, size(hi)[1])
    end
    Amatrix = vcat(h...)

    v, bulkjdims = [], Int[]
    for k in matrix_k
        h = []
        for j in matrix_j
            ind = findfirst(x->x in [[j; k]], Bparity)
            push!(h, Btensor[ind])
        end
        hj = vcat(h...)
        push!(v, hj)
        push!(bulkjdims, size(hj)[2])
    end
    Bmatrix = hcat(v...)

    atype = _arraytype(Btensor[1])
    C = atype(Amatrix) * atype(Bmatrix)
    for i in 1:length(matrix_i), j in 1:length(matrix_k)
        push!(parity, [matrix_i[i]; matrix_k[j]])
        idim, jdim = sum(bulkidims[1:i-1])+1:sum(bulkidims[1:i]), sum(bulkjdims[1:j-1])+1:sum(bulkjdims[1:j])
        push!(tensor, C[idim, jdim])
    end
end

function Z2tensor2tensor(A::Z2tensor{T,N}) where {T,N}
    atype = _arraytype(A.tensor[1])
    Tensor = atype(zeros(T, N))
    bits = map(x -> Int(ceil(log2(x))), N)
    @inbounds for i in CartesianIndices(Tensor)         
        sum(sum.(bitarray.(i.I .- 1, bits))) % 2 == 0 && (CUDA.@allowscalar Tensor[i] = A[i])
    end
    Tensor
end

function Z2bitselection(maxN::Int)
    q = [sum(bitarray(i-1,ceil(Int,log(2,maxN))))%2 for i=1:maxN]
    return [(q .== 0),(q .== 1)]
end

# have Bugs with CuArray, rely on https://github.com/JuliaGPU/CUDA.jl/issues/1304
function tensor2Z2tensor(tensor::AbstractArray{T,N}) where {T,N}
    A = zerosZ2(_arraytype(tensor), eltype(tensor), size(tensor)...)
    qlist = [Z2bitselection(size(tensor)[i]) for i =1:N]
    for i in CartesianIndices(A.parity)
        A.tensor[i] = tensor[[qlist[j][A.parity[i][j]+1] for j =1:N]...]
    end
    return A
end

# for OMEinsum contract to get number
function transpose(A::AbstractZ2Array{T,N}) where {T,N}
    tensor = map(x->transpose(x), A.tensor)
    Z2tensor(A.parity, tensor, N, 0)
end

function tr(A::AbstractZ2Array{T,N}) where {T,N}
    parity = A.parity
    tensor = A.tensor
    half = Int(length(parity[1])/2)
    s = 0.0
    @inbounds for i in 1:length(parity)
        parity[i][1:half] == parity[i][half+1:end] && (s += tr(tensor[i]))
    end
    s
end

# function einsum(::Tr, ixs, iy, xs::Tuple{<:AbstractZ2Array}, size_dict::Dict)
#     x = xs[1]
#     asZ2array(tr(x))
# end

# """
#     asZ2array(a::Number) where {T,N}

# fill `a` as a Z2Array
# """
# asZ2array(a::Number) = Z2tensor([Int[]], [[a]], (), 0)

# for [] backward
# function einsum(::Repeat, ixs, iy, xs::Tuple{<:AbstractZ2Array}, size_dict::Dict)
#     ix, x = ixs[1], xs[1]
#     @show ix, x
#     @debug "Repeat" ix => iy size(x)
#     ix1f = filter(i -> i ∈ ix, iy)
#     res = if ix1f != ix
#         einsum(Permutedims(), (ix,), ix1f, (x,), size_dict)
#     else
#         x
#     end
#     newshape = [l ∈ ix ? size_dict[l] : 1 for l in iy]
#     repeat_dims = [l ∈ ix ? 1 : size_dict[l] for l in iy]
#     repeat(reshape(res, newshape...), repeat_dims...)
# end

function _compactify!(y, x::AbstractZ2Array, indexer)
    x = Z2tensor2tensor(Array(x))
    @inbounds for ci in CartesianIndices(y)
        y[ci] = x[subindex(indexer, ci.I)]
    end
    return y
end

# only for ein"abab -> " case
# function _compactify!(y, x::AbstractZ2Array, indexer)
#     vcat(hcat(x.tensor[1],x.tensor[6]), hcat(x.tensor[3],x.tensor[8]))
# end

vec(A::AbstractZ2Array) = A

# only for order-three tensor's qr and lq
function qrpos!(A::AbstractZ2Array{T,N}) where {T,N}
    Qparity = Vector{Vector{Int}}()
    Rparity = Vector{Vector{Int}}()
    atype = _arraytype(A.tensor[1])
    Qtensor = Vector{atype{T}}()
    Rtensor = Vector{atype{T}}()

    bulkQR!(Qparity, Qtensor, Rparity, Rtensor, A, 0)
    bulkQR!(Qparity, Qtensor, Rparity, Rtensor, A, 1)
    Z2tensor(Qparity, Qtensor, N, A.division), Z2tensor(Rparity, Rtensor, (N[end], N[end]), 1)
end

function bulkQR!(Qparity, Qtensor, Rparity, Rtensor, A, p)
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

function lqpos!(A::AbstractZ2Array{T,N}) where {T,N}
    Lparity = Vector{Vector{Int}}()
    Qparity = Vector{Vector{Int}}()
    atype = _arraytype(A.tensor[1])
    Ltensor = Vector{atype{T}}()
    Qtensor = Vector{atype{T}}()

    bulkLQ!(Lparity, Ltensor, Qparity, Qtensor, A, 0)
    bulkLQ!(Lparity, Ltensor, Qparity, Qtensor, A, 1)
    Z2tensor(Lparity, Ltensor, (N[1], N[1]), 1), Z2tensor(Qparity, Qtensor, N, A.division)
end

function bulkLQ!(Lparity, Ltensor, Qparity, Qtensor, A, p)
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

# only for Z2 Matrix
function sysvd!(A::AbstractZ2Array{T,N}) where {T,N}
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
    Nm = min(N...)
    Z2tensor(parity, Utensor, (N[1],Nm), div), Z2tensor(parity, Stensor, (Nm,Nm), div), Z2tensor(parity, Vtensor, (Nm,N[2]), div)
end

"""
p = getparity(L::Int)

give the parity of length L
"""
function getparity(L::Int)
    p = Vector{Vector{Int}}()
    for i in CartesianIndices(Tuple(0:1 for i=1:L))
        sum(i.I) % 2 == 0 && push!(p, collect(i.I))
    end
    p
end

"""
div = division(a::NTuple{Na, Int}, b::NTuple{Nb, Int}) where {Na, Nb}

give the reshape division of b by a, where b is the original shape and a is the new shape
"""
function division(a::NTuple{Na, Int}, b::NTuple{Nb, Int}) where {Na, Nb}
    prod(a) != prod(b) && throw(error("$a and $b must have the same product"))
    Na > Nb && throw(error("$a must be shorter than $b"))
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
    div
end
"""
    Z2tensor(reparity, tensor, a, 1) = Z2reshape(A::AbstractZ2Array{T,N}, a::Int...) where {T,N}

reshape the Z2tensor to the new shape, but only for a <: 2^N
"""
Z2reshape(A::AbstractZ2Array, a::Tuple{Vararg{Int}}) = Z2reshape(A, a...)
function Z2reshape(A::AbstractZ2Array{T,N}, a::Int...) where {T,N}
    atype = _arraytype(A.tensor[1])
    if length(a) < length(N)
        div = division(a, N)
        reparity = map(x->[[sum(x[div[i] + 1 : div[i + 1]]) for i in 1:(length(div)-1)]...] .% 2, A.parity)
        # ureparity = unique(reparity) 
        ureparity = getparity(length(a))
        deven, dodd = bulkdims(a...) 
        tensor = Vector{atype{T}}(undef, length(ureparity))
        for i in 1:length(ureparity)
            p = ureparity[i]
            ind = findall(x->x in [p], reparity)
            dims = Tuple(p[i] == 0 ? deven[i] : dodd[i] for i in 1:length(a))
            resize = map(x->[[prod(x[div[i] + 1 : div[i + 1]]) for i in 1:(length(div)-1)]...], size.(A.tensor[ind]))
            tensor[i] = reshape(vcat(map((x,a) -> reshape(x,a...), A.tensor[ind], resize)...), dims)
        end
        Z2tensor(ureparity, tensor, a, 1)
    else
        reparity = getparity(length(a))
        tensor = Vector{atype{T}}(undef, length(reparity))
        div = division(N, a)
        parity = map(x->[[sum(x[div[i] + 1 : div[i + 1]]) for i in 1:(length(div)-1)]...] .% 2, reparity)
        uparity = A.parity
        deven, dodd = bulkdims(a...)
        for i in 1:length(uparity)
            p = uparity[i]
            ind = findall(x->x in [p], parity)
            dims = map(p -> Tuple(p[i] == 0 ? deven[i] : dodd[i] for i in 1:length(a)), reparity[ind])
            resize = map(x->[[prod(x[div[i] + 1 : div[i + 1]]) for i in 1:(length(div)-1)]...], dims)
            resize1 = Int[0, [resize[i][1] for i in 1:length(resize)]...]
            for j in 1:length(ind)
                re = (sum(resize1), resize[1][2:length(N)]...)
                ylen = [sum(resize1[1:j])+1:sum(resize1[1:j+1]), [(:) for _ in 1:length(N)-1]...]
                tensor[ind[j]] = reshape(reshape(A.tensor[i],re)[ylen...], dims[j])
            end
        end
        Z2tensor(reparity, tensor, a, 1)
    end
end

# for ein"abab ->"(A)[]
function dtr(A::AbstractZ2Array{T,N}) where {T,N}
    parity = A.parity
    tensor = A.tensor
    s = 0.0
    @inbounds for i in 1:length(parity)
        parity[i][1] == parity[i][3] && parity[i][2] == parity[i][4] && (s += Array(ein"abab ->"(tensor[i]))[])
    end
    s
end
