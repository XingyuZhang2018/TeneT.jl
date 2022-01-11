import Base: ==, +, -, *, /, ≈, size, reshape, permutedims, transpose, conj, show, similar, adjoint, copy, sqrt, getindex, setindex!, Array, broadcasted, vec, map, ndims, indexin, sum
using BitBasis
using CUDA
import CUDA: CuArray
import LinearAlgebra: tr, norm, dot, rmul!, axpy!, mul!, diag, Diagonal, lmul!
import OMEinsum: _compactify!, subindex, einsum, Tr, Repeat, tensorpermute
import Zygote: accum
export AbstractZ2Array, Z2tensor

"""
    parity_conserving(T::Array)
Transform an arbitray tensor which has arbitray legs and each leg have index 1 or 2 into parity conserving form
----
The following is faster but rely on updates of CUDA.jl(available in master branch)
function parity_conserving!(T::Union{Array,CuArray})
	bits = map(x -> Int(ceil(log2(x))), size(T))
    T[map(x->sum(sum.(bitarray.((Tuple(x).-1) ,bits))) % 2 !== 0 ,CartesianIndices(T))].=0
    return T
end
parity_conserving(T) = parity_conserving!(copy(T))
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
"""
    Z2tensor{T, N}

a struct to hold the N-order Z2 tensors
- `parity`: `2^(N-1)` `N`-length `0/1` Tuple
- `tensor`: `2^(N-1)` bulk tensor
- `size`  : original none-symetric array size
- `dims` size of `tensor`
- `division`: division location for reshape
"""
struct Z2tensor{T, N} <: AbstractZ2Array{T,N}
    parity::Tuple{Vararg{Tuple{Vararg{Int, N}}}}
    tensor::Vector{AbstractArray{T}}
    size::Tuple{Vararg{Int, N}}
    dims::Tuple{Vararg{Tuple{Vararg{Int, N}}}}
    division::Int
    function Z2tensor(parity::Tuple{Vararg{Tuple{Vararg{Int, N}}}}, tensor::Vector{<:AbstractArray{T}}, size::Tuple{Vararg{Int, N}}, dims::Tuple{Vararg{Tuple{Vararg{Int, N}}}}, division::Int) where {T,N}
        new{T, N}(parity, tensor, size, dims, division)
    end
end

size(A::AbstractZ2Array) = A.size
size(A::AbstractZ2Array, a) = size(A)[a]
conj(A::AbstractZ2Array) = Z2tensor(A.parity, map(conj, A.tensor), A.size, A.dims, A.division)
map(conj, A::AbstractZ2Array) = conj(A)
norm(A::AbstractZ2Array) = norm(A.tensor)

*(A::AbstractZ2Array, B::Number) = Z2tensor(A.parity, A.tensor * B, A.size, A.dims, A.division)
*(B::Number, A::AbstractZ2Array{T,N}) where {T,N} = A * B
/(A::AbstractZ2Array{T,N}, B::Number) where {T,N} = Z2tensor(A.parity, A.tensor / B, A.size, A.dims, A.division)
broadcasted(*, A::AbstractZ2Array, B::Number) = A * B
broadcasted(*, B::Number, A::AbstractZ2Array) = A * B
broadcasted(/, A::AbstractZ2Array, B::Number) = A / B

function +(A::AbstractZ2Array, B::AbstractZ2Array)
    if B.parity == A.parity
        Z2tensor(B.parity, A.tensor + B.tensor, B.size, B.dims, B.division)
    else
        exchangeind = indexin(B.parity, A.parity)
        Z2tensor(B.parity, A.tensor[exchangeind] + B.tensor, B.size, B.dims, B.division)
    end
end

function -(A::AbstractZ2Array, B::AbstractZ2Array)
    if B.parity == A.parity
        Z2tensor(B.parity, A.tensor - B.tensor, B.size, B.dims, B.division)
    else
        exchangeind = indexin(B.parity, A.parity)
        Z2tensor(B.parity, A.tensor[exchangeind] - B.tensor, B.size, B.dims, B.division)
    end
end

-(A::AbstractZ2Array) = Z2tensor(A.parity, map(-, A.tensor), A.size, A.dims, A.division)

indexin(A::Tuple, B::Tuple) = indexin(A, [B...])

CuArray(A::AbstractZ2Array) = Z2tensor(A.parity, map(CuArray, A.tensor), A.size, A.dims, A.division)
Array(A::AbstractZ2Array) = Z2tensor(A.parity, map(Array, A.tensor), A.size, A.dims, A.division)

function dot(A::AbstractZ2Array, B::AbstractZ2Array) 
    if A.parity == B.parity 
        dot(A.tensor, B.tensor)
    else
        exchangeind = indexin(A.parity, B.parity)
        dot(A.tensor, B.tensor[exchangeind])
    end
end

function ≈(A::AbstractZ2Array{TA,NA}, B::AbstractZ2Array{TB,NB}) where {TA,NA,TB,NB}
    NA != NB && throw(Base.error("$A and $B have different dimensions"))
    exchangeind = indexin(A.parity, B.parity)
    A.tensor ≈ B.tensor[exchangeind]
end

function show(::IOBuffer, A::Z2tensor)
    println("parity: \n", A.parity)
    println("dims: \n", A.dims)
    println("tensor: \n", A.tensor)
end

getindex(A::AbstractZ2Array, index::CartesianIndex) = getindex(A, index.I...)
function getindex(A::AbstractZ2Array, index::Int...)
    bits = map(x -> ceil(Int, log2(x)), size(A))
    parity = map((index, bits) -> sum(bitarray(index - 1, bits)) % 2, index, bits)
    sum(parity) % 2 != 0 && return 0.0
    ip = findfirst(x->x in [parity], A.parity)
    CUDA.@allowscalar A.tensor[ip][(Int.(floor.((index .-1 ) ./ 2)) .+ 1)...]
end

setindex!(A::AbstractZ2Array, x::Number, index::CartesianIndex) = setindex!(A, x, index.I...)
function setindex!(A::AbstractZ2Array, x::Number, index::Int...)
    bits = map(x -> ceil(Int, log2(x)), size(A))
    parity = map((index, bits) -> sum(bitarray(index - 1, bits)) % 2, index, bits)
    ip = findfirst(x->x in [parity], A.parity)
    CUDA.@allowscalar A.tensor[ip][(Int.(floor.((index .-1 ) ./ 2)) .+ 1)...] = x
end

"""
    deven, dodd = bulkdims(N::Int...)

find even and odd part dims of Z2 tensor bulk
"""
function bulkdims(N::Int...)
    bits = map(x -> ceil(Int, log2(x)), N)
    dodd = map((bits, N) -> sum([sum(bitarray(i - 1, bits)) % 2 for i = 1:N]), bits, N)
    deven = N .- dodd
    deven, dodd
end

function randZ2(atype, dtype, a...)
    L = length(a)
    deven, dodd = bulkdims(a...)
    parity = []
    tensor = Vector{atype{dtype}}()
    @inbounds for i in CartesianIndices(Tuple(0:1 for i=1:L))
        if sum(i.I) % 2 == 0
            push!(parity, i.I)
            dims = Tuple(i.I[j] == 0 ? deven[j] : dodd[j] for j in 1:L)
            push!(tensor, atype(rand(dtype, dims)))
        end
    end
    dims = Tuple(map(size, tensor))
    Z2tensor(Tuple(parity), tensor, a, dims, 1)
end

function zerosZ2(atype, dtype, a...)
    L = length(a)
    deven, dodd = bulkdims(a...)
    parity = []
    tensor = Vector{atype{dtype}}()
    @inbounds for i in CartesianIndices(Tuple(0:1 for i=1:L))
        if sum(i.I) % 2 == 0
            push!(parity, i.I)
            dims = Tuple(i.I[j] == 0 ? deven[j] : dodd[j] for j in 1:L)
            push!(tensor, atype(zeros(dtype, dims)))
        end
    end
    dims = Tuple(map(size, tensor))
    Z2tensor(Tuple(parity), tensor, a, dims, 1)
end

function IZ2(atype, dtype, D)
    deven, dodd = bulkdims(D, D)
    Z2tensor(((0, 0), (1, 1)), [atype{dtype}(I, deven...), atype{dtype}(I, dodd...)], (D, D), (deven, dodd), 1)
end

# only for OMEinsum binary permutedims before reshape
permutedims(A::AbstractZ2Array, perm) = tensorpermute(A, perm)
function tensorpermute(A::AbstractZ2Array, perm)
    length(perm) == 0 && return copy(A)
    parity = map(x -> x[collect(perm)], A.parity)
    exchangeind = indexin(A.parity, parity)
    tensor = map(x -> permutedims(x, perm), A.tensor)[exchangeind]
    dims = Tuple(map(x -> x[collect(perm)], A.dims)[exchangeind])
    Z2tensor(A.parity, tensor, A.size[collect(perm)], dims, A.division)
end

reshape(A::AbstractZ2Array, a::Tuple{Vararg{Int}}) = reshape(A, a...)
function reshape(A::AbstractZ2Array{T,N}, a::Int...) where {T,N}
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
        return Z2tensor(A.parity, tensor, A.size, A.dims, div)
    else
        tensor = map((x, y) -> reshape(x, y), Atensor, A.dims)
        return Z2tensor(A.parity, tensor, A.size, A.dims, A.division)
    end
end

"""
    *(A::AbstractZ2Array{TA,NA}, B::AbstractZ2Array{TB,NB}) where {TA,TB,NA,NB}

core code for Z2tensor product
"""
function *(A::AbstractZ2Array{TA,NA}, B::AbstractZ2Array{TB,NB}) where {TA,TB,NA,NB}
    parity = []
    dims = []
    atype = _arraytype(B.tensor[1])
    T = TA == ComplexF64 || TB == ComplexF64 ? ComplexF64 : Float64
    tensor = Vector{atype{T}}()
    divA, divB = A.division, B.division

    bulktimes!(parity, tensor, dims, A, B, 0)
    !(divA in [0, NA]) && !(divB in [0, NB]) && bulktimes!(parity, tensor, dims, A, B, 1)
    parity == [()] && return Array(tensor[1])[]
    Z2tensor(Tuple(parity), tensor, (size(A)[1:divA]..., size(B)[divB+1:end]...), Tuple(dims), divA)
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

    ind_A = findall(x->sum(x[divA+1:end]) % 2 == p, Aparity)
    matrix_j = unique(map(x->x[divA+1:end], Aparity[ind_A]))
    matrix_i = unique(map(x->x[1:divA], Aparity[ind_A]))
    ind_B = findall(x->x[1:divB] in matrix_j, Bparity)
    matrix_k = unique(map(x->x[divB+1:end], Bparity[ind_B]))

    #opt push!
    h, bulkidims, oribulkidims = [] , Int[], []
    for i in matrix_i
        v = []
        ind = 0
        for j in matrix_j
            ind = findfirst(x->x in [(i..., j...)], Aparity)
            push!(v, Atensor[ind])
        end
        push!(oribulkidims, Adims[ind][1:divA]) 
        hi = hcat(v...)
        push!(h, hi)
        push!(bulkidims, size(hi)[1])
    end
    Amatrix = vcat(h...)

    v, bulkjdims, oribulkjdims = [], Int[], []
    for k in matrix_k
        h = []
        ind = 0
        for j in matrix_j
            ind = findfirst(x->x in [(j..., k...)], Bparity)
            push!(h, Btensor[ind])
        end
        push!(oribulkjdims, Bdims[ind][divB+1:end])
        hj = vcat(h...)
        push!(v, hj)
        push!(bulkjdims, size(hj)[2])
    end
    Bmatrix = hcat(v...)

    atype = _arraytype(Btensor[1])
    C = atype(Amatrix) * atype(Bmatrix)
    for i in 1:length(matrix_i), j in 1:length(matrix_k)
        push!(parity, (matrix_i[i]..., matrix_k[j]...))
        push!(dims, (oribulkidims[i]..., oribulkjdims[j]...))
        idim, jdim = sum(bulkidims[1:i-1])+1:sum(bulkidims[1:i]), sum(bulkjdims[1:j-1])+1:sum(bulkjdims[1:j])
        push!(tensor, C[idim, jdim])
    end
end

function Z2bitselection(maxN::Int)
    q = [sum(bitarray(i-1,ceil(Int,log2(maxN)))) % 2 for i = 1:maxN]
    return [(q .== 0),(q .== 1)]
end

function Z2tensor2tensor(A::Z2tensor{T,N}) where {T,N}
    atype = _arraytype(A.tensor[1])
    tensor = zeros(T, size(A))
    parity = A.parity
    qlist = [Z2bitselection(size(A)[i]) for i = 1:N]
    for i in 1:length(parity)
        tensor[[qlist[j][parity[i][j]+1] for j = 1:N]...] = Array(A.tensor[i])
    end
    atype(tensor)
end

# have Bugs with CuArray, rely on https://github.com/JuliaGPU/CUDA.jl/issues/1304
function tensor2Z2tensor(A::AbstractArray{T,N}) where {T,N}
    atype = _arraytype(A)
    Aarray = Array(A)
    qlist = [Z2bitselection(size(A)[i]) for i = 1:N]
    parity = getparity(N)
    tensor = [atype(Aarray[[qlist[j][parity[i][j]+1] for j = 1:N]...]) for i in 1:length(parity)]
    dims = Tuple(map(size, tensor))
    Z2tensor(parity, tensor, size(A), dims, 1)
end

# for OMEinsum contract to get number
vec(A::AbstractZ2Array) = A

function transpose(A::AbstractZ2Array{T,N}) where {T,N}
    tensor = map(transpose, A.tensor)
    Z2tensor(A.parity, tensor, A.size, A.dims, 0)
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

function _compactify!(y, x::AbstractZ2Array, indexer)
    x = Z2tensor2tensor(Array(x))
    @inbounds for ci in CartesianIndices(y)
        y[ci] = x[subindex(indexer, ci.I)]
    end
    return y
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


# for Zygote compatibility
accum(A::AbstractZ2Array, B::AbstractZ2Array...) = +(A, B...)
sum(()) = 0

# for KrylovKit compatibility
rmul!(A::AbstractZ2Array, B::Number) = (map(x -> rmul!(x, B), A.tensor); A)

function lmul!(A::AbstractZ2Array{T,N}, B::AbstractZ2Array) where {T,N}
    C = A*B
    for i = 1:2^(N-1)
        setindex!(B.tensor, C.tensor[i], i)
    end
    return B
end

similar(A::AbstractZ2Array) = Z2tensor(A.parity, map(similar, A.tensor), A.size, A.dims, A.division)
similar(A::AbstractZ2Array, atype) = Z2tensor(A.parity, map(x -> atype(similar(x)), A.tensor), A.size, A.dims, A.division)
diag(A::AbstractZ2Array{T,N}) where {T,N} = CUDA.@allowscalar collect(Iterators.flatten(diag.(A.tensor)))
copy(A::AbstractZ2Array{T,N}) where {T,N} = Z2tensor(A.parity, map(copy, A.tensor), A.size, A.dims, A.division)

mul!(Y::AbstractZ2Array, A::AbstractZ2Array, B::Number) = (map((Y, A) -> mul!(Y, A, B), Y.tensor, A.tensor); Y)

function axpy!(α::Number, A::AbstractZ2Array, B::AbstractZ2Array)
    if B.parity == A.parity
        map((x,y) -> axpy!(α, x, y), A.tensor, B.tensor)
    else
        exchangeind = indexin(B.parity, A.parity)
        map((x,y) -> axpy!(α, x, y), A.tensor[exchangeind], B.tensor)
    end
    return B
end

# for leftorth and rightorth compatibility
Diagonal(A::AbstractZ2Array) = Z2tensor(A.parity, map(Diagonal, A.tensor), A.size, A.dims, A.division)
sqrt(A::AbstractZ2Array) = Z2tensor(A.parity, map(x->sqrt.(x), A.tensor), A.size, A.dims, A.division)
broadcasted(sqrt, A::AbstractZ2Array) = sqrt(A)

# only for order-three tensor's qr and lq
function qrpos!(A::AbstractZ2Array{T,N}) where {T,N}
    Qparity = []
    Rparity = []
    atype = _arraytype(A.tensor[1])
    Qtensor = Vector{atype{T}}()
    Rtensor = Vector{atype{T}}()

    bulkQR!(Qparity, Qtensor, Rparity, Rtensor, A, 0)
    bulkQR!(Qparity, Qtensor, Rparity, Rtensor, A, 1)
    Asize = A.size
    Adims = A.dims
    exchangeind = indexin(A.parity, Qparity)
    Z2tensor(A.parity, Qtensor[exchangeind], Asize, Adims, A.division), Z2tensor(Tuple(Rparity), Rtensor, (Asize[end], Asize[end]), Tuple(map(size, Rtensor)), 1)
end

function bulkQR!(Qparity, Qtensor, Rparity, Rtensor, A, p)
    Atensor = A.tensor
    Aparity = A.parity
    div = A.division

    ind_A = findall(x->sum(x[div+1:end]) % 2 == p, Aparity)
    matrix_j = unique(map(x->x[div+1:end], Aparity[ind_A]))
    matrix_i = unique(map(x->x[1:div], Aparity[ind_A]))

    ind = [findfirst(x->x in [(i..., matrix_j[1]...)], Aparity) for i in matrix_i]
    Amatrix = vcat(Atensor[ind]...)
    bulkidims = [size(Atensor[i],1) for i in ind]
    bulkjdims = [size(Amatrix, 2)]

    Q, R = qrpos!(Amatrix)
    for i in 1:length(matrix_i), j in 1:length(matrix_j)
        push!(Qparity, (matrix_i[i]..., matrix_j[j]...))
        idim, jdim = sum(bulkidims[1:i-1])+1:sum(bulkidims[1:i]), sum(bulkjdims[1:j-1])+1:sum(bulkjdims[1:j])
        push!(Qtensor, Q[idim, jdim])
    end
    for i in 1:length(matrix_j), j in 1:length(matrix_j)
        push!(Rparity, (matrix_j[i]..., matrix_j[j]...))
        idim, jdim = sum(bulkjdims[1:i-1])+1:sum(bulkjdims[1:i]), sum(bulkjdims[1:j-1])+1:sum(bulkjdims[1:j])
        push!(Rtensor, R[idim, jdim])
    end
end

function lqpos!(A::AbstractZ2Array{T,N}) where {T,N}
    Lparity = []
    Qparity = []
    atype = _arraytype(A.tensor[1])
    Ltensor = Vector{atype{T}}()
    Qtensor = Vector{atype{T}}()

    bulkLQ!(Lparity, Ltensor, Qparity, Qtensor, A, 0)
    bulkLQ!(Lparity, Ltensor, Qparity, Qtensor, A, 1)
    Asize = A.size
    Adims = A.dims
    exchangeind = indexin(A.parity, Qparity)
    Z2tensor(Tuple(Lparity), Ltensor, (Asize[1], Asize[1]), Tuple(map(size, Ltensor)), 1), Z2tensor(A.parity, Qtensor[exchangeind], Asize, Adims, A.division)
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
            ind = findfirst(x->x in [(i..., j...)], Aparity)
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
        push!(Lparity, (matrix_i[i]..., matrix_i[j]...))
        idim, jdim = sum(bulkidims[1:i-1])+1:sum(bulkidims[1:i]), sum(bulkidims[1:j-1])+1:sum(bulkidims[1:j])
        push!(Ltensor, L[idim, jdim])
    end
    for i in 1:length(matrix_i), j in 1:length(matrix_j)
        push!(Qparity, (matrix_i[i]..., matrix_j[j]...))
        idim, jdim = sum(bulkidims[1:i-1])+1:sum(bulkidims[1:i]), sum(bulkjdims[1:j-1])+1:sum(bulkjdims[1:j])
        push!(Qtensor, Q[idim, jdim])
    end
end

# for ' in ACCtoALAR of VUMPS
function adjoint(A::AbstractZ2Array{T,N}) where {T,N}
    div = A.division 
    parity = map(x->x[[div+1:end;1:div]], A.parity)
    exchangeind = indexin(A.parity, parity)
    tensor = map(adjoint, A.tensor)[exchangeind]
    # dims = Tuple(map(x -> x[[div+1:end;1:div]], A.dims)[exchangeind])
    Z2tensor(A.parity, tensor, A.size[[div+1:end;1:div]], A.dims[exchangeind], N - div)
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
    Nm = Tuple(map(x->min(x...), A.dims))
    N1 = Tuple(map((x, y) -> (x[1], y), A.dims, Nm))
    N2 = Tuple(map((x, y) -> (y, x[2]), A.dims, Nm))
    Asize = A.size
    sm = min(Asize...)
    Z2tensor(parity, Utensor, (Asize[1], sm), N1, div), Z2tensor(parity, Stensor, (sm, sm), (Nm, Nm), div), Z2tensor(parity, Vtensor, (sm, Asize[2]), N2, div)
end

"""
p = getparity(L::Int)

give the parity of length L
"""
function getparity(L::Int)
    p = []
    for i in CartesianIndices(Tuple(0:1 for i=1:L))
        sum(i.I) % 2 == 0 && push!(p, i.I)
    end
    Tuple(p)
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
    Z2reshape(A::AbstractZ2Array{T, N}, a::Int...) where {T, N}

Z2reshape only for shape `(D,D,D,D,D,D,D,D) <-> (D^2,D^2,D^2,D^2)` and `(χ,D,D,χ) <-> (χ,D^2,χ)`
"""
Z2reshape(A::AbstractZ2Array, a::Tuple{Vararg{Int}}) = Z2reshape(A, a...)
function Z2reshape(A::AbstractZ2Array{T, N}, a::Int...) where {T, N}
    atype = _arraytype(A.tensor[1])
    if N > length(a)
        div = division(a, size(A))
        reparity = Tuple([Tuple([sum(p[d]) % 2 for d in div]) for p in A.parity])
        redims = Tuple([Tuple([prod(dims[d]) for d in div]) for dims in A.dims])
        retensor = [reshape(t, s) for (t, s) in zip(Array.(A.tensor), redims)]
        ureparity = getparity(length(a))
        retensors = Vector{atype{T}}()
        for i in 1:length(ureparity)
            p = ureparity[i]
            bulkind = findall(x->x in [p], reparity)
            rebulkdims = Tuple(Int.(.+(redims[bulkind]...) ./ (length(bulkind) ./ length.(div))))
            rebulkdims1 = redims[bulkind[1]]
            silce = [[1:rebulkdims1[i], (rebulkdims1[i] == rebulkdims[i] ? 1 : 1+rebulkdims1[i]):rebulkdims[i]] for i in 1:length(rebulkdims)]
            tensor = atype(zeros(T, rebulkdims))
            bits = Int(log2(length(bulkind)))
            for j in 1:length(bulkind)
                choose = bitarray(j - 1, bits) .+ 1
                length(choose) == 1 && (choose = [choose[], choose[], choose[]])
                choosesilce = [silce[i][choose[i]] for i in 1:length(silce)]
                tensor[choosesilce...] = retensor[bulkind[j]]
            end
            push!(retensors, tensor)
        end
        dims = Tuple(map(size, retensors))
        Z2tensor(ureparity, atype.(retensors), a, dims, 1)
    else
        div = division(size(A), a)
        reparity = getparity(length(a))
        parity = Tuple([Tuple([sum(p[d]) % 2 for d in div]) for p in reparity])
        rebulkdims = bulkdims(a...)
        redims = Tuple([Tuple([rebulkdims[p[i] + 1][i] for i in 1:length(a)]) for p in reparity])
        dims = Tuple([Tuple([prod(dims[d]) for d in div]) for dims in redims])
        retensors = Array{Array,1}(undef, length(reparity))
        for i in 1:length(A.parity)
            p = A.parity[i]
            bulkind = findall(x->x in [p], parity)
            bulkdims = Int.(.+(dims[bulkind]...) ./ (length(bulkind) ./ length.(div)))
            bulkdims1 = dims[bulkind[1]]
            silce = [[1:bulkdims1[i], (bulkdims1[i] == bulkdims[i] ? 1 : 1+bulkdims1[i]):bulkdims[i]] for i in 1:length(bulkdims)]
            bits = Int(log2(length(bulkind)))
            for j in 1:length(bulkind)
                choose = bitarray(j - 1, bits) .+ 1
                length(choose) == 1 && (choose = [choose[], choose[], choose[]])
                choosesilce = [silce[i][choose[i]] for i in 1:length(silce)]
                retensors[bulkind[j]] = reshape(Array(A.tensor[i])[choosesilce...], redims[bulkind[j]])
            end
        end
        dims = Tuple(map(size, retensors))
        Z2tensor(reparity, atype.(retensors), a, dims, 1)
    end
end
