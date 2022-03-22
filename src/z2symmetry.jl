import Base: ==, +, -, *, /, ≈, size, reshape, permutedims, transpose, conj, show, similar, adjoint, copy, sqrt, getindex, setindex!, Array, broadcasted, vec, map, ndims, indexin, sum, zero
using BitBasis
using CUDA
import CUDA: CuArray
import LinearAlgebra: tr, norm, dot, rmul!, axpy!, mul!, diag, Diagonal, lmul!
import OMEinsum: _compactify!, subindex, einsum, Tr, Repeat, tensorpermute
import Zygote: accum
export Z2Array, AbstractSymmetricArray
export parity_conserving, randZ2, asZ2Array, asArray, Z2reshape

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

abstract type AbstractSymmetricArray{T,N} <: AbstractArray{T,N} end
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
    parity::Vector{<:Vector}
    dims::Vector{<:Vector}
    tensor::Vector{AbstractArray{T}}
    size::Tuple{Vararg{Int, N}}
    function Z2Array(parity::Vector{<:Vector}, dims::Vector{<:Vector}, tensor::Vector{<:AbstractArray{T}}, size::Tuple{Vararg{Int, N}}, ) where {T,N}
        new{T, N}(parity, dims, tensor, size)
    end
end

function show(::IOBuffer, A::Z2Array)
    println("parity: \n", A.parity)
    println("dims: \n", A.dims)
    println("tensor: \n", A.tensor)
end


"""
    [deven, dodd] = z2indexdims(Nmax::Int)

find even and odd part dims of each index in Z2 tensor 
"""
function z2indexdims(Nmax::Int)
    bit = ceil(Int, log2(Nmax))
    odddim = sum([sum(bitarray(i - 1, bit)) % 2 for i = 1:Nmax])
    evendim = Nmax - odddim
    [evendim, odddim]
end

"""
    [evendim, odddim] = z2fusedims(s::Vector{<:Vector{Int}})

find fuse dimensions of mutiple size
"""
function z2fusedims(s::Vector{<:Vector{Int}})
    dims = [0, 0]
    for i in CartesianIndices(Tuple(0:1 for i=1:length(s)))
        d = prod(map((dim,ind)->dim[ind], s, [i.I .+ 1...]))
        dims[sum(i.I) % 2 + 1] += d
    end
    dims
end

"""
    randZ2(atype, dtype, s)

matrix form of a rand Z2 tensor: s have form of ((a,b,c...), (d,e,f...))
"""
function randZ2(atype, dtype, s)
    parity = [[[0,1] for _ in 1:length(si)] for si in s]
    dims = [[z2indexdims(si[j]) for j in 1:length(si)] for si in s]
    bulkdims = z2fusedims.(dims)
    tensor = [atype(rand(dtype, bulkdims[1][i], bulkdims[2][i])) for i = 1:2]
    Z2Array(parity, dims, tensor, (s[1]..., s[2]...))
end

"""
    zerosZ2(atype, dtype, s)

matrix form of a zeros Z2 tensor: s have form of ((a,b,c...), (d,e,f...))
"""
function zerosZ2(atype, dtype, s)
    parity = [[[0,1] for _ in 1:length(si)] for si in s]
    dims = [[z2indexdims(si[j]) for j in 1:length(si)] for si in s]
    bulkdims = z2fusedims.(dims)
    tensor = [zeros(dtype, bulkdims[1][i], bulkdims[2][i]) for i = 1:2]
    Z2Array(parity, dims, tensor, (s[1]..., s[2]...))
end

"""
    IZ2(atype, dtype, D)

Identitiy Z2 matrix
"""
function IZ2(atype, dtype, D)
    dims = z2indexdims(D)
    tensor = [atype{dtype}(I, dims[i], dims[i]) for i = 1:2]
    Z2Array([[0,1],[0,1]], [dims, dims], tensor, (D, D))
end

zero(A::Z2Array) = Z2Array(A.parity, A.dims, map(zero, A.tensor), A.size)

function ≈(A::Z2Array{TA,NA}, B::Z2Array{TB,NB}) where {TA,NA,TB,NB}
    NA != NB && throw(Base.error("$A and $B have different dimensions"))
    A.tensor ≈ B.tensor
end

function Z2bitselection(maxN::Int)
    q = [sum(bitarray(i-1,ceil(Int,log2(maxN)))) % 2 for i = 1:maxN]
    return [(q .== 0),(q .== 1)]
end

function parityfuseparts(s::Vector{<:Vector{Int}})
    dims = [0, 0]
    parts = Vector{UnitRange{Int64}}()
    for i in CartesianIndices(Tuple(0:1 for i=1:length(s)))
        d = prod(map((dim,ind)->dim[ind], s, collect(i.I .+ 1)))
        p = sum(i.I) % 2 + 1
        push!(parts, dims[p]+1:dims[p]+d)
        dims[p] += d
    end
    parts
end

function asArray(A::Z2Array{T,N}) where {T,N}
    atype = _arraytype(A.tensor[1])
    tensor = zeros(T, size(A))
    Atensor = map(Array, A.tensor)
    qlist = [Z2bitselection(size(A)[i]) for i = 1:N]
    column = parityfuseparts(A.dims[1])
    row = parityfuseparts(A.dims[2])
    Lcol = length(A.parity[1])
    dims = [A.dims[1]..., A.dims[2]...]
    for i in CartesianIndices(Tuple(0:1 for i=1:N))
        if sum(i.I) % 2 == 0
            parts = [column[packbits(collect(i.I[1:Lcol]))+1], row[packbits(collect(i.I[Lcol+1:end]))+1]]
            p = sum(i.I[1:Lcol]) % 2 + 1
            tensor[[qlist[j][i.I[j]+1] for j = 1:N]...] = reshape(Atensor[p][parts...], map((dim, ind)->dim[ind+1], dims, i.I)...)
        end
    end
    atype(tensor)
end

# have Bugs with CUDA@v3.5.0, rely on https://github.com/JuliaGPU/CUDA.jl/issues/1304
# which is fixed in new vervion, but its allocation is abnormal
function asZ2Array(A::AbstractArray{T,N}, s) where {T,N}
    atype = _arraytype(A)
    Z2array = zerosZ2(atype, T, s)

    qlist = [Z2bitselection(size(A, i)) for i = 1:N]
    column = parityfuseparts(Z2array.dims[1])
    row = parityfuseparts(Z2array.dims[2])
    Lcol = length(s[1])
    dims = [Z2array.dims[1]..., Z2array.dims[2]...]
    for i in CartesianIndices(Tuple(0:1 for i=1:N))
        if sum(i.I) % 2 == 0
            parts = [column[packbits(collect(i.I[1:Lcol]))+1], row[packbits(collect(i.I[Lcol+1:end]))+1]]
            p = sum(i.I[1:Lcol]) % 2 + 1
            shape = map((dim, ind)->dim[ind+1], dims, i.I)
            Z2array.tensor[p][parts...] = reshape(A[[qlist[j][i.I[j]+1] for j = 1:N]...], prod(shape[1:Lcol]), prod(shape[Lcol+1:end]))
        end
    end
    Z2array
end

size(A::Z2Array) = A.size
size(A::Z2Array, a) = size(A)[a]
conj(A::Z2Array) = Z2Array(A.parity, A.dims, map(conj, A.tensor), A.size)
map(conj, A::Z2Array) = conj(A)
norm(A::Z2Array) = norm(A.tensor)

*(A::Z2Array, B::Number) = Z2Array(A.parity, A.dims, A.tensor * B, A.size)
*(B::Number, A::Z2Array{T,N}) where {T,N} = A * B
/(A::Z2Array{T,N}, B::Number) where {T,N} = Z2Array(A.parity, A.dims, A.tensor / B, A.size)
broadcasted(*, A::Z2Array, B::Number) = Z2Array(A.parity, A.dims, A.tensor .* B, A.size)
broadcasted(*, B::Number, A::Z2Array) = Z2Array(A.parity, A.dims, A.tensor .* B, A.size)
broadcasted(/, A::Z2Array, B::Number) = A / B

+(A::Z2Array, B::Z2Array) = Z2Array(A.parity, A.dims, A.tensor + B.tensor, A.size)
-(A::Z2Array, B::Z2Array) = Z2Array(A.parity, A.dims, A.tensor - B.tensor, A.size)

-(A::Z2Array) = Z2Array(A.parity, A.dims, map(-, A.tensor), A.size)

CuArray(A::Z2Array) = Z2Array(A.parity, A.dims, map(CuArray, A.tensor), A.size)
Array(A::Z2Array) = Z2Array(A.parity, A.dims, map(Array, A.tensor), A.size)

dot(A::Z2Array, B::Z2Array) = dot(A.tensor, B.tensor)

# getindex(A::Z2Array, index::CartesianIndex) = getindex(A, index.I...)
# function getindex(A::Z2Array, index::Int...)
#     bits = map(x -> ceil(Int, log2(x)), size(A))
#     parity = collect(map((index, bits) -> sum(bitarray(index - 1, bits)) % 2, index, bits))
#     sum(parity) % 2 != 0 && return 0.0
#     ip = findfirst(x->x in [parity], A.parity)
#     CUDA.@allowscalar A.tensor[ip][(Int.(floor.((index .-1 ) ./ 2)) .+ 1)...]
# end

# setindex!(A::Z2Array, x::Number, index::CartesianIndex) = setindex!(A, x, index.I...)
# function setindex!(A::Z2Array, x::Number, index::Int...)
#     bits = map(x -> ceil(Int, log2(x)), size(A))
#     parity = collect(map((index, bits) -> sum(bitarray(index - 1, bits)) % 2, index, bits))
#     ip = findfirst(x->x in [parity], A.parity)
#     CUDA.@allowscalar A.tensor[ip][(Int.(floor.((index .-1 ) ./ 2)) .+ 1)...] = x
# end

# only for OMEinsum binary permutedims before reshape
permutedims(A::Z2Array, perm) = tensorpermute(A, perm)
function tensorpermute(A::Z2Array{T, N}, perm) where {T, N}
    length(perm) == 0 && return copy(A)
    Asizep = A.size[collect(perm)]
    sp = length(perm) > 2 ? (Asizep[1:2], Asizep[3:end]) : (Tuple(Asizep[1]), Tuple(Asizep[2]))
    Ap = zerosZ2(_arraytype(A.tensor[1]), T, sp) # maybe initialize from A will be faster

    oricolumn = parityfuseparts(A.dims[1])
    orirow = parityfuseparts(A.dims[2])
    oriLcol = length(A.parity[1])
    column = parityfuseparts(Ap.dims[1])  # check if have same dims so that same parts
    row = parityfuseparts(Ap.dims[2])
    Lcol = length(perm) > 2 ? 2 : 1
    oridims = [A.dims[1]..., A.dims[2]...]
    dims = [Ap.dims[1]..., Ap.dims[2]...]
    for i in CartesianIndices(Tuple(0:1 for i=1:N))
        if sum(i.I) % 2 == 0
            oriparts = [oricolumn[packbits(collect(i.I[1:oriLcol]))+1], orirow[packbits(collect(i.I[oriLcol+1:end]))+1]]
            orip = sum(i.I[1:oriLcol]) % 2 + 1
            indp = collect(i.I)[collect(perm)]
            parts = [column[packbits(indp[1:Lcol])+1], row[packbits(indp[Lcol+1:end])+1]]
            p = sum(indp[1:Lcol]) % 2 + 1
            orishape = map((dim, ind)->dim[ind+1], oridims, i.I)
            shape = map((dim, ind)->dim[ind+1], dims, indp)
            Ap.tensor[p][parts...] .= reshape(permutedims(reshape(A.tensor[orip][oriparts...], orishape...), perm), prod(shape[1:Lcol]), prod(shape[Lcol+1:end]))
        end
    end
    return Ap
end

reshape(A::Z2Array, s::Tuple{Vararg{Int}}) = reshape(A, s...)
function reshape(A::Z2Array{T, N}, s::Int...) where {T, N}
    div = 1
    if length(s) < N
        sizeA = size(A)
        p = sizeA[1]
        while p != s[1]
            div += 1
            p *= sizeA[div]
        end
        if div == length(A.parity[1])
            return A
        else
            sr = [A.size[1:div], A.size[div+1:end]]
            Ar = zerosZ2(_arraytype(A.tensor[1]), T, sr) # maybe initialize from A will be faster

            oricolumn = parityfuseparts(A.dims[1])
            orirow = parityfuseparts(A.dims[2])
            oriLcol = length(A.parity[1])
            column = parityfuseparts(Ar.dims[1])  # check if have same dims so that same parts
            row = parityfuseparts(Ar.dims[2])

            dims = [A.dims[1]..., A.dims[2]...]
            for i in CartesianIndices(Tuple(0:1 for i=1:N))
                if sum(i.I) % 2 == 0
                    oriparts = [oricolumn[packbits(collect(i.I[1:oriLcol]))+1], orirow[packbits(collect(i.I[oriLcol+1:end]))+1]]
                    orip = sum(i.I[1:oriLcol]) % 2 + 1
                    parts = [column[packbits(collect(i.I[1:div]))+1], row[packbits(collect(i.I[div+1:end]))+1]]
                    p = sum(i.I[1:div]) % 2 + 1
                    shape = map((dim, ind)->dim[ind+1], dims, i.I)
                    Ar.tensor[p][parts...] .= reshape(A.tensor[orip][oriparts...], prod(shape[1:div]), prod(shape[div+1:end]))
                end
            end
            return Ar
        end
    else
        return A
    end
end

"""
    *(A::Z2Array{TA,NA}, B::Z2Array{TB,NB}) where {TA,TB,NA,NB}

core code for Z2Array product
"""
function *(A::Z2Array, B::Z2Array)
    parity = [A.parity[1], B.parity[2]]
    dims = [A.dims[1], B.dims[2]]
    LcolA = length(A.parity[1])
    LcolB = length(B.parity[1])
    siz = (A.size[1:LcolA]..., B.size[LcolB+1:end]...)
    tensor = [A.tensor[i]*B.tensor[i] for i in 1:2]
    Z2Array(parity, dims, tensor, siz)
end

# for OMEinsum contract to get number
# vec(A::Z2Array) = A

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
    x = asArray(Array(x))
    @inbounds for ci in CartesianIndices(y)
        y[ci] = x[subindex(indexer, ci.I)]
    end
    return y
end

broadcasted(*, A::Z2Array, B::Base.RefValue) = Z2Array(A.parity, A.tensor .* B, A.size, A.dims)
broadcasted(*, B::Base.RefValue, A::Z2Array) = Z2Array(A.parity, A.tensor .* B, A.size, A.dims)

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

similar(A::Z2Array) = Z2Array(A.parity, map(similar, A.tensor), A.size, A.dims)
similar(A::Z2Array, atype) = Z2Array(A.parity, map(x -> atype(similar(x)), A.tensor), A.size, A.dims)
diag(A::Z2Array{T,N}) where {T,N} = CUDA.@allowscalar collect(Iterators.flatten(diag.(A.tensor)))
copy(A::Z2Array{T,N}) where {T,N} = Z2Array(A.parity, map(copy, A.tensor), A.size, A.dims)

mul!(Y::Z2Array, A::Z2Array, B::Number) = (map((Y, A) -> mul!(Y, A, B), Y.tensor, A.tensor); Y)

function axpy!(α::Number, A::Z2Array, B::Z2Array)
    if B.parity == A.parity
        map((x,y) -> axpy!(α, x, y), A.tensor, B.tensor)
    else
        exchangeind = indexin(B.parity, A.parity)
        map((x,y) -> axpy!(α, x, y), A.tensor[exchangeind], B.tensor)
    end
    return B
end

# for leftorth and rightorth compatibility
Diagonal(A::Z2Array) = Z2Array(A.parity, map(Diagonal, A.tensor), A.size, A.dims)
sqrt(A::Z2Array) = Z2Array(A.parity, map(x->sqrt.(x), A.tensor), A.size, A.dims)
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
    Z2Array(Qparity, Qtensor, Asize, Adims[exchangeind]), Z2Array(Rparity, Rtensor, (Asize[end], Asize[end]), map(x -> [size(x)...], Rtensor), 1)
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
    Z2Array(Lparity, Ltensor, (Asize[1], Asize[1]), map(x -> [size(x)...], Ltensor), 1), Z2Array(Qparity, Qtensor, Asize, Adims[exchangeind])
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

# for ' in ACCtoALAR of VUMPS
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
    Z2reshape(A::Z2Array{T, N}, a::Int...) where {T, N}

Z2reshape only for shape `(D,D,D,D,D,D,D,D) <-> (D^2,D^2,D^2,D^2)` and `(χ,D,D,χ) <-> (χ,D^2,χ)`
"""
Z2reshape(A::Z2Array, a::Tuple{Vararg{Int}}) = Z2reshape(A, a...)
function Z2reshape(A::Z2Array{T, N}, a::Int...) where {T, N}
    atype = _arraytype(A.tensor[1])
    orderedparity = getparity(N)
    if orderedparity == A.parity
        Atensor = A.tensor
        Adims = A.dims
    else
        exchangeind = indexin(orderedparity, A.parity)
        Atensor = A.tensor[exchangeind]
        Adims = A.dims[exchangeind]
    end
    if N > length(a)
        div = division(a, size(A))
        reparity = [[sum(p[d]) % 2 for d in div] for p in orderedparity]
        redims = [[prod(dims[d]) for d in div] for dims in Adims]
        retensor = [reshape(t, s...) for (t, s) in zip(map(Array, Atensor), redims)]
        ureparity = getparity(length(a))
        retensors = Vector{atype{T}}()
        for i in 1:length(ureparity)
            p = ureparity[i]
            bulkind = findall(x->x in [p], reparity)
            rebulkdims = Int.(.+(redims[bulkind]...) ./ (length(bulkind) ./ length.(div)))
            rebulkdims1 = redims[bulkind[1]]
            silce = [[1:rebulkdims1[i], (rebulkdims1[i] == rebulkdims[i] ? 1 : 1+rebulkdims1[i]):rebulkdims[i]] for i in 1:length(rebulkdims)]
            tensor = atype(zeros(T, rebulkdims...))
            bits = Int(log2(length(bulkind)))
            for j in 1:length(bulkind)
                choose = bitarray(j - 1, bits) .+ 1
                length(choose) == 1 && (choose = [choose[], choose[], choose[]])
                choosesilce = [silce[i][choose[i]] for i in 1:length(silce)]
                tensor[choosesilce...] = retensor[bulkind[j]]
            end
            push!(retensors, tensor)
        end
        dims = map(x -> [size(x)...], retensors)
        Z2Array(ureparity, atype.(retensors), a, dims, 1)
    else
        div = division(size(A), a)
        reparity = getparity(length(a))
        parity = [[sum(p[d]) % 2 for d in div] for p in reparity]
        rebulkdims = bulkdims(a...)
        redims = [[rebulkdims[p[i] + 1][i] for i in 1:length(a)] for p in reparity]
        dims = [[prod(dims[d]) for d in div] for dims in redims]
        retensors = Array{Array,1}(undef, length(reparity))
        for i in 1:length(orderedparity)
            p = orderedparity[i]
            bulkind = findall(x->x in [p], parity)
            bulkdims = Int.(.+(dims[bulkind]...) ./ (length(bulkind) ./ length.(div)))
            bulkdims1 = dims[bulkind[1]]
            silce = [[1:bulkdims1[i], (bulkdims1[i] == bulkdims[i] ? 1 : 1+bulkdims1[i]):bulkdims[i]] for i in 1:length(bulkdims)]
            bits = Int(log2(length(bulkind)))
            for j in 1:length(bulkind)
                choose = bitarray(j - 1, bits) .+ 1
                length(choose) == 1 && (choose = [choose[], choose[], choose[]])
                choosesilce = [silce[i][choose[i]] for i in 1:length(silce)]
                retensors[bulkind[j]] = reshape(Array(Atensor[i])[choosesilce...], redims[bulkind[j]]...)
            end
        end
        dims = map(x -> [size(x)...], retensors)
        Z2Array(reparity, atype.(retensors), a, dims, 1)
    end
end
