import Base: ==, +, -, *, /, ≈, size, reshape, permutedims, transpose, conj, show, similar
import LinearAlgebra: tr, norm, dot, rmul!, axpy!, mul!
import OMEinsum: _compactify!, subindex
import Random: rand
using BitBasis
using CUDA

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

size(::AbstractZ2Array{T,N}) where {T,N} = N
conj(A::AbstractZ2Array{T,N}) where {T,N} = Z2tensor(A.parity, map(conj,A.tensor), N, A.division)
norm(A::AbstractZ2Array{T,N}) where {T,N} = norm(A.tensor)

*(A::AbstractZ2Array{T,N}, B::Number) where {T,N} = Z2tensor(A.parity, A.tensor .* B, N, A.division)
*(B::Number, A::AbstractZ2Array{T,N}) where {T,N} = A * B
/(A::AbstractZ2Array{T,N}, B::Number) where {T,N} = Z2tensor(A.parity, A.tensor ./ B, N, A.division)
rmul!(A::AbstractZ2Array{T,N}, B::Number) where {T,N} = Z2tensor(A.parity, rmul!.(A.tensor, B), N, A.division)
similar(A::AbstractZ2Array{T,N}) where {T,N} = Z2tensor(A.parity, similar(A.tensor), N, A.division)

function mul!(Y::AbstractZ2Array{T,N}, A::AbstractZ2Array{T,N}, B::Number) where {T,N}
    Y = A*B
    Z2tensor(Y.parity, Y.tensor, N, Y.division)
end

function dot(A::AbstractZ2Array{T,N}, B::AbstractZ2Array{T,N}) where {T,N}
    exchangeind = indexin(B.parity, A.parity)
    dot(A.tensor[exchangeind], B.tensor)
end

function axpy!(α::Number, A::AbstractZ2Array{T,N}, B::AbstractZ2Array{T,N}) where {T,N}
    exchangeind = indexin(B.parity, A.parity)
    Z2tensor(B.parity, map((x,y)->axpy!(α, x, y), A.tensor[exchangeind], B.tensor), N, B.division)
end

function ≈(A::AbstractZ2Array{T,N}, B::AbstractZ2Array{T,N}) where {T,N}
    A.tensor ≈ B.tensor && A.parity == B.parity
end

function show(io::IOBuffer, A::Z2tensor)
    @show A.parity A.tensor
end

function randZ2(atype, dtype, a...)
    L = length(a)
    deven = Int.(ceil.(a./2))
    dodd = a .- deven
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

function permutedims(A::AbstractZ2Array{T,N}, perm) where {T,N}
    parity = map(x->x[collect(perm)], A.parity)
    tensor = map(x->permutedims(x, perm), A.tensor)
    Z2tensor(parity, tensor, N[collect(perm)], 1)
end

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
        deven = Int.(ceil.(N./2))
        dodd = N .- deven
        parity = A.parity
        @inbounds for i in 1:length(tensor)
            dims = Tuple(parity[i][j] == 0 ? deven[j] : dodd[j] for j in 1:length(N))
            tensor[i] = reshape(tensor[i], dims)
        end
        return Z2tensor(parity, tensor, N, div)
    end
end

function *(A::AbstractZ2Array{T,NA}, B::AbstractZ2Array{T,NB}) where {T,NA,NB}
    parity = Vector{Vector{Int}}()
    atype = _arraytype(B.tensor[1])
    tensor = Vector{atype{T}}()
    divA, divB = A.division, B.division

    adddata!(parity, tensor, A, B, 0)
    if divA != length(NA) && divB != length(NB)
        adddata!(parity, tensor, A, B, 1)
    end
    parity == [Int64[]] && return Array(tensor[1])[]
    Z2tensor(parity, tensor, (NA[1:divA]..., NB[divB+1:end]...), 1)
end

function adddata!(parity, tensor, A, B, p)
    Aparity, Atensor = A.parity, A.tensor
    Bparity, Btensor = B.parity, B.tensor
    divA, divB = A.division, B.division

    ind_A = findall(x->sum(x[divA+1:end]) % 2 == p, Aparity)
    matrix_j = unique(map(x->x[divA+1:end], Aparity[ind_A]))
    matrix_i = unique(map(x->x[1:divA], Aparity[ind_A]))
    ind_B = findall(x->x[1:divB] in matrix_j, Bparity)
    matrix_k = unique(map(x->x[divB+1:end], Bparity[ind_B]))

    h, bulkidims = [] , Int[]
    # @show matrix_i matrix_j Aparity size(Atensor)
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

    C = Amatrix * Bmatrix
    for i in 1:length(matrix_i), j in 1:length(matrix_k)
        push!(parity, [matrix_i[i]; matrix_k[j]])
        idim, jdim = sum(bulkidims[1:i-1])+1:sum(bulkidims[1:i]), sum(bulkjdims[1:j-1])+1:sum(bulkjdims[1:j])
        push!(tensor, C[idim, jdim])
    end
end

function Z2tensor2tensor(A::Z2tensor{T,N}) where {T,N}
    tensor = A.tensor
    atype = _arraytype(tensor[1])
    Tensor = atype(zeros(T, N))
    parity = A.parity
    # div = A.division
    @inbounds for l in 1:length(parity)
        @inbounds for i in CartesianIndices(tensor[l])         
            # ind = 2 .* (i.I .- 1) .+ 1 .+ (mod(sum(parity[l][1:div]),2), parity[l][div+1:end]...)
            ind = 2 .* (i.I .- 1) .+ 1 .+ parity[l]
            CUDA.@allowscalar Tensor[ind...] = tensor[l][i]
        end
    end
    Tensor
end

function transpose(A::AbstractZ2Array{T,N}) where {T,N}
    tensor = map(x->transpose(x), A.tensor)
    Z2tensor(A.parity, tensor, N, 0)
end

function tr(A::AbstractZ2Array{T,N}) where {T,N}
    parity = A.parity
    tensor = A.tensor
    half = Int(length(parity[1])/2)
    s = 0
    @inbounds for i in 1:length(parity)
        parity[i][1:half] == parity[i][half+1:end] && (s += tr(tensor[i]))
    end
    s
end

function _compactify!(y, x::AbstractZ2Array, indexer)
    x = Array(Z2tensor2tensor(x))
    @inbounds for ci in CartesianIndices(y)
        y[ci] = x[subindex(indexer, ci.I)]
    end
    return y
end

function qrpos(A::AbstractZ2Array{T,N}) where {T,N}
    tensor = A.tensor
    parity = A.parity
    div = A.division
    Qtensor = []
    Rtensor = []
    @inbounds for i in 1:length(tensor)
        Q, R = qrpos(tensor[i])
        push!(Qtensor, Q)
        push!(Rtensor, R)
    end
    atype = _arraytype(tensor[1])
    Qtensor = Vector{atype{T}}(Qtensor)
    Rtensor = Vector{atype{T}}(Rtensor)
    Nm = min(N...)
    Z2tensor(parity, Qtensor, (N[1],Nm), div), Z2tensor(parity, Rtensor, (Nm,N[2]), div)
end

function lqpos(A::AbstractZ2Array{T,N}) where {T,N}
    tensor = A.tensor
    parity = A.parity
    div = A.division
    Ltensor = []
    Qtensor = []
    @inbounds for i in 1:length(tensor)
        Q, R = lqpos(tensor[i])
        push!(Ltensor, Q)
        push!(Qtensor, R)
    end
    atype = _arraytype(tensor[1])
    Ltensor = Vector{atype{T}}(Ltensor)
    Qtensor = Vector{atype{T}}(Qtensor)
    Nm = min(N...)
    Z2tensor(parity, Ltensor, (N[1],Nm), div), Z2tensor(parity, Qtensor, (Nm,N[2]), div)
end