import Base: ==, +, -, *, ≈, size, reshape, permutedims, transpose, conj, show
import LinearAlgebra: tr
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
    parity = []
    tensor = []
    for i in CartesianIndices(Tuple(0:1 for i=1:L))
        if sum(i.I) % 2 == 0
            push!(parity,collect(i.I))
            dims = Tuple(i.I[j] == 0 ? deven[j] : dodd[j] for j in 1:L)
            push!(tensor,atype(rand(dtype, dims)))
        end
    end
    parity = Vector{Vector{Int}}(parity)
    tensor = Vector{atype{dtype}}(tensor)
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
        for i in 1:length(tensor)
            s = size(tensor[i])
            tensor[i] = reshape(tensor[i], prod(s[1:div]), prod(s[div+1:end]))
        end
        return Z2tensor(A.parity, tensor, a, div)
    else
        deven = Int.(ceil.(a./2))
        dodd = a .- deven
        parity = A.parity
        for i in 1:length(tensor)
            dims = Tuple(parity[i][j] == 0 ? deven[j] : dodd[j] for j in 1:length(a))
            tensor[i] = reshape(tensor[i], dims)
        end
        return Z2tensor(parity, tensor, a, div)
    end
end

function *(A::AbstractZ2Array{T,NA}, B::AbstractZ2Array{T,NB}) where {T,NA,NB}
    Aparity, Atensor = A.parity, A.tensor
    Bparity, Btensor = B.parity, B.tensor
    LA, LB = length(Aparity), length(Bparity)
    divA, divB = A.division, B.division
    parity, tensor = [], []
    for i in 1:LA, j in 1:LB
        if Aparity[i][divA+1:end] == Bparity[j][1:divB] 
            push!(parity, [Aparity[i][1:divA];Bparity[j][divB+1:end]])
            push!(tensor, Atensor[i] * Btensor[j])
        end
    end
    parity, tensor = uniquetensors(parity,tensor)
    # @show parity, tensor
    parity == [Int64[]] && return Array(tensor[1])[]
    atype = _arraytype(tensor[1])
    parity = Vector{Vector{Int}}(parity)
    tensor = Vector{atype{T}}(tensor)
    Z2tensor(parity, tensor, (NA[1:divA]..., NB[divB+1:end]...), 1)
end

function uniquetensors(parity,tensor)
    uparity = unique(parity)
    utensor = []
    for upy in uparity
        i = findall(x->x in [upy], parity)
        push!(utensor, sum(tensor[i]))
    end
    uparity, utensor
end

function Z2tensor2tensor(A::Z2tensor{T,N}) where {T,N}
    tensor = A.tensor
    atype = _arraytype(tensor[1])
    Tensor = atype(zeros(T, N))
    parity = A.parity
    # div = A.division
    for l in 1:length(parity)
        for i in CartesianIndices(tensor[l])
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
    for i in 1:length(parity)
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
    for i in 1:length(tensor)
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
    for i in 1:length(tensor)
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