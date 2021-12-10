import Base: ==, +, -, *, ≈, size, reshape, permutedims, transpose, display
import LinearAlgebra: tr
import OMEinsum: tensorpermute, _compactify!, einsum, Diag
import Random: rand
using BitBasis
using CUDA

"""
    parity_conserving(T::Array)

Transform an arbitray tensor which has arbitray legs and each leg have index 1 or 2 into parity conserving form

"""
function parity_conserving(T::Union{Array,CuArray}) where V<:Real
	s = size(T)
	p = zeros(s)
	bits = map(x -> Int(ceil(log2(x))), s)
	for index in CartesianIndices(T)
		i = Tuple(index) .- 1
		sum(sum.(bitarray.(i,bits))) % 2 == 0 && (p[index] = 1)
	end
	p = _arraytype(T)(p)

	return reshape(p.*T,s...)
end

abstract type AbstractZ2Array{T,N} <: AbstractArray{T,N} end
struct Z2Matrix{T, N, T1 <: AbstractMatrix, T2 <: AbstractVector} <: AbstractZ2Array{T,N}
    even::T1
    odd::T1
    Ni::T2
    Nj::T2
    function Z2Matrix(even::T1, odd::T1, Ni::T2, Nj::T2) where {T1 <: AbstractMatrix, T2 <: AbstractVector}
        T, N = eltype(even), Tuple([Ni; Nj])
        new{T, N, T1, T2}(even, odd, Ni, Nj)
    end
end
struct Z2Vector{T, N, T1 <: AbstractMatrix, T2 <: AbstractVector} <: AbstractZ2Array{T,N}
    even::T1
    Ni::T2
    function Z2Vector(even::T1, Ni::T2) where {T1 <: AbstractMatrix, T2 <: AbstractVector}
        T, N = eltype(even), Tuple(Ni)
        new{T, N, T1,T2}(even, Ni)
    end
end

size(::AbstractZ2Array{T,N}) where {T,N} = N
function display(A::Z2Matrix)
    @show A.even
    @show A.odd
end

*(A::Z2Matrix, B::Z2Vector) = Z2Vector(A.even * B.even, A.Ni)
*(A::Z2Vector, B::Z2Matrix) = Z2Vector(A.even * B.even, B.Nj)
*(A::Z2Vector, B::Z2Vector) = CUDA.@allowscalar (A.even * B.even)[1]
*(A::Z2Matrix, B::Z2Matrix) = Z2Matrix(A.even * B.even, A.odd * B.odd, A.Ni, B.Nj)
+(A::Z2Matrix, B::Z2Matrix) = Z2Matrix(A.even + B.even, A.odd + B.odd, A.Ni, A.Nj)
-(A::Z2Matrix, B::Z2Matrix) = Z2Matrix(A.even - B.even, A.odd - B.odd, A.Ni, A.Nj)
≈(A::Z2Matrix, B::Z2Matrix) = (A.even ≈ B.even && A.odd ≈ B.odd)
≈(A::Z2Vector, B::Z2Vector) = A.even ≈ B.even

function bulksize(Ni,Nj)
    l1, l2 = length(Ni), length(Nj)
    if l2 == 0
        ieven = Int(ceil(prod(Ni)/2))
        return (ieven, 1)
    elseif l1 == 0
        jeven = Int(ceil(prod(Nj)/2))
        return (1, jeven)
    else
        ieven = Int(ceil(prod(Ni)/2))
        iodd = prod(Ni) - ieven
        jeven = Int(ceil(prod(Nj)/2))
        jodd = prod(Nj) - jeven
        return (ieven, jeven), (iodd, jodd)
    end
end

function randZ2(atype, dtype, a...)
    a = collect(Iterators.flatten(a))
    La = Int(ceil(length(a)/2))
    N = [a[1:La], a[La+1:end]]
    s = bulksize(N[1],N[2])
    Z2Matrix(atype(rand(dtype, s[1][1], s[1][2])), atype(rand(dtype, s[2][1], s[2][2])), N[1], N[2])
end

maptable(N::Vector) = Int.(ceil.(LinearIndices(Tuple(collect((0:k) for k in N.-1)))/2))

function indextoZ2index(a::Vector,b::Vector,Ci,Cj)
    s1, s2 = sum(a), sum(b) 
    l1, l2 = length(a), length(b)
    (s1 + s2) % 2 != 0 && throw(Base.error("$a $b is not in the parity conserving subspace"))
    s1 % 2 == 0 ? parity = :even : parity = :odd
    l1 == 0 ? (i = 1) : (i = Ci[a.+1...])
    l2 == 0 ? (j = 1) : (j = Cj[b.+1...])
    return (parity, i, j)
end

function permutedims(A::AbstractZ2Array{T}, b::AbstractArray) where {T}
    typeof(A) <: Z2Matrix ? Na = [A.Ni, A.Nj] : Na = [A.Ni, []]
    iter = Tuple(collect(Iterators.flatten(Na)))
    La = length(Na[1])
    Nb = similar(b)
    Nb[1] = collect(iter[b[1]])
    Nb[2] = collect(iter[b[2]])
    s = bulksize(Nb[1],Nb[2])
    Cbi, Cbj = [], [] 
    atype = _arraytype(A.even)
    if Nb[1] == []
        Cbj = maptable(Nb[2])
        B = Z2Vector(atype(zeros(T,s)), Nb[2])
    elseif Nb[2] == []
        Cbi = maptable(Nb[1])
        B = Z2Vector(atype(zeros(T,s)), Nb[1])
    else
        Cbi = maptable(Nb[1])
        Cbj = maptable(Nb[2])
        B = Z2Matrix(atype(zeros(T,s[1])), atype(zeros(T,s[2])), Nb[1], Nb[2])
    end
    Cai = maptable(Na[1])
    Caj = maptable(Na[2])
    for i in CartesianIndices(iter)
        ind = collect(Tuple(i)) .- 1
		if sum(ind) % 2 == 0
            sa = indextoZ2index(ind[1:La],ind[La+1:end],Cai,Caj)
            sb = indextoZ2index(ind[b[1]],ind[b[2]],Cbi,Cbj)
			CUDA.@allowscalar getfield(B,sb[1])[sb[2],sb[3]] = getfield(A,sa[1])[sa[2],sa[3]]
		end
	end
    return B
end

function Z2Matrix2tensor(A::AbstractZ2Array{T}) where {T}
    typeof(A) <: Z2Matrix ? N = [A.Ni, A.Nj] : N = [A.Ni, []]
    L = length(N[1])
    iter = Tuple(collect(Iterators.flatten(N)))
    Tensor = _arraytype(A.even)(zeros(T, iter))
    Ci = maptable(N[1])
    Cj = maptable(N[2])
    for i in CartesianIndices(iter)
        ind = collect(Tuple(i)) .- 1
        if sum(ind) % 2 == 0
            sa = indextoZ2index(ind[1:L],ind[L+1:end],Ci,Cj)
            CUDA.@allowscalar Tensor[i] = getfield(A,sa[1])[sa[2],sa[3]]
        end
    end
    return Tensor
end 

function reshape(A::AbstractZ2Array, a::Int...)
    s = size(A)
    L = length(s)
    typeof(A) <: Z2Matrix ? (prod(A.Ni) == a[1] && prod(A.Nj) == a[2] && return A) : (prod(A.Ni) == a && return A)
    l = 1
    p = s[1]
    while p != a[1]
        l += 1
        p *= s[l]
    end
    permutedims(A, [collect(1:l), collect(l+1:L)])
end

transpose(A::Z2Vector) = Z2Vector(transpose(A.even), A.Ni)
tr(A::Z2Matrix) = tr(A.even) + tr(A.odd)

tensorpermute(A::AbstractZ2Array, perm) = length(perm) == 0 ? copy(A) : permutedims(A, [collect(perm[1:length(A.Ni)]), collect(perm[length(A.Ni)+1:end])])

function einsum(::Diag, ixs, iy, xs::Tuple{<:AbstractZ2Array}, size_dict::Dict)
    ix, x = ixs[1], xs[1]
    @debug "Diag" ix => iy size.(x)
    x_in_y_locs = (Int[findfirst(==(x), iy) for x in ix]...,)
    if x_in_y_locs[1] == x_in_y_locs[2]
        x = permutedims(x, [[1,3], [2,4]])
    end
    reshape([diag(x.even); diag(x.odd)],x.Ni[1],x.Ni[1])
end

function qrpos(A::AbstractZ2Array)
    Qeven,Reven = qrpos(A.even)
    Qodd,Rodd = qrpos(A.odd)
    prod(A.Ni) <= prod(A.Nj) ? (Nm = A.Ni) : (Nm = A.Nj)
    Q = Z2Matrix(Qeven, Qodd, A.Ni, Nm)
    R = Z2Matrix(Reven, Rodd, Nm, A.Nj)
    return Q, R
end   

function lqpos(A::AbstractZ2Array)
    Leven,Qeven = lqpos(A.even)
    Lodd,Qodd = lqpos(A.odd)
    prod(A.Ni) <= prod(A.Nj) ? (Nm = A.Ni) : (Nm = A.Nj)
    L = Z2Matrix(Leven, Lodd, A.Ni, Nm)
    Q = Z2Matrix(Qeven, Qodd, Nm, A.Nj)
    return L, Q
end