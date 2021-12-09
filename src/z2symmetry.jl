import Base: ==, +, -, *, ≈, size, reshape, permutedims, transpose
import LinearAlgebra: tr
import OMEinsum: tensorpermute, _compactify!, einsum, Diag
using BitBasis

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
        T, N = eltype(even), Tuple(collect(Iterators.flatten((Ni, Nj))))
        new{T, N, T1,T2}(even, odd, Ni, Nj)
    end
end
struct Z2Vector{T, N, T1 <: AbstractMatrix, T2 <: AbstractVector} <: AbstractZ2Array{T,N}
    even::T1
    Ni::T2
    function Z2Vector(even::T1, Ni::T2) where {T1 <: AbstractMatrix, T2 <: AbstractVector}
        T, N = eltype(even), Tuple(collect(Iterators.flatten(Ni)))
        new{T, N, T1,T2}(even, Ni)
    end
end

size(A::Z2Matrix) = Tuple(collect(Iterators.flatten((A.Ni, A.Nj))))
size(A::Z2Vector) = A.Ni

*(A::Z2Matrix, B::Z2Vector) = Z2Vector(A.even * B.even, A.Ni)
*(A::Z2Vector, B::Z2Matrix) = Z2Vector(A.even * B.even, B.Nj)
*(A::Z2Vector, B::Z2Vector) = (A.even * B.even)[1]
*(A::Z2Matrix, B::Z2Matrix) = Z2Matrix(A.even * B.even, A.odd * B.odd, A.Ni, B.Nj)
+(A::Z2Matrix, B::Z2Matrix) = Z2Matrix(A.even + B.even, A.odd + B.odd, A.Ni, A.Nj)
-(A::Z2Matrix, B::Z2Matrix) = Z2Matrix(A.even - B.even, A.odd - B.odd, A.Ni, A.Nj)
≈(A::Z2Matrix, B::Z2Matrix) = (A.even ≈ B.even && A.odd ≈ B.odd)
≈(A::Z2Vector, B::Z2Vector) = A.even ≈ B.even

function maxbulkMatrix(Ni::Vector,Nj::Vector)
    l1, l2 = length(Ni), length(Nj)
    ieven, iodd = 1, 1
    jeven, jodd = 1, 1
    Ci = Int.(ceil.(LinearIndices(Tuple(collect((0:i) for i in Ni.-1)))/2))
    Cj = Int.(ceil.(LinearIndices(Tuple(collect((0:j) for j in Nj.-1)))/2))
    if l1 != 0 && l2 != 0
        i1 = Ci[Ni...]
        Nir = copy(Ni)
        Nir[1] -= 1
        i2 = Ci[Nir...]
        j1 = Cj[Nj...]
        Njr = copy(Nj)
        Njr[1] -= 1
        j2 = Cj[Njr...]
        sum(Ni .- 1) % 2 == 0 ? (ieven = i1; iodd = i2) : (ieven = i2; iodd = i1)
        sum(Nj .- 1) % 2 == 0 ? (jeven = j1; jodd = j2) : (jeven = j2; jodd = j1)
        return (ieven, jeven), (iodd, jodd)
    elseif l2 == 0
        i1 = Ci[Ni...]
        Nir = copy(Ni)
        Nir[1] -= 1
        i2 = Ci[Nir...]
        sum(Ni .- 1) % 2 == 0 ? (ieven = i1) : (ieven = i2)
        return (ieven, jeven)
    else
        j1 = Cj[Nj...]
        Njr = copy(Nj)
        Njr[1] -= 1
        j2 = Cj[Njr...]
        sum(Nj .- 1) % 2 == 0 ? (jeven = j1) : (jeven = j2)
        return (ieven, jeven)
    end
end

function sitetoZ2site(a::Vector,b::Vector,Ni::Vector,Nj::Vector)
    s1, s2 = sum(a), sum(b) 
    l1, l2 = length(a), length(b)
    (s1 + s2) % 2 != 0 && throw(Base.error("$a $b is not in the parity conserving subspace"))
    s1 % 2 == 0 ? parity = :even : parity = :odd
    if l1 != 0
        Ci = Int.(ceil.(LinearIndices(Tuple(collect((0:k) for k in Ni.-1)))/2))
        i = Ci[a.+1...]
    else
        i = 1
    end
    if l2 != 0
        Cj = Int.(ceil.(LinearIndices(Tuple(collect((0:k) for k in Nj.-1)))/2))
        j = Cj[b.+1...]
    else
        j = 1
    end
    return (parity, i, j)
end

function permutedims(A::AbstractZ2Array, b::AbstractArray)
    typeof(A) <: Z2Matrix ? Na = [A.Ni, A.Nj] : Na = [A.Ni, []]
    iter = Tuple(collect(Iterators.flatten(Na)))
    La = length(Na[1])
    Nb = similar(b)
    Nb[1] = collect(iter[b[1]])
    Nb[2] = collect(iter[b[2]])
    s = maxbulkMatrix(Nb[1],Nb[2])
    if Nb[1] == []
        B = Z2Vector(zeros(s), Nb[2])
    elseif Nb[2] == []
        B = Z2Vector(zeros(s), Nb[1])
    else
        B = Z2Matrix(zeros(s[1]), zeros(s[2]), Nb[1], Nb[2])
    end
    for i in CartesianIndices(iter)
        ind = collect(Tuple(i)) .- 1
		if sum(ind) % 2 == 0
            sa = sitetoZ2site(ind[1:La],ind[La+1:end],Na[1],Na[2])
            sb = sitetoZ2site(ind[b[1]],ind[b[2]],Nb[1],Nb[2])
			getfield(B,sb[1])[sb[2],sb[3]] = getfield(A,sa[1])[sa[2],sa[3]]
		end
	end
    return B
end

function Z2Matrix2tensor(A::AbstractZ2Array)
    typeof(A) <: Z2Matrix ? N = [A.Ni, A.Nj] : N = [A.Ni, []]
    L = length(N[1])
    iter = Tuple(collect(Iterators.flatten(N)))
    T = zeros(iter)
    for i in CartesianIndices(iter)
        ind = collect(Tuple(i)) .- 1
        if sum(ind) % 2 == 0
            sa = sitetoZ2site(ind[1:L],ind[L+1:end],N[1],N[2])
            T[i] = getfield(A,sa[1])[sa[2],sa[3]]
        end
    end
    return T
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