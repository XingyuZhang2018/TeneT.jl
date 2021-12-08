import Base: ==, +, -, *, ≈, size, reshape, permutedims
import OMEinsum: expand_unary, einsum, Permutedims, tensorpermute
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

abstract type AbstractZ2Array end
struct Z2Matrix{T1 <: AbstractMatrix, T2 <: AbstractVector} <: AbstractZ2Array
    even::T1
    odd::T1
    Nx::T2
    Ny::T2
    function Z2Matrix(even::T1, odd::T1, Nx::T2, Ny::T2) where {T1 <: AbstractMatrix, T2 <: AbstractVector}
        new{T1,T2}(even, odd, Nx, Ny)
    end
end

function size(A::Z2Matrix)
    return Tuple(collect(Iterators.flatten((A.Nx, A.Ny))))
end

function size(A::Z2Vector)
    return Tuple(collect(Iterators.flatten((A.N))))
end
struct Z2Vector{T1 <: AbstractMatrix, T2 <: AbstractVector} <: AbstractZ2Array
    even::T1
    N::T2
    function Z2Vector(even::T1, N::T2) where {T1 <: AbstractMatrix, T2 <: AbstractVector}
        new{T1,T2}(even, N)
    end
end

function *(A::Z2Matrix, B::Z2Vector)
    Z2Vector(A.even * B.even, A.Nx)
end

function *(A::Z2Vector, B::Z2Matrix)
    Z2Vector(A.even * B.even, B.Ny)
end

function *(A::Z2Matrix, B::Z2Matrix)
    Z2Matrix(A.even * B.even, A.odd * B.odd, A.Nx, B.Ny)
end

function +(A::Z2Matrix, B::Z2Matrix)
    match
    Z2Matrix(A.even + B.even, A.odd + B.odd, A.Nx, A.Ny)
end

function -(A::Z2Matrix, B::Z2Matrix)
    Z2Matrix(A.even - B.even, A.odd - B.odd, A.Nx, A.Ny)
end

function ≈(A::Z2Matrix, B::Z2Matrix)
    A.even ≈ B.even && A.odd ≈ B.odd
end

function ≈(A::Z2Vector, B::Z2Vector)
    A.even ≈ B.even
end

function sitetoZ2site(a::Vector,b::Vector,Na::Vector,Nb::Vector)
    s1, s2 = sum(a), sum(b) 
    l1, l2 = length(a), length(b)
    x, y = 0, 0
    (s1 + s2) % 2 != 0 && throw(Base.error("$a $b is not in the parity conserving subspace"))
    s1 % 2 == 0 ? parity = :even : parity = :odd
    if l1 != 0
        for i in 1:(l1-1)
            x += a[i] * Int(ceil(Na[i]/2)) 
        end
        x += Int(floor(a[l1]/2)) + 1
    else
        x = 1
    end
    if l2 != 0
        for i in 1:(l2-1)
            y += b[i] * Int(ceil(Nb[i]/2))
        end
        y += Int(floor(b[l2]/2)) + 1
    else
        y = 1
    end
    return (parity, x, y)
end

function permutedims(A::AbstractZ2Array, b::AbstractArray)
    typeof(A) <: Z2Matrix ? Na = [A.Nx, A.Ny] : Na = [A.N, []]
    iter = Tuple(collect(Iterators.flatten(Na)))
    La = length(Na[1])
    Nb = similar(b)
    Nb[1] = collect(iter[b[1]])
    Nb[2] = collect(iter[b[2]])
    if Nb[1] == []
        Nb[1] = [1]
        B = Z2Vector(zeros(Int(ceil(prod(Nb[1])/2)),Int(ceil(prod(Nb[2])/2))), Nb[2])
    elseif Nb[2] == []
        Nb[2] = [1]
        B = Z2Vector(zeros(Int(ceil(prod(Nb[1])/2)),Int(ceil(prod(Nb[2])/2))), Nb[1])
    else
        B = Z2Matrix(zeros(Int(ceil(prod(Nb[1])/2)),Int(ceil(prod(Nb[2])/2))), 
                     zeros(Int(ceil(prod(Nb[1])/2)),Int(ceil(prod(Nb[2])/2))), Nb[1], Nb[2])
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
    typeof(A) <: Z2Matrix ? N = [A.Nx, A.Ny] : N = [A.N, []]
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
    typeof(A) <: Z2Matrix ? (prod(A.Nx) == a && prod(A.Ny) == b && return A) : (prod(A.N) == a && return A)
    l = 1
    p = s[1]
    while p != a[1]
        l += 1
        p *= s[l]
    end
    permutedims(A, [collect(1:l), collect(l+1:L)])
end

function expand_unary(ix::Vector{T}, iy::Vector{T}, x::AbstractZ2Array, size_dict::Dict{T}) where T 
    iy_b = unique(iy)
    iy_a = filter(i->i ∈ ix, iy_b)
    y_a = if ix != iy_a
        einsum(Permutedims(), (ix,), iy_a, (x,), size_dict)
    else
        x
    end
    # repeat
    y_b = length(iy_a) != length(iy_b) ? einsum(Repeat(), (iy_a,), iy_b, (y_a,), size_dict) : y_a
    # duplicate
    length(iy_b) != length(iy) ? einsum(Duplicate(), (iy_b,), iy, (y_b,), size_dict) : y_b
end

function einsum(::Permutedims, ixs, iy, xs::Tuple{<:AbstractZ2Array}, size_dict)
    ix, x = ixs[1], xs[1]
    perm = ntuple(i -> findfirst(==(iy[i]), ix)::Int, length(iy))
    return tensorpermute(x, perm)
end

tensorpermute(A::AbstractZ2Array, perm) = length(perm) == 0 ? copy(A) : permutedims(A, [collect(perm[1:length(A.Nx)]), collect(perm[length(A.Nx)+1:end])])