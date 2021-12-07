import Base: ==, +, -, *, reshape
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

struct Z2Matrix{T <: AbstractMatrix}
    even::T
    odd::T
    function Z2Matrix(even::T, odd::T) where {T <: AbstractMatrix}
        new{T}(even, odd)
    end
end

function *(A::Z2Matrix, B::Z2Matrix)
    Z2Matrix(A.even * B.even, A.odd * B.odd)
end

function +(A::Z2Matrix, B::Z2Matrix)
    Z2Matrix(A.even + B.even, A.odd + B.odd)
end

function -(A::Z2Matrix, B::Z2Matrix)
    Z2Matrix(A.even - B.even, A.odd - B.odd)
end

function sitetoZ2site(a::Vector{Int},b::Vector{Int},Na::Vector{Int},Nb::Vector{Int})
    s1, s2 = sum(a), sum(b) 
    l1, l2 = length(a), length(b)
    x, y = 0, 0
    (s1 + s2) % 2 != 0 && throw(Base.error("$a $b is not in the parity conserving subspace"))
    s1 % 2 == 0 ? parity = :even : parity = :odd
    for i in 1:(l1-1)
        x += a[i] * Int(ceil(Na[i]/2)) 
    end
    x += Int(floor(a[l1]/2)) + 1
    for i in 1:(l2-1)
        y += b[i] * Int(ceil(Nb[i]/2))
    end
    y += Int(floor(b[l2]/2)) + 1
    return (parity, x, y)
end

function reshape(A::Z2Matrix, Na::AbstractArray, Nb::AbstractArray)
    B = Z2Matrix(zeros(Int(prod(Nb[1])/2),Int(prod(Nb[2])/2)), 
                  zeros(Int(prod(Nb[1])/2),Int(prod(Nb[2])/2)))
    iter = Tuple(collect(Iterators.flatten(Na)))
    La,Lb = map(length,[Na[1],Nb[1]])
    for i in CartesianIndices(iter)
        ind = collect(Tuple(i)) .- 1
		if sum(ind) % 2 == 0
            sa = sitetoZ2site(ind[1:La],ind[La+1:end],Na[1],Na[2])
            sb = sitetoZ2site(ind[1:Lb],ind[Lb+1:end],Nb[1],Nb[2])
			getfield(B,sb[1])[sb[2],sb[3]] = getfield(A,sa[1])[sa[2],sa[3]]
		end
	end
    return B
end


