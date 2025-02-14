"""
A StructArray is a wrapper around an array of arrays.
Its data is stored in a vector of arrays and can be accessed using matrix-like indexing.


For example, given a matrix `M` of arrays with the structure:
```
M = [A B;
     B A]
```
where M[1,1] = M[2,2] and M[1,2] = M[2,1], we can store the data efficiently in a vector:
```
V = [A, B]
```
and create a StructArray:
```
S = StructArray(V, [1 2; 2 1])
```
The StructArray maintains the original matrix-like access pattern:
```
S[1,1] = A
S[1,2] = B
S[2,1] = B
S[2,2] = A
```
"""
struct StructArray{A}
    data::A # Stores the actual array data
    pattern::Matrix{Int}  # Pattern matrix storing indices into data vector
    function StructArray(data, pattern)
        @assert length(data) == length(unique(pattern)) "the number of data is not equal to the number of unique elements in pattern"
        # @assert pattern[1,1] == 1 "the first element must be 1"
        new{typeof(data)}(data, pattern)
    end
end

# 实现必要的AbstractArray接口
Base.size(S::StructArray) = size(S.pattern)
Base.size(S::StructArray, i::Int) = size(S.pattern, i)
Base.getindex(S::StructArray, i::Int, j::Int) = S.data[S.pattern[i, j]]
Base.getindex(S::StructArray, i::Int, ::Colon) = S.data[S.pattern[i, :]]
Base.getindex(S::StructArray, ::Colon, j::Int) = S.data[S.pattern[:, j]]
Base.getindex(S::StructArray, i::Int) = S.data[S.pattern[i]]
Base.setindex!(S::StructArray, value, i::Int, j::Int) = (S.data[S.pattern[i, j]] = value)
Base.setindex!(S::StructArray, value, i::Int) = (S.data[S.pattern[i]] = value)
Base.vec(S::StructArray) = S
Base.similar(S::StructArray) = StructArray(similar(S.data), S.pattern)
Base.Array(S::StructArray) = StructArray(Array.(S.data), S.pattern)
Base.copy(S::StructArray) = StructArray(copy(S.data), S.pattern)
CUDA.CuArray(S::StructArray) = StructArray(CUDA.CuArray.(S.data), S.pattern)
LinearAlgebra.norm(S::StructArray) = norm(S.data)
LinearAlgebra.conj(S::StructArray) = StructArray(conj(S.data), S.pattern)
Base.isapprox(S::StructArray, T::StructArray; atol=1e-12) = ==(S.pattern, T.pattern) && isapprox(S.data, T.data; atol=atol)
function Base.show(io::IO, S::StructArray)
    println(io, "StructArray with pattern:")
    show(io, S.pattern)

    println(io, "\nand data:")
    show(io, S.data)
end

function Base.iterate(S::StructArray, state=1)
    unique_patterns = sort!(unique(S.pattern))
    state > length(unique_patterns) && return nothing
    return S.data[unique_patterns[state]], state + 1
end

function Base.collect(S::StructArray) 
    unique_patterns = unique(S.pattern)
    return [S.data[p] for p in unique_patterns]
end

function Base.circshift(S::StructArray, shift::NTuple{N,Int}) where N
    return StructArray(S.data, circshift(S.pattern, shift))
end