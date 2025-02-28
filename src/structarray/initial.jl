"""
    randSA([T=ComplexF64,] atype, pattern, sizes)

    randSA([T=ComplexF64,] S::StructArray, [sizes])



Randomly initialize a StructArray.

Arguments:
- `T`: Optional, type of random numbers, defaults to ComplexF64
- `atype`: Array type (e.g., Array, CuArray, etc.)
- `pattern`: Pattern matrix
- `sizes`: Vector of tuples specifying sizes for each array
- `S`: Existing StructArray to copy pattern and array type from

Examples:
```julia
randSA(Array, [1 2; 2 1], [(2,2), (3,3)])  # Using default ComplexF64 type
randSA(Float32, Array, [1 2; 2 1], [(2,2), (3,3)])  # Specifying Float32 type
```
"""
function rand(::Type{T}, spaces::Vector, pattern::Matrix{Int}) where {T<:Number}
    data = [rand(T, s) for s in spaces]
    return StructArray(data, pattern)
end

function rand(::Type{T}, pattern::Matrix{Int}) where {T<:Number}
    data = rand(T, length(unique(pattern)))
    return StructArray(data, pattern)
end

function rand!(S::StructArray{<:AbstractVector{<:Number}})
    return StructArray(rand!(S.data), S.pattern)
end

function rand!(S::StructArray)
    return StructArray(rand!.(S.data), S.pattern)
end