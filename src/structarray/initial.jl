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
function randSA(::Type{T}, atype, pattern::Matrix{Int}, sizes::Vector{NTuple{N,Int}}) where {T<:Number, N}
    data = [atype(rand(T, s...)) for s in sizes]
    return StructArray(data, pattern)
end

function randSA(::Type{T}, atype, pattern::Matrix{Int}) where {T<:Number}
    data = atype(rand(T, length(unique(pattern))))
    return StructArray(data, pattern)
end

# Method with default type
function randSA(atype, pattern::Matrix{Int}, sizes::Vector{NTuple{N,Int}}) where N
    return randSA(ComplexF64, atype, pattern, sizes)
end

function randSA(atype, pattern::Matrix{Int})
    return randSA(ComplexF64, atype, pattern)
end

function randSA(S::StructArray, pattern::Matrix{Int}, sizes::Vector{NTuple{N,Int}}) where N
    atype = _arraytype(S[1])
    T = eltype(S[1])
    return randSA(T, atype, pattern, sizes)
end

function randSA(S::StructArray, sizes::Vector{NTuple{N,Int}}) where N
    atype = _arraytype(S[1])
    T = eltype(S[1])
    return randSA(T, atype, S.pattern, sizes)
end

# Methods based on existing StructArray
function randSA(S::StructArray)
    atype = _arraytype(S[1])
    T = eltype(S[1])
    return randSA(T, atype, S.pattern, size.(S.data))
end

function ISA(::Type{T}, atype, pattern::Matrix{Int}, sizes::Vector{NTuple{N,Int}}) where {T<:Number, N}
    data = [atype{T}(I, s...) for s in sizes]
    return StructArray(data, pattern)
end

function ISA(S::StructArray, sizes::Vector{NTuple{N,Int}}) where N
    atype = _arraytype(S[1])
    pattern = S.pattern
    T = eltype(S[1])
    return ISA(T, atype, pattern, sizes)
end