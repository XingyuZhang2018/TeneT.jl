#helper functions to handle array types
_mattype(::Array{T}) where {T} = Matrix
_mattype(::CuArray{T}) where {T} = CuMatrix
_mattype(::Adjoint{T, CuArray{T, 2 ,B}}) where {T,B} = CuMatrix
_mattype(::Symmetric{T, CuArray{T, 2, B}}) where {T,B} = CuMatrix

_arraytype(::Array) = Array
_arraytype(::CuArray) = CuArray
_arraytype(::Diagonal{T, Vector{T}}) where {T} = Array
_arraytype(::Diagonal{T, CuArray{T, 1, B}}) where {T, B} = CuArray
_arraytype(::Z2tensor{T}) where {T} = Z2tensor
_arraytype(::Adjoint{T, Matrix{T}}) where {T} = Matrix
_arraytype(::Adjoint{T, CuArray{T, 2, B}}) where {T,B} = CuArray
_arraytype(::Transpose{T, Matrix{T}}) where {T} = Matrix
_arraytype(::Transpose{T, CuArray{T, 2, B}}) where {T,B} = CuArray

getsymmetry(::AbstractArray) = :none
getsymmetry(::AbstractZ2Array) = :Z2
getsymmetry(::AbstractU1Array) = :U1

randinitial(::Val{:none}, atype, dtype, a...; dir) = atype(rand(dtype, a...))
randinitial(::Val{:Z2}, atype, dtype, a...; dir) = randZ2(atype, dtype, a...)
randinitial(::Val{:U1}, atype, dtype, a...; dir) = randU1(atype, dtype, a...; dir = dir)

function randinitial(A::AbstractArray{T, N}, a...; dir) where {T, N}
    atype = typeof(A) <: Union{Array, CuArray} ? _arraytype(A) : _arraytype(A.tensor[1])
    randinitial(Val(getsymmetry(A)), atype, T, a...; dir = dir)
end

Iinitial(::Val{:none}, atype, dtype, D) = atype{dtype}(I, D, D)
Iinitial(::Val{:Z2}, atype, dtype, D) = IZ2(atype, dtype, D)
Iinitial(::Val{:U1}, atype, dtype, D; dir) = IU1(atype, dtype, D; dir = dir)

function Iinitial(A::AbstractArray{T, N}, D; dir) where {T, N}
    atype = typeof(A) <: Union{Array, CuArray} ? _arraytype(A) : _arraytype(A.tensor[1])
    Iinitial(Val(getsymmetry(A)), atype, ComplexF64, D; dir = dir)
end

zerosinitial(::Val{:none}, atype, dtype, a...) = atype(zeros(dtype, a...))
zerosinitial(::Val{:Z2}, atype, dtype, a...) = zerosZ2(atype, dtype, a...)
zerosinitial(::Val{:U1}, atype, dtype, a...; dir) = zerosU1(atype, dtype, a...; dir = dir)

function zerosinitial(A::AbstractArray{T, N}, a...; dir) where {T, N}
    atype = typeof(A) <: Union{Array, CuArray} ? _arraytype(A) : _arraytype(A.tensor[1])
    zerosinitial(Val(getsymmetry(A)), atype, T, a...; dir = dir)
end