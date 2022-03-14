#helper functions to handle array types
_mattype(::Array{T}) where {T} = Matrix
_mattype(::CuArray{T}) where {T} = CuMatrix
_mattype(::Adjoint{T, CuArray{T, 2 ,B}}) where {T,B} = CuMatrix
_mattype(::Symmetric{T, CuArray{T, 2, B}}) where {T,B} = CuMatrix

_arraytype(::Array) = Array
_arraytype(::CuArray) = CuArray
_arraytype(::Diagonal{T, Vector{T}}) where {T} = Array
_arraytype(::Diagonal{T, CuArray{T, 1, B}}) where {T, B} = CuArray
_arraytype(::Z2Array{T}) where {T} = Z2Array
_arraytype(::U1Array{T}) where {T} = U1Array
_arraytype(::Adjoint{T, Matrix{T}}) where {T} = Matrix
_arraytype(::Adjoint{T, CuArray{T, 2, B}}) where {T,B} = CuArray
_arraytype(::Transpose{T, Matrix{T}}) where {T} = Matrix
_arraytype(::Transpose{T, CuArray{T, 2, B}}) where {T,B} = CuArray

getsymmetry(::AbstractArray) = :none
getsymmetry(::Z2Array) = :Z2
getsymmetry(::U1Array) = :U1

randinitial(::Val{:none}, atype, dtype, a...; dir = :none) = atype(rand(dtype, a...))
randinitial(::Val{:Z2}, atype, dtype, a...; dir = :none) = randZ2(atype, dtype, a...)
randinitial(::Val{:U1}, atype, dtype, a...; dir) = randU1(atype, dtype, a...; dir = dir)

function randinitial(A::AbstractArray{T, N}, a...; dir) where {T, N}
    atype = typeof(A) <: Union{Array, CuArray} ? _arraytype(A) : _arraytype(A.tensor[1])
    randinitial(Val(getsymmetry(A)), atype, T, a...; dir = dir)
end

Iinitial(::Val{:none}, atype, dtype, D; dir = :none) = atype{dtype}(I, D, D)
Iinitial(::Val{:Z2}, atype, dtype, D; dir = :none) = IZ2(atype, dtype, D)
Iinitial(::Val{:U1}, atype, dtype, D; dir) = IU1(atype, dtype, D; dir = dir)

function Iinitial(A::AbstractArray{T, N}, D; dir) where {T, N}
    atype = typeof(A) <: Union{Array, CuArray} ? _arraytype(A) : _arraytype(A.tensor[1])
    Iinitial(Val(getsymmetry(A)), atype, ComplexF64, D; dir = dir)
end

zerosinitial(::Val{:none}, atype, dtype, a...; dir = :none) = atype(zeros(dtype, a...))
zerosinitial(::Val{:Z2}, atype, dtype, a...; dir = :none) = zerosZ2(atype, dtype, a...)
zerosinitial(::Val{:U1}, atype, dtype, a...; dir) = zerosU1(atype, dtype, a...; dir = dir)

function zerosinitial(A::AbstractArray{T, N}, a...; dir) where {T, N}
    atype = typeof(A) <: Union{Array, CuArray} ? _arraytype(A) : _arraytype(A.tensor[1])
    zerosinitial(Val(getsymmetry(A)), atype, T, a...; dir = dir)
end

asArray(A::AbstractArray) = A