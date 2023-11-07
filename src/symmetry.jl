export asArray, asSymmetryArray, symmetryreshape, getsymmetry, getdir
export _mattype, _arraytype
export randinitial, zerosinitial

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
_arraytype(::Transpose{T, Vector{T}}) where {T} = Vector
_arraytype(::Transpose{T, Matrix{T}}) where {T} = Matrix
_arraytype(::Transpose{T, CuArray{T, N, B}}) where {T,N,B} = CuArray

getsymmetry(::AbstractArray) = :none
getsymmetry(::Z2Array) = :Z2
getsymmetry(::U1Array) = :U1

getdir(::AbstractArray) = nothing
getdir(::Z2Array) = nothing

randinitial(::Val{:none}, atype, dtype, a...; kwarg...) = atype(rand(dtype, a...))
randinitial(::Val{:Z2}, atype, dtype, a...; kwarg...) = randZ2(atype, dtype, a...; kwarg...)
randinitial(::Val{:U1}, atype, dtype, a...; kwarg...) = randU1(atype, dtype, a...; kwarg...)

function randinitial(A::AbstractArray{T, N}, a...; kwarg...) where {T, N}
    atype = typeof(A) <: Union{Array, CuArray} ? _arraytype(A) : _arraytype(A.tensor)
    randinitial(Val(getsymmetry(A)), atype, T, a...; kwarg...)
end

Iinitial(::Val{:none}, atype, dtype, D; kwarg...) = atype{dtype}(I, D, D)
Iinitial(::Val{:Z2}, atype, dtype, D; kwarg...) = IZ2(atype, dtype, D; kwarg...)
Iinitial(::Val{:U1}, atype, dtype, D; kwarg...) = IU1(atype, dtype, D; kwarg...)

function Iinitial(A::AbstractArray{T, N}, D; kwarg...) where {T, N}
    atype = typeof(A) <: Union{Array, CuArray} ? _arraytype(A) : _arraytype(A.tensor)
    Iinitial(Val(getsymmetry(A)), atype, ComplexF64, D; kwarg...)
end

zerosinitial(::Val{:none}, atype, dtype, a...; kwarg...) = atype(zeros(dtype, a...))
zerosinitial(::Val{:Z2}, atype, dtype, a...; kwarg...) = zerosZ2(atype, dtype, a...; kwarg...)
zerosinitial(::Val{:U1}, atype, dtype, a...; kwarg...) = zerosU1(atype, dtype, a...; kwarg...)

function zerosinitial(A::AbstractArray{T, N}, a...; kwarg...) where {T, N}
    atype = typeof(A) <: Union{Array, CuArray} ? _arraytype(A) : _arraytype(A.tensor)
    zerosinitial(Val(getsymmetry(A)), atype, T, a...; kwarg...)
end

asArray(A::AbstractArray) = A

"""
    asSymmetryArray(A::AbstractArray, symmetry; dir = nothing)

Transform Array to a SymmetryArray.
now supports:
    `:none`
    `:Z2`
    `:U1`
"""
asSymmetryArray(A::AbstractArray, ::Val{:none}; kwarg...) = A
asSymmetryArray(A::AbstractArray, ::Val{:Z2}; kwarg...) = asZ2Array(A; kwarg...)
asSymmetryArray(A::AbstractArray, ::Val{:U1}; kwarg...) = asU1Array(A; kwarg...)

symmetryreshape(A::AbstractArray, s...; kwarg...) = reshape(A, s...), nothing
symmetryreshape(A::Z2Array, s...; kwarg...) = Z2reshape(A, s...; kwarg...)
symmetryreshape(A::U1Array, s...; kwarg...) = U1reshape(A, s...; kwarg...)