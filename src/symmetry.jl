export asArray, asSymmetryArray, symmetryreshape, getsymmetry, getdir
export _mattype, _arraytype
export randinitial, zerosinitial, Iinitial
export SymmetricType

@with_kw struct SymmetricType
    symmetry = Val(:U1)
    stype::AbstractSiteType
    atype = Array
    dtype = ComplexF64
end

#helper functions to handle array types
_mattype(::Array{T}) where {T} = Matrix
_mattype(::CuArray{T}) where {T} = CuMatrix
_mattype(::Adjoint{T, CuArray{T, 2 ,B}}) where {T,B} = CuMatrix
_mattype(::Symmetric{T, CuArray{T, 2, B}}) where {T,B} = CuMatrix

_arraytype(::Array) = Array
_arraytype(::CuArray) = CuArray
_arraytype(::Diagonal{T, Vector{T}}) where {T} = Array
_arraytype(::Diagonal{T, CuArray{T, 1, B}}) where {T, B} = CuArray
_arraytype(::U1Array{T}) where {T} = U1Array
_arraytype(::Adjoint{T, Matrix{T}}) where {T} = Matrix
_arraytype(::Adjoint{T, CuArray{T, 2, B}}) where {T,B} = CuArray
_arraytype(::Transpose{T, Vector{T}}) where {T} = Vector
_arraytype(::Transpose{T, Matrix{T}}) where {T} = Matrix
_arraytype(::Transpose{T, CuArray{T, N, B}}) where {T,N,B} = CuArray

getsymmetry(::AbstractArray) = :none
getsymmetry(::U1Array) = :U1

getdir(::AbstractArray) = nothing

randinitial(SD::SymmetricType, a...; kwarg...) = randinitial(SD.symmetry, SD.stype, SD.atype, SD.dtype, a...; kwarg...)
randinitial(::Val{:none}, atype, dtype, a...; kwarg...) = atype(rand(dtype, a...))
randinitial(::Val{:none}, sitetype::AbstractSiteType, atype, dtype, a...; kwarg...) = atype(rand(dtype, a...))
randinitial(::Val{:U1}, atype, dtype, a...; kwarg...) = randU1(atype, dtype, a...; kwarg...)
randinitial(::Val{:U1}, sitetype::AbstractSiteType, atype, dtype, a...; kwarg...) = randU1(sitetype,atype, dtype, a...; kwarg...)

function randinitial(A::AbstractArray{T, N}, a...; kwarg...) where {T, N}
    atype = typeof(A) <: Union{Array, CuArray} ? _arraytype(A) : _arraytype(A.tensor)
    randinitial(Val(getsymmetry(A)), atype, T, a...; ifZ2=A.ifZ2, kwarg...)
end

Iinitial(SD::SymmetricType, a...; kwarg...) = Iinitial(SD.symmetry, SD.stype, SD.atype, SD.dtype, a...; kwarg...)
Iinitial(::Val{:none}, atype, dtype, D; kwarg...) = atype{dtype}(I, D, D)
Iinitial(::Val{:none}, sitetype::AbstractSiteType, atype, dtype, D; kwarg...) = atype{dtype}(I, D, D)
Iinitial(::Val{:U1}, atype, dtype, D; kwarg...) = IU1(atype, dtype, D; kwarg...)
Iinitial(::Val{:U1}, sitetype::AbstractSiteType, atype, dtype, D; kwarg...) = IU1(sitetype, atype, dtype, D; kwarg...)

function Iinitial(A::AbstractArray{T, N}, D; kwarg...) where {T, N}
    atype = typeof(A) <: Union{Array, CuArray} ? _arraytype(A) : _arraytype(A.tensor)
    Iinitial(Val(getsymmetry(A)), atype, ComplexF64, D; ifZ2=A.ifZ2, kwarg...)
end

zerosinitial(::Val{:none}, atype, dtype, a...; kwarg...) = atype(zeros(dtype, a...))
zerosinitial(::Val{:U1}, atype, dtype, a...; kwarg...) = zerosU1(atype, dtype, a...; kwarg...)

function zerosinitial(A::AbstractArray{T, N}, a...; kwarg...) where {T, N}
    atype = typeof(A) <: Union{Array, CuArray} ? _arraytype(A) : _arraytype(A.tensor)
    zerosinitial(Val(getsymmetry(A)), atype, T, a...; ifZ2=A.ifZ2, kwarg...)
end

asArray(A::Union{Array, CuArray}) = A

"""
    asSymmetryArray(A::AbstractArray, symmetry; dir = nothing)

Transform Array to a SymmetryArray.
now supports:
    `:none`
    `:Z2`
    `:U1`
"""
asSymmetryArray(A::AbstractArray, ::Val{:none}; kwarg...) = A
asSymmetryArray(A::AbstractArray, ::Val{:U1}, sitetype::AbstractSiteType;  kwarg...) = asU1Array(sitetype, A; kwarg...)

symmetryreshape(A::AbstractArray, s...; kwarg...) = reshape(A, s...), nothing
symmetryreshape(A::U1Array, s...; kwarg...) = U1reshape(A, s...; kwarg...)