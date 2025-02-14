#helper functions to handle array types
_mattype(x::Array{T}) where {T} = Matrix
_mattype(x::CuArray{T}) where {T} = CuMatrix
_mattype(x::Adjoint{T, CuArray{T, 2 ,B}}) where {T,B} = CuMatrix
_mattype(x::Symmetric{T, CuArray{T, 2, B}}) where {T,B} = CuMatrix

_arraytype(x::Array{T}) where {T} = Array
_arraytype(x::CuArray{T}) where {T} = CuArray