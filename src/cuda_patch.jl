using CUDA
using LinearAlgebra

CUDA.allowscalar(false)

const CublasFloat = Union{Float64,Float32,ComplexF64,ComplexF32}
const CublasReal = Union{Float64,Float32}

import LinearAlgebra

# https://github.com/Jutho/KrylovKit.jl/issues/15#issuecomment-464839305
LinearAlgebra.mul!(y::CuArray, x::CuArray, α::T) where {T <: CublasFloat} = (y .= α .* x)
LinearAlgebra.axpy!(α::Complex{T}, x::CuArray{T}, y::CuArray{Complex{T}}) where {T <: CublasReal} = (y .+= α .*x)


#helper functions to handle array types
_mattype(x::Array{T}) where {T} = Matrix
_mattype(x::CuArray{T}) where {T} = CuMatrix
_mattype(x::Adjoint{T, CuArray{T, 2 ,B}}) where {T,B} = CuMatrix
_mattype(x::Symmetric{T, CuArray{T, 2, B}}) where {T,B} = CuMatrix

_arraytype(x::Array{T}) where {T} = Array
_arraytype(x::CuArray{T}) where {T} = CuArray