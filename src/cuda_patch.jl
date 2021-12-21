using CUDA
using LinearAlgebra

CUDA.allowscalar(false)

const CublasFloat = Union{Float64,Float32,ComplexF64,ComplexF32}
const CublasReal = Union{Float64,Float32}

import LinearAlgebra: mul!, axpy!

# https://github.com/Jutho/KrylovKit.jl/issues/15#issuecomment-464839305
mul!(y::CuArray, x::CuArray, α::T) where {T <: CublasFloat} = (y .= α .* x)
axpy!(α::Complex{T}, x::CuArray{T}, y::CuArray{Complex{T}}) where {T <: CublasReal} = (y .+= α .*x)