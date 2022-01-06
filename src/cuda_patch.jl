using CUDA
using LinearAlgebra
using OMEinsum
using Zygote: @adjoint
CUDA.allowscalar(false)

import Base: getindex, _to_subscript_indices, zero
import CUDA: CuArray

const CublasFloat = Union{Float64,Float32,ComplexF64,ComplexF32}
const CublasReal = Union{Float64,Float32}

import LinearAlgebra: mul!, axpy!

# https://github.com/Jutho/KrylovKit.jl/issues/15#issuecomment-464839305
mul!(y::CuArray, x::CuArray, α::T) where {T <: CublasFloat} = (y .= α .* x)
axpy!(α::Complex{T}, x::CuArray{T}, y::CuArray{Complex{T}}) where {T <: CublasReal} = (y .+= α .*x)

# for CuArray index [] to get scalar
getindex(A::CuArray{T,0,B}) where {T,B} = Array(A)[]

@adjoint function getindex(A::CuArray{T,0,B}) where {T,B}
    function back(Δ)
        return (CuArray([Δ]), )
    end
    return Array(A)[], back
end

# for ein"abab -> "(A)[]
dtr(A::AbstractArray) = ein"abab -> "(A)[]
