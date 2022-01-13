using CUDA
using LinearAlgebra
using OMEinsum
using Zygote: @adjoint
CUDA.allowscalar(false)

import Base: getindex

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
