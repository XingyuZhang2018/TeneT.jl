module TeneT

include("cuda_patch.jl")
include("environment.jl")
include("fixedpoint.jl")

abstract type AbstractSymmetricArray{T,N} <: AbstractArray{T,N} end
export AbstractSymmetricArray
include("sitetype.jl")
include("u1symmetry.jl")
include("symmetry.jl")

include("vumpsruntime.jl")
include("autodiff.jl")

end
