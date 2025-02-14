module TeneT

using CUDA
using LinearAlgebra
using KrylovKit
using Zygote
using OMEinsum
using Printf
using Parameters
using ChainRulesCore

import Base: +, -, *, getindex, Array
import LinearAlgebra: norm,  mul!
import VectorInterface: inner, scale, scale!!, scalartype, zerovector, add!!
import CUDA: CuArray
# import KrylovKit RealVec
export StructArray, randSA
export VUMPS, VUMPSRuntime, VUMPSEnv
export leading_boundary

CUDA.allowscalar(false)

include("defaults.jl")
include("structarray/base.jl")
include("structarray/initial.jl")
include("structarray/buffer.jl")
include("utilities.jl")
include("patch.jl")
include("environment.jl")
include("vumpsruntime.jl")
include("grassmann.jl")
include("autodiff.jl")


end
