module TeneT

using CUDA
using ChainRulesCore
using TensorKit
using LinearAlgebra
using KrylovKit
using Zygote
using Printf
using Parameters
using Random

import Base: +, -, *, getindex, Array
import Random: rand, rand!
import LinearAlgebra: norm,  mul!
import CUDA: CuArray

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
include("initial.jl")
include("contraction.jl")
include("environment.jl")
include("vumpsruntime.jl")
include("autodiff.jl")

end
