module TeneT

using AMDGPU
using Base.Threads
using CUDA
using LinearAlgebra
using KrylovKit
using Zygote
using OMEinsum
using Printf
using Parameters
using ChainRulesCore
using FileIO
using MPI

import Base: +, -, *, getindex, Array
import LinearAlgebra: norm,  mul!
import VectorInterface: inner, scale, scale!!, scalartype, zerovector, add!!
import CUDA: CuArray
# import KrylovKit RealVec
export StructArray, randSA
export VUMPS, VUMPSRuntime, VUMPSEnv
export leading_boundary
export save_rt, load_rt

CUDA.allowscalar(false)

include("defaults.jl")
include("structarray/base.jl")
include("structarray/initial.jl")
include("structarray/buffer.jl")
include("utilities.jl")
include("contraction/basic.jl")
include("contraction/for_loop.jl")
include("contraction/parallel_MPI.jl")
include("initial_env.jl")
include("environment.jl")
include("vumpsruntime.jl")
include("grassmann.jl")
include("autodiff.jl")

end
