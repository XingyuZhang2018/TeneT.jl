module TeneT

using CUDA
using LinearAlgebra
using KrylovKit
using Zygote
using OMEinsum
using Printf
using Parameters
using ChainRulesCore

import Base: +, -, *, getindex
import LinearAlgebra: norm,  mul!
import VectorInterface: inner, scale, scale!!, scalartype, zerovector
# import KrylovKit: RealVec
export SquareLattice, HoneycombLattice
export VUMPS, VUMPSRuntime, VUMPSEnv
export leading_boundary


include("defaults.jl")
include("utilities.jl")
include("patch.jl")
include("environment.jl")
include("vumpsruntime.jl")
include("grassmann.jl")
include("autodiff.jl")


end
