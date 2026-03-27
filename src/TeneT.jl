module TeneT

using LinearAlgebra, Random, Printf
using Parameters

using CUDA
using AMDGPU
using cuTENSOR

using TensorOperations

using Zygote, ChainRulesCore, ForwardDiff
using KrylovKit, VectorInterface

using OptimKit, LineSearches

using JLD2, FileIO

using MPI

using Base.Threads
CUDA.allowscalar(false)

# ============================================================================
# Core types and defaults
# ============================================================================

include("types.jl")
include("defaults.jl")

# ============================================================================
# StructArray (custom unit-cell array with pattern)
# ============================================================================

include("structarray/base.jl")
include("structarray/initial.jl")
include("structarray/buffer.jl")

# ============================================================================
# Utility functions
# ============================================================================

include("utils/gpu.jl")
include("utils/io.jl")
include("utils/misc.jl")

# ============================================================================
# Patch for other packages (e.g. Zygote, OptimKit)
# ============================================================================

include("patch/OptimKit_patch.jl")

# ============================================================================
# Contraction kernels
# ============================================================================

include("contraction/basic.jl")
include("contraction/forloop_parallel_MPI.jl")
include("contraction/observable.jl")

# ============================================================================
# Boundary algorithms (environment structs, then algorithm implementations)
# ============================================================================

include("boundary_algorithm/algorithm.jl")
include("boundary_algorithm/environment.jl")
include("boundary_algorithm/vumps/general.jl")
include("boundary_algorithm/vumps/plaquette.jl")
include("boundary_algorithm/vumps/c4v.jl")
include("boundary_algorithm/qrctm.jl")

# ============================================================================
# Automatic differentiation rules
# ============================================================================

include("autodiff/rules.jl")
include("autodiff/grassmann.jl")

# ============================================================================
# Hamiltonian models
# ============================================================================

include("models/basic_op.jl")
include("models/heisenberg/hamiltonian.jl")
include("models/heisenberg/energy.jl")

# ============================================================================
# iPEPS: interface, build, init, restriction, precondition, SU, energy, observable, optimize
# ============================================================================

include("ipeps_optimize/interface.jl")
include("ipeps_optimize/build_A.jl")
include("ipeps_optimize/init.jl")
include("ipeps_optimize/restriction.jl")
include("ipeps_optimize/precondition.jl")
include("ipeps_optimize/su_parameterization.jl")
include("ipeps_optimize/observable.jl")
include("ipeps_optimize/optimize.jl")

# ============================================================================
# Exports
# ============================================================================

export VUMPS
export QRCTM

export Heisenberg
export Square, Honeycomb, Kagome

export C4v_restriction

export GradientOptimize, init_ipeps, optimise_ipeps
export observable

end # module TeneT
