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
include("ipeps_optimize/build.jl")
include("ipeps_optimize/init.jl")
include("ipeps_optimize/restriction.jl")
include("ipeps_optimize/precondition.jl")
include("ipeps_optimize/su_parameterization.jl")
include("ipeps_optimize/observable.jl")
include("ipeps_optimize/optimize.jl")

# ============================================================================
# Exports
# ============================================================================

# --- Core types ---
export AbstractLattice, Square, Honeycomb, Kagome
export ContractionMode, General, Plaquette
export Algorithm
export iPEPSOptimize, HamiltonianModel

# --- Defaults ---
export Defaults

# --- StructArray ---
export StructArray
export randSA, ISA

# --- GPU / array utilities ---
export _arraytype, _mattype
export set_device_id!, get_device, get_device_id, device_count
export atype_device!
export reclaim

# --- I/O ---
export save_rt, load_rt, read_last_log

# --- Misc utilities ---
export leg3, leg4, leg5, leg8
export _to_front, _to_tail
export permute_fronttail
export simple_eig
export mcform
export checkpoint
export takagi_decomposition

# --- Contraction kernels ---
export ρmap
export FLmap, FRmap, ACmap, ACdmap
export FLmap_parallel, FRmap_parallel, ACmap_parallel, ACdmap_parallel
export Mumap_parallel

# --- Observable contractions ---
export oc_H_leg3, oc_V_leg3
export oc_H_leg4, oc_V_leg4
export oc1_leg3, oc1_leg4
export contract_n1, contract_o1
export contract_n2_H, contract_o2_H
export contract_n2_V, contract_o2_V
export contract_n_D, contract_o_D1, contract_o_D2
export contract_n3_H, contract_o3_H
export oc_D_leg4, oc_Q_4_corner

# --- Boundary algorithms ---
export CTMRG, QRCTM, FPCTM, PT, VUMPS

# --- Boundary environments ---
export VUMPSRuntime, VUMPSEnv, CTMEnv
export PlaquetteRuntime, PlaquetteVUMPSEnv
export update!
export leading_boundary

# --- VUMPS helpers ---
export qrpos, lqpos, selectpos
export ALCtoAC

# --- Models ---
export Heisenberg, J1J2, J1J2J3, SS, Kagome
export const_Sx, const_Sy, const_Sz
export hamiltonian, hamiltonian_trunc
export hamiltonian_onsite, hamiltonian_right, hamiltonian_down
export expectation_value

# --- iPEPS optimization parameter structs ---
export GradientOptimize, SUOptimize, FUOptimize

# --- iPEPS build / init ---
export build_A
export init_ipeps, init_ipeps_to_D, init_ipeps_perturbation, init_ipeps_from_small_D
export initialize_env

# --- iPEPS restriction / canonical forms ---
export _restriction_ipeps, C4v_restriction
export central_canonical1, central_canonical2
export pepsgeneral, pepsgeneral_Ac
export ARstoA, ARstoA1, ARstoA2
export local_gauge_contraction, guage_transfer
export find_local_min_norm_G, local_min_norm
export find_local_hermite_G, local_hermite
export to_mcf_ipeps, solve_balancing_gauge, local_min_norm_iter

# --- iPEPS preconditioning ---
export precondition_invese_single_envir
export precondition_invese_N
export precondition_invese_hessian
export precondition_invese_BP_envir
export precondition_invese_single_chi1_envir
export environment_FWAD

# --- iPEPS SU parameterization ---
export SU_parameterization
export hv_SU_update, one_bond_SU

# --- iPEPS energy / observable ---
export energy
export magnetization_value, cor_len_value
export observable, write_obs_log

# --- iPEPS optimization ---
export optimise_ipeps
export optimize_reload
export LBFGSState, save_lbfgs_state, load_lbfgs_state

# --- Autodiff ---
export project_AL, project_AL!, project_AR, project_AR!

end # module TeneT
