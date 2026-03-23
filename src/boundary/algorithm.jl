# Boundary contraction algorithm structs
# All algorithms are subtypes of Algorithm (defined in types.jl)

"""
    CTMRG <: Algorithm

Corner Transfer Matrix Renormalization Group algorithm.
"""
@kwdef mutable struct CTMRG <: Algorithm
    tol::Float64 = 1e-10
    maxiter::Int = 100
    miniter::Int = 1
    maxiter_ad::Int = 10
    miniter_ad::Int = 1
    output_interval::Int = 1
    verbosity::Int = Defaults.verbosity
    ifsimple_eig::Bool = true
end

"""
    QRCTM <: Algorithm

QR-based Corner Transfer Matrix algorithm.
"""
@kwdef mutable struct QRCTM <: Algorithm
    tol::Float64 = 1e-10
    maxiter::Int = 100
    miniter::Int = 1
    maxiter_ad::Int = 10
    miniter_ad::Int = 1
    output_interval::Int = 1
    verbosity::Int = Defaults.verbosity
    ifsimple_eig::Bool = true
    maxiter_power::Int = 1
    ifload_env::Bool = true
    ifsave_env::Bool = true
    ifparallel::Bool = false
    ifcheckpoint::Bool = false
    forloop_iter::Int = 1
end

"""
    FPCTM <: Algorithm

Fixed-Point Corner Transfer Matrix algorithm.
"""
@kwdef mutable struct FPCTM <: Algorithm
    tol::Float64 = 1e-10
    maxiter::Int = 100
    miniter::Int = 1
    maxiter_ad::Int = 10
    miniter_ad::Int = 1
    output_interval::Int = 1
    verbosity::Int = Defaults.verbosity
    ifsimple_eig::Bool = true
end

"""
    PT <: Algorithm

Power Transfer algorithm.
"""
@kwdef mutable struct PT <: Algorithm
    tol::Float64 = 1e-10
    maxiter::Int = 100
    miniter::Int = 1
    maxiter_ad::Int = 10
    miniter_ad::Int = 1
    output_interval::Int = 1
    verbosity::Int = Defaults.verbosity
    ifsimple_eig::Bool = true
    maxiter_power::Int = 10
    ifload_env::Bool = true
    ifsave_env::Bool = true
    ifparallel::Bool = false
    ifcheckpoint::Bool = false
    forloop_iter::Int = 1
end

"""
    VUMPS{M<:ContractionMode} <: Algorithm

Variational Uniform Matrix Product State algorithm, parameterized by
a `ContractionMode` (`General` or `Plaquette`).
"""
@kwdef mutable struct VUMPS{M<:ContractionMode} <: Algorithm
    mode::M = General()

    tol::Float64 = 1e-10
    maxiter::Int = 10
    miniter::Int = 1
    maxiter_ad::Int = 10
    miniter_ad::Int = 3
    forloop_iter::Int = 1
    power_iter::Int = 5
    power_iter_obs::Int = 20

    show_every::Int = 10
    verbosity::Int = Defaults.verbosity

    ifupdown::Bool = true
    ifdownfromup::Bool = false
    ifparallel::Bool = false
    ifsimple_eig::Bool = true
    ifcheckpoint::Bool = false
    ifgpu_cpu_combo::Bool = false
    iflinear_ad::Bool = false
end
