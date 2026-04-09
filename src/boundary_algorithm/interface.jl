# Boundary contraction algorithm structs
# All algorithms are subtypes of Algorithm (defined in types.jl)
"""
    VUMPS{F} <: Algorithm

Variational Uniform Matrix Product State algorithm
General
Plaquette{<:AbstractLattice}
C4v
"""
@kwdef mutable struct VUMPS{F <: ContractionMode} <: Algorithm
    tol::Float64 = 1e-10
    maxiter::Int = 10
    miniter::Int = 1
    maxiter_ad::Int = 10
    miniter_ad::Int = 3
    forloop_iter::Int = 1
    power_iter::Int = 1
    power_iter_ad::Int = 1
    power_iter_obs::Int = 20

    show_every::Int = 10
    verbosity::Int = Defaults.verbosity

    ifupdown::Bool = true
    ifdownfromup::Bool = false
    ifparallelupdown = false
    ifparallel::Bool = false
    ifsimple_eig::Bool = true
    ifcheckpoint::Bool = false
end

# Convenience: VUMPS(General(); kwargs...) or VUMPS(Plaquette(lattice); kwargs...)
VUMPS(::F; kwargs...) where {F <: ContractionMode} = VUMPS{F}(; kwargs...)

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
    show_every::Int = 1
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
    show_every::Int = 1
    verbosity::Int = Defaults.verbosity

    maxiter_power::Int = 1

    ifsimple_eig::Bool = true
    ifparallel::Bool = false
    ifcheckpoint::Bool = false
    forloop_iter::Int = 1
end
