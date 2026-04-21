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

    inner_etype::Union{Nothing, Type} = nothing
    inner_etype_final_steps::Int = 0
    simple_eig_polish_steps::Int = 0
    # When set (e.g. Float32), leading_boundary converts rt/M to this precision
    # at entry and runs the WHOLE VUMPS step (FLmap, QR, norm, eigsolve, ...) in
    # it. Polish iters (inner_etype_final_steps) convert rt back to original
    # precision. Alternative to `inner_etype` which only affects @tensor inside
    # FLmap/FRmap/ACmap. Mutually exclusive: set one OR the other, not both.
    whole_vumps_etype::Union{Nothing, Type} = nothing

    # Checkpointing for AD, four granularities (fine → coarse), all default
    # to `Plain()` (no checkpointing — fast, full tape). Opt in per level.
    #   segment_checkpoint — wraps each `_power_iter_segment` chunk inside
    #                         `simple_eig` (`checkpoint_every` power iters per
    #                         segment). Bounds the tape peak during a single
    #                         simple_eig execution. Set `Recompute()` to enable.
    #   inner_checkpoint   — wraps each FLmap/FRmap/ACmap/Cmap call inside
    #                         power iteration. Supports Plain/Recompute only
    #                         (Offload rejected at runtime).
    #   eig_checkpoint     — wraps the per-row `simple_eig` in leftenv /
    #                         rightenv / ACenv. Plain keeps the fast closure
    #                         path; Recompute/Offload go through the
    #                         `_simple_eig_*map` explicit-args wrappers.
    #   step_checkpoint    — wraps the whole `vumps_step`. Coarsest; typically
    #                         the biggest VRAM lever (memory says so).
    # Accepts `Plain()`/`Recompute()`/`Offload()` singletons, or a Symbol
    # (`:plain`, `:recompute`, `:offload`) via `Base.convert`.
    segment_checkpoint::CheckpointMethod = Plain()
    inner_checkpoint::CheckpointMethod   = Plain()
    eig_checkpoint::CheckpointMethod     = Plain()
    step_checkpoint::CheckpointMethod    = Plain()
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

    # Mixed-precision fields (same semantics as VUMPS; see VUMPS struct doc).
    # QRCTM doesn't use simple_eig, so simple_eig_polish_steps is functionally
    # inactive but kept for API symmetry.
    inner_etype::Union{Nothing, Type} = nothing
    inner_etype_final_steps::Int = 0
    simple_eig_polish_steps::Int = 0
    whole_vumps_etype::Union{Nothing, Type} = nothing
end
