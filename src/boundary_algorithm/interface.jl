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

    # M5 Slice2D (2D block-distributed) path. When a Slice2DGrid is set, `vumps_step`
    # routes to `vumps_step_slice2d` (block AL/AR/FL/FR, replicated C, QR gather seam).
    # nothing → serial path unchanged. Untyped (like ifparallelupdown) to avoid a
    # forward-reference to Slice2DGrid (defined later in the module).
    grid = nothing
    # New public parallel API. When set, this method object takes precedence over
    # legacy `ifparallel`/`grid` routing at the high-level VUMPS boundary.
    parallel_method = nothing
    # Opt-in forward-only distributed QR seam for Plaquette Slice2D observation runs.
    # The default gather QR seam remains in place for AD/optimization.
    distributed_qr::Bool = false

    inner_etype::Union{Nothing, Type} = nothing
    inner_etype_final_steps::Int = 0
    simple_eig_polish_steps::Int = 0
    # When set (e.g. Float32), leading_boundary converts rt/M to this precision
    # at entry and runs the WHOLE VUMPS step (FLmap, QR, norm, eigsolve, ...) in
    # it. Polish iters (inner_etype_final_steps) convert rt back to original
    # precision. Alternative to `inner_etype` which only affects @tensor inside
    # FLmap/FRmap/ACmap. Mutually exclusive: set one OR the other, not both.
    whole_vumps_etype::Union{Nothing, Type} = nothing

    # Checkpointing for AD, four granularities (fine → coarse).
    #
    # `ifcheckpoint::Bool` is a master switch:
    #   false (default) — every checkpoint defaults to Plain() (no
    #                     checkpointing — fast, full tape). Good for
    #                     small-D testing where everything fits.
    #   true            — checkpoints default to the "R2 winner" production
    #                     preset (D=10 χ=400 H200; 15% faster fg than
    #                     all-Plain at production scale; see
    #                     docs/2026-05-14-checkpoint-sweep-round2-B0-neighborhood.md).
    #
    # Individual `*_checkpoint` kwargs always win over the ifcheckpoint
    # default — pass them explicitly to override per layer.
    #
    #   segment_checkpoint — wraps each `_power_iter_segment` chunk inside
    #                         `simple_eig` (`checkpoint_every` power iters per
    #                         segment). Bounds the tape peak during a single
    #                         simple_eig execution. R2 default: OffloadRecompute.
    #   inner_checkpoint   — wraps each FLmap/FRmap/ACmap/Cmap call inside
    #                         power iteration. Supports Plain/Recompute only
    #                         (Offload rejected at runtime). R2 default: Plain.
    #   eig_checkpoint     — wraps the per-row `simple_eig` in leftenv /
    #                         rightenv / ACenv. Plain keeps the fast closure
    #                         path; Recompute/Offload go through the
    #                         `_simple_eig_*map` explicit-args wrappers.
    #                         R2 default: Recompute.
    #   subop_checkpoint   — wraps each individual subop inside vumps_step
    #                         (leftenv, rightenv, ACenv, Cenv, ALCtoAC,
    #                         ACCtoALAR). Between eig and step in the
    #                         granularity hierarchy. R2 default: OffloadRecompute.
    #   step_checkpoint    — wraps the whole `vumps_step`. Coarsest; typically
    #                         the biggest VRAM lever. R2 default: OffloadRecompute.
    #
    # Accepts `Plain()`/`Recompute()`/`Offload()` singletons, or a Symbol
    # (`:plain`, `:recompute`, `:offload`) via `Base.convert`.
    ifcheckpoint::Bool = false
    segment_checkpoint::CheckpointMethod = ifcheckpoint ? OffloadRecompute() : Plain()
    inner_checkpoint::CheckpointMethod   = Plain()
    eig_checkpoint::CheckpointMethod     = ifcheckpoint ? Recompute() : Plain()
    subop_checkpoint::CheckpointMethod   = ifcheckpoint ? OffloadRecompute() : Plain()
    step_checkpoint::CheckpointMethod    = ifcheckpoint ? OffloadRecompute() : Plain()
end

function _apply_parallel_method!(alg)
    method = alg.parallel_method
    method === nothing && return alg

    if method isa SerialMethod
        alg.ifparallel = false
    elseif method isa Slice1DMethod
        alg.ifparallel = true
    elseif method isa Slice2DMethod
        alg.ifparallel = false
    else
        throw(ArgumentError("Unsupported parallel_method $(typeof(method)). Use slice1D(...) or slice2D(...)."))
    end

    alg.forloop_iter = method.forloop_iter
    alg.inner_etype = method.inner_etype
    return alg
end

function _effective_grid(alg)
    _apply_parallel_method!(alg)
    return alg.parallel_method isa Slice2DMethod ? alg.parallel_method.grid : alg.grid
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
    QRCTMRG{F} <: Algorithm

QR-based Corner Transfer Matrix Renormalization Group algorithm.
"""
@kwdef mutable struct QRCTMRG{F <: ContractionMode} <: Algorithm
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
    # Per-step checkpoint method wrapping `qrctmrg_step` inside the AD loop.
    # Plain()            — no checkpointing (default)
    # Recompute()        — rerun qrctmrg_step on backward (trades compute for memory)
    # OffloadRecompute() — copy args to host after forward, copy back + rerun on backward
    # Offload()          — pb-capture walker swaps GPU captures for CPU copies; on
    #                      backward reload + run original pb (no forward recompute).
    step_checkpoint::CheckpointMethod = Plain()
    forloop_iter::Int = 1

    # Mixed-precision fields (same semantics as VUMPS; see VUMPS struct doc).
    # QRCTMRG doesn't use simple_eig, so simple_eig_polish_steps is functionally
    # inactive but kept for API symmetry.
    inner_etype::Union{Nothing, Type} = nothing
    inner_etype_final_steps::Int = 0
    simple_eig_polish_steps::Int = 0
    whole_vumps_etype::Union{Nothing, Type} = nothing
end

# Convenience: QRCTMRG(C4v(); kwargs...) mirrors VUMPS(C4v(); kwargs...).
QRCTMRG(::F; kwargs...) where {F <: ContractionMode} = QRCTMRG{F}(; kwargs...)
