# ============================================================================
# Unified checkpoint API
# ============================================================================
# Single entry point `checkpoint(method, f, args...; kwargs...)` dispatched on
# a singleton-type method selector:
#
#   Plain()     — identity, no adjoint override (plain autodiff)
#   Recompute() — identity forward, re-run on backward (classic checkpoint)
#   Offload()   — identity forward, copy args to host, reconstruct on backward
#
# Symbol sugar: `:plain` / `:none` / `:recompute` / `:offload` are accepted at
# the call site and as VUMPS field kwargs via `Base.convert`.
#
# Future methods (e.g. `Tape()`) add a new subtype + a `checkpoint(::Tape, ...)`
# method + a `Zygote.@adjoint` without touching existing code.

abstract type CheckpointMethod end
struct Plain     <: CheckpointMethod end
struct Recompute <: CheckpointMethod end
struct Offload   <: CheckpointMethod end

# ── Symbol normalization ─────────────────────────────────────────────────────
_ckpt_method(m::CheckpointMethod) = m
_ckpt_method(s::Symbol) = s === :plain || s === :none ? Plain()     :
                          s === :recompute            ? Recompute() :
                          s === :offload              ? Offload()   :
                          throw(ArgumentError("unknown checkpoint method: $s (expected :plain, :none, :recompute, or :offload)"))

# Let `VUMPS(; step_checkpoint=:offload, ...)` (or any struct field of type
# `CheckpointMethod`) accept a Symbol transparently.
Base.convert(::Type{CheckpointMethod}, s::Symbol) = _ckpt_method(s)

# ── Inner-map granularity constraint ────────────────────────────────────────
# Inner-map checkpoint (wrapping each FLmap/FRmap/ACmap/Cmap call inside
# power iteration) only supports Plain and Recompute. Offloading per-map is
# already marginal at the eig granularity due to StructArray pointer sharing,
# and inner-map is finer yet — reject it explicitly.
_assert_inner_method(::Union{Plain,Recompute}) = nothing
_assert_inner_method(::Offload) = throw(ArgumentError(
    "inner-map checkpoint only supports Plain/Recompute; " *
    "use eig_checkpoint=Offload() or step_checkpoint=Offload() for offload"))

# ── Symbol dispatcher ────────────────────────────────────────────────────────
checkpoint(m::Symbol, f, args...; kwargs...) = checkpoint(_ckpt_method(m), f, args...; kwargs...)

# ── Plain: no adjoint override ──────────────────────────────────────────────
checkpoint(::Plain, f, args...; kwargs...) = f(args...; kwargs...)

# ── Recompute: identity forward, re-run on backward ─────────────────────────
# See Zygote Checkpointing https://fluxml.ai/Zygote.jl/latest/adjoints/#Checkpointing-1
checkpoint(::Recompute, f, args...; kwargs...) = f(args...; kwargs...)

Zygote.@adjoint checkpoint(m::Recompute, f, args...; kwargs...) =
    f(args...; kwargs...),
    ȳ -> begin
        inner = Zygote._pullback((aa...) -> f(aa...; kwargs...), args...)[2](ȳ)
        (nothing, inner...)  # prepend ∂m = nothing
    end

# ── Offload: copy args to host after forward, reconstruct on backward ───────
# On GPU this trades device VRAM for host RAM + two one-way transfers: after
# the forward returns, the explicit args are copied to host and the outer
# Zygote tape only holds the CPU snapshot + the target atype. During backward
# the snapshot is copied back to the device and the forward is re-run under
# Zygote to produce the pullback.
#
# On CPU this collapses to `Recompute()` modulo a copy roundtrip.
#
# Caveat: only the explicit `args` are offloaded. Any large tensors captured
# inside `f` as a closure still live on the device.

_offload_to_host(x::AbstractArray{<:Number}) = Array(x)
_offload_to_host(x::AbstractArray) = map(_offload_to_host, x)
_offload_to_host(x::Tuple) = map(_offload_to_host, x)
_offload_to_host(x) = x

# Atype detection: returns Array / CuArray / ROCArray, or `nothing` if no
# array-like could be found in `x`. Specialisations for StructArray and
# runtime structs live in boundary_algorithm/environment.jl (where those
# types are defined).
_atype_of(x::AbstractArray{<:Number}) = _arraytype(x)
_atype_of(x::AbstractArray) = isempty(x) ? nothing : _atype_of(first(x))
_atype_of(x::Tuple) = begin
    for a in x
        at = _atype_of(a)
        at === nothing || return at
    end
    return nothing
end
_atype_of(x) = nothing

function _detect_target_atype(args)
    for a in args
        at = _atype_of(a)
        at === nothing || return at
    end
    return Array
end

# Reconstruct an on-device copy from a CPU copy, given the target atype.
# The pullback closure captures only `atype` and `args_cpu`, never the
# original `args`, so the device originals become eligible for GC/free.
_to_atype(atype, x::Array{<:Number}) = atype(x)
_to_atype(atype, x::AbstractArray) = map(a -> _to_atype(atype, a), x)
_to_atype(atype, x::Tuple) = map(a -> _to_atype(atype, a), x)
_to_atype(_atype, x) = x

checkpoint(::Offload, f, args...; kwargs...) = f(args...; kwargs...)

Zygote.@adjoint function checkpoint(m::Offload, f, args...; kwargs...)
    y = f(args...; kwargs...)
    atype = _detect_target_atype(args)
    args_cpu = map(_offload_to_host, args)
    # NOTE: the returned closure deliberately does NOT reference `args` so
    # that Julia will not capture it; only `args_cpu` + `atype` survive.
    return y, function(ȳ)
        args_dev = map(a -> _to_atype(atype, a), args_cpu)
        inner = Zygote._pullback((aa...) -> f(aa...; kwargs...), args_dev...)[2](ȳ)
        (nothing, inner...)
    end
end
