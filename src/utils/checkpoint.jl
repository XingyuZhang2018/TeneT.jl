# ============================================================================
# Unified checkpoint API
# ============================================================================
# Single entry point `checkpoint(method, f, args...; kwargs...)` dispatched on
# a singleton-type method selector. Four classes, ordered from cheapest to
# heaviest memory lever:
#
#   Plain()             — identity, no adjoint override (plain autodiff)
#   Recompute()         — identity forward, re-run forward on backward
#                         (classic checkpoint; GPU args still captured by pb)
#   OffloadRecompute()  — forward + copy args to host; on backward copy back
#                         to device and re-run forward under Zygote
#   Offload()           — forward once; walk the Zygote pullback closure,
#                         swap GPU captures for CPU copies; on backward
#                         reload captures and call pullback (no re-forward)
#
# Symbol sugar: `:plain` / `:none` / `:recompute` / `:offload_recompute` /
# `:offload` are accepted at the call site and as struct-field kwargs via
# `Base.convert`.
#
# Future methods add a new subtype + a `checkpoint(::Method, ...)` method + a
# `Zygote.@adjoint` without touching existing code.

abstract type CheckpointMethod end
struct Plain            <: CheckpointMethod end
struct Recompute        <: CheckpointMethod end
struct OffloadRecompute <: CheckpointMethod end
struct Offload          <: CheckpointMethod end

# ── Symbol normalization ─────────────────────────────────────────────────────
_ckpt_method(m::CheckpointMethod) = m
_ckpt_method(s::Symbol) = s === :plain || s === :none ? Plain()            :
                          s === :recompute            ? Recompute()        :
                          s === :offload_recompute    ? OffloadRecompute() :
                          s === :offload              ? Offload()          :
                          throw(ArgumentError("unknown checkpoint method: $s " *
                              "(expected :plain, :none, :recompute, :offload_recompute, or :offload)"))

# Let `VUMPS(; step_checkpoint=:offload, ...)` (or any struct field of type
# `CheckpointMethod`) accept a Symbol transparently.
Base.convert(::Type{CheckpointMethod}, s::Symbol) = _ckpt_method(s)

# ── Inner-map granularity constraint ────────────────────────────────────────
# Inner-map checkpoint (wrapping each FLmap/FRmap/ACmap/Cmap call inside
# power iteration) only supports Plain and Recompute. Offloading per-map is
# already marginal at the eig granularity due to StructArray pointer sharing,
# and inner-map is finer yet — reject it explicitly.
_assert_inner_method(::Union{Plain,Recompute}) = nothing
_assert_inner_method(::Union{Offload,OffloadRecompute}) = throw(ArgumentError(
    "inner-map checkpoint only supports Plain/Recompute; " *
    "use eig_checkpoint=Offload() or step_checkpoint=Offload() for offload"))

# ── Bond-checkpoint granularity constraint ──────────────────────────────────
# Bond-level checkpoint (wrapping each term inside `_contract_barebones` and
# each single observable in `_contract_one`) only supports Plain and
# Recompute. Offload at this level makes an independent CPU copy of the
# env args *per closure* — for a multi-site multi-term model (e.g. J1J2
# Plaquette: 4 plaquettes × 3 bond directions × 3 Heisenberg terms = 36
# closures, each holding 8 env tensors ≈ 2.4 GB) that reaches ~100 GB host
# RAM with zero GPU peak savings over Recompute (which shares env refs by
# pointer). Reject it to avoid silently blowing up host memory.
_assert_bond_method(::Union{Plain,Recompute}) = nothing
_assert_bond_method(::Union{Offload,OffloadRecompute}) = throw(ArgumentError(
    "bond_checkpoint=Offload() is not supported: bond-level closures share " *
    "env tensors by reference under Recompute() (zero extra GPU or CPU " *
    "cost), but Offload() makes independent CPU copies per closure — " *
    "typically 100+ GB host RAM for multi-site multi-term models — with no " *
    "GPU peak savings over Recompute. Use bond_checkpoint=Recompute() " *
    "(main memory lever); if you need additional host-offload headroom, " *
    "set obs_checkpoint=Offload() at the outer level instead."))

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

# ── Device detection helper ──────────────────────────────────────────────────
# Returns true for GPU-backed AbstractArrays (CuArray, ROCArray). Explicit
# per-type methods rather than `_arraytype(x) !== Array`: the walker can hit
# Julia-internal AbstractArrays (e.g. `Memory{UInt8}` backing `Dict.slots`
# under Julia 1.11) that have no `_arraytype` method, which would throw.
_is_gpu(x) = false
_is_gpu(x::CuArray)  = true
_is_gpu(x::ROCArray) = true

# ── Shared struct-walker primitives ──────────────────────────────────────────
# ChainRules rrules return pullback closures whose captured forward-pass
# values live in anonymous-closure struct fields. To swap GPU→CPU captures
# (or any other per-leaf transform) we walk the struct reflectively via
# `fieldnames`/`getfield`/`jl_new_structv` — the only way to reconstruct
# closure types that have no positional outer constructor.
#
# Closure types generated by anonymous functions and `let`-captures do not
# have a positional outer constructor (the only way to construct them is
# via `jl_new_structv` or `Expr(:new, …)`). `T(children...)` fails with
# MethodError. We therefore go directly through `ccall(:jl_new_structv, …)`,
# which is the low-level Julia primitive that `Expr(:new, …)` also lowers to.
# It works uniformly for:
#   • closures  — `var"#1#2"{…}`
#   • tuples    — `Tuple{T1, T2, …}`
#   • user structs with default constructors
#   • structs with @kw / parametric inner constructors
#
# If reconstruction raises (e.g. an inner constructor validates invariants
# we would break), we fall back to the original subtree and bump
# `_reconstruct_fail_count` for diagnostics. Numerical correctness is
# preserved — offload is skipped for that capture.
#
# AbstractArrays are treated as atomic leaves: they are either transformed
# (when `pred` matches) or returned as-is. The walker never descends into
# array elements or Array's internal struct fields (Julia 1.11 exposes
# `Array` as a struct with `:ref, :size`, which `isstructtype` reports `true`
# — we guard against walking that shape explicitly via the `AbstractArray`
# early-return).
#
# `_reconstruct_fail_count` is NOT thread-safe; it is a single-threaded
# diagnostic signal only.
#
# Origin: this walker is inlined from Wengert.jl Phase 1 (branch
# claude/pb-capture-offload, commit fc96b9a) where it was added to make
# Wengert's `@checkpoint` macro free pb GPU captures. TeneT adopted it
# directly inside the Zygote `Offload()` adjoint — no outer AD switch.

const _reconstruct_fail_count = Ref(0)
_reset_reconstruct_fail_count!() = (_reconstruct_fail_count[] = 0)

# Re-parameterize a possibly-parametric type so its type parameters match the
# types of the (possibly transformed) children. ChainRules-style pullback
# closures look like `times_pullback{CuArray,CuArray}` with one type param
# per captured field. After swapping `CuArray→Array` at the field level we
# must also rebuild the closure TYPE as `times_pullback{Array,Array}`,
# otherwise `jl_new_structv` raises `TypeError: fieldtype mismatch`.
#
# For non-parametric types (`wrapper === T`), return T unchanged. Fallback
# to T if wrapper application fails (e.g. wrong arity, restricted bounds);
# the calling walker will then either catch the subsequent `jl_new_structv`
# failure and bump `_reconstruct_fail_count`.
@inline function _reparameterize_type(::Type{T}, children) where {T}
    wrapper = T.name.wrapper
    wrapper === T && return T
    return try
        wrapper{map(typeof, Tuple(children))...}
    catch
        T
    end
end

# Low-level positional reconstruction that bypasses user-defined constructors.
@inline function _new_struct(::Type{T}, children) where {T}
    T_new = _reparameterize_type(T, children)
    return ccall(:jl_new_structv, Any,
                 (Any, Ptr{Any}, UInt32),
                 T_new, collect(Any, children), length(children))
end

function transform_captured_arrays(x, pred, transform)
    if x isa AbstractArray && pred(x)
        return transform(x)
    end
    x isa AbstractArray && return x
    T = typeof(x)
    !isstructtype(T) && return x
    isbitstype(T)    && return x
    fnames = fieldnames(T)
    isempty(fnames)  && return x
    children = map(fn -> transform_captured_arrays(getfield(x, fn), pred, transform), fnames)
    try
        return _new_struct(T, children)
    catch
        _reconstruct_fail_count[] += 1
        return x
    end
end

# Detect the first GPU `AbstractArray` anywhere in `x`'s struct tree and
# return its `_arraytype` (CuArray/ROCArray). Returns `nothing` if none is
# found. Used by `Offload()` to bind the GPU constructor for reload on
# backward, and as a short-circuit to skip the walker entirely on CPU.
_detect_gpu_atype(x::AbstractArray) = _is_gpu(x) ? _arraytype(x) : nothing
function _detect_gpu_atype(x)
    T = typeof(x)
    !isstructtype(T) && return nothing
    isbitstype(T)    && return nothing
    for fn in fieldnames(T)
        at = _detect_gpu_atype(getfield(x, fn))
        at === nothing || return at
    end
    return nothing
end

# ── OffloadRecompute: copy args to host after forward, reconstruct on backward
# On GPU this trades device VRAM for host RAM + two one-way transfers: after
# the forward returns, the explicit args are copied to host and the outer
# Zygote tape only holds the CPU snapshot + the target atype. During backward
# the snapshot is copied back to the device and the forward is re-run under
# Zygote to produce the pullback.
#
# On CPU this collapses to `Recompute()` modulo a copy roundtrip.
#
# Caveat: only the explicit `args` are offloaded. Any large tensors captured
# inside `f` as a closure still live on the device. Use `Offload()` (walker-
# based) for broader coverage at the price of closure-reflection complexity.

_offload_to_host(x::AbstractArray{<:Number}) = Array(x)
_offload_to_host(x::AbstractArray) = map(_offload_to_host, x)
_offload_to_host(x::Tuple) = map(_offload_to_host, x)
function _offload_to_host(x)
    T = typeof(x)
    (!isstructtype(T) || isbitstype(T) || isempty(fieldnames(T))) && return x
    children = map(fn -> _offload_to_host(getfield(x, fn)), fieldnames(T))
    try
        return _new_struct(T, children)
    catch
        _reconstruct_fail_count[] += 1
        return x
    end
end

# Atype detection: returns Array / CuArray / ROCArray, or `Array` fallback
# when no array-like is found. Handles AbstractArray{<:Number} directly,
# Tuple/AbstractArray{non-Number} via recursion, runtime structs (CTMEnv,
# VUMPSRuntime, ...) via field walk.
_atype_of(x::AbstractArray{<:Number}) = _arraytype(x)
_atype_of(x::AbstractArray) = isempty(x) ? nothing : _atype_of(first(x))
_atype_of(x::Tuple) = begin
    for a in x
        at = _atype_of(a)
        at === nothing || return at
    end
    return nothing
end
function _atype_of(x)
    T = typeof(x)
    !isstructtype(T) && return nothing
    isbitstype(T)    && return nothing
    for fn in fieldnames(T)
        at = _atype_of(getfield(x, fn))
        at === nothing || return at
    end
    return nothing
end

function _detect_target_atype(args)
    for a in args
        at = _atype_of(a)
        at === nothing || return at
    end
    return Array
end

# Reconstruct an on-device copy from a CPU copy, given the target atype.
# Dispatches on concrete `Array{<:Number}` (not AbstractArray) because after
# `_offload_to_host`, GPU leaves have been replaced with plain `Array`;
# other AbstractArray subtypes that weren't originally on GPU (SubArray,
# Transpose, …) pass through unchanged.
_to_atype(atype, x::Array{<:Number}) = atype(x)
_to_atype(atype, x::AbstractArray) = map(a -> _to_atype(atype, a), x)
_to_atype(atype, x::Tuple) = map(a -> _to_atype(atype, a), x)
function _to_atype(atype, x)
    T = typeof(x)
    (!isstructtype(T) || isbitstype(T) || isempty(fieldnames(T))) && return x
    children = map(fn -> _to_atype(atype, getfield(x, fn)), fieldnames(T))
    try
        return _new_struct(T, children)
    catch
        _reconstruct_fail_count[] += 1
        return x
    end
end

checkpoint(::OffloadRecompute, f, args...; kwargs...) = f(args...; kwargs...)

Zygote.@adjoint function checkpoint(m::OffloadRecompute, f, args...; kwargs...)
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

# ── Offload: walk pullback closure, swap GPU captures for CPU copies ────────
#
# `Zygote._pullback(f, args...)` returns `(y, pb)` where `pb` is a compiler-
# generated closure whose struct fields capture the forward-pass values it
# needs to compute the gradient (e.g. input arrays, intermediate activations).
# By default those captures pin GPU memory until the outer Zygote backward
# finally calls `pb(ȳ)`. For a deep iPEPS gradient (QRCTMRG AD loop + energy
# observable), that held memory can dominate GPU peak usage.
#
# `Offload()` walks `pb`'s captured struct fields reflectively and replaces
# every `AbstractArray` leaf satisfying `_is_gpu(x)` with its `Array(x)` CPU
# copy. The reconstructed `pb_cpu` holds only CPU state. On backward, the
# walker runs in reverse to materialise `pb_gpu` (reload leaves via the
# GPU `atype` constructor bound at forward time), call `pb_gpu(ȳ)`, and
# then let `pb_gpu` go out of scope so the reloaded device copies are freed
# at the end of that backward call.
#
# No forward recomputation on backward — the pullback runs exactly once.
#
# On CPU this is a near-no-op (`_detect_gpu_atype(pb)` returns `nothing`,
# the walker is skipped, `pb` passes through unchanged).
#
# vs `OffloadRecompute()`:
#   - Offload offloads ALL pb captures (args + intermediates + anything
#     Zygote bound into the closure) — broader coverage
#   - OffloadRecompute only offloads the explicit args; closure-captured
#     tensors inside `f` still live on device
#   - Offload has no forward recomputation; OffloadRecompute re-runs the
#     forward under Zygote on backward (2× forward cost)
#
# Caveat: the walker works for compiler-generated closures produced by
# ChainRules-style rrules. If a pullback captures something non-
# reconstructable (a struct with a restrictive inner constructor that
# validates invariants), the walker falls back to the original pb for that
# subtree and increments `_reconstruct_fail_count` — numerical correctness
# preserved, offload skipped for that capture.

checkpoint(::Offload, f, args...; kwargs...) = f(args...; kwargs...)

Zygote.@adjoint function checkpoint(m::Offload, f, args...; kwargs...)
    y, pb = Zygote._pullback((aa...) -> f(aa...; kwargs...), args...)
    # Detect GPU atype + short-circuit CPU runs. If no GPU array is found
    # anywhere in pb's struct tree, skip the walker entirely.
    atype = _detect_gpu_atype(pb)
    pb_cpu = atype === nothing ? pb :
             transform_captured_arrays(pb, _is_gpu, Array)
    return y, function(ȳ)
        # Reload CPU captures to device via the detected atype constructor.
        # Symmetric predicate to forward: match non-GPU numeric arrays.
        pb_gpu = atype === nothing ? pb :
                 transform_captured_arrays(pb_cpu,
                     x -> x isa AbstractArray{<:Number} && !_is_gpu(x),
                     atype)
        inner = pb_gpu(ȳ)
        (nothing, inner...)   # prepend ∂m = nothing
    end
end
