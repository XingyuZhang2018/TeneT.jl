# M2: map→chain glue. Chain declarations for the 21 basic.jl map methods,
# the engine toggle, and the engine_backward registry used by the
# forloop/parallel rrule reroute. Plan: docs/2026-06-12-chain-engine-m2-plan.md

# Global switch. Default false until M2 parity + the Sofia gate pass (Task 11).
const CHAIN_ENGINE = Ref(false)
set_chain_engine!(b::Bool) = (CHAIN_ENGINE[] = b; b)

# Numeric AbstractArrays (incl. SubArray views): leg aliases also admit
# Vector-of-arrays and StructArray, which the runtime API cannot contract;
# density is a census fact, not checked here.
_chainable(x) = x isa AbstractArray && eltype(x) <: Number
_chainable(t::Tuple) = all(_chainable, t)
_chainable(x, rest...) = _chainable(x) && _chainable(rest...)
use_chain_engine(args...) = CHAIN_ENGINE[] && _chainable(args...)

# Route a map through the engine, mirroring the kernels' inner_etype
# structure: downcast at entry, upcast at exit, identity when unset.
function _chain_map(ch::Chain, tensors::Tuple, inner_etype)
    if inner_etype === nothing || inner_etype == real(eltype(tensors[1]))
        return chain_apply(ch, tensors)
    end
    T_out = eltype(tensors[1])
    return T_out.(chain_apply(ch, map(t -> _downcast_eltype(inner_etype, t), tensors)))
end

# engine_backward(f, args, dOut) -> gradients in f's positional-arg order
# (Tuple-M arg ⇒ tuple gradient), or `nothing` when this (f, args) form has
# no chain — the rrule caller then falls back to Zygote. NEVER uses Zygote.
engine_backward(f, args, dOut) = nothing

# ─── FLmap family (leg4, leg8, leg5 pair via FLMAP_LEG5_CHAIN, leg5 single-M) ─
# Chain tensor order is (FL, ALd, M..., ALu) — the kernels' left-assoc order;
# map arg order is (FL, ALu, ALd, M). tensor_chain pins the @tensor temp
# layouts; FLMAP_LEG5_CHAIN keeps its Part-7 hand-kernel pins (chain_engine.jl).
const FLMAP_LEG4_CHAIN = tensor_chain(((:a,:d,:f), (:f,:g,:h), (:d,:g,:e,:b), (:a,:b,:c)), (:c,:e,:h))
const FLMAP_LEG8_CHAIN = tensor_chain(((:a,:e,:f,:i), (:i,:j,:k,:l), (:e,:f,:j,:k,:g,:h,:b,:c), (:a,:b,:c,:d)), (:d,:g,:h,:l))
const FLMAP_LEG5_CHAIN_1M = conj_variant(FLMAP_LEG5_CHAIN, 4)   # M2 = conj(M1), no materialization

function engine_backward(::typeof(FLmap), args::NTuple{4, Any}, dOut)
    _chainable(args...) || return nothing
    FL, ALu, ALd, M = args
    if M isa Tuple && length(M) == 2
        g = chain_backward(FLMAP_LEG5_CHAIN, (FL, ALd, M[1], M[2], ALu), dOut)
        # g = (dFL, dALd, dM1, dM2, dALu) → map order (dFL, dALu, dALd, dM-tuple)
        return (g[1], g[5], g[2], (g[3], g[4]))
    elseif M isa AbstractArray && ndims(M) == 5
        g = chain_backward(FLMAP_LEG5_CHAIN_1M, (FL, ALd, M, M, ALu), dOut)
        dM = g[3]; dM .+= g[4]; _free!(g[4])           # slot-sum, in place
        # g = (dFL, dALd, dM₁, dM₂, dALu) → map order (dFL, dALu, dALd, dM₁+dM₂)
        return (g[1], g[5], g[2], dM)
    elseif M isa AbstractArray && ndims(M) == 4
        g = chain_backward(FLMAP_LEG4_CHAIN, (FL, ALd, M, ALu), dOut)
        # g = (dFL, dALd, dM, dALu) → map order (dFL, dALu, dALd, dM)
        return (g[1], g[4], g[2], g[3])
    elseif M isa AbstractArray && ndims(M) == 8
        g = chain_backward(FLMAP_LEG8_CHAIN, (FL, ALd, M, ALu), dOut)
        # g = (dFL, dALd, dM, dALu) → map order (dFL, dALu, dALd, dM)
        return (g[1], g[4], g[2], g[3])
    end
    return nothing
end

# ─── FRmap family (leg4, leg8, leg5 pair, leg5 single-M) ────────────────────
# Chain tensor order is (ARd, FR, M..., ARu) — the kernels' @tensor written
# order; map arg order is (FR, ARu, ARd, M). NB: unlike FLmap the carried
# operand (ARd) is NOT the map's first arg, so the gradient permutations
# below differ from FLmap's — see the per-return comments.
const FRMAP_LEG4_CHAIN = tensor_chain(((:f,:g,:h), (:c,:e,:h), (:d,:g,:e,:b), (:a,:b,:c)), (:a,:d,:f))
const FRMAP_LEG5_CHAIN = tensor_chain(((:i,:j,:k,:l), (:d,:g,:h,:l), (:e,:j,:g,:b,:p), (:f,:k,:h,:c,:p), (:a,:b,:c,:d)), (:a,:e,:f,:i))
const FRMAP_LEG8_CHAIN = tensor_chain(((:i,:j,:k,:l), (:d,:g,:h,:l), (:e,:f,:j,:k,:g,:h,:b,:c), (:a,:b,:c,:d)), (:a,:e,:f,:i))
const FRMAP_LEG5_CHAIN_1M = conj_variant(FRMAP_LEG5_CHAIN, 4)   # M2 = conj(M1), no materialization

function engine_backward(::typeof(FRmap), args::NTuple{4, Any}, dOut)
    _chainable(args...) || return nothing
    FR, ARu, ARd, M = args
    if M isa Tuple && length(M) == 2
        g = chain_backward(FRMAP_LEG5_CHAIN, (ARd, FR, M[1], M[2], ARu), dOut)
        # g = (dARd, dFR, dM1, dM2, dARu) → map order (dFR, dARu, dARd, dM-tuple)
        return (g[2], g[5], g[1], (g[3], g[4]))
    elseif M isa AbstractArray && ndims(M) == 5
        g = chain_backward(FRMAP_LEG5_CHAIN_1M, (ARd, FR, M, M, ARu), dOut)
        dM = g[3]; dM .+= g[4]; _free!(g[4])           # slot-sum, in place
        # g = (dARd, dFR, dM₁, dM₂, dARu) → map order (dFR, dARu, dARd, dM₁+dM₂)
        return (g[2], g[5], g[1], dM)
    elseif M isa AbstractArray && ndims(M) == 4
        g = chain_backward(FRMAP_LEG4_CHAIN, (ARd, FR, M, ARu), dOut)
        # g = (dARd, dFR, dM, dARu) → map order (dFR, dARu, dARd, dM)
        return (g[2], g[4], g[1], g[3])
    elseif M isa AbstractArray && ndims(M) == 8
        g = chain_backward(FRMAP_LEG8_CHAIN, (ARd, FR, M, ARu), dOut)
        # g = (dARd, dFR, dM, dARu) → map order (dFR, dARu, dARd, dM)
        return (g[2], g[4], g[1], g[3])
    end
    return nothing
end

# ─── ACmap family (leg4, leg8, leg5 pair, leg5 single-M) ────────────────────
# Chain tensor order is (AC, FR, M..., FL) — the kernels' @tensor written
# order; map arg order is (AC, FL, FR, M). NB: the carried operand (AC) IS the
# map's first arg, but FL/FR are swapped relative to the chain order, so the
# gradient permutations differ from both FLmap's and FRmap's — see the
# per-return comments.
const ACMAP_LEG4_CHAIN = tensor_chain(((:a,:b,:c), (:c,:e,:h), (:d,:g,:e,:b), (:a,:d,:f)), (:f,:g,:h))
const ACMAP_LEG5_CHAIN = tensor_chain(((:a,:b,:c,:d), (:d,:g,:h,:l), (:e,:j,:g,:b,:p), (:f,:k,:h,:c,:p), (:a,:e,:f,:i)), (:i,:j,:k,:l))
const ACMAP_LEG8_CHAIN = tensor_chain(((:a,:b,:c,:d), (:d,:g,:h,:l), (:e,:f,:j,:k,:g,:h,:b,:c), (:a,:e,:f,:i)), (:i,:j,:k,:l))
const ACMAP_LEG5_CHAIN_1M = conj_variant(ACMAP_LEG5_CHAIN, 4)   # M2 = conj(M1), no materialization

function engine_backward(::typeof(ACmap), args::NTuple{4, Any}, dOut)
    _chainable(args...) || return nothing
    AC, FL, FR, M = args
    if M isa Tuple && length(M) == 2
        g = chain_backward(ACMAP_LEG5_CHAIN, (AC, FR, M[1], M[2], FL), dOut)
        # g = (dAC, dFR, dM1, dM2, dFL) → map order (dAC, dFL, dFR, dM-tuple)
        return (g[1], g[5], g[2], (g[3], g[4]))
    elseif M isa AbstractArray && ndims(M) == 5
        g = chain_backward(ACMAP_LEG5_CHAIN_1M, (AC, FR, M, M, FL), dOut)
        dM = g[3]; dM .+= g[4]; _free!(g[4])           # slot-sum, in place
        # g = (dAC, dFR, dM₁, dM₂, dFL) → map order (dAC, dFL, dFR, dM₁+dM₂)
        return (g[1], g[5], g[2], dM)
    elseif M isa AbstractArray && ndims(M) == 4
        g = chain_backward(ACMAP_LEG4_CHAIN, (AC, FR, M, FL), dOut)
        # g = (dAC, dFR, dM, dFL) → map order (dAC, dFL, dFR, dM)
        return (g[1], g[4], g[2], g[3])
    elseif M isa AbstractArray && ndims(M) == 8
        g = chain_backward(ACMAP_LEG8_CHAIN, (AC, FR, M, FL), dOut)
        # g = (dAC, dFR, dM, dFL) → map order (dAC, dFL, dFR, dM)
        return (g[1], g[4], g[2], g[3])
    end
    return nothing
end

# ─── ACdmap family (leg4, leg5 pair, leg5 single-M — NO leg8 method) ────────
# Chain tensor order is (ACd, FR, M..., FL) — the kernels' @tensor written
# order; map arg order is (ACd, FL, FR, M). Same arg layout as ACmap, so the
# gradient permutations below match ACmap's — see the per-return comments.
const ACDMAP_LEG4_CHAIN = tensor_chain(((:f,:g,:h), (:c,:e,:h), (:d,:g,:e,:b), (:a,:d,:f)), (:a,:b,:c))
const ACDMAP_LEG5_CHAIN = tensor_chain(((:i,:j,:k,:l), (:d,:g,:h,:l), (:e,:j,:g,:b,:p), (:f,:k,:h,:c,:p), (:a,:e,:f,:i)), (:a,:b,:c,:d))
const ACDMAP_LEG5_CHAIN_1M = conj_variant(ACDMAP_LEG5_CHAIN, 4)  # M2 = conj(M1), no materialization

function engine_backward(::typeof(ACdmap), args::NTuple{4, Any}, dOut)
    _chainable(args...) || return nothing
    ACd, FL, FR, M = args
    if M isa Tuple && length(M) == 2
        g = chain_backward(ACDMAP_LEG5_CHAIN, (ACd, FR, M[1], M[2], FL), dOut)
        # g = (dACd, dFR, dM1, dM2, dFL) → map order (dACd, dFL, dFR, dM-tuple)
        return (g[1], g[5], g[2], (g[3], g[4]))
    elseif M isa AbstractArray && ndims(M) == 5
        g = chain_backward(ACDMAP_LEG5_CHAIN_1M, (ACd, FR, M, M, FL), dOut)
        dM = g[3]; dM .+= g[4]; _free!(g[4])           # slot-sum, in place
        # g = (dACd, dFR, dM₁, dM₂, dFL) → map order (dACd, dFL, dFR, dM₁+dM₂)
        return (g[1], g[5], g[2], dM)
    elseif M isa AbstractArray && ndims(M) == 4
        g = chain_backward(ACDMAP_LEG4_CHAIN, (ACd, FR, M, FL), dOut)
        # g = (dACd, dFR, dM, dFL) → map order (dACd, dFL, dFR, dM)
        return (g[1], g[4], g[2], g[3])
    end
    return nothing
end

# ─── Cmap (leg3-FL, leg4-FL) ─────────────────────────────────────────────────
# Chain tensor order is (FL, C, FR) — the kernels' @tensor written order; map
# arg order is (C, FL, FR). Cmap gets NO engine_backward entry: the plan's map
# census shows Cmap never goes through forloop/parallel — it is differentiated
# directly in the Cenv loops, where the chain_apply rrule covers the
# engine-ON path. Cmap has no inner_etype kwarg ⇒ its guards call chain_apply
# directly (_chain_map is the inner_etype wrapper).
const CMAP_LEG3_CHAIN = tensor_chain(((:a,:c,:d), (:a,:b), (:b,:c,:e)), (:d,:e))
const CMAP_LEG4_CHAIN = tensor_chain(((:a,:c,:d,:e), (:a,:b), (:b,:c,:d,:f)), (:e,:f))

# ─── Mmap / Mumap / Mdmap (precondition transfer maps) ──────────────────────
# Chain tensor order for Mmap is (AC, FR, FL, ACd) — the kernel's left-assoc
# @tensor order; map arg order is (AC, ACd, FL, FR). NONE of the three M-maps
# gets an engine_backward entry: their only production consumers are
# forloop_sum/parallel_sum (precondition.jl), which have NO rrule — they are
# forward-only in production (census-verified). Zygote-gradability comes from
# the chain_apply rrule (Mmap) and the composed-glue rrules below + in
# autodiff/rules.jl (Mumap/Mdmap). No inner_etype kwarg ⇒ the basic.jl guards
# call chain_apply/_chain_Mumap/_chain_Mdmap directly (Cmap-style).
const MMAP_CHAIN = tensor_chain(((:a,:b,:c), (:c,:e,:h), (:a,:d,:f), (:f,:g,:h)), (:d,:g,:e,:b))

# Mumap/Mdmap are parenthesized TREES (X = AC*FR; Y = (FL*ACd)*Mu|Md;
# out = X*Y), expressed as two composed chains glued by _chain_Mumap /
# _chain_Mdmap. "B-side" = the right operand of the root contraction X*Y;
# @tensor lays those temps out by a different rule than the left-assoc
# carried temps tensor_pinned_inters models. Two kinds of pinned labels are
# at play: an "inner inter" is helper-derivable and frozen by choice (the
# pin just locks in what tensor_pinned_inters would derive), whereas the
# inner chains' `out` (Y) is NOT derivable and must be DECLARED from the
# Task 8 @macroexpand probe. The outer chains' ops[3] equals the inner
# chains' out and their derived single inter is the probe's X temp
# (asserted in test_chain_maps.jl).
const MUMAP_INNER_CHAIN = Chain(((:a,:e,:f,:i), (:i,:j,:k,:l), (:e,:j,:g,:b,:p)), (:a,:b,:g,:l,:f,:k,:p), ((:a,:f,:k,:l,:e,:j),))
const MUMAP_OUTER_CHAIN = tensor_chain(((:a,:b,:c,:d), (:d,:g,:h,:l), (:a,:b,:g,:l,:f,:k,:p)), (:f,:k,:h,:c,:p))
const MDMAP_INNER_CHAIN = Chain(((:a,:e,:f,:i), (:i,:j,:k,:l), (:f,:k,:h,:c,:p)), (:a,:c,:h,:l,:e,:j,:p), ((:a,:e,:j,:l,:f,:k),))
const MDMAP_OUTER_CHAIN = tensor_chain(((:a,:b,:c,:d), (:d,:g,:h,:l), (:a,:c,:h,:l,:e,:j,:p)), (:e,:j,:g,:b,:p))

function _chain_Mumap(AC, ACd, FL, FR, Mu)
    Y = chain_apply(MUMAP_INNER_CHAIN, (FL, ACd, Mu))
    out = chain_apply(MUMAP_OUTER_CHAIN, (AC, FR, Y))
    _free!(Y)
    return out
end
function _chain_Mdmap(AC, ACd, FL, FR, Md)
    Y = chain_apply(MDMAP_INNER_CHAIN, (FL, ACd, Md))
    out = chain_apply(MDMAP_OUTER_CHAIN, (AC, FR, Y))
    _free!(Y)
    return out
end

# ─── Corner maps LDmap/DRmap/RUmap/LUmap (chains #18-21) ────────────────────
# INTEGER labels, transcribed verbatim from the basic.jl kernels (the engine
# is label-type-agnostic; keeping the kernels' integers makes the transcription
# diffable against the @tensor bodies). Chain tensor order is the @tensor
# written order — for RUmap that is (U, R, M1, M2) while the map args are
# (R, U, M1, M2); the other three match their arg order. These maps are DEAD
# in src per the plan census (their only caller, oc_Q_22_getQ_CBE, is
# commented out) and never go through forloop/parallel — NO engine_backward
# entries; Zygote-gradability comes from the chain_apply rrule. No
# inner_etype kwarg ⇒ the basic.jl guards call chain_apply directly
# (Cmap-style). The parity tests ARE the spec (test_chain_maps.jl).
const LDMAP_CHAIN = tensor_chain(((1,5,6,9), (9,10,11,12), (5,10,7,2,13), (6,11,8,3,13)), (1,2,3,7,8,12))
const DRMAP_CHAIN = tensor_chain(((9,10,11,12), (4,7,8,12), (5,10,7,2,13), (6,11,8,3,13)), (9,5,6,2,3,4))
const RUMAP_CHAIN = tensor_chain(((1,2,3,4), (4,7,8,12), (5,10,7,2,13), (6,11,8,3,13)), (12,10,11,5,6,1))
const LUMAP_CHAIN = tensor_chain(((1,5,6,9), (1,2,3,4), (5,10,7,2,13), (6,11,8,3,13)), (9,10,11,7,8,4))
const LDMAP_CHAIN_1M = conj_variant(LDMAP_CHAIN, 4)   # M2 = conj(M1), no materialization
const DRMAP_CHAIN_1M = conj_variant(DRMAP_CHAIN, 4)
const RUMAP_CHAIN_1M = conj_variant(RUMAP_CHAIN, 4)
const LUMAP_CHAIN_1M = conj_variant(LUMAP_CHAIN, 4)
