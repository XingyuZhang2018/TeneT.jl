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
    FL, ALu, ALd, M = args
    if M isa Tuple && length(M) == 2 && _chainable(M)
        g = chain_backward(FLMAP_LEG5_CHAIN, (FL, ALd, M[1], M[2], ALu), dOut)
        return (g[1], g[5], g[2], (g[3], g[4]))
    elseif M isa AbstractArray && ndims(M) == 5
        g = chain_backward(FLMAP_LEG5_CHAIN_1M, (FL, ALd, M, M, ALu), dOut)
        dM = g[3]; dM .+= g[4]; _free!(g[4])           # slot-sum, in place
        return (g[1], g[5], g[2], dM)
    elseif M isa AbstractArray && ndims(M) == 4
        g = chain_backward(FLMAP_LEG4_CHAIN, (FL, ALd, M, ALu), dOut)
        return (g[1], g[4], g[2], g[3])
    elseif M isa AbstractArray && ndims(M) == 8
        g = chain_backward(FLMAP_LEG8_CHAIN, (FL, ALd, M, ALu), dOut)
        return (g[1], g[4], g[2], g[3])
    end
    return nothing
end
