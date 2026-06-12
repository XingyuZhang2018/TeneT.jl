# M2: map→chain glue. Chain declarations for the 21 basic.jl map methods,
# the engine toggle, and the engine_backward registry used by the
# forloop/parallel rrule reroute. Plan: docs/2026-06-12-chain-engine-m2-plan.md

# Global switch. Default false until M2 parity + the Sofia gate pass (Task 11).
const CHAIN_ENGINE = Ref(false)
set_chain_engine!(b::Bool) = (CHAIN_ENGINE[] = b; b)

# Dense numeric arrays (incl. SubArray views) only: leg aliases also admit
# Vector-of-arrays and StructArray, which the runtime API cannot contract.
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
