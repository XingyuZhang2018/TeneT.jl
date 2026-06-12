# Deterministic-memory contraction-chain engine (M1).
# Design: docs/2026-06-12-deterministic-contraction-engine-design.md
#
# A Chain declares a left-assoc pairwise sequence as index-label tuples; the
# executor runs TensorOperations' runtime tensorcontract (same cuTENSOR calls
# as @tensor) with every intermediate owned and _free!'d after its last use.
# M1 scope: no per-op conj flags (tuple-M maps only).

struct Chain{N, O}
    ops::NTuple{N, Tuple}    # index labels per operand; ops[1] is the carried tensor
    out::NTuple{O, Symbol}   # output labels of the final intermediate
end

# Labels of intermediate I_k = I_{k-1} ⋆ ops[k+1]: open labels of I_{k-1}
# (not shared with the op) followed by open labels of the op.
function _link_labels(labs::Tuple, op::Tuple)
    keep = filter(l -> !(l in op), labs)
    new  = filter(l -> !(l in labs), op)
    return (keep..., new...)
end

function chain_interlabels(ch::Chain{N}) where {N}
    ils = Vector{Tuple}(undef, N - 1)
    labs = ch.ops[1]
    for k in 2:N
        labs = k == N ? ch.out : _link_labels(labs, ch.ops[k])
        ils[k - 1] = labs
    end
    return Tuple(ils)
end

# FLmap leg5 (tuple-M), operand order (FL, ALd, M1, M2, ALu) — the current
# left-assoc order of the serial kernel and the cannon stage pipeline.
const FLMAP_LEG5_CHAIN = Chain(
    ((:a,:e,:f,:i), (:i,:j,:k,:l), (:e,:j,:g,:b,:p), (:f,:k,:h,:c,:p), (:a,:b,:c,:d)),
    (:d,:g,:h,:l))

"""
    chain_apply(ch, tensors) -> result

Run the chain left-assoc via runtime `tensorcontract`; owned intermediates
are `_free!`d as soon as consumed. `tensors` order matches `ch.ops`.
"""
function chain_apply(ch::Chain{N}, tensors::NTuple{N, Any}) where {N}
    acc  = tensors[1]
    labs = ch.ops[1]
    for k in 2:N
        IC  = k == N ? ch.out : _link_labels(labs, ch.ops[k])
        nxt = tensorcontract(IC, acc, labs, false, tensors[k], ch.ops[k], false)
        k > 2 && _free!(acc)     # acc owned from link k-1; inputs never freed
        acc, labs = nxt, IC
    end
    return acc
end
