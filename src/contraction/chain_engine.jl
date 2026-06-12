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

"""
    chain_apply_from1(ch, H, tensors_tail) -> result

Run links 2..N of the chain starting from a provided first intermediate
(H = I₁, e.g. ring-accumulated). `tensors_tail` are ops 3..N's tensors.
Frees the internally created intermediates, never H or the inputs.
"""
function chain_apply_from1(ch::Chain{N}, H, tensors_tail::Tuple) where {N}
    @assert length(tensors_tail) == N - 2
    acc  = H
    labs = _link_labels(ch.ops[1], ch.ops[2])
    for k in 3:N
        IC  = k == N ? ch.out : _link_labels(labs, ch.ops[k])
        nxt = tensorcontract(IC, acc, labs, false, tensors_tail[k - 2], ch.ops[k], false)
        k > 3 && _free!(acc)     # acc owned from link k-1; at k == 3 acc === H (caller's)
        acc, labs = nxt, IC
    end
    return acc
end

# Index2Tuple bookkeeping for the mutating tensorcontract! API (verified in
# TensorOperations 5.5.1, interface.jl:148):
#   tensorcontract!(C, A, pA, conjA, B, pB, conjB, pAB, α, β)
# pA = (open-A positions, contracted positions in A); pB = (contracted
# positions in B in the SAME order as pA's, open-B positions); pAB =
# (permutation of (openA..., openB...) onto IC, ()). Mirrors the convention of
# TensorOperations.contract_indices (contracted order = appearance in labsA).
function _index2tuples(labsA::Tuple, labsB::Tuple, IC::Tuple)
    contracted = filter(l -> l in labsB, labsA)
    openA      = filter(l -> !(l in labsB), labsA)
    openB      = filter(l -> !(l in labsA), labsB)
    oindA = map(l -> findfirst(isequal(l), labsA), openA)
    cindA = map(l -> findfirst(isequal(l), labsA), contracted)
    cindB = map(l -> findfirst(isequal(l), labsB), contracted)
    oindB = map(l -> findfirst(isequal(l), labsB), openB)
    openAB = (openA..., openB...)
    indCinoAB = map(l -> findfirst(isequal(l), openAB), IC)
    return (oindA, cindA), (cindB, oindB), (indCinoAB, ())
end

"""
    chain_link1_add!(H, ch, A, B) -> H

Accumulate the chain's first link into a caller buffer: `H += A ⋆ B` with the
labels of `ch.ops[1]`/`ch.ops[2]` (α=1, β=1 in the mutating `tensorcontract!`).
Building block for the cannon ring, mirroring `_cannon_stage1_add!`.
"""
function chain_link1_add!(H, ch::Chain, A, B)
    IH = _link_labels(ch.ops[1], ch.ops[2])
    pA, pB, pAB = _index2tuples(ch.ops[1], ch.ops[2], IH)
    tensorcontract!(H, A, pA, false, B, pB, false, pAB, 1, 1)
    return H
end

"""
    chain_link1_back(ch, dH, A, B) -> (dA, dB)

Adjoint of the chain's first link (I₁ = A ⋆ B): the generic pairwise adjoints
`dA = dH ⋆ conj(B)`, `dB = conj(A) ⋆ dH` with the labels of
`ch.ops[1]`/`ch.ops[2]`. Allocates both gradients; never frees its inputs.
Counterpart of `chain_link1_add!` for the cannon ring's per-(chunk, block)
dFL/dALd contributions.
"""
function chain_link1_back(ch::Chain, dH, A, B)
    IH = _link_labels(ch.ops[1], ch.ops[2])
    dA = tensorcontract(ch.ops[1], dH, IH, false, B, ch.ops[2], true)
    dB = tensorcontract(ch.ops[2], A, ch.ops[1], true, dH, IH, false)
    return dA, dB
end

"""
    chain_backward(ch, tensors, dOut) -> NTuple{N} of gradients

Recompute-style backward: rebuilds the intermediates I_1..I_{N-2} (the final
output I_{N-1} is NOT an adjoint operand and is not recomputed at all — same
as the cannon rrule, which recomputes H/T/G but never P), then walks the
reversed chain with the generic pairwise adjoints

    dB = conj(A) ⋆ dC,    dA = dC ⋆ conj(B)

(conj on the non-cotangent operand, matching the hand kernels). Every owned
array is freed right after its last consumer; caller-owned arrays (`tensors`,
`dOut`) are never freed. Gradients are returned in `ch.ops` order.
"""
function chain_backward(ch::Chain{N}, tensors::NTuple{N, Any}, dOut) where {N}
    # 1. forward recompute, KEEPING I_1..I_{N-2} (adjoint operands).
    inters = Vector{Any}(undef, N - 2)
    labs   = Vector{Tuple}(undef, N - 2)
    acc, l = tensors[1], ch.ops[1]
    for k in 2:(N - 1)
        IC  = _link_labels(l, ch.ops[k])
        acc = tensorcontract(IC, acc, l, false, tensors[k], ch.ops[k], false)
        inters[k - 1], labs[k - 1] = acc, IC
        l = IC
    end
    # 2. reversed walk k = N..2; dC is the cotangent of I_{k-1} entering step k.
    grads = Vector{Any}(undef, N)
    dC, lC = dOut, ch.out
    for k in N:-1:2
        Iprev  = k == 2 ? tensors[1] : inters[k - 2]
        lIprev = k == 2 ? ch.ops[1]  : labs[k - 2]
        # d(op_k) = conj(I_{k-1}) ⋆ dC
        grads[k] = tensorcontract(ch.ops[k], Iprev, lIprev, true, dC, lC, false)
        # dI_{k-1} = dC ⋆ conj(op_k)
        dprev = tensorcontract(lIprev, dC, lC, false, tensors[k], ch.ops[k], true)
        # Frees AFTER both contractions of this step (last consumers):
        k < N && _free!(dC)      # owned cotangent from step k+1; at k == N dC === dOut (caller's)
        k > 2 && _free!(Iprev)   # owned inters[k-2]; at k == 2 Iprev === tensors[1] (caller's)
        dC, lC = dprev, lIprev
    end
    grads[1] = dC                # dI_1-walked-to-d(op_1); owned, returned to caller
    return Tuple(grads)
end

"""
    chain_backward_from1(ch, H, tensors_tail, dOut) -> (dH, dtail...)

Backward of links 2..N only: cotangents for H (= I₁, e.g. ring-accumulated)
and the tail operands (ops 3..N, `tensors_tail` order). Recompute-style for
the internal intermediates: I₂..I_{N-2} are rebuilt from H (the final output
is never an adjoint operand), then the reversed walk runs the same generic
pairwise adjoints as `chain_backward` but stops at k = 3, returning the
carried cotangent dI₁ = dH first. Owned intermediates and cotangents are
freed after their last consumer; H, `tensors_tail`, `dOut` are caller-owned
and never freed.
"""
function chain_backward_from1(ch::Chain{N}, H, tensors_tail::Tuple, dOut) where {N}
    @assert length(tensors_tail) == N - 2
    # 1. forward recompute of I_2..I_{N-2}, starting from I_1 = H (provided).
    inters = Vector{Any}(undef, N - 2)
    labs   = Vector{Tuple}(undef, N - 2)
    inters[1], labs[1] = H, _link_labels(ch.ops[1], ch.ops[2])
    for k in 3:(N - 1)
        IC = _link_labels(labs[k - 2], ch.ops[k])
        inters[k - 1] = tensorcontract(IC, inters[k - 2], labs[k - 2], false,
                                       tensors_tail[k - 2], ch.ops[k], false)
        labs[k - 1] = IC
    end
    # 2. reversed walk k = N..3; dC is the cotangent of I_{k-1} entering step k.
    grads = Vector{Any}(undef, N - 2)    # grads[k-2] = d(op_k), tail order
    dC, lC = dOut, ch.out
    for k in N:-1:3
        Iprev, lIprev = inters[k - 2], labs[k - 2]
        # d(op_k) = conj(I_{k-1}) ⋆ dC
        grads[k - 2] = tensorcontract(ch.ops[k], Iprev, lIprev, true, dC, lC, false)
        # dI_{k-1} = dC ⋆ conj(op_k)
        dprev = tensorcontract(lIprev, dC, lC, false, tensors_tail[k - 2], ch.ops[k], true)
        # Frees AFTER both contractions of this step (last consumers):
        k < N && _free!(dC)      # owned cotangent from step k+1; at k == N dC === dOut (caller's)
        k > 3 && _free!(Iprev)   # owned inters[k-2]; at k == 3 Iprev === H (caller's)
        dC, lC = dprev, lIprev
    end
    return (dC, grads...)        # dC == dH after the k == 3 step; owned
end
