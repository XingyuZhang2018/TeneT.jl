"""
two-site contraction

```
                                            a ────┬──── d
a ────┬──d ──┬──── g                        │     bc    │
│     bc     ef    │                        ├─ ef─┼─gh ─┤  u
├─ hi─┼─ jk ─┼─lm ─┤                        i     jk    l
│     op     rs    │                        ├─ mn─┼─op ─┤  v
n ────┴──q ──┴──── t                        │     rs    │
      u      v                              q ────┴──── t
```
"""
function oc_12(FLo, ACu, A1u, A1d, ACd, FRo, ARu, A2u, A2d, ARd; forloop_iter, ifparallel)
    l = FLmap_parallel(FLo, ACu, ACd, (A1u, A1d); forloop_iter, ifparallel)
    l = FLmap_parallel(l, ARu, ARd, (A2u, A2d); forloop_iter, ifparallel)
    return dot(conj(l), FRo)
end

function oc_21(ACu, FLu, A1u, A1d, FRu, FLo, A2u, A2d, FRo, ACd; forloop_iter, ifparallel)
    u = ACmap_parallel(ACu, FLu, FRu, (A1u, A1d); forloop_iter, ifparallel)
    u = ACmap_parallel(u, FLo, FRo, (A2u, A2d); forloop_iter, ifparallel)
    return dot(conj(u), ACd)
end

# ============================================================================
# Two-site normalization and observable contractions (leg4 dispatch)
# ============================================================================

function contract_n_12(FLo, ACu, A1, ACd, FRo, ARu, A2, ARd; forloop_iter, ifparallel)
    return oc_12(FLo, ACu, A1, conj(A1), ACd, FRo, ARu, A2, conj(A2), ARd; forloop_iter, ifparallel)
end

function contract_o_12(FLo, ACu, A1, ACd, FRo, ARu, A2, ARd, O1, O2; forloop_iter, ifparallel)
    @tensor A1u[a,b,c,d,f] := A1[a,b,c,d,e] * O1[e,f]
    @tensor A2u[a,b,c,d,f] := A2[a,b,c,d,e] * O2[e,f]
    return oc_12(FLo, ACu, A1u, conj(A1), ACd, FRo, ARu, A2u, conj(A2), ARd; forloop_iter, ifparallel)
end

function contract_n_21(ACu, FLu, A1, FRu, FLo, A2, FRo, ACd; forloop_iter, ifparallel)
    return oc_21(ACu, FLu, A1, conj(A1), FRu, FLo, A2, conj(A2), FRo, ACd; forloop_iter, ifparallel)
end

function contract_o_21(ACu, FLu, A1, FRu, FLo, A2, FRo, ACd, O1, O2; forloop_iter, ifparallel)
    @tensor A1u[a,b,c,d,f] := A1[a,b,c,d,e] * O1[e,f]
    @tensor A2u[a,b,c,d,f] := A2[a,b,c,d,e] * O2[e,f]
    return oc_21(ACu, FLu, A1u, conj(A1), FRu, FLo, A2u, conj(A2), FRo, ACd; forloop_iter, ifparallel)
end

# ============================================================================
# Single-site contraction
# ============================================================================
"""
one-site contraction

```
a ────┬──── d
│     bc    │
├─ ef─┼─gh ─┤
│     jk    │
i ────┴──── l

```
"""
function oc_11(FLo, ACu, Au, Ad, ACd, FRo; forloop_iter, ifparallel)
    l = FLmap_parallel(FLo, ACu, ACd, (Au, Ad); forloop_iter, ifparallel)
    return dot(conj(l), FRo)
end

function contract_n_11(FLo, ACu, A, ACd, FRo; forloop_iter, ifparallel)
    return oc_11(FLo, ACu, A, conj(A), ACd, FRo; forloop_iter, ifparallel)
end

function contract_o_11(FLo, ACu, A, ACd, FRo, O; forloop_iter, ifparallel)
    @tensor AO[a,b,c,d,f] := A[a,b,c,d,e] * O[e,f]
    return oc_11(FLo, ACu, AO, conj(A), ACd, FRo; forloop_iter, ifparallel)
end

# ============================================================================
# Diagonal (next-nearest-neighbour) contractions — 2x2 corner
# ============================================================================

"""
    next near neighbour contraction for 2 site
```

a ────┬──c     c──┬──── a
│     bl          bl    │
├─ dm─┼─ eo    eo─┼─dm ─┤
f     gn          hq    i

f     gn          hq    i
├─ dm─┼─ jr   jr ─┼─dm ─┤
│     bl          bl    │
a ────┴──k     k──┴──── a
```

"""
function oc_Q_22(Q, FLu, FLo, ACu, ACd, FRu, FRo, ARu, ARd, Au11, Ad11, Au12, Ad12, Au21, Ad21, Au22, Ad22; forloop_iter, ifparallel)
    Q = FLmap_parallel(FLo, Q, ACd, (Au21, Ad21); forloop_iter, ifparallel)
    Q = ACdmap_parallel(ARd, Q, FRo, (Au22, Ad22); forloop_iter, ifparallel)
    Q = FRmap_parallel(FRu, ARu, Q, (Au12, Ad12); forloop_iter, ifparallel)
    Q = ACmap_parallel(ACu, FLu, Q, (Au11, Ad11); forloop_iter, ifparallel)

    return Q
end

function oc_Q_22_getQ_CBE(FLu, FLo, ACu, ACd, FRu, FRo, ARu, ARd, Au11, Ad11, Au12, Ad12, Au21, Ad21, Au22, Ad22; forloop_iter, ifparallel)
    χ, D = size(FLu)[[1,2]]
    Dp = ceil(Int, χ/D^2)

    LD = LDmap(FLo, ACd, Au21, Ad21)
    _, R = qr_for_ad(_to_front(LD))
    @tensor ARd[5,2,3,4] := R[5,1] * ARd[1,2,3,4]
    DR = DRmap(ARd, FRo, Au22, Ad22)
    _, R = qr_for_ad(_to_front(DR))
    @tensor FRu[1,2,3,5] := FRu[1,2,3,4] * R[5,4]
    RU = RUmap(FRu, ARu, Au12, Ad12)
    _, R = qr_for_ad(_to_front(RU))
    @tensor ACu[1,2,3,5] := ACu[1,2,3,4] * R[5,4]

    # method 1: SVD
    LU = LUmap(FLu, ACu, Au11, Ad11)
    F = svd(_to_front(LU))
    L = F.U[:,1:Dp] * Diagonal(F.S[1:Dp]) 
    Q,  = qr_for_ad(reshape(L, χ*D^2, Dp*D^2))
    Q = reshape(Q, χ, D, D, Dp*D^2)
    return Q

    # method 2: RSVD
    # Q = Zygote.@ignore _arraytype(FLu)(randn(eltype(FLu), χ,D,D,χ))
    # ACu = ACmap_parallel(ACu, FLu, Q, (Au11, Ad11); forloop_iter, ifparallel)
    # Q, = qr_for_ad(_to_front(ACu))
    # Q = reshape(Q, χ, D, D, χ)
    # return Q
end

function oc_22(FLu, FLo, ACu, ACd, FRu, FRo, ARu, ARd, Au11, Ad11, Au12, Ad12, Au21, Ad21, Au22, Ad22; forloop_iter, ifparallel)
    χ = size(FLu ,1)
    D1 = size(Au21, 4)
    D2 = size(Ad21, 4)
    # Q = oc_Q_22_getQ_CBE(FLu, FLo, ACu, ACd, FRu, FRo, ARu, ARd, Au11, Ad11, Au12, Ad12, Au21, Ad21, Au22, Ad22; forloop_iter, ifparallel)
    Q = Zygote.@ignore _arraytype(FLu)(randn(eltype(FLu), χ,D1,D2,χ))
    Q = oc_Q_22(Q, FLu, FLo, ACu, ACd, FRu, FRo, ARu, ARd, Au11, Ad11, Au12, Ad12, Au21, Ad21, Au22, Ad22; forloop_iter, ifparallel)
    Q, _ = qrpos(reshape(Q, χ*D1*D2, χ))
    Q = reshape(Q, χ,D1,D2,χ)

    QQ = oc_Q_22(Q, FLu, FLo, ACu, ACd, FRu, FRo, ARu, ARd, Au11, Ad11, Au12, Ad12, Au21, Ad21, Au22, Ad22; forloop_iter, ifparallel)
    return dot(Q, QQ)
end

function contract_n_22(FLu, FLo, ACu, ACd, FRu, FRo, ARu, ARd, A11, A12, A21, A22; forloop_iter, ifparallel)
    return oc_22(FLu, FLo, ACu, ACd, FRu, FRo, ARu, ARd, A11, conj(A11), A12, conj(A12), A21, conj(A21), A22, conj(A22); forloop_iter, ifparallel)
end

function contract_o_22_1(FLu, FLo, ACu, ACd, FRu, FRo, ARu, ARd, A11, A12, A21, A22, O1, O2; forloop_iter, ifparallel)
    @tensor Au11[a,b,c,d,f] := A11[a,b,c,d,e] * O1[e,f]
    @tensor Au22[a,b,c,d,f] := A22[a,b,c,d,e] * O2[e,f]
    return oc_22(FLu, FLo, ACu, ACd, FRu, FRo, ARu, ARd, Au11, conj(A11), A12, conj(A12), A21, conj(A21), Au22, conj(A22); forloop_iter, ifparallel)
end

function contract_o_22_2(FLu, FLo, ACu, ACd, FRu, FRo, ARu, ARd, A11, A12, A21, A22, O1, O2; forloop_iter, ifparallel)
    @tensor Au12[a,b,c,d,f] := A12[a,b,c,d,e] * O1[e,f]
    @tensor Au21[a,b,c,d,f] := A21[a,b,c,d,e] * O2[e,f]
    return oc_22(FLu, FLo, ACu, ACd, FRu, FRo, ARu, ARd, A11, conj(A11), Au12, conj(A12), Au21, conj(A21), A22, conj(A22); forloop_iter, ifparallel)
end

# ============================================================================
# Three-site contractions (J1J2J3)
# ============================================================================

function oc_13(FLo, ACu, ACd, FRo, ARu1, ARd1, ARu2, ARd2, A1u, A1d, A2u, A2d, A3u, A3d; forloop_iter, ifparallel)
    l = FLmap_parallel(FLo, ACu, ACd, (A1u, A1d); forloop_iter, ifparallel)
    l = FLmap_parallel(l, ARu1, ARd1, (A2u, A2d); forloop_iter, ifparallel)
    r = FRmap_parallel(FRo, ARu2, ARd2, (A3u, A3d); forloop_iter, ifparallel)
    return dot(conj(l), r)
end

function contract_n_13(FLo, ACu, ACd, FRo, ARu1, ARd1, ARu2, ARd2, A1, A2, A3; forloop_iter, ifparallel)
    return oc_13(FLo, ACu, ACd, FRo, ARu1, ARd1, ARu2, ARd2, A1, conj(A1), A2, conj(A2), A3, conj(A3); forloop_iter, ifparallel)
end

function contract_o_13(FLo, ACu, ACd, FRo, ARu1, ARd1, ARu2, ARd2, A1, A2, A3, O1, O2; forloop_iter, ifparallel)
    @tensor A1u[a,b,c,d,f] := A1[a,b,c,d,e] * O1[e,f]
    @tensor A3u[a,b,c,d,f] := A3[a,b,c,d,e] * O2[e,f]
    return oc_13(FLo, ACu, ACd, FRo, ARu1, ARd1, ARu2, ARd2, A1u, conj(A1), A2, conj(A2), A3u, conj(A3); forloop_iter, ifparallel)
end

function oc_31(ACu, ACd, FLu1, FRu1, FLu2, FRu2, FLo, FRo, A1u, A1d, A2u, A2d, A3u, A3d; forloop_iter, ifparallel)
    u = ACmap_parallel(ACu, FLu1, FRu1, (A1u, A1d); forloop_iter, ifparallel)
    u = ACmap_parallel(u, FLu2, FRu2, (A2u, A2d); forloop_iter, ifparallel)
    u = ACmap_parallel(u, FLo, FRo, (A3u, A3d); forloop_iter, ifparallel)
    return dot(conj(u), ACd)
end

function contract_n_31(ACu, ACd, FLu1, FRu1, FLu2, FRu2, FLo, FRo, A1, A2, A3; forloop_iter, ifparallel)
    return oc_31(ACu, ACd, FLu1, FRu1, FLu2, FRu2, FLo, FRo, A1, conj(A1), A2, conj(A2), A3, conj(A3); forloop_iter, ifparallel)
end

function contract_o_31(ACu, ACd, FLu1, FRu1, FLu2, FRu2, FLo, FRo, A1, A2, A3, O1, O2; forloop_iter, ifparallel)
    @tensor A1u[a,b,c,d,f] := A1[a,b,c,d,e] * O1[e,f]
    @tensor A3u[a,b,c,d,f] := A3[a,b,c,d,e] * O2[e,f]
    return oc_31(ACu, ACd, FLu1, FRu1, FLu2, FRu2, FLo, FRo, A1u, conj(A1), A2, conj(A2), A3u, conj(A3); forloop_iter, ifparallel)
end

# ============================================================================
# 2x3 contractions 
# ============================================================================
function oc_Q_23(Q, FLu, FLo, ACu, ACd, FRu, FRo, ARu1, ARd1, ARu2, ARd2, Au11, Ad11, Au12, Ad12, Au13, Ad13, Au21, Ad21, Au22, Ad22, Au23, Ad23; ifparallel, forloop_iter)
    χ = size(Q, 1)
    Iχ = Zygote.@ignore reshape(_arraytype(FLu){eltype(FLu)}(I(χ)), χ, 1,1, χ)

    Q = FLmap_parallel(FLo, Q, ACd, (Au21, Ad21); ifparallel, forloop_iter)
    Q = FLmap_parallel(Q, Iχ, ARd1, (Au22, Ad22); ifparallel, forloop_iter)
    Q = ACdmap_parallel(ARd2, Q, FRo, (Au23, Ad23); ifparallel, forloop_iter)
    Q = FRmap_parallel(FRu, ARu2, Q, (Au13, Ad13); ifparallel, forloop_iter)
    Q = FRmap_parallel(Q, ARu1, Iχ, (Au12, Ad12); ifparallel, forloop_iter)
    Q = ACmap_parallel(ACu, FLu, Q, (Au11, Ad11); ifparallel, forloop_iter)

    return Q
end

function oc_23(FLu, FLo, ACu, ACd, FRu, FRo, ARu1, ARd1, ARu2, ARd2, Au11, Ad11, Au12, Ad12, Au13, Ad13, Au21, Ad21, Au22, Ad22, Au23, Ad23; ifparallel, forloop_iter)
    χ = size(FLu ,1)
    D1 = size(Au21, 4)
    D2 = size(Ad21, 4)
    Q = Zygote.@ignore _arraytype(FLu)(randn(eltype(FLu), χ,D1,D2,χ))
    Q = oc_Q_23(Q, FLu, FLo, ACu, ACd, FRu, FRo, ARu1, ARd1, ARu2, ARd2, Au11, Ad11, Au12, Ad12, Au13, Ad13, Au21, Ad21, Au22, Ad22, Au23, Ad23; ifparallel, forloop_iter)
    Q, _ = TeneT.qrpos(reshape(Q, χ*D1*D2, χ))
    Q = reshape(Q, χ,D1,D2,χ)

    QQ = oc_Q_23(Q, FLu, FLo, ACu, ACd, FRu, FRo, ARu1, ARd1, ARu2, ARd2, Au11, Ad11, Au12, Ad12, Au13, Ad13, Au21, Ad21, Au22, Ad22, Au23, Ad23; ifparallel, forloop_iter)
    return dot(Q, QQ)
end

function contract_n_23(FLu, FLo, ACu, ACd, FRu, FRo, ARu1, ARd1, ARu2, ARd2, A11, A12, A13, A21, A22, A23; ifparallel, forloop_iter)
    return oc_23(FLu, FLo, ACu, ACd, FRu, FRo, ARu1, ARd1, ARu2, ARd2, A11, conj(A11), A12, conj(A12), A13, conj(A13), A21, conj(A21), A22, conj(A22), A23, conj(A23); ifparallel, forloop_iter)
end

function contract_o_23(FLu, FLo, ACu, ACd, FRu, FRo, ARu1, ARd1, ARu2, ARd2, A11, A12, A13, A21, A22, A23, O11, O12, O13, O21,O22, O23; ifparallel, forloop_iter)
    @tensor Au11[a,b,c,d,f] := A11[a,b,c,d,e] * O11[e,f]
    Ad11 = conj(A11)
    @tensor Au12[a,b,c,d,f] := A12[a,b,c,d,e] * O12[e,f]
    Ad12 = conj(A12)
    @tensor Au13[a,b,c,d,f] := A13[a,b,c,d,e] * O13[e,f]
    Ad13 = conj(A13)
    @tensor Au21[a,b,c,d,f] := A21[a,b,c,d,e] * O21[e,f]
    Ad21 = conj(A21)
    @tensor Au22[a,b,c,d,f] := A22[a,b,c,d,e] * O22[e,f]
    Ad22 = conj(A22)
    @tensor Au23[a,b,c,d,f] := A23[a,b,c,d,e] * O23[e,f]
    Ad23 = conj(A23)
    return oc_23(FLu, FLo, ACu, ACd, FRu, FRo, ARu1, ARd1, ARu2, ARd2, Au11, Ad11, Au12, Ad12, Au13, Ad13, Au21, Ad21, Au22, Ad22, Au23, Ad23; ifparallel, forloop_iter)
end
