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
function oc_H_leg4(FLo, ACu, A1u, A1d, ACd, FRo, ARu, A2u, A2d, ARd; forloop_iter, ifparallel)
    l = FLmap_parallel(FLo, ACu, ACd, (A1u, A1d); forloop_iter, ifparallel)
    r = FRmap_parallel(FRo, ARu, ARd, (A2u, A2d); forloop_iter, ifparallel)
    return dot(conj(l), r)
end

function oc_V_leg4(ACu, FLu, A1u, A1d, FRu, FLo, A2u, A2d, FRo, ACd; forloop_iter, ifparallel)
    u = ACmap_parallel(ACu, FLu, FRu, (A1u, A1d); forloop_iter, ifparallel)
    u = ACmap_parallel(u, FLo, FRo, (A2u, A2d); forloop_iter, ifparallel)
    return dot(conj(u), ACd)
end

# ============================================================================
# Two-site normalization and observable contractions (leg4 dispatch)
# ============================================================================

function contract_n2_H(FLo::leg4, ACu, A1, ACd, FRo, ARu, A2, ARd; forloop_iter, ifparallel)
    return oc_H_leg4(FLo, ACu, A1, conj(A1), ACd, FRo, ARu, A2, conj(A2), ARd; forloop_iter, ifparallel)
end

function contract_o2_H(FLo::leg4, ACu, A1, ACd, FRo, ARu, A2, ARd, O1, O2; forloop_iter, ifparallel)
    D1,D2,D3,D4,d = size(A1)
    Dh = size(O1, 3)
    @tensor A1u_tmp[a,b,c,i,d,f] := A1[a,b,c,d,e] * O1[e,f,i]
    A1u = reshape(A1u_tmp, D1,D2,D3*Dh,D4,d)
    D1,D2,D3,D4,d = size(A2)
    @tensor A2u_tmp[a,i,b,c,d,f] := A2[a,b,c,d,e] * O2[i,e,f]
    A2u = reshape(A2u_tmp, D1*Dh,D2,D3,D4,d)
    return oc_H_leg4(FLo, ACu, A1u, conj(A1), ACd, FRo, ARu, A2u, conj(A2), ARd; forloop_iter, ifparallel)
end

function contract_n2_V(ACu::leg4, FLu, A1, FRu, FLo, A2, FRo, ACd; forloop_iter, ifparallel)
    return oc_V_leg4(ACu, FLu, A1, conj(A1), FRu, FLo, A2, conj(A2), FRo, ACd; forloop_iter, ifparallel)
end

function contract_o2_V(ACu::leg4, FLu, A1, FRu, FLo, A2, FRo, ACd, O1, O2; forloop_iter, ifparallel)
    D1,D2,D3,D4,d = size(A1)
    Dh = size(O1, 3)
    @tensor A1u_tmp[a,b,i,c,d,f] := A1[a,b,c,d,e] * O1[e,f,i]
    A1u = reshape(A1u_tmp, D1,D2*Dh,D3,D4,d)
    D1,D2,D3,D4,d = size(A2)
    @tensor A2u_tmp[a,b,c,d,i,f] := A2[a,b,c,d,e] * O2[i,e,f]
    A2u = reshape(A2u_tmp, D1,D2,D3,D4*Dh,d)
    return oc_V_leg4(ACu, FLu, A1u, conj(A1), FRu, FLo, A2u, conj(A2), FRo, ACd; forloop_iter, ifparallel)
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
function oc1_leg4(FLo, ACu, Au, Ad, ACd, FRo; forloop_iter, ifparallel)
    l = FLmap_parallel(FLo, ACu, ACd, (Au, Ad); forloop_iter, ifparallel)
    return dot(conj(l), FRo)
end

function contract_n1(FLo::leg4, ACu, A, ACd, FRo; forloop_iter, ifparallel)
    return oc1_leg4(FLo, ACu, A, conj(A), ACd, FRo; forloop_iter, ifparallel)
end

function contract_o1(FLo::leg4, ACu, A, ACd, FRo, O; forloop_iter, ifparallel)
    @tensor AO[a,b,c,d,f] := A[a,b,c,d,e] * O[e,f]
    return oc1_leg4(FLo, ACu, AO, conj(A), ACd, FRo; forloop_iter, ifparallel)
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
function oc_Q_4_corner(Q, FLu, FLo, ACu, ACd, FRu, FRo, ARu, ARd, Au11, Ad11, Au12, Ad12, Au21, Ad21, Au22, Ad22; forloop_iter, ifparallel)
    Q = FLmap_parallel(FLo, Q, ACd, (Au21, Ad21); forloop_iter, ifparallel)
    Q = ACdmap_parallel(ARd, Q, FRo, (Au22, Ad22); forloop_iter, ifparallel)
    Q = FRmap_parallel(FRu, ARu, Q, (Au12, Ad12); forloop_iter, ifparallel)
    Q = ACmap_parallel(ACu, FLu, Q, (Au11, Ad11); forloop_iter, ifparallel)

    return Q
end

function oc_D_leg4(FLu, FLo, ACu, ACd, FRu, FRo, ARu, ARd, Au11, Ad11, Au12, Ad12, Au21, Ad21, Au22, Ad22; forloop_iter, ifparallel)
    χ = size(FLu ,1)
    D1 = size(Au21, 4)
    D2 = size(Ad21, 4)
    Q = Zygote.@ignore _arraytype(FLu)(randn(eltype(FLu), χ,D1,D2,χ))
    Q = oc_Q_4_corner(Q, FLu, FLo, ACu, ACd, FRu, FRo, ARu, ARd, Au11, Ad11, Au12, Ad12, Au21, Ad21, Au22, Ad22; forloop_iter, ifparallel)
    Q, _ = qrpos(reshape(Q, χ*D1*D2, χ))
    Q = reshape(Q, χ,D1,D2,χ)

    QQ = oc_Q_4_corner(Q, FLu, FLo, ACu, ACd, FRu, FRo, ARu, ARd, Au11, Ad11, Au12, Ad12, Au21, Ad21, Au22, Ad22; forloop_iter, ifparallel)
    return dot(Q, QQ)
end

function contract_n_D(FLu::leg4, FLo, ACu, ACd, FRu, FRo, ARu, ARd, A11, A12, A21, A22; forloop_iter, ifparallel)
    return oc_D_leg4(FLu, FLo, ACu, ACd, FRu, FRo, ARu, ARd, A11, conj(A11), A12, conj(A12), A21, conj(A21), A22, conj(A22); forloop_iter, ifparallel)
end

function contract_o_D1(FLu::leg4, FLo, ACu, ACd, FRu, FRo, ARu, ARd, A11, A12, A21, A22, O1, O2; forloop_iter, ifparallel)
    Dh = size(O1, 3)
    atype = _arraytype(O1)
    IDh = Zygote.@ignore atype(Matrix{Float64}(I, Dh, Dh))
    D1,D2,D3,D4,d = size(A11)
    @tensor Au11_tmp[a,b,c,i,d,n] := A11[a,b,c,d,e] * O1[e,n,i]
    Au11 = reshape(Au11_tmp, D1,D2,D3*Dh,D4,d)
    Ad11 = conj(A11)
    D1,D2,D3,D4,_ = size(A12)
    @tensor Au12_tmp[a,n,b,i,c,d,e] := A12[a,b,c,d,e] * IDh[n,i]
    Au12 = reshape(Au12_tmp, D1*Dh,D2*Dh,D3,D4,d)
    Ad12 = conj(A12)
    Au21 = A21
    Ad21 = conj(A21)
    D1,D2,D3,D4,_ = size(A22)
    @tensor Au22_tmp[a,b,c,d,i,n] := A22[a,b,c,d,e] * O2[i,e,n]
    Au22 = reshape(Au22_tmp, D1,D2,D3,D4*Dh,d)
    Ad22 = conj(A22)
    return oc_D_leg4(FLu, FLo, ACu, ACd, FRu, FRo, ARu, ARd, Au11, Ad11, Au12, Ad12, Au21, Ad21, Au22, Ad22; forloop_iter, ifparallel)
end

function contract_o_D2(FLu::leg4, FLo, ACu, ACd, FRu, FRo, ARu, ARd, A11, A12, A21, A22, O1, O2; forloop_iter, ifparallel)
    Dh = size(O1, 3)
    atype = _arraytype(O1)
    IDh = Zygote.@ignore atype(Matrix{Float64}(I, Dh, Dh))
    D1,D2,D3,D4,d = size(A11)
    @tensor Au11_tmp[a,b,n,c,i,d,e] := A11[a,b,c,d,e] * IDh[n,i]
    Au11 = reshape(Au11_tmp, D1,D2*Dh,D3*Dh,D4,d)
    Ad11 = conj(A11)
    D1,D2,D3,D4,_ = size(A12)
    @tensor Au12_tmp[a,i,b,c,d,n] := A12[a,b,c,d,e] * O1[e,n,i]
    Au12 = reshape(Au12_tmp, D1*Dh,D2,D3,D4,d)
    Ad12 = conj(A12)
    D1,D2,D3,D4,_ = size(A21)
    @tensor Au21_tmp[a,b,c,d,i,n] := A21[a,b,c,d,e] * O2[i,e,n]
    Au21 = reshape(Au21_tmp, D1,D2,D3,D4*Dh,d)
    Ad21 = conj(A21)
    Au22 = A22
    Ad22 = conj(A22)
    return oc_D_leg4(FLu, FLo, ACu, ACd, FRu, FRo, ARu, ARd, Au11, Ad11, Au12, Ad12, Au21, Ad21, Au22, Ad22; forloop_iter, ifparallel)
end

# ============================================================================
# Three-site contractions (J1J2J3)
# ============================================================================

function oc_H3_leg4(FLo, ACu, ACd, FRo, ARu1, ARd1, ARu2, ARd2, A1u, A1d, A2u, A2d, A3u, A3d; forloop_iter, ifparallel)
    l = FLmap_parallel(FLo, ACu, ACd, (A1u, A1d); forloop_iter, ifparallel)
    l = FLmap_parallel(l, ARu1, ARd1, (A2u, A2d); forloop_iter, ifparallel)
    r = FRmap_parallel(FRo, ARu2, ARd2, (A3u, A3d); forloop_iter, ifparallel)
    return dot(conj(l), r)
end

function contract_n3_H(FLo::leg4, ACu, ACd, FRo, ARu1, ARd1, ARu2, ARd2, A1, A2, A3; forloop_iter, ifparallel)
    return oc_H3_leg4(FLo, ACu, ACd, FRo, ARu1, ARd1, ARu2, ARd2, A1, conj(A1), A2, conj(A2), A3, conj(A3); forloop_iter, ifparallel)
end

function contract_o3_H(FLo::leg4, ACu, ACd, FRo, ARu1, ARd1, ARu2, ARd2, A1, A2, A3, O1, O2; forloop_iter, ifparallel)
    Dh = size(O1, 3)
    atype = _arraytype(O1)
    IDh = Zygote.@ignore atype(Matrix{Float64}(I, Dh, Dh))
    D1,D2,D3,D4,d = size(A1)
    @tensor A1u_tmp[a,b,c,i,d,f] := A1[a,b,c,d,e] * O1[e,f,i]
    A1u = reshape(A1u_tmp, D1,D2,D3*Dh,D4,d)
    A1d = conj(A1)
    D1,D2,D3,D4,_ = size(A2)
    @tensor A2u_tmp[a,n,b,c,i,d,e] := A2[a,b,c,d,e] * IDh[n,i]
    A2u = reshape(A2u_tmp, D1*Dh,D2,D3*Dh,D4,d)
    A2d = conj(A2)
    D1,D2,D3,D4,_ = size(A3)
    @tensor A3u_tmp[a,i,b,c,d,f] := A3[a,b,c,d,e] * O2[i,e,f]
    A3u = reshape(A3u_tmp, D1*Dh,D2,D3,D4,d)
    A3d = conj(A3)
    return oc_H3_leg4(FLo, ACu, ACd, FRo, ARu1, ARd1, ARu2, ARd2, A1u, A1d, A2u, A2d, A3u, A3d; forloop_iter, ifparallel)
end

function oc_V3_leg4(ACu, ACd, FLu1, FRu1, FLu2, FRu2, FLo, FRo, A1u, A1d, A2u, A2d, A3u, A3d; forloop_iter, ifparallel)
    u = ACmap_parallel(ACu, FLu1, FRu1, (A1u, A1d); forloop_iter, ifparallel)
    u = ACmap_parallel(u, FLu2, FRu2, (A2u, A2d); forloop_iter, ifparallel)
    u = ACmap_parallel(u, FLo, FRo, (A3u, A3d); forloop_iter, ifparallel)
    return dot(conj(u), ACd)
end

function contract_n3_V(ACu::leg4, ACd, FLu1, FRu1, FLu2, FRu2, FLo, FRo, A1, A2, A3; forloop_iter, ifparallel)
    return oc_V3_leg4(ACu, ACd, FLu1, FRu1, FLu2, FRu2, FLo, FRo, A1, conj(A1), A2, conj(A2), A3, conj(A3); forloop_iter, ifparallel)
end

function contract_o3_V(ACu::leg4, ACd, FLu1, FRu1, FLu2, FRu2, FLo, FRo, A1, A2, A3, O1, O2; forloop_iter, ifparallel)
    Dh = size(O1, 3)
    atype = _arraytype(O1)
    IDh = Zygote.@ignore atype(Matrix{Float64}(I, Dh, Dh))
    D1,D2,D3,D4,d = size(A1)
    @tensor A1u_tmp[a,b,i,c,d,f] := A1[a,b,c,d,e] * O1[e,f,i]
    A1u = reshape(A1u_tmp, D1,D2*Dh,D3,D4,d)
    A1d = conj(A1)
    D1,D2,D3,D4,_ = size(A2)
    @tensor A2u_tmp[a,b,n,c,d,i,e] := A2[a,b,c,d,e] * IDh[n,i]
    A2u = reshape(A2u_tmp, D1,D2*Dh,D3,D4*Dh,d)
    A2d = conj(A2)
    D1,D2,D3,D4,_ = size(A3)
    @tensor A3u_tmp[a,b,c,d,i,f] := A3[a,b,c,d,e] * O2[i,e,f]
    A3u = reshape(A3u_tmp, D1,D2,D3,D4*Dh,d)
    A3d = conj(A3)
    return oc_V3_leg4(ACu, ACd, FLu1, FRu1, FLu2, FRu2, FLo, FRo, A1u, A1d, A2u, A2d, A3u, A3d; forloop_iter, ifparallel)
end
