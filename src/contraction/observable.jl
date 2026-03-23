"""
two-site contraction

```
                                            a ────┬──── c
a ────┬──c ──┬──── f                        │     b     │
│     b      e     │                        ├─ e ─┼─ f ─┤
├─ g ─┼─  h ─┼─ i ─┤                        g     h     i
│     k      n     │                        ├─ j ─┼─ k ─┤
j ────┴──l ──┴──── o                        │     m     │
                                            l ────┴──── n
```
"""
function oc_H_leg3(FLo, ACu, M1, ACd, FRo, ARu, M2, ARd; forloop_iter, ifparallel)
    l = FLmap_parallel(FLo, ACu, ACd, M1; forloop_iter, ifparallel)
    r = FRmap_parallel(FRo, ARu, ARd, M2; forloop_iter, ifparallel)
    return sum(ein"abc,abc->"(l,r))
end

function oc_V_leg3(ACu, FLu, M1, FRu, FLo, M2, FRo, ACd; forloop_iter, ifparallel)
    u = ACmap_parallel(ACu, FLu, FRu, M1; forloop_iter, ifparallel)
    u = ACmap_parallel(u, FLo, FRo, M2; forloop_iter, ifparallel)
    return sum(ein"abc,abc->"(u,ACd))
end

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
    return sum(ein"abcd,abcd->"(l,r))
end

function oc_V_leg4(ACu, FLu, A1u, A1d, FRu, FLo, A2u, A2d, FRo, ACd; forloop_iter, ifparallel)
    u = ACmap_parallel(ACu, FLu, FRu, (A1u, A1d); forloop_iter, ifparallel)
    u = ACmap_parallel(u, FLo, FRo, (A2u, A2d); forloop_iter, ifparallel)
    return sum(ein"abcd,abcd->"(u,ACd))
end

# ============================================================================
# Two-site normalization and observable contractions (leg3 dispatch)
# ============================================================================

function contract_n2_H(FLo::leg3, ACu, A1, ACd, FRo, ARu, A2, ARd; forloop_iter, ifparallel)
    D1,D2,D3,D4,_ = size(A1)
    M1 = reshape(ein"abcde,fghme->afbgchdm"(A1, conj(A1)), D1^2,D2^2,D3^2,D4^2)
    D1,D2,D3,D4,_ = size(A2)
    M2 = reshape(ein"abcde,fghme->afbgchdm"(A2, conj(A2)), D1^2,D2^2,D3^2,D4^2)
    return oc_H_leg3(FLo, ACu, M1, ACd, FRo, ARu, M2, ARd; forloop_iter, ifparallel)
end

function contract_o2_H(FLo::leg3, ACu, A1, ACd, FRo, ARu, A2, ARd, O1, O2; forloop_iter, ifparallel)
    D1,D2,D3,D4,_ = size(A1)
    Dh = size(O1, 3)
    M1 = reshape(ein"(abcde,eni),fghmn->afbgchidm"(A1, O1, conj(A1)), D1^2,D2^2,D3^2*Dh,D4^2)
    D1,D2,D3,D4,_ = size(A2)
    M2 = reshape(ein"(abcde,ien),fghmn->afibgchdm"(A2, O2, conj(A2)), D1^2*Dh,D2^2,D3^2,D4^2)
    return oc_H_leg3(FLo, ACu, M1, ACd, FRo, ARu, M2, ARd; forloop_iter, ifparallel)
end

function contract_n2_V(ACu::leg3, FLu, A1, FRu, FLo, A2, FRo, ACd; forloop_iter, ifparallel)
    D1,D2,D3,D4,_ = size(A1)
    M1 = reshape(ein"abcde,fghme->afbgchdm"(A1, conj(A1)), D1^2,D2^2,D3^2,D4^2)
    D1,D2,D3,D4,_ = size(A2)
    M2 = reshape(ein"abcde,fghme->afbgchdm"(A2, conj(A2)), D1^2,D2^2,D3^2,D4^2)
    return oc_V_leg3(ACu, FLu, M1, FRu, FLo, M2, FRo, ACd; forloop_iter, ifparallel)
end

function contract_o2_V(ACu::leg3, FLu, A1, FRu, FLo, A2, FRo, ACd, O1, O2; forloop_iter, ifparallel)
    D1,D2,D3,D4,_ = size(A1)
    Dh = size(O1, 3)
    M1 = reshape(ein"(abcde,eni),fghmn->afbgichdm"(A1, O1, conj(A1)), D1^2,D2^2*Dh,D3^2,D4^2)
    D1,D2,D3,D4,_ = size(A2)
    M2 = reshape(ein"(abcde,ien),fghmn->afbgchdmi"(A2, O2, conj(A2)), D1^2,D2^2,D3^2,D4^2*Dh)
    return oc_V_leg3(ACu, FLu, M1, FRu, FLo, M2, FRo, ACd; forloop_iter, ifparallel)
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
    A1u = reshape(ein"abcde,efi->abcidf"(A1, O1), D1,D2,D3*Dh,D4,d)
    D1,D2,D3,D4,d = size(A2)
    A2u = reshape(ein"abcde,ief->aibcdf"(A2, O2), D1*Dh,D2,D3,D4,d)
    return oc_H_leg4(FLo, ACu, A1u, conj(A1), ACd, FRo, ARu, A2u, conj(A2), ARd; forloop_iter, ifparallel)
end

function contract_n2_V(ACu::leg4, FLu, A1, FRu, FLo, A2, FRo, ACd; forloop_iter, ifparallel)
    return oc_V_leg4(ACu, FLu, A1, conj(A1), FRu, FLo, A2, conj(A2), FRo, ACd; forloop_iter, ifparallel)
end

function contract_o2_V(ACu::leg4, FLu, A1, FRu, FLo, A2, FRo, ACd, O1, O2; forloop_iter, ifparallel)
    D1,D2,D3,D4,d = size(A1)
    Dh = size(O1, 3)
    A1u = reshape(ein"abcde,efi->abicdf"(A1, O1), D1,D2*Dh,D3,D4,d)
    D1,D2,D3,D4,d = size(A2)
    A2u = reshape(ein"abcde,ief->abcdif"(A2, O2), D1,D2,D3,D4*Dh,d)
    return oc_V_leg4(ACu, FLu, A1u, conj(A1), FRu, FLo, A2u, conj(A2), FRo, ACd; forloop_iter, ifparallel)
end

# ============================================================================
# Single-site contraction
# ============================================================================

"""
one-site contraction

```
a ────┬──── c
│     b     │
├─ e ─┼─ f ─┤
│     h     │
g ────┴──── i

```
"""
function oc1_leg3(FLo, ACu, M, ACd, FRo; forloop_iter, ifparallel)
    l = FLmap_parallel(FLo, ACu, ACd, M; forloop_iter, ifparallel)
    return sum(ein"abc,abc->"(l,FRo))
end

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
    return sum(ein"abcd,abcd->"(l,FRo))
end

function contract_n1(FLo::leg3, ACu, A, ACd, FRo; forloop_iter, ifparallel)
    D1,D2,D3,D4,_ = size(A)
    M = reshape(ein"abcde,fghme->afbgchdm"(A, conj(A)), D1^2,D2^2,D3^2,D4^2)
    return oc1_leg3(FLo, ACu, M, ACd, FRo; forloop_iter, ifparallel)
end

function contract_o1(FLo::leg3, ACu, A, ACd, FRo, O; forloop_iter, ifparallel)
    D1,D2,D3,D4,_ = size(A)
    M = reshape(ein"(abcde,en),fghmn->afbgchdm"(A, O, conj(A)), D1^2,D2^2,D3^2,D4^2)
    return oc1_leg3(FLo, ACu, M, ACd, FRo; forloop_iter, ifparallel)
end

function contract_n1(FLo::leg4, ACu, A, ACd, FRo; forloop_iter, ifparallel)
    return oc1_leg4(FLo, ACu, A, conj(A), ACd, FRo; forloop_iter, ifparallel)
end

function contract_o1(FLo::leg4, ACu, A, ACd, FRo, O; forloop_iter, ifparallel)
    return oc1_leg4(FLo, ACu, ein"abcde,ef->abcdf"(A, O), conj(A), ACd, FRo; forloop_iter, ifparallel)
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
    return sum(ein"abcd,abcd->"(conj(Q), QQ))
end

function contract_n_D(FLu::leg4, FLo, ACu, ACd, FRu, FRo, ARu, ARd, A11, A12, A21, A22; forloop_iter, ifparallel)
    return oc_D_leg4(FLu, FLo, ACu, ACd, FRu, FRo, ARu, ARd, A11, conj(A11), A12, conj(A12), A21, conj(A21), A22, conj(A22); forloop_iter, ifparallel)
end

function contract_o_D1(FLu::leg4, FLo, ACu, ACd, FRu, FRo, ARu, ARd, A11, A12, A21, A22, O1, O2; forloop_iter, ifparallel)
    Dh = size(O1, 3)
    atype = _arraytype(O1)
    IDh = Zygote.@ignore atype(Matrix{Float64}(I, Dh, Dh))
    D1,D2,D3,D4,d = size(A11)
    Au11 = reshape(ein"abcde,eni->abcidn"(A11, O1), D1,D2,D3*Dh,D4,d)
    Ad11 = conj(A11)
    D1,D2,D3,D4,_ = size(A12)
    Au12 = reshape(ein"abcde,ni->anbicde"(A12, IDh), D1*Dh,D2*Dh,D3,D4,d)
    Ad12 = conj(A12)
    Au21 = A21
    Ad21 = conj(A21)
    D1,D2,D3,D4,_ = size(A22)
    Au22 = reshape(ein"abcde,ien->abcdin"(A22, O2), D1,D2,D3,D4*Dh,d)
    Ad22 = conj(A22)
    return oc_D_leg4(FLu, FLo, ACu, ACd, FRu, FRo, ARu, ARd, Au11, Ad11, Au12, Ad12, Au21, Ad21, Au22, Ad22; forloop_iter, ifparallel)
end

function contract_o_D2(FLu::leg4, FLo, ACu, ACd, FRu, FRo, ARu, ARd, A11, A12, A21, A22, O1, O2; forloop_iter, ifparallel)
    Dh = size(O1, 3)
    atype = _arraytype(O1)
    IDh = Zygote.@ignore atype(Matrix{Float64}(I, Dh, Dh))
    D1,D2,D3,D4,d = size(A11)
    Au11 = reshape(ein"abcde,ni->abncide"(A11, IDh), D1,D2*Dh,D3*Dh,D4,d)
    Ad11 = conj(A11)
    D1,D2,D3,D4,_ = size(A12)
    Au12 = reshape(ein"abcde,eni->aibcdn"(A12, O1), D1*Dh,D2,D3,D4,d)
    Ad12 = conj(A12)
    D1,D2,D3,D4,_ = size(A21)
    Au21 = reshape(ein"abcde,ien->abcdin"(A21, O2), D1,D2,D3,D4*Dh,d)
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
    return sum(ein"abcd,abcd->"(l,r))
end

function contract_n3_H(FLo::leg4, ACu, ACd, FRo, ARu1, ARd1, ARu2, ARd2, A1, A2, A3; forloop_iter, ifparallel)
    return oc_H3_leg4(FLo, ACu, ACd, FRo, ARu1, ARd1, ARu2, ARd2, A1, conj(A1), A2, conj(A2), A3, conj(A3); forloop_iter, ifparallel)
end

function contract_o3_H(FLo::leg4, ACu, ACd, FRo, ARu1, ARd1, ARu2, ARd2, A1, A2, A3, O1, O2; forloop_iter, ifparallel)
    Dh = size(O1, 3)
    atype = _arraytype(O1)
    IDh = Zygote.@ignore atype(Matrix{Float64}(I, Dh, Dh))
    D1,D2,D3,D4,d = size(A1)
    A1u = reshape(ein"abcde,efi->abcidf"(A1, O1), D1,D2,D3*Dh,D4,d)
    A1d = conj(A1)
    D1,D2,D3,D4,_ = size(A2)
    A2u = reshape(ein"abcde,ni->anbcide"(A2, IDh), D1*Dh,D2,D3*Dh,D4,d)
    A2d = conj(A2)
    D1,D2,D3,D4,_ = size(A3)
    A3u = reshape(ein"abcde,ief->aibcdf"(A3, O2), D1*Dh,D2,D3,D4,d)
    A3d = conj(A3)
    return oc_H3_leg4(FLo, ACu, ACd, FRo, ARu1, ARd1, ARu2, ARd2, A1u, A1d, A2u, A2d, A3u, A3d; forloop_iter, ifparallel)
end

function oc_V3_leg4(ACu, ACd, FLu1, FRu1, FLu2, FRu2, FLo, FRo, A1u, A1d, A2u, A2d, A3u, A3d; forloop_iter, ifparallel)
    u = ACmap_parallel(ACu, FLu1, FRu1, (A1u, A1d); forloop_iter, ifparallel)
    u = ACmap_parallel(u, FLu2, FRu2, (A2u, A2d); forloop_iter, ifparallel)
    u = ACmap_parallel(u, FLo, FRo, (A3u, A3d); forloop_iter, ifparallel)
    return sum(ein"abcd,abcd->"(u,ACd))
end

function contract_n3_V(ACu::leg4, ACd, FLu1, FRu1, FLu2, FRu2, FLo, FRo, A1, A2, A3; forloop_iter, ifparallel)
    return oc_V3_leg4(ACu, ACd, FLu1, FRu1, FLu2, FRu2, FLo, FRo, A1, conj(A1), A2, conj(A2), A3, conj(A3); forloop_iter, ifparallel)
end

function contract_o3_V(ACu::leg4, ACd, FLu1, FRu1, FLu2, FRu2, FLo, FRo, A1, A2, A3, O1, O2; forloop_iter, ifparallel)
    Dh = size(O1, 3)
    atype = _arraytype(O1)
    IDh = Zygote.@ignore atype(Matrix{Float64}(I, Dh, Dh))
    D1,D2,D3,D4,d = size(A1)
    A1u = reshape(ein"abcde,efi->abicdf"(A1, O1), D1,D2*Dh,D3,D4,d)
    A1d = conj(A1)
    D1,D2,D3,D4,_ = size(A2)
    A2u = reshape(ein"abcde,ni->abncdie"(A2, IDh), D1,D2*Dh,D3,D4*Dh,d)
    A2d = conj(A2)
    D1,D2,D3,D4,_ = size(A3)
    A3u = reshape(ein"abcde,ief->abcdif"(A3, O2), D1,D2,D3,D4*Dh,d)
    A3d = conj(A3)
    return oc_V3_leg4(ACu, ACd, FLu1, FRu1, FLu2, FRu2, FLo, FRo, A1u, A1d, A2u, A2d, A3u, A3d; forloop_iter, ifparallel)
end
