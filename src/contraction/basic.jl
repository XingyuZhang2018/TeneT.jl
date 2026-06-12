"""
    _downcast_eltype(T, A) -> A'

Cast `A`'s element type to the smallest type compatible with real scalar type `T`
while preserving complex-ness. Used to lower precision inside FLmap/FRmap/ACmap
when `VUMPS.inner_etype` is set.

Rules:
- `T === nothing`               → return `A` unchanged (identity)
- `T === real(eltype(A))`       → return `A` unchanged (identity; no copy)
- `eltype(A) <: Complex`        → cast to `Complex{T}`
- otherwise                     → cast to `T`
"""
_downcast_eltype(::Nothing, A) = A
function _downcast_eltype(T::Type, A)
    Ta = eltype(A)
    if Ta <: Complex
        T === real(Ta) && return A
        return Complex{T}.(A)
    else
        T === Ta && return A
        return T.(A)
    end
end

# Specialization for StructArray: broadcast cast over the underlying data.
# `eltype(S::StructArray) = Any` so we cannot rely on the generic branch, and
# `T.(S)` fails because StructArray does not implement Broadcast. Cast every
# unique data entry and rebuild the StructArray with the same pattern.
_downcast_eltype(::Nothing, S::StructArray) = S
function _downcast_eltype(T::Type, S::StructArray)
    new_data = [_downcast_eltype(T, d) for d in S.data]
    return StructArray(new_data, S.pattern)
end

ALCtoAC_map(AL::leg3, C) = @tensor result[a,b,d] := AL[a,b,c] * C[c,d]
ALCtoAC_map(AL::leg4, C) = @tensor result[a,b,c,e] := AL[a,b,c,d] * C[d,e]
CTtoT(C, T::leg3) = @tensor T[a,c,d] := C[a,b] * T[b,c,d]
CTtoT(C, T::leg4) = @tensor T[a,c,d,e] := C[a,b] * T[b,c,d,e]
CTCtoT(C, T::leg3) = @tensor T[a,c,e] := C[a,b] * T[b,c,d] * C[d,e]
CTCtoT(C, T::leg4) = @tensor T[a,c,d,f] := C[a,b] * T[b,c,d,e] * C[e,f]

"""
    FLm = FLmap(ALu, ALd, M, FL)

```
  ┌──       ┌──  ALuᵢⱼ  ──                     a ────┬──── c
  │         │     │                            │     b     │
FLᵢⱼ₊₁ =   FLᵢⱼ ─ Mᵢⱼ   ──                     ├─ d ─┼─ e ─┤
  │         │     │                            │     g     │
  └──       └──  ALdᵢᵣⱼ  ─                     f ────┴──── h
```
"""
function FLmap(FL, ALu, ALd, M::leg4; inner_etype=nothing)
    use_chain_engine(FL, ALu, ALd, M) &&
        return _chain_map(FLMAP_LEG4_CHAIN, (FL, ALd, M, ALu), inner_etype)
    if inner_etype === nothing || inner_etype == real(eltype(FL))
        @tensor result[c,e,h] := FL[a,d,f] * ALd[f,g,h] * M[d,g,e,b] * ALu[a,b,c]
        return result
    else
        T_out = eltype(FL)
        FL_t  = _downcast_eltype(inner_etype, FL)
        ALu_t = _downcast_eltype(inner_etype, ALu)
        ALd_t = _downcast_eltype(inner_etype, ALd)
        M_t   = _downcast_eltype(inner_etype, M)
        @tensor result_t[c,e,h] := FL_t[a,d,f] * ALd_t[f,g,h] * M_t[d,g,e,b] * ALu_t[a,b,c]
        return T_out.(result_t)
    end
end
function FLmap(FL, ALu, ALd, M1::leg5, M2::leg5; inner_etype=nothing)
    use_chain_engine(FL, ALu, ALd, M1, M2) &&
        return _chain_map(FLMAP_LEG5_CHAIN, (FL, ALd, M1, M2, ALu), inner_etype)
    if inner_etype === nothing || inner_etype == real(eltype(FL))
        @tensor result[d,g,h,l] := FL[a,e,f,i] * ALd[i,j,k,l] * M1[e,j,g,b,p] * M2[f,k,h,c,p] * ALu[a,b,c,d]
        return result
    else
        T_out = eltype(FL)
        FL_t  = _downcast_eltype(inner_etype, FL)
        ALu_t = _downcast_eltype(inner_etype, ALu)
        ALd_t = _downcast_eltype(inner_etype, ALd)
        M1_t  = _downcast_eltype(inner_etype, M1)
        M2_t  = _downcast_eltype(inner_etype, M2)
        @tensor result_t[d,g,h,l] := FL_t[a,e,f,i] * ALd_t[i,j,k,l] * M1_t[e,j,g,b,p] * M2_t[f,k,h,c,p] * ALu_t[a,b,c,d]
        return T_out.(result_t)
    end
end
function FLmap(FL, ALu, ALd, M::leg8; inner_etype=nothing)
    use_chain_engine(FL, ALu, ALd, M) &&
        return _chain_map(FLMAP_LEG8_CHAIN, (FL, ALd, M, ALu), inner_etype)
    if inner_etype === nothing || inner_etype == real(eltype(FL))
        @tensor result[d,g,h,l] := FL[a,e,f,i] * ALd[i,j,k,l] * M[e,f,j,k,g,h,b,c] * ALu[a,b,c,d]
        return result
    else
        T_out = eltype(FL)
        FL_t  = _downcast_eltype(inner_etype, FL)
        ALu_t = _downcast_eltype(inner_etype, ALu)
        ALd_t = _downcast_eltype(inner_etype, ALd)
        M_t   = _downcast_eltype(inner_etype, M)
        @tensor result_t[d,g,h,l] := FL_t[a,e,f,i] * ALd_t[i,j,k,l] * M_t[e,f,j,k,g,h,b,c] * ALu_t[a,b,c,d]
        return T_out.(result_t)
    end
end

function FLmap(FL, ALu, ALd, M::leg5; inner_etype=nothing)
    use_chain_engine(FL, ALu, ALd, M) &&
        return _chain_map(FLMAP_LEG5_CHAIN_1M, (FL, ALd, M, M, ALu), inner_etype)
    return FLmap(FL, ALu, ALd, M, conj(M); inner_etype)
end
FLmap(FL, ALu, ALd, M::Tuple{leg5,leg5}; inner_etype=nothing) =
    FLmap(FL, ALu, ALd, M[1], M[2]; inner_etype)

function FLmap_C3v(FL, ALu, ALd, M1::leg4, M2::leg4, M3::leg4, M4::leg4; inner_etype=nothing)
    if inner_etype === nothing || inner_etype == real(eltype(FL))
        @tensor result[3,8,88,9] := FL[1,4,44,5] * ALu[1,2,22,3] * M1[4,7,2,10] * M2[44,77,22,10] * M3[6,8,7,11] * M4[66,88,77,11] * ALd[5,6,66,9]
        return result
    else
        T_out = eltype(FL)
        FL_t  = _downcast_eltype(inner_etype, FL)
        ALu_t = _downcast_eltype(inner_etype, ALu)
        ALd_t = _downcast_eltype(inner_etype, ALd)
        M1_t  = _downcast_eltype(inner_etype, M1)
        M2_t  = _downcast_eltype(inner_etype, M2)
        @tensor result_t[3,8,88,9] := FL_t[1,4,44,5] * ALu_t[1,2,22,3] * M1_t[4,7,2,10] * M2_t[44,77,22,10] * M3[6,8,7,11] * M4[66,88,77,11] * ALd_t[5,6,66,9]
        return T_out.(result_t)
    end
end

FLmap_C3v(FL, ALu, ALd, M::leg4; inner_etype=nothing) =
    FLmap_C3v(FL, ALu, ALd, M, conj(M), M, conj(M); inner_etype)
FLmap_C3v(FL, ALu, ALd, M1::leg4, M2::leg4; inner_etype=nothing) =
    FLmap_C3v(FL, ALu, ALd, M1, conj(M1), M2, conj(M2); inner_etype)

"""
    FRm = FRmap(ARu, ARd, M, FR, i)

```
    ── ARuᵢⱼ  ──┐          ──┐          a ────┬──── c
        │       │            │          │     b     │
    ── Mᵢⱼ   ──FRᵢⱼ  =    ──FRᵢⱼ₋₁      ├─ d ─┼─ e ─┤
        │       │            │          │     g     │
    ── ARdᵢᵣⱼ ──┘          ──┘          f ────┴──── h
```
"""
function FRmap(FR, ARu, ARd, M::leg4; inner_etype=nothing)
    use_chain_engine(FR, ARu, ARd, M) &&
        return _chain_map(FRMAP_LEG4_CHAIN, (ARd, FR, M, ARu), inner_etype)
    if inner_etype === nothing || inner_etype == real(eltype(FR))
        @tensor result[a,d,f] := ARd[f,g,h] * FR[c,e,h] * M[d,g,e,b] * ARu[a,b,c]
        return result
    else
        T_out = eltype(FR)
        FR_t  = _downcast_eltype(inner_etype, FR)
        ARu_t = _downcast_eltype(inner_etype, ARu)
        ARd_t = _downcast_eltype(inner_etype, ARd)
        M_t   = _downcast_eltype(inner_etype, M)
        @tensor result_t[a,d,f] := ARd_t[f,g,h] * FR_t[c,e,h] * M_t[d,g,e,b] * ARu_t[a,b,c]
        return T_out.(result_t)
    end
end
function FRmap(FR, ARu, ARd, M1::leg5, M2::leg5; inner_etype=nothing)
    use_chain_engine(FR, ARu, ARd, M1, M2) &&
        return _chain_map(FRMAP_LEG5_CHAIN, (ARd, FR, M1, M2, ARu), inner_etype)
    if inner_etype === nothing || inner_etype == real(eltype(FR))
        @tensor result[a,e,f,i] := ARd[i,j,k,l] * FR[d,g,h,l] * M1[e,j,g,b,p] * M2[f,k,h,c,p] * ARu[a,b,c,d]
        return result
    else
        T_out = eltype(FR)
        FR_t  = _downcast_eltype(inner_etype, FR)
        ARu_t = _downcast_eltype(inner_etype, ARu)
        ARd_t = _downcast_eltype(inner_etype, ARd)
        M1_t  = _downcast_eltype(inner_etype, M1)
        M2_t  = _downcast_eltype(inner_etype, M2)
        @tensor result_t[a,e,f,i] := ARd_t[i,j,k,l] * FR_t[d,g,h,l] * M1_t[e,j,g,b,p] * M2_t[f,k,h,c,p] * ARu_t[a,b,c,d]
        return T_out.(result_t)
    end
end
function FRmap(FR, ARu, ARd, M::leg8; inner_etype=nothing)
    use_chain_engine(FR, ARu, ARd, M) &&
        return _chain_map(FRMAP_LEG8_CHAIN, (ARd, FR, M, ARu), inner_etype)
    if inner_etype === nothing || inner_etype == real(eltype(FR))
        @tensor result[a,e,f,i] := ARd[i,j,k,l] * FR[d,g,h,l] * M[e,f,j,k,g,h,b,c] * ARu[a,b,c,d]
        return result
    else
        T_out = eltype(FR)
        FR_t  = _downcast_eltype(inner_etype, FR)
        ARu_t = _downcast_eltype(inner_etype, ARu)
        ARd_t = _downcast_eltype(inner_etype, ARd)
        M_t   = _downcast_eltype(inner_etype, M)
        @tensor result_t[a,e,f,i] := ARd_t[i,j,k,l] * FR_t[d,g,h,l] * M_t[e,f,j,k,g,h,b,c] * ARu_t[a,b,c,d]
        return T_out.(result_t)
    end
end

function FRmap(FR, ARu, ARd, M::leg5; inner_etype=nothing)
    use_chain_engine(FR, ARu, ARd, M) &&
        return _chain_map(FRMAP_LEG5_CHAIN_1M, (ARd, FR, M, M, ARu), inner_etype)
    return FRmap(FR, ARu, ARd, M, conj(M); inner_etype)
end
FRmap(FR, ARu, ARd, M::Tuple{leg5,leg5}; inner_etype=nothing) =
    FRmap(FR, ARu, ARd, M[1], M[2]; inner_etype)

"""
    ```
    ┌── ALuᵢⱼ  ──      ┌──        a──────┬──────c
    Lᵢⱼ   |        =   Lᵢⱼ₊₁      │      │      │
    └── ALdᵢᵣⱼ ──      └──        │      b      │
                                  │      │      │
                                  d──────┴──────e
    ```
"""
function Lmap(Lij, ALuij::leg3, ALdirj::leg3)
    @tensor result[c,e] := Lij[a,d] * ALdirj[d,b,e] * ALuij[a,b,c]
    return result
end

function Lmap(Lij, ALuij::leg4, ALdirj::leg4)
    @tensor result[c,e] := Lij[a,d] * ALdirj[d,b,f,e] * ALuij[a,b,f,c]
    return result
end

"""
    Rm = Rmap(FRi::Vector{<:AbstractTensorMap},
                ARui::Vector{<:AbstractTensorMap},
                ARdir::Vector{<:AbstractTensorMap},
                )

```
    ── ARuᵢⱼ  ──┐          ──┐           a──────┬──────c
        │       Rᵢⱼ  =       Rᵢⱼ₋₁       │      │      │
    ── ARdᵢᵣⱼ ──┘          ──┘           │      b      │
                                         │      │      │
                                         d──────┴──────e
```
"""
function Rmap(Ri, ARui::leg3, ARdir::leg3)
    @tensor result[a,d] := ARui[a,b,c] * Ri[c,e] * ARdir[d,b,e]
    return result
end

function Rmap(Ri, ARui::leg4, ARdir::leg4)
    @tensor result[a,d] := ARui[a,b,f,c] * Ri[c,e] * ARdir[d,b,f,e]
    return result
end

"""
    ACm = ACmap(ACij, FLj, FRj, Mj, II)

```
                                ┌─────── ACᵢⱼ ─────┐              a ────┬──── c
┌───── ACᵢ₊₁ⱼ ─────┐            │        │         │              │     b     │
│        │         │      =     FLᵢⱼ ─── Mᵢⱼ ───── FRᵢⱼ           ├─ d ─┼─ e ─┤
                                │        │         │              │     g     │
                                                                  f ────┴──── h

```
"""
function ACmap(AC, FL, FR, M::leg4; inner_etype=nothing)
    use_chain_engine(AC, FL, FR, M) &&
        return _chain_map(ACMAP_LEG4_CHAIN, (AC, FR, M, FL), inner_etype)
    if inner_etype === nothing || inner_etype == real(eltype(AC))
        @tensor result[f,g,h] := AC[a,b,c] * FR[c,e,h] * M[d,g,e,b] * FL[a,d,f]
        return result
    else
        T_out = eltype(AC)
        AC_t = _downcast_eltype(inner_etype, AC)
        FL_t = _downcast_eltype(inner_etype, FL)
        FR_t = _downcast_eltype(inner_etype, FR)
        M_t  = _downcast_eltype(inner_etype, M)
        @tensor result_t[f,g,h] := AC_t[a,b,c] * FR_t[c,e,h] * M_t[d,g,e,b] * FL_t[a,d,f]
        return T_out.(result_t)
    end
end
function ACmap(AC, FL, FR, M1::leg5, M2::leg5; inner_etype=nothing)
    use_chain_engine(AC, FL, FR, M1, M2) &&
        return _chain_map(ACMAP_LEG5_CHAIN, (AC, FR, M1, M2, FL), inner_etype)
    if inner_etype === nothing || inner_etype == real(eltype(AC))
        @tensor result[i,j,k,l] := AC[a,b,c,d] * FR[d,g,h,l] * M1[e,j,g,b,p] * M2[f,k,h,c,p] * FL[a,e,f,i]
        return result
    else
        T_out = eltype(AC)
        AC_t = _downcast_eltype(inner_etype, AC)
        FL_t = _downcast_eltype(inner_etype, FL)
        FR_t = _downcast_eltype(inner_etype, FR)
        M1_t = _downcast_eltype(inner_etype, M1)
        M2_t = _downcast_eltype(inner_etype, M2)
        @tensor result_t[i,j,k,l] := AC_t[a,b,c,d] * FR_t[d,g,h,l] * M1_t[e,j,g,b,p] * M2_t[f,k,h,c,p] * FL_t[a,e,f,i]
        return T_out.(result_t)
    end
end
function ACmap(AC, FL, FR, M::leg8; inner_etype=nothing)
    use_chain_engine(AC, FL, FR, M) &&
        return _chain_map(ACMAP_LEG8_CHAIN, (AC, FR, M, FL), inner_etype)
    if inner_etype === nothing || inner_etype == real(eltype(AC))
        @tensor result[i,j,k,l] := AC[a,b,c,d] * FR[d,g,h,l] * M[e,f,j,k,g,h,b,c] * FL[a,e,f,i]
        return result
    else
        T_out = eltype(AC)
        AC_t = _downcast_eltype(inner_etype, AC)
        FL_t = _downcast_eltype(inner_etype, FL)
        FR_t = _downcast_eltype(inner_etype, FR)
        M_t  = _downcast_eltype(inner_etype, M)
        @tensor result_t[i,j,k,l] := AC_t[a,b,c,d] * FR_t[d,g,h,l] * M_t[e,f,j,k,g,h,b,c] * FL_t[a,e,f,i]
        return T_out.(result_t)
    end
end

function ACmap(AC, FL, FR, M::leg5; inner_etype=nothing)
    use_chain_engine(AC, FL, FR, M) &&
        return _chain_map(ACMAP_LEG5_CHAIN_1M, (AC, FR, M, M, FL), inner_etype)
    return ACmap(AC, FL, FR, M, conj(M); inner_etype)
end
ACmap(AC, FL, FR, M::Tuple{leg5,leg5}; inner_etype=nothing) =
    ACmap(AC, FL, FR, M[1], M[2]; inner_etype)

"""
    Cmap(Cij, FLjp, FRj, II)

```
                    ┌────Cᵢⱼ ───┐            a ─── b
┌── Cᵢ₊₁ⱼ ──┐       │           │            │     │
│           │  =   FLᵢⱼ₊₁ ──── FRᵢⱼ          ├─ c ─┤
                    │           │            │     │
                                             d ─── e
```
"""
function Cmap(C, FL::leg3, FR)
    @tensor result[d,e] := FL[a,c,d] * C[a,b] * FR[b,c,e]
    return result
end
function Cmap(C, FL::leg4, FR)
    @tensor result[e,f] := FL[a,c,d,e] * C[a,b] * FR[b,c,d,f]
    return result
end

"""
    a ────┬──── c
    │     b     │
    ├─ d ─┼─ e ─┤
    │     g     │
    f ────┴──── h
"""
function ACdmap(ACd, FL, FR, M::leg4; inner_etype=nothing)
    if inner_etype === nothing || inner_etype == real(eltype(ACd))
        @tensor result[a,b,c] := ACd[f,g,h] * FR[c,e,h] * M[d,g,e,b] * FL[a,d,f]
        return result
    else
        T_out = eltype(ACd)
        ACd_t = _downcast_eltype(inner_etype, ACd)
        FL_t  = _downcast_eltype(inner_etype, FL)
        FR_t  = _downcast_eltype(inner_etype, FR)
        M_t   = _downcast_eltype(inner_etype, M)
        @tensor result_t[a,b,c] := ACd_t[f,g,h] * FR_t[c,e,h] * M_t[d,g,e,b] * FL_t[a,d,f]
        return T_out.(result_t)
    end
end
function ACdmap(ACd, FL, FR, M1::leg5, M2::leg5; inner_etype=nothing)
    if inner_etype === nothing || inner_etype == real(eltype(ACd))
        @tensor result[a,b,c,d] := ACd[i,j,k,l] * FR[d,g,h,l] * M1[e,j,g,b,p] * M2[f,k,h,c,p] * FL[a,e,f,i]
        return result
    else
        T_out = eltype(ACd)
        ACd_t = _downcast_eltype(inner_etype, ACd)
        FL_t  = _downcast_eltype(inner_etype, FL)
        FR_t  = _downcast_eltype(inner_etype, FR)
        M1_t  = _downcast_eltype(inner_etype, M1)
        M2_t  = _downcast_eltype(inner_etype, M2)
        @tensor result_t[a,b,c,d] := ACd_t[i,j,k,l] * FR_t[d,g,h,l] * M1_t[e,j,g,b,p] * M2_t[f,k,h,c,p] * FL_t[a,e,f,i]
        return T_out.(result_t)
    end
end

ACdmap(ACd, FL, FR, M::leg5; inner_etype=nothing) =
    ACdmap(ACd, FL, FR, M, conj(M); inner_etype)
ACdmap(ACd, FL, FR, M::Tuple{leg5,leg5}; inner_etype=nothing) =
    ACdmap(ACd, FL, FR, M[1], M[2]; inner_etype)

function Mmap(AC, ACd, FL, FR)
    @tensor result[d,g,e,b] := AC[a,b,c] * FR[c,e,h] * FL[a,d,f] * ACd[f,g,h]
    return result
end
function Mumap(AC, ACd, FL, FR, Mu)
    @tensor result[f,k,h,c,p] := (AC[a,b,c,d] * FR[d,g,h,l]) * ((FL[a,e,f,i] * ACd[i,j,k,l]) * Mu[e,j,g,b,p])
    return result
end
function Mdmap(AC, ACd, FL, FR, Md)
    @tensor result[e,j,g,b,p] := (AC[a,b,c,d] * FR[d,g,h,l]) * ((FL[a,e,f,i] * ACd[i,j,k,l]) * Md[f,k,h,c,p])
    return result
end

LDmap(L, D, M1::leg5, M2::leg5) = @tensor result[1,2,3,7,8,12] := L[1,5,6,9] * D[9,10,11,12] * M1[5,10,7,2,13] * M2[6,11,8,3,13]
DRmap(D, R, M1::leg5, M2::leg5) = @tensor result[9,5,6,2,3,4] := D[9,10,11,12] * R[4,7,8,12] * M1[5,10,7,2,13] * M2[6,11,8,3,13]
RUmap(R, U, M1::leg5, M2::leg5) = @tensor result[12,10,11,5,6,1] := U[1,2,3,4] * R[4,7,8,12] * M1[5,10,7,2,13] * M2[6,11,8,3,13]
LUmap(L, U, M1::leg5, M2::leg5) = @tensor result[9,10,11,7,8,4] := L[1,5,6,9] * U[1,2,3,4] * M1[5,10,7,2,13] * M2[6,11,8,3,13]

LDmap(L, D, M::leg5) = LDmap(L, D, M, conj(M))
DRmap(D, R, M::leg5) = DRmap(D, R, M, conj(M))
RUmap(R, U, M::leg5) = RUmap(R, U, M, conj(M))
LUmap(L, U, M::leg5) = LUmap(L, U, M, conj(M))

LDmap(L, D, M::Tuple{leg5,leg5}) = LDmap(L, D, M[1], M[2])
DRmap(D, R, M::Tuple{leg5,leg5}) = DRmap(D, R, M[1], M[2])
RUmap(R, U, M::Tuple{leg5,leg5}) = RUmap(R, U, M[1], M[2])
LUmap(L, U, M::Tuple{leg5,leg5}) = LUmap(L, U, M[1], M[2])
